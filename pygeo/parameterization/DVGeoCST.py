"""
==============================================================================
DVGeo: CST Parameterisation
==============================================================================
@Author : Eytan Adler, Alasdair Christison Gray
@Description : A DVGeo implementation based on the Class-Shape Transformation method
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from mpi4py import MPI
from scipy.special import factorial
from prefoil.utils import readCoordFile
from prefoil.airfoil import Airfoil

try:
    import matplotlib.pyplot as plt

    pltImport = True
except ModuleNotFoundError:
    pltImport = False


# ==============================================================================
# Extension modules
# ==============================================================================
from pygeo.parameterization.BaseDVGeo import BaseDVGeometry
from pygeo.parameterization.designVars import cstDV


class DVGeometryCST(BaseDVGeometry):
    r"""
    This class implements a 2D geometry parameterisation based on Brenda Kulfan's CST (Class-Shape Transformation) method.
    This class can work with 3D coordinates but will only change the point coordinates in one direction.

    The CST equation is as follows:

    :math:`y(x) = C(x) * S(x) + y_\text{te}x`

    Where C is the class function:

    :math:`C(x) = (x^{N1} + (1 - x)^{N2})`

    And S is the shape function, in this case a summation of Bernstein polynomials:

    :math:`S(x) = \sum_i^n w_i \binom{n}{i}x^i(1-x)^{n-i}`

    Here x is the normalized chordwise coordinate, ranging from 0 to 1 from front to the rear of the shape.

    This class works for extruded 2D airfoil geometries, but not 3D geometries that vary in the spanwise direction.

    Parameters
    ----------
    datFile : str
        Filename of dat file that represents the initial airfoil. The coordinates in this file will be used to
        determine the camber line, which is the dividing line to distinguish upper and lower surface points.
    idxChord : int, optional
        Index of the column in the point set to use as the chordwise (x in CST) coordinates, by default 0
    idxVertical : int, optional
        Index of the column in the point set to use as the vertical (y in CST) airfoil coordinates, by default 1
    comm : MPI communicator, optional
        Communicator for DVGeometryCST instance, by default MPI.COMM_WORLD
    isComplex : bool, optional
        Initialize variables to complex types where necessary, by default False
    debug : bool, optional
        Show plots when addPointSet is called to visually verify that it is correctly splitting
        the upper and lower surfaces of the airfoil points, by default False
    """

    def __init__(self, datFile, idxChord=0, idxVertical=1, comm=MPI.COMM_WORLD, isComplex=False, debug=False):
        super().__init__(datFile)
        self.xIdx = idxChord
        self.yIdx = idxVertical
        self.comm = comm
        self.isComplex = isComplex
        if isComplex:
            self.dtype = complex
            self.dtypeMPI = MPI.DOUBLE_COMPLEX
        else:
            self.dtype = float
            self.dtypeMPI = MPI.DOUBLE
        self.debug = debug
        if debug and not pltImport:
            raise ImportError("matplotlib.pyplot could not be imported and is required for DVGeoCST debug mode")

        # Store the DVs and flags to determine if the limited options have already been specified
        self.DVs = {}
        self.DVExists = {
            "upper": False,
            "lower": False,
            "n1_upper": False,
            "n2_upper": False,
            "n1_lower": False,
            "n2_lower": False,
            "chord": False,
        }

        # Default DVs to be copied for each point set
        self.defaultDV = {
            "upper": 0.1 * np.ones(8, dtype=self.dtype),
            "lower": -0.1 * np.ones(8, dtype=self.dtype),
            "n1_upper": np.array([0.5], dtype=self.dtype),
            "n2_upper": np.array([1.0], dtype=self.dtype),
            "n1_lower": np.array([0.5], dtype=self.dtype),
            "n2_lower": np.array([1.0], dtype=self.dtype),
            "n1": np.array([0.5], dtype=self.dtype),
            "n2": np.array([1.0], dtype=self.dtype),
            "chord": np.array([1.0], dtype=self.dtype),
        }

        # ========== Process the input airfoil and set variables accordingly ==========
        coords = readCoordFile(datFile)
        self.foilCoords = np.zeros_like(coords, dtype=self.dtype)
        self.foilCoords[:, self.xIdx] = coords[:, 0]
        self.foilCoords[:, self.yIdx] = coords[:, 1]

        # Set the leading and trailing edge x coordinates
        self.xMin = np.min(self.foilCoords[:, self.xIdx])
        self.xMax = np.max(self.foilCoords[:, self.xIdx])

        # Check that the leading edge is at y = 0
        idxLE = np.argmin(self.foilCoords[:, self.xIdx])
        yLE = self.foilCoords[idxLE, self.yIdx]
        if abs(yLE) > 1e-2:
            raise ValueError(f"Leading edge y (or idxVertical) value must equal zero, not {yLE}")

        # Determine the trailing edge thickness
        idxTE = np.where(self.foilCoords[:, self.xIdx] == self.xMax)[0]
        self.yUpperTE = np.max(self.foilCoords[idxTE, self.yIdx])
        self.yLowerTE = np.min(self.foilCoords[idxTE, self.yIdx])
        self.thicknessTE = self.yUpperTE - self.yLowerTE

        # Compute splines for the upper and lower surfaces (used to split the foil in addPointSet)
        self.foil = Airfoil(coords)
        self.upperSpline, self.lowerSpline = self.foil.splitAirfoil()

        # Fit CST parameters to the airfoil's upper and lower surface
        self.idxFoil = {}
        self.idxFoil["upper"], self.idxFoil["lower"] = self._splitUpperLower(self.foilCoords)
        self.defaultDV["chord"][0] = self.xMax - self.xMin
        for dvType in ["upper", "lower"]:
            self.defaultDV[dvType] = self.computeCSTfromCoords(
                self.foilCoords[self.idxFoil[dvType], self.xIdx],
                self.foilCoords[self.idxFoil[dvType], self.yIdx],
                self.defaultDV[dvType].size,
                N1=self.defaultDV[f"n1_{dvType}"],
                N2=self.defaultDV[f"n2_{dvType}"],
                dtype=self.dtype,
            )

    def addPointSet(self, points, ptName, **kwargs):
        """
        Add a set of coordinates to DVGeometry
        The is the main way that geometry in the form of a coordinate list is given to DVGeometry
        to be manipulated.
        This assumes...
            - Trailing edge is vertical or sharp and at maximum x values
            - The geometry is exclusively an extruded shape (no spanwise changes allowed)
            - The airfoil's leading edge is on the left (min x) and trailing edge is
              on the right (max x)
            - The airfoil's leading edge is at y (or idxVertical) equals zero (within :math:`10^{-2}`)

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed. These coordinates *should* all project into the interior of the geometry.
        ptName : str
            A user supplied name to associate with the set of coordinates.
            This name will need to be provided when updating the coordinates or when
            getting the derivatives of the coordinates.
        kwargs
            Any other parameters are ignored.
        """
        # Convert points to the type specified at initialization (with isComplex) and store the points
        points = points.astype(self.dtype)

        # Check that all points are within the airfoil x bounds
        if np.any(points[:, self.xIdx] < self.xMin) or np.any(points[:, self.xIdx] > self.xMax):
            raise ValueError(
                f'Points in the point set "{ptName}" have x coordinates outside'
                + f"the min and max x values in the initial dat file ({self.xMin} and {self.xMax})"
            )
        self.updated[ptName] = False
        self.points[ptName] = {
            "points": points,
            "xMax": self.xMax.copy(),
            "xMin": self.xMin.copy(),
            "thicknessTE": self.thicknessTE.copy(),
        }

        # Determine which points are on the upper and lower surfaces
        self.points[ptName]["upper"], self.points[ptName]["lower"] = self._splitUpperLower(points)

        # If debug mode is on, plot the upper and lower surface points
        if self.debug:
            fig = plt.figure()
            plt.scatter(
                self.points[ptName]["points"][:, self.xIdx][self.points[ptName]["upper"]],
                self.points[ptName]["points"][:, self.yIdx][self.points[ptName]["upper"]],
                c="b",
            )
            plt.scatter(
                self.points[ptName]["points"][:, self.xIdx][self.points[ptName]["lower"]],
                self.points[ptName]["points"][:, self.yIdx][self.points[ptName]["lower"]],
                c="r",
            )
            plt.scatter(
                self.points[ptName]["points"][:, self.xIdx],
                self.points[ptName]["points"][:, self.yIdx],
                s=3,
                c="k",
                zorder=3,
            )
            plt.legend(["Upper", "Lower"])
            plt.show()
            plt.close(fig)

    def addDV(self, dvName, dvType, dvNum=None, lower=None, upper=None, scale=1.0, default=None):
        """
        Add design variables to the DVGeometryCST object.

        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group
        dvType : str
            Define the type of CST design variable being added. Either the upper/lower surface class shape
            parameter DV can be defined (e.g., ``"N1_upper"``), or the DV for both the upper and lower surfaces' class shape
            parameter can be defined (e.g., ``"N1"``), but not both. The options (not case sensitive) are
                - ``"upper"``: upper surface CST coefficients (specify ``dvNum`` to define how many)
                - ``"lower"``: lower surface CST coefficients (specify ``dvNum`` to define how many)
                - ``"N1"``: first class shape parameter for both upper and lower surfaces (adds a single DV)
                - ``"N2"``: second class shape parameter for both upper and lower surfaces (adds a single DV)
                - ``"N1_upper"``: first class shape parameters for upper surface (adds a single DV)
                - ``"N1_lower"``: first class shape parameters for lower surface (adds a single DV)
                - ``"N2_upper"``: second class shape parameters for upper surface (adds a single DV)
                - ``"N2_lower"``: second class shape parameters for lower surface (adds a single DV)
                - ``"chord"``: chord length in whatever units the point set length is defined and scaled
                  to keep the leading edge at the same position (adds a single DV)
        dvNum : int
            If dvType is ``"upper"`` or ``"lower"``, use ``dvNum`` to specify the number of
            CST parameters to use. This must be given by the user for upper and lower DVs.
        lower : float or ndarray, optional
            The upper bound for the variable(s). This will be applied to
            all shape variables
        upper : float or ndarray, optional
            The upper bound for the variable(s). This will be applied to
            all shape variables
        scale : float, optional
            The scaling of the variables. A good approximate scale to
            start with is approximately 1.0/(upper-lower). This gives
            variables that are of order ~1.0.
        default : ndarray, optional
            Default value for design variable (must be same length as number of DVs added).

        Returns
        -------
        N : int
            The number of design variables added.
        """
        # Do some error checking
        if dvType.lower() not in [
            "upper",
            "lower",
            "n1",
            "n2",
            "n1_upper",
            "n1_lower",
            "n2_upper",
            "n2_lower",
            "chord",
        ]:
            raise ValueError(
                'dvType must be one of "upper", "lower", "N1", "N2", "N1_upper", "N1_lower", '
                + f'"N2_upper", "N2_lower", or "chord", not {dvType}'
            )
        dvType = dvType.lower()

        if dvType in ["upper", "lower"]:
            if dvNum is None:
                raise ValueError('dvNum must be specified if dvType is "upper" or "lower"')
        else:
            dvNum = 1

        # Check that a duplicate DV doesn't already exist
        if dvType in ["n1", "n2", "n1_upper", "n1_lower", "n2_upper", "n2_lower"]:
            if dvType in ["n1", "n2"]:  # if either of these is added, the individual lower and upper params can't be
                if self.DVExists[dvType + "_lower"]:
                    raise ValueError(f'"{dvType}" cannot be added when "{dvType}_lower" already exists')
                elif self.DVExists[dvType + "_upper"]:
                    raise ValueError(f'"{dvType}" cannot be added when "{dvType}_upper" already exists')
                else:
                    self.DVExists[dvType + "_lower"] = True
                    self.DVExists[dvType + "_upper"] = True
            else:  # the parameter that controls both the upper and lower surfaces simultaneously can't be added
                param = dvType.split("_")[0]  # either N1 or N2
                if self.DVExists[dvType]:
                    raise ValueError(f'"{dvType}" cannot be added when "{param}" or "{dvType}" already exist')
                else:
                    self.DVExists[dvType] = True
        else:
            if self.DVExists[dvType]:
                raise ValueError(f'"{dvType}" design variable already exists')
            else:
                self.DVExists[dvType] = True

        if dvName in self.DVs.keys():
            raise ValueError(f'A design variable with the name "{dvName}" already exists')

        # Set the default value
        if default is None:
            # If the number of CST parameters for the upper or lower surface is different than the
            # existing default, refit the airfoil with the new number of CST coefficients
            if dvType in ["upper", "lower"] and self.defaultDV[dvType].size != dvNum:
                self.defaultDV[dvType] = self.computeCSTfromCoords(
                    self.foilCoords[self.idxFoil[dvType], self.xIdx],
                    self.foilCoords[self.idxFoil[dvType], self.yIdx],
                    dvNum,
                    N1=self.defaultDV[f"n1_{dvType}"],
                    N2=self.defaultDV[f"n2_{dvType}"],
                    dtype=self.dtype,
                )
            default = self.defaultDV[dvType]
        else:
            if not isinstance(default, np.ndarray):
                raise ValueError(f"The default value for the {dvName} DV must be a NumPy array, not a {type(default)}")
            default = default.flatten()
            if default.size != dvNum:
                raise ValueError(
                    f"The default value for the {dvName} DV must have a length of {dvNum}, not {default.size}"
                )

            # Set new default
            self.defaultDV[dvType] = default.astype(self.dtype)
            if dvType in ["n1", "n2"]:
                self.defaultDV[f"{dvType}_lower"] = default.astype(self.dtype)
                self.defaultDV[f"{dvType}_upper"] = default.astype(self.dtype)

        # Add the DV to the internally-stored list
        self.DVs[dvName] = cstDV(
            name=dvName,
            value=default.astype(self.dtype),
            nVal=dvNum,
            lower=lower,
            upper=upper,
            scale=scale,
            dvType=dvType,
        )

        return dvNum

    def setDesignVars(self, dvDict):
        """
        Standard routine for setting design variables from a design variable dictionary.

        Parameters
        ----------
        dvDict : dict
            Dictionary of design variables. The keys of the dictionary must correspond to the design variable names.
            Any additional keys in the dictionary are simply ignored.
        """
        for dvName, dvVal in dvDict.items():
            if dvName in self.DVs:
                if dvVal.shape != self.DVs[dvName].value.shape:
                    raise ValueError(
                        f'Input shape of {dvVal.shape} for the DV named "{dvName}" does '
                        + f"not match the DV's shape of {self.DVs[dvName].value.shape}"
                    )
                self.DVs[dvName].value = dvVal.astype(self.dtype)

        # Flag all the pointSets as not being up to date
        for pointSet in self.updated:
            self.updated[pointSet] = False

    def getValues(self):
        """
        Generic routine to return the current set of design variables.
        Values are returned in a dictionary format that would be suitable for a subsequent call to setValues()

        Returns
        -------
        dvDict : dict
            Dictionary of design variables
        """
        # Format the dictonary into the desired shape
        DVs = {}
        for dvName in self.DVs.keys():
            DVs[dvName] = self.DVs[dvName].value

        return DVs

    def getVarNames(self, **kwargs):
        """
        Return a list of the design variable names. This is typically used when specifying a wrt= argument for pyOptSparse.

        Examples
        --------
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        return list(self.DVs.keys())

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, **kwargs):
        r"""
        This function computes sensitivity information.
        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}}^T \frac{dI}{d_{pt}}`

        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)
            This is the total derivative of the objective or function of interest with respect to the coordinates in 'ptSetName'.
            This can be a single array of size (Npt, 3) **or** a group of N vectors of size (N, Npt, 3).
            If you have many to do, it is faster to do many at once.
        ptSetName : str
            The name of set of points we are dealing with
        kwargs
            Any other parameters ignored, but this is maintained to allow the same
            interface as other DVGeo implementations.

        Returns
        -------
        dIdxDict : dict
            The dictionary containing the derivatives, suitable for pyOptSparse
        """
        # Unpack some useful variables
        desVars = self._unpackDVs()
        ptsX = self.points[ptSetName]["points"][:, self.xIdx]
        xMax = self.points[ptSetName]["xMax"]
        xMin = self.points[ptSetName]["xMin"]
        scaledX = (ptsX - xMin) / (xMax - xMin)
        idxUpper = self.points[ptSetName]["upper"]
        idxLower = self.points[ptSetName]["lower"]
        funcSens_local = {}

        # If dIdpt is a group of vectors, reorder the axes so it
        # is handled properly by the matrix multiplies
        dim = dIdpt.shape
        if len(dim) == 3:
            dIdpt = np.moveaxis(dIdpt, 0, -1)

        for dvName, DV in self.DVs.items():
            dvType = DV.type

            if dvType == "upper":
                dydUpperCST = self.computeCSTdydw(
                    scaledX[idxUpper], desVars["n1_upper"], desVars["n2_upper"], desVars["upper"], dtype=self.dtype
                )
                dydUpperCST *= desVars["chord"]
                funcSens_local[dvName] = dydUpperCST @ dIdpt[idxUpper, self.yIdx]
            elif dvType == "lower":
                dydLowerCST = self.computeCSTdydw(
                    scaledX[idxLower], desVars["n1_lower"], desVars["n2_lower"], desVars["lower"], dtype=self.dtype
                )
                dydLowerCST *= desVars["chord"]
                funcSens_local[dvName] = dydLowerCST @ dIdpt[idxLower, self.yIdx]
            elif dvType == "n1_upper":
                funcSens_local[dvName] = (
                    desVars["chord"]
                    * self.computeCSTdydN1(
                        scaledX[idxUpper], desVars["n1_upper"], desVars["n2_upper"], desVars["upper"], dtype=self.dtype
                    )
                    @ dIdpt[idxUpper, self.yIdx]
                )
            elif dvType == "n2_upper":
                funcSens_local[dvName] = (
                    desVars["chord"]
                    * self.computeCSTdydN2(
                        scaledX[idxUpper], desVars["n1_upper"], desVars["n2_upper"], desVars["upper"], dtype=self.dtype
                    )
                    @ dIdpt[idxUpper, self.yIdx]
                )
            elif dvType == "n1_lower":
                funcSens_local[dvName] = (
                    desVars["chord"]
                    * self.computeCSTdydN1(
                        scaledX[idxLower], desVars["n1_lower"], desVars["n2_lower"], desVars["lower"], dtype=self.dtype
                    )
                    @ dIdpt[idxLower, self.yIdx]
                )
            elif dvType == "n2_lower":
                funcSens_local[dvName] = (
                    desVars["chord"]
                    * self.computeCSTdydN2(
                        scaledX[idxLower], desVars["n1_lower"], desVars["n2_lower"], desVars["lower"], dtype=self.dtype
                    )
                    @ dIdpt[idxLower, self.yIdx]
                )
            elif dvType == "n1":
                funcSens_local[dvName] = (
                    desVars["chord"]
                    * self.computeCSTdydN1(
                        scaledX[idxUpper], desVars["n1_upper"], desVars["n2_upper"], desVars["upper"], dtype=self.dtype
                    )
                    @ dIdpt[idxUpper, self.yIdx]
                )
                funcSens_local[dvName] += (
                    desVars["chord"]
                    * self.computeCSTdydN1(
                        scaledX[idxLower], desVars["n1_lower"], desVars["n2_lower"], desVars["lower"], dtype=self.dtype
                    )
                    @ dIdpt[idxLower, self.yIdx]
                )
            elif dvType == "n2":
                funcSens_local[dvName] = (
                    desVars["chord"]
                    * self.computeCSTdydN2(
                        scaledX[idxUpper], desVars["n1_upper"], desVars["n2_upper"], desVars["upper"], dtype=self.dtype
                    )
                    @ dIdpt[idxUpper, self.yIdx]
                )
                funcSens_local[dvName] += (
                    desVars["chord"]
                    * self.computeCSTdydN2(
                        scaledX[idxLower], desVars["n1_lower"], desVars["n2_lower"], desVars["lower"], dtype=self.dtype
                    )
                    @ dIdpt[idxLower, self.yIdx]
                )
            else:  # chord
                dydchord = self.points[ptSetName]["points"][:, self.yIdx] / desVars["chord"]
                dxdchord = (ptsX - xMin) / desVars["chord"]
                funcSens_local[dvName] = dxdchord @ dIdpt[:, self.xIdx] + dydchord @ dIdpt[:, self.yIdx]

        # If the axes were reordered to handle a group of dIdpt vectors,
        # switch them back to the expected order for output
        if len(dim) == 3:
            for dvName in funcSens_local.keys():
                funcSens_local[dvName] = np.moveaxis(np.atleast_2d(funcSens_local[dvName]), 0, -1)

        if comm:
            funcSens = {}
            for dvName in funcSens_local.keys():
                funcSens[dvName] = comm.allreduce(funcSens_local[dvName], op=MPI.SUM)
        else:
            funcSens = funcSens_local

        return funcSens

    def totalSensitivityProd(self, vec, ptSetName, **kwargs):
        r"""
        This function computes sensitivity information.
        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}} \times\mathrm{vec}`.
        This is useful for forward AD mode.

        Parameters
        ----------
        vec : dictionary whose keys are the design variable names, and whose
              values are the derivative seeds of the corresponding design variable.
        ptSetName : str
            The name of set of points we are dealing with
        kwargs
            Any other parameters ignored, but this is maintained to allow the same
            interface as other DVGeo implementations.

        Returns
        -------
        xsdot : array (Nx3)
            Array with derivative seeds of the surface nodes.
        """
        # Unpack some useful variables
        desVars = self._unpackDVs()
        ptsX = self.points[ptSetName]["points"][:, self.xIdx]
        xMax = self.points[ptSetName]["xMax"]
        xMin = self.points[ptSetName]["xMin"]
        scaledX = (ptsX - xMin) / (xMax - xMin)
        idxUpper = self.points[ptSetName]["upper"]
        idxLower = self.points[ptSetName]["lower"]
        idxTE = np.full((self.points[ptSetName]["points"].shape[0],), True, dtype=bool)
        idxTE[idxUpper] = False
        idxTE[idxLower] = False
        xsdot = np.zeros_like(self.points[ptSetName]["points"], dtype=self.dtype)

        for dvName, dvSeed in vec.items():
            dvType = self.DVs[dvName].type

            if dvType == "upper":
                dydUpperCST = self.computeCSTdydw(
                    scaledX[idxUpper], desVars["n1_upper"], desVars["n2_upper"], desVars["upper"], dtype=self.dtype
                )
                dydUpperCST *= desVars["chord"]
                xsdot[idxUpper, self.yIdx] += dydUpperCST.T @ dvSeed
            if dvType == "lower":
                dydLowerCST = self.computeCSTdydw(
                    scaledX[idxLower], desVars["n1_lower"], desVars["n2_lower"], desVars["lower"], dtype=self.dtype
                )
                dydLowerCST *= desVars["chord"]
                xsdot[idxLower, self.yIdx] += dydLowerCST.T @ dvSeed
            if dvType == "n1_upper" or dvType == "n1":
                xsdot[idxUpper, self.yIdx] += (
                    dvSeed
                    * desVars["chord"]
                    * self.computeCSTdydN1(
                        scaledX[idxUpper], desVars["n1_upper"], desVars["n2_upper"], desVars["upper"], dtype=self.dtype
                    )
                )
            if dvType == "n2_upper" or dvType == "n2":
                xsdot[idxUpper, self.yIdx] += (
                    dvSeed
                    * desVars["chord"]
                    * self.computeCSTdydN2(
                        scaledX[idxUpper], desVars["n1_upper"], desVars["n2_upper"], desVars["upper"], dtype=self.dtype
                    )
                )
            if dvType == "n1_lower" or dvType == "n1":
                xsdot[idxLower, self.yIdx] += (
                    dvSeed
                    * desVars["chord"]
                    * self.computeCSTdydN1(
                        scaledX[idxLower], desVars["n1_lower"], desVars["n2_lower"], desVars["lower"], dtype=self.dtype
                    )
                )
            if dvType == "n2_lower" or dvType == "n2":
                xsdot[idxLower, self.yIdx] += (
                    dvSeed
                    * desVars["chord"]
                    * self.computeCSTdydN2(
                        scaledX[idxLower], desVars["n1_lower"], desVars["n2_lower"], desVars["lower"], dtype=self.dtype
                    )
                )
            if dvType == "chord":
                dydchord = self.points[ptSetName]["points"][:, self.yIdx] / desVars["chord"]
                dxdchord = (ptsX - xMin) / desVars["chord"]
                xsdot[:, self.yIdx] += dvSeed * dydchord
                xsdot[:, self.xIdx] += dvSeed * dxdchord

        return xsdot

    def addVariablesPyOpt(self, optProb):
        """
        Add the current set of variables to the optProb object.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which variables are added
        """
        # Add design variables to the problem
        for DV in self.DVs.values():
            optProb.addVarGroup(
                DV.name,
                DV.nVal,
                "c",
                value=DV.value,
                lower=DV.lower,
                upper=DV.upper,
                scale=DV.scale,
            )

    def update(self, ptSetName, **kwargs):
        """
        This is the main routine for returning coordinates that have
        been updated by design variables.

        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the
            given in an :func:`addPointSet()` call.
        kwargs
            Any other parameters ignored, but this is maintained to allow the same
            interface as other DVGeo implementations.

        Returns
        -------
        points : ndarray (N x 3)
            Updated point set coordinates.
        """
        desVars = self._unpackDVs()

        # Unpack the points to make variable names more accessible
        idxUpper = self.points[ptSetName]["upper"]
        idxLower = self.points[ptSetName]["lower"]
        idxTE = np.full((self.points[ptSetName]["points"].shape[0],), True, dtype=bool)
        idxTE[idxUpper] = False
        idxTE[idxLower] = False
        points = self.points[ptSetName]["points"]
        ptsX = points[:, self.xIdx]
        ptsY = points[:, self.yIdx]
        xMax = self.points[ptSetName]["xMax"]
        xMin = self.points[ptSetName]["xMin"]
        thicknessTE = self.points[ptSetName]["thicknessTE"]

        # Scale the airfoil to the range 0 to 1 in x direction
        shift = xMin
        chord = xMax - xMin
        scaledX = (ptsX - shift) / chord
        yTE = thicknessTE / chord / 2  # half the scaled trailing edge thickness

        ptsY[idxUpper] = desVars["chord"] * self.computeCSTCoordinates(
            scaledX[idxUpper], desVars["n1_upper"], desVars["n2_upper"], desVars["upper"], yTE, dtype=self.dtype
        )
        ptsY[idxLower] = desVars["chord"] * self.computeCSTCoordinates(
            scaledX[idxLower], desVars["n1_lower"], desVars["n2_lower"], desVars["lower"], -yTE, dtype=self.dtype
        )
        ptsY[idxTE] *= desVars["chord"] / chord

        # Scale the chord according to the chord DV
        points[:, self.xIdx] = (points[:, self.xIdx] - shift) * desVars["chord"] / chord + shift

        # Scale the point set's properties based on the new chord length
        self.points[ptSetName]["xMax"] = (xMax - shift) * desVars["chord"] / chord + shift
        self.points[ptSetName]["thicknessTE"] *= desVars["chord"] / chord

        self.updated[ptSetName] = True

        return points.copy()

    def getNDV(self):
        """
        Return the total number of design variables this object has.

        Returns
        -------
        nDV : int
            Total number of design variables
        """
        nDV = 0
        for DV in self.DVs.values():
            nDV += DV.nVal
        return nDV

    def printDesignVariables(self):
        """
        Print a formatted list of design variables to the screen
        """
        print("\nDVGeometryCST design variables")
        print("==============================")
        for DV in self.DVs.values():
            print(f"{DV.name} ({DV.type} type): {DV.value}")
        print("")

    def _unpackDVs(self):
        """
        Return the parameters needed for the airfoil shape calculation
        based on the DVs and default values. This requires a few extra
        checks to handle the multiple ways of parameterizing the class
        shape variables.

        Returns
        -------
        desVars : dict
            Dictionary containing the following airfoil shape parameters:
                `"upper"`: upper surface CST coefficients
                `"lower"`: lower surface CST coefficients
                `"n1_lower"`: first class shape parameter on lower surface
                `"n2_lower"`: second class shape parameter on lower surface
                `"n1_upper"`: first class shape parameter on upper surface
                `"n2_upper"`: second class shape parameter on upper surface
                `"chord"`: chord length
        """
        desVars = {}
        desVars["upper"] = self.defaultDV["upper"].copy()
        desVars["lower"] = self.defaultDV["lower"].copy()
        desVars["n1_upper"] = self.defaultDV["n1_upper"].copy()
        desVars["n2_upper"] = self.defaultDV["n2_upper"].copy()
        desVars["n1_lower"] = self.defaultDV["n1_lower"].copy()
        desVars["n2_lower"] = self.defaultDV["n2_lower"].copy()
        desVars["chord"] = self.defaultDV["chord"].copy()

        for DV in self.DVs.values():
            if DV.type in ["n1", "n2"]:
                desVars[f"{DV.type}_upper"] = DV.value
                desVars[f"{DV.type}_lower"] = DV.value
            else:
                desVars[DV.type] = DV.value

        return desVars

    def _splitUpperLower(self, points):
        """
        Figure out the indices of points on the upper and lower
        surfaces of the airfoil. This requires that the attributes
        self.xMax, self.lowerSpline, self.upperSpline, self.xIdx,
        and self.yIdx have already been set.

        Parameters
        ----------
        points : ndarray (Npts x 3)
            Point array to separate upper and lower surfaces

        Returns
        -------
        ndarray (1D)
            Indices of upper surface points (correspond to rows in points)
        ndarray (1D)
            Indices of lower surface points (correspond to rows in points)
        """
        # Determine which surface each point is on based on which spline it is closer to
        _, upperDist = self.upperSpline.projectPoint(points[:, [self.xIdx, self.yIdx]])
        _, lowerDist = self.lowerSpline.projectPoint(points[:, [self.xIdx, self.yIdx]])
        upperDist = np.linalg.norm(upperDist, axis=1)
        lowerDist = np.linalg.norm(lowerDist, axis=1)

        upperBool = upperDist < lowerDist
        lowerBool = np.logical_not(upperBool)

        # Find any trailing edge points that are not on the upper or lower
        # surface (on the surface of the blunt trailing edge)
        isTE = points[:, self.xIdx] == self.xMax
        tol = 1e-13
        isNotUpperSurface = points[:, self.yIdx] < self.yUpperTE - tol
        isNotLowerSurface = points[:, self.yIdx] > self.yLowerTE + tol
        isNotSurface = np.logical_and(isNotLowerSurface, isNotUpperSurface)
        idxTE = np.where(np.logical_and(isTE, isNotSurface))[0]

        # Remove points on the blunt trailing edge
        upperBool[idxTE] = False
        lowerBool[idxTE] = False

        return np.where(upperBool)[0], np.where(lowerBool)[0]

    @staticmethod
    def computeCSTCoordinates(x, N1, N2, w, yte, dtype=float):
        """
        Compute the vertical coordinates of a CST curve.

        This function assumes x has been normalized to the range [0,1].

        Parameters
        ----------
        x : ndarray (# pts,)
            x coordinates at which to compute the CST curve height
        N1 : float
            First class shape parameter
        N2 : float
            Second class shape parameter
        w : ndarray (# coeff,)
            CST coefficient array
        yte : float
            y coordinate of the trailing edge (used to define trailing edge thickness).
            Note that the trailing edge will be twice this thick, assuming the same ``yte``
            value is used for both the upper and lower surfaces.
        dtype : type, optional
            Type for instantiated arrays, by default float

        Returns
        -------
        ndarray (# pts,)
            y coordinates of the CST curve
        """
        C = DVGeometryCST.computeClassShape(x, N1, N2, dtype=dtype)
        S = DVGeometryCST.computeShapeFunctions(x, w, dtype=dtype)
        return C * S.sum(axis=0) + yte * x

    @staticmethod
    def computeClassShape(x, N1, N2, dtype=float):
        """
        Compute the class shape of a CST curve

        Parameters
        ----------
        x : ndarray (# pts,)
            x coordinates at which to compute the CST curve height
        N1 : float
            First class shape parameter
        N2 : float
            Second class shape parameter
        dtype : type, optional
            Type for instantiated arrays, by default float

        Returns
        -------
        ndarray (# pts,)
            y coordinates of the class shape
        """
        C = np.zeros_like(x, dtype=dtype)

        # 0 to the power of a complex number is undefined, so anywhere
        # x is 0 or 1, just keep C as zero (doesn't change the result for real)
        mask = np.logical_and(x != 0.0, x != 1.0)
        C[mask] = x[mask] ** N1 * (1.0 - x[mask]) ** N2

        return C

    @staticmethod
    def computeShapeFunctions(x, w, dtype=float):
        """Compute the Bernstein polynomial shape function of a CST curve

        This function assumes x has been normalized to the range [0,1].

        Parameters
        ----------
        x : ndarray (# pts,)
            x coordinates at which to compute the CST curve height
        w : ndarray (# coeff,)
            CST coefficient array
        dtype : type, optional
            Type for instantiated arrays, by default float

        Returns
        -------
        ndarray (# coeff, # pts)
            Bernstein polynomials for each CST coefficient
        """
        numCoeffs = len(w)
        order = numCoeffs - 1
        S = np.zeros((numCoeffs, len(x)), dtype=dtype)
        facts = factorial(np.arange(0, order + 1))
        for i in range(numCoeffs):
            binom = facts[-1] / (facts[i] * facts[order - i])
            S[i] = w[i] * binom * x ** (i) * (1.0 - x) ** (order - i)
        return S

    @staticmethod
    def computeCSTdydw(x, N1, N2, w, dtype=float):
        r"""Compute the derivatives of the height of a CST curve with respect to the shape function coefficients

        Given :math:`y = C(x) * sum [w_i * p_i(x)]`
        :math:`\frac{dy}{dw_i} = C(x) * p_i(x)`

        This function assumes x has been normalized to the range [0,1].

        Only the shape of w is used, not the values.

        Parameters
        ----------
        x : ndarray (# pts,)
            x coordinates at which to compute the CST curve height
        N1 : float
            First class shape parameter
        N2 : float
            Second class shape parameter
        w : ndarray (# coeff,)
            CST coefficient array
        dtype : type, optional
            Type for instantiated arrays, by default float

        Returns
        -------
        ndarray (# coeff, # pts)
            Derivatives of the y coordinates with respect to the CST coefficients
        """
        C = DVGeometryCST.computeClassShape(x, N1, N2, dtype=dtype)
        S = DVGeometryCST.computeShapeFunctions(x, np.ones_like(w), dtype=dtype)
        return C * S

    @staticmethod
    def computeCSTdydN1(x, N1, N2, w, dtype=float):
        r"""Compute the drivatives of the height of a CST curve with respect to N1

        Given :math:`y = C(x, N1, N2) * S(x)`
        :math:`\frac{dy}{dN1} = S(x) * \frac{dC}{dN1} = S(x) * C(x, N1, N2) * \ln{x}

        This function assumes x has been normalised to the range [0,1].

        Parameters
        ----------
        x : ndarray (# pts,)
            x coordinates at which to compute the CST curve height
        N1 : float
            First class shape parameter
        N2 : float
            Second class shape parameter
        w : ndarray (# coeff,)
            CST coefficient array
        dtype : type, optional
            Type for instantiated arrays, by default float

        Returns
        -------
        ndarray (# pts,)
            Derivative of the y coordinates with respect to the first class shape parameter
        """
        C = DVGeometryCST.computeClassShape(x[x != 0.0], N1, N2, dtype=dtype)
        S = DVGeometryCST.computeShapeFunctions(x[x != 0.0], w, dtype=dtype)
        dydN1 = np.zeros_like(x, dtype=dtype)
        dydN1[x != 0.0] = np.sum(S, axis=0) * C * np.log(x[x != 0.0])
        return dydN1

    @staticmethod
    def computeCSTdydN2(x, N1, N2, w, dtype=float):
        r"""Compute the drivatives of the height of a CST curve with respect to N2

        Given :math:`y = C(x, N1, N2) * S(x)`
        :math:`\frac{dy}{dN2} = S(x) * \frac{dC}{dN2} = S(x) * C(x, N1, N2) * \ln(1-x)`

        This function assumes x has been normalised to the range [0,1].

        Parameters
        ----------
        x : ndarray (# pts,)
            x coordinates at which to compute the CST curve height
        N1 : float
            First class shape parameter
        N2 : float
            Second class shape parameter
        w : ndarray (# coeff,)
            CST coefficient array
        dtype : type, optional
            Type for instantiated arrays, by default float

        Returns
        -------
        ndarray (# pts,)
            Derivative of the y coordinates with respect to the second class shape parameter
        """
        C = DVGeometryCST.computeClassShape(x[x != 1.0], N1, N2, dtype=dtype)
        S = DVGeometryCST.computeShapeFunctions(x[x != 1.0], w, dtype=dtype)
        dydN2 = np.zeros_like(x, dtype=dtype)
        dydN2[x != 1.0] = np.sum(S, axis=0) * C * np.log(1 - x[x != 1.0])
        return dydN2

    @staticmethod
    def computeCSTfromCoords(xCoord, yCoord, nCST, N1=0.5, N2=1.0, dtype=float):
        """
        Compute the CST coefficients that fit a set of airfoil
        coordinates (either for the upper or lower surface, not both).

        This function internally normalizes the x and y-coordinates.

        Parameters
        ----------
        xCoord : ndarray
            Upper or lower surface airfoil x-coordinates (same length
            as yCoord vector).
        yCoord : ndarray
            Upper or lower surface airfoil y-coordinates (same length
            as xCoord vector).
        nCST : int
            Number of CST coefficients to fit.
        N1 : float, optional
            First class shape parameter to assume in fitting, by default 0.5
        N2 : float, optional
            Second class shape parameter to assume in fitting, by default 1.0
        dtype : type, optional
            Type for instantiated arrays, by default float

        Returns
        -------
        np.ndarray (nCST,)
            CST coefficients fit to the airfoil surface.
        """
        # Normalize x and y
        chord = np.max(xCoord) - np.min(xCoord)
        xCoord = (xCoord - np.min(xCoord)) / chord
        yCoord /= chord

        # Compute the coefficients via linear least squares
        dydw = DVGeometryCST.computeCSTdydw(xCoord, N1, N2, np.ones(nCST), dtype=dtype)
        w = np.linalg.lstsq(dydw.T, yCoord, rcond=None)[0]
        return w

    @staticmethod
    def plotCST(upperCoeff, lowerCoeff, N1=0.5, N2=1.0, nPts=100, ax=None, **kwargs):
        """Simple utility to generate a plot from CST coefficients.

        Parameters
        ----------
        upperCoeff : ndarray
            One dimensional array of CST coefficients for the upper surface.
        lowerCoeff : ndarray
            One dimensional array of CST coefficients for the lower surface.
        N1 : float
            First class shape parameter.
        N2 : float
            Second class shape parameter.
        nPts : int, optional
            Number of coordinates to compute on each surface.
        ax : matplotlib Axes, optional
            Axes on which to plot airfoil.
        **kwargs
            Keyword arguments passed to matplotlib.pyplot.plot

        Returns
        -------
        matplotlib Axes
            Axes with airfoil plotted
        """
        if not pltImport:
            raise ImportError("matplotlib could not be imported and is required for plotCST")

        if ax is None:
            _ = plt.figure()
            ax = plt.gca()

        x = np.linspace(0, 1, nPts)
        yUpper = DVGeometryCST.computeCSTCoordinates(x, N1, N2, upperCoeff, 0.0)
        yLower = DVGeometryCST.computeCSTCoordinates(x, N1, N2, lowerCoeff, 0.0)

        ax.plot(x, yUpper, **kwargs)
        ax.plot(x, yLower, **kwargs)
        ax.set_aspect("equal")

        return ax
