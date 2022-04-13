"""
==============================================================================
DVGeo: CST Parameterisation
==============================================================================
@Author  :   Alasdair Christison Gray, Eytan Adler
@Description : A DVGeo implementation based on the Class-Shape Transformation method
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
from collections import OrderedDict
from copy import deepcopy

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from mpi4py import MPI
from scipy.special import factorial, comb
from scipy.spatial import Delaunay

# ==============================================================================
# Extension modules
# ==============================================================================


class DVGeometryCST:
    """
    This class implements a 2D geometry parameterisation based on Brenda Kulfan's CST (Class-Shape Transformation) method.
    This class can work with 3D coordinates but will only change the point coordinates in one direction.

    The CST equation is as follows:

    :math:`y(x) = C(x) * S(x) + y_\text{te}x`

    Where C is the class function:

    :math:`C(x) = (x^N1 + (1 - x)^N2)`

    And S is the shape function, in this case a summation of Bernstein polynomials:

    :math:`S(x) = \sum_i^n w_i \binom{n}{i}x^i(1-x)^{n-i}`

    Here x is the normalized chordwise coordinate, ranging from 0 to 1 from front to the rear of the shape.

    This class
    """

    def __init__(self, idxChord=0, idxVertical=1, comm=MPI.COMM_WORLD):
        """
        Initialize DVGeometryCST.

        Parameters
        ----------
        idxChord : int, optional
            Index of the column in the point set to use as the chordwise (x in CST) coordinates, by default 0
        idxVertical : int, optional
            Index of the column in the point set to use as the vertical (y in CST) airfoil coordinates, by default 1
        comm : MPI communicator
            Communicator for DVGeometryCST instance
        """
        self.points = OrderedDict()  # For each point set, it contains a dictionary with the coordinates,
        # indices of upper, lower, and trailing edge points, and the minimum and maximum chordwise coordinates
        self.updated = {}
        self.xIdx = idxChord
        self.yIdx = idxVertical
        self.comm = comm

        # Store the DVs and flags to determine if the limited options have already been specified
        # Each DV in the DVs dictionary (with the key as the DV name) contains
        #   "type": the DV's type among "upper", "lower", "n1", "n2", "n1_upper",
        #           "n1_lower", "n2_upper", "n2_lower", and "chord"
        #   "value": the DV's value, initialized to zero(s)
        #   "lower": lower bound
        #   "upper": upper bound
        #   "scale": variable scaling for optimizer
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
        self.rootDefaultDV = {
            "upper": np.array([0.17356, 0.14769, 0.17954, 0.12373, 0.16701, 0.12967, 0.14308, 0.13890]),  # NACA 0012
            "lower": -np.array([0.17356, 0.14769, 0.17954, 0.12373, 0.16701, 0.12967, 0.14308, 0.13890]),  # NACA 0012
            "n1_upper": np.array([0.5]),
            "n2_upper": np.array([1.0]),
            "n1_lower": np.array([0.5]),
            "n2_lower": np.array([1.0]),
            "n1": np.array([0.5]),
            "n2": np.array([1.0]),
            "chord": np.array([1.0]),
        }

        # Default DVs specific to each point set
        self.defaultDVs = {}

    def addPointSet(self, points, ptName, **kwargs):
        """
        Add a set of coordinates to DVGeometry
        The is the main way that geometry in the form of a coordinate list is given to DVGeometry to be manipulated.
        This assumes...
            - Trailing edge is vertical or sharp and at maximum x (or idxChord) values
            - The geometry is exclusively an extruded shape (no spanwise changes allowed)
            - The airfoil's leading edge is on the left (min x or idxChord) and trailing edge is
              on the right (max x or idxChord)
            - The airfoil's leading edge is at y (or idxVertical) equals zero (within 1e-2)
            - The current approach to split the upper and lower surfaces fits a 9th order polynomial to the airfoil
              coordinates and the upper surface is anything above and lower, anything below. The downside is this
              method may not work for airfoils with a very thin and cambered trailing edge.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed. These coordinates *should* all project into the interior of the geometry.
        ptName : str
            A user supplied name to associate with the set of coordinates.
            This name will need to be provided when updating the coordinates or when getting the derivatives of the coordinates.
        kwargs
            Any other parameters ignored, but this is maintained to allow the same
            interface as other DVGeo implementations.
        """
        # Only parts of the points may be passed in to each DVGeo instance on each proc, so we must share the entire point cloud
        nCols = 3  # number of dimensions for each point
        NLoc = points.shape[0]  # number of coordinates on current proc
        sizes = np.array(self.comm.allgather(NLoc), dtype="intc")  # number of points on each proc
        disp = np.hstack(([0], np.cumsum(sizes)))[:-1]  # starting index for each part of the distributed point set
        N = np.sum(sizes)  # total points in the point set
        pointsGlobal = np.zeros((N, 3), dtype=float)  # full point set

        # Gather one column at a time
        for col in range(nCols):
            # Copy data into 1D arrays so data is contiguous in memory for MPI
            pointsCol = points[:, col].copy()
            tempColGlobal = np.zeros(N, dtype=float)
            self.comm.Allgatherv([pointsCol, NLoc], [tempColGlobal, sizes, disp, MPI.DOUBLE])

            # Copy resulting column into global point array
            pointsGlobal[:, col] = tempColGlobal.copy()

        # Check that the leading edge is at y = 0
        idxLE = np.argmin(pointsGlobal[:, self.xIdx])
        yLE = pointsGlobal[idxLE, self.yIdx]
        if abs(yLE) > 1e-2:
            raise ValueError(f"Leading edge y (or idxVertical) value must equal zero, not {yLE}")

        # Trailing edge points are at maximum chord
        idxTE = np.where(pointsGlobal[:, self.xIdx] == np.max(pointsGlobal[:, self.xIdx]))[0]

        # The trailing edge point at the maximum y value begins the upper surface, minimum ends lower surface
        idxUpperTE = idxTE[np.argmax(pointsGlobal[idxTE, self.yIdx])]
        idxLowerTE = idxTE[np.argmin(pointsGlobal[idxTE, self.yIdx])]
        thicknessTE = pointsGlobal[idxUpperTE, self.yIdx] - pointsGlobal[idxLowerTE, self.yIdx]

        # Fit a polynomial to the airfoil to approximate the camber line
        # Upper surface is above that line and lower is below
        p = np.polyfit(pointsGlobal[:, self.xIdx], pointsGlobal[:, self.yIdx], 9)
        yCamberLine = np.polyval(p, pointsGlobal[:, self.xIdx])
        upperBool = yCamberLine < pointsGlobal[:, self.yIdx]
        upperBool[idxTE] = False
        upperBool[idxUpperTE] = True
        lowerBool = pointsGlobal[:, self.yIdx] <= yCamberLine
        lowerBool[idxTE] = False
        lowerBool[idxLowerTE] = True

        # Indices of the local points in the pointsGlobal array
        dispLocal = disp[self.comm.rank]  # displacement of current proc in global array
        idxLocal = np.arange(dispLocal, dispLocal + points.shape[0])

        # Add the points
        self.updated[ptName] = False
        self.points[ptName] = {
            "points": points,
            "upper": np.where(upperBool[idxLocal])[0],
            "lower": np.where(lowerBool[idxLocal])[0],
            "thicknessTE": thicknessTE,
            "xMin": np.min(pointsGlobal[:, self.xIdx]),
            "xMax": np.max(pointsGlobal[:, self.xIdx]),
        }

        # Set the default design variables based on the input airfoil
        # TODO: fit CST coefficients to the input airfoil so if upper or lower DVs aren't specified, it will keep the airfoil
        #       also, this will not currently work for CST shape variables with dvNum != 8 because the default is 8 (and will initialize pyOptSparse this way)
        self.defaultDVs[ptName] = deepcopy(self.rootDefaultDV)
        self.defaultDVs[ptName]["chord"] = np.array([self.points[ptName]["xMax"] - self.points[ptName]["xMin"]])

    def addDV(self, dvName, dvType, dvNum=None, lower=None, upper=None, scale=1.0):
        """
        Add one or more local design variables ot the DVGeometry
        object. Local variables are used for small shape modifications.

        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group

        dvType : str
            Define the type of CST design variable being added. Either the upper/lower surface class shape
            parameter DV can be defined (e.g., `"N1_upper"`), or the DV for both the upper and lower surfaces' class shape
            parameter can be defined (e.g., `"N1"`), but not both. The options (not case sensitive) are
                `"upper"`: upper surface CST coefficients (specify `dvNum` to define how many)
                `"lower"`: lower surface CST coefficients (specify `dvNum` to define how many)
                `"N1"`: first class shape parameter for both upper and lower surfaces (adds a single DV)
                `"N2"`: second class shape parameter for both upper and lower surfaces (adds a single DV)
                `"N1_upper"`: first class shape parameters for upper surface (adds a single DV)
                `"N1_lower"`: first class shape parameters for lower surface (adds a single DV)
                `"N2_upper"`: second class shape parameters for upper surface (adds a single DV)
                `"N2_lower"`: second class shape parameters for lower surface (adds a single DV)
                `"chord"`: chord length in whatever units the point set length is defined and scaled
                           to keep the leading edge at the same position (adds a single DV)

        dvNum : int
            If dvType is `"upper"` or `"lower"`, use `dvNum` to specify the number of
            CST parameters to use. This must be given by the user for upper and lower DVs.

        lower : float or ndarray
            The upper bound for the variable(s). This will be applied to
            all shape variables

        upper : float or ndarray
            The upper bound for the variable(s). This will be applied to
            all shape variables

        scale : float
            The scaling of the variables. A good approximate scale to
            start with is approximately 1.0/(upper-lower). This gives
            variables that are of order ~1.0.

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
                f'dvType must be one of "upper", "lower", "N1", "N2", "N1_upper", "N1_lower", '
                + f'"N2_upper", "N2_lower", or "chord" not {dvType}'
            )
        dvType = dvType.lower()

        if dvType in ["upper", "lower"] and dvNum is None:
            raise ValueError(f'dvNum must be specified if dvType is "upper" or "lower"')
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

        # Add the DV to the internally-stored list
        self.DVs[dvName] = {
            "type": dvType,
            "value": self.rootDefaultDV[dvType],
            "lower": lower,
            "upper": upper,
            "scale": scale,
        }

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
                if dvVal.shape != self.DVs[dvName]["value"].shape:
                    raise ValueError(
                        f'Input shape of {dvVal.shape} for the DV named "{dvName}" does '
                        + f"not match the DV's shape of {self.DVs[dvName]['value'].shape}"
                    )
                self.DVs[dvName]["value"] = dvVal

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
            DVs[dvName] = self.DVs[dvName]["value"]

        return DVs

    def pointSetUpToDate(self, ptSetName):
        """
        This is used externally to query if the object needs to update
        its pointset or not. Essentially what happens, is when
        update() is called with a point set, it the self.updated dict
        entry for pointSet is flagged as true. Here we just return
        that flag. When design variables are set, we then reset all
        the flags to False since, when DVs are set, nothing (in
        general) will up to date anymore.

        Parameters
        ----------
        ptSetName : str
            The name of the pointset to check.
        """
        if ptSetName in self.updated:
            return self.updated[ptSetName]
        else:
            return True

    def getVarNames(self, **kwargs):
        """
        Return a list of the design variable names. This is typically used when specifying a wrt= argument for pyOptSparse.

        Examples
        --------
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        return list(self.DVs.keys())

    def writeToFile(self, filename):
        # TODO generalize the writing to files?
        pass

    def totalSensitivity(self, dIdpt, ptSetName, **kwargs):
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
        vars = self._unpackDVs(ptSetName)
        ptsX = self.points[ptSetName]["points"][:, self.xIdx]
        scaledX = (ptsX - self.points[ptSetName]["xMin"]) / (
            self.points[ptSetName]["xMax"] - self.points[ptSetName]["xMin"]
        )
        idxUpper = self.points[ptSetName]["upper"]
        idxLower = self.points[ptSetName]["lower"]
        funcSens = {}

        # TODO: is this what needs to be done in the case where dIdpt is (N, Npt, 3)?
        dim = dIdpt.shape
        if len(dim) == 3:
            dIdpt = dIdpt.reshape((dim[1], dim[2], dim[0]))

        for dvName, DV in self.DVs.items():
            dvType = DV["type"]

            # TODO: these are wrong because they don't take into account the initial scaling of the
            #       x values and then how xMax changes with the new chord
            if dvType == "upper":
                dydUpperCST = self.computeCSTdydw(scaledX[idxUpper], vars["n1_upper"], vars["n2_upper"], vars["upper"])
                dydUpperCST *= vars["chord"]
                funcSens[dvName] = dydUpperCST @ dIdpt[idxUpper, self.yIdx]
            elif dvType == "lower":
                dydLowerCST = self.computeCSTdydw(scaledX[idxLower], vars["n1_lower"], vars["n2_lower"], vars["lower"])
                dydLowerCST *= vars["chord"]
                funcSens[dvName] = dydLowerCST @ dIdpt[idxLower, self.yIdx]
            elif dvType == "n1_upper":
                funcSens[dvName] = (
                    vars["chord"]
                    * self.computeCSTdydN1(scaledX[idxUpper], vars["n1_upper"], vars["n2_upper"], vars["upper"])
                    @ dIdpt[idxUpper, self.yIdx]
                )
            elif dvType == "n2_upper":
                funcSens[dvName] = (
                    vars["chord"]
                    * self.computeCSTdydN2(scaledX[idxUpper], vars["n1_upper"], vars["n2_upper"], vars["upper"])
                    @ dIdpt[idxUpper, self.yIdx]
                )
            elif dvType == "n1_lower":
                funcSens[dvName] = (
                    vars["chord"]
                    * self.computeCSTdydN1(scaledX[idxLower], vars["n1_lower"], vars["n2_lower"], vars["lower"])
                    @ dIdpt[idxLower, self.yIdx]
                )
            elif dvType == "n2_lower":
                funcSens[dvName] = (
                    vars["chord"]
                    * self.computeCSTdydN2(scaledX[idxLower], vars["n1_lower"], vars["n2_lower"], vars["lower"])
                    @ dIdpt[idxLower, self.yIdx]
                )
            elif dvType == "n1":
                funcSens[dvName] = (
                    vars["chord"]
                    * self.computeCSTdydN1(scaledX[idxUpper], vars["n1_upper"], vars["n2_upper"], vars["upper"])
                    @ dIdpt[idxUpper, self.yIdx]
                )
                funcSens[dvName] += (
                    vars["chord"]
                    * self.computeCSTdydN1(scaledX[idxLower], vars["n1_lower"], vars["n2_lower"], vars["lower"])
                    @ dIdpt[idxLower, self.yIdx]
                )
            elif dvType == "n2":
                funcSens[dvName] = (
                    vars["chord"]
                    * self.computeCSTdydN2(scaledX[idxUpper], vars["n1_upper"], vars["n2_upper"], vars["upper"])
                    @ dIdpt[idxUpper, self.yIdx]
                )
                funcSens[dvName] += (
                    vars["chord"]
                    * self.computeCSTdydN2(scaledX[idxLower], vars["n1_lower"], vars["n2_lower"], vars["lower"])
                    @ dIdpt[idxLower, self.yIdx]
                )
            else:  # chord
                dydchord = self.points[ptSetName]["points"][:, self.yIdx] / vars["chord"]
                dxdchord = (ptsX - self.points[ptSetName]["xMin"]) / vars["chord"]
                funcSens[dvName] = dydchord @ dIdpt[:, self.yIdx]
                funcSens[dvName] += dxdchord @ dIdpt[:, self.xIdx]

        if len(dim) == 3:
            for dvName in funcSens.keys():
                funcSens[dvName] = funcSens[dvName].reshape((dim[0], len(self.DVs[dvName]["value"])))

        return funcSens

    def totalSensitivityProd(self, vec, ptSetName, **kwargs):
        """
        This function computes sensitivity information.
        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}} \times\mathrm{vec}`
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
        xsdot : array (Nx3) -> Array with derivative seeds of the surface nodes.
        """
        # Unpack some useful variables
        vars = self._unpackDVs(ptSetName)
        ptsX = self.points[ptSetName]["points"][:, self.xIdx]
        scaledX = (ptsX - self.points[ptSetName]["xMin"]) / (
            self.points[ptSetName]["xMax"] - self.points[ptSetName]["xMin"]
        )
        idxUpper = self.points[ptSetName]["upper"]
        idxLower = self.points[ptSetName]["lower"]
        idxTE = np.full((self.points[ptSetName]["points"].shape[0],), True, dtype=bool)
        idxTE[idxUpper] = False
        idxTE[idxLower] = False
        xsdot = np.zeros(self.points[ptSetName]["points"].shape)

        for dvName, dvSeed in vec.items():
            dvType = self.DVs[dvName]["type"]

            # TODO: these are wrong because they don't take into account the initial scaling of the
            #       x values and then how xMax changes with the new chord
            if dvType == "upper":
                dydUpperCST = self.computeCSTdydw(scaledX[idxUpper], vars["n1_upper"], vars["n2_upper"], vars["upper"])
                dydUpperCST *= vars["chord"]
                xsdot[idxUpper, self.yIdx] += dydUpperCST.T @ dvSeed
            if dvType == "lower":
                dydLowerCST = self.computeCSTdydw(scaledX[idxLower], vars["n1_lower"], vars["n2_lower"], vars["lower"])
                dydLowerCST *= vars["chord"]
                xsdot[idxLower, self.yIdx] += dydLowerCST.T @ dvSeed
            if dvType == "n1_upper" or dvType == "n1":
                xsdot[idxUpper, self.yIdx] += (
                    dvSeed
                    * vars["chord"]
                    * self.computeCSTdydN1(scaledX[idxUpper], vars["n1_upper"], vars["n2_upper"], vars["upper"])
                )
            if dvType == "n2_upper" or dvType == "n2":
                xsdot[idxUpper, self.yIdx] += (
                    dvSeed
                    * vars["chord"]
                    * self.computeCSTdydN2(scaledX[idxUpper], vars["n1_upper"], vars["n2_upper"], vars["upper"])
                )
            if dvType == "n1_lower" or dvType == "n1":
                xsdot[idxLower, self.yIdx] += (
                    dvSeed
                    * vars["chord"]
                    * self.computeCSTdydN1(scaledX[idxLower], vars["n1_lower"], vars["n2_lower"], vars["lower"])
                )
            if dvType == "n2_lower" or dvType == "n2":
                xsdot[idxLower, self.yIdx] += (
                    dvSeed
                    * vars["chord"]
                    * self.computeCSTdydN2(scaledX[idxLower], vars["n1_lower"], vars["n2_lower"], vars["lower"])
                )
            if dvType == "chord":
                dydchord = self.points[ptSetName]["points"][:, self.yIdx] / vars["chord"]
                dxdchord = (ptsX - self.points[ptSetName]["xMin"]) / vars["chord"]
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
        for dvName, DV in self.DVs.items():
            optProb.addVarGroup(
                dvName,
                DV["value"].size,
                "c",
                value=DV["value"],
                lower=DV["lower"],
                upper=DV["upper"],
                scale=DV["scale"],
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
        vars = self._unpackDVs(ptSetName)

        # Unpack the points to make variable names more accessible
        idxUpper = self.points[ptSetName]["upper"]
        idxLower = self.points[ptSetName]["lower"]
        idxTE = np.full((self.points[ptSetName]["points"].shape[0],), True, dtype=bool)
        idxTE[idxUpper] = False
        idxTE[idxLower] = False
        thicknessTE = self.points[ptSetName]["thicknessTE"]
        points = self.points[ptSetName]["points"]
        ptsX = points[:, self.xIdx]
        ptsY = points[:, self.yIdx]

        # Scale the airfoil to the range 0 to 1 in x direction
        # TODO: the scaling is wrong if the point set is not the whole airfoil,
        #       which could be the case with DVCon's point sets
        shift = self.points[ptSetName]["xMin"]
        chord = self.points[ptSetName]["xMax"] - self.points[ptSetName]["xMin"]
        scaledX = (ptsX - shift) / chord
        yTE = thicknessTE / chord / 2  # half the scaled trailing edge thickness

        ptsY[idxUpper] = vars["chord"] * self.computeCSTCoordinates(
            scaledX[idxUpper], vars["n1_upper"], vars["n2_upper"], vars["upper"], yTE
        )
        ptsY[idxLower] = vars["chord"] * self.computeCSTCoordinates(
            scaledX[idxLower], vars["n1_lower"], vars["n2_lower"], vars["lower"], yTE
        )
        ptsY[idxTE] *= vars["chord"] / chord

        # Scale the chord according to the chord DV
        points[:, self.xIdx] = (points[:, self.xIdx] - shift) * vars["chord"] / chord + shift
        self.points[ptSetName]["xMax"] = (self.points[ptSetName]["xMax"] - shift) * vars["chord"] / chord + shift

        self.updated[ptSetName] = True

        return points

    def getNDV(self):
        """
        Return the total number of design variables this object has.

        Returns
        -------
        nDV : int
            Total number of design variables
        """
        return len(self.DVs)

    def printDesignVariables(self, directory):
        pass

    def writePointSet(self, name, fileName):
        pass

    def demoDesignVars(self, directory):
        pass

    def _unpackDVs(self, ptSetName):
        """
        Return the parameters needed for the airfoil shape calculation
        based on the DVs and default values. This requires a few extra
        checks to handle the multiple ways of parameterizing the class
        shape variables.

        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the
            given in an :func:`addPointSet()` call.

        Returns
        -------
        vars : dict
            Dictionary containing the following airfoil shape parameters:
                `"upper"`: upper surface CST coefficients
                `"lower"`: lower surface CST coefficients
                `"n1_lower"`: first class shape parameter on lower surface
                `"n2_lower"`: second class shape parameter on lower surface
                `"n1_upper"`: first class shape parameter on upper surface
                `"n2_upper"`: second class shape parameter on upper surface
                `"chord"`: chord length
        """
        vars = {}
        vars["upper"] = self.defaultDVs[ptSetName]["upper"].copy()
        vars["lower"] = self.defaultDVs[ptSetName]["lower"].copy()
        vars["n1_upper"] = self.defaultDVs[ptSetName]["n1_upper"].copy()
        vars["n2_upper"] = self.defaultDVs[ptSetName]["n2_upper"].copy()
        vars["n1_lower"] = self.defaultDVs[ptSetName]["n1_lower"].copy()
        vars["n2_lower"] = self.defaultDVs[ptSetName]["n2_lower"].copy()
        vars["chord"] = self.defaultDVs[ptSetName]["chord"].copy()

        for DV in self.DVs.values():
            if DV["type"] in ["n1", "n2"]:
                vars[f"{DV['type']}_upper"] = DV["value"]
                vars[f"{DV['type']}_lower"] = DV["value"]
            else:
                vars[DV["type"]] = DV["value"]

        return vars

    @staticmethod
    def computeCSTCoordinates(x, N1, N2, w, yte):
        """
        Compute the vertical coordinates of a CST curve

        This function assumes x has been normalised to the range [0,1]
        """
        C = DVGeometryCST.computeClassShape(x, N1, N2)
        S = DVGeometryCST.computeShapeFunctions(x, w)
        return C * S.sum(axis=0) + yte * x

    @staticmethod
    def computeClassShape(x, N1, N2):
        """
        Compute the class shape of a CST curve
        """
        return x**N1 * (1.0 - x) ** N2

    @staticmethod
    def computeShapeFunctions(x, w):
        """Compute the Bernstein polynomial shape function of a CST curve

        This function assumes x has been normalised to the range [0,1]
        """
        numCoeffs = len(w)
        order = numCoeffs - 1
        S = np.zeros((numCoeffs, len(x)), dtype=w.dtype)
        facts = factorial(np.arange(0, order + 1))
        for i in range(numCoeffs):
            binom = facts[-1] / (facts[i] * facts[order - i])
            S[i] = w[i] * binom * x ** (i) * (1.0 - x) ** (order - i)
        return S

    @staticmethod
    def computeCSTdydw(x, N1, N2, w):
        """Compute the drivatives of the height of a CST curve with respect to the shape function coefficients

        Given y = C(x) * sum [w_i * p_i(x)]
        dy/dw_i = C(x) * p_i(x)

        This function assumes x has been normalised to the range [0,1]
        """
        C = DVGeometryCST.computeClassShape(x, N1, N2)
        S = DVGeometryCST.computeShapeFunctions(x, np.ones_like(w))
        return C * S

    @staticmethod
    def computeCSTdydN1(x, N1, N2, w):
        """Compute the drivatives of the height of a CST curve with respect to N1

        Given y = C(x, N1, N2) * S(x)
        dy/dN1 = S(x) * dC/dN1 = S(x) * C(x, N1, N2) * ln(x)

        This function assumes x has been normalised to the range [0,1]
        """
        C = DVGeometryCST.computeClassShape(x, N1, N2)
        S = DVGeometryCST.computeShapeFunctions(x, w)
        return np.sum(S, axis=0) * C * np.log(x)

    @staticmethod
    def computeCSTdydN2(x, N1, N2, w):
        """Compute the drivatives of the height of a CST curve with respect to N2

        Given y = C(x, N1, N2) * S(x)
        dy/dN2 = S(x) * dC/dN2 = S(x) * C(x, N1, N2) * ln(1-x)

        This function assumes x has been normalised to the range [0,1]
        """
        C = DVGeometryCST.computeClassShape(x, N1, N2)
        S = DVGeometryCST.computeShapeFunctions(x, w)
        return np.sum(S, axis=0) * C * np.log(1 - x)

    def _orderAirfoilCoordinates(self, points, alpha=0.25):
        """
        Takes in a set of points and returns them sorted in the same
        order that they would be in a dat file (starting at the TE, traversing
        the upper surface to the LE, then around the lower surface back to the TE).
        This process computes the alpha shape to order the edges and is roughly
        based on https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points.

        Parameters
        ----------
        points : array, size (N,3)
            Set of airfoil coordinates in an arbitrary order.
        alpha : float, optional
            Value related to the edge size for the alpha shape (default is likely fine)

        Returns
        -------
        sortedPoints : array, size (N,3)
            Airfoil coordinates sorted in dat file order.
        """
        points2D = points[:, (self.xIdx, self.yIdx)]  # take only the chordwise and vertical coordinates
        assert points2D.shape[0] > 3, "Need at least four points"

        def add_edge(edges, i, j):
            """
            Add an edge between the i-th and j-th points,
            if not in the list already
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                assert (j, i) in edges, "Can't go twice over same directed edge right?"
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
                return
            edges.add((i, j))

        tri = Delaunay(points2D)
        edges = set()
        # Loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.simplices:
            pa = points2D[ia]
            pb = points2D[ib]
            pc = points2D[ic]
            # Computing radius of triangle circumcircle
            # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
            a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
            s = (a + b + c) / 2.0
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
            if circum_r < alpha:
                add_edge(edges, ia, ib)
                add_edge(edges, ib, ic)
                add_edge(edges, ic, ia)

        edges_arr = np.asarray(list(edges))
        n = np.shape(edges_arr)[0]
        edge_half = edges_arr[:, 0]
        edge_sort = np.sort(edge_half)

        return points[edge_sort]
