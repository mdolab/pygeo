"""
==============================================================================
DVGeo: CST Parameterisation
==============================================================================
@Author : Eytan Adler, Alasdair Christison Gray
@Description : A DVGeo implementation based on the Class-Shape Transformation method
"""

# Standard Python modules
import warnings

# External modules
from mpi4py import MPI
import numpy as np
from scipy.special import factorial

try:
    # External modules
    from prefoil.airfoil import Airfoil
    from prefoil.utils import readCoordFile

    prefoilInstalled = True
except ImportError:
    prefoilInstalled = False

try:
    # External modules
    import matplotlib.pyplot as plt

    pltImport = True
except ImportError:
    pltImport = False

# Local modules
from .BaseDVGeo import BaseDVGeometry
from .designVars import cstDV


class DVGeometryCST(BaseDVGeometry):
    r"""
    This class implements a 2D geometry parameterisation based on Brenda Kulfan's CST (Class-Shape Transformation) method.
    This class can work with 3D coordinates but will only change the point coordinates in one direction.

    The CST equation is as follows:

    :math:`y(x) = C(x) * S(x) + y_\text{te}x + w_\text{LE} x (1 - x)^{n + 0.5}`

    Where C is the class function:

    :math:`C(x) = x^{N1} (1 - x)^{N2}`

    And S is the shape function, in this case a summation of Bernstein polynomials:

    :math:`S(x) = \sum_{i=0}^n w_i \binom{n}{i}x^i(1-x)^{n-i}`

    Here x is the normalized chordwise coordinate, ranging from 0 to 1 from front to the rear of the shape.
    The last term is Kulfan's leading edge modification (LEM), where :math:`n` is the Bernstein polynomial order
    (:math:`n + 1` is the number of CST coefficients on the surface). Each surface has its own LEM weight. The weights
    are zero unless an LEM design variable (``"lem"``, ``"lem_upper"``, or ``"lem_lower"``) is added with a nonzero
    value or the initial fit includes the LEM (``fitLEM``).

    Assumptions about the point sets being added:

        - Dat file is ordered continuously around the airfoil and the beginning and end of
          the list is the trailing edge (no jumping around, but CW vs. CCW does not matter)
        - Geometry is exclusively an extruded shape (no spanwise changes allowed)
        - Airfoil's leading edge is on the left (min x) and trailing edge is on the right (max x)
        - Airfoil is not rotated (trailing edge and leading edge are close to y equals zero)

    Parameters
    ----------
    datFile : str
        Filename of dat file that represents the initial airfoil. The coordinates in this file will be used to
        determine the camber line, which is the dividing line to distinguish upper and lower surface points.
    numCST : int or list of two ints
        Number of CST parameters to use for the initial fit and the DVs (if DVs with type ``"upper"`` or ``"lower"``
        are added). If ``numCST`` is an int, the value will be used for both upper and lower. If it is a two-item list,
        the first value defines the number of upper CST coefficients and the second is the number of lower coefficients,
        by default 8.
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
    tolTE : float, optional
        Tolerance used to detect trailing edge corners on the airfoil. The value represents the angle difference
        in degrees between adjacent edges of the airfoil, by default 60 deg.
    name : string, optional
        Name of this DVGeo object, not necessary unless multiple DVGeos are used in one optimization.
    fitLEContinuity : bool, optional
        If True, the initial CST fit enforces ``lower[0] = -upper[0]``, which gives the upper and lower surfaces the
        same leading edge radius (curvature continuity at the leading edge), by default False.
        The upper and lower surfaces are then fit together in one least squares problem.
        Use this with :meth:`addLEContinuityConstraint` so the initial design satisfies the constraint.
    fitLEM : bool or str, optional
        Include Kulfan's leading edge modification (LEM) term (see :meth:`computeLEMShape`) in the initial CST fit,
        by default False. ``True`` (or ``"shared"``) fits one weight used on both surfaces, which models leading
        edge droop; ``"per_surface"`` fits separate upper and lower weights, as in Kulfan's original formulation.
        The fitted LEM weights are part of the geometry even if no LEM design variable is added. Combined with
        ``fitLEContinuity``, the LEM term captures leading edge droop that would otherwise be fit with unequal
        leading edge radii.
    """

    def __init__(
        self,
        datFile,
        numCST=8,
        idxChord=0,
        idxVertical=1,
        comm=MPI.COMM_WORLD,
        isComplex=False,
        debug=False,
        tolTE=60.0,
        name=None,
        fitLEContinuity=False,
        fitLEM=False,
    ):
        # Check if preFoil is installed before initializing.
        if not prefoilInstalled:
            raise ImportError("preFoil is not installed and is required to use DVGeometryCST.")

        super().__init__(datFile, name=name)

        # Error check the fitLEM input
        if fitLEM is True or fitLEM == "shared":
            fitLEM = "shared"
        elif fitLEM is False or fitLEM is None:
            fitLEM = None
        elif fitLEM != "per_surface":
            raise ValueError(f'fitLEM must be True, False, "shared", or "per_surface", not {fitLEM}')

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

        # Error check the numCST input
        if isinstance(numCST, int):
            self.nCSTUpper = numCST
            self.nCSTLower = numCST
        else:
            if isinstance(numCST, list):
                if len(numCST) != 2 or not isinstance(numCST[0], int) or not isinstance(numCST[1], int):
                    raise ValueError(f"numCST input of {numCST} is incorrect; must be int or list of two ints")
                else:
                    self.nCSTUpper = numCST[0]
                    self.nCSTLower = numCST[1]
            else:
                raise ValueError(f"numCST input of type {type(numCST)} is incorrect; must be int or list of two ints")

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
            "lem_upper": False,
            "lem_lower": False,
            "te_thickness": False,
        }

        # Default DVs to be copied for each point set
        self.defaultDV = {
            "upper": 0.1 * np.ones(self.nCSTUpper, dtype=self.dtype),
            "lower": -0.1 * np.ones(self.nCSTLower, dtype=self.dtype),
            "n1_upper": np.array([0.5], dtype=self.dtype),
            "n2_upper": np.array([1.0], dtype=self.dtype),
            "n1_lower": np.array([0.5], dtype=self.dtype),
            "n2_lower": np.array([1.0], dtype=self.dtype),
            "n1": np.array([0.5], dtype=self.dtype),
            "n2": np.array([1.0], dtype=self.dtype),
            "chord": np.array([1.0], dtype=self.dtype),
            "lem": np.array([0.0], dtype=self.dtype),
            "lem_upper": np.array([0.0], dtype=self.dtype),
            "lem_lower": np.array([0.0], dtype=self.dtype),
            "te_thickness": np.array([0.0], dtype=self.dtype),
        }

        # ========== Process the input airfoil and set variables accordingly ==========
        coords = readCoordFile(datFile)
        self.foilCoords = np.zeros_like(coords, dtype=self.dtype)
        self.foilCoords[:, self.xIdx] = coords[:, 0]
        self.foilCoords[:, self.yIdx] = coords[:, 1]

        # Determine if the dat file is closed (first and last points are the same); remove the duplicate point if so
        distance = np.linalg.norm(self.foilCoords[0, :] - self.foilCoords[-1, :])
        distTol = 1e-12
        if distance < distTol:
            self.foilCoords = self.foilCoords[:-1, :]

        # Check if the coordinates are clockwise or counter-clockwise by taking the cross-product of
        # vectors from adjacent points at the trailing edge
        vec1 = np.append(self.foilCoords[0] - self.foilCoords[-1], 0)
        vec2 = np.append(self.foilCoords[1] - self.foilCoords[0], 0)
        crossProduct = np.cross(vec1, vec2)

        # Flip the y-coordinates to counter-clockwise if they are clockwise (negative cross-product)
        if crossProduct[2] < 0:
            self.foilCoords = np.flip(self.foilCoords, self.xIdx)

        # Set the leading and trailing edge x coordinates
        self.xMin = np.min(self.foilCoords[:, self.xIdx])
        self.xMax = np.max(self.foilCoords[:, self.xIdx])

        # The airfoil file can have one or two points at the minimum x-coordinate
        idxLE = np.where(self.foilCoords[:, self.xIdx] == self.xMin)[0]
        if len(idxLE) > 2:
            raise ValueError(f"There can only be one or two points at the minimum x-coordinate, not {len(idxLE)}")

        # Check that the leading edge is at y = 0
        for idx in idxLE:
            yLE = self.foilCoords[idx, self.yIdx]
            if abs(yLE) > 1e-2:
                raise ValueError(f"Leading edge y (or idxVertical) value must equal zero, not {yLE}")

        # If there is one leading edge point, add a duplicate index to make setting idxFoil easier
        if len(idxLE) == 1:
            idxLE = np.repeat(idxLE, 2)

        # Traverse the airfoil surface to find the corner(s) defining the trailing edge (ignore anything in the front
        # half, chordwise, of the airfoil)
        cosTolTE = np.cos(np.deg2rad(tolTE))
        cornerIdx = []
        for idx in range(self.foilCoords.shape[0]):
            pt = self.foilCoords[idx, :]
            # Ignore if closer to the leading edge
            if pt[self.xIdx] - self.xMin < self.xMax - pt[self.xIdx]:
                continue
            edgePrev = pt - self.foilCoords[idx - 1, :]
            edgePrev /= np.linalg.norm(edgePrev)
            edgeNext = self.foilCoords[(idx + 1) % self.foilCoords.shape[0], :] - pt
            edgeNext /= np.linalg.norm(edgeNext)
            if np.dot(edgePrev, edgeNext) < cosTolTE:
                cornerIdx.append(idx)
        if len(cornerIdx) > 2:
            raise RuntimeError(
                "More than two corners in the airfoil identified when looking for the "
                + "trailing edge. If the actual airfoil geometry in the dat file has more "
                + "than two corners it is not supported. If not, try reducing the tolTE input."
            )
        elif len(cornerIdx) == 0:
            raise RuntimeError(
                "Zero corners in the airfoil identified when looking for the "
                + "trailing edge. If the actual airfoil geometry in the dat file has zero corners "
                + "(is a circle??) it is not supported. If not, try increasing the tolTE input."
            )

        # Airfoil is sharp if only one corner is detected
        self.sharp = len(cornerIdx) == 1

        # Save the trailing edge coordinates and surface indices
        self.idxFoil = {}
        if self.sharp:
            self.yUpperTE = np.array([0.0])
            self.yLowerTE = np.array([0.0])

            self.idxFoil["upper"] = np.arange(cornerIdx[0], idxLE[0] + 1)
            self.idxFoil["lower"] = np.concatenate(
                (np.arange(idxLE[1], self.foilCoords.shape[0]), np.array([cornerIdx[0]]))
            )
        else:
            self.coordUpperTE = self.foilCoords[cornerIdx[0]]
            self.coordLowerTE = self.foilCoords[cornerIdx[1]]

            self.yUpperTE = self.coordUpperTE[self.yIdx]
            self.yLowerTE = self.coordLowerTE[self.yIdx]

            self.idxFoil["upper"] = np.arange(cornerIdx[0], idxLE[0] + 1)
            self.idxFoil["lower"] = np.arange(idxLE[1], cornerIdx[1] + 1)

        # Compute splines for the upper and lower surfaces (used to split the foil in addPointSet).
        # preFoil defines the leading edge as the point furthest from the trailing edge
        if not self.sharp and cornerIdx[1] - cornerIdx[0] + 1 < self.foilCoords.shape[0]:
            # The dat file includes points along the blunt trailing edge face (e.g., dat files written by preFoil).
            # Leave them out of the splines; otherwise the lower surface spline runs up the trailing edge face and
            # points on the face are as close to it as to the trailing edge, so they may be snapped to the surface.
            surfaceCoords = np.real(self.foilCoords[cornerIdx[0] : cornerIdx[1] + 1][:, [self.xIdx, self.yIdx]])
            self.foil = Airfoil(surfaceCoords)
        else:
            self.foil = Airfoil(coords)
        self.upperSpline, self.lowerSpline = self.foil.splitAirfoil()

        # Fit CST parameters to the airfoil's upper and lower surface
        chord = self.xMax - self.xMin
        self.defaultDV["chord"][0] = chord
        self.defaultDV["te_thickness"][:] = (self.yUpperTE - self.yLowerTE) / chord
        yTE = {"upper": self.yUpperTE, "lower": self.yLowerTE}
        if self.comm.rank == 0:
            print(f"######## Fitting CST coefficients to coordinates in {datFile} ########")
            xPts = {dvType: self.foilCoords[self.idxFoil[dvType], self.xIdx] for dvType in ["upper", "lower"]}
            yPts = {dvType: self.foilCoords[self.idxFoil[dvType], self.yIdx] for dvType in ["upper", "lower"]}

            if fitLEContinuity or fitLEM is not None:
                # Fit both surfaces together so they can share the tied first coefficient and the LEM term
                upper, lower, lemUpper, lemLower = self.computeCSTfromCoordsJoint(
                    (xPts["upper"] - self.xMin) / chord,
                    yPts["upper"] / chord,
                    (xPts["lower"] - self.xMin) / chord,
                    yPts["lower"] / chord,
                    self.nCSTUpper,
                    self.nCSTLower,
                    yUpperTE=self.yUpperTE / chord,
                    yLowerTE=self.yLowerTE / chord,
                    N1Upper=self.defaultDV["n1_upper"],
                    N2Upper=self.defaultDV["n2_upper"],
                    N1Lower=self.defaultDV["n1_lower"],
                    N2Lower=self.defaultDV["n2_lower"],
                    tieLE=fitLEContinuity,
                    fitLEM=fitLEM,
                    dtype=self.dtype,
                )
                self.defaultDV["upper"] = upper.astype(self.dtype)
                self.defaultDV["lower"] = lower.astype(self.dtype)
                self.defaultDV["lem_upper"] = lemUpper.astype(self.dtype)
                self.defaultDV["lem_lower"] = lemLower.astype(self.dtype)
                # Default for the shared LEM DV (the two weights are equal with a shared fit)
                self.defaultDV["lem"] = (0.5 * (lemUpper + lemLower)).astype(self.dtype)
            else:
                for dvType in ["upper", "lower"]:
                    self.defaultDV[dvType] = self.computeCSTfromCoords(
                        xPts[dvType],
                        yPts[dvType],
                        self.defaultDV[dvType].size,
                        N1=self.defaultDV[f"n1_{dvType}"],
                        N2=self.defaultDV[f"n2_{dvType}"],
                        dtype=self.dtype,
                    )

            for dvType in ["upper", "lower"]:
                # Compute the quality of the fit by computing an L2 norm of the fit vs. the actual coordinates
                ptsFit = chord * self.computeCSTCoordinates(
                    (xPts[dvType] - self.xMin) / chord,
                    self.defaultDV[f"n1_{dvType}"],
                    self.defaultDV[f"n2_{dvType}"],
                    self.defaultDV[dvType],
                    yTE[dvType] / chord,
                    dtype=self.dtype,
                    wLE=self.defaultDV[f"lem_{dvType}"],
                )
                L2norm = np.sqrt(1 / ptsFit.size * np.sum((yPts[dvType] - ptsFit) ** 2))

                print(f"{dvType.capitalize()} surface")
                print(f"    L2 norm of coordinates in dat file versus fit coordinates: {L2norm}")
                print(f"    Fit CST coefficients: {self.defaultDV[dvType]}")
            if fitLEM is not None:
                print(
                    "Fit leading edge modification (LEM) weights (upper, lower): "
                    + f"{self.defaultDV['lem_upper']}, {self.defaultDV['lem_lower']}"
                )

        # Broadcast the fit DVs to the rest of the procs
        for dvType in ["upper", "lower", "lem", "lem_upper", "lem_lower"]:
            self.comm.Bcast([self.defaultDV[dvType], self.dtypeMPI])

    def addPointSet(self, points, ptName, boundTol=1e-10, **kwargs):
        """
        Add a set of coordinates to DVGeometry.
        The is the main way that geometry in the form of a coordinate list is given to DVGeometry
        to be manipulated.

        .. note::
            Even if ``isComplex=True``, the imaginary portion of coordinates passed in here
            is ignored when determining if a given point is on the upper or lower surface.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed.
        ptName : str
            A user supplied name to associate with the set of coordinates.
            This name will need to be provided when updating the coordinates or when
            getting the derivatives of the coordinates.
        boundTol : float, optional
            Small absolute deviation by which the airfoil coordinates can exceed the initial
            minimum and maximum x coordinates, by default 1e-10.
        \\*\\*kwargs
            Any other parameters are ignored.
        """
        # Convert points to the type specified at initialization (with isComplex) and store the points
        points = points.astype(self.dtype)

        # Check that all points are within the airfoil x bounds
        if np.any(points[:, self.xIdx] < self.xMin - boundTol) or np.any(points[:, self.xIdx] > self.xMax + boundTol):
            raise ValueError(
                f'Points in the point set "{ptName}" have x coordinates outside'
                + f"the min and max x values in the initial dat file ({self.xMin} and {self.xMax})"
            )
        self.updated[ptName] = False
        self.points[ptName] = {
            "points": points,
            "xMax": self.xMax.copy(),
            "xMin": self.xMin.copy(),
            "yUpperTE": self.yUpperTE.copy(),
            "yLowerTE": self.yLowerTE.copy(),
        }

        # Determine which points are on the upper and lower surfaces
        self.points[ptName]["upper"], self.points[ptName]["lower"] = self._splitUpperLower(points)

        # For the trailing edge thickness DV, store the trailing edge midpoint (normalized by chord) and the
        # fractional position of each point along the blunt trailing edge face (0 at the lower corner, 1 at
        # the upper corner). These are computed once here so later updates do not depend on the update history.
        chord = self.xMax - self.xMin
        self.points[ptName]["yMidTEScaled"] = 0.5 * (self.yUpperTE + self.yLowerTE) / chord
        etaTE = np.zeros(points.shape[0], dtype=self.dtype)
        if not self.sharp:
            etaTE[:] = (points[:, self.yIdx] - self.yLowerTE) / (self.yUpperTE - self.yLowerTE)
        self.points[ptName]["etaTE"] = etaTE

        # If debug mode is on, plot the upper and lower surface points
        if self.debug:
            # Gather all the plotting data on the root proc
            dataGlob = {}
            for name in ["points", "upper", "lower"]:
                # First, determine the sizes and displacements of the arrays from each proc for gather
                vecFlatLoc = self.points[ptName][name].flatten()
                if name in ["upper", "lower"]:
                    vecFlatLoc = vecFlatLoc.astype("intc")
                numLoc = vecFlatLoc.size
                sizes = np.array(self.comm.allgather(numLoc), dtype="intc")
                disp = np.array([np.sum(sizes[:i]) for i in range(self.comm.size)], dtype="intc")
                if name == "points":
                    dispPoints = disp.copy()
                numGlob = np.sum(sizes)

                # Send coordinates to root proc
                dtype = "intc"
                dtypeMPI = MPI.INT
                if name == "points":
                    dtype = self.dtype
                    dtypeMPI = self.dtypeMPI
                dataGlob[name] = np.zeros(numGlob, dtype=dtype)  # recv buffer

                # Shift the data by the displacement if it is local index data
                if name in ["upper", "lower"]:
                    vecFlatLoc += dispPoints[self.comm.rank] // 3

                # Finally, collect the data on the root proc
                self.comm.Gatherv([vecFlatLoc, numLoc], [dataGlob[name], sizes, disp, dtypeMPI])

            if self.comm.rank == 0:
                # Reshape the flatted coordinates
                coords = dataGlob["points"].reshape((dataGlob["points"].size // 3, 3))

                fig = plt.figure()
                plt.scatter(
                    coords[:, self.xIdx][dataGlob["upper"]],
                    coords[:, self.yIdx][dataGlob["upper"]],
                    c="b",
                )
                plt.scatter(
                    coords[:, self.xIdx][dataGlob["lower"]],
                    coords[:, self.yIdx][dataGlob["lower"]],
                    c="r",
                )
                plt.scatter(
                    coords[:, self.xIdx],
                    coords[:, self.yIdx],
                    s=3,
                    c="k",
                    zorder=3,
                )
                plt.legend(["Upper", "Lower"])
                plt.show()
                plt.close(fig)

            self.comm.Barrier()

    def addDV(self, dvName, dvType, lowerBound=None, upperBound=None, scale=1.0, default=None):
        """
        Add design variables to the DVGeometryCST object. For upper and lower CST coefficient DVs,
        the number of design variables is defined using the ``numCST`` parameter in DVGeoCST's
        init function.

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
                - ``"lem"``: one weight for Kulfan's leading edge modification (LEM) term, used on both the upper
                  and lower surfaces (adds a single DV). It changes the leading edge droop (camber line slope at the
                  leading edge) without changing the leading edge radius. See :meth:`computeLEMShape`.
                - ``"lem_upper"``: LEM weight on the upper surface (adds a single DV)
                - ``"lem_lower"``: LEM weight on the lower surface (adds a single DV). Separate upper and lower
                  weights are Kulfan's original formulation; their difference adds a thickness slope at the
                  leading edge, while their mean sets the droop. Either ``"lem"`` or the separate upper/lower LEM
                  DVs can be added, but not both.
                - ``"te_thickness"``: trailing edge thickness normalized by the chord (adds a single DV). The
                  trailing edge opens symmetrically about the midpoint of the dat file's trailing edge and points
                  on the trailing edge face move with it. Only available for blunt trailing edge airfoils.
                  A lower bound of zero is recommended, since a negative thickness makes the surfaces cross.
        lowerBound : float or ndarray, optional
            The lower bound for the variable(s). This will be applied to
            all shape variables
        upperBound : float or ndarray, optional
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
            "lem",
            "lem_upper",
            "lem_lower",
            "te_thickness",
        ]:
            raise ValueError(
                'dvType must be one of "upper", "lower", "N1", "N2", "N1_upper", "N1_lower", '
                + '"N2_upper", "N2_lower", "chord", "lem", "lem_upper", "lem_lower", '
                + f'or "te_thickness", not {dvType}'
            )
        dvType = dvType.lower()

        if dvType == "te_thickness" and self.sharp:
            raise ValueError(
                'The "te_thickness" DV requires a blunt trailing edge, but the airfoil in the dat file is sharp'
            )

        if dvType == "upper":
            dvNum = self.nCSTUpper
        elif dvType == "lower":
            dvNum = self.nCSTLower
        else:
            dvNum = 1

        # Check that a duplicate DV doesn't already exist
        if dvType in ["n1", "n2", "lem", "n1_upper", "n1_lower", "n2_upper", "n2_lower", "lem_upper", "lem_lower"]:
            if dvType in ["n1", "n2", "lem"]:  # if one of these is added, the individual lower and upper ones can't be
                if self.DVExists[dvType + "_lower"]:
                    raise ValueError(f'"{dvType}" cannot be added when "{dvType}_lower" already exists')
                elif self.DVExists[dvType + "_upper"]:
                    raise ValueError(f'"{dvType}" cannot be added when "{dvType}_upper" already exists')
                else:
                    self.DVExists[dvType + "_lower"] = True
                    self.DVExists[dvType + "_upper"] = True
            else:  # the parameter that controls both the upper and lower surfaces simultaneously can't be added
                param = dvType.split("_")[0]  # N1, N2, or LEM
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
            default = self.defaultDV[dvType]
            if dvType == "lem":
                # The shared LEM DV sets both surfaces' weights, so warn if it replaces two different fitted weights
                if np.real(self.defaultDV["lem_upper"][0]) != np.real(self.defaultDV["lem_lower"][0]):
                    if self.comm.rank == 0:
                        warnings.warn(
                            'The shared "lem" DV replaces the different upper and lower LEM weights from the fit '
                            + f"with their mean, {np.real(default[0])}, which changes the initial geometry. "
                            + 'Use the "lem_upper" and "lem_lower" DVs to keep them separate.',
                            stacklevel=2,
                        )
                self.defaultDV["lem_upper"] = default.astype(self.dtype)
                self.defaultDV["lem_lower"] = default.astype(self.dtype)
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
            if dvType in ["n1", "n2", "lem"]:
                self.defaultDV[f"{dvType}_lower"] = default.astype(self.dtype)
                self.defaultDV[f"{dvType}_upper"] = default.astype(self.dtype)

        # Add the DV to the internally-stored list
        self.DVs[dvName] = cstDV(
            name=dvName,
            value=default.astype(self.dtype),
            nVal=dvNum,
            lower=lowerBound,
            upper=upperBound,
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

    def getDesignVars(self):
        """
        Generic routine to return the current set of design variables.
        Values are returned in a dictionary format that would be suitable for a subsequent call to setDesignVars()

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

    def getDVBounds(self):
        """
        Return the bounds on the design variables.

        Returns
        -------
        lowerBounds : dict
            Dictionary of design variable lower bounds
        upperBounds : dict
            Dictionary of design variable upper bounds
        """
        lowerBounds = {}
        upperBounds = {}
        for dvName, dv in self.DVs.items():
            lowerBounds[dvName] = dv.lower.real if dv.lower is not None else None
            upperBounds[dvName] = dv.upper.real if dv.upper is not None else None
        return lowerBounds, upperBounds

    def getVarNames(self, **kwargs):
        """
        Return a list of the design variable names. This is typically used when specifying a wrt= argument for pyOptSparse.

        Examples
        --------
        >>> optProb.addCon(.....wrt=DVGeo.getVarNames())
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
        \\*\\*kwargs
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
        idxTE = np.full((self.points[ptSetName]["points"].shape[0],), True, dtype=bool)
        idxTE[idxUpper] = False
        idxTE[idxLower] = False
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
            elif dvType in ["lem", "lem_upper", "lem_lower"]:
                funcSens_local[dvName] = 0.0
                if dvType in ["lem", "lem_upper"]:
                    dydLEMUpper = desVars["chord"] * self.computeLEMShape(
                        scaledX[idxUpper], self.nCSTUpper, dtype=self.dtype
                    )
                    funcSens_local[dvName] = funcSens_local[dvName] + dydLEMUpper @ dIdpt[idxUpper, self.yIdx]
                if dvType in ["lem", "lem_lower"]:
                    dydLEMLower = desVars["chord"] * self.computeLEMShape(
                        scaledX[idxLower], self.nCSTLower, dtype=self.dtype
                    )
                    funcSens_local[dvName] = funcSens_local[dvName] + dydLEMLower @ dIdpt[idxLower, self.yIdx]
            elif dvType == "te_thickness":
                dydtUpper = 0.5 * desVars["chord"] * scaledX[idxUpper]
                dydtLower = -0.5 * desVars["chord"] * scaledX[idxLower]
                dydtTE = desVars["chord"] * (self.points[ptSetName]["etaTE"][idxTE] - 0.5)
                funcSens_local[dvName] = (
                    dydtUpper @ dIdpt[idxUpper, self.yIdx]
                    + dydtLower @ dIdpt[idxLower, self.yIdx]
                    + dydtTE @ dIdpt[idxTE, self.yIdx]
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
        \\*\\*kwargs
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
            if dvType in ["lem", "lem_upper"]:
                xsdot[idxUpper, self.yIdx] += (
                    dvSeed
                    * desVars["chord"]
                    * self.computeLEMShape(scaledX[idxUpper], self.nCSTUpper, dtype=self.dtype)
                )
            if dvType in ["lem", "lem_lower"]:
                xsdot[idxLower, self.yIdx] += (
                    dvSeed
                    * desVars["chord"]
                    * self.computeLEMShape(scaledX[idxLower], self.nCSTLower, dtype=self.dtype)
                )
            if dvType == "te_thickness":
                xsdot[idxUpper, self.yIdx] += dvSeed * desVars["chord"] * 0.5 * scaledX[idxUpper]
                xsdot[idxLower, self.yIdx] -= dvSeed * desVars["chord"] * 0.5 * scaledX[idxLower]
                xsdot[idxTE, self.yIdx] += dvSeed * desVars["chord"] * (self.points[ptSetName]["etaTE"][idxTE] - 0.5)

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

    def addLEContinuityConstraint(self, optProb, name="le_continuity"):
        """
        Add a linear equality constraint, ``upper[0] + lower[0] = 0``, to the optimization problem.

        With N1 = 0.5, the leading edge radius of each surface is set by its first CST coefficient
        (radius = w_0^2 / 2). Constraining the first upper and lower coefficients to have the same magnitude
        and opposite signs gives both surfaces the same leading edge radius, so the curvature is continuous
        around the leading edge. It also fixes the camber line slope at the leading edge to the value set by the
        LEM term, which is zero unless the initial fit includes it (``fitLEM=True``). Adding the ``"lem"`` DV lets
        the optimizer change the leading edge droop while keeping the curvature continuous.

        The initial design should satisfy this constraint, which can be done by initializing
        DVGeometryCST with ``fitLEContinuity=True``.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which the constraint is added. The upper and lower
            DVs must already be added to it (see :meth:`addVariablesPyOpt`).
        name : str, optional
            Name of the constraint, by default ``"le_continuity"``
        """
        dvNames = {}
        for dvType in ["upper", "lower"]:
            matches = [DV.name for DV in self.DVs.values() if DV.type == dvType]
            if len(matches) == 0:
                raise ValueError(
                    f'The leading edge continuity constraint requires a DV of type "{dvType}", but none was added'
                )
            dvNames[dvType] = matches[0]

        # Warn if the current design does not satisfy the constraint
        upper0 = np.real(self.DVs[dvNames["upper"]].value[0])
        lower0 = np.real(self.DVs[dvNames["lower"]].value[0])
        if abs(upper0 + lower0) > 1e-10 and self.comm.rank == 0:
            warnings.warn(
                f"The current design does not satisfy the leading edge continuity constraint (upper[0] = {upper0}, "
                + f"lower[0] = {lower0}). Initialize DVGeometryCST with fitLEContinuity=True to start from a "
                + "design that satisfies it.",
                stacklevel=2,
            )

        # Warn if the upper and lower surfaces have different N1 DVs
        if any(DV.type in ["n1_upper", "n1_lower"] for DV in self.DVs.values()) and self.comm.rank == 0:
            warnings.warn(
                "Separate N1 DVs on the upper and lower surfaces can change the leading edge shape differently on "
                + 'each surface, so they break leading edge continuity. Use the "n1" DV instead.',
                stacklevel=2,
            )

        jac = {}
        for dvType in ["upper", "lower"]:
            jac[dvNames[dvType]] = np.zeros((1, self.DVs[dvNames[dvType]].nVal))
            jac[dvNames[dvType]][0, 0] = 1.0

        optProb.addCon(name, lower=0.0, upper=0.0, linear=True, wrt=list(jac.keys()), jac=jac)

    def update(self, ptSetName, **kwargs):
        """
        This is the main routine for returning coordinates that have
        been updated by design variables.

        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the
            given in an :func:`addPointSet()` call.
        \\*\\*kwargs
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
        yUpperTE = self.points[ptSetName]["yUpperTE"]
        yLowerTE = self.points[ptSetName]["yLowerTE"]

        # Scale the airfoil to the range 0 to 1 in x direction
        shift = xMin
        chord = xMax - xMin
        scaledX = (ptsX - shift) / chord
        if self.DVExists["te_thickness"]:
            # The trailing edge opens symmetrically about the original trailing edge midpoint
            yMidTEScaled = self.points[ptSetName]["yMidTEScaled"]
            scaledYUpperTE = yMidTEScaled + 0.5 * desVars["te_thickness"]
            scaledYLowerTE = yMidTEScaled - 0.5 * desVars["te_thickness"]
        else:
            scaledYUpperTE = yUpperTE / chord
            scaledYLowerTE = yLowerTE / chord

        ptsY[idxUpper] = desVars["chord"] * self.computeCSTCoordinates(
            scaledX[idxUpper],
            desVars["n1_upper"],
            desVars["n2_upper"],
            desVars["upper"],
            scaledYUpperTE,
            dtype=self.dtype,
            wLE=desVars["lem_upper"],
        )
        ptsY[idxLower] = desVars["chord"] * self.computeCSTCoordinates(
            scaledX[idxLower],
            desVars["n1_lower"],
            desVars["n2_lower"],
            desVars["lower"],
            scaledYLowerTE,
            dtype=self.dtype,
            wLE=desVars["lem_lower"],
        )
        if self.DVExists["te_thickness"]:
            # Points on the blunt trailing edge face keep their fractional position between the corners
            etaTE = self.points[ptSetName]["etaTE"][idxTE]
            ptsY[idxTE] = desVars["chord"] * (scaledYLowerTE + etaTE * (scaledYUpperTE - scaledYLowerTE))
        else:
            ptsY[idxTE] *= desVars["chord"] / chord

        # Scale the chord according to the chord DV
        points[:, self.xIdx] = (points[:, self.xIdx] - shift) * desVars["chord"] / chord + shift

        # Scale the point set's properties based on the new chord length
        self.points[ptSetName]["xMax"] = (xMax - shift) * desVars["chord"] / chord + shift
        if self.DVExists["te_thickness"]:
            # Not used by later updates (they use the stored midpoint and the DV), but keeps the point set's
            # trailing edge coordinates consistent with the current geometry
            self.points[ptSetName]["yUpperTE"] = desVars["chord"] * scaledYUpperTE
            self.points[ptSetName]["yLowerTE"] = desVars["chord"] * scaledYLowerTE
        else:
            self.points[ptSetName]["yUpperTE"] *= desVars["chord"] / chord
            self.points[ptSetName]["yLowerTE"] *= desVars["chord"] / chord

        self.updated[ptSetName] = True

        return points.copy()

    def getOrigPoints(self, ptSetName):
        """Get the original coordinates for a point set. a.k.a the coordinates that were passed to :func:`addPointSet`.

        Parameters
        ----------
        ptSetName : str
            Name of the point set to return the original coordinates for.
        """
        return self.points[ptSetName]["points"]

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
        if self.comm.rank == 0:
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
                `"lem_upper"`: leading edge modification weight on the upper surface
                `"lem_lower"`: leading edge modification weight on the lower surface
                `"te_thickness"`: trailing edge thickness normalized by chord
        """
        desVars = {}
        desVars["upper"] = self.defaultDV["upper"].copy()
        desVars["lower"] = self.defaultDV["lower"].copy()
        desVars["n1_upper"] = self.defaultDV["n1_upper"].copy()
        desVars["n2_upper"] = self.defaultDV["n2_upper"].copy()
        desVars["n1_lower"] = self.defaultDV["n1_lower"].copy()
        desVars["n2_lower"] = self.defaultDV["n2_lower"].copy()
        desVars["chord"] = self.defaultDV["chord"].copy()
        desVars["lem_upper"] = self.defaultDV["lem_upper"].copy()
        desVars["lem_lower"] = self.defaultDV["lem_lower"].copy()
        desVars["te_thickness"] = self.defaultDV["te_thickness"].copy()

        for DV in self.DVs.values():
            if DV.type in ["n1", "n2", "lem"]:
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
        # Determine which surface (either upper, lower, or trailing edge) each point is
        # on based on which spline it is closest to
        # Upper (if it's complex, ignore the imaginary part since the spline doesn't handle that)
        _, upperDist = self.upperSpline.projectPoint(np.real(points[:, [self.xIdx, self.yIdx]]))
        upperDist = np.linalg.norm(upperDist, axis=1)
        # Lower
        _, lowerDist = self.lowerSpline.projectPoint(np.real(points[:, [self.xIdx, self.yIdx]]))
        lowerDist = np.linalg.norm(lowerDist, axis=1)
        # Trailing edge
        teDist = np.full_like(upperDist, np.inf)
        if not self.sharp:
            x0 = points[:, self.xIdx]
            y0 = points[:, self.yIdx]
            x1 = self.coordLowerTE[self.xIdx]
            y1 = self.coordLowerTE[self.yIdx]
            x2 = self.coordUpperTE[self.xIdx]
            y2 = self.coordUpperTE[self.yIdx]
            teDist = np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Determine if each point is on the upper surface if it's closer to the upper spline than
        # either the lower spline or the trailing edge line (and do the same for the lower surface)
        upperBool = np.logical_and(upperDist <= lowerDist, upperDist <= teDist)
        lowerBool = np.logical_and(lowerDist < upperDist, lowerDist <= teDist)

        return np.where(upperBool)[0], np.where(lowerBool)[0]

    @staticmethod
    def computeCSTCoordinates(x, N1, N2, w, yte, dtype=float, wLE=0.0):
        """
        Compute the vertical coordinates of a CST curve.

        This function assumes x has been normalized to the range [0,1]
        and that yte has been normalized by the chord.

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
        wLE : float, optional
            Weight of Kulfan's leading edge modification (LEM) term on this surface, by default 0.0.
            See :meth:`computeLEMShape`.

        Returns
        -------
        ndarray (# pts,)
            y coordinates of the CST curve
        """
        C = DVGeometryCST.computeClassShape(x, N1, N2, dtype=dtype)
        S = DVGeometryCST.computeShapeFunctions(x, w, dtype=dtype)
        y = C * S.sum(axis=0) + yte * x
        # Only add the LEM term if it is nonzero so the result is unchanged when it is not used
        if np.any(wLE != 0.0):
            y = y + wLE * DVGeometryCST.computeLEMShape(x, len(w), dtype=dtype)
        return y

    @staticmethod
    def computeLEMShape(x, n, dtype=float):
        r"""
        Compute the shape of Kulfan's leading edge modification (LEM) term

        :math:`y_\text{LEM}(x) = x (1 - x)^{N + 0.5}`

        where :math:`N = n - 1` is the order of the Bernstein polynomials for a surface with :math:`n` CST
        coefficients. Kulfan defines it by duplicating and reversing the last component shape,
        :math:`x^{N + 0.5} (1 - x)`. Each surface has its own LEM weight.

        Near the leading edge the LEM term behaves like :math:`x`, while the CST terms behave like
        :math:`\sqrt{x}`, so it does not change the leading edge radius of either surface.
        The same weight on both surfaces gives the camber line a finite slope at the leading edge
        (leading edge droop); different weights also add a thickness slope at the leading edge.

        This function assumes x has been normalized to the range [0,1].
        Points at or beyond the trailing edge (x >= 1) return zero.

        Parameters
        ----------
        x : ndarray (# pts,)
            x coordinates at which to compute the LEM shape
        n : int
            Number of CST coefficients on the surface
        dtype : type, optional
            Type for instantiated arrays, by default float

        Returns
        -------
        ndarray (# pts,)
            LEM shape at each x coordinate
        """
        x = np.asarray(x)
        y = np.zeros(x.shape, dtype=dtype)

        # A real power of a negative number is undefined, so anywhere x >= 1 (e.g., points
        # slightly past the trailing edge) just keep the LEM shape as zero
        mask = np.real(x) < 1.0
        y[mask] = x[mask] * (1.0 - x[mask]) ** (n - 0.5)

        return y

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
        r"""Compute the derivatives of the height of a CST curve with respect to N1

        Given :math:`y = C(x, N1, N2) * S(x)`

        :math:`\frac{dy}{dN1} = S(x) * \frac{dC}{dN1} = S(x) * C(x, N1, N2) * \ln{x}`

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
        r"""Compute the derivatives of the height of a CST curve with respect to N2

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
        yCoord = yCoord / chord

        # Find the y-coordinate at the trailing edge
        idxTE = np.argmax(xCoord)
        yTE = yCoord[idxTE]

        # Subtract the linear TE thickness function
        yMinusTE = yCoord - yTE * xCoord

        # Compute the coefficients via linear least squares
        dydw = DVGeometryCST.computeCSTdydw(xCoord, N1, N2, np.ones(nCST), dtype=dtype)
        w = np.linalg.lstsq(dydw.T, yMinusTE, rcond=None)[0]
        return w

    @staticmethod
    def computeCSTfromCoordsJoint(
        xUpper,
        yUpper,
        xLower,
        yLower,
        nCSTUpper,
        nCSTLower,
        yUpperTE=0.0,
        yLowerTE=0.0,
        N1Upper=0.5,
        N2Upper=1.0,
        N1Lower=0.5,
        N2Lower=1.0,
        tieLE=False,
        fitLEM=None,
        dtype=float,
    ):
        """
        Compute the CST coefficients that fit the upper and lower surfaces of an airfoil together
        in a single linear least squares problem. This allows the fit to enforce leading edge
        continuity (``lower[0] = -upper[0]``) and/or include the leading edge modification (LEM) term.

        This function assumes the x coordinates have been normalized to the range [0, 1] and that the
        y coordinates and trailing edge y coordinates have been normalized by the chord.

        Parameters
        ----------
        xUpper, yUpper : ndarray
            Normalized upper surface airfoil coordinates.
        xLower, yLower : ndarray
            Normalized lower surface airfoil coordinates.
        nCSTUpper, nCSTLower : int
            Number of CST coefficients to fit on the upper and lower surfaces.
        yUpperTE, yLowerTE : float, optional
            Normalized y coordinates of the upper and lower trailing edge corners, by default 0.0.
            The linear trailing edge thickness term is subtracted before fitting.
        N1Upper, N2Upper, N1Lower, N2Lower : float, optional
            Class shape parameters to assume in fitting, by default 0.5 and 1.0.
        tieLE : bool, optional
            Enforce ``lower[0] = -upper[0]`` (equal leading edge radii), by default False.
        fitLEM : None, bool, or str, optional
            Include the LEM term in the fit: ``None`` or ``False`` (default) for no LEM, ``True`` or ``"shared"``
            for one weight used on both surfaces, or ``"per_surface"`` for separate upper and lower weights.
            Without the LEM, the returned LEM weights are zero.
        dtype : type, optional
            Type for instantiated arrays, by default float

        Returns
        -------
        upper : ndarray (nCSTUpper,)
            Upper surface CST coefficients.
        lower : ndarray (nCSTLower,)
            Lower surface CST coefficients.
        lemUpper : ndarray (1,)
            Upper surface LEM weight.
        lemLower : ndarray (1,)
            Lower surface LEM weight.
        """
        if fitLEM is True:
            fitLEM = "shared"
        elif fitLEM is False:
            fitLEM = None
        nUpperPts = len(xUpper)
        nLowerPts = len(xLower)
        idxLowerStart = nCSTUpper
        idxLEMUpper = nCSTUpper + nCSTLower
        idxLEMLower = idxLEMUpper + 1

        # Columns are [upper coefficients, lower coefficients, upper LEM weight, lower LEM weight]
        A = np.zeros((nUpperPts + nLowerPts, nCSTUpper + nCSTLower + 2), dtype=dtype)
        A[:nUpperPts, :nCSTUpper] = DVGeometryCST.computeCSTdydw(
            xUpper, N1Upper, N2Upper, np.ones(nCSTUpper), dtype=dtype
        ).T
        A[nUpperPts:, idxLowerStart:idxLEMUpper] = DVGeometryCST.computeCSTdydw(
            xLower, N1Lower, N2Lower, np.ones(nCSTLower), dtype=dtype
        ).T
        A[:nUpperPts, idxLEMUpper] = DVGeometryCST.computeLEMShape(xUpper, nCSTUpper, dtype=dtype)
        A[nUpperPts:, idxLEMLower] = DVGeometryCST.computeLEMShape(xLower, nCSTLower, dtype=dtype)

        # Subtract the linear trailing edge thickness term from the coordinates
        b = np.concatenate((yUpper - yUpperTE * xUpper, yLower - yLowerTE * xLower)).astype(dtype)

        cols = list(range(A.shape[1]))
        if tieLE:
            # Substitute lower[0] = -upper[0] by folding the first lower coefficient's column into the first upper one
            A[:, 0] -= A[:, idxLowerStart]
            cols.remove(idxLowerStart)
        if fitLEM == "shared":
            # One weight on both surfaces: fold the lower LEM column into the upper one
            A[:, idxLEMUpper] += A[:, idxLEMLower]
            cols.remove(idxLEMLower)
        elif fitLEM is None:
            cols.remove(idxLEMUpper)
            cols.remove(idxLEMLower)

        params = np.zeros(A.shape[1], dtype=dtype)
        params[cols] = np.linalg.lstsq(A[:, cols], b, rcond=None)[0]
        if tieLE:
            params[idxLowerStart] = -params[0]
        if fitLEM == "shared":
            params[idxLEMLower] = params[idxLEMUpper]

        return (
            params[:nCSTUpper],
            params[idxLowerStart:idxLEMUpper],
            params[idxLEMUpper : idxLEMUpper + 1],
            params[idxLEMLower : idxLEMLower + 1],
        )

    @staticmethod
    def plotCST(upperCoeff, lowerCoeff, yUpperTE=0.0, yLowerTE=0.0, N1=0.5, N2=1.0, nPts=100, ax=None, **kwargs):
        """Simple utility to generate a plot from CST coefficients.

        Parameters
        ----------
        upperCoeff : ndarray
            One dimensional array of CST coefficients for the upper surface.
        lowerCoeff : ndarray
            One dimensional array of CST coefficients for the lower surface.
        yUpperTE : float, optional
            y-coordinate for the upper surface trailing edge point.
        yLowerTE : float, optional
            y-coordinate for the lower surface trailing edge point.
        N1 : float, optional
            First class shape parameter.
        N2 : float, optional
            Second class shape parameter.
        nPts : int, optional
            Number of coordinates to compute on each surface.
        ax : matplotlib Axes, optional
            Axes on which to plot airfoil.
        \\*\\*kwargs
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
        yUpper = DVGeometryCST.computeCSTCoordinates(x, N1, N2, upperCoeff, yUpperTE)
        yLower = DVGeometryCST.computeCSTCoordinates(x, N1, N2, lowerCoeff, yLowerTE)

        ax.plot(x, yUpper, **kwargs)
        ax.plot(x, yLower, **kwargs)
        ax.set_aspect("equal")

        return ax
