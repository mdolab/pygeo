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

    def addPointSet(self, points, ptName):
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

        # # Sort the points in the expected dat file order
        # points = self._orderAirfoilCoordinates(pointsGlobal)

        # Find the leading edge point to split upper and lower surfaces
        idxLE = np.argmin(pointsGlobal[:, self.xIdx])

        # Check that the leading edge is at y = 0
        if abs(pointsGlobal[idxLE, self.yIdx]) > 1e-2:
            raise ValueError(
                f"Leading edge y (or idxVertical) value must equal zero, not {pointsGlobal[idxLE, self.yIdx]}"
            )

        # Trailing edge points are at maximum chord
        idxTE = np.where(pointsGlobal[:, self.xIdx] == np.max(pointsGlobal[:, self.xIdx]))[0]

        # The trailing edge point at the maximum y value begins the upper surface, minimum ends lower surface
        idxUpperTE = idxTE[np.argmax(pointsGlobal[idxTE, self.yIdx])]
        idxLowerTE = idxTE[np.argmin(pointsGlobal[idxTE, self.yIdx])]

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

        # # Now we have the range of indices for upper and lower surfaces
        # idxUpper = np.arange(idxUpperTE, idxLE + 1)  # include leading and trailing edges
        # idxLower = np.arange(idxLE, idxLowerTE + 1)  # include leading and trailing edges

        self.points[ptName] = {
            "points": pointsGlobal,
            "upper": np.where(upperBool)[0],
            "lower": np.where(lowerBool)[0],
            "trailingEdge": idxTE,
            "xMin": np.min(pointsGlobal[:, self.xIdx]),
            "xMax": np.max(pointsGlobal[:, self.xIdx]),
        }

    def setDesignVars(self, dvDict):
        """
        Standard routine for setting design variables from a design variable dictionary.

        Parameters
        ----------
        dvDict : dict
            Dictionary of design variables. The keys of the dictionary must correspond to the design variable names.
            Any additional keys in the dictionary are simply ignored.
        """
        pass

    def getValues(self):
        """
        Generic routine to return the current set of design variables.
        Values are returned in a dictionary format that would be suitable for a subsequent call to setValues()

        Returns
        -------
        dvDict : dict
            Dictionary of design variables
        """
        pass

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

    def getVarNames(self):
        """
        Return a list of the design variable names. This is typically used when specifying a wrt= argument for pyOptSparse.

        Examples
        --------
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        pass

    def writeToFile(self, filename):
        # TODO generalize the writing to files?
        pass

    def totalSensitivity(self, dIdpt, ptSetName):
        r"""
        This function computes sensitivity information.
        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}}^T \frac{dI}{d_{pt}}`

        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)
            This is the total derivative of the objective or function of interest with respect to the coordinates in 'ptSetName'.
            This can be a single array of size (Npt, 3) **or** a group of N vectors of size (Npt, 3, N).
            If you have many to do, it is faster to do many at once.
        ptSetName : str
            The name of set of points we are dealing with

        Returns
        -------
        dIdxDict : dict
            The dictionary containing the derivatives, suitable for pyOptSparse
        """
        pass

    def totalSensitivityProd(self, vec, ptSetName):
        r"""
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

        Returns
        -------
        xsdot : array (Nx3) -> Array with derivative seeds of the surface nodes.
        """
        pass

    def addVariablesPyOpt(self, optProb):
        """
        Add the current set of variables to the optProb object.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which variables are added
        """
        pass

    def update(self, ptSetName, childDelta=True, config=None):
        """
        This is the main routine for returning coordinates that have
        been updated by design variables.

        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the
            given in an :func:`addPointSet()` call.

        childDelta : bool
            Return updates on child as a delta. The user should not
            need to ever change this parameter.

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable applies to *ALL* configurations.
        """
        pass

    def getNDV(self):
        """
        Return the total number of design variables this object has.

        Returns
        -------
        nDV : int
            Total number of design variables
        """
        pass

    def printDesignVariables(self, directory):
        pass

    def writePointSet(self, name, fileName):
        pass

    def demoDesignVars(self, directory):
        pass

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
    def computeCSTdydN1(x, N1, N2, w, yte):
        """Compute the drivatives of the height of a CST curve with respect to N1

        Given y = C(x, N1, N2) * S(x)
        dy/dN1 = S(x) * dC/dN1 = S(x) * C(x, N1, N2) * ln(x)

        This function assumes x has been normalised to the range [0,1]
        """
        C = DVGeometryCST.computeClassShape(x, N1, N2)
        S = DVGeometryCST.computeShapeFunctions(x, w)
        return np.sum(S, axis=0) * C * np.log(x)

    @staticmethod
    def computeCSTdydN2(x, N1, N2, w, yte):
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
