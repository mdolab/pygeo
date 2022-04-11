"""
BaseDVGeo

Holds a basic version of a DVGeo geometry object to be used with a parametric geometry engine
Enables the use of ESP (Engineering Sketch Pad) and OpenVSP (Open Vehicle Sketch Pad) with the MACH-Aero framework

"""

# ======================================================================
#         Imports
# ======================================================================
from abc import abstractmethod
from collections import OrderedDict
import time
import numpy as np
from mpi4py import MPI
from baseclasses.utils import Error
from .BaseDVGeo import BaseDVGeo


class DVGeoSketch(BaseDVGeo):
    """A class for manipulating parametric geometry

    The purpose of the BaseDVGeoSketchy class is to provide translation
    of the ESP/OpenVSP geometry engine to externally supplied surfaces. This
    allows the use of ESP/OpenVSP design variables to control the MACH
    framework.

    There are several import limitations:

    1. Since ESP and OpenVSP are surface based only, they cannot be used to
    parameterize a geometry that doesn't lie on the surface. This
    means it cannot be used for structural analysis. It is generally
    possible use most of the constraints DVConstraints since most of
    those points lie on the surface.

    2. It cannot handle *moving* intersection. A geometry with static
    intersections is fine as long as the intersection doesn't move

    3. It does not support complex numbers for the complex-step method.

    4. It does not support separate configurations.

    5. Because of limitations with ESP and OpenVSP, this class
    uses parallel finite differencing to obtain the required Jacobian
    matrices.

    Parameters
    ----------
    fileName : str
       filename of .vsp3 file (OpenVSP) or .csm file (ESP).

    comm : MPI Intra Comm
       Comm on which to build operate the object. This is used to
       perform embarrassingly parallel finite differencing. Defaults to
       MPI.COMM_WORLD.

    scale : float
       A global scale factor from the ESP/VSP geometry to incoming (CFD) mesh
       geometry. For example, if the ESP/VSP model is in inches, and the CFD
       in meters, scale=0.0254.

    """

    def __init__(self, fileName, comm=MPI.COMM_WORLD, scale=1.0, comps=[], projTol=0.01):

        if comm.rank == 0:
            print("Initializing DVGeometry")
            t0 = time.time()

        self.points = OrderedDict()
        self.pointSets = OrderedDict()
        self.ptSetNames = []
        self.updated = {}
        self.updatedJac = {}

    @abstractmethod
    def addPointSet(self, points, ptName, **kwargs):
        """
        Add a set of coordinates to DVGeometry
        The is the main way that geometry, in the form of a coordinate
        list is given to DVGeometry to be manipulated.
        """
        pass

    @abstractmethod
    def setDesignVars(self, dvDict):
        """
        Standard routine for setting design variables from a design
        variable dictionary.

        Parameters
        ----------
        dvDict : dict
            Dictionary of design variables. The keys of the dictionary
            must correspond to the design variable names. Any
            additional keys in the dv-dictionary are simply ignored.
        """
        pass

    def getValues(self):
        """
        Generic routine to return the current set of design
        variables. Values are returned in a dictionary format
        that would be suitable for a subsequent call to setValues()

        Returns
        -------
        dvDict : dict
            Dictionary of design variables
        """
        dvDict = OrderedDict()
        for dvName in self.DVs:
            dvDict[dvName] = self.DVs[dvName].value

        return dvDict

    @abstractmethod
    def update(self, ptSetName, config=None):
        pass

    def pointSetUpToDate(self, ptSetName):
        """
        This is used externally to query if the object needs to update
        its pointset or not. Essentially what happens, is when
        update() is called with a point set, the self.updated dict
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

    @abstractmethod
    def getNDV(self):
        """
        Return the number of DVs

        Returns
        _______
        len(self.DVs) : int
            number of design variables
        """
        pass

    def getVarNames(self, pyOptSparse=False):
        """
        Return a list of the design variable names. This is typically
        used when specifying a wrt= argument for pyOptSparse.

        Examples
        --------
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        return list(self.DVs.keys())

    @abstractmethod
    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        # TODO see if VSP and ESP can be reconciled
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}}^T \frac{dI}{d_{pt}}`

        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)

            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once.

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place.

        Returns
        -------
        dIdxDict : dict
            The dictionary containing the derivatives, suitable for
            pyOptSparse
        """
        pass

    @abstractmethod
    def totalSensitivityProd(self, vec, ptSetName, comm=None, config=None):
        # TODO see if VSP and ESP can be reconciled
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}} \ vec`

        This is useful for forward AD mode.

        Parameters
        ----------
        vec : dictionary whose keys are the design variable names, and whose
              values are the derivative seeds of the corresponding design variable.

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            inactive parameter, this has no effect on the final result
            because with this method, the reduction is performed externally

        Returns
        -------
        xsdot : array (Nx3) -> Array with derivative seeds of the surface nodes.
        """
        pass

    @abstractmethod
    def addVariable(self):
        """
        Add a design variable definition.
        """
        pass

    # TODO I think these have to stay separate - VSP doesn't have grouped DVs
    @abstractmethod
    def addVariablesPyOpt(self, optProb):
        """
        Add the current set of variables to the optProb object.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which variables are added
        """
        pass

    @abstractmethod
    def printDesignVariables(self):
        """
        Print a formatted list of design variables to the screen
        """
        pass

    # ----------------------------------------------------------------------- #
    #      THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER      #
    # ----------------------------------------------------------------------- #

    @abstractmethod
    def _updateModel(self):
        """
        Set each of the DVs in the internal ESP/VSP model
        """
        pass

    @abstractmethod
    def _updateProjectedPts(self):
        """
        Internally updates the coordinates of the projected points
        """
        pass

    @abstractmethod
    def _computeSurfJacobian(self):
        """
        This routine comptues the jacobian of the surface with respect
        to the design variables. Since our point sets are rigidly linked to
        the projection points, this is all we need to calculate. The input
        pointSets is a list or dictionary of pointSets to calculate the jacobian for.
        """
        pass


# TODO PointSet is the same for both - pull out on its own?
class PointSet:
    """Internal class for storing the projection details of each pointset"""

    def __init__(self, points, pts, geom, u, v):
        self.points = points
        self.pts = pts
        self.geom = geom
        self.u = u
        self.v = v
        self.offset = self.pts - self.points
        self.nPts = len(self.pts)
        self.jac = None
