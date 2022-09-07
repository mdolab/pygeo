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
from mpi4py import MPI
from .BaseDVGeo import BaseDVGeometry
from pyspline.utils import openTecplot, closeTecplot, writeTecplot1D


class DVGeoSketch(BaseDVGeometry):
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

    def __init__(self, fileName, comm=MPI.COMM_WORLD, scale=1.0, projTol=0.01):
        super().__init__(fileName=fileName)

        # this scales coordinates from model to mesh geometry
        self.modelScale = scale
        # and this scales coordinates from mesh to model geometry
        self.meshScale = 1.0 / scale
        self.projTol = projTol * self.meshScale  # default input is in meters.

        self.updatedJac = {}
        self.comm = comm

        # Initial list of DVs
        self.DVs = OrderedDict()

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
    def addVariable(self):
        """
        Add a design variable definition.
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
        for dvName in self.DVs:
            dv = self.DVs[dvName]
            optProb.addVarGroup(dvName, dv.nVal, "c", value=dv.value, lower=dv.lower, upper=dv.upper, scale=dv.scale)

    def writePointSet(self, name, fileName):
        """
        Write a given point set to a tecplot file

        Parameters
        ----------
        name : str
             The name of the point set to write to a file

        fileName : str
           Filename for tecplot file. Should have no extension, an
           extension will be added
        """
        coords = self.update(name)
        fileName = fileName + "_%s.dat" % name
        f = openTecplot(fileName, 3)
        writeTecplot1D(f, name, coords)
        closeTecplot(f)

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
