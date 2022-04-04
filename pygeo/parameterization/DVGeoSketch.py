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

# mdolab packages

from pygeo.parameterization.BaseDVGeo import BaseDVGeo


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

    def addPointSet(self, points, ptName, **kwargs):
        pass

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

        # Just dump in the values
        for key in dvDict:
            if key in self.DVs:
                self.DVs[key].value = dvDict[key]

        # we just need to set the design variables in the VSP model and we are done
        self._updateModel()

        # update the projected coordinates
        self._updateProjectedPts()

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

        # set the jacobian flag to false
        for ptName in self.pointSets:
            self.updatedJac[ptName] = False

        # TODO these are not exactly the same method, maybe change ESP

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

    def update(self, ptSetName, config=None):
        pass
        # TODO these look too dissimilar?

    def writeVSPFile(self, fileName, exportSet=0):
        """
        Take the current design and write a new VSP file

        Parameters
        ----------
        fileName : str
            Name of the output VSP file
        exportSet : int
            optional input parameter to select an export set in VSP
        """
        openvsp.WriteVSPFile(fileName, exportSet)
        # TODO can this be general?

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
        # TODO i think this is the same everywhere? just put in base base

    def getNDV(self):
        """
        Return the number of DVs

        Returns
        _______
        len(self.DVs) : int
            number of design variables
        """
        return len(self.DVs)  # TODO change the name of this in ESP, global doesn't make sense

    # this one is the same for both
    def getVarNames(self, pyOptSparse=False):
        """
        Return a list of the design variable names. This is typically
        used when specifying a wrt= argument for pyOptSparse.

        Examples
        --------
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        return list(self.DVs.keys())

    # TODO i think these are doing the same things in different orders. double check
    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
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

        # We may not have set the variables so the surf jac might not be computed.
        if self.pointSets[ptSetName].jac is None:
            # in this case, we updated our pts when we added our pointset,
            # therefore the reference pts are up to date.
            self._computeSurfJacobian()

        # if the jacobian for this pointset is not up to date
        # update all the points
        if not self.updatedJac[ptSetName]:
            self._computeSurfJacobian()

        # The following code computes the final sensitivity product:
        #
        #        T       T
        #   pXpt     pI
        #  ------  ------
        #   pXdv    pXpt
        #
        # Where I is the objective, Xpt are the externally coordinates
        # supplied in addPointSet

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])

        # reshape the dIdpt array from [N] * [nPt] * [3] to  [N] * [nPt*3]
        dIdpt = dIdpt.reshape((dIdpt.shape[0], dIdpt.shape[1] * 3))

        # jacobian of the projected points
        jac = self.pointSets[ptSetName].jac

        # local product
        dIdxT_local = jac.T.dot(dIdpt.T)

        # take the transpose to get the jacobian itself dI/dx
        dIdx_local = dIdxT_local.T

        # sum the results if we are provided a comm
        if comm:
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # Now convert to dict:
        dIdxDict = {}
        i = 0
        for dvName in self.DVs:
            dIdxDict[dvName] = np.array(dIdx[:, i]).T
            i += 1

        return dIdxDict

    def totalSensitivityProd(self, vec, ptSetName, comm=None, config=None):
        r"""
        This function computes sensitivity information.
        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}} \vec'`

        Parameters
        ----------
        vec : dictionary whose keys are the design variable names, and whose
              values are the derivative seeds of the corresponding design variable.
        ptSetName : str
            The name of set of points we are dealing with
        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place.

        Returns
        -------
        xsdot : array (Nx3) -> Array with derivative seeds of the surface nodes.
        """

        # We may not have set the variables so the surf jac might not be computed.
        if self.pointSets[ptSetName].jac is None:
            # in this case, we updated our pts when we added our pointset,
            # therefore the reference pts are up to date.
            self._computeSurfJacobian()

        # if the jacobian for this pointset is not up to date
        # update all the points
        if not self.updatedJac[ptSetName]:
            self._computeSurfJacobian()

        # vector that has all the derivative seeds of the design vars
        newvec = np.zeros(self.getNDV())

        # populate newvec
        for i, dv in enumerate(self.DVs):
            if dv in vec:
                newvec[i] = vec[dv]

        # we need to multiply with the surface jacobian
        dPt = self.pointSets[ptSetName].jac.dot(newvec)

        return dPt

    # totalSensitivityTransProd is only VSP

    @abstractmethod
    def addVariable(self):
        """
        Add a design variable definition.
        """
        pass

    # TODO VSP uses addVar not addVarGroup so this might not be equivalent
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
            optProb.addVarGroup(dv.name, dv.nVal, "c", value=dv.value, lower=dv.lower, upper=dv.upper, scale=dv.scale)

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
        Set each of the DVs for the respective parametric model.
        """
        pass

    @abstractmethod
    def _updateProjectedPts(self):
        """
        Internally updates the coordinates of the projected points.
        """
        pass

    @abstractmethod
    def _computeSurfJacobian(self):
        """
        Comptues the jacobian of the surface with respect to the design variables.
        """
        pass
