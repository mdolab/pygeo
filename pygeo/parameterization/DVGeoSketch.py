"""
BaseDVGeo

Holds a basic version of a DVGeo geometry object to be used with a parametric geometry engine
Enables the use of ESP (Engineering Sketch Pad) and OpenVSP (Open Vehicle Sketch Pad) with the MACH-Aero framework

"""

# Standard Python modules
from abc import abstractmethod
from collections import OrderedDict

# External modules
from mpi4py import MPI
import numpy as np
from pyspline.utils import closeTecplot, openTecplot, writeTecplot1D

# Local modules
from .BaseDVGeo import BaseDVGeometry
from .designVars import geoDVComposite


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

    def addCompositeDV(self, dvName, ptSetName=None, u=None, scale=None, comm=None):
        """
        Add composite DVs. Note that this is essentially a preprocessing call which only works in serial
        at the moment.

        Parameters
        ----------
        dvName : str
            The name of the composite DVs
        ptSetName : str, optional
            If the matrices need to be computed, then a point set must be specified, by default None
        u : ndarray, optional
            The u matrix used for the composite DV, by default None
        scale : float or ndarray, optional
            The scaling applied to this DV, by default None
        """
        NDV = self.getNDV()

        if u is not None:
            # we are after a square matrix
            if u.shape != (NDV, NDV):
                raise ValueError(f"The shapes don't match! Got shape = {u.shape} but NDV = {NDV}")
            if scale is None:
                raise ValueError("If u is provided, then scale must also be provided.")
            s = None
        else:
            if ptSetName is None:
                raise ValueError("If u and s need to be computed, you must specify the ptSetName")
            if not self.updatedJac[ptSetName]:
                self._computeSurfJacobian()

            J_full = self.pointSets[ptSetName].jac

            u, s, _ = np.linalg.svd(J_full.T, full_matrices=False)

            scale = np.sqrt(s)
            # normalize the scaling
            scale = scale * (NDV / np.sum(scale))

        # map the initial design variable values
        # we do this manually instead of calling self.mapVecToComp
        # because self.DVComposite.u isn't available yet
        values = u.T @ self.convertDictToSensitivity(self.getValues())

        self.DVComposite = geoDVComposite(dvName, values, NDV, u, scale=scale, s=s)

        self.useComposite = True

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

        if self.useComposite:
            dvDict = self.mapXDictToComp(dvDict)

        return dvDict

    def getVarNames(self, pyOptSparse=False):
        """
        Return a list of the design variable names. This is typically
        used when specifying a wrt= argument for pyOptSparse.

        Examples
        --------
        >>> optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        if not self.useComposite:
            names = list(self.DVs.keys())
        else:
            names = [self.DVComposite.name]

        return names

    def convertDictToSensitivity(self, dIdxDict):
        """
        This function performs the reverse operation of
        convertSensitivityToDict(); it transforms the dictionary back
        into an array. This function is important for the matrix-free
        interface.

        Parameters
        ----------
        dIdxDict : dictionary
           Dictionary of information keyed by this object's
           design variables

        Returns
        -------
        dIdx : array
           Flattened array of length getNDV().
        """
        DVCount = self.getNDV()

        dIdx = np.zeros(DVCount, dtype="d")
        i = 0
        for key in self.DVs:
            dv = self.DVs[key]
            dIdx[i : i + dv.nVal] = dIdxDict[key]
            i += dv.nVal

        return dIdx

    def convertSensitivityToDict(self, dIdx, out1D=False, useCompositeNames=False):
        """
        This function takes the result of totalSensitivity and
        converts it to a dict for use in pyOptSparse

        Parameters
        ----------
        dIdx : array
           Flattened array of length getNDV(). Generally it comes from
           a call to totalSensitivity()
        out1D : boolean
            If true, creates a 1D array in the dictionary instead of 2D.
            This function is used in the matrix-vector product calculation.
        useCompositeNames : boolean
            Whether the sensitivity dIdx is with respect to the composite DVs or the original DVGeo DVs.
            If False, the returned dictionary will have keys corresponding to the original set of geometric DVs.
            If True,  the returned dictionary will have replace those with a single key corresponding to the composite DV name.

        Returns
        -------
        dIdxDict : dictionary
           Dictionary of the same information keyed by this object's
           design variables
        """

        # compute the various DV offsets
        # DVCountGlobal= self._getDVOffsets()

        i = 0
        dIdxDict = {}
        for key in self.DVs:
            dv = self.DVs[key]
            if out1D:
                dIdxDict[key] = np.ravel(dIdx[:, i : i + dv.nVal])
            else:
                dIdxDict[key] = np.array(dIdx[:, i : i + dv.nVal])
            i += dv.nVal

        # replace other names with user
        if useCompositeNames and self.useComposite:
            array = []
            for _key, val in dIdxDict.items():
                array.append(val)
            array = np.column_stack(array)
            dIdxDict = {self.DVComposite.name: array}

        return dIdxDict

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
        # add the linear DV constraints that replace the existing bounds!
        if self.useComposite:
            dv = self.DVComposite
            optProb.addVarGroup(dv.name, dv.nVal, "c", value=dv.value, lower=dv.lower, upper=dv.upper, scale=dv.scale)
            lb = {}
            ub = {}

            for dvName in self.DVs:
                dv = self.DVs[dvName]
                lb[dvName] = dv.lower
                ub[dvName] = dv.upper

            lb = self.convertDictToSensitivity(lb)
            ub = self.convertDictToSensitivity(ub)

            optProb.addConGroup(
                f"{self.DVComposite.name}_con",
                self.getNDV(),
                lower=lb,
                upper=ub,
                scale=1.0,
                linear=True,
                wrt=self.DVComposite.name,
                jac={self.DVComposite.name: self.DVComposite.u},
            )
        else:
            for dvName in self.DVs:
                dv = self.DVs[dvName]
                optProb.addVarGroup(
                    dvName, dv.nVal, "c", value=dv.value, lower=dv.lower, upper=dv.upper, scale=dv.scale
                )

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
