"""
BaseDVGeo

Holds a basic version of a DVGeo geometry object
Enables the use of different geometry parameterizations (FFD, OpenVSP, ESP, etc) with the MACH-Aero framework
"""

# Standard Python modules
from abc import ABC, abstractmethod
from collections import OrderedDict
import copy


class BaseDVGeometry(ABC):
    """
    Abstract class for a basic geometry object
    """

    def __init__(self, fileName):
        self.fileName = fileName

        self.points = OrderedDict()
        self.pointSets = OrderedDict()
        self.updated = {}
        self.ptSetNames = []

    @abstractmethod
    def addPointSet(self, points, ptName):
        """
        Add a set of coordinates to DVGeometry
        The is the main way that geometry in the form of a coordinate list is given to DVGeometry to be manipulated.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed. These coordinates *should* all project into the interior of the geometry.
        ptName : str
            A user supplied name to associate with the set of coordinates.
            This name will need to be provided when updating the coordinates or when getting the derivatives of the coordinates.
        """
        pass

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
    def getNDV(self):
        """
        Return the total number of design variables this object has.

        Returns
        -------
        nDV : int
            Total number of design variables
        """
        pass

    @abstractmethod
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

    @abstractmethod
    def getVarNames(self):
        """
        Return a list of the design variable names. This is typically used when specifying a wrt= argument for pyOptSparse.

        Examples
        --------
        >>> optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        pass

    def pointSetUpToDate(self, ptSetName):
        """
        This is used externally to query if the object needs to update its pointset or not.
        Essentially what happens is when update() is called with a point set, the self.updated dict entry for pointSet is flagged as true.
        Here we just return that flag. When design variables are set, we then reset all the flags to False since,
        when DVs are set, nothing (in general) will be up to date anymore.

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

    @abstractmethod
    def totalSensitivity(self, dIdpt, ptSetName, comm=None):
        # TODO see if VSP and ESP can be reconciled
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

        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If comm is None, no reduction takes place.

        Returns
        -------
        dIdxDict : dic
            The dictionary containing the derivatives, suitable for pyOptSparse
        """
        pass

    @abstractmethod
    def totalSensitivityProd(self, vec, ptSetName):
        # TODO see if VSP and ESP can be reconciled
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}} \times\mathrm{vec}`

        This is useful for forward AD mode.

        Parameters
        ----------
        vec : dictionary whose keys are the design variable names
            and whose values are the derivative seeds of the corresponding design variable.

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If comm is None, no reduction takes place.

        Returns
        -------
        xsdot : array (Nx3) -> Array with derivative seeds of the surface nodes.
        """
        pass

    @abstractmethod
    def update(self, ptSetName):
        """
        This is the main routine for returning coordinates that have been updated by design variables.

        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the given in an :func:`addPointSet()` call.
        """
        pass

    def mapXDictToDVGeo(self, inDict):
        """
        Map a dictionary of DVs to the 'DVGeo' design, while keeping non-DVGeo DVs in place
        without modifying them

        Parameters
        ----------
        inDict : dict
            The dictionary of DVs to be mapped

        Returns
        -------
        dict
            The mapped DVs in the same dictionary format
        """
        # first make a copy so we don't modify in place
        inDictBase = inDict
        userVec = inDict[self.DVComposite.name]
        outVec = self.mapVecToDVGeo(userVec)
        outDict = self.convertSensitivityToDict(outVec.reshape(1, -1), out1D=True, useCompositeNames=False)
        # now merge inDict and outDict
        for key in inDict:
            outDict[key] = inDictBase[key]
        return outDict

    def mapXDictToComp(self, inDict):
        """
        The inverse of :func:`mapXDictToDVGeo`, where we map the DVs to the composite space

        Parameters
        ----------
        inDict : dict
            The DVs to be mapped

        Returns
        -------
        dict
            The mapped DVs
        """
        # first make a copy so we don't modify in place
        inDict = copy.deepcopy(inDict)
        userVec = self.convertDictToSensitivity(inDict)
        outVec = self.mapVecToComp(userVec)
        outDict = self.convertSensitivityToDict(outVec.reshape(1, -1), out1D=True, useCompositeNames=True)
        return outDict

    def mapVecToDVGeo(self, inVec):
        """
        This is the vector version of :func:`mapXDictToDVGeo`, where the actual mapping is done

        Parameters
        ----------
        inVec : ndarray
            The DVs in a single 1D array

        Returns
        -------
        ndarray
            The mapped DVs in a single 1D array
        """
        inVec = inVec.reshape(self.getNDV(), -1)
        outVec = self.DVComposite.u @ inVec
        return outVec.flatten()

    def mapVecToComp(self, inVec):
        """
        This is the vector version of :func:`mapXDictToComp`, where the actual mapping is done

        Parameters
        ----------
        inVec : ndarray
            The DVs in a single 1D array

        Returns
        -------
        ndarray
            The mapped DVs in a single 1D array
        """
        inVec = inVec.reshape(self.getNDV(), -1)
        outVec = self.DVComposite.u.transpose() @ inVec
        return outVec.flatten()

    def mapSensToComp(self, inVec):
        """
        Maps the sensitivity matrix to the composite design space

        Parameters
        ----------
        inVec : ndarray
            The sensitivities to be mapped

        Returns
        -------
        ndarray
            The mapped sensitivity matrix
        """
        outVec = inVec @ self.DVComposite.u  # this is the same as (self.DVComposite.u.T @ inVec.T).T
        return outVec
