"""
BaseDVGeo

Holds a basic version of a DVGeo geometry object
Enables the use of different geometry parameterizations (FFD, OpenVSP, ESP, etc) with the MACH-Aero framework
"""

from abc import abstractmethod
from typing import OrderedDict
from pyspline.utils import openTecplot, closeTecplot, writeTecplot1D, writeTecplot3D


class BaseDVGeo:
    """
    Abstract class for a basic geometry object
    """

    def init__(self, name, category, fileName):
        # TODO add docstring once all params are in
        self.name = name
        self.category = category
        self.fileName = fileName  # TODO - each type has a different file and build process - can this be expanded in the respective init or do we need a separate method?

        self.points = OrderedDict
        self.pointSets = OrderedDict
        self.updated = {}
        # TODO there is a lot of stuff in the ESP and VSP inits should the building be moved out?

    # TODO There are a lof of methods duplicated just between DVGeoESP and DVGeoVSP - should there be another class they inherit from under this one?

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
    def demoDesignVars(self, directory):
        """
        This function can be used to "test" the design variable parametrization
        for a given optimization problem. It should be called in the script
        after DVGeo has been set up. The function will loop through all the
        design variables and write out a deformed FFD volume for the upper
        and lower bound of every design variable. It can also write out
        deformed point sets and surface meshes.
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
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        pass

    @abstractmethod
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
    def printDesignVariables(self, directory):
        """
        Print a formatted list of design variables to the screen
        """
        pass

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

    @abstractmethod
    def writePointSet(self, name, fileName):
        """
        Write a given point set to a tecplot file

        Parameters
        ----------
        name : str
             The name of the point set to write to a file

        fileName : str
           Filename for tecplot file. Should have no extension, an extension will be added
        """
        coords = self.update(name, childDelta=True)
        fileName = fileName + "_%s.dat" % name
        f = openTecplot(fileName, 3)
        writeTecplot1D(f, name, coords)
        closeTecplot(f)

    @abstractmethod
    def writeToFile(self, filename):
        # TODO generalize the writing to files?
        pass

    # TODO should there be a base class for design variables? regular has global and local, VSP/ESP has one general type
