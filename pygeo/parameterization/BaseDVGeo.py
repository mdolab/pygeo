"""
BaseDVGeo

Holds a basic version of a DVGeo geometry storage object
Enables the use of different geometry parameterizations (OpenVSP, ESP, etc) with the MACH-Aero framework
"""

from abc import abstractmethod
from typing import OrderedDict


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
    def writeToFile(self, filename):
        # TODO generalize the writing to files?
        pass

    @abstractmethod
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
        vec : dictionary whose keys are the design variable names, and whose
              values are the derivative seeds of the corresponding design variable.

        ptSetName : str
            The name of set of points we are dealing with

        Returns
        -------
        xsdot : array (Nx3) -> Array with derivative seeds of the surface nodes.
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
    def printDesignVariables(self, directory):
        pass

    @abstractmethod
    def writePointSet(self, name, fileName):
        pass

    @abstractmethod
    def demoDesignVars(self, directory):
        pass

    # TODO should there be a base class for design variables? regular has global and local, VSP/ESP has one general type
