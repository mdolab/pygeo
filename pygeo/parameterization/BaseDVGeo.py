"""
BaseDVGeo

Holds a basic version of a DVGeo geometry object
Enables the use of different geometry parameterizations (FFD, OpenVSP, ESP, etc) with the MACH-Aero framework
"""

# Standard Python modules
from abc import ABC, abstractmethod
from collections import OrderedDict
import copy
from difflib import get_close_matches

# External modules
import numpy as np
from scipy.optimize import least_squares
import scipy.sparse as sp


class BaseDVGeometry(ABC):
    """
    Abstract class for a basic geometry object
    """

    def __init__(self, fileName, name):
        self.fileName = fileName
        self.name = name

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

    def getValues(self):
        DeprecationWarning(
            "getValues() is deprecated and will be removed in pyGeo version 1.18. Use getDesignVars() instead. "
        )
        return self.getDesignVars()

    @abstractmethod
    def getDesignVars(self):
        """
        Generic routine to return the current set of design variables.
        Values are returned in a dictionary format that would be suitable for a subsequent call to setDesignVars()

        Returns
        -------
        dvDict : dict
            Dictionary of design variables
        """
        pass

    @abstractmethod
    def getDVBounds(self):
        """
        Return the bounds on the design variables.

        Returns
        -------
        lowerBounds : dict
            Dictionary of design variable lower bounds. The keys are the design variable names and the values are the lower bounds.
        upperBounds : dict
            Dictionary of design variable upper bounds. The keys are the design variable names and the values are
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

    @abstractmethod
    def getOrigPoints(self, ptSetName):
        """Get the original coordinates for a point set. a.k.a the coordinates that were passed to :func:`addPointSet`.

        Parameters
        ----------
        ptSetName : str
            Name of the point set to return the original coordinates for.
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

    def fitDVGeo(self, refDVGeo, ptSetName, pointFraction=None, callback=None, **kwargs):
        """
        Fit the design variables of this DVGeo object to match the pointset of another DVGeo object. This could be used,
        for example, to fit an ESP/VSP model to a geometry optimized using an FFD.

        A more general version of this capability is provided by the :func:`fitPointset` method, which fits an existing
        pointset in this DVGeo object to an arbitrary set of coordinates. This may be useful if the points you are
        trying to fit to do not come from a DVGeo object.

        Parameters
        ----------
        refDVGeo : BaseDVGeometry
            The reference DVGeometry object to fit to. The fit will be performed based on the current DVs set in this object.
        ptSetName : str
            The name of the pointset to fit.
        pointFraction : float
            Fraction of points to use for fitting. This can be useful to reduce the cost of fitting for a large pointset.
            However, the downsampling is random, so results may vary between calls. By default, all points are used.
        callback : function
            A user-supplied function to be called at each iteration. The function is called as ``callback(self, ptSetName, iteration)``
            where ``self`` is the DVGeo object being fit, ``ptSetName`` is the name of the pointset being fit, and
            ``iteration`` is the current iteration number.
        **kwargs :
            Additional keyword arguments to be passed to scipy's least squares function.

        Returns
        -------
        dict
            Design variable values that most closely fit this DVGeo to the reference pointset
        """
        # Get the reference point coordinates from the reference DVGeo object and poterntially downsample them
        refPointCoords = refDVGeo.update(ptSetName)
        numPoints = refPointCoords.shape[0]
        if pointFraction is not None:
            if pointFraction <= 0.0 or pointFraction > 1.0:
                raise ValueError("pointFraction must be between 0 and 1.")
            numPointsToFit = int(numPoints * pointFraction)
            indices = np.random.choice(numPoints, numPointsToFit, replace=False)
            refPointCoords = refPointCoords[indices, :]
        else:
            indices = np.arange(numPoints)

        # If the pointset we're using already exists in this DVGeo object, and we're not downsampling, we can use it
        # directly, after checking that it is the same. Otherwise we need to add a new pointset
        addNewPointSet = True
        if ptSetName in self.points and pointFraction is None:
            # Check that the pointset has the same number of points as the reference pointset
            ptSetsMatch = self.points[ptSetName].shape[0] == numPoints and np.allclose(
                self.points[ptSetName][indices, :], refPointCoords
            )
            if ptSetsMatch:
                addNewPointSet = False
            else:
                Warning(
                    f"Point set '{ptSetName}' already exists but does not match the reference coordinates. Adding a new pointset to use for fitting."
                )

        if addNewPointSet:
            # add the pointset, using the original coordinates
            origPointCoords = refDVGeo.getOrigPoints(ptSetName)[indices, :]
            newPtSetName = f"{ptSetName}_fit"
            ptSetName = newPtSetName
            self.addPointSet(origPointCoords, ptSetName)

        # Now we can fit the pointset to the reference coordinates
        return self.fitPointset(ptSetName, refPointCoords, callback=callback, **kwargs)

    def fitPointset(self, ptSetName, newPointCoords, callback=None, **kwargs):
        """Solve a least squares problem to find the design variables values that map a pointset as closely as possible
        to a new set of coordinates.

        Parameters
        ----------
        ptSetName : str
            The name of the pointset to fit. Must be one of the pointsets added to the DVGeo object.
        newPointCoords : array, size (N, 3)
            The new coordinates to fit the pointset to. Must have the same number of points as the pointset being fit
            and have the same ordering.
        callback : function
            A user-supplied function to be called at each iteration. The function is called as ``callback(self, ptSetName, iteration)``
            where ``self`` is the DVGeo object being fit, ``ptSetName`` is the name of the pointset being fit, and
            ``iteration`` is the current iteration number.
        **kwargs :
            Additional keyword arguments to be passed to scipy's `least_squares` function.

        Returns
        -------
        dict
            Design variable values that most closely fit the pointset to the new points.
        """
        # Verify that the pointset exists and that there are the correct number of points to fit
        if ptSetName not in self.ptSetNames:
            closestMatch = get_close_matches(ptSetName, self.ptSetNames, n=1, cutoff=0.0)[0]
            raise ValueError(f"Point set '{ptSetName}' not found. Did you mean '{closestMatch}'?")
        numPoints = self.points[ptSetName].shape[0]
        numPointsToFit = newPointCoords.shape[0]
        if numPoints != numPointsToFit:
            raise ValueError(
                f"Point set '{ptSetName}' has {numPoints} points, " f"but {numPointsToFit} points were provided to fit."
            )

        # Get the initial design variable values and their bounds
        initialDVDict = self.getDesignVars()
        lowerBoundsDict, upperBoundsDict = self.getDVBounds()
        initialDVArray = self.convertDictToSensitivity(initialDVDict).flatten()
        lowerBoundsArray = self.convertDictToSensitivity(lowerBoundsDict).flatten()
        upperBoundsArray = self.convertDictToSensitivity(upperBoundsDict).flatten()

        iteration_counter = [0]

        def computePointCoords(x):
            """Compute the coordinates of the pointset given the design variables."""
            # Set the design variables
            self.setDesignVars(self.convertSensitivityToDict(x.reshape(1, -1), out1D=True))
            # Update the pointset
            return self.update(ptSetName).reshape(-1, 3)

        def computeCoordinateResiduals(x):
            """Compute the residuals in terms of the coordinates."""
            updatedPoints = computePointCoords(x)
            if callback is not None:
                callback(self, ptSetName, iteration_counter[0])
            iteration_counter[0] += 1
            return (updatedPoints - newPointCoords).flatten()

        def computeCoordinateJacobian(x):
            """Compute the Jacobian of the coordinates."""

            # Create a seed vector for the design variables
            numDVs = self.getNDV()
            seed = np.zeros(numDVs)
            rowInd = []
            colInd = []
            values = []
            for ii in range(numDVs):
                seed[ii] = 1.0
                # Propogate seed through to the point coordinates
                coordSens = self.totalSensitivityProd(
                    self.convertSensitivityToDict(seed.reshape(1, -1)), ptSetName
                ).flatten()

                nonZeroInd = np.nonzero(coordSens)[0]
                numNonZero = nonZeroInd.size
                if numNonZero > 0:
                    values += list(coordSens[nonZeroInd])
                    colInd += [ii] * numNonZero
                    rowInd.extend(nonZeroInd)

                seed[ii] = 0.0
            return sp.csr_array((values, (rowInd, colInd)), shape=(3 * numPoints, numDVs))

        # Set some default kwargs for the least_squares function
        defaultKwargs = {"xtol": 1e-8, "ftol": 1e-8, "gtol": 1e-2, "verbose": 2, "max_nfev": 100}
        # Update the default kwargs with any user-supplied kwargs
        kwargs.update(defaultKwargs)
        # Now solve the least squares problem using scipy's least_squares function
        result = least_squares(
            computeCoordinateResiduals,
            initialDVArray,
            jac=computeCoordinateJacobian,
            bounds=(lowerBoundsArray, upperBoundsArray),
            **kwargs,
        )

        # Convert the result back to a dictionary of design variables
        dvDict = self.convertSensitivityToDict(result.x.reshape(1, -1), out1D=True)

        return dvDict
