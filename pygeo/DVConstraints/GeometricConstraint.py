class GeometricConstraint(object):
    """
    This is a generic base class for all of the geometric constraints.

    """

    def __init__(self, name, nCon, lower, upper, scale, DVGeo, addToPyOpt):
        """
        General init function. Every constraint has these functions
        """
        self.name = name
        self.nCon = nCon
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        return

    def setDesignVars(self, x):
        """
        take in the design var vector from pyopt and set the variables for this constraint
        This function is constraint specific, so the baseclass doesn't implement anything.
        """
        pass

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary.
        This function is constraint specific, so the baseclass doesn't implement anything.

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        pass

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary
        This function is constraint specific, so the baseclass doesn't implement anything.

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        pass

    def getVarNames(self):
        """
        return the var names relevant to this constraint. By default, this is the DVGeo
        variables, but some constraints may extend this to include other variables.
        """
        return self.DVGeo.getVarNames()

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addConGroup(
                self.name, self.nCon, lower=self.lower, upper=self.upper, scale=self.scale, wrt=self.getVarNames()
            )

    def addVariablesPyOpt(self, optProb):
        """
        Add the variables to pyOpt, if the flag is set
        """
        # if self.addToPyOpt:
        #     optProb.addVarGroup(self.name, self.nCon, lower=self.lower,
        #                         upper=self.upper, scale=self.scale,
        #                         wrt=self.getVarNames())

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        This function is constraint specific, so the baseclass doesn't implement anything.
        """
        pass
