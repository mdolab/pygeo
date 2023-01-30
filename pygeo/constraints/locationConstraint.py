# External modules
import numpy as np

# Local modules
from .baseConstraint import GeometricConstraint


class LocationConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of location
    constraints. One of these objects is created each time a
    addLocationConstraints1D call is
    made. The user should not have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords.flatten()), lower, upper, scale, DVGeo, addToPyOpt)
        self.coords = coords
        self.scaled = scaled

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths
        self.X0 = np.zeros(self.nCon)
        X = self.coords.flatten()
        for i in range(self.nCon):
            self.X0[i] = X[i]

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        X = self.coords.flatten()
        if self.scaled:
            for i in range(self.nCon):
                X[i] /= self.X0[i]

        funcs[self.name] = X

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dTdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))
            counter = 0
            for i in range(self.coords.shape[0]):
                for j in range(self.coords.shape[1]):
                    dTdPt[counter][i][j] = 1.0
                    if self.scaled:
                        dTdPt[counter][i][j] /= self.X0[i]
                    counter += 1

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dTdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) - 1))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) - 1):
            handle.write("%d %d\n" % (i + 1, i + 2))
