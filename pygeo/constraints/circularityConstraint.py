# ======================================================================
#         Imports
# ======================================================================
import numpy as np
from . import geo_utils, pyGeo
from pyspline import Curve
from mpi4py import MPI
from scipy.sparse import csr_matrix
from collections import OrderedDict


class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a expliclty raised exception.
    """

    def __init__(self, message):
        msg = "\n+" + "-" * 78 + "+" + "\n" + "| DVCon Error: "
        i = 14
        for word in message.split():
            if len(word) + i + 1 > 78:  # Finish line and start new one
                msg += " " * (78 - i) + "|\n| " + word + " "
                i = 1 + len(word) + 1
            else:
                msg += word + " "
                i += len(word) + 1
        msg += " " * (78 - i) + "|\n" + "+" + "-" * 78 + "+" + "\n"
        print(msg)
        Exception.__init__(self)


class Warning(object):
    """
    Format a warning message
    """

    def __init__(self, message):
        msg = "\n+" + "-" * 78 + "+" + "\n" + "| DVConstraints Warning: "
        i = 24
        for word in message.split():
            if len(word) + i + 1 > 78:  # Finish line and start new one
                msg += " " * (78 - i) + "|\n| " + word + " "
                i = 1 + len(word) + 1
            else:
                msg += word + " "
                i += len(word) + 1
        msg += " " * (78 - i) + "|\n" + "+" + "-" * 78 + "+" + "\n"
        print(msg)


class CircularityConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of circularity
    constraint. One of these objects is created each time a
    addCircularityConstraints call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, center, coords, lower, upper, scale, DVGeo, addToPyOpt):
        self.name = name
        self.center = np.array(center).reshape((1, 3))
        self.coords = coords
        self.nCon = self.coords.shape[0] - 1
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(
            self, self.name, self.nCon, self.lower, self.upper, self.scale, self.DVGeo, self.addToPyOpt
        )

        self.X = np.zeros(self.nCon)

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name + "coords")
        self.DVGeo.addPointSet(self.center, self.name + "center")

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name + "coords", config=config)
        self.center = self.DVGeo.update(self.name + "center", config=config)

        self._computeLengths(self.center, self.coords, self.X)

        funcs[self.name] = self.X

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
            dLndPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))
            dLndCn = np.zeros((self.nCon, self.center.shape[0], self.center.shape[1]))

            xb = np.zeros(self.nCon)
            for con in range(self.nCon):
                centerb = dLndCn[con, 0, :]
                coordsb = dLndPt[con, :, :]
                xb[:] = 0.0
                xb[con] = 1.0
                # reflength2 = 0
                # for i in range(3):
                #     reflength2 = reflength2 + (center[i]-coords[0,i])**2
                reflength2 = np.sum((self.center - self.coords[0, :]) ** 2)
                reflength2b = 0.0
                for i in range(self.nCon):
                    # length2 = 0
                    # for j in range(3):
                    #     length2 = length2 + (center[j]-coords[i+1, j])**2
                    length2 = np.sum((self.center - self.coords[i + 1, :]) ** 2)

                    if length2 / reflength2 == 0.0:
                        tempb1 = 0.0
                    else:
                        tempb1 = xb[i] / (2.0 * np.sqrt(length2 / reflength2) * reflength2)
                    length2b = tempb1
                    reflength2b = reflength2b - length2 * tempb1 / reflength2
                    xb[i] = 0.0
                    for j in reversed(range(3)):
                        tempb0 = 2 * (self.center[0, j] - self.coords[i + 1, j]) * length2b
                        centerb[j] = centerb[j] + tempb0
                        coordsb[i + 1, j] = coordsb[i + 1, j] - tempb0
                for j in reversed(range(3)):  # DO i=3,1,-1
                    tempb = 2 * (self.center[0, j] - self.coords[0, j]) * reflength2b
                    centerb[j] = centerb[j] + tempb
                    coordsb[0, j] = coordsb[0, j] - tempb

            tmpPt = self.DVGeo.totalSensitivity(dLndPt, self.name + "coords", config=config)
            tmpCn = self.DVGeo.totalSensitivity(dLndCn, self.name + "center", config=config)
            tmpTotal = {}
            for key in tmpPt:
                tmpTotal[key] = tmpPt[key] + tmpCn[key]

            funcsSens[self.name] = tmpTotal

    def _computeLengths(self, center, coords, X):
        """
        compute the lengths from the center and coordinates
        """
        reflength2 = np.sum((center - coords[0, :]) ** 2)
        for i in range(self.nCon):
            length2 = np.sum((self.center - self.coords[i + 1, :]) ** 2)
            X[i] = np.sqrt(length2 / reflength2)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s_coords\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) - 1))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write("%f %f %f\n" % (self.coords[i, 0], self.coords[i, 1], self.coords[i, 2]))

        for i in range(len(self.coords) - 1):
            handle.write("%d %d\n" % (i + 1, i + 2))

        handle.write("Zone T=%s_center\n" % self.name)
        handle.write("Nodes = 2, Elements = 1 ZONETYPE=FELINESEG\n")
        handle.write("DATAPACKING=POINT\n")
        handle.write("%f %f %f\n" % (self.center[0, 0], self.center[0, 1], self.center[0, 2]))
        handle.write("%f %f %f\n" % (self.center[0, 0], self.center[0, 1], self.center[0, 2]))
        handle.write("%d %d\n" % (1, 2))
