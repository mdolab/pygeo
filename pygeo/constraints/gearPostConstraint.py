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


class GearPostConstraint(GeometricConstraint):
    """
    This class is used to represet a single volume constraint. The
    parameter list is explained in the addVolumeConstaint() of
    the DVConstraints class
    """

    def __init__(
        self,
        name,
        wimpressCalc,
        up,
        down,
        thickLower,
        thickUpper,
        thickScaled,
        MACFracLower,
        MACFracUpper,
        DVGeo,
        addToPyOpt,
    ):

        self.name = name
        self.wimpress = wimpressCalc
        self.thickLower = thickLower
        self.thickUpper = thickUpper
        self.thickScaled = thickScaled
        self.MACFracLower = MACFracLower
        self.MACFracUpper = MACFracUpper
        self.coords = np.array([up, down])
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, None, None, None, None, self.DVGeo, self.addToPyOpt)
        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name)

        # Compute the reference length
        self.D0 = np.linalg.norm(self.coords[0] - self.coords[1])

    def evalFunctions(self, funcs, config):

        # Update the gear post locations
        self.coords = self.DVGeo.update(self.name, config=config)

        # Compute the thickness constraint
        D = np.linalg.norm(self.coords[0] - self.coords[1])
        if self.thickScaled:
            D = D / self.D0

        # Compute the values we need from the wimpress calc
        wfuncs = {}
        self.wimpress.evalFunctions(wfuncs)

        # Now the constraint value is
        postLoc = 0.5 * (self.coords[0, 0] + self.coords[1, 0])
        locCon = (postLoc - wfuncs["xLEMAC"]) / wfuncs["MAC"]

        # Final set of two constrains
        funcs[self.name + "_thick"] = D
        funcs[self.name + "_MAC"] = locCon

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

            wfuncs = {}
            self.wimpress.evalFunctions(wfuncs)

            wSens = {}
            self.wimpress.evalFunctionsSens(wSens)

            # Accumulate the derivative into p1b and p2b
            p1b, p2b = geo_utils.eDist_b(self.coords[0, :], self.coords[1, :])
            if self.thickScaled:
                p1b /= self.D0
                p2b /= self.D0

            funcsSens[self.name + "_thick"] = self.DVGeo.totalSensitivity(
                np.array([[p1b, p2b]]), self.name, config=config
            )

            # And now we need the sensitivty of the conLoc calc
            p1b[:] = 0
            p2b[:] = 0
            p1b[0] += 0.5 / wfuncs["MAC"]
            p2b[0] += 0.5 / wfuncs["MAC"]

            tmpSens = self.DVGeo.totalSensitivity(np.array([[p1b, p2b]]), self.name, config=config)

            # And we need the sensitity of conLoc wrt 'xLEMAC' and 'MAC'
            postLoc = 0.5 * (self.coords[0, 0] + self.coords[1, 0])
            for key in wSens["xLEMAC"]:
                tmpSens[key] -= wSens["xLEMAC"][key] / wfuncs["MAC"]
                tmpSens[key] += wfuncs["xLEMAC"] / wfuncs["MAC"] ** 2 * wSens["MAC"][key]
                tmpSens[key] -= postLoc / wfuncs["MAC"] ** 2 * wSens["MAC"][key]
            funcsSens[self.name + "_MAC"] = tmpSens

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addCon(
                self.name + "_thick", lower=self.thickLower, upper=self.thickUpper, wrt=self.DVGeo.getVarNames()
            )

            optProb.addCon(
                self.name + "_MAC", lower=self.MACFracLower, upper=self.MACFracUpper, wrt=self.DVGeo.getVarNames()
            )
