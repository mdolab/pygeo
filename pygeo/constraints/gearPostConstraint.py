# ======================================================================
#         Imports
# ======================================================================
import numpy as np
from .. import geo_utils
from .baseConstraint import GeometricConstraint


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
        super().__init__(name, None, None, None, None, DVGeo, addToPyOpt)

        self.wimpress = wimpressCalc
        self.thickLower = thickLower
        self.thickUpper = thickUpper
        self.thickScaled = thickScaled
        self.MACFracLower = MACFracLower
        self.MACFracUpper = MACFracUpper
        self.coords = np.array([up, down])

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

            # And now we need the sensitivity of the conLoc calc
            p1b[:] = 0
            p2b[:] = 0
            p1b[0] += 0.5 / wfuncs["MAC"]
            p2b[0] += 0.5 / wfuncs["MAC"]

            tmpSens = self.DVGeo.totalSensitivity(np.array([[p1b, p2b]]), self.name, config=config)

            # And we need the sensitivity of conLoc wrt 'xLEMAC' and 'MAC'
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

    def writeTecplot(self, handle):
        raise NotImplementedError()
