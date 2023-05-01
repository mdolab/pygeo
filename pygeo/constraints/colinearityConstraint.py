# External modules
import numpy as np

# Local modules
from .. import geo_utils
from .baseConstraint import GeometricConstraint


class ColinearityConstraint(GeometricConstraint):
    """
    DVConstraints representation of a colinearity constraint.
    Constrain that all of the points provided stay colinear with the
    specified axis.
    One of these objects is created each time an
    addColinearityConstraint call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, axis, origin, coords, lower, upper, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords), lower, upper, scale, DVGeo, addToPyOpt)

        # create the output array
        self.X = np.zeros(self.nCon)

        # The first thing we do is convert v1 and v2 to coords
        self.axis = axis
        self.origin = origin
        self.coords = coords

        # Now embed the coordinates and origin into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.origin, self.name + "origin", compNames=compNames)
        self.DVGeo.addPointSet(self.coords, self.name + "coords", compNames=compNames)

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
        self.origin = self.DVGeo.update(self.name + "origin", config=config)

        # # Compute the direction from each point to the origin
        # dirVec = self.origin-self.coords

        # # compute the cross product with the desired axis. Cross product
        # # will be zero if the direction vector is the same as the axis
        # resultDir = np.cross(self.axis,dirVec)

        # for i in range(len(resultDir)):
        #     self.X[i] = geo_utils.euclideanNorm(resultDir[i,:])
        self.X = geo_utils.norm.computeDistToAxis(self.origin, self.coords, self.axis)

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
            dCdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))
            dCdOrigin = np.zeros((self.nCon, self.origin.shape[0], self.origin.shape[1]))
            dCdAxis = np.zeros((self.nCon, self.axis.shape[0], self.axis.shape[1]))

            # Compute the direction from each point to the origin
            # for i in range(n):
            #     for j in range(3):
            #         dirvec[i, j] = origin[j] - coords[i, j]
            dirVec = self.origin - self.coords

            # axisb = 0.0
            # dirvecb = 0.0
            # for i in range(self.nCon):
            #     resultdir = np.cross(axis, dirvec[i, :])
            #     self.X[i] = 0
            #     for j in range(3):
            #         self.X[i] = self.X[i] + resultdir[j]**2
            resultDir = np.cross(self.axis, dirVec)
            tmpX = np.zeros(self.nCon)
            for i in range(len(resultDir)):
                # self.X[i] = geo_utils.euclideanNorm(resultDir[i,:])
                for j in range(3):
                    tmpX[i] += resultDir[i, j] ** 2

            resultdirb = np.zeros(3)
            dirvecb = np.zeros_like(dirVec)
            xb = np.zeros(self.nCon)
            for con in range(self.nCon):
                originb = dCdOrigin[con, 0, :]
                coordsb = dCdPt[con, :, :]
                axisb = dCdAxis[con, 0, :]
                xb[:] = 0.0
                xb[con] = 1.0

                for i in range(self.nCon):
                    if tmpX[i] == 0.0:
                        xb[i] = 0.0
                    else:
                        xb[i] = xb[i] / (2.0 * np.sqrt(tmpX[i]))

                    resultdirb[:] = 0.0
                    for j in reversed(range(3)):  # DO j=3,1,-1
                        resultdirb[j] = resultdirb[j] + 2 * resultDir[i, j] * xb[i]

                    xb[i] = 0.0
                    # CALL CROSS_B(axis, axisb, dirvec(i, :), dirvecb(i, :), resultdirb)
                    axisb, dirvecb[i, :] = geo_utils.cross_b(self.axis[0, :], dirVec[i, :], resultdirb)

                # coordsb = 0.0
                # originb = 0.0
                for i in reversed(range(len(coordsb))):  # DO i=n,1,-1
                    for j in reversed(range(3)):  # DO j=3,1,-1
                        originb[j] = originb[j] + dirvecb[i, j]
                        coordsb[i, j] = coordsb[i, j] - dirvecb[i, j]
                        dirvecb[i, j] = 0.0

            tmpPt = self.DVGeo.totalSensitivity(dCdPt, self.name + "coords", config=config)
            tmpOrigin = self.DVGeo.totalSensitivity(dCdOrigin, self.name + "origin", config=config)

            tmpTotal = {}
            for key in tmpPt:
                tmpTotal[key] = tmpPt[key] + tmpOrigin[key]

            tmpTotal[self.name + "axis"] = dCdAxis

            funcsSens[self.name] = tmpTotal

    def addVariablesPyOpt(self, optProb):
        """
        Add the axis variable for the colinearity constraint to pyOpt
        """

        if self.addToPyOpt:
            optProb.addVarGroup(
                self.name, self.nVal, "c", value=self.value, lower=self.lower, upper=self.upper, scale=self.scale
            )

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """
        handle.write("Zone T=%s_coords\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords) + 1, len(self.coords)))
        handle.write("DATAPACKING=POINT\n")
        handle.write(f"{self.origin[0, 0]:f} {self.origin[0, 1]:f} {self.origin[0, 2]:f}\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords)):
            handle.write("%d %d\n" % (i + 1, i + 2))
