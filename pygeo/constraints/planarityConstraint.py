# External modules
import numpy as np

# Local modules
from .. import geo_utils
from .baseConstraint import GeometricConstraint


class PlanarityConstraint(GeometricConstraint):
    """
    DVConstraints representation of a surface planarity constraint.
    Constrain that all of the points on this surface are co-planar.
    One of these objects is created each time an
    addPlanarityConstraint call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, axis, origin, p0, v1, v2, lower, upper, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, 1, lower, upper, scale, DVGeo, addToPyOpt)

        # create the output array
        self.X = np.zeros(self.nCon)
        self.n = len(p0)

        # The first thing we do is convert v1 and v2 to coords
        self.axis = axis
        self.p0 = p0
        self.p1 = v1 + p0
        self.p2 = v2 + p0
        self.origin = origin

        # Now embed the coordinates and origin into DVGeo
        # with the name provided:
        # TODO this is duplicating a DVGeo pointset (same as the surface which originally created the constraint)
        # issue 53
        self.DVGeo.addPointSet(self.p0, self.name + "p0", compNames=compNames)
        self.DVGeo.addPointSet(self.p1, self.name + "p1", compNames=compNames)
        self.DVGeo.addPointSet(self.p2, self.name + "p2", compNames=compNames)
        self.DVGeo.addPointSet(self.origin, self.name + "origin", compNames=compNames)

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.p0 = self.DVGeo.update(self.name + "p0", config=config)
        self.p1 = self.DVGeo.update(self.name + "p1", config=config)
        self.p2 = self.DVGeo.update(self.name + "p2", config=config)
        self.origin = self.DVGeo.update(self.name + "origin", config=config)

        allPoints = np.vstack([self.p0, self.p1, self.p2])

        # Compute the distance from the origin to each point
        dist = allPoints - self.origin

        # project it onto the axis
        self.X[0] = 0
        for i in range(self.n * 3):
            self.X[0] += np.dot(self.axis, dist[i, :]) ** 2
        self.X[0] = np.sqrt(self.X[0])
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
            dPdp0 = np.zeros((self.nCon, self.p0.shape[0], self.p0.shape[1]))
            dPdp1 = np.zeros((self.nCon, self.p1.shape[0], self.p1.shape[1]))

            dPdp2 = np.zeros((self.nCon, self.p2.shape[0], self.p2.shape[1]))

            dPdO = np.zeros((self.nCon, self.origin.shape[0], self.origin.shape[1]))

        # copy data into all points array
        # allpoints(1:n) = p0
        # allpoints(n:2*n) = p1
        # allpoints(2*n:3*n) = p2
        allPoints = np.vstack([self.p0, self.p1, self.p2])

        # Compute the distance from the origin to each point
        # for i in range(n*3):#DO i=1,n*3
        #     for j in range(3):#DO j=1,3
        #         dist(i, j) = allpoints(i, j) - origin(j)
        dist = allPoints - self.origin

        scalardist = np.zeros(self.n * 3)
        tmpX = 0
        for i in range(self.n * 3):
            scalardist[i] = np.dot(self.axis, dist[i, :])
            tmpX += scalardist[i] ** 2

        xb = np.zeros(self.nCon)
        axisb = np.zeros(3)

        scalardistb = np.zeros(self.n * 3)
        allpointsb = np.zeros((self.n * 3, 3))
        distb = np.zeros((self.n * 3, 3))
        for con in range(self.nCon):
            p0b = dPdp0[con, :, :]
            p1b = dPdp1[con, :, :]
            p2b = dPdp2[con, :, :]
            originb = dPdO[con, 0, :]
            axisb[:] = 0.0
            originb[:] = 0.0
            scalardistb[:] = 0.0
            allpointsb[:, :] = 0.0
            distb[:, :] = 0.0
            xb[:] = 0
            xb[con] = 1.0
            if self.X[0] == 0.0:
                xb[con] = 0.0
            else:
                xb[con] = xb[con] / (2.0 * np.sqrt(tmpX))

            for i in reversed(range(self.n * 3)):  # DO i=3*n,1,-1
                scalardistb[i] = scalardistb[i] + 2.0 * scalardist[i] * xb[con]  # /(self.n*3)
                # CALL DOT_B(axis, axisb, dist(i, :), distb(i, :), scalardist(i), &
                #            &        scalardistb(i))
                axisb, distb[i, :] = geo_utils.dot_b(self.axis, dist[i, :], scalardistb[i])
                scalardistb[i] = 0.0
                for j in reversed(range(3)):  # DO j=3,1,-1
                    allpointsb[i, j] = allpointsb[i, j] + distb[i, j]
                    originb[j] = originb[j] - distb[i, j]
                    distb[i, j] = 0.0

            p2b[:, :] = 0.0
            p2b[:, :] = allpointsb[2 * self.n : 3 * self.n]
            allpointsb[2 * self.n : 3 * self.n] = 0.0
            p1b[:, :] = 0.0
            p1b[:, :] = allpointsb[self.n : 2 * self.n]
            allpointsb[self.n : 2 * self.n] = 0.0
            p0b[:, :] = 0.0
            p0b[:, :] = allpointsb[0 : self.n]

            # map back to DVGeo
            tmpp0 = self.DVGeo.totalSensitivity(dPdp0, self.name + "p0", config=config)
            tmpp1 = self.DVGeo.totalSensitivity(dPdp1, self.name + "p1", config=config)
            tmpp2 = self.DVGeo.totalSensitivity(dPdp2, self.name + "p2", config=config)
            tmpO = self.DVGeo.totalSensitivity(dPdO, self.name + "origin", config=config)

            tmpTotal = {}
            for key in tmpp0:
                tmpTotal[key] = tmpp0[key] + tmpp1[key] + tmpp2[key] + tmpO[key]

            funcsSens[self.name] = tmpTotal

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s_surface\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FETRIANGLE\n" % (3 * self.n, self.n))
        handle.write("DATAPACKING=POINT\n")
        for i in range(self.n):
            handle.write(f"{self.p0[i, 0]:f} {self.p0[i, 1]:f} {self.p0[i, 2]:f}\n")
        for i in range(self.n):
            handle.write(f"{self.p1[i, 0]:f} {self.p1[i, 1]:f} {self.p1[i, 2]:f}\n")

        for i in range(self.n):
            handle.write(f"{self.p2[i, 0]:f} {self.p2[i, 1]:f} {self.p2[i, 2]:f}\n")

        for i in range(self.n):
            handle.write("%d %d %d\n" % (i + 1, i + self.n + 1, i + self.n * 2 + 1))

        handle.write("Zone T=%s_center\n" % self.name)
        handle.write("Nodes = 2, Elements = 1 ZONETYPE=FELINESEG\n")
        handle.write("DATAPACKING=POINT\n")
        handle.write(f"{self.origin[0, 0]:f} {self.origin[0, 1]:f} {self.origin[0, 2]:f}\n")
        handle.write(f"{self.origin[0, 0]:f} {self.origin[0, 1]:f} {self.origin[0, 2]:f}\n")
        handle.write("%d %d\n" % (1, 2))
