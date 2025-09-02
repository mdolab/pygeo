# External modules
import numpy as np

# Local modules
from .. import geo_utils
from .baseConstraint import GeometricConstraint


class RadiusConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of radius of curvature
    constraints. One of these objects is created each time a
    addLERadiusConstraints call is made. The user should not have
    to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords) // 3, lower, upper, scale, DVGeo, addToPyOpt)

        self.coords = coords
        self.scaled = scaled

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths
        self.r0, self.c0 = self.computeCircle(self.coords)

    def splitPointSets(self, coords):
        p1 = coords[: self.nCon]
        p2 = coords[self.nCon : self.nCon * 2]
        p3 = coords[self.nCon * 2 :]
        return p1, p2, p3

    def computeReferenceFrames(self, coords):
        p1, p2, p3 = self.splitPointSets(coords)

        # Compute origin and unit vectors (xi, eta) of 2d space
        origin = (p1 + p2) / 2.0
        nxi = p1 - origin
        neta = p3 - origin
        for i in range(self.nCon):
            nxi[i] /= geo_utils.euclideanNorm(nxi[i])
            neta[i] /= geo_utils.euclideanNorm(neta[i])

        # Compute component of eta in the xi direction
        eta_on_xi = np.einsum("ij,ij->i", nxi, neta)
        xi_of_eta = np.einsum("ij,i->ij", nxi, eta_on_xi)

        # Remove component of eta in the xi direction
        neta = neta - xi_of_eta
        for i in range(self.nCon):
            neta[i] /= geo_utils.euclideanNorm(neta[i])

        return origin, nxi, neta

    def computeCircle(self, coords):
        """
        A circle in a 2D coordinate system is defined by the equation:

            A*xi**2 + A*eta**2 + B*xi + C*eta + D = 0

        First, we get the coordinates of our three points in 2D reference space.
        Then, we can get the coefficients A, B, C, and D by solving for some
        determinants. Then the radius and center of the circle can be calculated
        from:

            x = -B / 2 / A
            y = -C / 2 / A
            r = sqrt((B**2 + C**2 - 4*A*D) / 4 / A**2)

        Finally, we convert the reference coordinates of the center back into
        3D space.
        """
        p1, p2, p3 = self.splitPointSets(coords)

        # Compute origin and unit vectors (xi, eta) of 2d space
        origin, nxi, neta = self.computeReferenceFrames(coords)

        # Compute xi component of p1, p2, and p3
        xi1 = np.einsum("ij,ij->i", p1 - origin, nxi)
        xi2 = np.einsum("ij,ij->i", p2 - origin, nxi)
        xi3 = np.einsum("ij,ij->i", p3 - origin, nxi)

        # Compute eta component of p1, p2, and p3
        eta1 = np.einsum("ij,ij->i", p1 - origin, neta)
        eta2 = np.einsum("ij,ij->i", p2 - origin, neta)
        eta3 = np.einsum("ij,ij->i", p3 - origin, neta)

        # Compute the radius of curvature
        A = xi1 * (eta2 - eta3) - eta1 * (xi2 - xi3) + xi2 * eta3 - xi3 * eta2
        B = (xi1**2 + eta1**2) * (eta3 - eta2) + (xi2**2 + eta2**2) * (eta1 - eta3) + (xi3**2 + eta3**2) * (eta2 - eta1)
        C = (xi1**2 + eta1**2) * (xi2 - xi3) + (xi2**2 + eta2**2) * (xi3 - xi1) + (xi3**2 + eta3**2) * (xi1 - xi2)
        D = (
            (xi1**2 + eta1**2) * (xi3 * eta2 - xi2 * eta3)
            + (xi2**2 + eta2**2) * (xi1 * eta3 - xi3 * eta1)
            + (xi3**2 + eta3**2) * (xi2 * eta1 - xi1 * eta2)
        )

        xiC = -B / 2 / A
        etaC = -C / 2 / A
        r = np.sqrt((B**2 + C**2 - 4 * A * D) / 4 / A**2)

        # Convert center coordinates back
        center = origin + nxi * xiC[:, None] + neta * etaC[:, None]

        return r, center

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
        r, _ = self.computeCircle(self.coords)
        if self.scaled:
            r /= self.r0
        funcs[self.name] = r

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
            # This is the sensitivity of the radius of curvature w.r.t. the
            # coordinates of each of the three points that make it up
            # row 0: dr0dp0x dr0dp0y dr0dp0z dr0dp1x dr0dp1y dr0dp1z dr0dp2x ...
            # row 1: dr1dp0x dr1dp0y dr1dp0z dr1dp1x dr1dp1y dr1dp1z dr1dp2x ...
            # :
            drdPt = np.zeros((self.nCon, 9))

            coords = self.coords.astype("D")
            for i in range(3):  # loop over pts at given slice
                for j in range(3):  # loop over coordinates in pt
                    coords[i * self.nCon : (i + 1) * self.nCon, j] += 1e-40j
                    r, _ = self.computeCircle(coords)

                    drdPt[:, i * 3 + j] = r.imag / 1e-40
                    coords[i * self.nCon : (i + 1) * self.nCon, j] -= 1e-40j

            # We now need to convert to the 3d sparse matrix form of the jacobian.
            # We need the derivative of each radius w.r.t. all of the points
            # in coords w.r.t. all of the coordinates for a given point.
            # So the final matrix dimensions are (ncon, ncon*3, 3)

            # We also have to scale the sensitivities if scale is True.
            if self.scaled:
                eye = np.diag(1 / self.r0)
            else:
                eye = np.eye(self.nCon)
            drdPt_sparse = np.einsum("ij,jk->ijk", eye, drdPt)
            drdPt_sparse = drdPt_sparse.reshape(self.nCon, self.nCon * 3, 3)
            drdPt_sparse = np.hstack([drdPt_sparse[:, ::3, :], drdPt_sparse[:, 1::3, :], drdPt_sparse[:, 2::3, :]])

            funcsSens[self.name] = self.DVGeo.totalSensitivity(drdPt_sparse, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """
        r, c = self.computeCircle(self.coords)

        # Compute origin and unit vectors (xi, eta) of 2d space
        _, nxi, neta = self.computeReferenceFrames(self.coords)

        nres = 50
        theta = np.linspace(0, 2 * np.pi, nres + 1)[:-1]
        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (self.nCon * nres, self.nCon * nres))
        handle.write("DATAPACKING=POINT\n")
        for i in range(self.nCon):
            cos_part = np.outer(np.cos(theta), nxi * r[i])
            sin_part = np.outer(np.sin(theta), neta * r[i])
            x = c[i, 0] + cos_part[:, 0] + sin_part[:, 0]
            y = c[i, 1] + cos_part[:, 1] + sin_part[:, 1]
            z = c[i, 2] + cos_part[:, 2] + sin_part[:, 2]

            for j in range(nres):
                handle.write(f"{x[j]:f} {y[j]:f} {z[j]:f}\n")

        for i in range(self.nCon):
            for j in range(nres):
                handle.write("%d %d\n" % (i * nres + j + 1, i * nres + (j + 1) % nres + 1))
