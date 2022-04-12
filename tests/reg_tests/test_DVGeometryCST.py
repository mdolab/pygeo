"""
==============================================================================
DVGeoCST: Test suite for the DVGeoCST module.
==============================================================================
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest
import os
from parameterized import parameterized_class

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from prefoil.preFoil import readCoordFile
from mpi4py import MPI

# ==============================================================================
# Extension modules
# ==============================================================================
from pygeo import DVGeometryCST

# LEUpper is true if the leading edge point is considered to be on the upper surface
airfoils = [{"fName": "naca2412.dat", "LEUpper": False}, {"fName": "naca0012.dat", "LEUpper": True}]


class DVGeometryCSTUnitTest(unittest.TestCase):

    N_PROCS = 1

    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.sensTol = 1e-10
        self.coordTol = 1e-10
        self.maxNumCoeff = 10
        self.x = np.linspace(0, 1, 100)
        self.yte = 1e-3
        self.CS_delta = 1e-200

    def test_ClassShape(self):
        """Test that for w_i = 1, the class shape has the expected shape"""
        N1 = 0.5
        N2 = 1.0
        yExact = np.sqrt(self.x) * (1 - self.x)
        for n in range(1, self.maxNumCoeff + 1):
            w = np.ones(n)
            y = DVGeometryCST.computeClassShape(self.x, N1, N2)
            np.testing.assert_allclose(y, yExact, atol=self.coordTol, rtol=self.coordTol)

    def test_ShapeFunctions(self):
        """Test that the shape functions sum to 1 when all weights are 1"""
        for n in range(1, self.maxNumCoeff + 1):
            w = np.ones(n)
            y = DVGeometryCST.computeShapeFunctions(self.x, w)
            np.testing.assert_allclose(y.sum(axis=0), 1.0, atol=self.coordTol, rtol=self.coordTol)

    def test_dydN1(self):
        """Test the derivatives of the CST curve height w.r.t N1"""
        N1 = self.rng.random(1)
        N2 = self.rng.random(1)
        for n in range(1, self.maxNumCoeff + 1):
            w = self.rng.random(n)
            y0 = DVGeometryCST.computeCSTCoordinates(self.x, N1, N2, w, self.yte)
            dydN1 = DVGeometryCST.computeCSTdydN1(self.x, N1, N2, w, self.yte)
            dydN1_CS = (
                np.imag(DVGeometryCST.computeCSTCoordinates(self.x, N1 + self.CS_delta * 1j, N2, w, self.yte))
                / self.CS_delta
            )
            np.testing.assert_allclose(dydN1, dydN1_CS, atol=self.sensTol, rtol=self.sensTol)

    def test_dydN2(self):
        """Test the derivatives of the CST curve height w.r.t N2"""
        N1 = self.rng.random(1)
        N2 = self.rng.random(1)
        for n in range(1, self.maxNumCoeff + 1):
            w = self.rng.random(n)
            y0 = DVGeometryCST.computeCSTCoordinates(self.x, N1, N2, w, self.yte)
            dydN2 = DVGeometryCST.computeCSTdydN2(self.x, N1, N2, w, self.yte)
            dydN2_CS = (
                np.imag(DVGeometryCST.computeCSTCoordinates(self.x, N1, N2 + self.CS_delta * 1j, w, self.yte))
                / self.CS_delta
            )
            np.testing.assert_allclose(dydN2, dydN2_CS, atol=self.sensTol, rtol=self.sensTol)

    def test_dydw(self):
        """Test the derivatives of the CST curve height w.r.t N2"""
        N1 = self.rng.random(1)
        N2 = self.rng.random(1)
        for n in range(1, self.maxNumCoeff + 1):
            w = self.rng.random(n)
            y0 = DVGeometryCST.computeCSTCoordinates(self.x, N1, N2, w, self.yte)
            dydw = DVGeometryCST.computeCSTdydw(self.x, N1, N2, w)
            dydw_CS = np.zeros((n, self.x.size), dtype=float)
            w = w.astype(complex)
            for i in range(n):
                w[i] += self.CS_delta * 1j
                dydw_CS[i, :] = (
                    np.imag(DVGeometryCST.computeCSTCoordinates(self.x, N1, N2, w, self.yte)) / self.CS_delta
                )
                w[i] -= self.CS_delta * 1j

            np.testing.assert_allclose(dydw, dydw_CS, atol=self.sensTol, rtol=self.sensTol)


@parameterized_class(airfoils)
class DVGeometryCSTPointSetSerial(unittest.TestCase):
    # Test in serial
    N_PROCS = 1

    def setUp(self):
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.comm = MPI.COMM_WORLD
        self.DVGeo = DVGeometryCST(comm=self.comm)

    def test_addPointSet_sorted(self):
        # Read in NACA 2412 coordinates to test with and split up the surfaces
        coords = readCoordFile(os.path.join(self.curDir, self.fName))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(
            0, idxLE + self.LEUpper
        )  # TODO: should the LE be in both upper and lower? I think probably not
        idxLower = np.arange(idxLE + self.LEUpper, coords.shape[0])
        thickTE = coords[0, 1] - coords[-1, 1]

        self.DVGeo.addPointSet(coords, "test")

        np.testing.assert_equal(idxUpper, self.DVGeo.points["test"]["upper"])
        np.testing.assert_equal(idxLower, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(thickTE, self.DVGeo.points["test"]["thicknessTE"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])

    def test_addPointSet_randomized(self):
        # Read in NACA 2412 coordinates to test with and split up the surfaces
        coords = readCoordFile(os.path.join(self.curDir, self.fName))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(0, idxLE + self.LEUpper)
        idxLower = np.arange(idxLE + self.LEUpper, coords.shape[0])
        thickTE = coords[0, 1] - coords[-1, 1]

        # Randomize the index order (do indices so we can track where they end up)
        rng = np.random.default_rng(1)
        idx = np.arange(0, coords.shape[0])  # maps from the original index to the new one
        rng.shuffle(idx)
        coordsRand = np.zeros(coords.shape)
        coordsRand[idx, :] = coords
        idxUpperRand = np.sort(idx[idxUpper])
        idxLowerRand = np.sort(idx[idxLower])

        self.DVGeo.addPointSet(coordsRand, "test")

        np.testing.assert_equal(idxUpperRand, self.DVGeo.points["test"]["upper"])
        np.testing.assert_equal(idxLowerRand, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(thickTE, self.DVGeo.points["test"]["thicknessTE"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])

    def test_addPointSet_bluntTE(self):  # includes a blunt trailing edge with points along it
        # Read in NACA 2412 coordinates to test with and split up the surfaces
        coords = readCoordFile(os.path.join(self.curDir, self.fName))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        nPointsTE = 6  # total points on the trailing edge
        pointsTE = np.ones((nPointsTE - 2, 3), dtype=float)
        pointsTE[:, 2] = 0  # z coordinates are zero
        pointsTE[:, 1] = np.linspace(coords[-1, 1] + 1e-4, coords[0, 1] - 1e-4, pointsTE.shape[0])
        coords = np.vstack((coords, pointsTE))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(0, idxLE + self.LEUpper)
        idxLower = np.arange(idxLE + self.LEUpper, coords.shape[0] - nPointsTE + 2)
        thickTE = coords[0, 1] - coords[coords.shape[0] - nPointsTE + 1, 1]

        self.DVGeo.addPointSet(coords, "test")

        np.testing.assert_equal(idxUpper, self.DVGeo.points["test"]["upper"])
        np.testing.assert_equal(idxLower, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(thickTE, self.DVGeo.points["test"]["thicknessTE"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])


@parameterized_class(airfoils)
class DVGeometryCSTPointSetParallel(unittest.TestCase):
    # Test in parallel
    N_PROCS = 4

    def setUp(self):
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.comm = MPI.COMM_WORLD
        self.DVGeo = DVGeometryCST(comm=self.comm)

    def test_addPointSet_sorted(self):
        # Read in NACA 2412 coordinates to test with and split up the surfaces
        coords = readCoordFile(os.path.join(self.curDir, self.fName))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])

        isUpper = np.zeros(coords.shape[0])
        isUpper[: idxLE + self.LEUpper] = 1
        isUpper = isUpper == 1
        isLower = np.logical_not(isUpper)

        thickTE = coords[0, 1] - coords[-1, 1]

        # Divide up the points among the procs (mostly evenly, but not quite to check the harder case)
        nPerProc = int(coords.shape[0] // 3.5)
        rank = self.comm.rank
        if self.comm.rank < self.comm.size - 1:  # all but last proc takes nPerProc elements
            self.DVGeo.addPointSet(coords[rank * nPerProc : (rank + 1) * nPerProc, :], "test")
            idxUpper = np.where(isUpper[rank * nPerProc : (rank + 1) * nPerProc])[0]
            idxLower = np.where(isLower[rank * nPerProc : (rank + 1) * nPerProc])[0]
        else:
            self.DVGeo.addPointSet(coords[rank * nPerProc :, :], "test")
            idxUpper = np.where(isUpper[rank * nPerProc :])[0]
            idxLower = np.where(isLower[rank * nPerProc :])[0]

        np.testing.assert_equal(idxUpper, self.DVGeo.points["test"]["upper"])
        np.testing.assert_equal(idxLower, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(thickTE, self.DVGeo.points["test"]["thicknessTE"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])

    def test_addPointSet_randomized(self):
        # Read in NACA 2412 coordinates to test with and split up the surfaces
        coords = readCoordFile(os.path.join(self.curDir, self.fName))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])

        isUpper = np.zeros(coords.shape[0])
        isUpper[: idxLE + self.LEUpper] = 1
        isUpper = isUpper == 1
        isLower = np.logical_not(isUpper)

        thickTE = coords[0, 1] - coords[-1, 1]

        # Randomize the index order (do indices so we can track where they end up)
        rng = np.random.default_rng(1)
        idx = np.arange(0, coords.shape[0])  # maps from the original index to the new one
        rng.shuffle(idx)
        coordsRand = np.zeros(coords.shape)
        isUpperRand = np.full(isUpper.shape, False)
        isLowerRand = np.full(isLower.shape, False)
        coordsRand[idx, :] = coords
        isUpperRand[idx] = isUpper
        isLowerRand[idx] = isLower

        # Divide up the points among the procs (mostly evenly, but not quite to check the harder case)
        nPerProc = int(coordsRand.shape[0] // 3.5)
        rank = self.comm.rank
        if self.comm.rank < self.comm.size - 1:  # all but last proc takes nPerProc elements
            self.DVGeo.addPointSet(coordsRand[rank * nPerProc : (rank + 1) * nPerProc, :], "test")
            idxUpper = np.where(isUpperRand[rank * nPerProc : (rank + 1) * nPerProc])[0]
            idxLower = np.where(isLowerRand[rank * nPerProc : (rank + 1) * nPerProc])[0]
        else:
            self.DVGeo.addPointSet(coordsRand[rank * nPerProc :, :], "test")
            idxUpper = np.where(isUpperRand[rank * nPerProc :])[0]
            idxLower = np.where(isLowerRand[rank * nPerProc :])[0]

        np.testing.assert_equal(idxUpper, self.DVGeo.points["test"]["upper"])
        np.testing.assert_equal(idxLower, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(thickTE, self.DVGeo.points["test"]["thicknessTE"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])


if __name__ == "__main__":
    unittest.main()
