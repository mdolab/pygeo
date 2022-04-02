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

class DVGeometryCSTUnitTest(unittest.TestCase):

    N_PROCS = 1

    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.sensTol = 1e-10
        self.coordTol = 1e-10
        self.maxNumCoeff = 10
        self.x = np.linspace(0,1,100)
        self.yte = 1e-3

    def test_ClassShape(self):
        """Test that for w_i = 1, the class shape has the expected shape
        """
        N1 = 0.5
        N2 = 1.0
        yExact = np.sqrt(self.x)*(1-self.x)
        for n in range(1, self.maxNumCoeff+1):
            w = np.ones(n)
            y = DVGeometryCST.computeClassShape(self.x, N1, N2)
            np.testing.assert_allclose(y, yExact, atol=self.coordTol, rtol=self.coordTol)

    def test_ShapeFunctions(self):
        """Test that the shape functions sum to 1 when all weights are 1"""
        for n in range(1, self.maxNumCoeff+1):
            w = np.ones(n)
            y = DVGeometryCST.computeShapeFunctions(self.x, w)
            np.testing.assert_allclose(y.sum(axis=0), 1.0, atol=self.coordTol, rtol=self.coordTol)

    def test_dYdN1(self):
        """Test the derivatives of the CST curve height w.r.t N1"""
        N1 = self.rng.random(1)
        N2 = self.rng.random(1)
        for n in range(1, self.maxNumCoeff+1):
            w = self.rng.random(n)
            y0 = DVGeometryCST.computeCSTCoordinates(self.x, N1, N2, w, self.yte)
            dydN1 = DVGeometryCST.computeCSTdydN1(self.x, N1, N2, w, self.yte)
            dydN1_CS = np.imag(DVGeometryCST.computeCSTdydN1(self.x, N1+1e-200*1j, N2, w, self.yte))*1e200
            np.testing.assert_allclose(dydN1, dydN1_CS, atol=self.sensTol, rtol=self.sensTol)


class DVGeometryCSTPointSetSerial(unittest.TestCase):
    # Test in serial
    N_PROCS = 1

    def setUp(self):
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.comm = MPI.COMM_WORLD
        self.DVGeo = DVGeometryCST(comm=self.comm)

    def test_addPointSet_sorted(self):
        # Read in NACA 2412 coordinates to test with and split up the surfaces
        coords = readCoordFile(os.path.join(self.curDir, "naca2412.dat"))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(0, idxLE)  # TODO: should the LE be in both upper and lower? I think probably not
        idxLower = np.arange(idxLE, coords.shape[0])
        idxTE = np.array([0, coords.shape[0] - 1])

        self.DVGeo.addPointSet(coords, "test")

        np.testing.assert_equal(idxUpper, self.DVGeo.points["test"]["upper"])
        np.testing.assert_equal(idxLower, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(idxTE, self.DVGeo.points["test"]["trailingEdge"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])

    def test_addPointSet_randomized(self):
        # Read in NACA 2412 coordinates to test with and split up the surfaces
        coords = readCoordFile(os.path.join(self.curDir, "naca2412.dat"))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(0, idxLE)  # TODO: should the LE be in both upper and lower? I think probably not
        idxLower = np.arange(idxLE, coords.shape[0])
        idxTE = np.array([0, coords.shape[0] - 1])

        # Randomize the index order (do indices so we can track where they end up)
        rng = np.random.default_rng(1)
        idx = np.arange(0, coords.shape[0])  # maps from the original index to the new one
        rng.shuffle(idx)
        coordsRand = np.zeros(coords.shape)
        coordsRand[idx, :] = coords
        idxUpperRand = np.sort(idx[idxUpper])
        idxLowerRand = np.sort(idx[idxLower])
        idxTERand = np.sort(idx[idxTE])

        self.DVGeo.addPointSet(coordsRand, "test")

        np.testing.assert_equal(idxUpperRand, self.DVGeo.points["test"]["upper"])
        np.testing.assert_equal(idxLowerRand, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(idxTERand, self.DVGeo.points["test"]["trailingEdge"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])

    def test_addPointSet_bluntTE(self):  # includes a blunt trailing edge with points along it
        # Read in NACA 2412 coordinates to test with and split up the surfaces
        coords = readCoordFile(os.path.join(self.curDir, "naca2412.dat"))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        nPointsTE = 6  # total points on the trailing edge
        pointsTE = np.ones((nPointsTE - 2, 3), dtype=float)
        pointsTE[:, 2] = 0  # z coordinates are zero
        pointsTE[:, 1] = np.linspace(coords[-1, 1] + 1e-4, coords[0, 1] - 1e-4, pointsTE.shape[0])
        coords = np.vstack((coords, pointsTE))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(0, idxLE)
        idxLower = np.arange(idxLE, coords.shape[0] - nPointsTE + 2)
        idxTE = np.zeros(nPointsTE)
        idxTE[1:] = np.arange(coords.shape[0] - nPointsTE + 1, coords.shape[0])

        self.DVGeo.addPointSet(coords, "test")

        plt.figure()
        plt.scatter(coords[self.DVGeo.points["test"]["upper"], 0], coords[self.DVGeo.points["test"]["upper"], 1], marker='^')
        plt.scatter(coords[self.DVGeo.points["test"]["lower"], 0], coords[self.DVGeo.points["test"]["lower"], 1], marker='v')
        plt.scatter(coords[self.DVGeo.points["test"]["trailingEdge"], 0], coords[self.DVGeo.points["test"]["trailingEdge"], 1], marker='>')
        plt.show()

        np.testing.assert_equal(idxUpper, self.DVGeo.points["test"]["upper"])
        np.testing.assert_equal(idxLower, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(idxTE, self.DVGeo.points["test"]["trailingEdge"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])


class DVGeometryCSTPointSetParallel(unittest.TestCase):
    # Test in parallel
    N_PROCS = 4

    def setUp(self):
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.comm = MPI.COMM_WORLD
        self.DVGeo = DVGeometryCST(comm=self.comm)

    def test_addPointSet_sorted(self):
        # Read in NACA 2412 coordinates to test with and split up the surfaces
        coords = readCoordFile(os.path.join(self.curDir, "naca2412.dat"))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(0, idxLE)
        idxLower = np.arange(idxLE, coords.shape[0])
        idxTE = np.array([0, coords.shape[0] - 1])

        # Divide up the points among the procs (mostly evenly, but not quite to check the harder case)
        nPerProc = int(coords.shape[0] // 3.5)
        rank = self.comm.rank
        if self.comm.rank < self.comm.size - 1:  # all but last proc takes nPerProc elements
            self.DVGeo.addPointSet(coords[rank * nPerProc : (rank + 1) * nPerProc, :], "test")
        else:
            self.DVGeo.addPointSet(coords[rank * nPerProc:, :], "test")

        np.testing.assert_equal(idxUpper, self.DVGeo.points["test"]["upper"])
        np.testing.assert_equal(idxLower, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(idxTE, self.DVGeo.points["test"]["trailingEdge"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])

    def test_addPointSet_randomized(self):
        # Read in NACA 2412 coordinates to test with and split up the surfaces
        coords = readCoordFile(os.path.join(self.curDir, "naca2412.dat"))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(0, idxLE)
        idxLower = np.arange(idxLE, coords.shape[0])
        idxTE = np.array([0, coords.shape[0] - 1])

        # Randomize the index order (do indices so we can track where they end up)
        rng = np.random.default_rng(1)
        idx = np.arange(0, coords.shape[0])  # maps from the original index to the new one
        rng.shuffle(idx)
        coordsRand = np.zeros(coords.shape)
        coordsRand[idx, :] = coords
        idxUpperRand = np.sort(idx[idxUpper])
        idxLowerRand = np.sort(idx[idxLower])
        idxTERand = np.sort(idx[idxTE])

        # Divide up the points among the procs (mostly evenly, but not quite to check the harder case)
        nPerProc = int(coordsRand.shape[0] // 3.5)
        rank = self.comm.rank
        if self.comm.rank < self.comm.size - 1:  # all but last proc takes nPerProc elements
            self.DVGeo.addPointSet(coordsRand[rank * nPerProc : (rank + 1) * nPerProc, :], "test")
        else:
            self.DVGeo.addPointSet(coordsRand[rank * nPerProc:, :], "test")

        np.testing.assert_equal(idxUpperRand, self.DVGeo.points["test"]["upper"])
        np.testing.assert_equal(idxLowerRand, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(idxTERand, self.DVGeo.points["test"]["trailingEdge"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])

if __name__ == '__main__':
    unittest.main()