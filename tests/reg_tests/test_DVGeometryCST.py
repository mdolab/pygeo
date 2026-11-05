"""
==============================================================================
DVGeometryCST: Test suite for the DVGeometryCST module.
==============================================================================
"""

# Standard Python modules
from math import comb
import os
import tempfile
import unittest
import warnings

# External modules
from baseclasses import BaseRegTest
from mpi4py import MPI
import numpy as np
from parameterized import parameterized_class
import psutil

try:
    # External modules
    from prefoil.utils import readCoordFile

    prefoilImported = True
    # First party modules
    from pygeo import DVGeometryCST
except ImportError:
    prefoilImported = False

try:
    # External modules
    from pyoptsparse import Optimization

    pyoptsparseImported = True
except ImportError:
    pyoptsparseImported = False

baseDir = os.path.dirname(os.path.abspath(__file__))
inputDir = os.path.join(baseDir, "..", "..", "input_files")

# LEUpper is true if the leading edge (minimum x) point is considered to be on the upper surface
airfoils = [
    {"fName": "naca2412.dat", "LEUpper": False},
    {"fName": "naca0012.dat", "LEUpper": True},
    {"fName": "e63.dat", "LEUpper": False},
]

airfoils_cst_reg = [
    {"name": "naca2412"},
    {"name": "naca0012"},
    {"name": "naca0012_closed"},
    {"name": "naca0012_clockwise"},
    {"name": "naca0012_sharp"},
    {"name": "naca0012_zeroLE"},
    {"name": "e63"},
]

# Parameterization of design variables
DVs = [
    {"dvName": "upper", "dvNum": 5},
    {"dvName": "lower", "dvNum": 5},
    {"dvName": "n1", "dvNum": 1},
    {"dvName": "n2", "dvNum": 1},
    {"dvName": "n1_upper", "dvNum": 1},
    {"dvName": "n1_lower", "dvNum": 1},
    {"dvName": "n2_upper", "dvNum": 1},
    {"dvName": "n2_lower", "dvNum": 1},
    {"dvName": "chord", "dvNum": 1},
    {"dvName": "lem", "dvNum": 1},
    {"dvName": "lem_upper", "dvNum": 1},
    {"dvName": "lem_lower", "dvNum": 1},
    {"dvName": "te_thickness", "dvNum": 1},
]

# Design variables to test with points on the blunt trailing edge face and in parallel
DVsTE = [
    {"dvName": "upper", "dvNum": 4},
    {"dvName": "lower", "dvNum": 4},
    {"dvName": "chord", "dvNum": 1},
    {"dvName": "lem", "dvNum": 1},
    {"dvName": "lem_upper", "dvNum": 1},
    {"dvName": "lem_lower", "dvNum": 1},
    {"dvName": "te_thickness", "dvNum": 1},
]


def getCoordsWithTEFace(datFile, nPointsTE=6):
    """Read a blunt trailing edge dat file and add points along the trailing edge face (x = 1)"""
    coords = readCoordFile(datFile)
    coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
    pointsTE = np.ones((nPointsTE - 2, 3), dtype=float)
    pointsTE[:, 2] = 0.0
    pointsTE[:, 1] = np.linspace(coords[-1, 1] + 1e-4, coords[0, 1] - 1e-4, pointsTE.shape[0])
    return np.vstack((coords, pointsTE))


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
@parameterized_class(airfoils)
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
            dydN1 = DVGeometryCST.computeCSTdydN1(self.x, N1, N2, w)
            dydN1_CS = (
                np.imag(
                    DVGeometryCST.computeCSTCoordinates(self.x, N1 + self.CS_delta * 1j, N2, w, self.yte, dtype=complex)
                )
                / self.CS_delta
            )
            np.testing.assert_allclose(dydN1, dydN1_CS, atol=self.sensTol, rtol=self.sensTol)

    def test_dydN2(self):
        """Test the derivatives of the CST curve height w.r.t N2"""
        N1 = self.rng.random(1)
        N2 = self.rng.random(1)
        for n in range(1, self.maxNumCoeff + 1):
            w = self.rng.random(n)
            dydN2 = DVGeometryCST.computeCSTdydN2(self.x, N1, N2, w)
            dydN2_CS = (
                np.imag(
                    DVGeometryCST.computeCSTCoordinates(self.x, N1, N2 + self.CS_delta * 1j, w, self.yte, dtype=complex)
                )
                / self.CS_delta
            )
            np.testing.assert_allclose(dydN2, dydN2_CS, atol=self.sensTol, rtol=self.sensTol)

    def test_dydw(self):
        """Test the derivatives of the CST curve height w.r.t N2"""
        N1 = self.rng.random(1)
        N2 = self.rng.random(1)
        for n in range(1, self.maxNumCoeff + 1):
            w = self.rng.random(n)
            dydw = DVGeometryCST.computeCSTdydw(self.x, N1, N2, w)
            dydw_CS = np.zeros((n, self.x.size), dtype=float)
            w = w.astype(complex)
            for i in range(n):
                w[i] += self.CS_delta * 1j
                dydw_CS[i, :] = (
                    np.imag(DVGeometryCST.computeCSTCoordinates(self.x, N1, N2, w, self.yte, dtype=complex))
                    / self.CS_delta
                )
                w[i] -= self.CS_delta * 1j

            np.testing.assert_allclose(dydw, dydw_CS, atol=self.sensTol, rtol=self.sensTol)

    def test_fitCST(self):
        """Test the CST parameter fitting"""
        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(os.path.join(inputDir, self.fName))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(0, idxLE + self.LEUpper)
        idxLower = np.arange(idxLE + self.LEUpper, coords.shape[0])
        yUpperTE = coords[0, 1]
        yLowerTE = coords[-1, 1]
        N1 = 0.5
        N2 = 1.0

        # Normalize the x-coordinates
        xMin = np.min(coords[:, 0])
        xMax = np.max(coords[:, 0])
        chord = xMax - xMin
        xScaledUpper = (coords[idxUpper, 0] - xMin) / chord
        xScaledLower = (coords[idxLower, 0] - xMin) / chord

        for nCST in range(2, 10):
            # Fit the CST parameters and then compute the coordinates
            # with those parameters and check that it's close
            upperCST = DVGeometryCST.computeCSTfromCoords(coords[idxUpper, 0], coords[idxUpper, 1], nCST, N1=N1, N2=N2)
            lowerCST = DVGeometryCST.computeCSTfromCoords(coords[idxLower, 0], coords[idxLower, 1], nCST, N1=N1, N2=N2)
            fitCoordsUpper = (
                DVGeometryCST.computeCSTCoordinates(xScaledUpper, N1, N2, upperCST, yUpperTE / chord) * chord
            )
            fitCoordsLower = (
                DVGeometryCST.computeCSTCoordinates(xScaledLower, N1, N2, lowerCST, yLowerTE / chord) * chord
            )

            # Loosen the tolerances for the challenging e63 airfoil
            if self.fName == "e63.dat":
                if nCST < 4:
                    atol = 5e-3
                    rtol = 1e-1
                else:
                    atol = 2e-3
                    rtol = 1e-1
            else:
                atol = 1e-3
                rtol = 1e-1

            np.testing.assert_allclose(fitCoordsUpper, coords[idxUpper, 1], atol=atol, rtol=rtol)
            np.testing.assert_allclose(fitCoordsLower, coords[idxLower, 1], atol=atol, rtol=rtol)


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
class DVGeometryCSTLEMUnitTest(unittest.TestCase):
    """
    Tests for Kulfan's leading edge modification (LEM) term, x * (1 - x)^(N + 0.5), where N = n - 1 is the
    Bernstein polynomial order and n is the number of CST coefficients on the surface
    """

    N_PROCS = 1

    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.x = np.linspace(0, 1, 101)
        self.CS_delta = 1e-200
        self.tol = 1e-14

    def test_LEMShape_values(self):
        """The LEM shape matches Kulfan's x * (1 - x)^(n - 0.5) for n CST coefficients"""
        for n in range(1, 10):
            yExact = self.x * (1 - self.x) ** (n - 0.5)
            y = DVGeometryCST.computeLEMShape(self.x, n)
            np.testing.assert_allclose(y, yExact, atol=self.tol, rtol=self.tol)

    def test_LEMShape_hand_value(self):
        """Hand-checked value: x = 0.25, n = 2 gives 0.25 * 0.75^1.5"""
        y = DVGeometryCST.computeLEMShape(np.array([0.25]), 2)
        np.testing.assert_allclose(y, [0.16237976320958225], atol=self.tol, rtol=self.tol)

    def test_LEMShape_past_trailing_edge(self):
        """Points slightly past the trailing edge (x > 1) must give zero, not NaN"""
        x = np.array([1.0, 1.0 + 1e-12, 1.0 + 1e-6])
        y = DVGeometryCST.computeLEMShape(x, 8)
        np.testing.assert_array_equal(y, np.zeros(3))
        yComplex = DVGeometryCST.computeLEMShape(x.astype(complex), 8, dtype=complex)
        np.testing.assert_array_equal(yComplex, np.zeros(3))

    def test_LEMShape_complex_step(self):
        """Complex step through x agrees with the analytic derivative d/dx [x (1-x)^(n+1/2)]"""
        n = 8
        x = self.x[1:-1]
        dydxExact = (1 - x) ** (n - 0.5) - (n - 0.5) * x * (1 - x) ** (n - 1.5)
        dydxCS = np.imag(DVGeometryCST.computeLEMShape(x + self.CS_delta * 1j, n, dtype=complex)) / self.CS_delta
        np.testing.assert_allclose(dydxCS, dydxExact, atol=1e-12, rtol=1e-12)

    def test_CSTCoordinates_with_LEM(self):
        """computeCSTCoordinates adds wLE * x * (1 - x)^(n - 0.5), with n the number of CST weights"""
        N1 = 0.5
        N2 = 1.0
        yte = 1e-3
        for n in range(2, 10):
            w = self.rng.random(n) - 0.5
            wLE = 0.3
            yNoLEM = DVGeometryCST.computeCSTCoordinates(self.x, N1, N2, w, yte)
            yLEM = DVGeometryCST.computeCSTCoordinates(self.x, N1, N2, w, yte, wLE=wLE)
            np.testing.assert_allclose(
                yLEM - yNoLEM, wLE * self.x * (1 - self.x) ** (n - 0.5), atol=self.tol, rtol=self.tol
            )

    def test_CSTCoordinates_default_LEM_unchanged(self):
        """Without wLE the CST coordinates are bit-for-bit identical to C(x) S(x) + yte x"""
        N1 = 0.5
        N2 = 1.0
        yte = 1e-3
        w = self.rng.random(6)
        C = DVGeometryCST.computeClassShape(self.x, N1, N2)
        S = DVGeometryCST.computeShapeFunctions(self.x, w)
        np.testing.assert_array_equal(
            DVGeometryCST.computeCSTCoordinates(self.x, N1, N2, w, yte), C * S.sum(axis=0) + yte * self.x
        )


@parameterized_class(airfoils_cst_reg)
class DVGeometryCSTFitRegTest(unittest.TestCase):
    N_PROCS = 1

    def train_cst_fit(self):
        self.test_cst_fit(train=True)

    def test_cst_fit(self, train=False):
        datFile = os.path.join(inputDir, f"{self.name}.dat")
        DVGeo = DVGeometryCST(datFile, numCST=8)

        upperCST = DVGeo.defaultDV["upper"]
        lowerCST = DVGeo.defaultDV["lower"]

        if self.name in ["naca0012_closed", "naca0012_clockwise"]:
            # The closed and clockwise versions of naca0012.dat should have the same CST coefficients
            # as the original, so use the same ref file
            refName = "naca0012"
        else:
            refName = self.name

        refFile = os.path.join(baseDir, "ref", f"test_DVGeometryCST_{refName}.ref")
        tol = 1e-12
        with BaseRegTest(refFile, train=train) as handler:
            # Regression test the upper surface CST coefficients
            handler.root_add_val("upperCST", upperCST, tol=tol)
            if "naca0012" in self.name:
                # Test that the coefficients are symmetric for symmetric airfoils
                np.testing.assert_allclose(upperCST, -lowerCST, rtol=tol)
            else:
                # Regression test the lower surface CST coefficients for asymmetric airfoils
                handler.root_add_val("lowerCST", lowerCST, tol=tol)


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
@parameterized_class(airfoils)
class DVGeometryCSTPointSetSerial(unittest.TestCase):
    # Test in serial
    N_PROCS = 1

    def setUp(self):
        self.datFile = os.path.join(inputDir, self.fName)
        self.comm = MPI.COMM_WORLD
        self.DVGeo = DVGeometryCST(self.datFile, comm=self.comm)

    def test_addPointSet_sorted(self):
        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(self.datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])
        # Don't include the points at the corners of the trailing edge because it's not guaranteed
        # that they'll be included in the upper and lower surface (which is ok)
        idxUpper = np.arange(1, idxLE + self.LEUpper)
        idxLower = np.arange(idxLE + self.LEUpper, coords.shape[0] - 1)
        yUpperTE = coords[0, 1]
        yLowerTE = coords[-1, 1]

        self.DVGeo.addPointSet(coords, "test")

        # Arrays are short so this is fast enough
        for idx in idxUpper:
            self.assertIn(idx, self.DVGeo.points["test"]["upper"])
        for idx in idxLower:
            self.assertIn(idx, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(yUpperTE, self.DVGeo.points["test"]["yUpperTE"])
        np.testing.assert_equal(yLowerTE, self.DVGeo.points["test"]["yLowerTE"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])

    def test_addPointSet_randomized(self):
        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(self.datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(0, idxLE + self.LEUpper)
        idxLower = np.arange(idxLE + self.LEUpper, coords.shape[0])
        yUpperTE = coords[0, 1]
        yLowerTE = coords[-1, 1]

        # Randomize the index order (do indices so we can track where they end up)
        rng = np.random.default_rng(1)
        # Maps from the original index to the new one (e.g., the first value is the index the first coordinate ends up at)
        idxShuffle = np.arange(0, coords.shape[0])
        rng.shuffle(idxShuffle)
        coordsRand = np.zeros(coords.shape)
        coordsRand[idxShuffle, :] = coords
        idxUpperRand = np.sort(idxShuffle[idxUpper])
        idxLowerRand = np.sort(idxShuffle[idxLower])

        self.DVGeo.addPointSet(coordsRand, "test")

        # Don't include the points at the corners of the trailing edge because it's not guaranteed
        # that they'll be included in the upper and lower surface (which is ok)
        # Arrays are short so this is fast enough
        for idx in idxUpperRand:
            if idx != idxShuffle[0]:
                self.assertIn(idx, self.DVGeo.points["test"]["upper"])
        for idx in idxLowerRand:
            if idx != idxShuffle[-1]:
                self.assertIn(idx, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(yUpperTE, self.DVGeo.points["test"]["yUpperTE"])
        np.testing.assert_equal(yLowerTE, self.DVGeo.points["test"]["yLowerTE"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])

    def test_addPointSet_bluntTE(self):  # includes a blunt trailing edge with points along it
        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(self.datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        nPointsTE = 6  # total points on the trailing edge
        pointsTE = np.ones((nPointsTE - 2, 3), dtype=float)
        pointsTE[:, 2] = 0  # z coordinates are zero
        pointsTE[:, 1] = np.linspace(coords[-1, 1] + 1e-4, coords[0, 1] - 1e-4, pointsTE.shape[0])
        coords = np.vstack((coords, pointsTE))
        idxLE = np.argmin(coords[:, 0])
        # Don't include the points at the corners of the trailing edge because it's not guaranteed
        # that they'll be included in the upper and lower surface (which is ok)
        idxUpper = np.arange(1, idxLE + self.LEUpper)
        idxLower = np.arange(idxLE + self.LEUpper, coords.shape[0] - nPointsTE + 2 - 1)
        yUpperTE = coords[0, 1]
        yLowerTE = coords[coords.shape[0] - nPointsTE + 1, 1]

        self.DVGeo.addPointSet(coords, "test")

        # Arrays are short so this is fast enough
        for idx in idxUpper:
            self.assertIn(idx, self.DVGeo.points["test"]["upper"])
        for idx in idxLower:
            self.assertIn(idx, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(yUpperTE, self.DVGeo.points["test"]["yUpperTE"])
        np.testing.assert_equal(yLowerTE, self.DVGeo.points["test"]["yLowerTE"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
@parameterized_class(airfoils)
class DVGeometryCSTPointSetParallel(unittest.TestCase):
    # Test in parallel
    numPhysicalCores = psutil.cpu_count(logical=False)
    N_PROCS = int(np.clip(numPhysicalCores, 2, 16))

    def setUp(self):
        self.datFile = os.path.join(inputDir, self.fName)
        self.comm = MPI.COMM_WORLD
        self.DVGeo = DVGeometryCST(self.datFile, comm=self.comm)

    def test_addPointSet_sorted(self):
        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(self.datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])

        isUpper = np.zeros(coords.shape[0])
        isUpper[: idxLE + self.LEUpper] = 1
        isUpper = isUpper == 1
        isLower = np.logical_not(isUpper)

        yUpperTE = coords[0, 1]
        yLowerTE = coords[-1, 1]

        # Divide up the points among the procs (mostly evenly, but not quite to check the harder case)
        if self.N_PROCS == 1:
            nPerProc = coords.shape[0]
        else:
            nPerProc = int(coords.shape[0] // (self.N_PROCS - 0.5))
        rank = self.comm.rank
        if self.comm.rank < self.comm.size - 1:  # all but last proc takes nPerProc elements
            self.DVGeo.addPointSet(coords[rank * nPerProc : (rank + 1) * nPerProc, :], "test")
            idxUpper = np.where(isUpper[rank * nPerProc : (rank + 1) * nPerProc])[0]
            idxLower = np.where(isLower[rank * nPerProc : (rank + 1) * nPerProc])[0]
        else:
            self.DVGeo.addPointSet(coords[rank * nPerProc :, :], "test")
            idxUpper = np.where(isUpper[rank * nPerProc :])[0]
            idxLower = np.where(isLower[rank * nPerProc :])[0]

        # Don't include the points at the corners of the trailing edge because it's not guaranteed
        # that they'll be included in the upper and lower surface (which is ok)
        # Arrays are short so this is fast enough
        for idx in idxUpper:
            if idx != 0:
                self.assertIn(idx, self.DVGeo.points["test"]["upper"])
        for idx in idxLower:
            if idx != coords.shape[0] - 1:
                self.assertIn(idx, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(yUpperTE, self.DVGeo.points["test"]["yUpperTE"])
        np.testing.assert_equal(yLowerTE, self.DVGeo.points["test"]["yLowerTE"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])

    def test_addPointSet_randomized(self):
        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(self.datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])

        isUpper = np.zeros(coords.shape[0])
        isUpper[: idxLE + self.LEUpper] = 1
        isUpper = isUpper == 1
        isLower = np.logical_not(isUpper)

        yUpperTE = coords[0, 1]
        yLowerTE = coords[-1, 1]

        # Randomize the index order (do indices so we can track where they end up)
        rng = np.random.default_rng(1)
        # Maps from the original index to the new one (e.g., the first value is the index the first coordinate ends up at)
        idxShuffle = np.arange(0, coords.shape[0])
        rng.shuffle(idxShuffle)
        # idxShuffle = (idxShuffle + 10) % coords.shape[0]
        coordsRand = np.zeros(coords.shape)
        isUpperRand = np.full(isUpper.shape, False)
        isLowerRand = np.full(isLower.shape, False)
        coordsRand[idxShuffle, :] = coords
        isUpperRand[idxShuffle] = isUpper
        isLowerRand[idxShuffle] = isLower

        # Maps from the shuffled index to the original one (e.g., the first value is
        # the original index of the first value in the shuffled array)
        idxInverseShuffle = np.zeros(idxShuffle.shape[0])
        idxInverseShuffle[idxShuffle] = np.arange(0, coords.shape[0])

        # Divide up the points among the procs (mostly evenly, but not quite to check the harder case)
        nPerProc = int(coordsRand.shape[0] // 3.5)
        rank = self.comm.rank
        if self.comm.rank < self.comm.size - 1:  # all but last proc takes nPerProc elements
            self.DVGeo.addPointSet(coordsRand[rank * nPerProc : (rank + 1) * nPerProc, :], "test", rank=self.comm.rank)
            idxUpper = np.where(isUpperRand[rank * nPerProc : (rank + 1) * nPerProc])[0]
            idxLower = np.where(isLowerRand[rank * nPerProc : (rank + 1) * nPerProc])[0]

            # Figure out the local indices where the first and last coordinates in the dat file ended up
            idxStart = np.argwhere(0 == idxInverseShuffle[rank * nPerProc : (rank + 1) * nPerProc])
            idxEnd = np.argwhere(coords.shape[0] == idxInverseShuffle[rank * nPerProc : (rank + 1) * nPerProc])
        else:
            self.DVGeo.addPointSet(coordsRand[rank * nPerProc :, :], "test", rank=self.comm.rank)
            idxUpper = np.where(isUpperRand[rank * nPerProc :])[0]
            idxLower = np.where(isLowerRand[rank * nPerProc :])[0]

            # Figure out the local indices where the first and last coordinates in the dat file ended up
            idxStart = np.argwhere(0 == idxInverseShuffle[rank * nPerProc :])
            idxEnd = np.argwhere(coords.shape[0] == idxInverseShuffle[rank * nPerProc :])

        # Turn the single element array to a number or None if the first
        # or last points aren't in this partition
        if idxStart.size > 0:
            idxStart = idxStart.item()
        else:
            idxStart = None
        if idxEnd.size > 0:
            idxEnd = idxEnd.item()
        else:
            idxEnd = None

        # Don't include the points at the corners of the trailing edge because it's not guaranteed
        # that they'll be included in the upper and lower surface (which is ok)
        # Arrays are short so this is fast enough
        for idx in idxUpper:
            if idx != idxStart:
                self.assertIn(idx, self.DVGeo.points["test"]["upper"])
        for idx in idxLower:
            if idx != idxEnd:
                self.assertIn(idx, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(yUpperTE, self.DVGeo.points["test"]["yUpperTE"])
        np.testing.assert_equal(yLowerTE, self.DVGeo.points["test"]["yLowerTE"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
class DVGeometryCSTSharpOrClosed(unittest.TestCase):
    # Test in serial
    N_PROCS = 1

    def test_addPointSet_sharp(self):
        datFile = os.path.join(inputDir, "naca0012_sharp.dat")
        comm = MPI.COMM_WORLD
        DVGeo = DVGeometryCST(datFile, comm=comm)

        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0]) + 1

        # Don't include the points at the corners of the trailing edge because it's not guaranteed
        # that they'll be included in the upper and lower surface (which is ok)
        idxUpper = np.arange(1, idxLE)
        idxLower = np.arange(idxLE, coords.shape[0] - 1)
        yUpperTE = 0.0
        yLowerTE = 0.0

        DVGeo.addPointSet(coords, "test")

        # Arrays are short so this is fast enough
        for idx in idxUpper:
            self.assertIn(idx, DVGeo.points["test"]["upper"])
        for idx in idxLower:
            self.assertIn(idx, DVGeo.points["test"]["lower"])
        np.testing.assert_equal(yUpperTE, DVGeo.points["test"]["yUpperTE"])
        np.testing.assert_equal(yLowerTE, DVGeo.points["test"]["yLowerTE"])
        self.assertEqual(min(coords[:, 0]), DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), DVGeo.points["test"]["xMax"])
        self.assertTrue(DVGeo.sharp)

    def test_addPointSet_closed(self):
        datFile = os.path.join(inputDir, "naca0012_closed.dat")
        comm = MPI.COMM_WORLD
        DVGeo = DVGeometryCST(datFile, comm=comm)

        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0]) + 1

        # Don't include the points at the corners of the trailing edge because it's not guaranteed
        # that they'll be included in the upper and lower surface (which is ok)
        idxUpper = np.arange(1, idxLE)
        idxLower = np.arange(idxLE, coords.shape[0] - 2)
        yUpperTE = coords[0, 1]
        yLowerTE = coords[-2, 1]

        DVGeo.addPointSet(coords, "test")

        # Arrays are short so this is fast enough
        for idx in idxUpper:
            self.assertIn(idx, DVGeo.points["test"]["upper"])
        for idx in idxLower:
            self.assertIn(idx, DVGeo.points["test"]["lower"])
        np.testing.assert_equal(yUpperTE, DVGeo.points["test"]["yUpperTE"])
        np.testing.assert_equal(yLowerTE, DVGeo.points["test"]["yLowerTE"])
        self.assertEqual(min(coords[:, 0]), DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), DVGeo.points["test"]["xMax"])
        self.assertFalse(DVGeo.sharp)


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
@parameterized_class(DVs)
class DVGeometryCSTSensitivity(unittest.TestCase):
    # Test in serial
    N_PROCS = 1

    def setUp(self):
        self.datFile = os.path.join(inputDir, "naca2412.dat")
        self.rng = np.random.default_rng(1)
        self.comm = MPI.COMM_WORLD
        if self.dvName in ["upper", "lower"]:
            numCST = self.dvNum
        else:
            numCST = 4
        self.DVGeo = DVGeometryCST(self.datFile, comm=self.comm, isComplex=True, numCST=numCST)

        # Read in airfoil coordinates (use NACA 2412)
        coords = readCoordFile(self.datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))  # z-coordinates
        self.coords = coords.astype(complex)
        idxLE = np.argmin(coords[:, 0])
        self.idxUpper = np.arange(0, idxLE)
        self.idxLower = np.arange(idxLE, coords.shape[0])
        self.thickTE = coords[0, 1] - coords[-1, 1]
        self.ptName = "pt"

        self.sensTol = 1e-10
        self.coordTol = 1e-10
        self.CS_delta = 1e-200

    def test_DV_sensitivityProd(self):
        """
        Test DVGeo.totalSensitivityProd for all design variables
        """
        self.DVGeo.addDV(self.dvName, dvType=self.dvName)
        self.DVGeo.addPointSet(self.coords, self.ptName)

        # Set DV to random values
        self.DVGeo.setDesignVars({self.dvName: self.rng.random(self.dvNum)})
        self.DVGeo.update(self.ptName)

        DVs = self.DVGeo.getDesignVars()

        # First compute the analytic ones with the built in function
        sensProd = []
        for i in range(self.dvNum):
            vec = np.zeros(self.dvNum)
            vec[i] = 1
            sensProd.append(np.real(self.DVGeo.totalSensitivityProd({self.dvName: vec}, self.ptName)))

        # Then check them against doing it with complex step
        valDV = DVs[self.dvName]
        for i in range(DVs[self.dvName].size):
            valDV[i] += self.CS_delta * 1j
            self.DVGeo.setDesignVars({self.dvName: valDV})
            pertCoords = self.DVGeo.update(self.ptName)
            valDV[i] -= self.CS_delta * 1j

            dXdDV = np.imag(pertCoords) / self.CS_delta
            np.testing.assert_allclose(sensProd[i], dXdDV, atol=self.sensTol, rtol=self.sensTol)

    def test_DV_sensitivity_simple(self):
        """
        Test DVGeo.totalSensitivity for all design variables with dIdXpt of all ones
        """
        self.DVGeo.addDV(self.dvName, dvType=self.dvName)
        self.DVGeo.addPointSet(self.coords, self.ptName)

        # Set DV to random values
        self.DVGeo.setDesignVars({self.dvName: self.rng.random(self.dvNum)})
        self.DVGeo.update(self.ptName)

        # dIdXpt of all ones means the total sensitivities will just be the sum of the
        # derivatives at each of the coordianates
        dIdXpt = np.ones_like(self.coords)

        DVs = self.DVGeo.getDesignVars()

        # First compute the analytic ones with the built in function
        sens = np.real(self.DVGeo.totalSensitivity(dIdXpt, self.ptName)[self.dvName])

        # Then check them against doing it with complex step
        valDV = DVs[self.dvName]
        sensCS = np.zeros(self.dvNum)
        for i in range(DVs[self.dvName].size):
            valDV[i] += self.CS_delta * 1j
            self.DVGeo.setDesignVars({self.dvName: valDV})
            pertCoords = self.DVGeo.update(self.ptName)
            valDV[i] -= self.CS_delta * 1j

            dXdDV = np.imag(pertCoords) / self.CS_delta

            # Sum the derivatives
            sensCS[i] = np.sum(dXdDV)

        np.testing.assert_allclose(sens, sensCS, atol=self.sensTol, rtol=self.sensTol)

    def test_DV_sensitivity_parallel(self):
        """
        Test DVGeo.totalSensitivity for all design variables with dIdXpt containing
        three different Npts x 3 arrays (another possible input)
        """
        self.DVGeo.addDV(self.dvName, dvType=self.dvName)
        self.DVGeo.addPointSet(self.coords, self.ptName)

        # Set DV to random values
        self.DVGeo.setDesignVars({self.dvName: self.rng.random(self.dvNum)})
        self.DVGeo.update(self.ptName)

        # dIdXpt of all ones means the total sensitivities will just be the sum of the
        # derivatives at each of the coordianates
        dIdXpt = np.ones_like(self.coords)
        coeff = np.array([0.1, 0.5, 1.0])
        dIdXptVectorized = np.array([coeff[0] * dIdXpt, coeff[1] * dIdXpt, coeff[2] * dIdXpt])

        DVs = self.DVGeo.getDesignVars()

        # First compute the analytic ones with the built in function
        sens = np.real(self.DVGeo.totalSensitivity(dIdXptVectorized, self.ptName, comm=self.comm)[self.dvName])

        # Then check them against doing it with complex step
        valDV = DVs[self.dvName]
        sensCS = np.zeros((dIdXptVectorized.shape[0], self.dvNum))
        for i in range(DVs[self.dvName].size):
            valDV[i] += self.CS_delta * 1j
            self.DVGeo.setDesignVars({self.dvName: valDV})
            pertCoords = self.DVGeo.update(self.ptName)
            valDV[i] -= self.CS_delta * 1j

            dXdDV = np.imag(pertCoords) / self.CS_delta

            # Sum the derivatives
            for j in range(sensCS.shape[0]):
                sensCS[j, i] = coeff[j] * np.sum(dXdDV)

        np.testing.assert_allclose(sens, sensCS, atol=self.sensTol, rtol=self.sensTol)


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
@parameterized_class(DVsTE)
class DVGeometryCSTSensitivityTEFace(unittest.TestCase):
    """
    Sensitivities with points along the blunt trailing edge face, which are
    moved by the trailing edge thickness DV but not by the CST coefficients
    """

    N_PROCS = 1

    def setUp(self):
        self.datFile = os.path.join(inputDir, "naca2412.dat")
        self.rng = np.random.default_rng(3)
        self.DVGeo = DVGeometryCST(self.datFile, comm=MPI.COMM_WORLD, isComplex=True, numCST=4)
        self.coords = getCoordsWithTEFace(self.datFile).astype(complex)
        self.ptName = "pt"
        self.sensTol = 1e-10
        self.CS_delta = 1e-200

    def _setRandomDVs(self):
        self.DVGeo.addDV(self.dvName, dvType=self.dvName)
        self.DVGeo.addPointSet(self.coords, self.ptName)
        default = np.real(self.DVGeo.getDesignVars()[self.dvName])
        self.DVGeo.setDesignVars({self.dvName: default + 0.05 * self.rng.random(self.dvNum)})
        self.DVGeo.update(self.ptName)

    def _complexStepJacobian(self):
        """Return dX/dDV computed with complex step, shape (dvNum, nPts, 3)"""
        valDV = self.DVGeo.getDesignVars()[self.dvName].copy()
        jac = []
        for i in range(self.dvNum):
            valDV[i] += self.CS_delta * 1j
            self.DVGeo.setDesignVars({self.dvName: valDV})
            jac.append(np.imag(self.DVGeo.update(self.ptName)) / self.CS_delta)
            valDV[i] -= self.CS_delta * 1j
        self.DVGeo.setDesignVars({self.dvName: valDV})
        self.DVGeo.update(self.ptName)
        return np.array(jac)

    def test_sensitivityProd(self):
        self._setRandomDVs()
        jacCS = self._complexStepJacobian()
        for i in range(self.dvNum):
            vec = np.zeros(self.dvNum)
            vec[i] = 1.0
            sensProd = np.real(self.DVGeo.totalSensitivityProd({self.dvName: vec}, self.ptName))
            np.testing.assert_allclose(sensProd, jacCS[i], atol=self.sensTol, rtol=self.sensTol)

    def test_sensitivity(self):
        self._setRandomDVs()
        jacCS = self._complexStepJacobian()
        dIdXpt = self.rng.random(self.coords.shape)
        sens = np.real(self.DVGeo.totalSensitivity(dIdXpt, self.ptName)[self.dvName])
        sensCS = np.einsum("ijk,jk->i", jacCS, dIdXpt)
        np.testing.assert_allclose(sens.flatten(), sensCS, atol=self.sensTol, rtol=self.sensTol)


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
@parameterized_class(DVsTE)
class DVGeometryCSTSensitivityParallel(unittest.TestCase):
    """Sensitivities with the point set split across two procs, so the allreduce in totalSensitivity is exercised"""

    N_PROCS = 2

    def setUp(self):
        self.datFile = os.path.join(inputDir, "naca2412.dat")
        self.comm = MPI.COMM_WORLD
        self.DVGeo = DVGeometryCST(self.datFile, comm=self.comm, isComplex=True, numCST=4, fitLEM=True)
        coords = getCoordsWithTEFace(self.datFile)

        # Interleave the points between procs so each one has upper, lower, and trailing edge points
        self.coords = coords[self.comm.rank :: self.comm.size].astype(complex)
        self.ptName = "pt"
        self.sensTol = 1e-10
        self.CS_delta = 1e-200

    def test_sensitivity_parallel(self):
        rng = np.random.default_rng(5)  # same seed on all procs so the DVs match
        self.DVGeo.addDV(self.dvName, dvType=self.dvName)
        self.DVGeo.addPointSet(self.coords, self.ptName)
        default = np.real(self.DVGeo.getDesignVars()[self.dvName])
        self.DVGeo.setDesignVars({self.dvName: default + 0.05 * rng.random(self.dvNum)})
        self.DVGeo.update(self.ptName)

        dIdXpt = np.ones(self.coords.shape)
        sens = np.real(self.DVGeo.totalSensitivity(dIdXpt, self.ptName, comm=self.comm)[self.dvName])

        valDV = self.DVGeo.getDesignVars()[self.dvName].copy()
        sensCSLocal = np.zeros(self.dvNum)
        for i in range(self.dvNum):
            valDV[i] += self.CS_delta * 1j
            self.DVGeo.setDesignVars({self.dvName: valDV})
            sensCSLocal[i] = np.sum(np.imag(self.DVGeo.update(self.ptName)) / self.CS_delta)
            valDV[i] -= self.CS_delta * 1j
        sensCS = self.comm.allreduce(sensCSLocal, op=MPI.SUM)

        np.testing.assert_allclose(sens.flatten(), sensCS, atol=self.sensTol, rtol=self.sensTol)


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
class DVGeometryCSTFitParallel(unittest.TestCase):
    N_PROCS = 2

    def test_fitLEM_same_on_all_procs(self):
        """The fitted LEM weight is computed on the root proc and must be broadcast to all procs"""
        comm = MPI.COMM_WORLD
        DVGeo = DVGeometryCST(os.path.join(inputDir, "naca2412.dat"), comm=comm, numCST=4, fitLEM=True)
        lem = DVGeo.defaultDV["lem"]
        allLEM = comm.allgather(lem.copy())
        for val in allLEM:
            np.testing.assert_array_equal(val, allLEM[0])
        self.assertNotEqual(np.real(lem[0]), 0.0)


@unittest.skipUnless(prefoilImported and pyoptsparseImported, "preFoil and pyOptSparse are required")
class DVGeometryCSTCoverage(unittest.TestCase):
    """
    Cases that exercise the parts of the LEM, trailing edge thickness, joint fit, and constraint helper code
    that the tests above do not: different numbers of coefficients on each surface, a scaled and shifted
    airfoil with an asymmetric trailing edge, and combinations of DVs.
    """

    N_PROCS = 1

    def setUp(self):
        self.tmpDir = tempfile.TemporaryDirectory()
        self.naca2412 = os.path.join(inputDir, "naca2412.dat")
        self.CS_delta = 1e-200
        self.tol = 1e-10

    def tearDown(self):
        self.tmpDir.cleanup()

    def _checkSensitivities(self, DVGeo, pts, dvName, value=None):
        """Compare totalSensitivity and totalSensitivityProd with complex step for each entry of a DV"""
        rng = np.random.default_rng(11)
        if value is not None:
            DVGeo.setDesignVars({dvName: value})
        DVGeo.update("p")
        v = np.real(DVGeo.getDesignVars()[dvName]).copy()
        dIdpt = rng.random(pts.shape)
        sens = np.real(np.atleast_1d(DVGeo.totalSensitivity(dIdpt, "p")[dvName]).flatten())
        for i in range(v.size):
            vc = v.astype(complex)
            vc[i] += self.CS_delta * 1j
            DVGeo.setDesignVars({dvName: vc})
            dXdDV = np.imag(DVGeo.update("p")) / self.CS_delta
            DVGeo.setDesignVars({dvName: v})
            DVGeo.update("p")
            seed = np.zeros(v.size)
            seed[i] = 1.0
            prod = np.real(DVGeo.totalSensitivityProd({dvName: seed}, "p"))
            np.testing.assert_allclose(prod, dXdDV, atol=self.tol, rtol=self.tol)
            np.testing.assert_allclose(sens[i], np.sum(dXdDV * dIdpt), atol=self.tol, rtol=self.tol)

    def _writeDat(self, name, xUpper, yUpper, xLower, yLower):
        """Write upper (TE -> LE) and lower (LE -> TE) surfaces, given from LE to TE, to a dat file"""
        coords = np.vstack((np.column_stack((xUpper[::-1], yUpper[::-1])), np.column_stack((xLower[1:], yLower[1:]))))
        fileName = os.path.join(self.tmpDir.name, name)
        np.savetxt(fileName, coords)
        return fileName, coords

    def _scaledShiftedDat(self):
        """Synthetic airfoil with chord 2, LE at x = 0.5, and a trailing edge that is not centered at y = 0"""
        upper = np.array([0.17, 0.16, 0.2, 0.15, 0.22, 0.18, 0.2, 0.17])
        lower = np.array([-0.17, -0.1, -0.08, -0.12, -0.02, -0.05, 0.02, 0.05])
        x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 121)))
        yUpper, yLower = kulfanCoordinates(x, upper, lower, 0.25, 0.004)
        # Shift the trailing edge midpoint to y = 0.003 (chord-normalized)
        yUpper += 0.003 * x
        yLower += 0.003 * x
        scale, shift = 2.0, 0.5
        fileName, coords = self._writeDat(
            "scaled.dat", scale * x + shift, scale * yUpper, scale * x + shift, scale * yLower
        )
        return fileName, coords, upper, lower

    def test_lem_sensitivity_unequal_nCST(self):
        """The LEM exponent depends on each surface's number of coefficients in the sensitivities"""
        DVGeo = DVGeometryCST(self.naca2412, numCST=[6, 3], isComplex=True)
        DVGeo.addDV("lem", dvType="lem")
        pts = getCoordsWithTEFace(self.naca2412).astype(complex)
        DVGeo.addPointSet(pts, "p")
        self._checkSensitivities(DVGeo, pts, "lem", np.array([0.1]))

    def test_joint_fit_round_trip_unequal_nCST(self):
        """The LEM exponent depends on each surface's number of coefficients in the joint fit"""
        upper = np.array([0.17, 0.16, 0.2, 0.15, 0.22, 0.18])
        lower = np.array([-0.15, -0.1, -0.08, 0.02])
        x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 121)))
        yUpper = kulfanCoordinates(x, upper, upper, 0.25, 0.004)[0]
        yLower = kulfanCoordinates(x, lower, lower, 0.25, 0.004)[1]
        fileName, _ = self._writeDat("unequal.dat", x, yUpper, x, yLower)
        DVGeo = DVGeometryCST(fileName, numCST=[6, 4], fitLEM=True)
        np.testing.assert_allclose(DVGeo.defaultDV["upper"], upper, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lower"], lower, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lem"], [0.25], atol=1e-8)

    def test_joint_fit_scaled_shifted(self):
        """The joint fit normalizes the coordinates by the chord and leading edge position"""
        fileName, _, upper, lower = self._scaledShiftedDat()
        DVGeo = DVGeometryCST(fileName, numCST=8, fitLEM=True, fitLEContinuity=True)
        np.testing.assert_allclose(DVGeo.defaultDV["upper"], upper, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lower"], lower, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lem"], [0.25], atol=1e-8)

    def test_te_thickness_asymmetric_TE_scaled(self):
        """The trailing edge opens and closes about the dat file's trailing edge midpoint"""
        fileName, coords, _, _ = self._scaledShiftedDat()
        DVGeo = DVGeometryCST(fileName, numCST=8, fitLEM=True)
        DVGeo.addDV("te", dvType="te_thickness")
        pts = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        DVGeo.addPointSet(pts, "p")
        np.testing.assert_allclose(DVGeo.update("p"), pts, atol=1e-10)
        DVGeo.setDesignVars({"te": np.array([0.0])})
        y = DVGeo.update("p")
        np.testing.assert_allclose(y[0, 1], y[-1, 1], atol=1e-14)
        np.testing.assert_allclose(y[0, 1], 0.5 * (pts[0, 1] + pts[-1, 1]), atol=1e-12)

    def test_te_face_with_chord_dv(self):
        """Trailing edge face points with the chord and trailing edge thickness DVs together"""
        DVGeo = DVGeometryCST(self.naca2412, numCST=4, isComplex=True)
        DVGeo.addDV("c", dvType="chord")
        DVGeo.addDV("te", dvType="te_thickness")
        pts = getCoordsWithTEFace(self.naca2412).astype(complex)
        DVGeo.addPointSet(pts, "p")
        DVGeo.setDesignVars({"c": np.array([1.7])})
        self._checkSensitivities(DVGeo, pts, "te", np.array([0.01]))
        self._checkSensitivities(DVGeo, pts, "c")

    def test_lem_complex_step_at_zero(self):
        """Complex step through the LEM DV works at its default value of zero"""
        DVGeo = DVGeometryCST(self.naca2412, numCST=4, isComplex=True)
        DVGeo.addDV("lem", dvType="lem")
        pts = getCoordsWithTEFace(self.naca2412).astype(complex)
        DVGeo.addPointSet(pts, "p")
        self._checkSensitivities(DVGeo, pts, "lem", np.array([0.0]))

    def _optProb(self, DVGeo):
        optProb = Optimization("test", lambda x: x)
        DVGeo.addVariablesPyOpt(optProb)
        return optProb

    def test_no_warning_when_tied(self):
        DVGeo = DVGeometryCST(self.naca2412, numCST=4, fitLEContinuity=True)
        DVGeo.addDV("up", dvType="upper")
        DVGeo.addDV("low", dvType="lower")
        optProb = self._optProb(DVGeo)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            DVGeo.addLEContinuityConstraint(optProb)

    def test_warns_with_separate_N1_lower(self):
        DVGeo = DVGeometryCST(self.naca2412, numCST=4, fitLEContinuity=True)
        DVGeo.addDV("up", dvType="upper")
        DVGeo.addDV("low", dvType="lower")
        DVGeo.addDV("n1l", dvType="n1_lower")
        optProb = self._optProb(DVGeo)
        with self.assertWarns(UserWarning):
            DVGeo.addLEContinuityConstraint(optProb)

    def test_near_corner_points_on_surfaces(self):
        """With trailing edge face points in the dat file, points near each corner stay on their surface"""
        coords = readCoordFile(self.naca2412)
        nFace = 8
        yFace = np.linspace(coords[-1, 1], coords[0, 1], nFace + 2)[1:-1]
        face = np.column_stack((np.full(nFace, coords[0, 0] - 1e-9), yFace))
        fileName = os.path.join(self.tmpDir.name, "face.dat")
        np.savetxt(fileName, np.vstack((coords, face, coords[:1])))
        DVGeo = DVGeometryCST(fileName, numCST=8)

        # Points along the last surface segment at each corner (alternating upper and lower)
        pts = []
        for a in [0.25, 0.5, 0.75]:
            pts.append(a * coords[0] + (1 - a) * coords[1])
            pts.append(a * coords[-1] + (1 - a) * coords[-2])
        pts = np.hstack((np.array(pts), np.zeros((len(pts), 1))))
        DVGeo.addPointSet(pts, "p")
        np.testing.assert_array_equal(np.sort(DVGeo.points["p"]["upper"]), [0, 2, 4])
        np.testing.assert_array_equal(np.sort(DVGeo.points["p"]["lower"]), [1, 3, 5])


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
class DVGeometryCSTDatWithTEFace(unittest.TestCase):
    """
    Dat files written by preFoil (e.g., in the MACH-Aero airfoil tutorials) are closed loops that include
    points along the blunt trailing edge face. Those points must be treated as trailing edge points,
    not snapped to the upper or lower surface.
    """

    N_PROCS = 1

    def setUp(self):
        self.tmpDir = tempfile.TemporaryDirectory()
        coords = readCoordFile(os.path.join(inputDir, "naca2412.dat"))
        nFace = 8
        yFace = np.linspace(coords[-1, 1], coords[0, 1], nFace + 2)[1:-1]
        # preFoil's face points are not exactly on the line between the corners (roundoff-level offsets),
        # so put them slightly inside it. They are still much closer to the face than to either surface.
        face = np.column_stack((np.full(nFace, coords[0, 0] - 1e-9), yFace))
        # Upper TE -> LE -> lower TE -> up the TE face -> back to the upper TE (closed)
        self.datCoords = np.vstack((coords, face, coords[:1]))
        self.datFile = os.path.join(self.tmpDir.name, "naca2412_teface.dat")
        np.savetxt(self.datFile, self.datCoords)
        self.idxFace = np.arange(coords.shape[0], coords.shape[0] + nFace)

    def tearDown(self):
        self.tmpDir.cleanup()

    def test_face_points_are_trailing_edge_points(self):
        DVGeo = DVGeometryCST(self.datFile, numCST=8)
        coords = np.hstack((self.datCoords, np.zeros((self.datCoords.shape[0], 1))))
        DVGeo.addPointSet(coords, "pt")
        surface = np.concatenate((DVGeo.points["pt"]["upper"], DVGeo.points["pt"]["lower"]))
        for idx in self.idxFace:
            self.assertNotIn(idx, surface)

        # The face points do not move when the geometry is updated with the initial DVs
        np.testing.assert_allclose(DVGeo.update("pt")[self.idxFace], coords[self.idxFace], atol=1e-14)


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
class DVGeometryCSTLEMTEUpdate(unittest.TestCase):
    """Check how the LEM and trailing edge thickness DVs change the point set"""

    N_PROCS = 1

    def setUp(self):
        self.datFile = os.path.join(inputDir, "naca2412.dat")
        self.nUpper = 5
        self.nLower = 4
        self.DVGeo = DVGeometryCST(self.datFile, numCST=[self.nUpper, self.nLower])
        self.coords = getCoordsWithTEFace(self.datFile)
        self.ptName = "pt"

        # Hand-computed quantities from the dat file (independent of DVGeometryCST)
        datCoords = readCoordFile(self.datFile)
        self.xMin = np.min(datCoords[:, 0])
        self.chord = np.max(datCoords[:, 0]) - self.xMin
        self.yUpperTE = datCoords[0, 1]
        self.yLowerTE = datCoords[-1, 1]
        self.tol = 1e-13

    def _indices(self):
        upper = self.DVGeo.points[self.ptName]["upper"]
        lower = self.DVGeo.points[self.ptName]["lower"]
        te = np.setdiff1d(np.arange(self.coords.shape[0]), np.concatenate((upper, lower)))
        return upper, lower, te

    def test_lem_shifts_both_surfaces(self):
        self.DVGeo.addDV("lem", dvType="lem")
        self.DVGeo.addPointSet(self.coords, self.ptName)
        y0 = self.DVGeo.update(self.ptName)[:, 1].copy()

        wLE = 0.2
        self.DVGeo.setDesignVars({"lem": np.array([wLE])})
        y1 = self.DVGeo.update(self.ptName)[:, 1]

        upper, lower, te = self._indices()
        self.assertGreater(te.size, 0)
        x = (self.coords[:, 0] - self.xMin) / self.chord
        lemUpper = np.where(x < 1, x * np.abs(1 - x) ** (self.nUpper - 0.5), 0.0)
        lemLower = np.where(x < 1, x * np.abs(1 - x) ** (self.nLower - 0.5), 0.0)
        np.testing.assert_allclose(y1[upper] - y0[upper], self.chord * wLE * lemUpper[upper], atol=self.tol)
        np.testing.assert_allclose(y1[lower] - y0[lower], self.chord * wLE * lemLower[lower], atol=self.tol)
        np.testing.assert_allclose(y1[te], y0[te], atol=self.tol)

    def test_te_thickness_default_is_dat_thickness(self):
        self.DVGeo.addDV("te", dvType="te_thickness")
        np.testing.assert_allclose(
            self.DVGeo.getDesignVars()["te"], [(self.yUpperTE - self.yLowerTE) / self.chord], atol=1e-15
        )

    def test_te_thickness_default_geometry_unchanged(self):
        """Adding the trailing edge thickness DV at its default value does not move any points"""
        DVGeoRef = DVGeometryCST(self.datFile, numCST=[self.nUpper, self.nLower])
        DVGeoRef.addPointSet(self.coords, self.ptName)
        yRef = DVGeoRef.update(self.ptName)

        self.DVGeo.addDV("te", dvType="te_thickness")
        self.DVGeo.addPointSet(self.coords, self.ptName)
        np.testing.assert_allclose(self.DVGeo.update(self.ptName), yRef, atol=self.tol)

    def test_te_thickness_opens_symmetrically(self):
        self.DVGeo.addDV("te", dvType="te_thickness")
        self.DVGeo.addPointSet(self.coords, self.ptName)
        y0 = self.DVGeo.update(self.ptName)[:, 1].copy()

        dt = 0.01
        t0 = (self.yUpperTE - self.yLowerTE) / self.chord
        self.DVGeo.setDesignVars({"te": np.array([t0 + dt])})
        y1 = self.DVGeo.update(self.ptName)[:, 1]

        upper, lower, te = self._indices()
        x = (self.coords[:, 0] - self.xMin) / self.chord
        np.testing.assert_allclose(y1[upper] - y0[upper], self.chord * dt / 2 * x[upper], atol=self.tol)
        np.testing.assert_allclose(y1[lower] - y0[lower], -self.chord * dt / 2 * x[lower], atol=self.tol)

        # Points on the trailing edge face keep their fractional position between the corners
        eta = (self.coords[te, 1] - self.yLowerTE) / (self.yUpperTE - self.yLowerTE)
        np.testing.assert_allclose(y1[te] - y0[te], self.chord * dt * (eta - 0.5), atol=self.tol)

    def test_update_idempotent(self):
        self.DVGeo.addDV("lem", dvType="lem")
        self.DVGeo.addDV("te", dvType="te_thickness")
        self.DVGeo.addDV("chord", dvType="chord")
        self.DVGeo.addPointSet(self.coords, self.ptName)

        dvs = {"lem": np.array([0.1]), "te": np.array([0.02]), "chord": np.array([1.3])}
        self.DVGeo.setDesignVars(dvs)
        first = self.DVGeo.update(self.ptName).copy()
        np.testing.assert_allclose(self.DVGeo.update(self.ptName), first, atol=self.tol)

        # Go somewhere else (including a closed trailing edge) and come back
        self.DVGeo.setDesignVars({"lem": np.array([-0.3]), "te": np.array([0.0]), "chord": np.array([0.7])})
        self.DVGeo.update(self.ptName)
        self.DVGeo.setDesignVars(dvs)
        np.testing.assert_allclose(self.DVGeo.update(self.ptName), first, atol=self.tol)

    def test_te_thickness_sharp_raises(self):
        DVGeo = DVGeometryCST(os.path.join(inputDir, "naca0012_sharp.dat"))
        with self.assertRaises(ValueError):
            DVGeo.addDV("te", dvType="te_thickness")

    def test_duplicate_lem_raises(self):
        self.DVGeo.addDV("lem", dvType="lem")
        with self.assertRaises(ValueError):
            self.DVGeo.addDV("lem2", dvType="lem")

    def test_duplicate_te_thickness_raises(self):
        self.DVGeo.addDV("te", dvType="te_thickness")
        with self.assertRaises(ValueError):
            self.DVGeo.addDV("te2", dvType="te_thickness")


def kulfanCoordinates(x, upper, lower, wLE, teThickness, wLELower=None):
    """
    Independent reference implementation of Kulfan's CST airfoil with the leading edge modification
    (N1 = 0.5, N2 = 1): the LEM term on each surface is w * x * (1 - x)^(N + 0.5), with N the Bernstein order.
    wLE is the upper surface LEM weight and also the lower one unless wLELower is given.
    Returns the upper and lower surface y coordinates at x.
    """
    if wLELower is None:
        wLELower = wLE

    def surface(w, wLEM):
        n = len(w) - 1
        bernstein = sum(w[i] * comb(n, i) * x**i * (1 - x) ** (n - i) for i in range(n + 1))
        return np.sqrt(x) * (1 - x) * bernstein + wLEM * x * (1 - x) ** (n + 0.5)

    return surface(upper, wLE) + x * teThickness / 2, surface(lower, wLELower) - x * teThickness / 2


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
class DVGeometryCSTJointFit(unittest.TestCase):
    """Fitting options that enforce leading edge continuity and/or include the LEM term"""

    N_PROCS = 1

    def setUp(self):
        self.tmpDir = tempfile.TemporaryDirectory()
        self.naca2412 = os.path.join(inputDir, "naca2412.dat")

        # Synthetic blunt trailing edge airfoil with known Kulfan parameters
        self.upper = np.array([0.17, 0.16, 0.2, 0.15, 0.22, 0.18, 0.2, 0.17])
        self.lower = np.array([-0.15, -0.1, -0.08, -0.12, -0.02, -0.05, 0.02, 0.05])
        self.wLE = 0.25
        self.teThickness = 0.004
        self.syntheticUntied = self._writeSynthetic("untied.dat", self.upper, self.lower)
        lowerTied = self.lower.copy()
        lowerTied[0] = -self.upper[0]
        self.lowerTied = lowerTied
        self.syntheticTied = self._writeSynthetic("tied.dat", self.upper, lowerTied)

    def tearDown(self):
        self.tmpDir.cleanup()

    def _writeSynthetic(self, name, upper, lower):
        x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 121)))
        yUpper, yLower = kulfanCoordinates(x, upper, lower, self.wLE, self.teThickness)
        coords = np.vstack((np.column_stack((x[::-1], yUpper[::-1])), np.column_stack((x[1:], yLower[1:]))))
        fileName = os.path.join(self.tmpDir.name, name)
        np.savetxt(fileName, coords)
        return fileName

    def _fitError(self, DVGeo, datFile):
        """RMS error between the dat file coordinates and the geometry they are mapped to by the fit"""
        coords = readCoordFile(datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        DVGeo.addPointSet(coords, "dat")
        return np.sqrt(np.mean((DVGeo.update("dat")[:, 1] - coords[:, 1]) ** 2))

    def test_round_trip_untied(self):
        """An airfoil generated with Kulfan's formula (shared LEM weight) is recovered exactly by the LEM fit"""
        DVGeo = DVGeometryCST(self.syntheticUntied, numCST=8, fitLEM=True)
        np.testing.assert_allclose(DVGeo.defaultDV["upper"], self.upper, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lower"], self.lower, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lem"], [self.wLE], atol=1e-8)
        self.assertLess(self._fitError(DVGeo, self.syntheticUntied), 1e-10)

    def test_round_trip_tied(self):
        """A tied airfoil with LEM is recovered exactly by the tied LEM fit"""
        DVGeo = DVGeometryCST(self.syntheticTied, numCST=8, fitLEM=True, fitLEContinuity=True)
        np.testing.assert_allclose(DVGeo.defaultDV["upper"], self.upper, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lower"], self.lowerTied, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lem"], [self.wLE], atol=1e-8)
        self.assertLess(self._fitError(DVGeo, self.syntheticTied), 1e-10)

    def test_tied_fit_is_exactly_continuous(self):
        """The tied fit gives lower[0] = -upper[0] exactly, even for a cambered airfoil"""
        for fitLEM in [False, True]:
            DVGeo = DVGeometryCST(self.naca2412, numCST=8, fitLEContinuity=True, fitLEM=fitLEM)
            self.assertEqual(DVGeo.defaultDV["lower"][0], -DVGeo.defaultDV["upper"][0])

    def test_tied_fit_without_LEM_has_zero_LEM(self):
        DVGeo = DVGeometryCST(self.naca2412, numCST=8, fitLEContinuity=True)
        np.testing.assert_array_equal(DVGeo.defaultDV["lem"], [0.0])

    def test_LEM_improves_tied_fit(self):
        """For a cambered airfoil, the LEM term absorbs the leading edge droop that the tie removes"""
        errTied = self._fitError(DVGeometryCST(self.naca2412, numCST=8, fitLEContinuity=True), self.naca2412)
        errTiedLEM = self._fitError(
            DVGeometryCST(self.naca2412, numCST=8, fitLEContinuity=True, fitLEM=True), self.naca2412
        )
        self.assertLess(errTiedLEM, 0.5 * errTied)


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
class DVGeometryCSTPerSurfaceLEM(unittest.TestCase):
    """Kulfan's per-surface LEM weights: the "lem_upper" and "lem_lower" DVs and fitLEM="per_surface\""""

    N_PROCS = 1

    def setUp(self):
        self.tmpDir = tempfile.TemporaryDirectory()
        self.naca2412 = os.path.join(inputDir, "naca2412.dat")
        self.upper = np.array([0.17, 0.16, 0.2, 0.15, 0.22, 0.18, 0.2, 0.17])
        self.lower = np.array([-0.17, -0.1, -0.08, -0.12, -0.02, -0.05, 0.02, 0.05])
        self.wLEUpper = 0.25
        self.wLELower = -0.1
        self.tol = 1e-13

    def tearDown(self):
        self.tmpDir.cleanup()

    def _writeSynthetic(self, upper, lower, name="ps.dat"):
        x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 121)))
        yUpper, yLower = kulfanCoordinates(x, upper, lower, self.wLEUpper, 0.004, wLELower=self.wLELower)
        coords = np.vstack((np.column_stack((x[::-1], yUpper[::-1])), np.column_stack((x[1:], yLower[1:]))))
        fileName = os.path.join(self.tmpDir.name, name)
        np.savetxt(fileName, coords)
        return fileName

    def test_round_trip_per_surface_tied(self):
        """A tied airfoil with different upper and lower LEM weights is recovered by the per-surface fit"""
        DVGeo = DVGeometryCST(
            self._writeSynthetic(self.upper, self.lower), numCST=8, fitLEContinuity=True, fitLEM="per_surface"
        )
        np.testing.assert_allclose(DVGeo.defaultDV["upper"], self.upper, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lower"], self.lower, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lem_upper"], [self.wLEUpper], atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lem_lower"], [self.wLELower], atol=1e-8)

    def test_round_trip_per_surface_untied_unequal_nCST(self):
        upper = self.upper[:6]
        lower = np.array([-0.15, -0.1, -0.08, 0.02])
        DVGeo = DVGeometryCST(self._writeSynthetic(upper, lower, "u.dat"), numCST=[6, 4], fitLEM="per_surface")
        np.testing.assert_allclose(DVGeo.defaultDV["upper"], upper, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lower"], lower, atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lem_upper"], [self.wLEUpper], atol=1e-8)
        np.testing.assert_allclose(DVGeo.defaultDV["lem_lower"], [self.wLELower], atol=1e-8)

    def test_shared_fit_sets_both_surfaces(self):
        DVGeo = DVGeometryCST(self.naca2412, numCST=8, fitLEContinuity=True, fitLEM=True)
        self.assertEqual(DVGeo.defaultDV["lem_upper"][0], DVGeo.defaultDV["lem_lower"][0])
        self.assertEqual(DVGeo.defaultDV["lem"][0], DVGeo.defaultDV["lem_upper"][0])
        self.assertNotEqual(DVGeo.defaultDV["lem"][0], 0.0)

    def test_lem_upper_moves_only_upper(self):
        nUpper, nLower = 5, 4
        DVGeo = DVGeometryCST(self.naca2412, numCST=[nUpper, nLower])
        DVGeo.addDV("lu", dvType="lem_upper")
        DVGeo.addDV("ll", dvType="lem_lower")
        coords = getCoordsWithTEFace(self.naca2412)
        DVGeo.addPointSet(coords, "p")
        y0 = DVGeo.update("p")[:, 1].copy()
        DVGeo.setDesignVars({"lu": np.array([0.2]), "ll": np.array([-0.3])})
        y1 = DVGeo.update("p")[:, 1]
        upper, lower = DVGeo.points["p"]["upper"], DVGeo.points["p"]["lower"]
        datCoords = readCoordFile(self.naca2412)
        xMin = datCoords[:, 0].min()
        chord = datCoords[:, 0].max() - xMin
        x = (coords[:, 0] - xMin) / chord
        shape = lambda n: np.where(x < 1, x * np.abs(1 - x) ** (n - 0.5), 0.0)  # noqa: E731
        np.testing.assert_allclose(y1[upper] - y0[upper], chord * 0.2 * shape(nUpper)[upper], atol=self.tol)
        np.testing.assert_allclose(y1[lower] - y0[lower], chord * -0.3 * shape(nLower)[lower], atol=self.tol)

    def test_per_surface_sensitivity_unequal_nCST(self):
        """Each per-surface LEM DV uses its own surface's number of coefficients in the sensitivities"""
        for dvName in ["lem_upper", "lem_lower"]:
            DVGeo = DVGeometryCST(self.naca2412, numCST=[6, 3], isComplex=True)
            DVGeo.addDV(dvName, dvType=dvName)
            pts = getCoordsWithTEFace(self.naca2412).astype(complex)
            DVGeo.addPointSet(pts, "p")
            DVGeo.setDesignVars({dvName: np.array([0.1])})
            DVGeo.update("p")
            rng = np.random.default_rng(2)
            dIdpt = rng.random(pts.shape)
            sens = np.real(np.atleast_1d(DVGeo.totalSensitivity(dIdpt, "p")[dvName]))
            prod = np.real(DVGeo.totalSensitivityProd({dvName: np.array([1.0])}, "p"))
            DVGeo.setDesignVars({dvName: np.array([0.1 + 1e-200j])})
            dXdDV = np.imag(DVGeo.update("p")) / 1e-200
            np.testing.assert_allclose(prod, dXdDV, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(sens, [np.sum(dXdDV * dIdpt)], atol=1e-10, rtol=1e-10)

    def test_lem_and_lem_upper_exclusive(self):
        DVGeo = DVGeometryCST(self.naca2412, numCST=4)
        DVGeo.addDV("lem", dvType="lem")
        with self.assertRaises(ValueError):
            DVGeo.addDV("lu", dvType="lem_upper")

    def test_lem_lower_and_lem_exclusive(self):
        DVGeo = DVGeometryCST(self.naca2412, numCST=4)
        DVGeo.addDV("ll", dvType="lem_lower")
        with self.assertRaises(ValueError):
            DVGeo.addDV("lem", dvType="lem")

    def test_duplicate_lem_upper(self):
        DVGeo = DVGeometryCST(self.naca2412, numCST=4)
        DVGeo.addDV("lu", dvType="lem_upper")
        with self.assertRaises(ValueError):
            DVGeo.addDV("lu2", dvType="lem_upper")

    def test_invalid_fitLEM(self):
        with self.assertRaises(ValueError):
            DVGeometryCST(self.naca2412, numCST=4, fitLEM="both")

    def test_shared_dv_after_per_surface_fit_warns(self):
        """Adding the shared "lem" DV replaces two different fitted weights with one, so warn"""
        DVGeo = DVGeometryCST(self._writeSynthetic(self.upper, self.lower), numCST=8, fitLEM="per_surface")
        with self.assertWarns(UserWarning):
            DVGeo.addDV("lem", dvType="lem")


@unittest.skipUnless(prefoilImported and pyoptsparseImported, "preFoil and pyOptSparse are required")
class DVGeometryCSTLEContinuityConstraint(unittest.TestCase):
    """The helper that adds the linear constraint upper[0] + lower[0] = 0 to a pyOptSparse problem"""

    N_PROCS = 1

    def setUp(self):
        self.datFile = os.path.join(inputDir, "naca2412.dat")

    def _constraintValue(self, con, dvs):
        """Evaluate a linear constraint from its user-supplied Jacobian"""
        return sum(np.asarray(con.jac[dvName]) @ dvs[dvName] for dvName in con.wrt)

    def test_constraint_is_linear_equality_on_first_coefficients(self):
        DVGeo = DVGeometryCST(self.datFile, numCST=[5, 4], fitLEContinuity=True)
        DVGeo.addDV("up", dvType="upper")
        DVGeo.addDV("low", dvType="lower")
        DVGeo.addDV("lem", dvType="lem")
        optProb = Optimization("test", lambda x: x)
        DVGeo.addVariablesPyOpt(optProb)
        DVGeo.addLEContinuityConstraint(optProb)

        con = optProb.constraints["le_continuity"]
        self.assertTrue(con.linear)
        self.assertEqual(con.ncon, 1)
        np.testing.assert_array_equal(con.lower, [0.0])
        np.testing.assert_array_equal(con.upper, [0.0])
        self.assertEqual(sorted(con.wrt), ["low", "up"])

        dvs = {"up": np.array([0.2, 1.0, 2.0, 3.0, 4.0]), "low": np.array([-0.15, 5.0, 6.0, 7.0])}
        np.testing.assert_allclose(self._constraintValue(con, dvs), [0.05], atol=1e-15)

    def test_custom_name(self):
        DVGeo = DVGeometryCST(self.datFile, numCST=4, fitLEContinuity=True)
        DVGeo.addDV("up", dvType="upper")
        DVGeo.addDV("low", dvType="lower")
        optProb = Optimization("test", lambda x: x)
        DVGeo.addVariablesPyOpt(optProb)
        DVGeo.addLEContinuityConstraint(optProb, name="nose")
        self.assertIn("nose", optProb.constraints)

    def test_missing_lower_raises(self):
        DVGeo = DVGeometryCST(self.datFile, numCST=4)
        DVGeo.addDV("up", dvType="upper")
        optProb = Optimization("test", lambda x: x)
        DVGeo.addVariablesPyOpt(optProb)
        with self.assertRaises(ValueError):
            DVGeo.addLEContinuityConstraint(optProb)

    def test_warns_when_initial_design_violates_constraint(self):
        """The default (untied) fit of a cambered airfoil violates the constraint at the start"""
        DVGeo = DVGeometryCST(self.datFile, numCST=4)
        DVGeo.addDV("up", dvType="upper")
        DVGeo.addDV("low", dvType="lower")
        optProb = Optimization("test", lambda x: x)
        DVGeo.addVariablesPyOpt(optProb)
        with self.assertWarns(UserWarning):
            DVGeo.addLEContinuityConstraint(optProb)

    def test_warns_with_separate_N1(self):
        """Different N1 on the upper and lower surfaces also breaks leading edge continuity"""
        DVGeo = DVGeometryCST(self.datFile, numCST=4, fitLEContinuity=True)
        DVGeo.addDV("up", dvType="upper")
        DVGeo.addDV("low", dvType="lower")
        DVGeo.addDV("n1u", dvType="n1_upper")
        optProb = Optimization("test", lambda x: x)
        DVGeo.addVariablesPyOpt(optProb)
        with self.assertWarns(UserWarning):
            DVGeo.addLEContinuityConstraint(optProb)


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
class TestFunctionality(unittest.TestCase):
    """
    This class tests that some simple methods run without errors.
    """

    def test_plotCST(self):
        DVGeometryCST.plotCST(np.ones(4), np.ones(3), 1e-3, 1e-3)

    def test_print(self):
        nUpper = 5
        nLower = 3
        self.DVGeo = DVGeometryCST(os.path.join(inputDir, "naca2412.dat"), numCST=[nUpper, nLower])

        self.DVGeo.addDV("upper", dvType="upper")
        self.DVGeo.addDV("lower", dvType="lower")
        self.DVGeo.addDV("n1", dvType="n1")
        self.DVGeo.addDV("n2", dvType="n2")
        self.DVGeo.addDV("chord", dvType="chord")
        self.DVGeo.printDesignVariables()

    def test_getNDV(self):
        nUpper = 5
        nLower = 3
        nOther = 3  # N1, N2, and chord
        self.DVGeo = DVGeometryCST(os.path.join(inputDir, "naca2412.dat"), numCST=[nUpper, nLower])

        self.DVGeo.addDV("upper", dvType="upper")
        self.DVGeo.addDV("lower", dvType="lower")
        self.DVGeo.addDV("n1", dvType="n1")
        self.DVGeo.addDV("n2", dvType="n2")
        self.DVGeo.addDV("chord", dvType="chord")

        self.assertEqual(nUpper + nLower + nOther, self.DVGeo.getNDV())

    def test_getDesignVars(self):
        nUpper = 5
        nLower = 3
        self.DVGeo = DVGeometryCST(os.path.join(inputDir, "naca2412.dat"), numCST=[nUpper, nLower])

        upper = np.full((nUpper,), 0.3)
        lower = 0.1 * np.ones(nLower)
        N1 = np.array([0.4])
        N2_lower = np.array([1.2])
        chord = np.array([0.5])

        self.DVGeo.addDV("upper", dvType="upper", default=upper)
        self.DVGeo.addDV("lower", dvType="lower")
        self.DVGeo.addDV("n1", dvType="n1")
        self.DVGeo.addDV("n2_lower", dvType="n2_lower")
        self.DVGeo.addDV("chord", dvType="chord")

        DVs = {
            "upper": upper,
            "lower": lower,
            "n1": N1,
            "n2_lower": N2_lower,
            "chord": chord,
        }
        self.DVGeo.setDesignVars(DVs)

        valDVs = self.DVGeo.getDesignVars()

        for dvName in DVs.keys():
            np.testing.assert_array_equal(DVs[dvName], valDVs[dvName])

    def test_getVarNames(self):
        self.DVGeo = DVGeometryCST(os.path.join(inputDir, "naca2412.dat"))

        dvNames = ["amy", "joesph", "maryann", "tobysue", "sir blue bus"]

        self.DVGeo.addDV(dvNames[0], dvType="upper")
        self.DVGeo.addDV(dvNames[1], dvType="lower")
        self.DVGeo.addDV(dvNames[2], dvType="n1")
        self.DVGeo.addDV(dvNames[3], dvType="n2_lower")
        self.DVGeo.addDV(dvNames[4], dvType="chord")

        names = self.DVGeo.getVarNames()

        for name in names:
            self.assertTrue(name in dvNames)


@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
class TestErrorChecking(unittest.TestCase):
    def setUp(self):
        self.DVGeo = DVGeometryCST(os.path.join(inputDir, "naca2412.dat"), numCST=4)

    def test_addPointSet_min_out_of_bounds(self):
        points = np.array(
            [
                [0.5, 0.1, 4.0],
                [-0.5, 0.1, 4.0],
            ]
        )
        with self.assertRaises(ValueError):
            self.DVGeo.addPointSet(points, "bjork")

    def test_addPointSet_max_out_of_bounds(self):
        points = np.array(
            [
                [0.5, 0.1, 4.0],
                [1.5, 0.1, 4.0],
            ]
        )
        with self.assertRaises(ValueError):
            self.DVGeo.addPointSet(points, "jacobo")

    def test_addDV_invalid_type(self):
        with self.assertRaises(ValueError):
            self.DVGeo.addDV("samantha", dvType="this is an invalid type")

    def test_addDV_duplicate_n1(self):
        self.DVGeo.addDV("silver baboon", dvType="n1", default=np.array([0.4]))
        with self.assertRaises(ValueError):
            self.DVGeo.addDV("panda express", dvType="n1_upper")

    def test_addDV_duplicate_n1_reverse(self):
        self.DVGeo.addDV("candace", dvType="n1_lower")
        with self.assertRaises(ValueError):
            self.DVGeo.addDV("richard", dvType="n1")

    def test_addDV_duplicate_n2(self):
        self.DVGeo.addDV("harry", dvType="n2")
        with self.assertRaises(ValueError):
            self.DVGeo.addDV("bobby", dvType="n2_upper")

    def test_addDV_duplicate_n2_reverse(self):
        self.DVGeo.addDV("bob haimes", dvType="n2_lower")
        with self.assertRaises(ValueError):
            self.DVGeo.addDV("hannah", dvType="n2")

    def test_addDV_duplicate_same_type(self):
        self.DVGeo.addDV("ali", dvType="upper")
        with self.assertRaises(ValueError):
            self.DVGeo.addDV("anil", dvType="upper")

    def test_addDV_duplicate_same_name(self):
        self.DVGeo.addDV("josh", dvType="upper")
        with self.assertRaises(ValueError):
            self.DVGeo.addDV("josh", dvType="lower")

    def test_addDV_duplicate_invalid_default_type(self):
        with self.assertRaises(ValueError):
            self.DVGeo.addDV("timo", dvType="chord", default=5.0)

    def test_addDV_duplicate_invalid_default_size(self):
        with self.assertRaises(ValueError):
            self.DVGeo.addDV("brick", dvType="upper", default=np.array([5.0, 1]))

    def test_setDesignVars_invalid_shape(self):
        self.DVGeo.addDV("mafa", dvType="upper")
        with self.assertRaises(ValueError):
            self.DVGeo.setDesignVars({"mafa": np.array([1.0, 3.0])})


if __name__ == "__main__":
    unittest.main()
