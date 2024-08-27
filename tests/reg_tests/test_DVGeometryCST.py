"""
==============================================================================
DVGeometryCST: Test suite for the DVGeometryCST module.
==============================================================================
"""

# Standard Python modules
import os
import unittest

# External modules
from baseclasses import BaseRegTest
from mpi4py import MPI
import numpy as np
from parameterized import parameterized_class

try:
    # External modules
    from prefoil.utils import readCoordFile

    prefoilImported = True
    # First party modules
    from pygeo import DVGeometryCST
except ImportError:
    prefoilImported = False

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
]


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
    N_PROCS = 4

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
        if idxStart:
            idxStart = idxStart.item()
        else:
            idxStart = None
        if idxEnd:
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

        DVs = self.DVGeo.getValues()

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

        DVs = self.DVGeo.getValues()

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

        DVs = self.DVGeo.getValues()

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

    def test_getValues(self):
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

        valDVs = self.DVGeo.getValues()

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
