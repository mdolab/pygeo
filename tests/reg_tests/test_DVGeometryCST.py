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
from prefoil.utils import readCoordFile
from mpi4py import MPI

# ==============================================================================
# Extension modules
# ==============================================================================
from pygeo import DVGeometryCST

# LEUpper is true if the leading edge (minimum x) point is considered to be on the upper surface
airfoils = [
    {"fName": "naca2412.dat", "LEUpper": False},
    {"fName": "naca0012.dat", "LEUpper": True},
    {"fName": "e63.dat", "LEUpper": False}
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
        self.curDir = os.path.abspath(os.path.dirname(__file__))

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
            y0 = DVGeometryCST.computeCSTCoordinates(self.x, N1, N2, w, self.yte)
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
            y0 = DVGeometryCST.computeCSTCoordinates(self.x, N1, N2, w, self.yte)
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
        coords = readCoordFile(os.path.join(self.curDir, self.fName))
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(0, idxLE + self.LEUpper)
        idxLower = np.arange(idxLE + self.LEUpper, coords.shape[0])
        yTE = coords[0, 1]
        N1 = 0.5
        N2 = 1.0

        for nCST in range(2, 10):
            # Fit the CST parameters and then compute the coordinates
            # with those parameters and check that it's close
            upperCST = DVGeometryCST.computeCSTfromCoords(coords[idxUpper, 0], coords[idxUpper, 1], nCST, N1=N1, N2=N2)
            lowerCST = DVGeometryCST.computeCSTfromCoords(coords[idxLower, 0], coords[idxLower, 1], nCST, N1=N1, N2=N2)
            fitCoordsUpper = DVGeometryCST.computeCSTCoordinates(coords[idxUpper, 0], N1, N2, upperCST, yTE)
            fitCoordsLower = DVGeometryCST.computeCSTCoordinates(coords[idxLower, 0], N1, N2, lowerCST, -yTE)

            # Loosen the tolerances for the challenging e63 airfoil
            if self.fName == "e63.dat":
                if nCST < 4:
                    atol = 1e-1
                    rtol = 1.
                else:
                    atol = 1e-2
                    rtol = 6e-1
            else:
                atol = 1e-3
                rtol = 1e-1

            np.testing.assert_allclose(fitCoordsUpper, coords[idxUpper, 1], atol=atol, rtol=rtol)
            np.testing.assert_allclose(fitCoordsLower, coords[idxLower, 1], atol=atol, rtol=rtol)


@parameterized_class(airfoils)
class DVGeometryCSTPointSetSerial(unittest.TestCase):
    # Test in serial
    N_PROCS = 1

    def setUp(self):
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.datFile = os.path.join(self.curDir, self.fName)
        self.comm = MPI.COMM_WORLD
        self.DVGeo = DVGeometryCST(self.datFile, comm=self.comm)

    def test_addPointSet_sorted(self):
        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(self.datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        idxLE = np.argmin(coords[:, 0])
        idxUpper = np.arange(0, idxLE + self.LEUpper)
        idxLower = np.arange(idxLE + self.LEUpper, coords.shape[0])
        thickTE = coords[0, 1] - coords[-1, 1]

        self.DVGeo.addPointSet(coords, "test")

        np.testing.assert_equal(idxUpper, self.DVGeo.points["test"]["upper"])
        np.testing.assert_equal(idxLower, self.DVGeo.points["test"]["lower"])
        np.testing.assert_equal(thickTE, self.DVGeo.points["test"]["thicknessTE"])
        self.assertEqual(min(coords[:, 0]), self.DVGeo.points["test"]["xMin"])
        self.assertEqual(max(coords[:, 0]), self.DVGeo.points["test"]["xMax"])

    def test_addPointSet_randomized(self):
        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(self.datFile)
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
        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(self.datFile)
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
        self.datFile = os.path.join(self.curDir, self.fName)
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
        # Read in airfoil coordinates to test with and split up the surfaces
        coords = readCoordFile(self.datFile)
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


@parameterized_class(DVs)
class DVGeometryCSTSensitivity(unittest.TestCase):
    # Test in serial
    N_PROCS = 1

    def setUp(self):
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.datFile = os.path.join(self.curDir, "naca2412.dat")
        self.rng = np.random.default_rng(1)
        self.comm = MPI.COMM_WORLD
        self.DVGeo = DVGeometryCST(self.datFile, comm=self.comm, isComplex=True)

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
        self.DVGeo.addDV(self.dvName, dvType=self.dvName, dvNum=self.dvNum)
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
            sensProd.append(self.DVGeo.totalSensitivityProd({self.dvName: vec}, self.ptName).astype(float))

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
        self.DVGeo.addDV(self.dvName, dvType=self.dvName, dvNum=self.dvNum)
        self.DVGeo.addPointSet(self.coords, self.ptName)

        # Set DV to random values
        self.DVGeo.setDesignVars({self.dvName: self.rng.random(self.dvNum)})
        self.DVGeo.update(self.ptName)

        # dIdXpt of all ones means the total sensitivities will just be the sum of the
        # derivatives at each of the coordianates
        dIdXpt = np.ones_like(self.coords)

        DVs = self.DVGeo.getValues()

        # First compute the analytic ones with the built in function
        sens = self.DVGeo.totalSensitivity(dIdXpt, self.ptName)[self.dvName].astype(float)

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
        self.DVGeo.addDV(self.dvName, dvType=self.dvName, dvNum=self.dvNum)
        self.DVGeo.addPointSet(self.coords, self.ptName)

        # Set DV to random values
        self.DVGeo.setDesignVars({self.dvName: self.rng.random(self.dvNum)})
        self.DVGeo.update(self.ptName)

        # dIdXpt of all ones means the total sensitivities will just be the sum of the
        # derivatives at each of the coordianates
        dIdXpt = np.ones_like(self.coords)
        coeff = np.array([0.1, 0.5, 1.])
        dIdXptVectorized = np.array([coeff[0] * dIdXpt, coeff[1] * dIdXpt, coeff[2] * dIdXpt])

        DVs = self.DVGeo.getValues()

        # First compute the analytic ones with the built in function
        sens = self.DVGeo.totalSensitivity(dIdXptVectorized, self.ptName)[self.dvName].astype(float)

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


if __name__ == "__main__":
    unittest.main()
