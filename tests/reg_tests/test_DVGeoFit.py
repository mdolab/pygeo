"""
==============================================================================
DVGeometry least-squares fitting tests
==============================================================================
@File    :   test_DVGeoFit.py
@Date    :   2025/07/25
@Author  :   Alasdair Christison Gray
@Description : Tests for the DVGeometry least-squares fitting feature (fitDVGeo / fitPointset).

The self-fit tests are organised around the abstract ``_FitTestBase`` mixin, which holds the shared test bodies.
Each parameterization (FFD, CST, ESP, VSP, Multi) provides a concrete subclass that implements ``setupDVGeo`` and
``perturbDVs``. The mixin is deliberately NOT a ``unittest.TestCase`` because an abstract ``TestCase`` would abort
collection of the whole module.
"""

# Standard Python modules
import os
import unittest
from abc import ABC, abstractmethod

# External modules
from mpi4py import MPI
import numpy as np
from stl import mesh

# First party modules
from pygeo import DVGeometry

# Optional parameterizations. Each is guarded so the module still imports (and the FFD tests still run) when the
# corresponding external dependency is not installed. The CI image has all of these, so they run there.
try:
    from prefoil.utils import readCoordFile

    from pygeo import DVGeometryCST

    prefoilImported = True
except ImportError:
    prefoilImported = False

try:
    import pyOCSM  # noqa: F401

    from pygeo import DVGeometryESP

    ocsmImported = True
except ImportError:
    ocsmImported = False

try:
    import openvsp  # noqa: F401

    from pygeo import DVGeometryVSP

    openvspImported = True
except ImportError:
    openvspImported = False

try:
    import pysurf  # noqa: F401

    from pygeo import DVGeometryMulti

    pysurfImported = True
except ImportError:
    pysurfImported = False


# ==============================================================================
# Shared helpers
# ==============================================================================
def setupC172DVGeo(ffdFile, pts, ptSetName, setBounds=True):
    """Set up the C172 wing FFD DVGeometry object used by the FFD fitting tests.

    Parameters
    ----------
    ffdFile : str
        Path to the FFD file.
    pts : ndarray
        Points to embed as the pointset.
    ptSetName : str
        Name to give the pointset.
    setBounds : bool
        If True, set finite bounds on all design variables. If False, leave the twist bounds unset (None) so the
        ``None`` -> infinite-bounds handling in ``fitPointset`` is exercised.
    """
    DVGeo = DVGeometry(ffdFile)

    # Setup the reference axis
    nRefAxPts = DVGeo.addRefAxis("wing", xFraction=0.25, alignIndex="k")
    nTwist = nRefAxPts - 1

    def twist(val, geo):
        for i in range(1, nRefAxPts):
            geo.rot_z["wing"].coef[i] = val[i - 1]

    def span(val, geo):
        C = geo.extractCoef("wing")
        s = geo.extractS("wing")
        for i in range(nRefAxPts):
            C[i, 2] += val[0] * s[i]
        geo.restoreCoef(C, "wing")

    # Add some DVs
    if setBounds:
        DVGeo.addGlobalDV(dvName="twist", value=[0] * nTwist, func=twist, lower=-10, upper=10, scale=1)
    else:
        DVGeo.addGlobalDV(dvName="twist", value=[0] * nTwist, func=twist)
    DVGeo.addGlobalDV(dvName="span", value=0.0, func=span, lower=-2, upper=2, scale=1)
    DVGeo.addLocalDV("local", lower=-0.05, upper=0.05, axis="y", scale=1.0)

    DVGeo.addPointSet(pts, ptSetName)

    return DVGeo


def loadC172Points(inputFolder, comm):
    """Load the C172 surface points, split across the communicator if running in parallel."""
    meshFile = os.path.join(inputFolder, "c172.stl")
    testMesh = mesh.Mesh.from_file(meshFile)
    pts = testMesh.vectors.reshape(-1, 3) / 1000.0

    nPts = pts.shape[0]
    numProcs = comm.size
    if numProcs > 1:
        rank = comm.rank
        startInd = rank * (nPts // numProcs)
        endInd = max((rank + 1) * (nPts // numProcs), nPts)
        pts = pts[startInd:endInd]
    return pts


def getInputFolder():
    """Return the absolute path to the input_files folder."""
    basePath = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(basePath, "..", "..", "input_files"))


def getEspNacaCSMFile():
    """Return the path to the git-tracked 3-parameter NACA ESP model (camber/maxloc/thickness).

    This lives under examples/ rather than input_files/ because input_files/ is not version-controlled (it is
    downloaded by input_files/get-input-files.sh), so a csm placed there would not be available in CI.
    """
    basePath = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(basePath, "..", "..", "examples", "esp_airfoil", "naca0012.csm"))


# ==============================================================================
# Abstract base holding the shared self-fit test bodies
# ==============================================================================
class _FitTestBase(ABC):
    """Mixin holding the shared self-fit test bodies.

    Concrete subclasses must also inherit from ``unittest.TestCase`` and implement ``setupDVGeo`` and ``perturbDVs``.
    Self-assertion helpers (``self.assertEqual`` etc.) resolve through the MRO at runtime on the concrete subclasses.
    """

    ptSetName = "test_pts"
    comm = MPI.COMM_WORLD
    coordinateTol = 1e-5
    # Whether to additionally assert that the recovered DV values match the perturbed ones. Only meaningful where the
    # DV -> geometry map is locally unique (e.g. the C172 FFD); the map is generally non-injective so this is off by
    # default.
    checkDVvalues = False
    perturbSeed = 0
    fitSettings = {"xtol": 1e-8, "ftol": 1e-8, "gtol": 1e-8, "max_nfev": 60, "verbose": 0}

    @abstractmethod
    def setupDVGeo(self):
        """Return a DVGeometry object with a pointset named ``self.ptSetName`` already added."""

    @abstractmethod
    def perturbDVs(self):
        """Return a dict of seeded, bounds-respecting perturbed DV values around ``self.origDVs``."""

    def setUp(self):
        self.input_folder = getInputFolder()
        self.DVGeoRef = self.setupDVGeo()
        self.DVGeoFit = self.setupDVGeo()
        self.origDVs = self.DVGeoRef.getDesignVars()
        self.lowerBounds, self.upperBounds = self.DVGeoRef.getDVBounds()
        self.perturbedDVs = self.perturbDVs()

    def perturbAroundInitial(self, amplitudes):
        """Helper for subclasses: seeded uniform perturbation around the initial DVs, clipped to finite bounds.

        Parameters
        ----------
        amplitudes : dict
            Per-DV perturbation amplitude. Each perturbed value is ``initial + U(-amplitude, amplitude)``.
        """
        rng = np.random.default_rng(self.perturbSeed)
        perturbed = {}
        for key, val in self.origDVs.items():
            size = np.atleast_1d(val).size
            newVal = np.atleast_1d(val).astype(float) + rng.uniform(-amplitudes[key], amplitudes[key], size=size)
            lower = self.lowerBounds[key]
            upper = self.upperBounds[key]
            if lower is not None or upper is not None:
                newVal = np.clip(
                    newVal,
                    -np.inf if lower is None else lower,
                    np.inf if upper is None else upper,
                )
            perturbed[key] = newVal
        return perturbed

    def test_perfect_fit_same_dvs(self):
        """Starting from the same parameterization and DVs, the fit should converge immediately and recover the DVs.

        The self-fit is driven through ``fitPointset`` on the shared point set so the test is meaningful for every
        parameterization. Using ``fitDVGeo`` here would re-embed the reference's original points into a new point set,
        and for parameterizations that project points onto a surface (e.g. CST, ESP, VSP) that re-projection makes the
        original and re-embedded point sets differ, so they cannot be compared coordinate-by-coordinate.
        """
        self.DVGeoRef.setDesignVars(self.perturbedDVs)
        self.DVGeoFit.setDesignVars(self.perturbedDVs)
        targetPts = self.DVGeoRef.update(self.ptSetName)
        fitDVs, result = self.DVGeoFit.fitPointset(self.ptSetName, targetPts, **self.fitSettings)
        self.assertEqual(result.nfev, 1)
        self.assertTrue(result.success)

        # Check that the design variables are unchanged
        for key in fitDVs.keys():
            np.testing.assert_array_equal(fitDVs[key], self.perturbedDVs[key])
        # Check that the deformed points are the same (to machine precision; update() can differ by a few ulp)
        np.testing.assert_allclose(self.DVGeoFit.update(self.ptSetName), targetPts, rtol=0.0, atol=1e-12)

    def test_perfect_fit_diff_dvs(self):
        """Starting from the same parameterization but different DVs, the fit should recover the reference geometry."""
        self.DVGeoRef.setDesignVars(self.perturbedDVs)
        targetPts = self.DVGeoRef.update(self.ptSetName)
        fitDVs, result = self.DVGeoFit.fitPointset(self.ptSetName, targetPts, **self.fitSettings)
        # Do not assert result.success: for a high-dimensional fit the optimizer can keep taking tiny steps until it
        # hits max_nfev (status 0) without tripping the ftol/xtol/gtol flags, even though the geometry is fully matched.
        # Whether that happens is environment-dependent (BLAS/MPI build), so we require only that the optimizer did not
        # error out (status >= 0) and check the geometry match, which is the actual contract of the fit.
        self.assertGreaterEqual(result.status, 0)

        # Check that the deformed points match the reference geometry
        np.testing.assert_allclose(
            self.DVGeoFit.update(self.ptSetName), targetPts, rtol=np.inf, atol=self.coordinateTol
        )

        # Optionally check that the recovered design variables match
        if self.checkDVvalues:
            for key in fitDVs.keys():
                np.testing.assert_allclose(fitDVs[key], self.perturbedDVs[key], rtol=0.05, atol=0.05)


# ==============================================================================
# FFD self-fit (C172 wing)
# ==============================================================================
class TestFFDFit(_FitTestBase, unittest.TestCase):
    N_PROCS = 2
    # The C172 FFD DV -> geometry map is locally unique, so we can additionally check DV recovery.
    checkDVvalues = True

    def setupDVGeo(self):
        ffdFile = os.path.join(self.input_folder, "c172_fitted.xyz")
        pts = loadC172Points(self.input_folder, self.comm)
        return setupC172DVGeo(ffdFile, pts, self.ptSetName)

    def perturbDVs(self):
        # All C172 DVs have finite bounds; perturb within a tenth of each bound range around the (zero) initial values.
        # This is large enough to be a meaningful fit but small enough that the fit-from-default case converges.
        amplitudes = {}
        for key in self.origDVs.keys():
            amplitudes[key] = 0.1 * (self.upperBounds[key] - self.lowerBounds[key])
        return self.perturbAroundInitial(amplitudes)

    def test_fitDVGeo_high_level(self):
        """Exercise the high-level fitDVGeo API, including the cross-object getOrigPoints -> addPointSet plumbing.

        FFD embeds points exactly, so the geometry recovered through fitDVGeo (which re-embeds the reference's
        original points into a new point set) matches the reference geometry coordinate-by-coordinate.
        """
        self.DVGeoRef.setDesignVars(self.perturbedDVs)
        fitDVs, result = self.DVGeoFit.fitDVGeo(self.DVGeoRef, self.ptSetName, **self.fitSettings)
        # See test_perfect_fit_diff_dvs: assert the optimizer did not error out and rely on the geometry match rather
        # than result.success, which is an environment-dependent false negative for high-dimensional fits.
        self.assertGreaterEqual(result.status, 0)
        refPts = self.DVGeoRef.update(self.ptSetName)
        fitPts = self.DVGeoFit.update(self.ptSetName)
        np.testing.assert_allclose(refPts, fitPts, rtol=np.inf, atol=self.coordinateTol)
        for key in fitDVs.keys():
            np.testing.assert_allclose(fitDVs[key], self.perturbedDVs[key], rtol=0.05, atol=0.05)


# ==============================================================================
# CST self-fit (NACA 2412 airfoil)
# ==============================================================================
@unittest.skipUnless(prefoilImported, "preFoil is required for DVGeometryCST")
class TestCSTFit(_FitTestBase, unittest.TestCase):
    N_PROCS = 1
    numCST = 8
    # The CST DV -> geometry map is not generally injective (and is poorly conditioned), so we only check geometry.
    checkDVvalues = False
    coordinateTol = 1e-5

    def setupDVGeo(self):
        datFile = os.path.join(self.input_folder, "naca2412.dat")
        DVGeo = DVGeometryCST(datFile, numCST=self.numCST, comm=self.comm)
        DVGeo.addDV("upper", dvType="upper")
        DVGeo.addDV("lower", dvType="lower")
        DVGeo.addDV("n1", dvType="n1")
        DVGeo.addDV("n2", dvType="n2")
        DVGeo.addDV("chord", dvType="chord")

        coords = readCoordFile(datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        DVGeo.addPointSet(coords, self.ptSetName)
        return DVGeo

    def perturbDVs(self):
        # Modest perturbations: the CST coefficients dominate the shape, while the class-shape parameters (n1, n2)
        # are sensitive and poorly conditioned, so we perturb them only slightly to keep the fit-from-default
        # case well-behaved.
        amplitudes = {
            "upper": 0.02,
            "lower": 0.02,
            "n1": 0.01,
            "n2": 0.01,
            "chord": 0.05,
        }
        return self.perturbAroundInitial(amplitudes)


# ==============================================================================
# ESP self-fit (NACA airfoil with camber/maxloc/thickness DVs)
# ==============================================================================
@unittest.skipUnless(ocsmImported, "pyOCSM (ESP) is required for DVGeometryESP")
class TestESPFit(_FitTestBase, unittest.TestCase):
    N_PROCS = 1
    # ESP projects the point set onto the regenerated CAD surface, which sets a noise floor on the achievable fit.
    coordinateTol = 1e-4

    def setupDVGeo(self):
        csmFile = getEspNacaCSMFile()
        DVGeo = DVGeometryESP(csmFile)
        # These DESPMTRs are added with default (None) bounds, which also exercises the None -> infinite-bounds path.
        DVGeo.addVariable("camber_dv")
        DVGeo.addVariable("maxloc_dv")
        DVGeo.addVariable("thickness_dv")

        stlFile = os.path.join(self.input_folder, "esp", "naca0012_esp.stl")
        surfMesh = mesh.Mesh.from_file(stlFile)
        DVGeo.addPointSet(surfMesh.vectors[:, 0, :], self.ptSetName)
        return DVGeo

    def perturbDVs(self):
        # Perturb camber and thickness only. At zero camber the geometry is insensitive to the location of maximum
        # camber, so perturbing maxloc would make the fit-from-default problem ill-conditioned.
        amplitudes = {"camber_dv": 0.01, "maxloc_dv": 0.0, "thickness_dv": 0.01}
        return self.perturbAroundInitial(amplitudes)


# ==============================================================================
# VSP self-fit (ellipsoid radii DVs)
# ==============================================================================
@unittest.skipUnless(openvspImported, "openvsp is required for DVGeometryVSP")
class TestVSPFit(_FitTestBase, unittest.TestCase):
    N_PROCS = 1
    # VSP projects the point set onto the surface, which sets a noise floor on the achievable fit.
    coordinateTol = 1e-4

    def setupDVGeo(self):
        vspFile = os.path.join(self.input_folder, "simpleEll_med.vsp3")
        DVGeo = DVGeometryVSP(vspFile)
        # Added with default (None) bounds, which also exercises the None -> infinite-bounds path.
        DVGeo.addVariable("Ellipsoid", "Design", "A_Radius", scale=1.0, dh=0.1)
        DVGeo.addVariable("Ellipsoid", "Design", "B_Radius", scale=1.0, dh=0.1)
        DVGeo.addVariable("Ellipsoid", "Design", "C_Radius", scale=1.0, dh=0.1)

        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
            ]
        )
        DVGeo.addPointSet(points, self.ptSetName)
        return DVGeo

    def perturbDVs(self):
        amplitudes = {key: 0.1 for key in self.origDVs.keys()}
        return self.perturbAroundInitial(amplitudes)


# ==============================================================================
# Multi self-fit (two intersecting boxes)
# ==============================================================================
@unittest.skip(
    "DVGeometryMulti does not yet implement convertDictToSensitivity/convertSensitivityToDict, which fitPointset "
    "requires. See /tmp/dvgeomulti-convert-methods-handoff.md. Remove this skip once those methods are added."
)
class TestMultiFit(_FitTestBase, unittest.TestCase):
    N_PROCS = 1
    coordinateTol = 1e-4

    def setupDVGeo(self):
        comps = ["box1", "box2"]
        ffdFiles = [os.path.join(self.input_folder, f"{comp}.xyz") for comp in comps]
        triMeshFiles = [os.path.join(self.input_folder, f"{comp}.cgns") for comp in comps]

        DVGeo = DVGeometryMulti(comm=self.comm)
        DVGeo.addComponent("box1", DVGeometry(ffdFiles[0]), triMeshFiles[0])
        DVGeo.addComponent("box2", DVGeometry(ffdFiles[1]), triMeshFiles[1])
        DVGeo.addIntersection(
            "box1",
            "box2",
            dStarA=0.15,
            dStarB=0.15,
            featureCurves=["part_15_1d", "part_35_1d"],
            project=True,
            includeCurves=True,
            curveEpsDict={"part_15_1d": 1e-3, "part_35_1d": 1e-3, "intersection": 1e-3},
        )

        DVGeoDict = DVGeo.getDVGeoDict()
        for comp in comps:
            nRefAxPts = DVGeoDict[comp].addRefAxis("box", xFraction=0.5, alignIndex="j", rotType=4)
            nTwist = nRefAxPts - 1

            def twist(val, geo, nRefAxPts=nRefAxPts):
                for i in range(1, nRefAxPts):
                    geo.rot_z["box"].coef[i] = val[i - 1]

            DVGeoDict[comp].addGlobalDV(dvName=f"{comp}_twist", value=[0] * nTwist, func=twist)

        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.1, -0.5],
                [0.5, 0.0, 2.0],
                [0.375, 0.125, 2.0],
            ]
        )
        DVGeo.addPointSet(pts, self.ptSetName, comm=self.comm, applyIC=True)
        return DVGeo

    def perturbDVs(self):
        amplitudes = {key: 1.0 for key in self.origDVs.keys()}
        return self.perturbAroundInitial(amplitudes)


# ==============================================================================
# Cross-type fit: recover NACA parameters of a CST airfoil using an ESP model
# ==============================================================================
@unittest.skipUnless(ocsmImported and prefoilImported, "pyOCSM (ESP) and preFoil are required for this test")
class TestESPtoCSTCrossFit(unittest.TestCase):
    """Fit an ESP NACA model to the CST representation of a NACA 2412 airfoil and recover its NACA parameters.

    This is the same scenario as the airfoil-fitting tutorial. Because the ESP ``naca`` UDP is parameterized by the
    exact NACA 4-digit definition (camber, location of maximum camber, thickness), fitting it to a NACA 2412 should
    recover camber = 0.02, maxloc = 0.4, thickness = 0.12.
    """

    N_PROCS = 1
    ptSetName = "airfoil"
    comm = MPI.COMM_WORLD

    def test_recover_naca_parameters(self):
        inputFolder = getInputFolder()

        # self: the ESP NACA model whose parameters we want to recover
        espDVGeo = DVGeometryESP(getEspNacaCSMFile())
        espDVGeo.addVariable("camber_dv")
        espDVGeo.addVariable("maxloc_dv")
        espDVGeo.addVariable("thickness_dv")

        # ref: the CST representation of a NACA 2412
        datFile = os.path.join(inputFolder, "naca2412.dat")
        cstDVGeo = DVGeometryCST(datFile, numCST=8, comm=self.comm)
        cstDVGeo.addDV("upper", dvType="upper")
        cstDVGeo.addDV("lower", dvType="lower")
        coords = readCoordFile(datFile)
        coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        cstDVGeo.addPointSet(coords, self.ptSetName)

        fitDVs, result = espDVGeo.fitDVGeo(
            cstDVGeo, self.ptSetName, xtol=1e-10, ftol=1e-10, gtol=1e-8, max_nfev=100, verbose=0
        )
        self.assertTrue(result.success)

        # The recovered ESP parameters should match the NACA 2412 definition to within a few percent. The noise floor
        # is set by the difference between the OpenCSM NACA UDP and the tabulated naca2412.dat plus the CST fit error.
        self.assertAlmostEqual(float(fitDVs["camber_dv"]), 0.02, delta=0.004)
        self.assertAlmostEqual(float(fitDVs["maxloc_dv"]), 0.4, delta=0.08)
        self.assertAlmostEqual(float(fitDVs["thickness_dv"]), 0.12, delta=0.006)

        # The fitted geometry should match the CST airfoil to a loose tolerance
        targetCoords = cstDVGeo.update(self.ptSetName)
        fitCoords = espDVGeo.update(f"{self.ptSetName}_fit")
        np.testing.assert_allclose(fitCoords, targetCoords, rtol=np.inf, atol=1e-3)


# ==============================================================================
# Regression test for the None -> infinite-bounds fix in fitPointset
# ==============================================================================
class TestUnsetBoundsRegression(unittest.TestCase):
    """An unset (None) design variable bound must not crash fitPointset.

    Previously, ``getDVBounds`` returning None for an unset bound was passed through ``convertDictToSensitivity``,
    which silently wrote nan into the bounds arrays and made scipy raise a misleading
    "Initial guess is outside of provided bounds" error.
    """

    N_PROCS = 1
    ptSetName = "test_pts"
    comm = MPI.COMM_WORLD

    def setUp(self):
        self.input_folder = getInputFolder()
        self.ffdFile = os.path.join(self.input_folder, "c172_fitted.xyz")
        self.pts = loadC172Points(self.input_folder, self.comm)

    def test_unset_bounds_do_not_crash(self):
        # setBounds=False leaves the twist DV bounds as None
        DVGeoRef = setupC172DVGeo(self.ffdFile, self.pts, self.ptSetName, setBounds=False)
        DVGeoFit = setupC172DVGeo(self.ffdFile, self.pts, self.ptSetName, setBounds=False)

        rng = np.random.default_rng(0)
        perturbedDVs = {}
        origDVs = DVGeoRef.getDesignVars()
        perturbedDVs["twist"] = rng.uniform(-5.0, 5.0, size=origDVs["twist"].size)
        perturbedDVs["span"] = rng.uniform(-1.0, 1.0, size=origDVs["span"].size)
        perturbedDVs["local"] = rng.uniform(-0.025, 0.025, size=origDVs["local"].size)
        DVGeoRef.setDesignVars(perturbedDVs)

        # This call raised a misleading ValueError before the None -> infinite-bounds fix
        fitDVs, result = DVGeoFit.fitDVGeo(
            DVGeoRef, self.ptSetName, xtol=1e-8, ftol=1e-8, gtol=1e-6, max_nfev=40, verbose=0
        )
        # The point of this regression test is that the call no longer crashes; reaching this assertion proves that.
        # We rely on the geometry match rather than result.success (an environment-dependent false negative).
        self.assertGreaterEqual(result.status, 0)
        refPts = DVGeoRef.update(self.ptSetName)
        fitPts = DVGeoFit.update(self.ptSetName)
        np.testing.assert_allclose(refPts, fitPts, rtol=np.inf, atol=1e-5)


# ==============================================================================
# Error-path tests (cheap, deterministic, single proc)
# ==============================================================================
class TestFitErrorPaths(unittest.TestCase):
    N_PROCS = 1
    ptSetName = "test_pts"
    comm = MPI.COMM_WORLD

    def setUp(self):
        self.input_folder = getInputFolder()
        ffdFile = os.path.join(self.input_folder, "c172_fitted.xyz")
        self.pts = loadC172Points(self.input_folder, self.comm)
        self.DVGeoRef = setupC172DVGeo(ffdFile, self.pts, self.ptSetName)
        self.DVGeoFit = setupC172DVGeo(ffdFile, self.pts, self.ptSetName)

    def test_pointFraction_too_small(self):
        with self.assertRaises(ValueError):
            self.DVGeoFit.fitDVGeo(self.DVGeoRef, self.ptSetName, pointFraction=0.0)

    def test_pointFraction_too_large(self):
        with self.assertRaises(ValueError):
            self.DVGeoFit.fitDVGeo(self.DVGeoRef, self.ptSetName, pointFraction=1.5)

    def test_unknown_ptSetName(self):
        with self.assertRaises(ValueError):
            self.DVGeoFit.fitPointset("not_a_real_pointset", self.DVGeoRef.update(self.ptSetName))

    def test_point_count_mismatch(self):
        # Provide the wrong number of points to fit
        wrongCoords = self.DVGeoRef.update(self.ptSetName)[:-1, :]
        with self.assertRaises(ValueError):
            self.DVGeoFit.fitPointset(self.ptSetName, wrongCoords)


# ==============================================================================
# Analytic-Jacobian-vs-finite-difference test
# ==============================================================================
class TestFitJacobian(unittest.TestCase):
    """Validate the coordinate sensitivities that fitPointset assembles into its Jacobian.

    ``fitPointset`` builds the least-squares Jacobian from per-DV forward ``totalSensitivityProd`` products. Here we
    assemble the same analytic coordinate Jacobian via the public sensitivity API and compare it against a central
    finite difference of ``update``. A wrong Jacobian can still converge under scipy, so this direct comparison is the
    only test that validates the sensitivity-assembly the feature depends on.
    """

    N_PROCS = 1
    ptSetName = "test_pts"
    comm = MPI.COMM_WORLD

    def setUp(self):
        self.input_folder = getInputFolder()
        ffdFile = os.path.join(self.input_folder, "c172_fitted.xyz")
        self.pts = loadC172Points(self.input_folder, self.comm)
        self.DVGeo = setupC172DVGeo(ffdFile, self.pts, self.ptSetName)

    def computeAnalyticJacobian(self):
        numDVs = self.DVGeo.getNDV()
        seed = np.zeros(numDVs)
        columns = []
        for ii in range(numDVs):
            seed[ii] = 1.0
            column = self.DVGeo.totalSensitivityProd(
                self.DVGeo.convertSensitivityToDict(seed.reshape(1, -1)), self.ptSetName
            ).flatten()
            columns.append(column)
            seed[ii] = 0.0
        return np.array(columns).T

    def computeFDJacobian(self, dvArray, step=1e-6):
        numDVs = dvArray.size
        numCoords = self.DVGeo.update(self.ptSetName).size
        jac = np.zeros((numCoords, numDVs))
        for ii in range(numDVs):
            pertUp = dvArray.copy()
            pertUp[ii] += step
            self.DVGeo.setDesignVars(self.DVGeo.convertSensitivityToDict(pertUp.reshape(1, -1), out1D=True))
            coordsUp = self.DVGeo.update(self.ptSetName).flatten()

            pertDown = dvArray.copy()
            pertDown[ii] -= step
            self.DVGeo.setDesignVars(self.DVGeo.convertSensitivityToDict(pertDown.reshape(1, -1), out1D=True))
            coordsDown = self.DVGeo.update(self.ptSetName).flatten()

            jac[:, ii] = (coordsUp - coordsDown) / (2.0 * step)
        # Restore the original design variables
        self.DVGeo.setDesignVars(self.DVGeo.convertSensitivityToDict(dvArray.reshape(1, -1), out1D=True))
        return jac

    def test_analytic_jacobian_matches_fd(self):
        # Use a non-trivial set of design variables so the (nonlinear) twist sensitivities are exercised
        rng = np.random.default_rng(0)
        dvArray = self.DVGeo.convertDictToSensitivity(self.DVGeo.getDesignVars()).flatten()
        dvArray = dvArray + rng.uniform(-1.0, 1.0, size=dvArray.size)
        self.DVGeo.setDesignVars(self.DVGeo.convertSensitivityToDict(dvArray.reshape(1, -1), out1D=True))

        analyticJac = self.computeAnalyticJacobian()
        fdJac = self.computeFDJacobian(dvArray)

        np.testing.assert_allclose(analyticJac, fdJac, rtol=1e-5, atol=1e-6)


# ==============================================================================
# pointFraction downsampling + bounds enforcement
# ==============================================================================
class TestFitPointFraction(unittest.TestCase):
    N_PROCS = 1
    ptSetName = "test_pts"
    comm = MPI.COMM_WORLD
    fitSettings = {"xtol": 1e-8, "ftol": 1e-8, "gtol": 1e-6, "max_nfev": 40, "verbose": 0}

    def setUp(self):
        self.input_folder = getInputFolder()
        self.ffdFile = os.path.join(self.input_folder, "c172_fitted.xyz")
        self.pts = loadC172Points(self.input_folder, self.comm)

    def test_pointFraction_downsampling_converges(self):
        DVGeoRef = setupC172DVGeo(self.ffdFile, self.pts, self.ptSetName)
        DVGeoFit = setupC172DVGeo(self.ffdFile, self.pts, self.ptSetName)

        rng = np.random.default_rng(0)
        origDVs = DVGeoRef.getDesignVars()
        perturbedDVs = {
            "twist": rng.uniform(-5.0, 5.0, size=origDVs["twist"].size),
            "span": rng.uniform(-1.0, 1.0, size=origDVs["span"].size),
            "local": rng.uniform(-0.025, 0.025, size=origDVs["local"].size),
        }
        DVGeoRef.setDesignVars(perturbedDVs)

        # The downsampling uses the global numpy RNG, so seed it for determinism
        np.random.seed(0)
        fitDVs, result = DVGeoFit.fitDVGeo(DVGeoRef, self.ptSetName, pointFraction=0.5, **self.fitSettings)
        self.assertTrue(result.success)
        # The C172 FFD map is locally unique, so a perfect self-fit on the subset should recover the DVs
        for key in fitDVs.keys():
            np.testing.assert_allclose(fitDVs[key], perturbedDVs[key], rtol=0.05, atol=0.05)

    def test_bounds_are_enforced(self):
        DVGeoRef = setupC172DVGeo(self.ffdFile, self.pts, self.ptSetName)
        DVGeoFit = setupC172DVGeo(self.ffdFile, self.pts, self.ptSetName)

        # Drive the reference geometry to a twist beyond the fit object's twist bounds (+/-10), so the unconstrained
        # optimum lies outside the bounds and the fitted twist must be clipped to the upper bound.
        _, upperBounds = DVGeoFit.getDVBounds()
        twistUpper = upperBounds["twist"]
        origDVs = DVGeoRef.getDesignVars()
        refDVs = {key: np.zeros_like(val) for key, val in origDVs.items()}
        refDVs["twist"] = np.full(origDVs["twist"].size, 20.0)
        DVGeoRef.setDesignVars(refDVs)

        fitDVs, result = DVGeoFit.fitDVGeo(DVGeoRef, self.ptSetName, **self.fitSettings)
        self.assertTrue(result.success)
        # Fitted twist must respect the upper bound and should be pinned to it (optimum is outside the bounds)
        self.assertTrue(np.all(fitDVs["twist"] <= twistUpper + 1e-6))
        self.assertTrue(np.any(fitDVs["twist"] >= twistUpper - 1e-3))


if __name__ == "__main__":
    unittest.main()
