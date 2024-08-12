import unittest
import os
import numpy as np
from parameterized import parameterized_class
from stl import mesh

import commonUtils
from pygeo.mphys import OM_DVGEOCOMP
from pyspline import Curve

try:
    from openmdao.api import IndepVarComp, Problem, Group
    from openmdao.utils.assert_utils import assert_near_equal, assert_check_totals

    omInstalled = True

except ImportError:
    omInstalled = False

try:
    # External modules
    import pyOCSM  # noqa

    # First party modules
    from pygeo import DVGeometryESP

    ocsmInstalled = True

except ImportError:
    ocsmInstalled = False

# input files for all DVGeo types
input_path = os.path.dirname(os.path.abspath(__file__))
parentFFDFile = os.path.join(input_path, "../../input_files/outerBoxFFD.xyz")
childFFDFile = os.path.join(input_path, "../../input_files/simpleInnerFFD.xyz")
espBox = os.path.join(input_path, "../input_files/esp/box.csm")

# parameters for FFD-based DVGeo tests
childName = "childFFD"
globalDVFuncParamsParent = ["mainX", -1.0, commonUtils.mainAxisPoints]
localDVFuncParamsParent = ["xdir"]
globalDVFuncParamsChild = ["nestedX", -0.5, commonUtils.childAxisPoints, childName]

globalDVParent = {"funcName": "nom_addGlobalDV", "funcParams": globalDVFuncParamsParent, "lower": -1.0, "upper": 0.0, "val": -1.0}
localDVParent = {"funcName": "nom_addLocalDV", "funcParams": localDVFuncParamsParent, "lower": -1.0, "upper": 1.0, "val": 12*[0.0]}
globalDVChild = {"funcName": "nom_addGlobalDV", "funcParams": globalDVFuncParamsChild, "lower": -1.0, "upper": 0.0, "val": -1.0}

ffd_test_params = [
    {"name": "MPhys_FFD_oneFFD_global", "dvInfo": [globalDVParent]},
    {"name": "MPhys_FFD_oneFFD_local", "dvInfo": [localDVParent]},
    {"name": "MPhys_FFD_childFFD_global", "dvInfo": [globalDVParent, globalDVChild]},
]

# DVConstraints functionals to test
test_params_constraints_box = [
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {"name": "func", "ptList": [[0.0, 0.0, 0.1], [0.0, 0.0, 7.9]], "nCon": 3, "axis": [1, 0, 0], "scaled": False},
        "valCheck": 2 * np.ones(3),
        "valTol": 1e-4,
    },
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {"name": "func", "ptList": [[0.0, 0.0, 0.1], [0.0, 0.0, 7.9]], "nCon": 5, "axis": [0, 1, 0], "scaled": False},
        "valCheck": np.ones(5),
        "valTol": 3e-5,
    },
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {"name": "func", "ptList": [[-0.5, 0.0, 4.0], [0.5, 0.0, 4.0]], "nCon": 5, "axis": [0, 0, 1], "scaled": False},
        "valCheck": 8 * np.ones(5),
    },
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {"name": "func", "ptList": [[0.0, 0.0, 0.1], [0.0, 0.0, 7.9]], "nCon": 3, "axis": [1, 0, 0], "scaled": False, "projected": True},
        "valCheck": 2 * np.ones(3),
        "valTol": 2e-4,
    },
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {"name": "func", "ptList": [[0.0, 0.0, 0.1], [0.0, 0.0, 7.9]], "nCon": 5, "axis": [0, 1, 0], "scaled": False, "projected": True},
        "valCheck": np.ones(5),
        "valTol": 2e-4,
    },
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {"name": "func", "ptList": [[-0.5, 0.0, 4.0], [0.5, 0.0, 4.0]], "nCon": 5, "axis": [0, 0, 1], "scaled": False, "projected": True},
        "valCheck": 8 * np.ones(5),
    },
    {
        "conFunc": "nom_addThicknessConstraints2D",
        "kwargs": {"name": "func", "leList": [[-0.25, 0.0, 0.1], [-0.25, 0.0, 7.9]], "teList": [[0.75, 0.0, 0.1], [0.75, 0.0, 7.9]], "nSpan": 2, "nChord": 3, "scaled": False},
        "valCheck": np.ones(6),
    },
    {
        "conFunc": "nom_addThicknessConstraints2D",
        "kwargs": {"name": "func", "leList": [[0.0, -0.25, 0.1], [0.0, -0.25, 7.9]], "teList": [[0.0, 0.25, 0.1], [0.0, 0.25, 7.9]], "nSpan": 2, "nChord": 3, "scaled": False},
        "valCheck": 2 * np.ones(6),
    },
    {
        "conFunc": "nom_addThicknessConstraints2D",
        "kwargs": {"name": "func", "leList": [[-0.5, -0.25, 0.1], [0.5, -0.25, 0.1]], "teList": [[-0.5, 0.25, 0.1], [0.5, 0.25, 0.1]], "nSpan": 2, "nChord": 3, "scaled": False},
        "valCheck": 8 * np.ones(6),
    },
    {
        "conFunc": "nom_addThicknessConstraints2D",
        "kwargs": {"name": "func", "leList": [[-0.5, -0.25, 0.1], [0.5, -0.25, 0.1]], "teList": [[-0.5, 0.25, 0.1], [0.5, 0.25, 0.1]], "nSpan": 2, "nChord": 3, "scaled": False, "projected": True},
        "valCheck": 8 * np.ones(6),
    },
    {
        "conFunc": "nom_addVolumeConstraint",
        "kwargs": {"name": "func", "leList": [[-0.5, -0.25, 0.1], [0.5, -0.25, 0.1]], "teList": [[-0.5, 0.25, 0.1], [0.5, 0.25, 0.1]], "nSpan": 4, "nChord": 4, "scaled": False},
        "valCheck": 4.0,
        "valTol": 1e-4,
    },
    {"conFunc": "nom_addSurfaceAreaConstraint", "kwargs": {"name": "func", "scaled": False}, "valCheck": 52.0},
    {"conFunc": "nom_addProjectedAreaConstraint", "kwargs": {"name": "func", "axis": "x", "scaled": False}, "valCheck": 8.0, "valTol": 3e-2},
    {"conFunc": "nom_addProjectedAreaConstraint", "kwargs": {"name": "func", "axis": "y", "scaled": False}, "valCheck": 16.0, "valTol": 3e-2},
    {"conFunc": "nom_addProjectedAreaConstraint", "kwargs": {"name": "func", "axis": "z", "scaled": False}, "valCheck": 2.0, "valTol": 3e-2},
]

@unittest.skipUnless(omInstalled, "OpenMDAO is required to test the pyGeo MPhys wrapper")
@parameterized_class(ffd_test_params)
class TestDVGeoMPhysFFD(unittest.TestCase):
    def setUp(self):
        # give the OM Group access to the test case attributes
        dvInfo = self.dvInfo

        class FFDGroup(Group):
            def setup(self):
                self.add_subsystem("dvs", IndepVarComp(), promotes=["*"])
                self.add_subsystem("geometry", OM_DVGEOCOMP(file=parentFFDFile, type="ffd"))

            def configure(self):
                self.geometry.nom_addChild(childFFDFile, childName=childName)

                points = np.zeros([2, 3])
                points[0, :] = [0.25, 0, 0]
                points[1, :] = [-0.25, 0, 0]
                ptName = "testPoints"
                self.geometry.nom_addPointSet(points.flatten(), ptName)

                # create a reference axis for the parent
                axisPoints = [[-1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
                c1 = Curve(X=axisPoints, k=2)
                self.geometry.nom_addRefAxis("mainAxis", curve=c1, axis="y")

                # create a reference axis for the child
                axisPoints = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
                c1 = Curve(X=axisPoints, k=2)
                self.geometry.nom_addRefAxis("nestedAxis", childName=childName, curve=c1, axis="y")

                for dv in dvInfo:
                    dvName = dv["funcParams"][0]
                    getattr(self.geometry, dv["funcName"])(*dv["funcParams"])
                    
                    self.dvs.add_output(dvName, dv["val"])
                    self.connect(dvName, f"geometry.{dvName}")
                    self.add_design_var(dvName, upper=dv["upper"], lower=dv["lower"])

                self.add_constraint(f"geometry.{ptName}")

        self.prob = Problem(model=FFDGroup())

    def test_run_model(self):
        self.prob.setup()
        self.prob.run_model()

    def test_deriv_fwd(self):
        self.prob.setup(mode="fwd")
        self.prob.run_model()

        totals = self.prob.check_totals(step=1e-7, out_stream=None)
        assert_check_totals(totals)

    def test_deriv_rev(self):
        self.prob.setup(mode="rev")
        self.prob.run_model()

        totals = self.prob.check_totals(step=1e-7, out_stream=None)
        assert_check_totals(totals)


@unittest.skipUnless(omInstalled, "OpenMDAO is required to test the pyGeo MPhys wrapper")
@parameterized_class(test_params_constraints_box)
class TestDVConMPhysBox(unittest.TestCase):
    def setUp(self):
        # Random number generator
        self.rand = np.random.default_rng(1)

    def get_box_prob(self):
        """
        Generate an OpenMDAO problem with the OM_DVGEOCOMP component with the
        functional dictated by the parameterized class.
        """
        # Parameterized values
        conFunc = self.conFunc
        kwargs = self.kwargs

        meshFile = os.path.join(input_path, "../../input_files/2x1x8_rectangle.stl")
        ffdFile = os.path.join(input_path, "../../input_files/2x1x8_rectangle.xyz")
        xFraction = 0.5
        meshScale = 1.0

        class BoxGeo(Group):
            def setup(self):
                self.geo = self.add_subsystem("geometry", OM_DVGEOCOMP(file=ffdFile, type="ffd"), promotes=["*"])

            def configure(self):
                # Get the mesh from the STL
                testMesh = mesh.Mesh.from_file(meshFile)
                # dim 0 is triangle index
                # dim 1 is each vertex of the triangle
                # dim 2 is x, y, z dimension

                p0 = testMesh.vectors[:, 0, :] * meshScale
                v1 = testMesh.vectors[:, 1, :] * meshScale - p0
                v2 = testMesh.vectors[:, 2, :] * meshScale - p0
                self.geo.nom_setConstraintSurface([p0, v1, v2], addToDVGeo=False)

                # Add the geometric functional
                getattr(self.geo, conFunc)(**kwargs)

                # Add DVs
                nRefAxPts = self.geo.nom_addRefAxis("wing", xFraction=xFraction, alignIndex="k")
                self.nTwist = nRefAxPts - 1

                def twist(val, geo):
                    for i in range(1, nRefAxPts):
                        geo.rot_z["wing"].coef[i] = val[i - 1]

                self.geo.nom_addGlobalDV(dvName="twist", value=[0] * self.nTwist, func=twist)
                self.geo.nom_addLocalDV("local", axis="y")

                self.add_design_var("twist")
                self.add_design_var("local")
                self.add_objective(kwargs["name"])

        p = Problem(model=BoxGeo())
        return p

    def test_undeformed_vals(self):
        """
        Test the value of the functional on the baseline geometry.
        """
        p = self.get_box_prob()
        p.setup()
        p.run_model()
        val = p.get_val(self.kwargs["name"])
        tol = 1e-5 if not hasattr(self, "valTol") else self.valTol
        assert_near_equal(val, self.valCheck, tolerance=tol)

    def test_deformed_derivs_fwd(self):
        """
        Test the total derivatives in forward mode on a random perturbation to the baseline.
        """
        p = self.get_box_prob()
        p.setup(mode="fwd")

        # Pick some random deformed state
        p.set_val("twist", self.rand.random() * 10)
        p.set_val("local", self.rand.random() * 10)

        p.run_model()

        # Check total derivatives using a directional derivatives
        totals = p.check_totals(step=1e-6, out_stream=None, directional=False)
        assert_check_totals(totals, atol=1e-5, rtol=3e-5)

    def test_deformed_derivs_rev(self):
        """
        Test the total derivatives in reverse mode on a random perturbation to the baseline.
        """
        p = self.get_box_prob()
        p.setup(mode="rev")

        # Pick some random deformed state
        p.set_val("twist", self.rand.random() * 10)
        p.set_val("local", self.rand.random() * 10)

        p.run_model()

        # Check total derivatives using a directional derivatives
        totals = p.check_totals(step=1e-6, out_stream=None, directional=False)
        assert_check_totals(totals, atol=1e-5, rtol=3e-5)


# parameters for ESP-based DVGeo tests
esp_test_params = [{"N_PROCS": 1, "name": "serial"}, {"N_PROCS": 4, "name": "parallel_4procs"}]

@unittest.skipUnless(mphysInstalled and ocsmInstalled, "OpenMDAO, MPhys, and ESP are required to test the ESP part of the pyGeo MPhys wrapper")
@parameterized_class(esp_test_params)
class TestDVGeoMPhysESP(unittest.TestCase):
    def setUp(self):
        self.input_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def modelSetup(self):

        class ESPGroup(Multipoint):
            def setup(self):
                self.add_subsystem("dvs", IndepVarComp(), promotes=["*"])
                self.add_subsystem("geometry", OM_DVGEOCOMP(file=espBox, type="esp"))

            def configure(self):

                # add a point set on the surface
                vertex1 = np.array([-2.0, -2.0, -2.0])
                vertex2 = np.array([1.5, 1.5, 1.5])
                left = np.array([-2.0, -1.1, -1.1])
                right = np.array([1.5, -1.2, -0.1])
                front = np.array([0.25, 1.5, 0.3])
                back = np.array([1.2, -2.0, -0.3])
                top = np.array([0.0, 0.1, 1.5])
                bottom = np.array([-1.9, -1.1, -2.0])
                initpts = np.vstack([vertex1, vertex2, left, right, front, back, top, bottom, left, right])
                distglobal = self.geometry.nom_addPointSet.addPointSet(initpts.flatten(), "mypts", cache_projections=False)
                self.assertAlmostEqual(distglobal, 0.0, 8)
                DVGeo._updateModel()
                DVGeo._updateProjectedPts()
                self.assertTrue(DVGeo.pointSetUpToDate)
                self.assertAlmostEqual(np.linalg.norm(initpts - DVGeo.pointSets["mypts"].proj_pts), 0.0, 10)

                for dv in dvInfo:
                    self.geometry.nom_addESPVariable()
                    
                    self.dvs.add_output(dvName, dv["val"])
                    self.connect(dvName, f"geometry.{dvName}")
                    self.add_design_var(dvName, upper=dv["upper"], lower=dv["lower"])

                self.add_constraint(f"geometry.{ptName}")

        prob = Problem(model=ESPGroup())
        prob.setup(mode="rev")

        return prob
    
    def test_run_model(self):
        self.prob.run_model()

    def testDVs(self):
        self.prob.run_model()

        data = self.prob.check_totals(step=1e-7, compact_print=True)
        for _, err in data.items():

            rel_err = err["rel error"]
            assert_near_equal(rel_err.forward, 0.0, 1e-5)

if __name__ == "__main__":
    unittest.main()
