# Standard Python modules
import copy
import os
import unittest

# External modules
from baseclasses.utils import Error
import commonUtils
import numpy as np
from parameterized import parameterized_class
from pyspline import Curve
from stl import mesh

# First party modules
from pygeo.mphys import OM_DVGEOCOMP

try:
    # External modules
    from openmdao.api import Group, IndepVarComp, Problem
    from openmdao.utils.assert_utils import assert_near_equal

    omInstalled = True

except ImportError:
    omInstalled = False

try:
    # External modules
    import pyOCSM  # noqa: F401

    ocsmInstalled = True

except ImportError:
    ocsmInstalled = False


# input files for all DVGeo types
input_path = os.path.dirname(os.path.abspath(__file__))
outerFFD = os.path.join(input_path, "..", "..", "input_files", "outerBoxFFD.xyz")
innerFFD = os.path.join(input_path, "..", "..", "input_files", "simpleInnerFFD.xyz")
rectFFD = os.path.join(input_path, "..", "..", "input_files", "2x1x8_rectangle.xyz")
espBox = os.path.join(input_path, "..", "..", "input_files", "esp", "box.csm")

# parameters for FFD-based DVGeo tests
childName = "childFFD"
globalDVFuncParamsParent = ["mainX", -1.0, commonUtils.mainAxisPoints]
localDVFuncParamsParent = ["xdir"]
globalDVFuncParamsChild = ["nestedX", -0.5, commonUtils.childAxisPoints, childName]
shapeFuncParamsParent = ["shapeFunc", []]

globalDVParent = {
    "funcName": "nom_addGlobalDV",
    "funcParams": globalDVFuncParamsParent,
    "lower": -1.0,
    "upper": 0.0,
    "val": -1.0,
}
localDVParent = {
    "funcName": "nom_addLocalDV",
    "funcParams": localDVFuncParamsParent,
    "lower": -1.0,
    "upper": 1.0,
    "val": 12 * [0.0],
}
globalDVChild = {
    "funcName": "nom_addGlobalDV",
    "funcParams": globalDVFuncParamsChild,
    "lower": -1.0,
    "upper": 0.0,
    "val": -1.0,
}
shapeFuncDV = {
    "funcName": "nom_addShapeFunctionDV",
    "funcParams": shapeFuncParamsParent,
    "lower": -10.0,
    "upper": 10.0,
    "val": np.zeros((2)),
}

ffd_test_params = [
    {
        "name": "MPhys_FFD_oneFFD_global",
        "parentFFD": outerFFD,
        "childFFD": None,
        "dvInfo": [globalDVParent],
    },  # test_DVGeometry #1
    {
        "name": "MPhys_FFD_oneFFD_global+local",
        "parentFFD": outerFFD,
        "childFFD": None,
        "dvInfo": [globalDVParent, localDVParent],
    },  # test_DVGeometry #2
    {
        "name": "MPhys_FFD_childFFD_global",
        "parentFFD": outerFFD,
        "childFFD": innerFFD,
        "dvInfo": [globalDVParent, globalDVChild],
    },  # test_DVGeometry #3
    {
        "name": "MPhys_FFD_shapeFunc",
        "parentFFD": rectFFD,
        "childFFD": None,
        "dvInfo": [shapeFuncDV],
    },  # test_DVGeometry test_shape_functions
]

# DVConstraints functionals to test
test_params_constraints_box = [
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {
            "name": "func",
            "ptList": [[0.0, 0.0, 0.1], [0.0, 0.0, 7.9]],
            "nCon": 3,
            "axis": [1, 0, 0],
            "scaled": False,
        },
        "valCheck": 2 * np.ones(3),
        "valTol": 1e-4,
    },
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {
            "name": "func",
            "ptList": [[0.0, 0.0, 0.1], [0.0, 0.0, 7.9]],
            "nCon": 5,
            "axis": [0, 1, 0],
            "scaled": False,
        },
        "valCheck": np.ones(5),
        "valTol": 3e-5,
    },
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {
            "name": "func",
            "ptList": [[-0.5, 0.0, 4.0], [0.5, 0.0, 4.0]],
            "nCon": 5,
            "axis": [0, 0, 1],
            "scaled": False,
        },
        "valCheck": 8 * np.ones(5),
    },
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {
            "name": "func",
            "ptList": [[0.0, 0.0, 0.1], [0.0, 0.0, 7.9]],
            "nCon": 3,
            "axis": [1, 0, 0],
            "scaled": False,
            "projected": True,
        },
        "valCheck": 2 * np.ones(3),
        "valTol": 2e-4,
    },
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {
            "name": "func",
            "ptList": [[0.0, 0.0, 0.1], [0.0, 0.0, 7.9]],
            "nCon": 5,
            "axis": [0, 1, 0],
            "scaled": False,
            "projected": True,
        },
        "valCheck": np.ones(5),
        "valTol": 2e-4,
    },
    {
        "conFunc": "nom_addThicknessConstraints1D",
        "kwargs": {
            "name": "func",
            "ptList": [[-0.5, 0.0, 4.0], [0.5, 0.0, 4.0]],
            "nCon": 5,
            "axis": [0, 0, 1],
            "scaled": False,
            "projected": True,
        },
        "valCheck": 8 * np.ones(5),
    },
    {
        "conFunc": "nom_addThicknessConstraints2D",
        "kwargs": {
            "name": "func",
            "leList": [[-0.25, 0.0, 0.1], [-0.25, 0.0, 7.9]],
            "teList": [[0.75, 0.0, 0.1], [0.75, 0.0, 7.9]],
            "nSpan": 2,
            "nChord": 3,
            "scaled": False,
        },
        "valCheck": np.ones(6),
    },
    {
        "conFunc": "nom_addThicknessConstraints2D",
        "kwargs": {
            "name": "func",
            "leList": [[0.0, -0.25, 0.1], [0.0, -0.25, 7.9]],
            "teList": [[0.0, 0.25, 0.1], [0.0, 0.25, 7.9]],
            "nSpan": 2,
            "nChord": 3,
            "scaled": False,
        },
        "valCheck": 2 * np.ones(6),
    },
    {
        "conFunc": "nom_addThicknessConstraints2D",
        "kwargs": {
            "name": "func",
            "leList": [[-0.5, -0.25, 0.1], [0.5, -0.25, 0.1]],
            "teList": [[-0.5, 0.25, 0.1], [0.5, 0.25, 0.1]],
            "nSpan": 2,
            "nChord": 3,
            "scaled": False,
        },
        "valCheck": 8 * np.ones(6),
    },
    {
        "conFunc": "nom_addThicknessConstraints2D",
        "kwargs": {
            "name": "func",
            "leList": [[-0.5, -0.25, 0.1], [0.5, -0.25, 0.1]],
            "teList": [[-0.5, 0.25, 0.1], [0.5, 0.25, 0.1]],
            "nSpan": 2,
            "nChord": 3,
            "scaled": False,
            "projected": True,
        },
        "valCheck": 8 * np.ones(6),
    },
    {
        "conFunc": "nom_addVolumeConstraint",
        "kwargs": {
            "name": "func",
            "leList": [[-0.5, -0.25, 0.1], [0.5, -0.25, 0.1]],
            "teList": [[-0.5, 0.25, 0.1], [0.5, 0.25, 0.1]],
            "nSpan": 4,
            "nChord": 4,
            "scaled": False,
        },
        "valCheck": 4.0,
        "valTol": 1e-4,
    },
    {"conFunc": "nom_addSurfaceAreaConstraint", "kwargs": {"name": "func", "scaled": False}, "valCheck": 52.0},
    {
        "conFunc": "nom_addProjectedAreaConstraint",
        "kwargs": {"name": "func", "axis": "x", "scaled": False},
        "valCheck": 8.0,
        "valTol": 3e-2,
    },
    {
        "conFunc": "nom_addProjectedAreaConstraint",
        "kwargs": {"name": "func", "axis": "y", "scaled": False},
        "valCheck": 16.0,
        "valTol": 3e-2,
    },
    {
        "conFunc": "nom_addProjectedAreaConstraint",
        "kwargs": {"name": "func", "axis": "z", "scaled": False},
        "valCheck": 2.0,
        "valTol": 3e-2,
    },
]


@unittest.skipUnless(omInstalled, "OpenMDAO is required to test the pyGeo MPhys wrapper")
@parameterized_class(ffd_test_params)
class TestDVGeoMPhysFFD(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        # give the OM Group access to the test case attributes
        dvInfo = self.dvInfo
        parentFFD = self.parentFFD
        childFFD = self.childFFD

        class FFDGroup(Group):
            def setup(self):
                self.add_subsystem("dvs", IndepVarComp(), promotes=["*"])
                self.add_subsystem("geometry", OM_DVGEOCOMP(file=parentFFD, type="ffd"))

            def configure(self):
                # get the DVGeo object out of the geometry component
                DVGeo = self.geometry.nom_getDVGeo()

                # embed a dummy pointset in the parent FFD
                points = np.zeros([2, 3])
                points[0, :] = [0.25, 0, 0]
                points[1, :] = [-0.25, 0, 0]
                ptName = "testPoints"
                self.geometry.nom_addPointSet(points.flatten(), ptName, distributed=False)

                # create a reference axis for the parent
                axisPoints = [[-1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
                c1 = Curve(X=axisPoints, k=2)
                self.geometry.nom_addRefAxis("mainAxis", curve=c1, axis="y")

                # local index is needed for shape function DV
                lidx = DVGeo.getLocalIndex(0)

                # add child FFD if necessary for the test
                if childFFD is not None:
                    self.geometry.nom_addChild(innerFFD, childName=childName)

                    # create a reference axis for the child
                    axisPoints = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
                    c1 = Curve(X=axisPoints, k=2)
                    self.geometry.nom_addRefAxis("nestedAxis", childName=childName, curve=c1, axis="y")

                # add each DV to the geometry
                for dv in dvInfo:
                    dvName = dv["funcParams"][0]

                    # parameters for shape func DV aren't known until the DVGeo object is created
                    if dv["funcName"] == "nom_addShapeFunctionDV":
                        dv["funcParams"][1] = commonUtils.getShapeFunc(lidx)

                    # call the function being tested
                    getattr(self.geometry, dv["funcName"])(*dv["funcParams"])

                    # OM stuff
                    self.dvs.add_output(dvName, dv["val"])
                    self.connect(dvName, f"geometry.{dvName}")
                    self.add_design_var(dvName, upper=dv["upper"], lower=dv["lower"])

                self.add_constraint(f"geometry.{ptName}")

        self.prob = Problem(model=FFDGroup(), reports=False)

    def test_run_model(self):
        self.prob.setup()
        self.prob.run_model()

    def test_deriv_fwd(self):
        self.prob.setup(mode="fwd")
        self.prob.run_model()

        totals = self.prob.check_totals(step=1e-7, out_stream=None)
        commonUtils.assert_check_totals(totals, atol=1e-6, rtol=1e-6)

    def test_deriv_rev(self):
        self.prob.setup(mode="rev")
        self.prob.run_model()

        totals = self.prob.check_totals(step=1e-5, out_stream=None)
        commonUtils.assert_check_totals(totals, atol=1e-5, rtol=1e-5)


@unittest.skipUnless(omInstalled, "OpenMDAO is required to test the pyGeo MPhys wrapper")
@parameterized_class(test_params_constraints_box)
class TestDVConMPhysBox(unittest.TestCase):
    def setUp(self):
        # Random number generator
        self.rand = np.random.default_rng(1)

    def get_box_prob(self, **kwargs):
        """
        Generate an OpenMDAO problem with the OM_DVGEOCOMP component with the
        functional dictated by the parameterized class. Custom keyword arguments
        can be passed in, which will override any specified in the parameterized
        constraint information.
        """
        # Parameterized values
        conFunc = self.conFunc
        paramKwargs = copy.deepcopy(self.kwargs)

        # Update the parameterized constraint keyword arguments with any manually specified ones
        paramKwargs.update(kwargs)

        meshFile = os.path.join(input_path, "..", "..", "input_files", "2x1x8_rectangle.stl")
        ffdFile = os.path.join(input_path, "..", "..", "input_files", "2x1x8_rectangle.xyz")
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
                getattr(self.geo, conFunc)(**paramKwargs)

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
                self.add_objective(paramKwargs["name"])

        p = Problem(model=BoxGeo(), reports=False)
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
        if "addProjectedAreaConstraint" in self.conFunc:
            # Use some random axis to avoid ill-conditioned derivatives
            p = self.get_box_prob(axis=np.array([0.5, 3, -1]))
        else:
            p = self.get_box_prob()
        p.setup(mode="fwd")

        # Pick some random deformed state
        p.set_val("twist", self.rand.random() * 10)
        p.set_val("local", self.rand.random() * 10)

        p.run_model()

        totals = p.check_totals(step=1e-6, out_stream=None)
        commonUtils.assert_check_totals(totals, atol=1e-5, rtol=1e-4)

    def test_deformed_derivs_rev(self):
        """
        Test the total derivatives in reverse mode on a random perturbation to the baseline.
        """
        if "addProjectedAreaConstraint" in self.conFunc:
            # Use some random axis to avoid ill-conditioned derivatives caused by triangle
            # elements with normals orthogonal to the projection direction
            p = self.get_box_prob(axis=np.array([0.5, 3, -1]))
        else:
            p = self.get_box_prob()
        p.setup(mode="rev")

        # Pick some random deformed state
        p.set_val("twist", self.rand.random() * 10)
        p.set_val("local", self.rand.random() * 10)

        p.run_model()

        totals = p.check_totals(step=1e-5, out_stream=None)
        commonUtils.assert_check_totals(totals, atol=5e-5, rtol=5e-5)


# parameters for ESP-based DVGeo tests
fullESPDV = {"name": "cubex0", "lower": np.array([-10.0]), "upper": np.array([10.0]), "scale": 0.1, "dh": 0.0001}
simpleESPDV = {"name": "cubey0"}
midESPDV = {"name": "cubez0", "lower": np.array([-10.0]), "upper": np.array([10.0])}

esp_test_params = [
    {"name": "serial", "N_PROCS": 1, "dvInfo": [fullESPDV, simpleESPDV, midESPDV]},
]


@unittest.skipUnless(
    omInstalled and ocsmInstalled,
    "OpenMDAO, MPhys, and ESP are required to test the ESP part of the pyGeo MPhys wrapper",
)
@parameterized_class(esp_test_params)
class TestDVGeoMPhysESP(unittest.TestCase):
    def setUp(self):
        # give the OM Group access to the test case attributes
        dvInfo = self.dvInfo

        class ESPGroup(Group):
            def setup(self):
                self.add_subsystem("dvs", IndepVarComp(), promotes=["*"])
                self.add_subsystem("geometry", OM_DVGEOCOMP(file=espBox, type="esp"))

            def configure(self):
                # get the DVGeo object out of the geometry component
                DVGeo = self.geometry.nom_getDVGeo()

                # add a point set on the surface
                vertex1 = np.array([-2.0, -2.0, -2.0])
                vertex2 = np.array([1.5, 1.5, 1.5])
                left = np.array([-2.0, -1.1, -1.1])
                right = np.array([1.5, -1.2, -0.1])
                front = np.array([0.25, 1.5, 0.3])
                back = np.array([1.2, -2.0, -0.3])
                top = np.array([0.0, 0.1, 1.5])
                bottom = np.array([-1.9, -1.1, -2.0])
                self.initpts = np.vstack([vertex1, vertex2, left, right, front, back, top, bottom, left, right])

                ptName = "mypts"
                self.distglobal = self.geometry.nom_addPointSet(self.initpts.flatten(), ptName, cache_projections=False)
                self.projPts = DVGeo.pointSets[ptName].proj_pts
                DVGeo._updateModel()
                DVGeo._updateProjectedPts()

                for dv in dvInfo:
                    if "upper" not in dv:
                        dv["upper"] = None

                    if "lower" not in dv:
                        dv["lower"] = None

                    if "scale" not in dv:
                        dv["scale"] = None

                    if "dh" not in dv:
                        dv["dh"] = 0.001

                    dvName = dv["name"]
                    self.geometry.nom_addESPVariable(dvName, dh=dv["dh"])

                    self.dvs.add_output(dvName)
                    self.connect(dvName, f"geometry.{dvName}")

                    self.add_design_var(dvName, upper=dv["upper"], lower=dv["lower"], scaler=dv["scale"])

                self.add_constraint(f"geometry.{ptName}")

        self.prob = Problem(model=ESPGroup(), reports=False)

    def test_run_model(self):
        self.prob.setup()
        with self.assertRaises(Error):
            try:
                self.prob.model.geometry.nom_addESPVariable("cubew0")
            except Error as e:
                mes = e.message
                raise e

        self.assertEqual(mes, 'User specified design parameter name "cubew0" which was not found in the CSM file')

        assert_near_equal(self.prob.model.distglobal, 0.0, 8)
        assert_near_equal(np.linalg.norm(self.prob.model.initpts - self.prob.model.projPts), 0.0, 10)

        self.prob.run_model()

    def test_deriv_fwd(self):
        self.prob.setup(mode="fwd")
        self.prob.run_model()

        totals = self.prob.check_totals(step=1e-7, out_stream=None)
        commonUtils.assert_check_totals(totals, atol=1e-6, rtol=1e-6)

    def test_deriv_rev(self):
        self.prob.setup(mode="rev")
        self.prob.run_model()

        totals = self.prob.check_totals(step=1e-5, out_stream=None)
        commonUtils.assert_check_totals(totals, atol=1e-5, rtol=1e-5)


@unittest.skipUnless(omInstalled, "OpenMDAO is required to test the pyGeo MPhys wrapper")
class TestGetDVGeoError(unittest.TestCase):
    # Make sure we get an error if we try to call nom_getDVGeo before setup
    def test_getDVGeo_error(self):
        class BadGroup(Group):
            def setup(self):
                geometryComp = OM_DVGEOCOMP(file=outerFFD, type="ffd")
                self.add_subsystem("geometry", geometryComp, promotes=["*"])
                geometryComp.nom_getDVGeo()

        prob = Problem(model=BadGroup(), reports=False)
        with self.assertRaises(RuntimeError):
            prob.setup()


if __name__ == "__main__":
    unittest.main()
