# Standard Python modules
import os
import unittest

# External modules
from baseclasses import BaseRegTest
from mpi4py import MPI
import numpy as np
from parameterized import parameterized_class
from stl import mesh

# First party modules
from pygeo import DVConstraints, DVGeometry

try:
    # External modules
    import geograd  # noqa: F401

    geogradInstalled = True
except ImportError:
    geogradInstalled = False

try:
    # External modules
    import pysurf  # noqa: F401

    pysurfInstalled = True
except ImportError:
    pysurfInstalled = False

if pysurfInstalled:
    # First party modules
    from pygeo import DVGeometryMulti


def evalFunctionsSensFD(DVGeo, DVCon, fdstep=1e-2):
    funcs = {}
    DVCon.evalFunctions(funcs, includeLinear=True)
    # make a deep copy of this
    outdims = {}
    for key in funcs.keys():
        val = funcs[key]
        if isinstance(val, np.ndarray):
            outdims[key] = val.shape[0]
            funcs[key] = val.copy()
        elif isinstance(val, (list, tuple)):
            outdims[key] = len(val)
        else:
            outdims[key] = 1

    xDV = DVGeo.getValues()
    indims = {}
    for key in xDV.keys():
        val = xDV[key]
        indims[key] = val.shape[0]

    # setup the output data structure
    funcsSens = {}
    for outkey in funcs.keys():
        funcsSens[outkey] = {}
        for inkey in xDV.keys():
            nRows = outdims[outkey]
            nCols = indims[inkey]
            funcsSens[outkey][inkey] = np.zeros((nRows, nCols))
    # now do finite differencing
    for inkey in xDV.keys():
        baseVar = xDV[inkey].copy()
        nDV = len(baseVar)
        for array_ind in range(nDV):
            xDV[inkey][array_ind] = baseVar[array_ind] + fdstep
            DVGeo.setDesignVars(xDV)
            funcs_fd = {}
            DVCon.evalFunctions(funcs_fd, includeLinear=True)
            for outkey in funcs.keys():
                temp_a = funcs_fd[outkey]
                temp_b = funcs[outkey]
                diff = temp_a - temp_b
                deriv_temp = diff / fdstep
                funcsSens[outkey][inkey][:, array_ind] = deriv_temp
            xDV[inkey][array_ind] = baseVar[array_ind]
    DVGeo.setDesignVars(xDV)
    DVCon.evalFunctions({})
    return funcsSens


def generic_test_base(DVGeo, DVCon, handler, checkDerivs=True, fdstep=1e-4):
    linear_constraint_keywords = ["lete", "monotonic", "linear_constraint"]
    funcs = {}
    DVCon.evalFunctions(funcs, includeLinear=True)
    handler.root_add_dict("funcs_base", funcs, rtol=1e-6, atol=1e-6)
    funcsSens = {}
    DVCon.evalFunctionsSens(funcsSens, includeLinear=True)
    # regress the derivatives
    if checkDerivs:
        handler.root_add_dict("derivs_base", funcsSens, rtol=1e-6, atol=1e-6)
        funcsSensFD = evalFunctionsSensFD(DVGeo, DVCon, fdstep=fdstep)
        for outkey in funcs.keys():
            for inkey in DVGeo.getValues().keys():
                try:
                    analytic = funcsSens[outkey][inkey]
                    fd = funcsSensFD[outkey][inkey]
                    handler.assert_allclose(analytic, fd, name="finite_diff_check", rtol=1e-3, atol=1e-3)
                except KeyError:
                    if any(sbstr in outkey for sbstr in linear_constraint_keywords):
                        # linear constraints only have their affected DVs in the dict
                        pass
                    else:
                        raise
    return funcs, funcsSens


test_params = [
    {
        # Standard one-level FFD
        "name": "standard",
        "child": False,
        "multi": False,
    },
    {
        # Deforming child FFD with a stationary parent FFD
        "name": "child",
        "child": True,
        "multi": False,
    },
    {
        # One deforming component FFD and a stationary component FFD
        # The components do not intersect
        "name": "multi",
        "child": False,
        "multi": True,
    },
]


@parameterized_class(test_params)
class RegTestPyGeo(unittest.TestCase):
    N_PROCS = 1

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.comm = MPI.COMM_WORLD

        # Skip multi component test if DVGeometryMulti cannot be imported (i.e. pySurf is not installed)
        if self.multi and not pysurfInstalled:
            self.skipTest("requires pySurf")

    def generate_dvgeo_dvcon(self, geometry, addToDVGeo=False, intersected=False):
        """
        This function creates the DVGeometry and DVConstraints objects for each geometry used in this class.

        The C172 wing represents a typical use case with twist and shape variables.

        The rectangular box is primarily used to test unscaled constraint function
        values against known values for thickness, volume, and surface area.

        The BWB is used for the triangulated surface and volume constraint tests.

        The RAE 2822 wing is used for the curvature constraint test.
        """

        if geometry == "c172":
            meshFile = os.path.join(self.base_path, "../../input_files/c172.stl")
            ffdFile = os.path.join(self.base_path, "../../input_files/c172.xyz")
            xFraction = 0.25
            meshScale = 1e-3
        elif geometry == "box":
            meshFile = os.path.join(self.base_path, "../../input_files/2x1x8_rectangle.stl")
            ffdFile = os.path.join(self.base_path, "../../input_files/2x1x8_rectangle.xyz")
            xFraction = 0.5
            meshScale = 1
        elif geometry == "bwb":
            meshFile = os.path.join(self.base_path, "../../input_files/bwb.stl")
            ffdFile = os.path.join(self.base_path, "../../input_files/bwb.xyz")
            xFraction = 0.25
            meshScale = 1
        elif geometry == "rae2822":
            ffdFile = os.path.join(self.base_path, "../../input_files/deform_geometry_ffd.xyz")
            xFraction = 0.25

        DVGeo = DVGeometry(ffdFile, child=self.child)
        if self.multi:
            # Use the nozzle FFD as the stationary component because it is outside all other FFD volumes
            nozzleFile = os.path.join(self.base_path, "../../input_files/nozzleFFD.xyz")
            DVGeoNozzle = DVGeometry(nozzleFile)
            # Set up the DVGeometryMulti object
            DVGeoMulti = DVGeometryMulti()
            DVGeoMulti.addComponent("deforming", DVGeo)
            DVGeoMulti.addComponent("stationary", DVGeoNozzle)

        DVCon = DVConstraints()
        nRefAxPts = DVGeo.addRefAxis("wing", xFraction=xFraction, alignIndex="k")
        self.nTwist = nRefAxPts - 1

        if self.child:
            parentFFD = os.path.join(self.base_path, "../../input_files/parent.xyz")
            self.parentDVGeo = DVGeometry(parentFFD)
            self.parentDVGeo.addChild(DVGeo)
            DVCon.setDVGeo(self.parentDVGeo)
        elif self.multi:
            DVCon.setDVGeo(DVGeoMulti)
        else:
            DVCon.setDVGeo(DVGeo)

        # Add design variables
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z["wing"].coef[i] = val[i - 1]

        DVGeo.addGlobalDV(dvName="twist", value=[0] * self.nTwist, func=twist, lower=-10, upper=10, scale=1)
        DVGeo.addLocalDV("local", lower=-0.5, upper=0.5, axis="y", scale=1)

        # RAE 2822 does not have a DVCon surface so we just return
        if geometry == "rae2822":
            return DVGeo, DVCon

        # Get the mesh from the STL
        testMesh = mesh.Mesh.from_file(meshFile)
        # dim 0 is triangle index
        # dim 1 is each vertex of the triangle
        # dim 2 is x, y, z dimension

        p0 = testMesh.vectors[:, 0, :] * meshScale
        v1 = testMesh.vectors[:, 1, :] * meshScale - p0
        v2 = testMesh.vectors[:, 2, :] * meshScale - p0
        DVCon.setSurface([p0, v1, v2], addToDVGeo=addToDVGeo)

        # Add the blob surface for the BWB
        if geometry == "bwb":
            objFile = os.path.join(self.base_path, "../../input_files/blob_bwb_wing.stl")
            testObj = mesh.Mesh.from_file(objFile)
            p0b = testObj.vectors[:, 0, :]
            v1b = testObj.vectors[:, 1, :] - p0b
            v2b = testObj.vectors[:, 2, :] - p0b
            if intersected:
                p0b = p0b + np.array([0.0, 0.3, 0.0])
            DVCon.setSurface([p0b, v1b, v2b], name="blob")

        return DVGeo, DVCon

    def wing_test_twist(self, DVGeo, DVCon, handler):
        funcs = {}
        funcsSens = {}
        # change the DVs
        xDV = DVGeo.getValues()
        xDV["twist"] = np.linspace(0, 10, len(xDV["twist"]))
        if self.child:
            # Twist needs to be set on the parent FFD to get accurate derivatives
            self.parentDVGeo.setDesignVars(xDV)
        else:
            DVGeo.setDesignVars(xDV)
        # check the constraint values changed
        DVCon.evalFunctions(funcs, includeLinear=True)

        handler.root_add_dict("funcs_twisted", funcs, rtol=1e-6, atol=1e-6)
        # check the derivatives are still right
        DVCon.evalFunctionsSens(funcsSens, includeLinear=True)
        # regress the derivatives
        handler.root_add_dict("derivs_twisted", funcsSens, rtol=1e-6, atol=1e-6)
        return funcs, funcsSens

    def wing_test_deformed(self, DVGeo, DVCon, handler):
        funcs = {}
        funcsSens = {}
        xDV = DVGeo.getValues()
        np.random.seed(37)
        xDV["local"] = np.random.normal(0.0, 0.05, len(xDV["local"]))
        DVGeo.setDesignVars(xDV)
        DVCon.evalFunctions(funcs, includeLinear=True)
        DVCon.evalFunctionsSens(funcsSens, includeLinear=True)
        handler.root_add_dict("funcs_deformed", funcs, rtol=1e-6, atol=1e-6)
        handler.root_add_dict("derivs_deformed", funcsSens, rtol=1e-6, atol=1e-6)
        return funcs, funcsSens

    def test_thickness1D(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_thickness1D.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            ptList = [[0.8, 0.0, 0.1], [0.8, 0.0, 5.0]]
            DVCon.addThicknessConstraints1D(ptList, nCon=10, axis=[0, 1, 0])

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler, checkDerivs=True)
            # 1D thickness should be all ones at the start
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_0"], np.ones(10), name="thickness_base", rtol=1e-7, atol=1e-7
            )

            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            # 1D thickness shouldn't change much under only twist
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_0"], np.ones(10), name="thickness_twisted", rtol=1e-2, atol=1e-2
            )

            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_thickness1D_box(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_thickness1D_box.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("box")

            ptList = [[0.0, 0.0, 0.1], [0.0, 0.0, 5.0]]
            ptList2 = [[-0.5, 0.0, 2.0], [0.5, 0.0, 2.0]]
            DVCon.addThicknessConstraints1D(ptList, nCon=3, axis=[0, 1, 0], scaled=False)
            DVCon.addThicknessConstraints1D(ptList, nCon=3, axis=[1, 0, 0], scaled=False)
            DVCon.addThicknessConstraints1D(ptList2, nCon=3, axis=[0, 0, 1], scaled=False)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            # Check that unscaled thicknesses are computed correctly at baseline
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_0"], np.ones(3), name="thickness_base", rtol=1e-7, atol=1e-7
            )
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_1"], 2.0 * np.ones(3), name="thickness_base", rtol=1e-7, atol=1e-7
            )
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_2"], 8.0 * np.ones(3), name="thickness_base", rtol=1e-7, atol=1e-7
            )

    def test_thickness2D(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_thickness2D.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            leList = [[0.7, 0.0, 0.1], [0.7, 0.0, 5.0]]
            teList = [[0.9, 0.0, 0.1], [0.9, 0.0, 5.0]]

            DVCon.addThicknessConstraints2D(leList, teList, 5, 5)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            # 2D thickness should be all ones at the start
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_0"], np.ones(25), name="thickness_base", rtol=1e-7, atol=1e-7
            )

            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            # 2D thickness shouldn't change much under only twist
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_0"], np.ones(25), name="thickness_twisted", rtol=1e-2, atol=1e-2
            )

            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_thickness2D_nSpanList(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_thickness2D.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            leList = [[0.7, 0.0, 0.1], [0.7, 0.0, 1.325], [0.7, 0.0, 5.0]]
            teList = [[0.9, 0.0, 0.1], [0.9, 0.0, 1.325], [0.9, 0.0, 5.0]]

            # Use a list for nSpan instead of an integer
            DVCon.addThicknessConstraints2D(leList, teList, [1, 4], 5)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            # 2D thickness should be all ones at the start
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_0"], np.ones(25), name="thickness_base", rtol=1e-7, atol=1e-7
            )

            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            # 2D thickness shouldn't change much under only twist
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_0"], np.ones(25), name="thickness_twisted", rtol=1e-2, atol=1e-2
            )

            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_thickness2D_box(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_thickness2D_box.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("box")

            leList = [[-0.25, 0.0, 0.1], [-0.25, 0.0, 7.9]]
            teList = [[0.75, 0.0, 0.1], [0.75, 0.0, 7.9]]

            leList2 = [[0.0, -0.25, 0.1], [0.0, -0.25, 7.9]]
            teList2 = [[0.0, 0.25, 0.1], [0.0, 0.25, 7.9]]

            leList3 = [[-0.5, -0.25, 0.1], [0.5, -0.25, 0.1]]
            teList3 = [[-0.5, 0.25, 0.1], [0.5, 0.25, 0.1]]

            DVCon.addThicknessConstraints2D(leList, teList, 2, 2, scaled=False)
            DVCon.addThicknessConstraints2D(leList2, teList2, 2, 2, scaled=False)
            DVCon.addThicknessConstraints2D(leList3, teList3, 2, 2, scaled=False)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            # Check that unscaled thicknesses are computed correctly at baseline
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_0"], np.ones(4), name="thickness_base", rtol=1e-7, atol=1e-7
            )
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_1"], 2.0 * np.ones(4), name="thickness_base", rtol=1e-7, atol=1e-7
            )
            handler.assert_allclose(
                funcs["DVCon1_thickness_constraints_2"], 8.0 * np.ones(4), name="thickness_base", rtol=1e-7, atol=1e-7
            )

    def test_volume(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_volume.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            leList = [[0.7, 0.0, 0.1], [0.7, 0.0, 5.0]]
            teList = [[0.9, 0.0, 0.1], [0.9, 0.0, 5.0]]

            DVCon.addVolumeConstraint(leList, teList, 5, 5)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            # Volume should be normalized to 1 at the start
            handler.assert_allclose(
                funcs["DVCon1_volume_constraint_0"], np.ones(1), name="volume_base", rtol=1e-7, atol=1e-7
            )

            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            # Volume shouldn't change much with twist only
            handler.assert_allclose(
                funcs["DVCon1_volume_constraint_0"], np.ones(1), name="volume_twisted", rtol=1e-2, atol=1e-2
            )

            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_volume_nSpanList(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_volume.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            leList = [[0.7, 0.0, 0.1], [0.7, 0.0, 1.325], [0.7, 0.0, 5.0]]
            teList = [[0.9, 0.0, 0.1], [0.9, 0.0, 1.325], [0.9, 0.0, 5.0]]

            # Use a list for nSpan instead of an integer
            DVCon.addVolumeConstraint(leList, teList, [1, 4], 5)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            # Volume should be normalized to 1 at the start
            handler.assert_allclose(
                funcs["DVCon1_volume_constraint_0"], np.ones(1), name="volume_base", rtol=1e-7, atol=1e-7
            )

            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            # Volume shouldn't change much with twist only
            handler.assert_allclose(
                funcs["DVCon1_volume_constraint_0"], np.ones(1), name="volume_twisted", rtol=1e-2, atol=1e-2
            )

            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_volume_box(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_volume_box.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("box")

            # this projects in the z direction which is of dimension 8
            # 1x0.5x8 = 4
            leList = [[-0.5, -0.25, 0.1], [0.5, -0.25, 0.1]]
            teList = [[-0.5, 0.25, 0.1], [0.5, 0.25, 0.1]]

            DVCon.addVolumeConstraint(leList, teList, 4, 4, scaled=False)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            # Check that unscaled volume is computed correctly at baseline
            handler.assert_allclose(
                funcs["DVCon1_volume_constraint_0"], 4.0 * np.ones(1), name="volume_base", rtol=1e-7, atol=1e-7
            )

    def test_LeTe(self, train=False, refDeriv=False):
        """
        LeTe constraint test using the iLow, iHigh method
        """
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_LeTe.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            if self.child:
                DVCon.addLeTeConstraints(0, "iLow", childIdx=0)
                DVCon.addLeTeConstraints(0, "iHigh", childIdx=0)
            elif self.multi:
                DVCon.addLeTeConstraints(0, "iLow", comp="deforming")
                DVCon.addLeTeConstraints(0, "iHigh", comp="deforming")
            else:
                DVCon.addLeTeConstraints(0, "iLow")
                DVCon.addLeTeConstraints(0, "iHigh")

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            # LeTe constraints should be all zero at the start
            for i in range(2):
                handler.assert_allclose(
                    funcs["DVCon1_lete_constraint_" + str(i)], np.zeros(4), name="lete_" + str(i), rtol=1e-7, atol=1e-7
                )

            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            # Global DVs should produce no change, especially twist
            for i in range(2):
                handler.assert_allclose(
                    funcs["DVCon1_lete_constraint_" + str(i)],
                    np.zeros(4),
                    name="lete_twisted_" + str(i),
                    rtol=1e-7,
                    atol=1e-7,
                )
            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_thicknessToChord(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_thicknessToChord.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            ptList = [[0.8, 0.0, 0.1], [0.8, 0.0, 5.0]]
            DVCon.addThicknessToChordConstraints1D(ptList, nCon=10, axis=[0, 1, 0], chordDir=[1, 0, 0])

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(
                funcs["DVCon1_thickness_to_chord_constraints_0"], np.ones(10), name="toverc_base", rtol=1e-7, atol=1e-7
            )

            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            handler.assert_allclose(
                funcs["DVCon1_thickness_to_chord_constraints_0"],
                np.ones(10),
                name="toverc_twisted",
                rtol=1e-3,
                atol=1e-3,
            )

            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_surfaceArea(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_surfaceArea.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            DVCon.addSurfaceAreaConstraint()

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(
                funcs["DVCon1_surfaceArea_constraints_0"], np.ones(1), name="surface_area_base", rtol=1e-7, atol=1e-7
            )

            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            handler.assert_allclose(
                funcs["DVCon1_surfaceArea_constraints_0"], np.ones(1), name="surface_area_twisted", rtol=1e-3, atol=1e-3
            )

            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_surfaceArea_box(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_surfaceArea_box.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("box")

            DVCon.addSurfaceAreaConstraint(scaled=False)
            # 2x1x8 box has surface area 2*(8*2+1*2+8*1) = 52
            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(
                funcs["DVCon1_surfaceArea_constraints_0"],
                52.0 * np.ones(1),
                name="surface_area_base",
                rtol=1e-7,
                atol=1e-7,
            )

    def test_projectedArea(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_projectedArea.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            DVCon.addProjectedAreaConstraint()

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(
                funcs["DVCon1_projectedArea_constraints_0"],
                np.ones(1),
                name="projected_area_base",
                rtol=1e-7,
                atol=1e-7,
            )

            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)

            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_projectedArea_box(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_projectedArea_box.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("box")

            DVCon.addProjectedAreaConstraint(scaled=False)
            DVCon.addProjectedAreaConstraint(axis="z", scaled=False)
            DVCon.addProjectedAreaConstraint(axis="x", scaled=False)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler, checkDerivs=False)
            handler.assert_allclose(
                funcs["DVCon1_projectedArea_constraints_0"],
                8 * 2 * np.ones(1),
                name="projected_area_base",
                rtol=1e-7,
                atol=1e-7,
            )
            handler.assert_allclose(
                funcs["DVCon1_projectedArea_constraints_1"],
                1 * 2 * np.ones(1),
                name="projected_area_base",
                rtol=1e-7,
                atol=1e-7,
            )
            handler.assert_allclose(
                funcs["DVCon1_projectedArea_constraints_2"],
                8 * 1 * np.ones(1),
                name="projected_area_base",
                rtol=1e-7,
                atol=1e-7,
            )

    def test_circularity(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_circularity.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            DVCon.addCircularityConstraint(
                origin=[0.8, 0.0, 2.5],
                rotAxis=[0.0, 0.0, 1.0],
                radius=0.1,
                zeroAxis=[0.0, 1.0, 0.0],
                angleCW=180.0,
                angleCCW=180.0,
                nPts=10,
            )

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(
                funcs["DVCon1_circularity_constraints_0"], np.ones(9), name="circularity_base", rtol=1e-7, atol=1e-7
            )

            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            handler.assert_allclose(
                funcs["DVCon1_circularity_constraints_0"], np.ones(9), name="circularity_twisted", rtol=1e-7, atol=1e-7
            )
            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_colinearity(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_colinearity.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            DVCon.addColinearityConstraint(
                np.array([0.7, 0.0, 1.0]), lineAxis=np.array([0.0, 0.0, 1.0]), distances=[0.0, 1.0, 2.5]
            )

            # Skip derivatives check here because true zero values cause difficulties for the partials
            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler, checkDerivs=False)
            handler.assert_allclose(
                funcs["DVCon1_colinearity_constraints_0"], np.zeros(3), name="colinearity_base", rtol=1e-7, atol=1e-7
            )

            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_linearConstraintShape(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_linearConstraintShape.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")
            lIndex = DVGeo.getLocalIndex(0)
            indSetA = []
            indSetB = []
            for i in range(lIndex.shape[0]):
                indSetA.append(lIndex[i, 0, 0])
                indSetB.append(lIndex[i, 0, 1])
            if self.child:
                DVCon.addLinearConstraintsShape(
                    indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0, upper=0, childIdx=0
                )
            elif self.multi:
                DVCon.addLinearConstraintsShape(
                    indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0, upper=0, comp="deforming"
                )
            else:
                DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0, upper=0)
            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_compositeVolume_box(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_compositeVolume_box.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("box")

            # this projects in the z direction which is of dimension 8
            # 1x0.5x8 = 4
            leList = [[-0.5, -0.25, 0.1], [0.5, -0.25, 0.1]]
            teList = [[-0.5, 0.25, 0.1], [0.5, 0.25, 0.1]]

            # this projects in the x direction which is of dimension 2
            # 2x0.6x7.8 = 9.36
            leList2 = [[0.0, -0.25, 0.1], [0.0, -0.25, 7.9]]
            teList2 = [[0.0, 0.35, 0.1], [0.0, 0.35, 7.9]]
            DVCon.addVolumeConstraint(leList, teList, 4, 4, scaled=False, addToPyOpt=False)
            DVCon.addVolumeConstraint(leList2, teList2, 4, 4, scaled=False, addToPyOpt=False)
            vols = ["DVCon1_volume_constraint_0", "DVCon1_volume_constraint_1"]
            DVCon.addCompositeVolumeConstraint(vols=vols, scaled=False)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            # Check that unscaled volumes are computed correctly at baseline
            handler.assert_allclose(
                funcs["DVCon1_volume_constraint_0"], 4.0 * np.ones(1), name="volume1_base", rtol=1e-7, atol=1e-7
            )
            handler.assert_allclose(
                funcs["DVCon1_volume_constraint_1"], 9.36 * np.ones(1), name="volume2_base", rtol=1e-7, atol=1e-7
            )
            handler.assert_allclose(
                funcs["DVCon1_composite_volume_constraint_2"],
                13.36 * np.ones(1),
                name="volume_composite_base",
                rtol=1e-7,
                atol=1e-7,
            )

    def test_location_box(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_location_box.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("box")

            ptList = [[0.0, 0.0, 0.0], [0.0, 0.0, 8.0]]
            ptList2 = [[0.0, 0.2, 0.0], [0.0, -0.2, 8.0]]

            # TODO this constraint seems buggy. for example, when scaled, returns a bunch of NaNs
            DVCon.addLocationConstraints1D(ptList=ptList, nCon=10, scaled=False)
            DVCon.addProjectedLocationConstraints1D(ptList=ptList2, nCon=10, scaled=False, axis=[0.0, 1.0, 0.0])
            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)

            exact_vals = np.zeros((30,))
            exact_vals[2::3] = np.linspace(0, 8, 10)
            # should be 10 evenly spaced points along the z axis originating from 0,0,0
            handler.assert_allclose(
                funcs["DVCon1_location_constraints_0"], exact_vals, name="locations_match", rtol=1e-7, atol=1e-7
            )
            handler.assert_allclose(
                funcs["DVCon1_location_constraints_1"],
                exact_vals,
                name="projected_locations_match",
                rtol=1e-7,
                atol=1e-7,
            )

    def test_planarity_box(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_planarity_box.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("box")

            DVCon.addPlanarityConstraint(origin=[0.0, 0.5, 0.0], planeAxis=[0.0, 1.0, 0.0])
            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)

    def test_planarity_tri(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_planarity_tri.ref")
        with BaseRegTest(refFile, train=train) as handler:
            # Set up a dummy DVGeo and DVCon
            DVGeo, DVCon = self.generate_dvgeo_dvcon("box")

            # Set a DVConstraints surface consisting of a simple plane with 2 triangles
            p0 = np.zeros(shape=(2, 3))
            p1 = np.zeros(shape=(2, 3))
            p2 = np.zeros(shape=(2, 3))

            vertex1 = np.array([0.5, -0.25, 0.0])
            vertex2 = np.array([0.5, -0.25, 4.0])
            vertex3 = np.array([-0.5, -0.25, 0.0])
            vertex4 = np.array([-0.5, -0.25, 4.0])

            p0[:, :] = vertex1
            p2[:, :] = vertex4
            p1[0, :] = vertex2
            p1[1, :] = vertex3

            v1 = p1 - p0
            v2 = p2 - p0
            DVCon.setSurface([p0, v1, v2], name="tri")

            DVCon.addPlanarityConstraint(origin=[0.0, -0.25, 2.0], planeAxis=[0.0, 1.0, 0.0], surfaceName="tri")

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler, checkDerivs=False)

            # this should be coplanar and the planarity constraint should be zero
            handler.assert_allclose(
                funcs["DVCon1_planarity_constraints_0"], np.zeros(1), name="planarity", rtol=1e-7, atol=1e-7
            )

    def test_monotonic(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_monotonic.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            if self.child:
                DVCon.addMonotonicConstraints("twist", childIdx=0)
                DVCon.addMonotonicConstraints("twist", start=1, stop=2, childIdx=0)
            elif self.multi:
                DVCon.addMonotonicConstraints("twist", comp="deforming")
                DVCon.addMonotonicConstraints("twist", start=1, stop=2, comp="deforming")
            else:
                DVCon.addMonotonicConstraints("twist")
                DVCon.addMonotonicConstraints("twist", start=1, stop=2)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(
                funcs["DVCon1_monotonic_constraint_0"], np.zeros(2), name="monotonicity", rtol=1e-7, atol=1e-7
            )
            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            handler.assert_allclose(
                funcs["DVCon1_monotonic_constraint_0"],
                -5.0 * np.ones(2),
                name="monotonicity_twisted",
                rtol=1e-7,
                atol=1e-7,
            )

            funcs = {}
            funcsSens = {}
            # change the DVs arbitrarily
            xDV = DVGeo.getValues()
            xDV["twist"][0] = 1.0
            xDV["twist"][1] = -3.5
            xDV["twist"][2] = -2.5
            DVGeo.setDesignVars(xDV)
            # check the constraint values changed
            DVCon.evalFunctions(funcs, includeLinear=True)
            handler.root_add_dict("funcs_arb_twist", funcs, rtol=1e-6, atol=1e-6)
            # check the derivatives are still right
            DVCon.evalFunctionsSens(funcsSens, includeLinear=True)
            # regress the derivatives
            handler.root_add_dict("derivs_arb_twist", funcsSens, rtol=1e-6, atol=1e-6)
            handler.assert_allclose(
                funcs["DVCon1_monotonic_constraint_0"],
                np.array([4.5, -1.0]),
                name="monotonicity_arb_twist",
                rtol=1e-7,
                atol=1e-7,
            )
            handler.assert_allclose(
                funcs["DVCon1_monotonic_constraint_1"],
                np.array([-1.0]),
                name="monotonicity_arb_twist_1",
                rtol=1e-7,
                atol=1e-7,
            )

    @unittest.skipUnless(geogradInstalled, "requires geograd")
    def test_triangulatedSurface(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_triangulatedSurface.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("bwb", addToDVGeo=True)

            DVCon.addTriangulatedSurfaceConstraint(
                self.comm, "default", "default", "blob", None, rho=10.0, addToPyOpt=True
            )
            DVCon.addTriangulatedSurfaceConstraint(
                self.comm, "default", "default", "blob", None, rho=1000.0, addToPyOpt=True
            )

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler, fdstep=1e-3)
            handler.assert_allclose(
                funcs["DVCon1_trisurf_constraint_0_KS"], 0.34660627481696404, name="KS", rtol=1e-7, atol=1e-7
            )
            handler.assert_allclose(funcs["DVCon1_trisurf_constraint_0_perim"], 0.0, name="perim", rtol=1e-7, atol=1e-7)
            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    @unittest.skipUnless(geogradInstalled, "requires geograd")
    def test_triangulatedSurface_intersected(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_triangulatedSurface_intersected.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("bwb", addToDVGeo=True, intersected=True)

            DVCon.addTriangulatedSurfaceConstraint(
                self.comm, "default", "default", "blob", None, rho=10.0, addToPyOpt=True
            )

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            np.testing.assert_array_less(np.zeros(1), funcs["DVCon1_trisurf_constraint_0_perim"])

    def test_triangulatedVolume_box(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_triangulatedVolume_box.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("box", addToDVGeo=True)

            DVCon.addTriangulatedVolumeConstraint(scaled=False, name="unscaled_vol_con")
            DVCon.addTriangulatedVolumeConstraint(scaled=True)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            # Check that the scaled and unscaled volumes are computed correctly at baseline
            handler.assert_allclose(
                funcs["DVCon1_trivolume_constraint_1"], 1.0, name="scaled_volume_base", rtol=1e-7, atol=1e-7
            )
            handler.assert_allclose(funcs["unscaled_vol_con"], 16.0, name="unscaled_volume_base", rtol=1e-7, atol=1e-7)

    def test_triangulatedVolume_bwb(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_triangulatedVolume_bwb.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("bwb", addToDVGeo=True)

            DVCon.addTriangulatedVolumeConstraint(scaled=False, name="unscaled_vol_con")
            DVCon.addTriangulatedVolumeConstraint(scaled=True)

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)

            # Check that the scaled and unscaled volumes are computed correctly at baseline
            handler.assert_allclose(
                funcs["DVCon1_trivolume_constraint_1"], 1.0, name="scaled_volume_base", rtol=1e-7, atol=1e-7
            )
            # BWB volume computed with meshmixer
            handler.assert_allclose(
                funcs["unscaled_vol_con"], 1103.57, name="unscaled_volume_base", rtol=1e-7, atol=1e-7
            )

    def test_curvature(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_curvature.ref")
        with BaseRegTest(refFile, train=train) as handler:
            # Use the RAE 2822 wing because we have a PLOT3D surface file for it
            DVGeo, DVCon = self.generate_dvgeo_dvcon("rae2822", addToDVGeo=True)
            surfFile = os.path.join(self.base_path, "../../input_files/deform_geometry_wing.xyz")

            # Add both scaled and unscaled curvature constraints
            DVCon.addCurvatureConstraint(surfFile, curvatureType="mean")
            DVCon.addCurvatureConstraint(surfFile, curvatureType="mean", scaled=False, name="unscaled_curvature_con")

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler, fdstep=1e-5)
            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_curvature1D(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_curvature1D.ref")
        with BaseRegTest(refFile, train=train) as handler:
            # Use the RAE 2822 wing because we have a PLOT3D surface file for it
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172", addToDVGeo=True)

            # Add both scaled and unscaled curvature constraints
            startP = [0.1, 0, 1.0]
            endP = [1.5, 0, 1.0]
            axis = [0, 1, 0]
            nPts = 10
            DVCon.addCurvatureConstraint1D(startP, endP, nPts, axis, "mean")
            DVCon.addCurvatureConstraint1D(
                startP, endP, nPts, axis, "mean", scaled=False, name="unscaled_curvature_con"
            )
            DVCon.addCurvatureConstraint1D(startP, endP, nPts, axis, "aggregated", KSCoeff=10.0)
            DVCon.addCurvatureConstraint1D(
                startP, endP, nPts, axis, "aggregated", KSCoeff=10.0, scaled=False, name="unscaled_curvature_con"
            )

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler, checkDerivs=False)
            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)

    def test_LERadius(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_LERadius.ref")
        with BaseRegTest(refFile, train=train) as handler:
            DVGeo, DVCon = self.generate_dvgeo_dvcon("c172")

            leList = [[1e-4, 0, 1e-3], [1e-3, 0, 2.5], [0.15, 0, 5.0]]

            # Add both scaled and unscaled LE radius constraints
            DVCon.addLERadiusConstraints(leList, 5, [0, 1, 0], [-1, 0, 0])
            DVCon.addLERadiusConstraints(leList, 5, [0, 1, 0], [-1, 0, 0], scaled=False, name="unscaled_radius_con")

            funcs, funcsSens = generic_test_base(DVGeo, DVCon, handler)
            funcs, funcsSens = self.wing_test_twist(DVGeo, DVCon, handler)
            funcs, funcsSens = self.wing_test_deformed(DVGeo, DVCon, handler)


@unittest.skipUnless(geogradInstalled, "requires geograd")
class RegTestGeograd(unittest.TestCase):
    N_PROCS = 1

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.comm = MPI.COMM_WORLD

    def test_triangulatedSurface_intersected_2DVGeos(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_DVConstraints_triangulatedSurface_intersected_2DVGeos.ref")
        with BaseRegTest(refFile, train=train) as handler:
            meshFile = os.path.join(self.base_path, "../../input_files/bwb.stl")
            objFile = os.path.join(self.base_path, "../../input_files/blob_bwb_wing.stl")
            ffdFile = os.path.join(self.base_path, "../../input_files/bwb.xyz")
            testMesh = mesh.Mesh.from_file(meshFile)
            testObj = mesh.Mesh.from_file(objFile)
            # test mesh dim 0 is triangle index
            # dim 1 is each vertex of the triangle
            # dim 2 is x, y, z dimension

            # create a DVGeo object with a few local thickness variables
            DVGeo1 = DVGeometry(ffdFile)
            nRefAxPts = DVGeo1.addRefAxis("wing", xFraction=0.25, alignIndex="k")
            self.nTwist = nRefAxPts - 1

            def twist(val, geo):
                for i in range(1, nRefAxPts):
                    geo.rot_z["wing"].coef[i] = val[i - 1]

            DVGeo1.addGlobalDV(dvName="twist", value=[0] * self.nTwist, func=twist, lower=-10, upper=10, scale=1)
            DVGeo1.addLocalDV("local", lower=-0.5, upper=0.5, axis="y", scale=1)

            # create a DVGeo object with a few local thickness variables
            DVGeo2 = DVGeometry(ffdFile, name="blobdvgeo")
            DVGeo2.addLocalDV("local_2", lower=-0.5, upper=0.5, axis="y", scale=1)

            # check that DVGeos with duplicate var names are not allowed
            DVGeo3 = DVGeometry(ffdFile)
            DVGeo3.addLocalDV("local", lower=-0.5, upper=0.5, axis="y", scale=1)

            # create a DVConstraints object for the wing
            DVCon = DVConstraints()
            DVCon.setDVGeo(DVGeo1)
            DVCon.setDVGeo(DVGeo2, name="second")
            with self.assertRaises(ValueError):
                DVCon.setDVGeo(DVGeo3, name="third")

            p0 = testMesh.vectors[:, 0, :]
            v1 = testMesh.vectors[:, 1, :] - p0
            v2 = testMesh.vectors[:, 2, :] - p0
            DVCon.setSurface([p0, v1, v2], addToDVGeo=True)
            p0b = testObj.vectors[:, 0, :]
            v1b = testObj.vectors[:, 1, :] - p0b
            v2b = testObj.vectors[:, 2, :] - p0b
            p0b = p0b + np.array([0.0, 0.3, 0.0])
            DVCon.setSurface([p0b, v1b, v2b], name="blob", addToDVGeo=True, DVGeoName="second")

            DVCon.addTriangulatedSurfaceConstraint(
                self.comm, "default", "default", "blob", "second", rho=10.0, addToPyOpt=True
            )

            funcs = {}
            DVCon.evalFunctions(funcs, includeLinear=True)
            handler.root_add_dict("funcs_base", funcs, rtol=1e-6, atol=1e-6)
            funcsSens = {}
            DVCon.evalFunctionsSens(funcsSens, includeLinear=True)
            # regress the derivatives
            handler.root_add_dict("derivs_base", funcsSens, rtol=1e-6, atol=1e-6)

            # FD check DVGeo1
            funcsSensFD = evalFunctionsSensFD(DVGeo1, DVCon, fdstep=1e-3)
            at_least_one_var = False
            for outkey in funcs.keys():
                for inkey in DVGeo1.getValues().keys():
                    analytic = funcsSens[outkey][inkey]
                    fd = funcsSensFD[outkey][inkey]
                    handler.assert_allclose(analytic, fd, name="finite_diff_check", rtol=1e-3, atol=1e-3)
                    # make sure there are actually checks happening
                    self.assertTrue(np.abs(np.sum(fd)) > 1e-10)
                    at_least_one_var = True
            self.assertTrue(at_least_one_var)

            # FD check DVGeo2
            funcsSensFD = evalFunctionsSensFD(DVGeo2, DVCon, fdstep=1e-3)
            at_least_one_var = False
            for outkey in funcs.keys():
                for inkey in DVGeo2.getValues().keys():
                    analytic = funcsSens[outkey][inkey]
                    fd = funcsSensFD[outkey][inkey]
                    handler.assert_allclose(analytic, fd, name="finite_diff_check", rtol=1e-3, atol=1e-3)
                    self.assertTrue(np.abs(np.sum(fd)) > 1e-10)
                    at_least_one_var = True
            self.assertTrue(at_least_one_var)


if __name__ == "__main__":
    unittest.main()
