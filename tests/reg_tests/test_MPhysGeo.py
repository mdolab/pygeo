import unittest
import os
import numpy as np
from parameterized import parameterized, parameterized_class

import commonUtils
from pygeo.mphys import OM_DVGEOCOMP
from pyspline import Curve

try:
    from mphys.multipoint import Multipoint
    from openmdao.api import IndepVarComp, Problem
    from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

    mphysInstalled = True

except ImportError:
    mphysInstalled = False

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

@unittest.skipUnless(mphysInstalled, "OpenMDAO and MPhys are required to test the pyGeo MPhys wrapper")
@parameterized_class(ffd_test_params)
class TestDVGeoMPhysFFD(unittest.TestCase):
    def setUp(self):
        # give the OM Group access to the test case attributes
        dvInfo = self.dvInfo

        class FFDGroup(Multipoint):
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

        prob = Problem(model=FFDGroup())
        prob.setup(mode="rev")

        self.prob = prob
    
    def test_run_model(self):
        self.prob.run_model()

    def testDVs(self):
        self.prob.run_model()

        data = self.prob.check_totals(step=1e-7, compact_print=True)
        for _, err in data.items():

            rel_err = err["rel error"]
            assert_near_equal(rel_err.forward, 0.0, 1e-5)

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
