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
    from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal, assert_check_totals

    mphysInstalled = True

except ImportError:
    mphysInstalled = False

input_path = os.path.dirname(os.path.abspath(__file__))
parentFFDFile = os.path.join(input_path, "../../input_files/outerBoxFFD.xyz")
childFFDFile = os.path.join(input_path, "../../input_files/simpleInnerFFD.xyz")
espBox = os.path.join(input_path, "../input_files/esp/box.csm")

ffdPoints = np.array(())

globalDVFuncParams = ["mainX", -1.0, commonUtils.mainAxisPoints]
localDVFuncParams = ["xdir"]

test_params = [
    {"name": "MPhys_FFD_global", "funcNames": ["nom_addGlobalDV"], "funcParams": [globalDVFuncParams], "lower": [-1.0], "upper": [1.0], "val": [-1.0]},
    {"name": "MPhys_FFD_local", "funcNames": ["nom_addLocalDV"], "funcParams": [localDVFuncParams], "lower": [-1.0], "upper": [1.0], "val": [12*[0.0]]},
]

@unittest.skipUnless(mphysInstalled, "OpenMDAO and MPhys are required to test the pyGeo MPhys wrapper")
@parameterized_class(test_params)
class TestDVGeoMPhysFFD(unittest.TestCase):
    def setUp(self):
        # give the OM Group access to the test case attributes
        # points = self.points
        # ptName = "pts"
        funcNames = self.funcNames
        funcParams = self.funcParams
        upper = self.upper
        lower = self.lower
        val = self.val

        class FFDGroup(Multipoint):
            def setup(self):
                self.add_subsystem("dvs", IndepVarComp(), promotes=["*"])
                self.add_subsystem("geometry", OM_DVGEOCOMP(file=parentFFDFile, type="ffd"))

            def configure(self):
                self.geometry.nom_addChild(childFFDFile, childName="child")

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
                self.geometry.nom_addRefAxis("nestedAxis", childName="child", curve=c1, axis="y")

                for ii, func in enumerate(funcNames):
                    dvName = funcParams[ii][0]
                    getattr(self.geometry, func)(*funcParams[ii])
                    
                    self.dvs.add_output(dvName, val[ii])
                    self.connect(dvName, f"geometry.{dvName}")
                    self.add_design_var(dvName, upper=upper[ii], lower=lower[ii])

                self.add_constraint(f"geometry.{ptName}")

        prob = Problem(model=FFDGroup())
        prob.setup(mode="rev", force_alloc_complex=True)

        self.prob = prob
    
    def test_run_model(self):
        self.prob.run_model()

    def testDVs(self):
        self.prob.run_model()

        totals = self.prob.check_totals(step=1e-7, compact_print=True)
        assert_check_totals(totals)

if __name__ == "__main__":
    unittest.main()
