# Standard Python modules
import os
import unittest

# External modules
from baseclasses import BaseRegTest
import numpy as np

# First party modules
from pygeo import pyGeo

baseDir = os.path.dirname(os.path.abspath(__file__))


class TestPyGeo(unittest.TestCase):
    def setUp(self):
        self.refFile = os.path.join(baseDir, "ref/test_pyGeo.ref")

    def train(self):
        with BaseRegTest(self.refFile, train=True) as handler:
            self.regTest(handler, clash=False)
            self.regTest(handler, clash=True)
            handler.writeRef()

    def test_original(self):
        with BaseRegTest(self.refFile, train=False) as handler:
            self.regTest(handler, clash=False)

    def test_clash(self):
        with BaseRegTest(self.refFile, train=False) as handler:
            self.regTest(handler, clash=True)

    def regTest(self, handler, clash=False):
        inputDir = os.path.join(baseDir, "../../input_files")

        if clash:
            # This case, which uses two different airfoils with very similar point distributions, lead to an error due
            # to floating point precision issues in the clash detection algorithm. So now we test that it runs without
            # error.
            airfoil_list = [os.path.join(inputDir, "CRMTailRoot.dat"), os.path.join(inputDir, "CRMTailTip.dat")]
        else:
            airfoil_list = [os.path.join(inputDir, "rae2822.dat")] * 2

        naf = len(airfoil_list)

        # Wing definition (Common to both)
        x = [0.0, 7.5]
        y = [0.0, 0.0]
        z = [0.0, 14.0]
        offset = np.zeros((naf, 2))
        rot_x = [0.0, 0.0]
        rot_y = [0.0, 0.0]
        rot_z = [0.0, 0.0]
        chord = [5.0, 1.5]

        # Run pyGeo
        wing = pyGeo(
            "liftingSurface",
            xsections=airfoil_list,
            scale=chord,
            offset=offset,
            x=x,
            y=y,
            z=z,
            rotX=rot_x,
            rotY=rot_y,
            rotZ=rot_z,
            tip="rounded",
            bluntTe=True,
            squareTeTip=True,
            teHeight=0.25 * 0.0254,
        )

        for isurf in range(wing.nSurf):
            wing.surfs[isurf].computeData()
        surf = wing.surfs[isurf].data
        label = "clash" if clash else "original"
        handler.root_add_val(f"{label} sum of surface data", sum(surf.flatten()), tol=1e-10)


if __name__ == "__main__":
    unittest.main()
