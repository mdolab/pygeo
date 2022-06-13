# Imports
import numpy as np
from pygeo import pyGeo
import unittest
import os
from baseclasses import BaseRegTest

baseDir = os.path.dirname(os.path.abspath(__file__))


class TestPyGeo(unittest.TestCase):
    def setUp(self):
        self.refFile = os.path.join(baseDir, "ref/test_pyGeo.ref")

    def train(self):
        with BaseRegTest(self.refFile, train=True) as handler:
            self.regTest(handler)
            handler.writeRef()

    def test(self):
        with BaseRegTest(self.refFile, train=False) as handler:
            self.regTest(handler)

    def regTest(self, handler):
        dirName = os.path.join(baseDir, "../../input_files")

        # Airfoil file
        airfoil_list = [dirName + "/rae2822.dat"] * 2
        naf = len(airfoil_list)  # number of airfoils

        # Wing definition
        # Airfoil leading edge positions
        x = [0.0, 7.5]
        y = [0.0, 0.0]
        z = [0.0, 14.0]
        offset = np.zeros((naf, 2))  # x-y offset applied to airfoil position before scaling

        # Airfoil rotations
        rot_x = [0.0, 0.0]
        rot_y = [0.0, 0.0]
        rot_z = [0.0, 0.0]

        # Airfoil scaling
        chord = [5.0, 1.5]  # chord lengths

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
        handler.root_add_val("sum of surface data", sum(surf.flatten()), tol=1e-10)
