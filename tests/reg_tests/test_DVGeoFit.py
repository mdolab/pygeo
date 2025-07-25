"""
==============================================================================
DVGeometry least-squares fitting tests
==============================================================================
@File    :   test_DVGeoFit.py
@Date    :   2025/07/25
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import unittest
import numpy as np

# ==============================================================================
# External Python modules
# ==============================================================================
from stl import mesh
from pygeo import DVGeometry


class TestDVGeoFit(unittest.TestCase):
    N_PROCS = 1

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.input_folder = os.path.abspath(os.path.join(self.base_path, "..", "..", "input_files"))
        np.random.seed(0)

    def setupDVGeo(self, ffdFile, pts):
        """Setup a DVGeometry object for the C172 case"""
        DVGeo = DVGeometry(ffdFile)

        # Setup the reference axis
        nRefAxPts = DVGeo.addRefAxis("wing", xFraction=0.25, alignIndex="k")
        nTwist = nRefAxPts - 1

        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z["wing"].coef[i] = val[i - 1]

        # Add some DVs
        DVGeo.addGlobalDV(dvName="twist", value=[0] * nTwist, func=twist, lower=-10, upper=10, scale=1)
        DVGeo.addLocalDV("local", lower=-0.5, upper=0.5, axis="y", scale=1.0)

        DVGeo.addPointSet(pts, "test_pts")

        return DVGeo

    def test_c172_fit(self):
        """Fit a C172 FFD to a deformed version of itself"""
        # ==============================================================================
        # Setup the FFD
        # ==============================================================================
        ffdFile = os.path.join(self.input_folder, "c172.xyz")
        meshFile = os.path.join(self.input_folder, "c172.stl")
        testMesh = mesh.Mesh.from_file(meshFile)
        pts = testMesh.vectors.reshape(-1, 3) / 1000.0

        DVGeo = self.setupDVGeo(ffdFile, pts)

        # ==============================================================================
        # Create a copy of the FFD and deform it
        # ==============================================================================
        DVGeo_deformed = self.setupDVGeo(ffdFile, pts)

        # Set random design variables
        xDV = DVGeo_deformed.getDesignVars()
        lowerBounds, uppperBounds = DVGeo_deformed.getDVBounds()

        randomDVs = {}
        for key in xDV.keys():
            randomDVs[key] = np.random.uniform(lowerBounds[key], uppperBounds[key], size=xDV[key].size)
        DVGeo_deformed.setDesignVars(randomDVs)

        # ==============================================================================
        # Fit the original FFD to the deformed points
        # ==============================================================================
        xDV_fit_dict, result = DVGeo.fitDVGeo(DVGeo_deformed, "test_pts", xtol=1e-6, ftol=1e-6, gtol=1e-6)
        DVGeo.setDesignVars(xDV_fit_dict)

        # ==============================================================================
        # Check the results
        # ==============================================================================
        # Check that the deformed points are the same
        new_pts = DVGeo.update("test_pts")
        deformed_pts = DVGeo_deformed.update("test_pts")
        np.testing.assert_allclose(new_pts, deformed_pts, rtol=1e-4, atol=1e-4)

        # Check that the design variables are the same
        xDV_fit = DVGeo.getDesignVars()
        for key in xDV.keys():
            np.testing.assert_allclose(xDV[key], xDV_fit[key], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
