"""
==============================================================================
DVGeometry least-squares fitting tests
==============================================================================
@File    :   test_DVGeoFit.py
@Date    :   2025/07/25
@Author  :   Alasdair Christison Gray
@Description :
"""

# Standard Python modules
import os
import unittest

# External modules
import numpy as np
from stl import mesh
from mpi4py import MPI

# First party modules
from pygeo import DVGeometry


class TestDVGeoFit(unittest.TestCase):
    N_PROCS = 2
    ptSetName = "test_pts"
    comm = MPI.COMM_WORLD

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.input_folder = os.path.abspath(os.path.join(self.base_path, "..", "..", "input_files"))
        np.random.seed(0)

        # ==============================================================================
        # Setup the FFD
        # ==============================================================================
        self.ffdFile = os.path.join(self.input_folder, "c172_fitted.xyz")
        self.meshFile = os.path.join(self.input_folder, "c172.stl")
        self.testMesh = mesh.Mesh.from_file(self.meshFile)
        self.pts = self.testMesh.vectors.reshape(-1, 3) / 1000.0
        # Split the points between the two procs
        nPts = self.pts.shape[0]
        numProcs = self.comm.size
        if numProcs > 1:
            rank = self.comm.rank
            startInd = rank * (nPts // numProcs)
            endInd = max((rank + 1) * (nPts // numProcs), nPts)
            self.pts = self.pts[startInd:endInd]

        self.DVGeoRef = self.setupDVGeo(self.ffdFile, self.pts)
        self.DVGeoFit = self.setupDVGeo(self.ffdFile, self.pts)
        self.origDVs = self.DVGeoRef.getDesignVars()
        lowerBounds, uppperBounds = self.DVGeoRef.getDVBounds()

        self.perturbedDVs = {}
        for key in self.origDVs.keys():
            self.perturbedDVs[key] = np.random.uniform(lowerBounds[key], uppperBounds[key], size=self.origDVs[key].size)

    def setupDVGeo(self, ffdFile, pts):
        """Setup a DVGeometry object for the C172 case"""
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
        DVGeo.addGlobalDV(dvName="twist", value=[0] * nTwist, func=twist, lower=-10, upper=10, scale=1)
        DVGeo.addGlobalDV(dvName="span", value=0.0, func=span, lower=-2, upper=2, scale=1)
        DVGeo.addLocalDV("local", lower=-0.05, upper=0.05, axis="y", scale=1.0)

        DVGeo.addPointSet(pts, self.ptSetName)

        return DVGeo

    def test_c172_perfect_fit_same_dvs(self):
        """
        Test that if we start from the same parameterization and DVs, the fitting should return the same values
        """

        self.DVGeoRef.setDesignVars(self.perturbedDVs)
        self.DVGeoFit.setDesignVars(self.perturbedDVs)
        fitDVs, result = self.DVGeoFit.fitDVGeo(self.DVGeoRef, self.ptSetName, xtol=1e-6, ftol=1e-6, gtol=1e-4)
        self.assertEqual(result.nfev, 1)
        self.assertEqual(result.success, True)

        # Check that the design variables are the same
        for key in fitDVs.keys():
            np.testing.assert_array_equal(fitDVs[key], self.perturbedDVs[key])
        # Check that the deformed points are the same
        ref_pts = self.DVGeoRef.update(self.ptSetName)
        fit_pts = self.DVGeoFit.update(self.ptSetName)
        np.testing.assert_array_equal(ref_pts, fit_pts)

    def test_c172_perfect_fit_diff_dvs(self):
        """
        Test that if we start from the same parameterization but with different DVs, the fitting should converge to the
        same geometry and similar DV values.
        """
        self.DVGeoRef.setDesignVars(self.perturbedDVs)
        fitDVs, result = self.DVGeoFit.fitDVGeo(
            self.DVGeoRef, self.ptSetName, xtol=1e-6, ftol=1e-6, gtol=1e-5, max_nfev=40
        )
        self.assertEqual(result.success, True)
        # Check that the deformed points are the same
        ref_pts = self.DVGeoRef.update(self.ptSetName)
        fit_pts = self.DVGeoFit.update(self.ptSetName)
        np.testing.assert_allclose(ref_pts, fit_pts, rtol=np.inf, atol=1e-5)

        # Check that the design variables are somewhat similar
        for key in fitDVs.keys():
            np.testing.assert_allclose(fitDVs[key], self.perturbedDVs[key], rtol=0.05, atol=0.05)


if __name__ == "__main__":
    unittest.main()
