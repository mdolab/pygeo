import os
import unittest
import numpy as np
from mpi4py import MPI
from baseclasses import BaseRegTest
from baseclasses.utils import Error
from pygeo import DVGeometry

try:
    from pygeo import DVGeometryMulti

    missing_pysurf = False
except ImportError:
    missing_pysurf = True

baseDir = os.path.dirname(os.path.abspath(__file__))
inputDir = os.path.join(baseDir, "../../input_files")


@unittest.skipIf(missing_pysurf, "requires pySurf")
class TestDVGeoMulti(unittest.TestCase):

    N_PROCS = 1

    def test_boxes(self, train=False):

        # box1 and box2 intersect
        # box3 does not intersect anything
        comps = ["box1", "box2", "box3"]
        ffdFiles = [os.path.join(inputDir, f"{comp}.xyz") for comp in comps]
        triMeshFiles = [os.path.join(inputDir, f"{comp}.cgns") for comp in comps]

        # Set up component DVGeo objects
        DVGeoBox1 = DVGeometry(ffdFiles[0])
        DVGeoBox2 = DVGeometry(ffdFiles[1])
        DVGeoBox3 = DVGeometry(ffdFiles[2])

        # Set up DVGeometryMulti object
        DVGeo = DVGeometryMulti()
        DVGeo.addComponent("box1", DVGeoBox1, triMeshFiles[0])
        DVGeo.addComponent("box2", DVGeoBox2, triMeshFiles[1])
        DVGeo.addComponent("box3", DVGeoBox3, None)

        # Define some feature curves
        featureCurves = [
            # Curves on box1
            "part_15_1d",
            # Curves on box2
            "part_35_1d",
            "part_37_1d",
            "part_39_1d",
        ]
        curveEpsDict = {
            # Curves on box1
            "part_15_1d": 1e-3,
            # Curves on box2
            "part_35_1d": 1e-3,
            "part_37_1d": 1e-3,
            "part_39_1d": 1e-3,
            # Intersection curve
            "intersection": 1e-3,
        }

        # Track some intersecting surfaces
        trackSurfaces = {
            # box1
            "part_14": 1e-3,
            # box2
            "part_39": 1e-3,
        }

        # Exclude some intersecting surfaces
        excludeSurfaces = {
            # box1
            "part_15": 1e-3,
            # box2
            "part_40": 1e-3,
        }

        # Add the intersection between box1 and box2
        DVGeo.addIntersection(
            "box1",
            "box2",
            dStarA=0.15,
            dStarB=0.15,
            featureCurves=featureCurves,
            project=True,
            includeCurves=True,
            curveEpsDict=curveEpsDict,
            trackSurfaces=trackSurfaces,
            excludeSurfaces=excludeSurfaces,
        )

        # Add a few design variables
        DVGeoDict = DVGeo.getDVGeoDict()
        for comp in comps:
            # Create reference axis
            nRefAxPts = DVGeoDict[comp].addRefAxis("box", xFraction=0.5, alignIndex="j", rotType=4)
            nTwist = nRefAxPts - 1

            # Set up a twist variable
            def twist(val, geo):
                for i in range(1, nRefAxPts):
                    geo.rot_z["box"].coef[i] = val[i - 1]

            DVGeoDict[comp].addGlobalDV(dvName=f"{comp}_twist", value=[0] * nTwist, func=twist)

        # Define a test point set
        pts = np.array(
            [
                # Points on box1 away from intersection
                [0.0, 0.0, 0.0],
                [0.5, 0.1, -0.5],
                # Points on box2 away from intersection
                [0.5, 0.0, 2.0],
                [0.375, 0.125, 2.0],
                # Point on box3
                [2.5, 0.5, 0.0],
                # Point on curve part_15_1d
                [0.3, 0.0, 0.5],
                # Point on curve part_35_1d
                [0.75, -0.25, 0.6],
                # Point on curve part_37_1d
                [0.25, -0.25, 0.6],
                # Point on curve part_39_1d
                [0.25, 0.25, 0.6],
                # Point on intersection curve
                [0.25, 0.1, 0.5],
                # Point on tracked surface part_14, box1
                [0.5, -0.3, 0.5],
                # Point on tracked surface part_39, box2
                [0.25, 0.12, 0.51],
                # Point on excluded surface part_15, box1
                [0.5, 0.3, 0.5],
                # Point on excluded surface part_40, box2
                [0.375, 0.25, 0.6],
                # Other points near the intersection
                [0.25, 0.251, 0.5],
                [0.5, 0.251, 0.5],
                [0.51, 0.25, 0.4],
                [0.75, 0.25, 0.6],
                [0.5, 0.25, 0.6],
                [0.25, 0.5, 0.6],
                [0.5, -0.25, 0.6],
                [0.25, -0.5, 0.6],
            ]
        )

        # Add the point set
        ptSetName = "test_set"
        comm = MPI.COMM_WORLD
        DVGeo.addPointSet(pts, ptSetName, comm=comm, applyIC=True)

        # Apply twist to the two intersecting boxes
        dvDict = DVGeo.getValues()
        dvDict["box1_twist"] = 2
        dvDict["box2_twist"] = 2
        DVGeo.setDesignVars(dvDict)

        # Update the point set
        ptsUpdated = DVGeo.update(ptSetName)

        # Regression test the updated points
        refFile = os.path.join(baseDir, "ref/test_DVGeometryMulti.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.par_add_val("ptsUpdated", ptsUpdated, tol=1e-14)

        # Now we will test the derivatives

        # Build a dIdpt array
        # We will have nNodes*3 many functions of interest
        nNodes = pts.shape[0]
        dIdpt = np.zeros((nNodes * 3, nNodes, 3))

        # Set the seeds to one such that we get individual derivatives for each coordinate of each point
        # The first function of interest gets the first coordinate of the first point
        # The second function gets the second coordinate of first point, and so on
        for i in range(nNodes):
            for j in range(3):
                dIdpt[i * 3 + j, i, j] = 1

        # Get the derivatives from DVGeo
        funcSens = DVGeo.totalSensitivity(dIdpt, ptSetName)

        # Perturb the DVs with finite differences and compute FD gradients
        dvDict = DVGeo.getValues()
        funcSensFD = {}
        dh = 1e-5

        for x in dvDict:

            nx = len(dvDict[x])
            funcSensFD[x] = np.zeros((nx, nNodes * 3))
            for i in range(nx):

                xRef = dvDict[x][i].copy()

                # Compute the central difference
                dvDict[x][i] = xRef + dh
                DVGeo.setDesignVars(dvDict)
                ptsNewPlus = DVGeo.update(ptSetName)

                dvDict[x][i] = xRef - dh
                DVGeo.setDesignVars(dvDict)
                ptsNewMinus = DVGeo.update(ptSetName)

                funcSensFD[x][i, :] = (ptsNewPlus.flatten() - ptsNewMinus.flatten()) / (2 * dh)

                # Set the DV back to the original value
                dvDict[x][i] = xRef.copy()

        # Check that the analytic derivatives are consistent with FD
        for x in dvDict:
            np.testing.assert_allclose(funcSens[x].T, funcSensFD[x], rtol=1e-4, atol=1e-10)

        # Test that adding a point outside any FFD raises an Error
        with self.assertRaises(Error):
            DVGeo.addPointSet(np.array([[-1.0, 0.0, 0.0]]), "test_error")


if __name__ == "__main__":
    unittest.main()
