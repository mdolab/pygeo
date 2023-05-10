# Standard Python modules
import os
import unittest

# External modules
from baseclasses import BaseRegTest
from baseclasses.utils import Error
from mpi4py import MPI
import numpy as np

# First party modules
from pygeo import DVGeometry

try:
    # External modules
    import pysurf  # noqa: F401

    pysurfInstalled = True
except ImportError:
    pysurfInstalled = False

if pysurfInstalled:
    # First party modules
    from pygeo import DVGeometryMulti

baseDir = os.path.dirname(os.path.abspath(__file__))
inputDir = os.path.join(baseDir, "../../input_files")


@unittest.skipUnless(pysurfInstalled, "requires pySurf")
class TestDVGeoMulti(unittest.TestCase):
    N_PROCS = 1

    def train_boxes(self, train=True):
        self.test_boxes(train=train)

    def test_boxes(self, train=False):
        # box1 and box2 intersect
        # box3 does not intersect anything
        comps = ["box1", "box2", "box3"]
        ffdFiles = [os.path.join(inputDir, f"{comp}.xyz") for comp in comps]
        triMeshFiles = [os.path.join(inputDir, f"{comp}.cgns") for comp in comps]

        # Set up real component DVGeo objects
        DVGeoBox1_real = DVGeometry(ffdFiles[0])
        DVGeoBox2_real = DVGeometry(ffdFiles[1])
        DVGeoBox3_real = DVGeometry(ffdFiles[2])

        # Set up complex component DVGeo objects
        DVGeoBox1_complex = DVGeometry(ffdFiles[0], isComplex=True)
        DVGeoBox2_complex = DVGeometry(ffdFiles[1], isComplex=True)
        DVGeoBox3_complex = DVGeometry(ffdFiles[2], isComplex=True)

        # Set up real DVGeometryMulti object
        DVGeo_real = DVGeometryMulti()
        DVGeo_real.addComponent("box1", DVGeoBox1_real, triMeshFiles[0])
        DVGeo_real.addComponent("box2", DVGeoBox2_real, triMeshFiles[1])
        DVGeo_real.addComponent("box3", DVGeoBox3_real, None)

        # Set up complex DVGeometryMulti object
        DVGeo_complex = DVGeometryMulti(isComplex=True)
        DVGeo_complex.addComponent("box1", DVGeoBox1_complex, triMeshFiles[0])
        DVGeo_complex.addComponent("box2", DVGeoBox2_complex, triMeshFiles[1])
        DVGeo_complex.addComponent("box3", DVGeoBox3_complex, None)

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

        # Define a name and comm for the point set
        ptSetName = "test_set"
        comm = MPI.COMM_WORLD

        # Set up the complex and real DVGeoMulti objects
        for DVGeo in [DVGeo_complex, DVGeo_real]:
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
                anisotropy=[1.0, 1.0, 0.8],
            )

            # Add a few design variables
            DVGeoDict = DVGeo.getDVGeoDict()
            for comp in comps:
                # Create reference axis
                nRefAxPts = DVGeoDict[comp].addRefAxis("box", xFraction=0.5, alignIndex="j", rotType=4)
                nTwist = nRefAxPts - 1

                # Set up a twist variable
                def twist(val, geo, nRefAxPts=nRefAxPts):
                    for i in range(1, nRefAxPts):
                        geo.rot_z["box"].coef[i] = val[i - 1]

                DVGeoDict[comp].addGlobalDV(dvName=f"{comp}_twist", value=[0] * nTwist, func=twist)

            # Set the correct dtype for the point set
            if DVGeo == DVGeo_complex:
                pts_dtype = pts.astype(complex)
            else:
                pts_dtype = pts

            # Add the point set
            DVGeo.addPointSet(pts_dtype, ptSetName, comm=comm, applyIC=True)

            # Apply twist to the two intersecting boxes
            dvDict = DVGeo.getValues()
            dvDict["box1_twist"] = 2
            dvDict["box2_twist"] = 2
            DVGeo.setDesignVars(dvDict)

            # Update the point set
            ptsUpdated = DVGeo.update(ptSetName)

        # Regression test the updated points for the real DVGeo
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

        # Get the derivatives from the real DVGeoMulti object
        funcSens = DVGeo_real.totalSensitivity(dIdpt, ptSetName)

        # Compute FD and CS derivatives
        dvDict_real = DVGeo_real.getValues()
        funcSensFD = {}
        dvDict_complex = DVGeo_complex.getValues()
        funcSensCS = {}

        stepSize_FD = 1e-5
        stepSize_CS = 1e-200

        for x in dvDict_real:
            nx = len(dvDict_real[x])
            funcSensFD[x] = np.zeros((nx, nNodes * 3))
            funcSensCS[x] = np.zeros((nx, nNodes * 3))
            for i in range(nx):
                xRef_real = dvDict_real[x][i].copy()
                xRef_complex = dvDict_complex[x][i].copy()

                # Compute the central difference
                dvDict_real[x][i] = xRef_real + stepSize_FD
                DVGeo_real.setDesignVars(dvDict_real)
                ptsNewPlus = DVGeo_real.update(ptSetName)

                dvDict_real[x][i] = xRef_real - stepSize_FD
                DVGeo_real.setDesignVars(dvDict_real)
                ptsNewMinus = DVGeo_real.update(ptSetName)

                funcSensFD[x][i, :] = (ptsNewPlus.flatten() - ptsNewMinus.flatten()) / (2 * stepSize_FD)

                # Set the real DV back to the original value
                dvDict_real[x][i] = xRef_real.copy()

                # Compute complex step derivative
                dvDict_complex[x][i] = xRef_complex + stepSize_CS * 1j
                DVGeo_complex.setDesignVars(dvDict_complex)
                ptsNew = DVGeo_complex.update(ptSetName)

                funcSensCS[x][i, :] = np.imag(ptsNew.flatten()) / stepSize_CS

                # Set the complex DV back to the original value
                dvDict_complex[x][i] = xRef_complex.copy()

        # Check that the analytic derivatives are consistent with FD and CS
        for x in dvDict_real:
            np.testing.assert_allclose(funcSens[x].T, funcSensFD[x], rtol=1e-4, atol=1e-10)
            np.testing.assert_allclose(funcSens[x].T, funcSensCS[x], rtol=1e-4, atol=1e-10)

        # Test that adding a point outside any FFD raises an Error
        with self.assertRaises(Error):
            DVGeo.addPointSet(np.array([[-1.0, 0.0, 0.0]]), "test_error")


if __name__ == "__main__":
    unittest.main()
