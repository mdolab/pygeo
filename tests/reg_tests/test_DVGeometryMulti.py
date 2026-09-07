# Standard Python modules
from copy import deepcopy
import os
import unittest

# External modules
from baseclasses import BaseRegTest
from baseclasses.utils import Error
from mpi4py import MPI
import numpy as np
from parameterized import parameterized_class
import psutil

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

# Run the boxes test in series and in parallel
numPhysicalCores = psutil.cpu_count(logical=False)
N_PROCS_CUR = int(np.clip(numPhysicalCores, 2, 16))
test_params = [
    {
        "name": "serial",
        "N_PROCS": 1,
    },
    {
        "name": "parallel",
        "N_PROCS": N_PROCS_CUR,
    },
]


@parameterized_class(test_params)
@unittest.skipUnless(pysurfInstalled, "requires pySurf")
class TestDVGeoMulti(unittest.TestCase):
    def train_boxes(self, train=True):
        self.test_boxes(train=train)

    def test_boxes(self, train=False):
        # box1 and box2 intersect
        # box3 does not intersect anything
        comps = ["box1", "box2", "box3"]
        ffdFiles = [os.path.join(inputDir, f"{comp}.xyz") for comp in comps]
        triMeshFiles = [os.path.join(inputDir, f"{comp}.cgns") for comp in comps]

        # Define the communicator
        comm = MPI.COMM_WORLD

        # Set up real component DVGeo objects
        DVGeoBox1_real = DVGeometry(ffdFiles[0])
        DVGeoBox2_real = DVGeometry(ffdFiles[1])
        DVGeoBox3_real = DVGeometry(ffdFiles[2])

        # Set up complex component DVGeo objects
        DVGeoBox1_complex = DVGeometry(ffdFiles[0], isComplex=True)
        DVGeoBox2_complex = DVGeometry(ffdFiles[1], isComplex=True)
        DVGeoBox3_complex = DVGeometry(ffdFiles[2], isComplex=True)

        # Set up real DVGeometryMulti object
        DVGeo_real = DVGeometryMulti(comm=comm)
        DVGeo_real.addComponent("box1", DVGeoBox1_real, triMeshFiles[0])
        DVGeo_real.addComponent("box2", DVGeoBox2_real, triMeshFiles[1])
        DVGeo_real.addComponent("box3", DVGeoBox3_real, None)

        # Set up complex DVGeometryMulti object
        DVGeo_complex = DVGeometryMulti(comm=comm, isComplex=True)
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

        # Define a name for the point set
        ptSetName = "test_set"

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

        # Compute the processor sizes with integer division
        sizes = np.zeros(comm.size, dtype="intc")
        nPtsGlobal = pts.shape[0]
        sizes[:] = nPtsGlobal // comm.size

        # Add the leftovers
        sizes[: nPtsGlobal % comm.size] += 1

        # Compute the processor displacements
        disp = np.zeros(comm.size + 1, dtype="intc")
        disp[1:] = np.cumsum(sizes)

        # Split up the point set
        localPts = pts[disp[comm.rank] : disp[comm.rank + 1]]

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
                pts_dtype = localPts.astype(complex)
            else:
                pts_dtype = localPts

            # Add the point set
            DVGeo.addPointSet(pts_dtype, ptSetName, comm=comm, applyIC=True)

            # Apply twist to the two intersecting boxes
            dvDict = DVGeo.getDesignVars()
            dvDict["box1_twist"] = 2
            dvDict["box2_twist"] = 2
            DVGeo.setDesignVars(dvDict)

            # Update the point set
            ptsUpdated = DVGeo.update(ptSetName)

        # Create the send buffer
        procPoints = ptsUpdated.flatten()
        sendbuf = [procPoints, sizes[comm.rank] * 3]

        # Create the receiving buffer
        globalPoints = np.zeros(nPtsGlobal * 3)
        recvbuf = [globalPoints, sizes * 3, disp[0:-1] * 3, MPI.DOUBLE]

        # Allgather the updated coordinates
        comm.Allgatherv(sendbuf, recvbuf)

        # Reshape into a nPtsGlobal, 3 array
        ptsUpdated = globalPoints.reshape((nPtsGlobal, 3))

        # Regression test the updated points for the real DVGeo
        refFile = os.path.join(baseDir, "ref/test_DVGeometryMulti.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_add_val("ptsUpdated", ptsUpdated, tol=1e-14)

        # Now we will test the derivatives

        # Build a dIdpt array
        # We will have nPtsGlobal*3 many functions of interest
        nPtsLocal = localPts.shape[0]
        dIdpt = np.zeros((nPtsGlobal * 3, nPtsLocal, 3))

        # Set the seeds to one such that we get individual derivatives for each coordinate of each point
        # The first function of interest gets the first coordinate of the first point
        # The second function gets the second coordinate of first point, and so on
        for i in range(disp[comm.rank], disp[comm.rank + 1]):
            for j in range(3):
                dIdpt[i * 3 + j, i - disp[comm.rank], j] = 1

        # Get the derivatives from the real DVGeoMulti object
        funcSens = DVGeo_real.totalSensitivity(dIdpt, ptSetName, comm=comm)

        # Compute FD and CS derivatives
        dvDict_real = DVGeo_real.getDesignVars()
        funcSensFD = {}
        dvDict_complex = DVGeo_complex.getDesignVars()
        funcSensCS = {}

        stepSize_FD = 1e-5
        stepSize_CS = 1e-200

        for x in dvDict_real:
            nx = len(dvDict_real[x])
            funcSensFD[x] = np.zeros((nx, nPtsGlobal * 3))
            funcSensCS[x] = np.zeros((nx, nPtsGlobal * 3))
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

                funcSensFD[x][i, disp[comm.rank] * 3 : disp[comm.rank + 1] * 3] = (
                    ptsNewPlus.flatten() - ptsNewMinus.flatten()
                ) / (2 * stepSize_FD)

                # Set the real DV back to the original value
                dvDict_real[x][i] = xRef_real.copy()

                # Compute complex step derivative
                dvDict_complex[x][i] = xRef_complex + stepSize_CS * 1j
                DVGeo_complex.setDesignVars(dvDict_complex)
                ptsNew = DVGeo_complex.update(ptSetName)

                funcSensCS[x][i, disp[comm.rank] * 3 : disp[comm.rank + 1] * 3] = (
                    np.imag(ptsNew.flatten()) / stepSize_CS
                )

                # Set the complex DV back to the original value
                dvDict_complex[x][i] = xRef_complex.copy()

        # Check that the analytic derivatives are consistent with FD and CS
        for x in dvDict_real:
            funcSensFD[x] = comm.allreduce(funcSensFD[x])
            funcSensCS[x] = comm.allreduce(funcSensCS[x])

            np.testing.assert_allclose(funcSens[x].T, funcSensFD[x], rtol=1e-4, atol=1e-10)
            np.testing.assert_allclose(funcSens[x].T, funcSensCS[x], rtol=1e-4, atol=1e-10)

        # Test that adding a point outside any FFD raises an Error
        with self.assertRaises(Error):
            DVGeo.addPointSet(np.array([[-1.0, 0.0, 0.0]]), "test_error")

    def test_slidingCurves(self):
        # box1 and box2 intersect
        comps = ["box1", "box2"]
        ffdFiles = [os.path.join(inputDir, f"{comp}.xyz") for comp in comps]
        triMeshFiles = [os.path.join(inputDir, f"{comp}.cgns") for comp in comps]

        # Define the communicator
        comm = MPI.COMM_WORLD

        # Set up real component DVGeo objects
        DVGeoBox1 = DVGeometry(ffdFiles[0])
        DVGeoBox2 = DVGeometry(ffdFiles[1])

        # Set up real DVGeometryMulti object
        DVGeo = DVGeometryMulti(comm=comm)
        DVGeo.addComponent("box1", DVGeoBox1, triMeshFiles[0])
        DVGeo.addComponent("box2", DVGeoBox2, triMeshFiles[1])

        # Define some feature curves
        featureCurves = {
            # Curves on box1
            "part_22_1d": None,
            "part_23_1d": None,
            # Curves on box2
            "part_35_1d": 1,
            "part_37_1d": 1,
            "part_39_1d": 1,
        }
        curveEpsDict = {
            # Curves on box1
            "part_22_1d": 1e-3,
            "part_23_1d": 1e-3,
            # Curves on box2
            "part_35_1d": 1e-3,
            "part_37_1d": 1e-3,
            "part_39_1d": 1e-3,
            # Intersection curve
            "intersection": 1e-3,
        }

        slidingCurves = [
            # Curves on box1
            "part_22_1d",
            "part_23_1d",
        ]

        # Define a name for the point set
        ptSetName = "test_set"

        # Define a test point set
        pts = np.array(
            [
                [1.0, -0.4, 0.5],  # curve 22
                [1.0, -0.2, 0.5],  # curve 22
                [1.0, 0.2, 0.5],  # curve 23
                [1.0, 0.4, 0.5],  # curve 23
            ]
        )

        # Compute the processor sizes with integer division
        sizes = np.zeros(comm.size, dtype="intc")
        nPtsGlobal = pts.shape[0]
        sizes[:] = nPtsGlobal // comm.size

        # Add the leftovers
        sizes[: nPtsGlobal % comm.size] += 1

        # Compute the processor displacements
        disp = np.zeros(comm.size + 1, dtype="intc")
        disp[1:] = np.cumsum(sizes)

        # Split up the point set
        localPts = pts[disp[comm.rank] : disp[comm.rank + 1]]

        # Add the intersection between box1 and box2
        DVGeo.addIntersection(
            "box1",
            "box2",
            dStarA=1.0,
            dStarB=0.15,
            featureCurves=featureCurves,
            project=True,
            includeCurves=True,
            slidingCurves=slidingCurves,
            curveEpsDict=curveEpsDict,
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

        # Add the point set
        DVGeo.addPointSet(localPts, ptSetName, comm=comm, applyIC=True)

        # Apply twist to box 2
        dvDict = DVGeo.getDesignVars()
        dvDict["box2_twist"] = 10
        DVGeo.setDesignVars(dvDict)

        # Update the point set
        ptsUpdated = DVGeo.update(ptSetName)

        # Create the send buffer
        procPoints = ptsUpdated.flatten()
        sendbuf = [procPoints, sizes[comm.rank] * 3]

        # Create the receiving buffer
        globalPoints = np.zeros(nPtsGlobal * 3)
        recvbuf = [globalPoints, sizes * 3, disp[0:-1] * 3, MPI.DOUBLE]

        # Allgather the updated coordinates
        comm.Allgatherv(sendbuf, recvbuf)

        # Reshape into a nPtsGlobal, 3 array
        ptsUpdated = globalPoints.reshape((nPtsGlobal, 3))

        # Test that the X and Z coordinates are unchanged and Y coordinates are changed
        np.testing.assert_allclose(pts[:, 0], ptsUpdated[:, 0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(pts[:, 2], ptsUpdated[:, 2], rtol=1e-10, atol=1e-10)

        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(pts[:, 1], ptsUpdated[:, 1], rtol=1e-4, atol=1e-10)


@unittest.skipUnless(pysurfInstalled, "requires pySurf")
class TestDVGeoMultiEdgeCases(unittest.TestCase):
    N_PROCS = 1

    def test_trackSurfaces_shared_points(self):
        """
        Tests that points shared between two tracked surfaces are handled properly
        """

        comps = ["box1", "box2"]
        ffdFiles = [os.path.join(inputDir, f"{comp}.xyz") for comp in comps]
        triMeshFiles = [os.path.join(inputDir, f"{comp}.cgns") for comp in comps]

        # Set up component DVGeo objects
        DVGeoBox1 = DVGeometry(ffdFiles[0])
        DVGeoBox2 = DVGeometry(ffdFiles[1])

        # Set up DVGeometryMulti object
        DVGeo = DVGeometryMulti()
        DVGeo.addComponent("box1", DVGeoBox1, triMeshFiles[0])
        DVGeo.addComponent("box2", DVGeoBox2, triMeshFiles[1])

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

        # Track adjacent surfaces on both components
        trackSurfaces = {
            # box1
            "part_14": 1e-3,
            "part_15": 1e-3,
            # box2
            "part_38": 1e-3,
            "part_39": 1e-3,
        }

        # Define a test point set
        pts = np.array(
            [
                # Point on the curve between tracked surfaces part_14 and part_15, box1
                [0.15, 0, 0.5],
                # Point on the curve between tracked surfaces part_38 and part_39, box2
                [0.25, 0, 0.6],
            ]
        )

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
        )

        # Define a name and comm for the point set
        ptSetName = "test_set"
        comm = MPI.COMM_WORLD

        # Add the point set
        DVGeo.addPointSet(pts, ptSetName, comm=comm, applyIC=True)

        # Check that updating the point set runs without errors
        DVGeo.update(ptSetName)


# fillet doesn't require pySurf
class TestDVGeoMultiFillet(unittest.TestCase):
    def set_up_fillet(self, cmplx):
        # Define the communicator
        comm = MPI.COMM_WORLD

        # test FFDs
        input_file_dir = os.path.join(os.path.dirname(__file__), "..", "..", "input_files")
        compA_FFD = os.path.join(input_file_dir, "compA.xyz")
        compB_FFD = os.path.join(input_file_dir, "compB.xyz")

        # manual definition of surface and curve pointsets
        compAPtSet = np.array(((-3.0, 0.0, 0.0), (-2.0, 0.0, 0.0)), dtype=float)
        compBPtSet = np.array(((3.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=float)
        filletPtSet = np.array(((-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=float)

        compACurve = np.array(((-2.0, 0.0, 0.0)), dtype=float)
        compBCurve = np.array(((2.0, 0.0, 0.0)), dtype=float)

        # create DVGeo objects
        compADVGeo = DVGeometry(compA_FFD, child=False, isComplex=cmplx)
        compBDVGeo = DVGeometry(compB_FFD, child=False, isComplex=cmplx)
        DVGeo = DVGeometryMulti(filletIntersection=True, debug=False, isComplex=cmplx, comm=comm)

        # add components to DVGeo
        DVGeo.addComponent("compA", DVGeo=compADVGeo)
        DVGeo.addComponent("compB", DVGeo=compBDVGeo)
        DVGeo.addComponent("fillet", DVGeo=None)

        # set up DVs
        compACtlPts = compADVGeo.getLocalIndex(0)
        compBCtlPts = compBDVGeo.getLocalIndex(0)

        compAShapes = [{compACtlPts[1, 1, 1]: np.array((0, 0, 1))}]
        compBShapes = [{compBCtlPts[1, 1, 1]: np.array((0, 1, 1))}]

        compADVGeo.addShapeFunctionDV("shapeA", compAShapes, lower=-1, upper=1)
        compBDVGeo.addShapeFunctionDV("shapeB", compBShapes, lower=-1, upper=1)

        # set up intersection
        DVGeo.addIntersection("compA", "compB", "fillet")
        DVGeo.addCurve("compA", curvePtsArray=compACurve)
        DVGeo.addCurve("compB", curvePtsArray=compBCurve)

        # add pointsets
        compAPtSetName = "compA_surf_points"
        compBPtSetName = "compB_surf_points"
        filletPtSetName = "fillet_surf_points"
        ptSets = [compAPtSetName, compBPtSetName, filletPtSetName]

        DVGeo.addPointSet(compAPtSet, compAPtSetName, familyName="compA", applyIC=True)
        DVGeo.addPointSet(compBPtSet, compBPtSetName, familyName="compB", applyIC=True)
        DVGeo.addPointSet(filletPtSet, filletPtSetName, familyName="fillet", applyIC=True)

        return DVGeo, ptSets

    def apply_DV(self, DVGeo, ptSetNames, val1, val2=None):
        dvDict = DVGeo.getDesignVars()
        dvDict.update({"shapeA": val1})
        if val2 is not None:
            dvDict.update({"shapeB": val2})
        DVGeo.setDesignVars(dvDict)

        [DVGeo.update(name) for name in ptSetNames]

    def deriv_fd(self, pts, ptSetName, DVGeo, filletPtSetName, fillet):
        nNodes = pts.shape[0]
        dIdpt = np.zeros((nNodes * 3, nNodes, 3))

        for i in range(nNodes):
            for j in range(3):
                dIdpt[i * 3 + j, i, j] = 1

        funcSens = DVGeo.totalSensitivity(dIdpt, ptSetName)

        dvDict_real = DVGeo.getDesignVars()
        funcSensFD = {}

        stepSize_FD = 1e-5
        nNodes = pts.shape[0]

        dvList = dvDict_real.keys()

        for x in dvList:
            nx = len(dvDict_real[x])
            funcSensFD[x] = np.zeros((nx, nNodes * 3))

            for i in range(nx):
                xRef_real = deepcopy(dvDict_real[x][i])

                # Compute the central difference
                dvDict_real[x][i] = xRef_real + stepSize_FD
                DVGeo.setDesignVars(dvDict_real)
                ptsNewPlus = DVGeo.update(ptSetName).copy()

                dvDict_real[x][i] = xRef_real - stepSize_FD
                DVGeo.setDesignVars(dvDict_real)
                ptsNewMinus = DVGeo.update(ptSetName).copy()

                funcSensFD[x][i, :] = (ptsNewPlus.flatten() - ptsNewMinus.flatten()) / (2 * stepSize_FD)

                # Set the real DV back to the original value
                dvDict_real[x][i] = deepcopy(xRef_real)

        # zero out the points on the curve if this is the fillet pointset
        if ptSetName is filletPtSetName:
            allInd = deepcopy(fillet.compAInterInd)
            allInd.extend(fillet.compBInterInd)

            for x in dvDict_real:
                deriv = funcSensFD[x].T
                for i in range(nNodes):
                    if i in (allInd):
                        deriv[3 * i : 3 * i + 3] = 0
                funcSensFD[x] = deriv.T

        return funcSens, funcSensFD, dvDict_real

    def test_deform(self):
        # set up non-complex DVGeo
        DVGeo, ptSetNames = self.set_up_fillet(False)
        compAPtSet_orig = DVGeo.points[ptSetNames[0]].points
        compBPtSet_orig = DVGeo.points[ptSetNames[1]].points
        filletPtSet_orig = DVGeo.points[ptSetNames[2]].points

        # apply deformation to DV
        self.apply_DV(DVGeo, ptSetNames, 5, 0)
        compAPtSet_updated = DVGeo.points[ptSetNames[0]].points
        compBPtSet_updated = DVGeo.points[ptSetNames[1]].points
        filletPtSet_updated = DVGeo.points[ptSetNames[2]].points

        # component A and fillet should change in z coordinates
        np.testing.assert_array_less(compAPtSet_orig[:, 2], compAPtSet_updated[:, 2])
        np.testing.assert_array_less(filletPtSet_orig[:, 2], filletPtSet_updated[:, 2])

        # component B shouldn't move at all
        np.testing.assert_allclose(compBPtSet_orig, compBPtSet_updated, rtol=1e-8, atol=1e-8)

        # fillet points should overlap with the component "curves"
        np.testing.assert_allclose(compAPtSet_updated[1], filletPtSet_updated[0], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(compBPtSet_updated[1], filletPtSet_updated[2], rtol=1e-6, atol=1e-8)

    def test_deform_cmplx(self):
        # set up complex DVGeo
        DVGeo, ptSetNames = self.set_up_fillet(True)
        compAPtSet_orig = DVGeo.points[ptSetNames[0]].points
        compBPtSet_orig = DVGeo.points[ptSetNames[1]].points
        filletPtSet_orig = DVGeo.points[ptSetNames[2]].points

        # apply deformation to DV
        self.apply_DV(DVGeo, ptSetNames, 5, 0)
        compAPtSet_updated = DVGeo.points[ptSetNames[0]].points
        compBPtSet_updated = DVGeo.points[ptSetNames[1]].points
        filletPtSet_updated = DVGeo.points[ptSetNames[2]].points

        # component A and fillet should change in z coordinates
        np.testing.assert_array_less(compAPtSet_orig[:, 2], compAPtSet_updated[:, 2])
        np.testing.assert_array_less(filletPtSet_orig[:, 2], filletPtSet_updated[:, 2])

        # component B shouldn't move at all
        np.testing.assert_allclose(compBPtSet_orig, compBPtSet_updated, rtol=1e-8, atol=1e-8)

        # fillet points should overlap with the component "curves"
        np.testing.assert_allclose(compAPtSet_updated[1], filletPtSet_updated[0], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(compBPtSet_updated[1], filletPtSet_updated[2], rtol=1e-6, atol=1e-8)

    def test_deriv_compA(self):
        DVGeo, ptSetNames = self.set_up_fillet(False)
        funcSens, funcSensFD, dvDict_real = self.deriv_fd(
            DVGeo.points[ptSetNames[0]].points, ptSetNames[0], DVGeo, ptSetNames[2], DVGeo.comps["fillet"]
        )

        for x in dvDict_real:
            np.testing.assert_allclose(funcSens[x].T, funcSensFD[x], rtol=1e-4, atol=1e-10)

    def test_deriv_fillet(self):
        DVGeo, ptSetNames = self.set_up_fillet(False)
        funcSens, funcSensFD, dvDict_real = self.deriv_fd(
            DVGeo.points[ptSetNames[2]].points, ptSetNames[2], DVGeo, ptSetNames[2], DVGeo.comps["fillet"]
        )
        for x in dvDict_real:
            np.testing.assert_allclose(funcSens[x].T, funcSensFD[x], rtol=1e-4, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
