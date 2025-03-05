# Standard Python modules
from collections import OrderedDict
import copy
import os
import time
import unittest

# External modules
from baseclasses import BaseRegTest
from baseclasses.utils import Error
import numpy as np
from parameterized import parameterized_class
import psutil
from stl import mesh

try:
    # External modules
    from mpi4py import MPI

    mpiInstalled = True
except ImportError:
    mpiInstalled = False

if mpiInstalled:
    try:
        # External modules
        import pyOCSM  # noqa

        # First party modules
        from pygeo import DVGeometryESP

        ocsmInstalled = True
    except ImportError:
        ocsmInstalled = False


numPhysicalCores = psutil.cpu_count(logical=False)
N_PROCS_CUR = int(np.clip(numPhysicalCores, 2, 16))
test_params = [{"N_PROCS": 1, "name": "serial"}, {"N_PROCS": N_PROCS_CUR, "name": "parallel"}]


@unittest.skipUnless(mpiInstalled and ocsmInstalled, "MPI and pyOCSM are required.")
@parameterized_class(test_params)
class TestPyGeoESP_BasicCube(unittest.TestCase):
    # to be tested in serial and parallel automatically
    N_PROCS = 1

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.input_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def setup_cubemodel(self):
        # load the box model and build the box model
        csmFile = os.path.join(self.input_path, "../input_files/esp/box.csm")
        DVGeo = DVGeometryESP(csmFile)
        self.assertIsNotNone(DVGeo)

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
        distglobal = DVGeo.addPointSet(initpts, "mypts", cache_projections=False)
        self.assertAlmostEqual(distglobal, 0.0, 8)

        # evaluate the points and check that they match
        DVGeo._updateModel()
        DVGeo._updateProjectedPts()
        self.assertTrue(DVGeo.pointSetUpToDate)
        self.assertAlmostEqual(np.linalg.norm(initpts - DVGeo.pointSets["mypts"].proj_pts), 0.0, 10)

        return DVGeo, initpts

    def setup_cubemodel_analytic_jac(self):
        jacpt0 = np.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]  # x  # y
        )  # z
        jacpt1 = np.array(
            [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]  # x  # y
        )  # z
        jacpt2 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.9 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.9 / 3.5],
            ]
        )  # z
        jacpt3 = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.8 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.9 / 3.5],
            ]
        )  # z
        jacpt4 = np.array(
            [
                [1.0, 0.0, 0.0, 2.25 / 3.50, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 2.30 / 3.50],
            ]
        )  # z
        jacpt5 = np.array(
            [
                [1.0, 0.0, 0.0, 3.20 / 3.50, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.70 / 3.50],
            ]
        )  # z
        jacpt6 = np.array(
            [
                [1.0, 0.0, 0.0, 2.0 / 3.5, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 2.1 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )  # z
        jacpt7 = np.array(
            [
                [1.0, 0.0, 0.0, 0.1 / 3.5, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.9 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )  # z
        ordered_analytic_jac = np.concatenate(
            [jacpt0, jacpt1, jacpt2, jacpt3, jacpt4, jacpt5, jacpt6, jacpt7, jacpt2, jacpt3], axis=0
        ).reshape(10, 3, 6)
        return ordered_analytic_jac

    def test_load_a_model(self):
        # load the box model and build the box model
        csmFile = os.path.join(self.input_path, "../input_files/esp/box.csm")
        DVGeometryESP(csmFile)

    def test_save_cadfile(self):
        write_fullpath = os.path.join(self.input_path, "reg_tests/fullpath_" + str(self.N_PROCS) + ".step")
        DVGeo, initpts = self.setup_cubemodel()
        if DVGeo.comm.rank == 0:
            try:
                os.remove(write_fullpath)
            except OSError:
                pass
        DVGeo.writeCADFile(write_fullpath)
        DVGeo.comm.barrier()
        time.sleep(0.1)
        self.assertTrue(os.path.exists(write_fullpath))

        # check that bad file extension raises a Python error
        with self.assertRaises(IOError):
            DVGeo.writeCADFile("relpath.wrongext")

    def test_write_csmfile(self):
        DVGeo, initpts = self.setup_cubemodel()
        write_fullpath = os.path.join(self.input_path, "reg_tests/fullpath_" + str(self.N_PROCS) + ".csm")
        if DVGeo.comm.rank == 0:
            try:
                os.remove(write_fullpath)
            except OSError:
                pass
        DVGeo.writeCSMFile(write_fullpath)
        DVGeo.comm.barrier()
        time.sleep(0.1)
        self.assertTrue(os.path.exists(write_fullpath))
        # check that bad file extension raises a Python error
        with self.assertRaises(IOError):
            DVGeo.writeCADFile("relpath.wrongext")

    def test_add_desvars(self):
        # load the box model and build the box model
        csmFile = os.path.join(self.input_path, "../input_files/esp/box.csm")
        DVGeo = DVGeometryESP(csmFile)
        self.assertIsNotNone(DVGeo)

        # add variables with a mix of optional arguments
        DVGeo.addVariable("cubex0", lower=np.array([-10.0]), upper=np.array([10.0]), scale=0.1, dh=0.0001)
        self.assertEqual(DVGeo.getNDV(), 1)
        DVGeo.addVariable("cubey0")
        self.assertEqual(DVGeo.getNDV(), 2)
        DVGeo.addVariable("cubez0", lower=np.array([-10.0]), upper=np.array([10.0]))
        self.assertEqual(DVGeo.getNDV(), 3)

        # try to add a variable that isn't in the CSM file
        with self.assertRaises(Error):
            DVGeo.addVariable("cubew0")

    def test_add_pointset(self):
        DVGeo, initpts = self.setup_cubemodel()

    def test_updated_points(self):
        DVGeo, initpts = self.setup_cubemodel()

        DVGeo.addVariable("cubey0")
        DVGeo.setDesignVars({"cubey0": np.array([4.2000])}, updateJacobian=False)
        npts = initpts.shape[0]
        self.assertAlmostEqual(np.sum(DVGeo.pointSets["mypts"].proj_pts[:, 1] - initpts[:, 1]) / npts, 6.2, 10)
        DVGeo.addVariable("cubedz")
        DVGeo.setDesignVars({"cubedz": np.array([9.5])}, updateJacobian=False)
        self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[1, 2], 7.5)
        self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[0, 2], -2.0)

    def test_finite_precision(self):
        DVGeo, initpts = self.setup_cubemodel()

        DVGeo.addVariable("cubey0")
        DVGeo.setDesignVars({"cubey0": np.array([4.2 + 1e-12])}, updateJacobian=False)
        self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[0, 1] - 4.2, 1e-12, 15)
        DVGeo.addVariable("cubedz")
        DVGeo.setDesignVars({"cubedz": np.array([9.5 - 1e-12])}, updateJacobian=False)
        self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[1, 2] - 7.5, -1e-12, 15)

    def test_serial_finite_difference(self):
        # this test checks the underlying jacobian itself, not the public API
        # TODO write tests for the public API
        DVGeo, initpts = self.setup_cubemodel()
        for designvarname in ["cubex0", "cubey0", "cubez0", "cubedx", "cubedy", "cubedz"]:
            DVGeo.addVariable(designvarname)
        # check the FD derivatives
        initpts_cache = initpts.copy()
        dvdict_cache = DVGeo.DVs.copy()
        self.assertFalse(DVGeo.updatedJac["mypts"])
        DVGeo._computeSurfJacobian(fd=True)
        self.assertTrue(DVGeo.updatedJac["mypts"])
        npts = initpts.shape[0]
        ndvs = DVGeo.getNDV()
        # check the jacobian results match analytic result
        testjac = DVGeo.pointSets["mypts"].jac.reshape(npts, 3, ndvs)
        analyticjac = self.setup_cubemodel_analytic_jac()

        for ipt in range(npts):
            self.assertAlmostEqual(np.sum(np.abs(testjac[ipt, :, :] - analyticjac[ipt, :, :])), 0)

        # check that the point set hasn't changed after running the FDs
        self.assertAlmostEqual(np.sum(np.abs(initpts_cache - DVGeo.pointSets["mypts"].proj_pts)), 0.0)
        # check that the DV dict hasn't changed
        for key in dvdict_cache:
            self.assertAlmostEqual(np.sum(np.abs(DVGeo.DVs[key].value - dvdict_cache[key].value)), 0.0)

    def test_jacobian_arbitrary_added_order(self):
        # this test checks the underlying jacobian itself, not the public API
        DVGeo, initpts = self.setup_cubemodel()
        # switch up the order of DVs added
        for designvarname in ["cubey0", "cubedx", "cubedy", "cubex0", "cubedz", "cubez0"]:
            DVGeo.addVariable(designvarname)
        # check the FD derivatives
        DVGeo._computeSurfJacobian(fd=True)
        npts = initpts.shape[0]
        ndvs = DVGeo.getNDV()
        # check the jacobian results match analytic result
        testjac = DVGeo.pointSets["mypts"].jac.reshape(npts, 3, ndvs)
        ordered_analyticjac = self.setup_cubemodel_analytic_jac()
        analyticjac = np.zeros((npts, 3, ndvs))

        # get original variable ordering
        orig_var_order = ["cubex0", "cubey0", "cubez0", "cubedx", "cubedy", "cubedz"]
        # reorder the analytic jacobian
        for idv, designvarname in enumerate(orig_var_order):
            dv_ind = DVGeo.DVs[designvarname].globalStartInd
            analyticjac[:, :, dv_ind] = ordered_analyticjac[:, :, idv]
            self.assertNotEqual(dv_ind, idv)

        for ipt in range(npts):
            self.assertAlmostEqual(np.sum(np.abs(testjac[ipt, :, :] - analyticjac[ipt, :, :])), 0)

    def train_composite(self, train=True):
        self.test_composite(train=train)

    def test_composite(self, train=False):
        """
        ESP wing test with DVComposite
        """

        refFile = os.path.join(self.input_path, "reg_tests/ref/test_DVGeometryESP_02.ref")

        csmFile = os.path.join(self.input_path, "../input_files/esp/box.csm")
        DVGeo = DVGeometryESP(csmFile)
        self.assertIsNotNone(DVGeo)

        # add variables with a mix of optional arguments
        DVGeo.addVariable("cubex0", lower=np.array([-10.0]), upper=np.array([10.0]), scale=0.1, dh=0.0001)
        self.assertEqual(DVGeo.getNDV(), 1)
        DVGeo.addVariable("cubey0")
        self.assertEqual(DVGeo.getNDV(), 2)
        DVGeo.addVariable("cubez0", lower=np.array([-10.0]), upper=np.array([10.0]))
        self.assertEqual(DVGeo.getNDV(), 3)

        _, initpts = self.setup_cubemodel()

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("ESP NACA 0012 composite derivative test")
            dh = 1e-6
            # extract node coordinates and save them in a numpy array
            npts = initpts.shape[0]

            coor = initpts
            # Add this pointSet to DVGeo
            DVGeo.addPointSet(coor, "test_points")
            DVGeo.addCompositeDV("espComp", "test_points")

            # We will have nNodes*3 many functions of interest...
            dIdpt = np.zeros((npts * 3, npts, 3))

            # set the seeds to one in the following fashion:
            # first function of interest gets the first coordinate of the first point
            # second func gets the second coord of first point etc....
            for i in range(npts):
                for j in range(3):
                    dIdpt[i * 3 + j, i, j] = 1
            # first get the dvgeo result
            funcSens = DVGeo.totalSensitivity(dIdpt.copy(), "test_points")

            # now perturb the design with finite differences and compute FD gradients
            DVs = OrderedDict()

            for dvName in DVGeo.DVs:
                DVs[dvName] = DVGeo.DVs[dvName].value

            funcSensFD = {}

            inDict = copy.deepcopy(DVs)
            userVec = DVGeo.convertDictToSensitivity(inDict)
            DVvalues = DVGeo.convertSensitivityToDict(userVec.reshape(1, -1), out1D=True, useCompositeNames=False)

            # We flag the composite DVs to be false to not to make any changes on default mode.
            DVGeo.useComposite = False

            count = 0
            for x in DVvalues:
                # perturb the design
                xRef = DVvalues[x].copy()
                DVvalues[x] += dh

                DVGeo.setDesignVars(DVvalues)

                # get the new points
                coorNew = DVGeo.update("test_points")

                # calculate finite differences
                funcSensFD[x] = (coorNew.flatten() - coor.flatten()) / dh

                # set back the DV
                DVvalues[x] = xRef.copy()
                count = count + 1

            DVGeo.useComposite = True
            # now loop over the values and compare
            # when this is run with multiple procs, VSP sometimes has a bug
            # that leads to different procs having different spanwise
            # u-v distributions. as a result, the final values can differ up to 1e-5 levels
            # this issue does not come up if this tests is ran with a single proc
            biggest_deriv = 1e-16

            DVCount = DVGeo.getNDV()

            funcSensFDMat = np.zeros((coorNew.size, DVCount), dtype="d")
            i = 0

            for key in DVGeo.DVs:
                funcSensFDMat[:, i] = funcSensFD[key].T
                i += 1

            # Now we need to map our FD derivatives to composite
            funcSensFDMat = DVGeo.mapSensToComp(funcSensFDMat)
            funcSensFDDict = DVGeo.convertSensitivityToDict(funcSensFDMat, useCompositeNames=True)

            # Now we check how much the derivatives deviates from each other
            counter = 0
            for x in [DVGeo.DVComposite.name]:
                err = np.array(funcSens[x]) - np.array(funcSensFDDict[x])
                maxderiv = np.max(np.abs(funcSens[x]))
                normalizer = np.median(np.abs(funcSensFDDict[x]))
                if np.abs(normalizer) < 1:
                    normalizer = np.ones(1)
                normalized_error = err / normalizer
                if maxderiv > biggest_deriv:
                    biggest_deriv = maxderiv
                handler.assert_allclose(normalized_error, 0.0, name=f"{x}_grad_normalized_error", rtol=1e0, atol=5e-5)
                counter = counter + 1

            # make sure that at least one derivative is nonzero
            self.assertGreater(biggest_deriv, 0.005)

            Composite_FFD = DVGeo.getValues()
            handler.root_add_val("Composite DVs :", Composite_FFD["espComp"], rtol=1e-12, atol=1e-12)


@unittest.skipUnless(mpiInstalled and ocsmInstalled, "MPI and pyOCSM are required.")
class TestPyGeoESP_BasicCube_Distributed(unittest.TestCase):
    N_PROCS = 3

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.input_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.comm = MPI.COMM_WORLD

    def setup_cubemodel(self):
        # load the box model and build the box model
        csmFile = os.path.join(self.input_path, "../input_files/esp/box.csm")
        DVGeo = DVGeometryESP(csmFile)
        self.assertIsNotNone(DVGeo)

        # add a point set on the surface
        # distri
        vertex1 = np.array([-2.0, -2.0, -2.0])
        vertex2 = np.array([1.5, 1.5, 1.5])
        left = np.array([-2.0, -1.1, -1.1])
        right = np.array([1.5, -1.2, -0.1])
        front = np.array([0.25, 1.5, 0.3])
        back = np.array([1.2, -2.0, -0.3])
        top = np.array([0.0, 0.1, 1.5])
        bottom = np.array([-1.9, -1.1, -2.0])
        # distribute the pointset
        if self.comm.rank == 0:
            initpts = np.vstack([vertex1, vertex2, left, right])
        elif self.comm.rank == 1:
            initpts = np.vstack([front, back, top])
        elif self.comm.rank == 2:
            initpts = np.vstack([bottom, left, right])
        else:
            raise ValueError("Too many procs")

        distglobal = DVGeo.addPointSet(initpts, "mypts", cache_projections=False)
        self.assertAlmostEqual(distglobal, 0.0, 8)

        # evaluate the points and check that they match
        DVGeo._updateModel()
        DVGeo._updateProjectedPts()
        self.assertTrue(DVGeo.pointSetUpToDate)
        self.assertAlmostEqual(np.linalg.norm(initpts - DVGeo.pointSets["mypts"].proj_pts), 0.0, 10)

        return DVGeo, initpts

    def setup_cubemodel_analytic_jac(self):
        jacpt0 = np.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]  # x  # y
        )  # z
        jacpt1 = np.array(
            [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]  # x  # y
        )  # z
        jacpt2 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.9 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.9 / 3.5],
            ]
        )  # z
        jacpt3 = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.8 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.9 / 3.5],
            ]
        )  # z
        jacpt4 = np.array(
            [
                [1.0, 0.0, 0.0, 2.25 / 3.50, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 2.30 / 3.50],
            ]
        )  # z
        jacpt5 = np.array(
            [
                [1.0, 0.0, 0.0, 3.20 / 3.50, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.70 / 3.50],
            ]
        )  # z
        jacpt6 = np.array(
            [
                [1.0, 0.0, 0.0, 2.0 / 3.5, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 2.1 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )  # z
        jacpt7 = np.array(
            [
                [1.0, 0.0, 0.0, 0.1 / 3.5, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.9 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )  # z
        if self.comm.rank == 0:
            ordered_analytic_jac = np.concatenate([jacpt0, jacpt1, jacpt2, jacpt3], axis=0).reshape(4, 3, 6)
        elif self.comm.rank == 1:
            ordered_analytic_jac = np.concatenate([jacpt4, jacpt5, jacpt6], axis=0).reshape(3, 3, 6)
        elif self.comm.rank == 2:
            ordered_analytic_jac = np.concatenate([jacpt7, jacpt2, jacpt3], axis=0).reshape(3, 3, 6)
        return ordered_analytic_jac

    def test_load_a_model(self):
        # load the box model and build the box model
        csmFile = os.path.join(self.input_path, "../input_files/esp/box.csm")
        DVGeometryESP(csmFile)

    def test_add_desvars(self):
        # load the box model and build the box model
        csmFile = os.path.join(self.input_path, "../input_files/esp/box.csm")
        DVGeo = DVGeometryESP(csmFile)
        self.assertIsNotNone(DVGeo)

        # add variables with a mix of optional arguments
        DVGeo.addVariable("cubex0", lower=np.array([-10.0]), upper=np.array([10.0]), scale=0.1, dh=0.0001)
        self.assertEqual(DVGeo.getNDV(), 1)
        DVGeo.addVariable("cubey0")
        self.assertEqual(DVGeo.getNDV(), 2)
        DVGeo.addVariable("cubez0", lower=np.array([-10.0]), upper=np.array([10.0]))
        self.assertEqual(DVGeo.getNDV(), 3)

        # try to add a variable that isn't in the CSM file
        with self.assertRaises(Error):
            DVGeo.addVariable("cubew0")

    def test_add_pointset(self):
        DVGeo, initpts = self.setup_cubemodel()

    def test_updated_points(self):
        DVGeo, initpts = self.setup_cubemodel()

        DVGeo.addVariable("cubey0")
        DVGeo.setDesignVars({"cubey0": np.array([4.2000])}, updateJacobian=False)
        npts = initpts.shape[0]
        self.assertAlmostEqual(np.sum(DVGeo.pointSets["mypts"].proj_pts[:, 1] - initpts[:, 1]) / npts, 6.2, 10)
        DVGeo.addVariable("cubedz")
        DVGeo.setDesignVars({"cubedz": np.array([9.5])}, updateJacobian=False)
        if self.comm.rank == 0:
            self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[1, 2], 7.5)
            self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[0, 2], -2.0)

    def test_parallel_finite_difference(self):
        # this test checks the underlying jacobian itself, not the public API
        # TODO write tests for the public API
        DVGeo, initpts = self.setup_cubemodel()
        for designvarname in ["cubex0", "cubey0", "cubez0", "cubedx", "cubedy", "cubedz"]:
            DVGeo.addVariable(designvarname)
        # check the FD derivatives
        initpts_cache = initpts.copy()
        dvdict_cache = DVGeo.DVs.copy()
        self.assertFalse(DVGeo.updatedJac["mypts"])
        DVGeo._computeSurfJacobian(fd=True)
        self.assertTrue(DVGeo.updatedJac["mypts"])
        npts = initpts.shape[0]
        ndvs = DVGeo.getNDV()
        # check the jacobian results match analytic result
        testjac = DVGeo.pointSets["mypts"].jac.reshape(npts, 3, ndvs)
        analyticjac = self.setup_cubemodel_analytic_jac()

        for ipt in range(npts):
            self.assertAlmostEqual(np.sum(np.abs(testjac[ipt, :, :] - analyticjac[ipt, :, :])), 0)

        # check that the point set hasn't changed after running the FDs
        self.assertAlmostEqual(np.sum(np.abs(initpts_cache - DVGeo.pointSets["mypts"].proj_pts)), 0.0)
        # check that the DV dict hasn't changed
        for key in dvdict_cache:
            self.assertAlmostEqual(np.sum(np.abs(DVGeo.DVs[key].value - dvdict_cache[key].value)), 0.0)

    def test_jacobian_arbitrary_added_order(self):
        # this test checks the underlying jacobian itself, not the public API
        DVGeo, initpts = self.setup_cubemodel()
        # switch up the order of DVs added
        for designvarname in ["cubey0", "cubedx", "cubedy", "cubex0", "cubedz", "cubez0"]:
            DVGeo.addVariable(designvarname)
        # check the FD derivatives
        DVGeo._computeSurfJacobian(fd=True)
        npts = initpts.shape[0]
        ndvs = DVGeo.getNDV()
        # check the jacobian results match analytic result
        testjac = DVGeo.pointSets["mypts"].jac.reshape(npts, 3, ndvs)
        ordered_analyticjac = self.setup_cubemodel_analytic_jac()
        analyticjac = np.zeros((npts, 3, ndvs))

        # get original variable ordering
        orig_var_order = ["cubex0", "cubey0", "cubez0", "cubedx", "cubedy", "cubedz"]
        # reorder the analytic jacobian
        for idv, designvarname in enumerate(orig_var_order):
            dv_ind = DVGeo.DVs[designvarname].globalStartInd
            analyticjac[:, :, dv_ind] = ordered_analyticjac[:, :, idv]
            self.assertNotEqual(dv_ind, idv)

        for ipt in range(npts):
            self.assertAlmostEqual(np.sum(np.abs(testjac[ipt, :, :] - analyticjac[ipt, :, :])), 0)


@unittest.skipUnless(mpiInstalled and ocsmInstalled, "MPI and pyOCSM are required.")
class TestPyGeoESP_BasicCube_Distributed_OneProcBlank(unittest.TestCase):
    N_PROCS = 4

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.input_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.comm = MPI.COMM_WORLD

    def setup_cubemodel(self):
        # load the box model and build the box model
        csmFile = os.path.join(self.input_path, "../input_files/esp/box.csm")
        DVGeo = DVGeometryESP(csmFile)
        self.assertIsNotNone(DVGeo)

        # add a point set on the surface
        # distri
        vertex1 = np.array([-2.0, -2.0, -2.0])
        vertex2 = np.array([1.5, 1.5, 1.5])
        left = np.array([-2.0, -1.1, -1.1])
        right = np.array([1.5, -1.2, -0.1])
        front = np.array([0.25, 1.5, 0.3])
        back = np.array([1.2, -2.0, -0.3])
        top = np.array([0.0, 0.1, 1.5])
        bottom = np.array([-1.9, -1.1, -2.0])
        # distribute the pointset
        if self.comm.rank == 0:
            initpts = np.vstack([vertex1, vertex2, left, right])
        elif self.comm.rank == 1:
            initpts = np.vstack([front, back, top])
        elif self.comm.rank == 2:
            initpts = np.array([]).reshape((0, 3))
        elif self.comm.rank == 3:
            initpts = np.vstack([bottom, left, right])
        else:
            raise ValueError("Too many procs")
        distglobal = DVGeo.addPointSet(initpts, "mypts", cache_projections=False)

        self.assertAlmostEqual(distglobal, 0.0, 8)

        # evaluate the points and check that they match
        DVGeo._updateModel()
        DVGeo._updateProjectedPts()
        self.assertTrue(DVGeo.pointSetUpToDate)
        self.assertAlmostEqual(np.linalg.norm(initpts - DVGeo.pointSets["mypts"].proj_pts), 0.0, 10)

        return DVGeo, initpts

    def setup_cubemodel_analytic_jac(self):
        jacpt0 = np.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]  # x  # y
        )  # z
        jacpt1 = np.array(
            [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]  # x  # y
        )  # z
        jacpt2 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.9 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.9 / 3.5],
            ]
        )  # z
        jacpt3 = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.8 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.9 / 3.5],
            ]
        )  # z
        jacpt4 = np.array(
            [
                [1.0, 0.0, 0.0, 2.25 / 3.50, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 2.30 / 3.50],
            ]
        )  # z
        jacpt5 = np.array(
            [
                [1.0, 0.0, 0.0, 3.20 / 3.50, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.70 / 3.50],
            ]
        )  # z
        jacpt6 = np.array(
            [
                [1.0, 0.0, 0.0, 2.0 / 3.5, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 2.1 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )  # z
        jacpt7 = np.array(
            [
                [1.0, 0.0, 0.0, 0.1 / 3.5, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0, 0.9 / 3.5, 0.0],  # y
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )  # z
        if self.comm.rank == 0:
            ordered_analytic_jac = np.concatenate([jacpt0, jacpt1, jacpt2, jacpt3], axis=0).reshape(4, 3, 6)
        elif self.comm.rank == 1:
            ordered_analytic_jac = np.concatenate([jacpt4, jacpt5, jacpt6], axis=0).reshape(3, 3, 6)
        elif self.comm.rank == 2:
            ordered_analytic_jac = np.array([]).reshape(0, 3, 6)
        elif self.comm.rank == 3:
            ordered_analytic_jac = np.concatenate([jacpt7, jacpt2, jacpt3], axis=0).reshape(3, 3, 6)
        return ordered_analytic_jac

    def test_add_pointset(self):
        DVGeo, initpts = self.setup_cubemodel()

    def test_updated_points(self):
        DVGeo, initpts = self.setup_cubemodel()

        DVGeo.addVariable("cubey0")
        DVGeo.setDesignVars({"cubey0": np.array([4.2000])}, updateJacobian=False)
        npts = initpts.shape[0]
        if self.comm.rank != 2:
            self.assertAlmostEqual(np.sum(DVGeo.pointSets["mypts"].proj_pts[:, 1] - initpts[:, 1]) / npts, 6.2, 10)
        DVGeo.addVariable("cubedz")
        DVGeo.setDesignVars({"cubedz": np.array([9.5])}, updateJacobian=False)
        if self.comm.rank == 0:
            self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[1, 2], 7.5)
            self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[0, 2], -2.0)
        elif self.comm.rank == 1:
            self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[0, 2], -2.0 + (0.3 + 2.0) * (9.5 / 3.5))
            self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[1, 2], -2.0 + (-0.3 + 2.0) * (9.5 / 3.5))
        elif self.comm.rank == 3:
            self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[0, 2], -2.0)
            self.assertAlmostEqual(DVGeo.pointSets["mypts"].proj_pts[1, 2], -2.0 + (-1.1 + 2.0) * (9.5 / 3.5))

    def test_parallel_finite_difference(self):
        # this test checks the underlying jacobian itself, not the public API
        # TODO write tests for the public API
        DVGeo, initpts = self.setup_cubemodel()
        for designvarname in ["cubex0", "cubey0", "cubez0", "cubedx", "cubedy", "cubedz"]:
            DVGeo.addVariable(designvarname)
        # check the FD derivatives
        initpts_cache = initpts.copy()
        dvdict_cache = DVGeo.DVs.copy()
        self.assertFalse(DVGeo.updatedJac["mypts"])
        DVGeo._computeSurfJacobian(fd=True)
        self.assertTrue(DVGeo.updatedJac["mypts"])
        npts = initpts.shape[0]
        ndvs = DVGeo.getNDV()
        # check the jacobian results match analytic result
        testjac = DVGeo.pointSets["mypts"].jac.reshape(npts, 3, ndvs)
        analyticjac = self.setup_cubemodel_analytic_jac()

        if self.comm.rank != 2:
            for ipt in range(npts):
                self.assertAlmostEqual(np.sum(np.abs(testjac[ipt, :, :] - analyticjac[ipt, :, :])), 0)

            # check that the point set hasn't changed after running the FDs
            self.assertAlmostEqual(np.sum(np.abs(initpts_cache - DVGeo.pointSets["mypts"].proj_pts)), 0.0)
            # check that the DV dict hasn't changed
            for key in dvdict_cache:
                self.assertAlmostEqual(np.sum(np.abs(DVGeo.DVs[key].value - dvdict_cache[key].value)), 0.0)


@unittest.skipUnless(mpiInstalled and ocsmInstalled, "MPI and pyOCSM are required.")
@parameterized_class(test_params)
class TestPyGeoESP_NACAFoil(unittest.TestCase):
    # serial and parallel handled automatically
    N_PROCS = 1

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.input_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.comm = MPI.COMM_WORLD

    def setup_airfoilmodel(self, kulfan=False, projtol=0.01):
        # load the csm file and pointset file
        if kulfan:
            csmFile = os.path.join(self.input_path, "../input_files/esp/naca0012_kulfan.csm")
            max_dist_tol = 2
        else:
            csmFile = os.path.join(self.input_path, "../input_files/esp/naca0012.csm")
            max_dist_tol = 3
        stlFile = os.path.join(self.input_path, "../input_files/esp/naca0012_esp.stl")

        DVGeo = DVGeometryESP(csmFile, projTol=projtol)
        self.assertIsNotNone(DVGeo)

        testobj = mesh.Mesh.from_file(stlFile)
        # test mesh dim 0 is triangle index
        # dim 1 is each vertex of the triangle
        # dim 2 is x, y, z dimension
        p0 = testobj.vectors[:, 0, :]
        p1 = testobj.vectors[:, 1, :]
        p2 = testobj.vectors[:, 2, :]
        distglobal1 = DVGeo.addPointSet(p0, "airfoil_p0")
        distglobal2 = DVGeo.addPointSet(p1, "airfoil_p1")
        distglobal3 = DVGeo.addPointSet(p2, "airfoil_p2")

        distglobal = np.max(np.array([distglobal1, distglobal2, distglobal3]))
        self.assertAlmostEqual(distglobal, 0.0, max_dist_tol)

        # evaluate the points and check that they match
        DVGeo._updateModel()
        DVGeo._updateProjectedPts()
        self.assertTrue(DVGeo.pointSetUpToDate)
        updated_dist_max = np.max(np.sqrt(np.sum((p0 - DVGeo.pointSets["airfoil_p0"].proj_pts) ** 2, axis=1)))
        self.assertAlmostEqual(updated_dist_max, 0.0, max_dist_tol)
        updated_dist_max = np.max(np.sqrt(np.sum((p1 - DVGeo.pointSets["airfoil_p1"].proj_pts) ** 2, axis=1)))
        self.assertAlmostEqual(updated_dist_max, 0.0, max_dist_tol)
        updated_dist_max = np.max(np.sqrt(np.sum((p2 - DVGeo.pointSets["airfoil_p2"].proj_pts) ** 2, axis=1)))
        self.assertAlmostEqual(updated_dist_max, 0.0, max_dist_tol)
        return DVGeo, [p0, p1, p2]

    def test_add_pointset(self):
        DVGeo, initpts = self.setup_airfoilmodel()

    def test_add_pointset_tighter_tolerance(self):
        with self.assertRaises(ValueError):
            DVGeo, initpts = self.setup_airfoilmodel(projtol=1e-5)

    def test_add_desvars(self):
        DVGeo, initpts = self.setup_airfoilmodel()
        DVGeo.addVariable("nacacode", lower=np.array([8]), upper=np.array([15]), scale=1, dh=0.001)
        self.assertEqual(DVGeo.getNDV(), 1)

    def test_point_mismatch(self):
        # load the wrong pointset on purpose
        csmFile = os.path.join(self.input_path, "../input_files/esp/naca0010.csm")
        stlFile = os.path.join(self.input_path, "../input_files/esp/naca0012_esp.stl")

        DVGeo = DVGeometryESP(csmFile)
        self.assertIsNotNone(DVGeo)

        testobj = mesh.Mesh.from_file(stlFile)
        # test mesh dim 0 is triangle index
        # dim 1 is each vertex of the triangle
        # dim 2 is x, y, z dimension
        p0 = testobj.vectors[:, 0, :]
        with self.assertRaises(ValueError):
            distglobal1 = DVGeo.addPointSet(p0, "airfoil_p0")
            self.assertGreater(distglobal1, 0.01)

    def test_parallel_finite_difference(self, train=False):
        np.random.seed(1)
        DVGeo, initpts = self.setup_airfoilmodel(kulfan=True)
        DVGeo.addVariable("cst_u", lower=np.zeros((13,)), upper=np.ones((13,)), scale=1, dh=0.0001)
        DVGeo.addVariable("cst_l", lower=-np.ones((13,)), upper=np.zeros((13,)), scale=1, dh=0.0001)

        refFile = os.path.join(self.input_path, "reg_tests/ref/test_DVGeometryESP_01.ref")
        pointset_names = ["airfoil_p0", "airfoil_p1", "airfoil_p2"]
        for pointset_name in pointset_names:
            self.assertFalse(DVGeo.updatedJac[pointset_name])
        DVGeo._computeSurfJacobian(fd=True)
        for pointset_name in pointset_names:
            self.assertTrue(DVGeo.updatedJac[pointset_name])

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("ESP NACA 0012 derivative test")
            npts = initpts[0].shape[0]
            dIdpt = np.random.rand(1, npts, 3)
            for pointset_name in pointset_names:
                dIdx = DVGeo.totalSensitivity(dIdpt, pointset_name)
                handler.root_add_dict("dIdx_" + pointset_name, dIdx, rtol=1e-7, atol=1e-7)


# TODO test pointset caching?
# TODO test total derivative API on an actual distributed pointset?

if __name__ == "__main__":
    unittest.main()
