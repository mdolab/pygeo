import os
import shutil
import unittest
import numpy as np
from baseclasses import BaseRegTest
import commonUtils
from pygeo import DVGeometry, DVConstraints
from stl import mesh


class RegTestPyGeo(unittest.TestCase):

    N_PROCS = 1

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def train_1(self, train=True, refDeriv=True):
        self.test_1(train=train, refDeriv=refDeriv)

    def test_1(self, train=False, refDeriv=False):
        """
        Test 1: Basic FFD, global DVs
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_01.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 1: Basic FFD, global DVs")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)
            # create global DVs on the parent
            DVGeo.addGlobalDV("mainX", -1.0, commonUtils.mainAxisPoints, lower=-1.0, upper=0.0, scale=1.0)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_2(self, train=True, refDeriv=True):
        self.test_2(train=train, refDeriv=refDeriv)

    def test_2(self, train=False, refDeriv=False):
        """
        Test 2: Basic FFD, global DVs and local DVs
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_02.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 2: Basic FFD, global DVs and local DVs")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            # create global DVs on the parent
            DVGeo.addGlobalDV("mainX", -1.0, commonUtils.mainAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_3(self, train=True, refDeriv=True):
        self.test_3(train=train, refDeriv=refDeriv)

    def test_3(self, train=False, refDeriv=False):
        """
        Test 3: Basic + Nested FFD, global DVs only
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_03.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 3: Basic + Nested FFD, global DVs only")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            # create global DVs on the parent
            DVGeo.addGlobalDV("mainX", -1.0, commonUtils.mainAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            # create global DVs on the child
            DVGeoChild.addGlobalDV("nestedX", -0.5, commonUtils.childAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_4(self, train=True, refDeriv=True):
        self.test_4(train=train, refDeriv=refDeriv)

    def test_4(self, train=False, refDeriv=False):
        """
        Test 4: Basic + Nested FFD, global DVs and local DVs on parent global on child
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_04.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 4: Basic + Nested FFD, global DVs and local DVs on parent global on child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            # create global DVs on the parent
            DVGeo.addGlobalDV("mainX", -1.0, commonUtils.mainAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            # create global DVs on the child
            DVGeoChild.addGlobalDV("nestedX", -0.5, commonUtils.childAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_5(self, train=True, refDeriv=True):
        self.test_5(train=train, refDeriv=refDeriv)

    def test_5(self, train=False, refDeriv=False):
        """
        Test 5: Basic + Nested FFD, global DVs and local DVs on both parent and child
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_05.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 5: Basic + Nested FFD, global DVs and local DVs on both parent and child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            # create global DVs on the parent
            DVGeo.addGlobalDV("mainX", -1.0, commonUtils.mainAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)
            # create global DVs on the child
            DVGeoChild.addGlobalDV("nestedX", -0.5, commonUtils.childAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_6(self, train=True, refDeriv=True):
        self.test_6(train=train, refDeriv=refDeriv)

    def test_6(self, train=False, refDeriv=False):
        """
        Test 6: Basic + Nested FFD, local DVs only
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_06.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 6: Basic + Nested FFD, local DVs only")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_7(self, train=True, refDeriv=True):
        self.test_7(train=train, refDeriv=refDeriv)

    def test_7(self, train=False, refDeriv=False):
        """
        Test 7: Basic + Nested FFD, local DVs only on parent
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_07.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 7: Basic + Nested FFD, local DVs only on parent")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_8(self, train=True, refDeriv=True):
        self.test_8(train=train, refDeriv=refDeriv)

    def test_8(self, train=False, refDeriv=False):
        """
        Test 8: Basic + Nested FFD, local DVs only on child
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_08.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 8: Basic + Nested FFD, local DVs only on child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_9(self, train=True, refDeriv=True):
        self.test_9(train=train, refDeriv=refDeriv)

    def test_9(self, train=False, refDeriv=False):
        """
        Test 9: Basic + Nested FFD, local DVs only on parent, global on child
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_09.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 9: Basic + Nested FFD, local DVs only on parent, global on child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            DVGeoChild.addGlobalDV("nestedX", -0.5, commonUtils.childAxisPoints, lower=-1.0, upper=0.0, scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_10(self, train=True, refDeriv=True):
        self.test_10(train=train, refDeriv=refDeriv)

    def test_10(self, train=False, refDeriv=False):
        """
        Test 10: Basic + Nested FFD, local DVs only on parent, global and local on child
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_10.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 10: Basic + Nested FFD, local DVs only on parent, global and local on child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            DVGeoChild.addGlobalDV("nestedX", -0.5, commonUtils.childAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def test_10_rot0(self, train=False, refDeriv=False):
        """
        Test 10: Basic + Nested FFD, local DVs only on parent, global and local on child
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_10.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 10: Basic + Nested FFD, local DVs only on parent, global and local on child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path, rotType=0)

            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            DVGeoChild.addGlobalDV("nestedX", -0.5, commonUtils.childAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def test_10_rot7(self, train=False, refDeriv=False):
        """
        Test 10: Basic + Nested FFD, local DVs only on parent, global and local on child
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_10.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 10: Basic + Nested FFD, local DVs only on parent, global and local on child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path, rotType=7)

            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            DVGeoChild.addGlobalDV("nestedX", -0.5, commonUtils.childAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_11(self, train=True, refDeriv=True):
        self.test_11(train=train, refDeriv=refDeriv)

    def test_11(self, train=False, refDeriv=False):
        """
        Test 11: Basic + Nested FFD, global DVs and local DVs on parent local on child
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_11.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 11: Basic + Nested FFD, global DVs and local DVs on parent local on child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            # create global DVs on the parent
            DVGeo.addGlobalDV("mainX", -1.0, commonUtils.mainAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            # create global DVs on the child
            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def test_11_rot0(self, train=False, refDeriv=False):
        """
        Test 11: Basic + Nested FFD, global DVs and local DVs on parent local on child
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_11.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print(
                "Test 11: Basic + Nested FFD, global DVs and local DVs on parent local on child, using rotType=0"
            )
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path, rotType=0)

            # create global DVs on the parent
            DVGeo.addGlobalDV("mainX", -1.0, commonUtils.mainAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            # create global DVs on the child
            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def test_11_rot7(self, train=False, refDeriv=False):
        """
        Test 11: Basic + Nested FFD, global DVs and local DVs on parent local on child
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_11.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print(
                "Test 11: Basic + Nested FFD, global DVs and local DVs on parent local on child, using rotType=7"
            )
            DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path, rotType=7)

            # create global DVs on the parent
            DVGeo.addGlobalDV("mainX", -1.0, commonUtils.mainAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            # create global DVs on the child
            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    # -------------------
    # D8 Tests
    # -------------------
    def train_12(self, train=True, refDeriv=True):
        self.test_12(train=train, refDeriv=refDeriv)

    def test_12(self, train=False, refDeriv=False):
        """
        Test 12: D8 FFD, global DVs
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_12.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 12: D8 FFD, global DVs")
            DVGeo, DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            # create global DVs on the parent
            axisX = [0.0, 26.0, 30.5, 32.5, 34.0]
            DVGeo.addGlobalDV("mainX", axisX, commonUtils.mainAxisPoints, lower=0.0, upper=35.0, scale=1.0)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

            handler.root_print("Test 12b: D8 FFD, random DV perturbation on test 10")
            xDV = DVGeo.getValues()
            for key in xDV:
                np.random.seed(42)
                xDV[key] += np.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

    def train_13(self, train=True, refDeriv=True):
        self.test_13(train=train, refDeriv=refDeriv)

    def test_13(self, train=False, refDeriv=False):
        """
        Test 13: D8 FFD, global DVs and local DVs
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_13.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 13: D8 FFD, global DVs and local DVs")
            DVGeo, DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            # create global DVs on the parent
            axisX = [0.0, 26.0, 30.5, 32.5, 34.0]
            DVGeo.addGlobalDV("mainX", axisX, commonUtils.mainAxisPoints, lower=0.0, upper=35.0, scale=1.0)
            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

            handler.root_print("Test 13b: D8 FFD, random DV perturbation on test 11")
            xDV = DVGeo.getValues()

            for key in xDV:
                np.random.seed(42)
                xDV[key] += np.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

    def train_14(self, train=True, refDeriv=True):
        self.test_14(train=train, refDeriv=refDeriv)

    def test_14(self, train=False, refDeriv=False):
        """
        Test 14
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_14.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 14: D8 + Nozzle FFD, global DVs only")
            DVGeo, DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            # create global DVs on the parent
            axisX = [0.0, 26.0, 30.5, 32.5, 34.0]
            DVGeo.addGlobalDV("mainX", axisX, commonUtils.mainAxisPointsD8, lower=0.0, upper=35.0, scale=1.0)
            # create global DVs on the child
            childAxisX = [32.4, 34]
            DVGeoChild.addGlobalDV(
                "nestedX", childAxisX, commonUtils.childAxisPointsD8, lower=0.0, upper=35.0, scale=1.0
            )
            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

            handler.root_print("Test 14b: D8 + Nozzle FFD, random DV perturbation on test 12")
            xDV = DVGeo.getValues()
            for key in xDV:
                np.random.seed(42)
                xDV[key] += np.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

    def train_15(self, train=True, refDeriv=True):
        self.test_15(train=train, refDeriv=refDeriv)

    def test_15(self, train=False, refDeriv=False):
        """
        Test 15
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_15.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 15: D8 + Nozzle FFD, global DVs and local DVs on parent global on child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            # create global DVs on the parent
            axisX = [0.0, 26.0, 30.5, 32.5, 34.0]
            DVGeo.addGlobalDV("mainX", axisX, commonUtils.mainAxisPoints, lower=0.0, upper=35.0, scale=1.0)
            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            # create global DVs on the child
            childAxisX = [32.4, 34]
            DVGeoChild.addGlobalDV("nestedX", childAxisX, commonUtils.childAxisPoints, lower=0.0, upper=35.0, scale=1.0)
            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

            handler.root_print("Test 15b: D8 + Nozzle FFD, random DV perturbation on test 13")
            xDV = DVGeo.getValues()
            for key in xDV:
                np.random.seed(42)
                xDV[key] += np.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

    def train_16(self, train=True, refDeriv=True):
        self.test_16(train=train, refDeriv=refDeriv)

    def test_16(self, train=False, refDeriv=False):
        """
        Test 16
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_16.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 16: D8 + Nozzle FFD,  global DVs and local DVs on both parent and child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            # create global DVs on the parent
            axisX = [0.0, 26.0, 30.5, 32.5, 34.0]
            DVGeo.addGlobalDV("mainX", axisX, commonUtils.mainAxisPoints, lower=0.0, upper=35.0, scale=1.0)
            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)
            # create global DVs on the child
            childAxisX = [32.4, 34]
            DVGeoChild.addGlobalDV("nestedX", childAxisX, commonUtils.childAxisPoints, lower=0.0, upper=35.0, scale=1.0)
            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

            handler.root_print("Test 14b: D8 + Nozzle FFD, random DV perturbation on test 14")
            xDV = DVGeo.getValues()
            for key in xDV:
                np.random.seed(42)
                xDV[key] += np.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

    def train_17(self, train=True, refDeriv=True):
        self.test_17(train=train, refDeriv=refDeriv)

    def test_17(self, train=False, refDeriv=False):
        """
        Test 17
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_17.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 17: D8 + Nozzle FFD, local DVs only")
            DVGeo, DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

            handler.root_print("Test 17b: D8 + Nozzle FFD, random DV perturbationon test 15")
            xDV = DVGeo.getValues()
            for key in xDV:
                np.random.seed(42)
                xDV[key] += np.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

    def train_18(self, train=True, refDeriv=True):
        self.test_18(train=train, refDeriv=refDeriv)

    def test_18(self, train=False, refDeriv=False):
        """
        Test 18
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_18.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 18: D8 + Nozzle FFD, local DVs only on parent, global on child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            childAxisX = [32.4, 34]
            DVGeoChild.addGlobalDV("nestedX", childAxisX, commonUtils.childAxisPoints, lower=0.0, upper=35.0, scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

            handler.root_print("Test 18b: D8 + Nozzle FFD, random DV perturbationon test 16")
            xDV = DVGeo.getValues()
            for key in xDV:
                np.random.seed(42)
                xDV[key] += np.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

    def train_19(self, train=True, refDeriv=True):
        self.test_19(train=train, refDeriv=refDeriv)

    def test_19(self, train=False, refDeriv=False):
        """
        Test 19
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_19.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 19: D8 + Nozzle FFD, local DVs only on parent, global and local on child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            childAxisX = [32.4, 34]
            DVGeoChild.addGlobalDV("nestedX", childAxisX, commonUtils.childAxisPoints, lower=0.0, upper=35.0, scale=1.0)
            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

            handler.root_print("Test 19b: D8 + Nozzle FFD,  random DV perturbationon test 17")
            xDV = DVGeo.getValues()
            for key in xDV:
                np.random.seed(42)
                xDV[key] += np.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

    def train_20(self, train=True, refDeriv=True):
        self.test_20(train=train, refDeriv=refDeriv)

    def test_20(self, train=False, refDeriv=False):
        """
        Test 20
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_20.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 20: D8 + Nozzle FFD, global DVs and local DVs on parent local on child")
            DVGeo, DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            # create global DVs on the parent
            axisX = [0.0, 26.0, 30.5, 32.5, 34.0]
            DVGeo.addGlobalDV("mainX", axisX, commonUtils.mainAxisPoints, lower=0.0, upper=35.0, scale=1.0)
            # create local DVs on the parent
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            # create global DVs on the child
            DVGeoChild.addLocalDV("childxdir", lower=-1.1, upper=1.1, axis="x", scale=1.0)
            DVGeoChild.addLocalDV("childydir", lower=-1.1, upper=1.1, axis="y", scale=1.0)
            DVGeoChild.addLocalDV("childzdir", lower=-1.1, upper=1.1, axis="z", scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

            handler.root_print("Test 20b: D8 + Nozzle FFD,  random DV perturbationon test 18")
            xDV = DVGeo.getValues()
            for key in xDV:
                np.random.seed(42)
                xDV[key] += np.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

    def train_21(self, train=True, refDeriv=True):
        self.test_21(train=train, refDeriv=refDeriv)

    def test_21(self, train=False, refDeriv=False):
        """
        Test 21
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_21.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 21: Axisymmetric FFD, global and local DVs")

            # Test with a single point along the 45 ` degree theta direction
            DVGeo = commonUtils.setupDVGeoAxi(self.base_path)

            DVGeo.addGlobalDV("mainAxis", np.zeros(1), commonUtils.mainAxisPointAxi)

            DVGeo.addLocalDV("x_axis", lower=-2, upper=2, axis="x")
            DVGeo.addLocalDV("z_axis", lower=-2, upper=2, axis="z")
            DVGeo.addLocalDV("y_axis", lower=-2, upper=2, axis="y")

            ptName = "point"
            s_pts = np.array(
                [
                    [0, 0.5, 0.5],
                ],
                dtype="float",
            )
            DVGeo.addPointSet(points=s_pts, ptName=ptName)

            # generate dIdPt
            nPt = 3
            dIdPt = np.zeros([nPt, 1, 3])
            dIdPt[0, 0, 0] = 1.0
            dIdPt[1, 0, 1] = 1.0
            dIdPt[2, 0, 2] = 1.0

            if refDeriv:
                # Generate reference derivatives
                refPoints = DVGeo.update(ptName)
                nPt = 3 * refPoints.shape[0]
                step = 1e-5
                J_fd = commonUtils.totalSensitivityFD(DVGeo, nPt, ptName, step)
                handler.root_add_dict(J_fd, rtol=1e-7, atol=1e-7)

            else:
                # Compute the analytic derivatives
                dIdx = DVGeo.totalSensitivity(dIdPt, ptName)
                handler.root_add_dict("dIdx", dIdx, rtol=1e-7, atol=1e-7)

    def test_spanwise_dvs(self, train=False, refDeriv=False):
        """
        Test spanwise_dvs
        """
        # refFile = os.path.join(self.base_path,'ref/test_DVGeometry_spanwise_dvs.ref')
        # with BaseRegTest(refFile, train=train) as handler:
        #     handler.root_print("Test spanwise local variables writing function")

        meshfile = os.path.join(self.base_path, "../../input_files/c172.stl")
        ffdfile = os.path.join(self.base_path, "../../input_files/c172.xyz")
        testmesh = mesh.Mesh.from_file(meshfile)
        # test mesh dim 0 is triangle index
        # dim 1 is each vertex of the triangle
        # dim 2 is x, y, z dimension

        # create a DVGeo object with a few local thickness variables
        DVGeo = DVGeometry(ffdfile)
        DVGeo.addSpanwiseLocalDV("shape", "i", lower=-0.5, upper=0.5, axis="y", scale=1.0)

        # create a DVConstraints object for the wing
        DVCon = DVConstraints()
        DVCon.setDVGeo(DVGeo)
        p0 = testmesh.vectors[:, 0, :] / 1000
        v1 = testmesh.vectors[:, 1, :] / 1000 - p0
        v2 = testmesh.vectors[:, 2, :] / 1000 - p0
        DVCon.setSurface([p0, v1, v2])

        leList = [[0.7, 0.0, 0.1], [0.7, 0.0, 2.4]]
        teList = [[0.9, 0.0, 0.1], [0.9, 0.0, 2.4]]

        nSpan = 10
        nChord = 10
        name = "thickness_con"
        DVCon.addThicknessConstraints2D(leList, teList, nSpan, nChord, name=name)

        size = DVGeo._getNDVSpanwiseLocal()

        np.random.seed(0)
        DVGeo.setDesignVars({"shape": (np.random.rand(size) - 0.5)})

        funcs = {}
        DVCon.evalFunctions(funcs)
        # print(funcs)

        for i in range(nChord):
            for j in range(nSpan - 1):
                np.testing.assert_allclose(funcs[name][i * nChord + j + 1], funcs[name][i * nChord + j], rtol=2e-15)

    def train_23_xyzFraction(self, train=True):
        self.test_23_xyzFraction(train=train)

    def test_23_xyzFraction(self, train=False):
        """
        Test 23
        This test verifies the correct implementation of the generalized `xFraction`, `yFraction` (and indirectly `zFraction`)
        Given an arbitrary input for the in-plane location of the reference axis nodes, the test sets up the axis object and compares the nodes location with a reference file.
        As the geometry of the FFD box is simple, the values can be also hand calculated:
        xFraction = 0.3, FFD x interval [-1,1] ---> 0.6 displacement from x min (% displ calculated from LE=xmin) --> x = -0.4
        yFraction = 0.6, FFD y interval [-0.5,0.5] ---> 0.6 displacement from y max (% displ calculated from top of the box=ymax) --> x = -0.1
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_23.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test generalized axis node location section in plane")
            DVGeo = DVGeometry(os.path.join(self.base_path, "../../input_files/2x1x8_rectangle.xyz"))
            xfraction = 0.3
            yfraction = 0.6
            rotType = 0
            DVGeo.addRefAxis("RefAx", xFraction=xfraction, yFraction=yfraction, alignIndex="k", rotType=rotType)
            nodes_loc = DVGeo.axis["RefAx"]["curve"].X

            handler.root_add_val("RefAxis_nodes_coord", nodes_loc, rtol=1e-12, atol=1e-12)

    def train_24_rot0_nonaligned(self, train=True, refDeriv=False):
        self.test_24_rot0_nonaligned(train=train, refDeriv=refDeriv)

    def test_24_rot0_nonaligned(self, train=False, refDeriv=False):
        """
        Test 24
        This test ensures that the scaling attributes (scale_x, scale_y, and scale_z) are effective when rotType=0 is selected.
        Moreover, this test ensures that rotType=0 reference axis can handle (given appropriate input parameters) FFD blocks that are not aligned with the main system of reference, e.g. the blades of a 3-bladed wind turbine rotor.
        The newly added input parameters rot0ang and rot0axis are used to provide the user control on this.
        The operations that pyGeo performs for this test are the following:
        We start from an initial "vertical" FFD box which, using the combination of rotType=0, rot0ang=-90, and rot0axis=[1,0,0] for addRefAxis(), is first rotated to have its "spanwise" axis along the y axis.
        Then, the script scales the 2nd section along the z axis for a "thickness" increase and the 4th section along the x axis for "chord" increase, it adds a +/- 30 deg twist respectively, and finally rotates the deformed FFD back in the initial position.
        The twist is added to ensure that the operation order is maintained, and the scaling preserves the orthogonality of the FFD in the section plane.
        This is a particular case as the FFD box is already aligned with the main axis and we "flip" the y and z axes, but the same criteria can be applied to a general rotation.
        """
        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_24.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test twist and scaling for FFDs non-aligned to main system of reference")
            DVGeo = DVGeometry(os.path.join(self.base_path, "../../input_files/2x1x8_rectangle.xyz"))
            rotType = 0
            xfraction = 0.5
            nRefAxPts = DVGeo.addRefAxis("RefAx", xFraction=xfraction, alignIndex="k", rotType=rotType, rot0ang=-90)

            fix_root_sect = 1
            nTwist = nRefAxPts - fix_root_sect

            DVGeo.addGlobalDV(dvName="twist", value=[0] * nTwist, func=commonUtils.twist, lower=-90, upper=90, scale=1)
            DVGeo.addGlobalDV(
                dvName="thickness", value=[1.0] * nTwist, func=commonUtils.thickness, lower=0.7, upper=5.0, scale=1
            )
            DVGeo.addGlobalDV(
                dvName="chord", value=[1.0] * nTwist, func=commonUtils.chord, lower=0.7, upper=5.0, scale=1
            )

            commonUtils.testSensitivities(DVGeo, refDeriv, handler, pointset=2)

            x = DVGeo.getValues()

            # Modifying the twist
            keyName = "twist"
            twistTest = [30, 0, -30]
            x[keyName] = twistTest

            # Modifying the chord
            keyName = "thickness"
            thickTest = [3.0, 1.0, 1.0]
            x[keyName] = thickTest

            # Modifying the chord
            keyName = "chord"
            chordTest = [1.0, 1.0, 2.0]
            x[keyName] = chordTest

            DVGeo.setDesignVars(x)

            DVGeo.update("testPoints")
            FFD_coords = DVGeo.FFD.coef.copy()

            handler.root_add_val("Updated FFD coordinates", FFD_coords, rtol=1e-12, atol=1e-12)

    def test_demoDesignVars(self):
        DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)
        DVGeo.addChild(DVGeoChild)

        # Add DVs to the child
        globalVarName = "nestedX"
        localVarName = "childXDir"
        DVGeoChild.addGlobalDV(globalVarName, -0.5, commonUtils.childAxisPoints, lower=-1.0, upper=0.0, scale=1.0)
        DVGeoChild.addLocalDV(localVarName, lower=-1.1, upper=1.1, axis="x", scale=1.0)

        # Add a simple point set
        ptName = "point"
        pts = np.array(
            [
                [0, 0.5, 0.5],
            ],
            dtype="float",
        )
        DVGeo.addPointSet(points=pts, ptName=ptName)

        # Demo DVs with just the FFD
        DVGeo.demoDesignVars(self.base_path)

        # Files that we expect to be output based on the DVs added
        refNames = sorted(
            [
                f"{localVarName}_000_iter_000",
                f"{localVarName}_000_iter_001",
                f"{localVarName}_001_iter_000",
                f"{localVarName}_001_iter_001",
                f"{localVarName}_002_iter_000",
                f"{localVarName}_002_iter_001",
                f"{localVarName}_003_iter_000",
                f"{localVarName}_003_iter_001",
                f"{localVarName}_004_iter_000",
                f"{localVarName}_004_iter_001",
                f"{localVarName}_005_iter_000",
                f"{localVarName}_005_iter_001",
                f"{localVarName}_006_iter_000",
                f"{localVarName}_006_iter_001",
                f"{localVarName}_007_iter_000",
                f"{localVarName}_007_iter_001",
                f"{globalVarName}_000_iter_000",
                f"{globalVarName}_000_iter_001",
            ]
        )
        ffdRef = [name + ".dat" for name in refNames]
        pointSetRef = [name + f"_{ptName}.dat" for name in refNames]

        # Set paths
        ffdPath = os.path.join(self.base_path, "ffd")
        pointSetPath = os.path.join(self.base_path, "pointset")
        surfPath = os.path.join(self.base_path, "surf")

        # Check that the generated FFD files match the expected result
        ffdFiles = sorted(os.listdir(ffdPath))
        self.assertEqual(ffdFiles, ffdRef)

        # Check that there are no other directories created
        with self.assertRaises(FileNotFoundError):
            os.listdir(pointSetPath)
            os.listdir(surfPath)

        # Delete FFD files
        shutil.rmtree(ffdPath)

        # Demo DVs with a point set
        DVGeo.demoDesignVars(self.base_path, pointSet=ptName)

        # Check that the FFD and point set files match the expected result
        ffdFiles = sorted(os.listdir(ffdPath))
        pointSetFiles = sorted(os.listdir(pointSetPath))
        self.assertEqual(ffdFiles, ffdRef)
        self.assertEqual(pointSetFiles, pointSetRef)

        # Delete FFD and point set files
        shutil.rmtree(ffdPath)
        shutil.rmtree(pointSetPath)

    def test_writeRefAxes(self):
        DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)
        DVGeo.addChild(DVGeoChild)

        # Add a simple point set
        ptName = "point"
        pts = np.array(
            [
                [0, 0.5, 0.5],
            ],
            dtype="float",
        )
        DVGeo.addPointSet(points=pts, ptName=ptName)

        # Write out the axes
        axesPath = os.path.join(self.base_path, "axis")
        DVGeo.writeRefAxes(axesPath)

        # Check that files were written
        self.assertTrue(os.path.isfile(axesPath + "_parent.dat"))
        self.assertTrue(os.path.isfile(axesPath + "_child000.dat"))

        # Delete axis files
        os.remove(axesPath + "_parent.dat")
        os.remove(axesPath + "_child000.dat")

    def train_ffdSplineOrder(self, train=True, refDeriv=True):
        self.test_ffdSplineOrder(train=train, refDeriv=refDeriv)

    def test_ffdSplineOrder(self, train=False, refDeriv=False):
        """
        Test custom FFD spline order
        """
        refFile = os.path.join(self.base_path, "ref/test_ffd_spline_order.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test custom FFD spline order")
            ffdfile = os.path.join(self.base_path, "../../input_files/deform_geometry_ffd.xyz")
            DVGeo = DVGeometry(ffdfile, kmax=6)

            # create local DVs
            DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
            DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
            DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler, pointset=3)


if __name__ == "__main__":
    unittest.main()

    # import xmlrunner
    # unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))
