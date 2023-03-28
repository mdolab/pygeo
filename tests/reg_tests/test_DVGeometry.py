# Standard Python modules
from collections import OrderedDict
import copy
import os
import shutil
import unittest

# External modules
from baseclasses import BaseRegTest
import commonUtils
import numpy as np
from stl import mesh

# First party modules
from pygeo import DVConstraints, DVGeometry


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

    def train_25_composite(self, train=True, refDeriv=True):
        self.test_25_composite(train=train, refDeriv=refDeriv)

    def test_25_composite(self, train=False, refDeriv=False):
        """
        D8 test with DVComposite
        """

        refFile = os.path.join(self.base_path, "ref/test_DVGeometry_25.ref")

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test composite DVs")
            DVGeo, DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            # create global DVs on the parent
            axisX = [0.0, 26.0, 30.5, 32.5, 34.0]
            DVGeo.addGlobalDV("mainX", axisX, commonUtils.mainAxisPoints, lower=0.0, upper=35.0, scale=1.0)

            self.assertIsNotNone(DVGeo)

            # create test points
            nPoints = 50
            points = np.zeros([nPoints, 3])
            for i in range(nPoints):
                nose = 0.01
                tail = 34.0
                delta = (tail - nose) / nPoints
                points[i, :] = [nose + i * delta, 1.0, 0.5]

            # add points to the geometry object
            ptName = "test_points"
            DVGeo.addPointSet(points, ptName)
            dh = 1e-6

            DVGeo.addCompositeDV("ffdComp", "test_points")

            # We will have nNodes*3 many functions of interest...
            # dIdpt = np.random.rand(1, npts, 3)
            dIdpt = np.zeros((nPoints * 3, nPoints, 3))

            # set the seeds to one in the following fashion:
            # first function of interest gets the first coordinate of the first point
            # second func gets the second coord of first point etc....
            for i in range(nPoints):
                for j in range(3):
                    dIdpt[i * 3 + j, i, j] = 1

            # first get the dvgeo result
            funcSens = DVGeo.totalSensitivity(dIdpt.copy(), "test_points")

            # now perturb the design with finite differences and compute FD gradients
            DVGeo.useComposite = False
            DVGeo_DV = DVGeo.getValues()
            DVs = OrderedDict()

            for dvName in DVGeo_DV:
                DVs[dvName] = DVGeo_DV[dvName]

            funcSensFD = {}

            inDict = copy.deepcopy(DVs)
            userVec = DVGeo.convertDictToSensitivity(inDict)
            DVvalues = DVGeo.convertSensitivityToDict(userVec.reshape(1, -1), out1D=True, useCompositeNames=False)

            # We flag the composite DVs to be false to not to make any changes on default mode.

            count = 0
            for x in DVvalues:
                # perturb the design
                xRef = DVvalues[x].copy()
                DVvalues[x] += dh

                DVGeo.setDesignVars(DVvalues)

                # get the new points
                coorNew = DVGeo.update("test_points")

                # calculate finite differences
                funcSensFD[x] = (coorNew.flatten() - points.flatten()) / dh

                # set back the DV
                DVvalues[x] = xRef.copy()
                count = count + 1
            funcSensFD = commonUtils.totalSensitivityFD(DVGeo, nPoints * 3, ptName, step=1e-6)

            DVGeo.useComposite = True

            biggest_deriv = 1e-16

            DVCount = DVGeo.getNDV()

            funcSensFDMat = np.zeros((coorNew.size, DVCount), dtype="d")

            i = 0

            for key in DVGeo_DV:
                nVal = len(DVGeo_DV[key])
                funcSensFDMat[:, i : i + nVal] = funcSensFD[key]
                i += 1

            # Now we need to map our FD derivatives to composite
            funcSensFDMat = DVGeo.mapSensToComp(funcSensFDMat)
            funcSensFDDict = DVGeo.convertSensitivityToDict(funcSensFDMat, useCompositeNames=True)

            # Now we check how much the derivatives deviates from each other
            counter = 0
            for x in [DVGeo.DVComposite.name]:
                err = np.array(funcSens[x]) - np.array(funcSensFDDict[x])
                maxderiv = np.max(np.abs(funcSens[x]))

                if maxderiv > biggest_deriv:
                    biggest_deriv = maxderiv
                handler.assert_allclose(err, 0.0, name=f"{x}_grad_error", rtol=1e-10, atol=1e-6)
                counter = counter + 1

            # make sure that at least one derivative is nonzero
            self.assertGreater(biggest_deriv, 0.005)

            # composite DV
            DVGeo.complex = False
            Composite_FFD = DVGeo.getValues()
            handler.root_add_val("Composite DVs :", Composite_FFD["ffdComp"], rtol=1e-12, atol=1e-12)

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
        self.assertTrue(os.path.isfile(axesPath + "_child0.dat"))

        # Delete axis files
        os.remove(axesPath + "_parent.dat")
        os.remove(axesPath + "_child0.dat")

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

    def test_spanDV(self, train=False):
        """
        Test span DV
        If the design FFD coef are not reset between updating the points and the derivative routines this will fail.
        """
        DVGeo = DVGeometry(os.path.join(self.base_path, "../../input_files/2x1x8_rectangle.xyz"))

        DVGeo.addRefAxis("RefAx", xFraction=0.5, alignIndex="k", rotType=0, rot0ang=-90)
        DVGeo.addGlobalDV(dvName="span", value=0.5, func=commonUtils.span, lower=0.1, upper=10, scale=1)

        points = np.zeros([2, 3])
        points[0, :] = [0.25, 0.4, 4]
        points[1, :] = [-0.8, 0.2, 7]
        ptName = "testPoints"
        DVGeo.addPointSet(points, ptName)

        nPt = points.size
        dIdx_FD = commonUtils.totalSensitivityFD(DVGeo, nPt, ptName)

        dIdPt = np.zeros([nPt, 2, 3])
        dIdPt[0, 0, 0] = 1.0
        dIdPt[1, 0, 1] = 1.0
        dIdPt[2, 0, 2] = 1.0
        dIdPt[3, 1, 0] = 1.0
        dIdPt[4, 1, 1] = 1.0
        dIdPt[5, 1, 2] = 1.0
        dIdx = DVGeo.totalSensitivity(dIdPt, ptName)

        np.testing.assert_allclose(dIdx["span"], dIdx_FD["span"], atol=1e-15)

    def test_spanDV_child(self, train=False):
        """
        Test span DV with child
        """
        DVGeo, DVGeoChild = commonUtils.setupDVGeo(self.base_path)

        # add design variables
        DVGeoChild.addGlobalDV(dvName="span", value=0.5, func=commonUtils.spanX, lower=0.1, upper=10, scale=1)
        DVGeo.addChild(DVGeoChild)

        points = np.zeros([2, 3])
        points[0, :] = [0.25, 0, 0]
        points[1, :] = [-0.25, 0, 0]
        ptName = "testPoints"
        DVGeo.addPointSet(points, ptName)

        nPt = points.size
        dIdx_FD = commonUtils.totalSensitivityFD(DVGeo, nPt, ptName)

        dIdPt = np.zeros([nPt, 2, 3])
        dIdPt[0, 0, 0] = 1.0
        dIdPt[1, 0, 1] = 1.0
        dIdPt[2, 0, 2] = 1.0
        dIdPt[3, 1, 0] = 1.0
        dIdPt[4, 1, 1] = 1.0
        dIdPt[5, 1, 2] = 1.0
        dIdx = DVGeo.totalSensitivity(dIdPt, ptName)

        np.testing.assert_allclose(dIdx["span"], dIdx_FD["span"], atol=1e-15)

    def test_embedding_solver(self):
        DVGeo = DVGeometry(os.path.join(self.base_path, "../../input_files/fuselage_ffd_severe.xyz"))

        test_points = [
            # Points that work with the linesearch fix on the pyspline projection code.
            # These points work with a reasonable iteration count (we have 50 for now).
            [0.49886, 0.31924, 0.037167],
            [0.49845, 0.32658, 0.039511],
            [0.76509, 0.29709, 0.037575],
            # The list of points below are much more problematic. The new fix can handle
            # some of them but they require a very large iteration count. The FFD here
            # is ridiculously difficult to embed, but leaving these points here because
            # they are a great test of robustness if someone wants to improve the solver.
            # even more down the line.
            # [0.76474, 0.30461, 0.039028],
            # [0.49988, 0.29506, 0.031219],
            # [0.49943, 0.30642, 0.03374],
            # [0.49792, 0.33461, 0.042548],
            # [0.49466, 0.35848, 0.06916],
            # [0.49419, 0.34003, 0.092855],
            # [0.49432, 0.33345, 0.09765],
            # [0.49461, 0.31777, 0.10775],
            # [0.49465, 0.31347, 0.11029],
            # [0.62736, 0.31001, 0.037233],
            # [0.76401, 0.32044, 0.042354],
            # [0.49322, 0.25633, 0.13751],
            # [0.49358, 0.26432, 0.13435],
        ]

        DVGeo.addPointSet(test_points, "test", nIter=50)

        # we evaluate the points. if the embedding fails, the points will not be identical
        new_points = DVGeo.update("test")

        np.testing.assert_allclose(test_points, new_points, atol=1e-15)

    def test_coord_xfer(self):
        DVGeo, _ = commonUtils.setupDVGeo(self.base_path)

        # create local DVs
        DVGeo.addLocalDV("xdir", lower=-1.0, upper=1.0, axis="x", scale=1.0)
        DVGeo.addLocalDV("ydir", lower=-1.0, upper=1.0, axis="y", scale=1.0)
        DVGeo.addLocalDV("zdir", lower=-1.0, upper=1.0, axis="z", scale=1.0)

        def coordXfer(coords, mode="fwd", applyDisplacement=True):
            rot_mat = np.array(
                [
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0],
                ]
            )

            if mode == "fwd":
                # apply the rotation first
                coords_new = np.dot(coords, rot_mat)

                # then the translation
                if applyDisplacement:
                    coords_new[:, 2] -= 5.0
            elif mode == "bwd":
                # apply the operations in reverse
                coords_new = coords.copy()
                if applyDisplacement:
                    coords_new[:, 2] += 5.0

                # and the rotation. note the rotation matrix is transposed
                # for switching the direction of rotation
                coords_new = np.dot(coords_new, rot_mat.T)

            return coords_new

        test_points = np.array(
            [
                # this point is normally outside the FFD volume,
                # but after the coordinate transfer,
                # it should be inside the FFD
                [0.5, 0.5, -4.5],
            ]
        )

        DVGeo.addPointSet(test_points, "test", coordXfer=coordXfer)

        # check if we can query the same point back
        pts_new = DVGeo.update("test")

        np.testing.assert_allclose(test_points, pts_new, atol=1e-15)

        # check derivatives
        nPt = test_points.size
        dIdx_FD = commonUtils.totalSensitivityFD(DVGeo, nPt, "test")

        dIdPt = np.zeros([3, 1, 3])
        dIdPt[0, 0, 0] = 1.0
        dIdPt[1, 0, 1] = 1.0
        dIdPt[2, 0, 2] = 1.0
        dIdx = DVGeo.totalSensitivity(dIdPt, "test")

        np.testing.assert_allclose(dIdx["xdir"], dIdx_FD["xdir"], atol=1e-15)
        np.testing.assert_allclose(dIdx["ydir"], dIdx_FD["ydir"], atol=1e-15)
        np.testing.assert_allclose(dIdx["zdir"], dIdx_FD["zdir"], atol=1e-15)

        # also test the fwd AD
        dIdxFwd = {
            "xdir": np.zeros((3, 12)),
            "zdir": np.zeros((3, 12)),
            "ydir": np.zeros((3, 12)),
        }
        # need to do it one DV at a time
        for ii in range(12):
            seed = np.zeros(12)
            seed[ii] = 1.0

            dIdxFwd["xdir"][:, ii] = DVGeo.totalSensitivityProd({"xdir": seed}, "test")
            dIdxFwd["ydir"][:, ii] = DVGeo.totalSensitivityProd({"ydir": seed}, "test")
            dIdxFwd["zdir"][:, ii] = DVGeo.totalSensitivityProd({"zdir": seed}, "test")

        np.testing.assert_allclose(dIdx["xdir"], dIdxFwd["xdir"], atol=1e-15)
        np.testing.assert_allclose(dIdx["ydir"], dIdxFwd["ydir"], atol=1e-15)
        np.testing.assert_allclose(dIdx["zdir"], dIdxFwd["zdir"], atol=1e-15)

    def train_custom_ray_projections(self, train=True):
        self.test_custom_ray_projections(train=train)

    def test_custom_ray_projections(self, train=False):
        """
        Custom Ray Projections Test
        This test checks the custom ray projection code.
        """
        refFile = os.path.join(self.base_path, "ref/test_custom_ray_projections.ref")
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test generalized axis node location section in plane")
            DVGeo = DVGeometry(os.path.join(self.base_path, "../../input_files/2x1x8_rectangle.xyz"))
            xfraction = 0.5
            yfraction = 0.5
            rotType = 0
            axis = np.array([0.0, 1.0, 1.0])

            DVGeo.addRefAxis(
                "RefAxis", axis=axis, xFraction=xfraction, yFraction=yfraction, alignIndex="k", rotType=rotType
            )
            DVGeo._finalize()

            handler.root_add_val("links_x", DVGeo.links_x, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

    # import xmlrunner
    # unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))
