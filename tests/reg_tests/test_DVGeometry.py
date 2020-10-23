from __future__ import print_function
import os
import unittest
import numpy
from baseclasses import BaseRegTest
import commonUtils 
from pygeo import geo_utils, DVGeometry


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
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_01.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 1: Basic FFD, global DVs")
            DVGeo,DVGeoChild = commonUtils.setupDVGeo(self.base_path)
            #create global DVs on the parent
            DVGeo.addGeoDVGlobal('mainX', -1.0, commonUtils.mainAxisPoints,
                                lower=-1., upper=0., scale=1.0)
                                
            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_2(self, train=True, refDeriv=True):
        self.test_2(train=train, refDeriv=refDeriv)

    def test_2(self, train=False, refDeriv=False):
        """
        Test 2: Basic FFD, global DVs and local DVs
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_02.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 2: Basic FFD, global DVs and local DVs")
            DVGeo,DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            #create global DVs on the parent
            DVGeo.addGeoDVGlobal('mainX', -1.0, commonUtils.mainAxisPoints,
                                lower=-1., upper=0., scale=1.0)
            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)


    def train_3(self, train=True, refDeriv=True):
        self.test_3(train=train, refDeriv=refDeriv)

    def test_3(self, train=False, refDeriv=False):
        """
        Test 3: Basic + Nested FFD, global DVs only
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_03.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 3: Basic + Nested FFD, global DVs only")
            DVGeo,DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            #create global DVs on the parent
            DVGeo.addGeoDVGlobal('mainX', -1.0, commonUtils.mainAxisPoints,
                                lower=-1., upper=0., scale=1.0)
            #create global DVs on the child
            DVGeoChild.addGeoDVGlobal('nestedX', -0.5, commonUtils.childAxisPoints,
                                    lower=-1., upper=0., scale=1.0)
            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)


    def train_4(self, train=True, refDeriv=True):
        self.test_4(train=train, refDeriv=refDeriv)

    def test_4(self, train=False, refDeriv=False):
        """
        Test 4: Basic + Nested FFD, global DVs and local DVs on parent global on child
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_04.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 4: Basic + Nested FFD, global DVs and local DVs on parent global on child")
            DVGeo,DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            #create global DVs on the parent
            DVGeo.addGeoDVGlobal('mainX', -1.0, commonUtils.mainAxisPoints,
                                lower=-1., upper=0., scale=1.0)
            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            #create global DVs on the child
            DVGeoChild.addGeoDVGlobal('nestedX', -0.5, commonUtils.childAxisPoints,
                                    lower=-1., upper=0., scale=1.0)
            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)



    def train_5(self, train=True, refDeriv=True):
        self.test_5(train=train, refDeriv=refDeriv)

    def test_5(self, train=False, refDeriv=False):
        """
        Test 5: Basic + Nested FFD, global DVs and local DVs on both parent and child
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_05.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 5: Basic + Nested FFD, global DVs and local DVs on both parent and child")
            DVGeo,DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            #create global DVs on the parent
            DVGeo.addGeoDVGlobal('mainX', -1.0, commonUtils.mainAxisPoints,
                                lower=-1., upper=0., scale=1.0)
            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)
            #create global DVs on the child
            DVGeoChild.addGeoDVGlobal('nestedX', -0.5, commonUtils.childAxisPoints,
                                    lower=-1., upper=0., scale=1.0)
            DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
            DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
            DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)


    def train_6(self, train=True, refDeriv=True):
        self.test_6(train=train, refDeriv=refDeriv)

    def test_6(self, train=False, refDeriv=False):
        """
        Test 6: Basic + Nested FFD, local DVs only
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_06.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 6: Basic + Nested FFD, local DVs only")
            DVGeo,DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
            DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
            DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)


    def train_7(self, train=True, refDeriv=True):
        self.test_7(train=train, refDeriv=refDeriv)

    def test_7(self, train=False, refDeriv=False):
        """
        Test 7: Basic + Nested FFD, local DVs only on parent
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_07.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 7: Basic + Nested FFD, local DVs only on parent")
            DVGeo,DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)

    def train_8(self, train=True, refDeriv=True):
        self.test_8(train=train, refDeriv=refDeriv)

    def test_8(self, train=False, refDeriv=False):
        """
        Test 8: Basic + Nested FFD, local DVs only on child
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_08.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 8: Basic + Nested FFD, local DVs only on child")
            DVGeo,DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
            DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
            DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)


    def train_9(self, train=True, refDeriv=True):
        self.test_9(train=train, refDeriv=refDeriv)

    def test_9(self, train=False, refDeriv=False):
        """
        Test 9: Basic + Nested FFD, local DVs only on parent, global on child
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_09.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 9: Basic + Nested FFD, local DVs only on parent, global on child")
            DVGeo,DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            DVGeoChild.addGeoDVGlobal('nestedX', -0.5, commonUtils.childAxisPoints,
                                    lower=-1., upper=0., scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)


    def train_10(self, train=True, refDeriv=True):
        self.test_10(train=train, refDeriv=refDeriv)

    def test_10(self, train=False, refDeriv=False):
        """
        Test 10: Basic + Nested FFD, local DVs only on parent, global and local on child
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_10.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 10: Basic + Nested FFD, local DVs only on parent, global and local on child")
            DVGeo,DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            DVGeoChild.addGeoDVGlobal('nestedX', -0.5, commonUtils.childAxisPoints,
                                    lower=-1., upper=0., scale=1.0)
            DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
            DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
            DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivities(DVGeo, refDeriv, handler)


    def train_11(self, train=True, refDeriv=True):
        self.test_11(train=train, refDeriv=refDeriv)

    def test_11(self, train=False, refDeriv=False):
        """
        Test 11: Basic + Nested FFD, global DVs and local DVs on parent local on child
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_11.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 11: Basic + Nested FFD, global DVs and local DVs on parent local on child")
            DVGeo,DVGeoChild = commonUtils.setupDVGeo(self.base_path)

            #create global DVs on the parent
            DVGeo.addGeoDVGlobal('mainX', -1.0, commonUtils.mainAxisPoints,
                                lower=-1., upper=0., scale=1.0)
            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            #create global DVs on the child
            DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
            DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
            DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

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
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_12.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 12: D8 FFD, global DVs")
            DVGeo,DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            #create global DVs on the parent
            axisX = [0.,26.,30.5,32.5, 34.0]
            DVGeo.addGeoDVGlobal('mainX', axisX , commonUtils.mainAxisPoints,
                                lower=0., upper=35., scale=1.0)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)
            
            handler.root_print("Test 12b: D8 FFD, random DV perturbation on test 10")
            xDV = DVGeo.getValues()
            for key in xDV:
                numpy.random.seed(42)
                xDV[key]+=numpy.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

    def train_13(self, train=True, refDeriv=True):
        self.test_13(train=train, refDeriv=refDeriv)

    def test_13(self, train=False, refDeriv=False):
        """
        Test 13: D8 FFD, global DVs and local DVs
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_13.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 13: D8 FFD, global DVs and local DVs")
            DVGeo,DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            #create global DVs on the parent
            axisX = [0.,26.,30.5,32.5, 34.0]
            DVGeo.addGeoDVGlobal('mainX', axisX , commonUtils.mainAxisPoints,
                                lower=0., upper=35., scale=1.0)
            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)
        
            handler.root_print("Test 13b: D8 FFD, random DV perturbation on test 11")
            xDV = DVGeo.getValues()
            
            for key in xDV:
                numpy.random.seed(42)
                xDV[key]+=numpy.random.rand(len(xDV[key]))
            
            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)



    def train_14(self, train=True, refDeriv=True):
        self.test_14(train=train, refDeriv=refDeriv)

    def test_14(self, train=False, refDeriv=False):
        """
        Test 14
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_14.ref')
        with BaseRegTest(refFile, train=train) as handler:            
            handler.root_print("Test 14: D8 + Nozzle FFD, global DVs only")
            DVGeo,DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            #create global DVs on the parent
            axisX = [0.,26.,30.5,32.5, 34.0]
            DVGeo.addGeoDVGlobal('mainX', axisX , commonUtils.mainAxisPointsD8,
                                lower=0., upper=35., scale=1.0)
            #create global DVs on the child
            childAxisX = [32.4, 34]
            DVGeoChild.addGeoDVGlobal('nestedX',childAxisX, commonUtils.childAxisPointsD8,
                                    lower=0., upper=35., scale=1.0)
            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)
            
            handler.root_print("Test 14b: D8 + Nozzle FFD, random DV perturbation on test 12")
            xDV = DVGeo.getValues()
            for key in xDV:
                numpy.random.seed(42)
                xDV[key]+=numpy.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)
            

    def train_15(self, train=True, refDeriv=True):
        self.test_15(train=train, refDeriv=refDeriv)

    def test_15(self, train=False, refDeriv=False):
        """
        Test 15
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_15.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 15: D8 + Nozzle FFD, global DVs and local DVs on parent global on child")
            DVGeo,DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            #create global DVs on the parent
            axisX = [0.,26.,30.5,32.5, 34.0]
            DVGeo.addGeoDVGlobal('mainX', axisX , commonUtils.mainAxisPoints,
                                lower=0., upper=35., scale=1.0)
            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            #create global DVs on the child
            childAxisX = [32.4, 34]
            DVGeoChild.addGeoDVGlobal('nestedX',childAxisX, commonUtils.childAxisPoints,
                                    lower=0., upper=35., scale=1.0)
            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)
            
            handler.root_print("Test 15b: D8 + Nozzle FFD, random DV perturbation on test 13")
            xDV = DVGeo.getValues()
            for key in xDV:
                numpy.random.seed(42)
                xDV[key]+=numpy.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)


    def train_16(self, train=True, refDeriv=True):
        self.test_16(train=train, refDeriv=refDeriv)

    def test_16(self, train=False, refDeriv=False):
        """
        Test 16
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_16.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 16: D8 + Nozzle FFD,  global DVs and local DVs on both parent and child")
            DVGeo,DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            #create global DVs on the parent
            axisX = [0.,26.,30.5,32.5, 34.0]
            DVGeo.addGeoDVGlobal('mainX', axisX , commonUtils.mainAxisPoints,
                                lower=0., upper=35., scale=1.0)
            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)
            #create global DVs on the child
            childAxisX = [32.4, 34]
            DVGeoChild.addGeoDVGlobal('nestedX',childAxisX, commonUtils.childAxisPoints,
                                    lower=0., upper=35., scale=1.0)
            DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
            DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
            DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)
            
            handler.root_print("Test 14b: D8 + Nozzle FFD, random DV perturbation on test 14")
            xDV = DVGeo.getValues()
            for key in xDV:
                numpy.random.seed(42)
                xDV[key]+=numpy.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

    def train_17(self, train=True, refDeriv=True):
        self.test_17(train=train, refDeriv=refDeriv)

    def test_17(self, train=False, refDeriv=False):
        """
        Test 17
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_17.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 17: D8 + Nozzle FFD, local DVs only")
            DVGeo,DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
            DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
            DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)
            
            handler.root_print("Test 17b: D8 + Nozzle FFD, random DV perturbationon test 15")
            xDV = DVGeo.getValues()
            for key in xDV:
                numpy.random.seed(42)
                xDV[key]+=numpy.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)



    def train_18(self, train=True, refDeriv=True):
        self.test_18(train=train, refDeriv=refDeriv)

    def test_18(self, train=False, refDeriv=False):
        """
        Test 18
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_18.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 18: D8 + Nozzle FFD, local DVs only on parent, global on child")
            DVGeo,DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            childAxisX = [32.4, 34]
            DVGeoChild.addGeoDVGlobal('nestedX',childAxisX, commonUtils.childAxisPoints,
                                    lower=0., upper=35., scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

            handler.root_print("Test 18b: D8 + Nozzle FFD, random DV perturbationon test 16")
            xDV = DVGeo.getValues()
            for key in xDV:
                numpy.random.seed(42)
                xDV[key]+=numpy.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)


    def train_19(self, train=True, refDeriv=True):
        self.test_19(train=train, refDeriv=refDeriv)

    def test_19(self, train=False, refDeriv=False):
        """
        Test 19
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_19.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 19: D8 + Nozzle FFD, local DVs only on parent, global and local on child")
            DVGeo,DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            childAxisX = [32.4, 34]
            DVGeoChild.addGeoDVGlobal('nestedX',childAxisX, commonUtils.childAxisPoints,
                                    lower=0., upper=35., scale=1.0)
            DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
            DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
            DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)

            handler.root_print("Test 19b: D8 + Nozzle FFD,  random DV perturbationon test 17")
            xDV = DVGeo.getValues()
            for key in xDV:
                numpy.random.seed(42)
                xDV[key]+=numpy.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)
    


    def train_20(self, train=True, refDeriv=True):
        self.test_20(train=train, refDeriv=refDeriv)

    def test_20(self, train=False, refDeriv=False):
        """
        Test 20
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_20.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 20: D8 + Nozzle FFD, global DVs and local DVs on parent local on child")
            DVGeo,DVGeoChild = commonUtils.setupDVGeoD8(self.base_path, refDeriv)

            #create global DVs on the parent
            axisX = [0.,26.,30.5,32.5, 34.0]
            DVGeo.addGeoDVGlobal('mainX', axisX , commonUtils.mainAxisPoints,
                                lower=0., upper=35., scale=1.0)
            #create local DVs on the parent
            DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
            DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
            DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

            #create global DVs on the child
            DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
            DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
            DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

            DVGeo.addChild(DVGeoChild)

            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)
            
            handler.root_print("Test 20b: D8 + Nozzle FFD,  random DV perturbationon test 18")
            xDV = DVGeo.getValues()
            for key in xDV:
                numpy.random.seed(42)
                xDV[key]+=numpy.random.rand(len(xDV[key]))

            DVGeo.setDesignVars(xDV)
            commonUtils.testSensitivitiesD8(DVGeo, refDeriv, handler)


    def train_21(self, train=True, refDeriv=True):
        self.test_21(train=train, refDeriv=refDeriv)

    def test_21(self, train=False, refDeriv=False):
        """
        Test 21
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_21.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 21: Axisymmetric FFD, global and local DVs")

            # Test with a single point along the 45 ` degree theta direction            
            DVGeo = commonUtils.setupDVGeoAxi(self.base_path)
          
            DVGeo.addGeoDVGlobal('mainAxis', numpy.zeros(1), commonUtils.mainAxisPointAxi)

            DVGeo.addGeoDVLocal('x_axis', lower=-2, upper=2, axis="x")
            DVGeo.addGeoDVLocal('z_axis', lower=-2, upper=2, axis="z")
            DVGeo.addGeoDVLocal('y_axis', lower=-2, upper=2, axis="y")

            ptName = "point"
            s_pts = numpy.array([[0, .5, .5],], dtype="float")
            DVGeo.addPointSet(points=s_pts, ptName=ptName)

        # generate dIdPt
            nPt = 3
            dIdPt = numpy.zeros([nPt,1,3])
            dIdPt[0,0,0] = 1.0
            dIdPt[1,0,1] = 1.0
            dIdPt[2,0,2] = 1.0

            if refDeriv:
                # Generate reference derivatives
                refPoints = DVGeo.update(ptName)
                nPt = 3*refPoints.shape[0]
                step = 1e-5
                J_fd = commonUtils.totalSensitivityFD(DVGeo,nPt,ptName,step)
                handler.root_add_dict(J_fd,rtol=1e-7,atol=1e-7)

            else:
                # Compute the analytic derivatives
                dIdx = DVGeo.totalSensitivity(dIdPt,ptName)
                handler.root_add_dict('dIdx',dIdx,rtol=1e-7,atol=1e-7)


    def train_22(self, train=True, refDeriv=True):
        self.test_22(train=train, refDeriv=refDeriv)

    def test_22(self, train=False, refDeriv=False):
        """
        Test 22
        """
        refFile = os.path.join(self.base_path,'ref/test_DVGeometry_22.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test FFD writing function")

            # Write duplicate of outerbox FFD
            axes = ['i', 'k', 'j']
            slices = numpy.array([
                # Slice 1
                [[[-1, -1, -1], [-1, 1, -1]],
                [[-1, -1, 1], [-1, 1, 1]]],
                # Slice 2
                [[[1, -1, -1], [1, 1, -1]],
                [[1, -1, 1], [1, 1, 1]]],
                # Slice 3
                [[[2, -1, -1], [2, 1, -1]],
                [[2, -1, 1], [2, 1, 1]]],
            ])

            N0 = [2,2]
            N1 = [2,2]
            N2 = [2,2]

            copyName = os.path.join(self.base_path,'../inputFiles/test1.xyz')
            geo_utils.write_wing_FFD_file(copyName, slices, N0, N1, N2, axes=axes)

            # Load original and duplicate
            origFFD = DVGeometry(os.path.join(self.base_path,'../inputFiles/outerBoxFFD.xyz'))
            copyFFD = DVGeometry(copyName)
            norm_diff = numpy.linalg.norm(origFFD.FFD.coef - copyFFD.FFD.coef)
            handler.par_add_norm('norm', norm_diff, rtol=1e-7, atol=1e-7)
            os.remove(copyName)

            

if __name__ == '__main__':
    unittest.main()

    #import xmlrunner
    #unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))


