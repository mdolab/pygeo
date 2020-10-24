from __future__ import print_function
import os
import unittest
import numpy as np
from baseclasses import BaseRegTest
import commonUtils 
from pygeo import geo_utils, DVGeometry, DVConstraints
from stl import mesh

class RegTestPyGeo(unittest.TestCase):

    N_PROCS = 1

    def setUp(self):
        # Store the path where this current script lives 
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def generate_dvgeo_dvcon_c172(self):
        meshfile = os.path.join(self.base_path, '../inputFiles/c172.stl')
        ffdfile = os.path.join(self.base_path, '../inputFiles/c172.xyz')
        testmesh = mesh.Mesh.from_file(meshfile)
        # test mesh dim 0 is triangle index
        # dim 1 is each vertex of the triangle
        # dim 2 is x, y, z dimension

        # create a DVGeo object with a few local thickness variables
        DVGeo = DVGeometry(ffdfile)
        nRefAxPts = DVGeo.addRefAxis("wing", xFraction=0.25, alignIndex="k")
        self.nTwist = nRefAxPts - 1
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z["wing"].coef[i] = val[i - 1]
        DVGeo.addGeoDVGlobal(dvName="twist", value=[0] * self.nTwist, func=twist, lower=-10, upper=10, scale=1)
        DVGeo.addGeoDVLocal("local", lower=-0.5, upper=0.5, axis="y", scale=1)

        # create a DVConstraints object for the wing
        DVCon =DVConstraints()
        DVCon.setDVGeo(DVGeo)
        p0 = testmesh.vectors[:,0,:] / 1000
        v1 = testmesh.vectors[:,1,:] / 1000 - p0
        v2 = testmesh.vectors[:,2,:] / 1000 - p0
        DVCon.setSurface([p0, v1, v2])

        return DVGeo, DVCon

    def c172_test_base(self, DVGeo, DVCon, handler):
        funcs = dict()
        DVCon.evalFunctions(funcs, includeLinear=True)
        handler.root_add_dict('funcs_base', funcs, rtol=1e-6, atol=1e-6)
        funcsSens=dict()
        DVCon.evalFunctionsSens(funcsSens, includeLinear=True)
        # regress the derivatives
        handler.root_add_dict('derivs_base', funcsSens, rtol=1e-6, atol=1e-6)

        return funcs, funcsSens

    def c172_test_twist(self, DVGeo, DVCon, handler):
        funcs = dict()
        funcsSens = dict()
        # change the DVs
        xDV = DVGeo.getValues()
        xDV['twist'] = np.linspace(0, 10, self.nTwist)
        DVGeo.setDesignVars(xDV)
        # check the constraint values changed
        DVCon.evalFunctions(funcs, includeLinear=True)

        handler.root_add_dict('funcs_twisted', funcs, rtol=1e-6, atol=1e-6)
        # check the derivatives are still right
        DVCon.evalFunctionsSens(funcsSens, includeLinear=True)
        # regress the derivatives
        handler.root_add_dict('derivs_twisted', funcsSens, rtol=1e-6, atol=1e-6)
        return funcs, funcsSens

    def c172_test_deformed(self, DVGeo, DVCon, handler):
        funcs = dict()
        funcsSens = dict()
        xDV = DVGeo.getValues()
        np.random.seed(37)
        xDV['local'] = np.random.normal(0.0, 0.05, 32)
        DVGeo.setDesignVars(xDV)
        DVCon.evalFunctions(funcs, includeLinear=True)
        DVCon.evalFunctionsSens(funcsSens, includeLinear=True)
        handler.root_add_dict('funcs_deformed', funcs, rtol=1e-6, atol=1e-6)
        handler.root_add_dict('derivs_deformed', funcsSens, rtol=1e-6, atol=1e-6)
        return funcs, funcsSens

    def train_1(self, train=True, refDeriv=True):
        self.test_1(train=train, refDeriv=refDeriv)
    
    def test_1(self, train=False, refDeriv=False):
        """
        Test 1: 1D Thickness Constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_01.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 1: 1D thickness constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            ptList = [[0.8, 0.0, 0.1],[0.8, 0.0, 5.0]]
            DVCon.addThicknessConstraints1D(ptList, nCon=10, axis=[0,1,0])


            funcs, funcsSens = self.c172_test_base(DVGeo, DVCon, handler)
            # 1D thickness should be all ones at the start
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_0'], np.ones(10), 
                                    name='thickness_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            # 1D thickness shouldn't change much under only twist
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_0'], np.ones(10), 
                                    name='thickness_twisted', rtol=1e-2, atol=1e-2)
            
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

    def train_2(self, train=True, refDeriv=True):
        self.test_2(train=train, refDeriv=refDeriv)
    
    def test_2(self, train=False, refDeriv=False):
        """
        Test 2: 2D Thickness Constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_02.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 2: 2D thickness constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            leList = [[0.7, 0.0, 0.1],[0.7, 0.0, 5.0]]
            teList = [[0.9, 0.0, 0.1],[0.9, 0.0, 5.0]]

            DVCon.addThicknessConstraints2D(leList, teList, 5, 5)


            funcs, funcsSens = self.c172_test_base(DVGeo, DVCon, handler)
            # 2D thickness should be all ones at the start
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_0'], np.ones(25), 
                                    name='thickness_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            # 2D thickness shouldn't change much under only twist
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_0'], np.ones(25), 
                                    name='thickness_twisted', rtol=1e-2, atol=1e-2)
            
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)
    
    def train_3(self, train=True, refDeriv=True):
        self.test_3(train=train, refDeriv=refDeriv)
    
    def test_3(self, train=False, refDeriv=False):
        """
        Test 3: Volume Constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_03.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 3: Volume constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            leList = [[0.7, 0.0, 0.1],[0.7, 0.0, 5.0]]
            teList = [[0.9, 0.0, 0.1],[0.9, 0.0, 5.0]]

            DVCon.addVolumeConstraint(leList, teList, 5, 5)


            funcs, funcsSens = self.c172_test_base(DVGeo, DVCon, handler)
            # Volume should be normalized to 1 at the start
            handler.assert_allclose(funcs['DVCon1_volume_constraint_0'], np.ones(1), 
                                    name='volume_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            # Volume shouldn't change much with twist only
            handler.assert_allclose(funcs['DVCon1_volume_constraint_0'], np.ones(1), 
                                    name='volume_twisted', rtol=1e-2, atol=1e-2)
            
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

    def train_4(self, train=True, refDeriv=True):
        self.test_4(train=train, refDeriv=refDeriv)
    
    def test_4(self, train=False, refDeriv=False):
        """
        Test 4: LeTe Constraint using the ilow, ihigh method
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_04.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 4: LETE constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            DVCon.addLeTeConstraints(0, 'iLow')
            DVCon.addLeTeConstraints(0, 'iHigh')


            funcs, funcsSens = self.c172_test_base(DVGeo, DVCon, handler)
            # LeTe constraints should be all zero at the start
            for i in range(2):
                handler.assert_allclose(funcs['DVCon1_lete_constraint_'+str(i)], np.zeros(4), 
                                        name='lete_'+str(i), rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            # Global DVs should produce no change, especially twist
            for i in range(2):
                handler.assert_allclose(funcs['DVCon1_lete_constraint_'+str(i)], np.zeros(4), 
                                        name='lete_twisted_'+str(i), rtol=1e-7, atol=1e-7)
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

    def train_5(self, train=True, refDeriv=True):
        self.test_5(train=train, refDeriv=refDeriv)
    
    def test_5(self, train=False, refDeriv=False):
        """
        Test 5: Thickness-to-chord constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_05.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 5: t/c constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            ptList = [[0.8, 0.0, 0.1],[0.8, 0.0, 5.0]]
            DVCon.addThicknessToChordConstraints1D(ptList, nCon=10, axis=[0,1,0], chordDir=[1,0,0])

            funcs, funcsSens = self.c172_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_thickness_to_chord_constraints_0'], np.ones(10), 
                                    name='toverc_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_thickness_to_chord_constraints_0'], np.ones(10), 
                                    name='toverc_twisted', rtol=1e-3, atol=1e-3)

            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

    def train_6(self, train=True, refDeriv=True):
        self.test_6(train=train, refDeriv=refDeriv)
    
    def test_6(self, train=False, refDeriv=False):
        """
        Test 6: Thickness-to-chord constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_06.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 6: t/c constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            DVCon.addSurfaceAreaConstraint()

            funcs, funcsSens = self.c172_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_surfaceArea_constraints_0'], np.ones(1), 
                                    name='surface_area_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_surfaceArea_constraints_0'], np.ones(1), 
                                    name='surface_area_twisted', rtol=1e-3, atol=1e-3)
                                    
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

if __name__ == '__main__':
    unittest.main()

    #import xmlrunner
    #unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))


