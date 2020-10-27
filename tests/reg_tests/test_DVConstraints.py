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

    def evalFunctionsSensFD(self, DVGeo, DVCon, fdstep=1e-4):
        funcs = dict()
        DVCon.evalFunctions(funcs, includeLinear=True)
        # make a deep copy of this
        outdims = dict()
        for key in funcs.keys():
            val = funcs[key]
            if isinstance(val, np.ndarray):
                outdims[key] = val.shape[0]
                funcs[key] = val.copy()   
            elif isinstance(val, (list, tuple)):
                outdims[key] = len(val)
            else:
                outdims[key] = 1

        xDV = DVGeo.getValues()
        indims = dict()
        for key in xDV.keys():
            val = xDV[key]
            indims[key] = val.shape[0]

        # setup the output data structure
        funcsSens = dict()
        for outkey in funcs.keys():
            funcsSens[outkey] = dict()
            for inkey in xDV.keys():
                nRows = outdims[outkey]
                nCols = indims[inkey]
                funcsSens[outkey][inkey] = np.zeros((nRows, nCols))
        # now do finite differencing
        for inkey in xDV.keys():
            baseVar = xDV[inkey].copy()
            nDV = len(baseVar)
            for array_ind in range(nDV):
                xDV[inkey][array_ind] = baseVar[array_ind] + fdstep
                DVGeo.setDesignVars(xDV)
                funcs_fd = dict()
                DVCon.evalFunctions(funcs_fd, includeLinear=True)
                for outkey in funcs.keys():
                    temp_a = funcs_fd[outkey]
                    temp_b = funcs[outkey]
                    diff = temp_a - temp_b
                    deriv_temp = diff / fdstep
                    funcsSens[outkey][inkey][:,array_ind] = deriv_temp
                xDV[inkey][array_ind] = baseVar[array_ind]
        DVGeo.setDesignVars(xDV)
        DVCon.evalFunctions(dict())
        return funcsSens

    def generate_dvgeo_dvcon_rect(self):
        meshfile = os.path.join(self.base_path, '../inputFiles/2x1x8_rectangle.stl')
        ffdfile = os.path.join(self.base_path, '../inputFiles/2x1x8_rectangle.xyz')
        testmesh = mesh.Mesh.from_file(meshfile)
        # test mesh dim 0 is triangle index
        # dim 1 is each vertex of the triangle
        # dim 2 is x, y, z dimension

        # create a DVGeo object with a few local thickness variables
        DVGeo = DVGeometry(ffdfile)
        nRefAxPts = DVGeo.addRefAxis("wing", xFraction=0.5, alignIndex="k")
        self.nTwist = nRefAxPts - 1
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z["wing"].coef[i] = val[i - 1]
        DVGeo.addGeoDVGlobal(dvName="twist", value=[0] * self.nTwist, func=twist, lower=-10, upper=10, scale=1)
        DVGeo.addGeoDVLocal("local", lower=-0.5, upper=0.5, axis="y", scale=1)

        # create a DVConstraints object for the wing
        DVCon =DVConstraints()
        DVCon.setDVGeo(DVGeo)
        p0 = testmesh.vectors[:,0,:]
        v1 = testmesh.vectors[:,1,:] - p0
        v2 = testmesh.vectors[:,2,:] - p0
        DVCon.setSurface([p0, v1, v2])

        return DVGeo, DVCon

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

    def generic_test_base(self, DVGeo, DVCon, handler, checkDerivs=True):
        linear_constraint_keywords = ['lete', 'monotonic', 'linear_constraint']
        funcs = dict()
        DVCon.evalFunctions(funcs, includeLinear=True)
        handler.root_add_dict('funcs_base', funcs, rtol=1e-6, atol=1e-6)
        funcsSens=dict()
        DVCon.evalFunctionsSens(funcsSens, includeLinear=True)
        # regress the derivatives
        if checkDerivs:
            handler.root_add_dict('derivs_base', funcsSens, rtol=1e-6, atol=1e-6)
            funcsSensFD = self.evalFunctionsSensFD(DVGeo, DVCon, fdstep=1e-4)
            for outkey in funcs.keys():
                for inkey in DVGeo.getValues().keys():
                    try: 
                        analytic = funcsSens[outkey][inkey]
                        fd = funcsSensFD[outkey][inkey]
                        handler.assert_allclose(analytic, fd, 
                            name='finite_diff_check', rtol=1e-3, atol=1e-3)
                    except KeyError:
                        if any(sbstr in outkey for sbstr in linear_constraint_keywords):
                            # linear constraints only have their affected DVs in the dict
                            pass
                        else:
                            raise
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


            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler, checkDerivs=True)
            # 1D thickness should be all ones at the start
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_0'], np.ones(10), 
                                    name='thickness_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            # 1D thickness shouldn't change much under only twist
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_0'], np.ones(10), 
                                    name='thickness_twisted', rtol=1e-2, atol=1e-2)
            
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

    def test_1b(self, train=False, refDeriv=False):
        """
        Test 1b: 1D Thickness Constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_01b.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 1b: 1D thickness constraint, rectangular box")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_rect()

            ptList = [[0.0, 0.0, 0.1],[0.0, 0.0, 5.0]]
            ptList2 = [[-0.5, 0.0, 2.0],[0.5, 0.0, 2.0]]
            DVCon.addThicknessConstraints1D(ptList, nCon=3, axis=[0,1,0], scaled=False)
            DVCon.addThicknessConstraints1D(ptList, nCon=3, axis=[1,0,0], scaled=False)
            DVCon.addThicknessConstraints1D(ptList2, nCon=3, axis=[0,0,1], scaled=False)

            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            # check that unscaled thicknesses are being computed correctly at baseline
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_0'], np.ones(3), 
                                    name='thickness_base', rtol=1e-7, atol=1e-7)
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_1'], 2.0*np.ones(3), 
                                    name='thickness_base', rtol=1e-7, atol=1e-7)
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_2'], 8.0*np.ones(3), 
                                    name='thickness_base', rtol=1e-7, atol=1e-7)
    
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


            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            # 2D thickness should be all ones at the start
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_0'], np.ones(25), 
                                    name='thickness_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            # 2D thickness shouldn't change much under only twist
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_0'], np.ones(25), 
                                    name='thickness_twisted', rtol=1e-2, atol=1e-2)
            
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)
    
    def test_2b(self, train=False, refDeriv=False):
        """
        Test 2b: 2D Thickness Constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_02b.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 2: 2D thickness constraint, rectangular box")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_rect()

            leList = [[-0.25, 0.0, 0.1],[-0.25, 0.0, 7.9]]
            teList = [[0.75, 0.0, 0.1],[0.75, 0.0, 7.9]]

            leList2 = [[0.0, -0.25, 0.1],[0.0, -0.25, 7.9]]
            teList2 = [[0.0, 0.25, 0.1],[0.0, 0.25, 7.9]]

            leList3 = [[-0.5, -0.25, 0.1],[0.5, -0.25, 0.1]]
            teList3 = [[-0.5, 0.25, 0.1],[0.5, 0.25, 0.1]]

            DVCon.addThicknessConstraints2D(leList, teList, 2, 2, scaled=False)
            DVCon.addThicknessConstraints2D(leList2, teList2, 2, 2, scaled=False)
            DVCon.addThicknessConstraints2D(leList3, teList3, 2, 2, scaled=False)

            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            # 2D thickness should be all ones at the start
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_0'], np.ones(4), 
                                    name='thickness_base', rtol=1e-7, atol=1e-7)
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_1'], 2.0*np.ones(4), 
                                    name='thickness_base', rtol=1e-7, atol=1e-7)
            handler.assert_allclose(funcs['DVCon1_thickness_constraints_2'], 8.0*np.ones(4), 
                                    name='thickness_base', rtol=1e-7, atol=1e-7)
    
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


            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            # Volume should be normalized to 1 at the start
            handler.assert_allclose(funcs['DVCon1_volume_constraint_0'], np.ones(1), 
                                    name='volume_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            # Volume shouldn't change much with twist only
            handler.assert_allclose(funcs['DVCon1_volume_constraint_0'], np.ones(1), 
                                    name='volume_twisted', rtol=1e-2, atol=1e-2)
            
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

    def test_3b(self, train=False, refDeriv=False):
        """
        Test 3b: Volume Constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_03b.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 3b: Volume constraint, rectangular box")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_rect()

            # this projects in the z direction which is of dimension 8
            # 1x0.5x8 = 4
            leList = [[-0.5, -0.25, 0.1],[0.5, -0.25, 0.1]]
            teList = [[-0.5, 0.25, 0.1],[0.5, 0.25, 0.1]]

            DVCon.addVolumeConstraint(leList, teList, 4, 4, scaled=False)

            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            # Volume should be normalized to 1 at the start
            handler.assert_allclose(funcs['DVCon1_volume_constraint_0'], 4.0*np.ones(1), 
                                    name='volume_base', rtol=1e-7, atol=1e-7)

    def test_4(self, train=False, refDeriv=False):
        """
        Test 4: LeTe Constraint using the ilow, ihigh method

        There's no need to test this with the rectangular box
        because it doesn't depend on a projected pointset (only the FFD)
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_04.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 4: LETE constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            DVCon.addLeTeConstraints(0, 'iLow')
            DVCon.addLeTeConstraints(0, 'iHigh')


            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
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

    def test_5(self, train=False, refDeriv=False):
        """
        Test 5: Thickness-to-chord constraint

        There's no need to test this with the rectangular box
        because it doesn't depend on a projected pointset (only the FFD)
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_05.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 5: t/c constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            ptList = [[0.8, 0.0, 0.1],[0.8, 0.0, 5.0]]
            DVCon.addThicknessToChordConstraints1D(ptList, nCon=10, axis=[0,1,0], chordDir=[1,0,0])

            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_thickness_to_chord_constraints_0'], np.ones(10), 
                                    name='toverc_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_thickness_to_chord_constraints_0'], np.ones(10), 
                                    name='toverc_twisted', rtol=1e-3, atol=1e-3)

            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

    def test_6(self, train=False, refDeriv=False):
        """
        Test 6: Surface area constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_06.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 6: surface area constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            DVCon.addSurfaceAreaConstraint()

            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_surfaceArea_constraints_0'], np.ones(1), 
                                    name='surface_area_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_surfaceArea_constraints_0'], np.ones(1), 
                                    name='surface_area_twisted', rtol=1e-3, atol=1e-3)
                                    
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

    def test_6b(self, train=False, refDeriv=False):
        """
        Test 6b: Surface area constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_06b.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 6: surface area constraint, rectangular box")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_rect()

            DVCon.addSurfaceAreaConstraint(scaled=False)
            # 2x1x8 box has surface area 2*(8*2+1*2+8*1) = 52
            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_surfaceArea_constraints_0'], 52.*np.ones(1), 
                                    name='surface_area_base', rtol=1e-7, atol=1e-7)

    def test_7(self, train=False, refDeriv=False):
        """
        Test 7: Projected area constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_07.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 7: projected area constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            DVCon.addProjectedAreaConstraint()

            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_projectedArea_constraints_0'], np.ones(1), 
                                    name='projected_area_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
                                    
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

    def test_7b(self, train=False, refDeriv=False):
        """
        Test 7b: Projected area constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_07b.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 7b: projected area constraint, rectangular box")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_rect()

            DVCon.addProjectedAreaConstraint(scaled=False)
            DVCon.addProjectedAreaConstraint(axis='z', scaled=False)
            DVCon.addProjectedAreaConstraint(axis='x', scaled=False)

            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler, checkDerivs=False)
            handler.assert_allclose(funcs['DVCon1_projectedArea_constraints_0'], 8*2*np.ones(1), 
                                    name='projected_area_base', rtol=1e-7, atol=1e-7)
            handler.assert_allclose(funcs['DVCon1_projectedArea_constraints_1'], 1*2*np.ones(1), 
                                    name='projected_area_base', rtol=1e-7, atol=1e-7)
            handler.assert_allclose(funcs['DVCon1_projectedArea_constraints_2'], 8*1*np.ones(1), 
                                    name='projected_area_base', rtol=1e-7, atol=1e-7)                                        

    def test_8(self, train=False, refDeriv=False):
        """
        Test 8: Circularity constraint

        No need to test this with the rectangular box
        because it only depends on the FFD, no projected points
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_08.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 8: Circularity constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            DVCon.addCircularityConstraint(origin=[0.8, 0.0, 2.5], rotAxis=[0., 0., 1.], 
                                           radius=0.1, zeroAxis=[0.,1.,0.], angleCW=180., angleCCW=180.,
                                           nPts=10)

            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_circularity_constraints_0'], np.ones(9), 
                                    name='circularity_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_circularity_constraints_0'], np.ones(9), 
                                    name='circularity_twisted', rtol=1e-7, atol=1e-7)            
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

    def test_9(self, train=False, refDeriv=False):
        """
        Test 9: Colinearity constraint

        No need to test this with the rectangular box
        because it only depends on the FFD, no projected points
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_09.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 9: Colinearity constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            DVCon.addColinearityConstraint(np.array([0.7, 0.0, 1.0]), lineAxis=np.array([0.,0.,1.]), 
                                           distances=[0., 1., 2.5])

            # Skip derivatives check here because true zero values cause difficulties for the partials
            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler, checkDerivs=False)
            handler.assert_allclose(funcs['DVCon1_colinearity_constraints_0'], np.zeros(3), 
                                    name='colinearity_base', rtol=1e-7, atol=1e-7)
            
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)       
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)

    def test_10(self, train=False, refDeriv=False):
        """
        Test 10: LinearConstraintShape

        No need to test this with the rectangular box
        because it only depends on the FFD, no projected points
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_10.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 10: LinearConstraintShape, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()
            lIndex = DVGeo.getLocalIndex(0)
            indSetA = []; indSetB = [];
            for i in range(lIndex.shape[0]):
                indSetA.append(lIndex[i, 0, 0])
                indSetB.append(lIndex[i, 0, 1])
            DVCon.addLinearConstraintsShape(indSetA, indSetB,
                                           factorA=1.0, factorB=-1.0,
                                           lower=0, upper=0)
            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)       
            funcs, funcsSens = self.c172_test_deformed(DVGeo, DVCon, handler)
    
    def test_11(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_11.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 11: CompositeVolumeConstraint, rectangular box")
            DVGeo, DVCon = self.generate_dvgeo_dvcon_rect()

            # this projects in the z direction which is of dimension 8
            # 1x0.5x8 = 4
            leList = [[-0.5, -0.25, 0.1],[0.5, -0.25, 0.1]]
            teList = [[-0.5, 0.25, 0.1],[0.5, 0.25, 0.1]]
            
            # this projects in the x direction which is of dimension 2
            # 2x0.6x7.8 = 9.36
            leList2 = [[0., -0.25, 0.1],[0., -0.25, 7.9]]
            teList2 = [[0., 0.35, 0.1],[0., 0.35, 7.9]]
            DVCon.addVolumeConstraint(leList, teList, 4, 4, scaled=False, addToPyOpt=False)
            DVCon.addVolumeConstraint(leList2, teList2, 4, 4, scaled=False, addToPyOpt=False)
            vols = ['DVCon1_volume_constraint_0',
                    'DVCon1_volume_constraint_1']
            DVCon.addCompositeVolumeConstraint(vols=vols, scaled=False)


            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            # Volume should be normalized to 1 at the start
            handler.assert_allclose(funcs['DVCon1_volume_constraint_0'], 4.0*np.ones(1), 
                                    name='volume1_base', rtol=1e-7, atol=1e-7)
            handler.assert_allclose(funcs['DVCon1_volume_constraint_1'], 9.36*np.ones(1), 
                                    name='volume2_base', rtol=1e-7, atol=1e-7)
            handler.assert_allclose(funcs['DVCon1_composite_volume_constraint_2'], 13.36*np.ones(1), 
                                    name='volume_composite_base', rtol=1e-7, atol=1e-7)

    def test_12(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_12.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 12: LocationConstraints1D, rectangular box")
            DVGeo, DVCon = self.generate_dvgeo_dvcon_rect()

            ptList = [[0.0, 0.0, 0.0],[0.0, 0.0, 8.0]]
            ptList2 = [[0.0, 0.2, 0.0],[0.0, -0.2, 8.0]]

            # TODO this constraint seems buggy. for example, when scaled, returns a bunch of NaNs
            DVCon.addLocationConstraints1D(ptList=ptList, nCon=10, scaled=False)
            DVCon.addProjectedLocationConstraints1D(ptList=ptList2, nCon=10, scaled=False, axis=[0.,1.,0.])
            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)

            exact_vals = np.zeros((30,))
            exact_vals[2::3] = np.linspace(0,8,10)
            # should be 10 evenly spaced points along the z axis originating from 0,0,0
            handler.assert_allclose(funcs['DVCon1_location_constraints_0'], 
                                    exact_vals, 
                                    name='locations_match', rtol=1e-7, atol=1e-7)
            handler.assert_allclose(funcs['DVCon1_location_constraints_1'], 
                                    exact_vals, 
                                    name='projected_locations_match', rtol=1e-7, atol=1e-7)

    def test_13(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_13.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 13: PlanarityConstraint, rectangular box")
            DVGeo, DVCon = self.generate_dvgeo_dvcon_rect()

            DVCon.addPlanarityConstraint(origin=[0.,0.5,0.0], planeAxis=[0.,1.,0.])
            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
    
    def test_13b(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_13b.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 13: PlanarityConstraint, rectangular box")
            ffdfile = os.path.join(self.base_path, '../inputFiles/2x1x8_rectangle.xyz')
            DVGeo = DVGeometry(ffdfile)
            DVGeo.addGeoDVLocal("local", lower=-0.5, upper=0.5, axis="y", scale=1)

            # create a DVConstraints object with a simple plane consisting of 2 triangles
            DVCon =DVConstraints()
            DVCon.setDVGeo(DVGeo)

            p0 = np.zeros(shape=(2,3))
            p1 = np.zeros(shape=(2,3))
            p2 = np.zeros(shape=(2,3))

            vertex1 = np.array([0.5, -0.25, 0.0])
            vertex2 = np.array([0.5, -0.25, 4.0])
            vertex3 = np.array([-0.5, -0.25, 0.0])
            vertex4 = np.array([-0.5, -0.25, 4.0])

            p0[:,:] = vertex1
            p2[:,:] = vertex4
            p1[0,:] = vertex2
            p1[1,:] = vertex3

            v1 = p1 - p0
            v2 = p2 - p0
            DVCon.setSurface([p0, v1, v2])

            DVCon.addPlanarityConstraint(origin=[0.,-0.25,2.0], planeAxis=[0.,1.,0.])

            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler, checkDerivs=False)

            # this should be coplanar and the planarity constraint shoudl be zero
            handler.assert_allclose(funcs['DVCon1_planarity_constraints_0'], np.zeros(1), 
                                    name='planarity', rtol=1e-7, atol=1e-7)

    def test_14(self, train=False, refDeriv=False):
        """
        Test 14: Monotonic constraint
        """
        refFile = os.path.join(self.base_path,'ref/test_DVConstraints_14.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 14: Monotonic constraint, C172 wing")

            DVGeo, DVCon = self.generate_dvgeo_dvcon_c172()

            DVCon.addMonotonicConstraints("twist")
            DVCon.addMonotonicConstraints("twist", start=1, stop=2)


            funcs, funcsSens = self.generic_test_base(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_monotonic_constraint_0'], np.zeros(2), 
                                    name='monotonicity', rtol=1e-7, atol=1e-7)
            funcs, funcsSens = self.c172_test_twist(DVGeo, DVCon, handler)
            handler.assert_allclose(funcs['DVCon1_monotonic_constraint_0'], -5.0*np.ones(2), 
                                    name='monotonicity_twisted', rtol=1e-7, atol=1e-7)

            funcs = dict()
            funcsSens = dict()
            # change the DVs arbitrarily
            xDV = DVGeo.getValues()
            xDV['twist'][0] = 1.0
            xDV['twist'][1] = -3.5
            xDV['twist'][2] = -2.5
            DVGeo.setDesignVars(xDV)
            # check the constraint values changed
            DVCon.evalFunctions(funcs, includeLinear=True)
            handler.root_add_dict('funcs_arb_twist', funcs, rtol=1e-6, atol=1e-6)
            # check the derivatives are still right
            DVCon.evalFunctionsSens(funcsSens, includeLinear=True)
            # regress the derivatives
            handler.root_add_dict('derivs_arb_twist', funcsSens, rtol=1e-6, atol=1e-6)
            handler.assert_allclose(funcs['DVCon1_monotonic_constraint_0'], np.array([4.5, -1.0]), 
                                    name='monotonicity_arb_twist', rtol=1e-7, atol=1e-7)
            handler.assert_allclose(funcs['DVCon1_monotonic_constraint_1'], np.array([-1.0]), 
                                    name='monotonicity_arb_twist_1', rtol=1e-7, atol=1e-7)

if __name__ == '__main__':
    unittest.main()

    #import xmlrunner
    #unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))


