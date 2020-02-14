from __future__ import print_function
import os
import unittest
import numpy
import copy
from baseclasses import BaseRegTest
import commonUtils
from pygeo import DVGeometry, geo_utils

class RegTestPyGeo(unittest.TestCase):

    N_PROCS = 1

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def make_cube_ffd(self, file_name, x0, y0, z0, dx, dy, dz):
        # Write cube ffd with i along x-axis, j along y-axis, and k along z-axis
        axes = ['k', 'j', 'i']
        slices = numpy.array(
            # Slice 1
            [[[[x0, y0, z0], [x0+dx, y0, z0]],
            [[x0, y0+dy, z0], [x0+dx, y0+dy, z0]]],
            # Slice 2
            [[[x0, y0, z0+dz], [x0+dx, y0, z0+dz]],
            [[x0, y0+dy, z0+dz], [x0+dx, y0+dy, z0+dz]]]],
            dtype='d'
        )

        N0 = [2]
        N1 = [2]
        N2 = [2]

        geo_utils.write_wing_FFD_file(file_name, slices, N0, N1, N2, axes=axes)

    def setup_blocks(self):
        # Make tiny FFD
        ffd_name = os.path.join(self.base_path,'../inputFiles/tiny_cube.xyz')
        self.make_cube_ffd(ffd_name, 1, 1, 1, 1, 1, 1)
        tiny = DVGeometry(ffd_name, child=True)
        tiny.addRefAxis('y', xFraction=0.5, alignIndex='j')

        # Make tiny FFD
        ffd_name = os.path.join(self.base_path,'../inputFiles/small_cube.xyz')
        self.make_cube_ffd(ffd_name, 0, 0, 0, 2, 2, 2)
        small = DVGeometry(ffd_name, child=True)
        small.addRefAxis('y', xFraction=0.5, alignIndex='j')

        # Make big FFD
        ffd_name = os.path.join(self.base_path,'../inputFiles/big_cube.xyz')
        self.make_cube_ffd(ffd_name, 0, 0, 0, 3, 3, 3)
        big = DVGeometry(ffd_name)
        big.addRefAxis('x', xFraction=0.5, alignIndex='i')
        big.addChild(small)
        small.addChild(tiny)

        # Add point set
        points = numpy.array([
                # [0.5, 0.5, 0.5],
                [1.25, 1.25, 1.25],
                # [1.5, 1.5, 1.5]
                ])
        big.addPointSet(points, 'X')

        # Add translation design variables
        def translate(val, geo):
            C = geo.extractCoef('y')
            for i in range(len(val)):
                C[:,i] += val[i]
            geo.restoreCoef(C, 'y')

        tiny.addGeoDVGlobal('moveTiny', [0]*3, translate)
        small.addGeoDVGlobal('moveSmall', [0]*3, translate)

        # Add rotation design variables
        def rotate_y(val, geo):
            geo.rot_y['y'].coef[:] = val[0]

        def rotate_x(val, geo):
            geo.rot_x['x'].coef[:] = val[0]

        tiny.addGeoDVGlobal('rotTiny', 0, rotate_y)
        small.addGeoDVGlobal('rotSmall', 0, rotate_y)
        big.addGeoDVGlobal('rotBig', 0, rotate_x)

        # Add shape variables
        small.addGeoDVLocal('smallLocal', axis='x')
        tiny.addGeoDVSectionLocal('tinySectionLocal', secIndex='j', axis=1,
                                  orient0='i', orient2='ffd')

        return big, small, tiny

    def compute_values(self, DVGeo, handler, refDeriv):
        # Calculate updated point coordinates
        Xnew = DVGeo.update('X')
        handler.root_add_val(Xnew, 1e-12, 1e-12, msg='Updated points')

        # Need to get design variables so that we can reset the Jacobians
        # for each call
        x = DVGeo.getValues()

        # Calculate Jacobians
        DVGeo.setDesignVars(x)
        DVGeo.computeTotalJacobian('X')
        Jac = DVGeo.JT['X'].toarray()

        DVGeo.setDesignVars(x)
        DVGeo.computeTotalJacobianCS('X')
        JacCS = DVGeo.JT['X']

        DVGeo.setDesignVars(x)
        DVGeo.computeTotalJacobianFD('X')
        JacFD = DVGeo.JT['X']

        if refDeriv:
            handler.root_add_val(JacCS, 1e-12, 1e-12, msg='Check jacobian')
        else:
            handler.root_add_val(Jac, 1e-12, 1e-12, msg='Check jacobian')

        # print(Jac-JacCS)
        # print(Jac-JacFD)
        # print('\n', Jac)
        # print('\n', JacCS)
        # print('\n', JacFD)

        # Test that they are equal to eachother
        numpy.testing.assert_allclose(Jac, JacCS, rtol=1e-12, atol=1e-12,
                                        err_msg='Analytic vs complex-step')
        numpy.testing.assert_allclose(Jac, JacFD, rtol=1e-6, atol=1e-6,
                                        err_msg='Analytic vs finite difference')

        # Create dIdPt with one function for each point coordinate
        Npt = 1
        dIdPt = numpy.zeros([Npt*3, Npt,3])
        for i in range(Npt):
            dIdPt[i*3:(i+1)*3,i] = numpy.eye(3)

        # sens = commonUtils.totalSensitivityFD(DVGeo, Npt*3, 'X', step=1e-6)
        # print(sens)
        # exit()

        # Test sensitivity dictionaries
        if refDeriv:
            # Generate reference from finite differences
            sens = commonUtils.totalSensitivityFD(DVGeo, Npt*3, 'X', step=1e-6)
            handler.root_add_dict(sens, 1e-6, 1e-6, msg='Check sens dict')
        else:
            # Compute the analytic derivatives
            sens = DVGeo.totalSensitivity(dIdPt, 'X')
            handler.root_add_dict(sens, 1e-6, 1e-6, msg='Check sens dict')

    def train_1(self, train=True, refDeriv=True):
        self.test_1(train=train, refDeriv=refDeriv)

    def test_1(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path,'ref/test_Blocks_01.ref')

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 1")

            big, small, tiny = self.setup_blocks()

            # Rotate small cube by 45 and then rotate tiny cube by -45
            x = {}
            x['rotSmall'] = 45
            x['rotTiny'] = -45
            big.setDesignVars(x)

            # Compute tests
            self.compute_values(big, handler, refDeriv)

    def train_2(self, train=True, refDeriv=True):
        self.test_2(train=train, refDeriv=refDeriv)

    def test_2(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path,'ref/test_Blocks_02.ref')

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 2")

            big, small, tiny = self.setup_blocks()

            # Modify design variables
            x = {}
            x['rotBig'] = 10
            x['moveSmall'] = [0.5, -1, 0.7]
            x['rotTiny'] = -20
            big.setDesignVars(x)

            # Compute tests
            self.compute_values(big, handler, refDeriv)

    def train_3(self, train=True, refDeriv=True):
        self.test_3(train=train, refDeriv=refDeriv)

    def test_3(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path,'ref/test_Blocks_03.ref')

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 3")

            big, small, tiny = self.setup_blocks()

            # Modify design variables
            x = big.getValues()
            x['rotSmall'] = 10
            numpy.random.seed(11)
            x['tinySectionLocal'] = numpy.random.random(*x['tinySectionLocal'].shape)
            big.setDesignVars(x)

            # Compute tests
            self.compute_values(big, handler, refDeriv)