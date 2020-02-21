
from __future__ import print_function
import os
import unittest
import numpy
import copy
from baseclasses import BaseRegTest
from pygeo import DVGeometry, geo_utils


class RegTestPyGeo(unittest.TestCase):

    N_PROCS = 1

    # def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        # self.base_path = os.path.dirname(os.path.abspath(__file__))


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



    def test_parent_shape_child_rot(self, train=False, refDeriv=False):
        ffd_name = '../../tests/inputFiles/small_cube.xyz'
        self.make_cube_ffd(ffd_name, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8)
        small = DVGeometry(ffd_name, child=True)
        small.addRefAxis('ref', xFraction=0.5, alignIndex='j')
        
        
        x0 = 0.0
        y0 = 0.0 
        z0 = 0.0
        dx = 1.0
        dy = 1.0 
        dz = 1.0
        
        axes = ['k', 'j', 'i']
        slices = numpy.array(
            # Slice 1
            [[[ [x0, y0, z0], [x0+dx, y0, z0],    [x0+2*dx, y0, z0] ],
            [[x0, y0+dy, z0], [x0+dx, y0+dy, z0], [x0+2*dx, y0+dy, z0]]],
            # Slice 2
            [[[x0, y0, z0+dz], [x0+dx, y0, z0+dz], [x0+2*dx, y0, z0+dz]],
            [[x0, y0+dy, z0+dz], [x0+dx, y0+dy, z0+dz], [x0+2*dx, y0+dy, z0+dz]]]],
            dtype='d'
        )

        N0 = [2]
        N1 = [2]
        N2 = [3]
        ffd_name = '../../tests/inputFiles/big_cube.xyz'

        geo_utils.write_wing_FFD_file(ffd_name, slices, N0, N1, N2, axes=axes)
        big = DVGeometry(ffd_name)
        big.addRefAxis('ref', xFraction=0.5, alignIndex='j')
        big.addChild(small)

        # Add point set
        points = numpy.array([[0.5, 0.5, 0.5]])
        big.addPointSet(points, 'X')

        # Add only translation variables
        add_vars(big, 'big', local='z', rotate='y')
        add_vars(small, 'small', rotate='y')


        ang = 45
        ang_r = numpy.deg2rad(ang)

        # Modify design variables
        x = big.getValues()

        # add a local shape change
        x['local_z_big'] = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        big.setDesignVars(x)
        Xs = big.update('X')

        # add a rotation of the child FFD
        x['rotate_y_small'] = ang
        big.setDesignVars(x)
        Xrot_ffd = big.update('X')


        # the modification caused by the child FFD should be the same as rotating the deformed point of the parent
        # (you would think)
        rot_mat = numpy.array([[numpy.cos(ang_r) , 0, numpy.sin(ang_r) ],
                                [0, 1, 0],
                                [-numpy.sin(ang_r), 0, numpy.cos(ang_r)  ]])
        Xrot = numpy.dot(rot_mat,( Xs - points) .T) + points.T


        numpy.testing.assert_array_almost_equal(Xrot_ffd.T, Xrot)

'''
The following are some helper functions for setting up the design variables for
the different test cases.
'''
def add_vars(geo, name, translate=False, rotate=None, scale=None, local=None, slocal=False):

    if translate:
        dvName = 'translate_{}'.format(name)
        geo.addGeoDVGlobal(dvName=dvName, value=[0]*3, func=f_translate)

    if rotate is not None:
        rot_funcs = {
            'x':f_rotate_x,
            'y':f_rotate_y,
            'z':f_rotate_z,
            'theta':f_rotate_theta
        }
        assert(rotate in rot_funcs.keys())

        dvName = 'rotate_{}_{}'.format(rotate, name)
        dvFunc = rot_funcs[rotate]
        geo.addGeoDVGlobal(dvName=dvName, value=0, func=dvFunc)

    if local is not None:
        assert(local in ['x', 'y', 'z'])
        dvName = 'local_{}_{}'.format(local, name)
        geo.addGeoDVLocal(dvName, axis=local)

    if slocal:
        dvName = 'sectionlocal_{}'.format(name)
        geo.addGeoDVSectionLocal(dvName, secIndex='j', axis=1, orient0='i', orient2='ffd')

def f_translate(val, geo):
    C = geo.extractCoef('ref')
    for i in range(len(val)):
        C[:,i] += val[i]
    geo.restoreCoef(C, 'ref')

def f_rotate_x(val, geo):
    geo.rot_x['ref'].coef[:] = val[0]

def f_rotate_y(val, geo):
    geo.rot_y['ref'].coef[:] = val[0]

def f_rotate_z(val, geo):
    geo.rot_z['ref'].coef[:] = val[0]

def f_rotate_theta(val, geo):
    geo.rot_theta['ref'].coef[:] = val[0]
            

if __name__ == '__main__':
    unittest.main()
