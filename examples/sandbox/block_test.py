# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time
 
# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross,shape,alltrue

# =============================================================================
# Extension modules
# =============================================================================

from mdo_import_helper import *
exec(import_modules('pyGeo','pySpline','geo_utils'))


grid = pyGeo.pyBlock('plot3d',file_name='dv_blocking.fmt',file_type='ascii')
grid.doConnectivity()

grid.writeTecplot('blocks.dat',vols=True,orig=False,coef=False)

print len(grid.topo.g_index)

N = grid.topo.l_index[3][0,-1,-1]

grid.coef[N] += [0,0,20]
grid._updateVolumeCoef()
grid.writeTecplot('blocks_mod.dat',vols=True,orig=False,coef=False)

#grid = pyGeo.pyBlock('plot3d',file_name='test.xyz',file_type='binary')

#grid.writeFEAPCorners('corners.in')

