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
exec(import_modules('pyBlock','pySpline','geo_utils'))

# DV_Volume Examples
grid = pyBlock.pyBlock('plot3d',file_name='dv_blocking.fmt',file_type='ascii')
grid.doConnectivity('grid.con')
grid.writeBvol('dv_volume.bvol',binary=True)
grid = pyBlock.pyBlock('bvol',file_name='dv_volume.bvol',file_type='binary')
grid.writeTecplot('blocks.dat',vols=True,orig=False,coef=True)
sys.exit(0)

# SGS Wing Case
# grid = pyBlock.pyBlock('plot3d',file_name='sgs.xyz',file_type='binary')
# grid.doConnectivity('grid.con')
# grid.writeBvol('sgs.bvol',binary=True)
# print len(grid.topo.g_index)

# grid = pyBlock.pyBlock('bvol',file_name='sgs.bvol',file_type='binary')
# grid.writeTecplot('blocks.dat',vols=True,orig=False,coef=True)

# DPW4 Case
#grid = pyBlock.pyBlock('plot3d',file_name='test.xyz',file_type='binary')
#grid.doConnectivity('grid.con')
#grid.writeBvol('dpw.bvol',binary=True)

grid = pyBlock.pyBlock('bvol',file_name='dpw.bvol',file_type='binary')
grid.writeTecplot('blocks.dat',vols=True,orig=False,coef=True,tecio=True)



# print len(grid.topo.g_index)

# N = grid.topo.l_index[3][0,-1,-1]

# grid.coef[N] += [0,0,20]
# grid._updateVolumeCoef()
# grid.writeTecplot('blocks_mod.dat',vols=True,orig=False,coef=False)

# #grid = pyGeo.pyBlock('plot3d',file_name='test.xyz',file_type='binary')

# #grid.writeFEAPCorners('corners.in')

