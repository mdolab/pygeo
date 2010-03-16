# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time
#import psyco
#psyco.full()

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


# print 'DV Volume Example'
# grid = pyBlock.pyBlock('plot3d',file_name='dv_blocking.fmt',file_type='ascii')
# grid.doConnectivity('dv.con')
# grid.writeBvol('dv_volume.bvol',binary=True)
# grid.writePlot3d('dv_volume_new.xyz')

# grid = pyBlock.pyBlock('bvol',file_name='dv_volume.bvol',file_type='binary')
# grid.doConnectivity('dv.con')
# grid.writeTecplot('dv.dat',vols=True,orig=False,coef=True,tecio=True)


#print ' '
# print 'SGS Wing Example'
# grid = pyBlock.pyBlock('plot3d',file_name='sgs.xyz',file_type='ascii')
# grid.doConnectivity('sgs.con')
# grid.fitGlobal()
# sys.exit(0)
# grid.writeTecplot('sgs1.dat',vols=True,orig=False,coef=False,tecio=False)
# grid.writeBvol('sgs.bvol',binary=False)

# grid = pyBlock.pyBlock('bvol',file_name='sgs.bvol',file_type='ascii')
# grid.doConnectivity('sgs.con')
# grid.writeTecplot('sgs2.dat',vols=True,orig=False,coef=False,tecio=False)



print ' '
print 'DPW4 Example'
grid = pyBlock.pyBlock('plot3d',file_name='test.xyz',file_type='binary')
grid.doConnectivity('dpw.con')
grid.fitGlobal()
sys.exit(0)
grid.writeTecplot('blocks.dat',vols=True,orig=True,coef=False,tecio=True)
grid.writePlot3d('dpw_new.xyz',binary=False)
grid.writeBvol('dpw.bvol',binary=True)

#grid = pyBlock.pyBlock('bvol',file_name='dpw.bvol',file_type='binary')
#grid.doConnectivity('dpw.con')
