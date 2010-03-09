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
grid.writeTecplot('blocks.dat',vols=False,orig=True,coef=False)
