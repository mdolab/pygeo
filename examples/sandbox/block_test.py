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
exec(import_modules('pyGeo','pySpline'))

grid = pyGeo.pyBlock('plot3d',file_name='test.xyz',file_type='binary')
grid.doConnectivity()
#grid.writeTecplot('blocks.dat',tecio=False)
