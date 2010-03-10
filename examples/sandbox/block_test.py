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

#grid = pyGeo.pyBlock('plot3d',file_name='dv_blocking.fmt',file_type='ascii')

#for ivol in xrange(grid.nVol):
#    print grid.vols[ivol].Nu,grid.vols[ivol].Nv,grid.vols[ivol].Nw

# pts = []
# for ivol in xrange(grid.nVol):
#     for i in xrange(grid.vols[ivol].Nu):
#         for j in xrange(grid.vols[ivol].Nv):
#             for k in xrange(grid.vols[ivol].Nw):
#                 pts.append(grid.vols[ivol].X[i,j,k])


# un,link = pointReduce(pts)
# print 'Unique Nodes:',len(un)


grid = pyGeo.pyBlock('plot3d',file_name='test.xyz',file_type='binary')
grid.doConnectivity()
#grid.writeTecplot('blocks.dat',vols=False,orig=True,coef=False)
