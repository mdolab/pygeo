# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross

# =============================================================================
# Extension modules
# =============================================================================

# pySpline 
sys.path.append('../pySpline/python')

#cfd-csm pre (Optional)
sys.path.append('../../pyHF/pycfd-csm/python/')

#pyGeo
import pyGeo2 as pyGeo

# This script reads a surfaced-based plot3d file as typically
# outputted by aerosurf. It then creates a b-spline surfaces for each
# surface patch.
timeA = time.time()
aircraft = pyGeo.pyGeo('plot3d',file_name='full_aircraft.xyz')
#aircraft = pyGeo.pyGeo('plot3d',file_name='test.xyz')
#aircraft = pyGeo.pyGeo('iges',file_name='sailplane_split.igs')
# #del aircraft.surfs[0]
# #del aircraft.surfs[0]
# #del aircraft.surfs[0]
# #del aircraft.surfs[0]
# del aircraft.surfs[8]
# del aircraft.surfs[8]
# del aircraft.surfs[8]
# #del aircraft.surfs[4]
# #del aircraft.surfs[4]
# #del aircraft.surfs[4]
# #del aircraft.surfs[4]
# #del aircraft.surfs[4]
# # del aircraft.surfs[4]
# # del aircraft.surfs[4]
# # del aircraft.surfs[5]
#del aircraft.surfs[1]
#del aircraft.surfs[2]
#del aircraft.surfs[4]
#del aircraft.surfs[0]
#del aircraft.surfs[0]
#del aircraft.surfs[0]
#del aircraft.surfs[0]
del aircraft.surfs[8]
del aircraft.surfs[8]
del aircraft.surfs[8]

aircraft.nSurf = 20

aircraft.calcEdgeConnectivity(1e-1,2e-1)
aircraft.writeEdgeConnectivity('aircraft.con')
aircraft.propagateKnotVectors()
#aircraft.fitSurfaces()
timeA = time.time()
aircraft.writeTecplot('full_aircraft.dat')
timeB =time.time()
print 'Write time is:',timeB-timeA
print 'full time',time.time()-timeA


