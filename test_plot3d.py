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

#pyOPT
sys.path.append(os.path.abspath('../../../../pyACDT/pyACDT/Optimization/pyOpt/'))

# pySpline
sys.path.append('../pySpline/python')

#pyOPT
sys.path.append(os.path.abspath('../../pyACDT/pyACDT/Optimization/pyOpt/'))

#pySNOPT
sys.path.append(os.path.abspath('../../pyACDT/pyACDT/Optimization/pyOpt/pySNOPT'))

#pyGeo
import pyGeo

# This script reads a surfaced-based plot3d file as typically
# outputted by aerosurf. It then creates a b-spline surfaces for each
# surface patch.
timeA = time.time()
aircraft = pyGeo.pyGeo('plot3d',file_name='full_aircraft.xyz')
aircraft = pyGeo.pyGeo('plot3d',file_name='test.xyz')
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
# # del aircraft.surfs[8]
# # del aircraft.surfs[8]
# # del aircraft.surfs[8]

#aircraft.nPatch = 17

aircraft.calcEdgeConnectivity(1e-1,2e-1)
aircraft.writeEdgeConnectivity('aircraft.con')

aircraft.writeTecplot('full_aircraft.dat')

print 'full time',time.time()-timeA

