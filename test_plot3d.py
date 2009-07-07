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
#aircraft = pyGeo.pyGeo('iges',file_name='sailplane_split.igs')
#aircraft.nPatch = 10
aircraft.calcEdgeConnectivity(1e-1,1e-1)
aircraft.writeEdgeConnectivity('edge.con')
aircraft.writeTecplot('full_aircraft.dat')

print 'full time',time.time()-timeA

for i in xrange(aircraft.nPatch):
    print aircraft.surfs[i].edge_con,aircraft.surfs[i].master_edge
