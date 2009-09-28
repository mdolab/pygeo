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
sys.path.append('../../pySpline/python')

#pyGeo
sys.path.append('../')
import pyGeo

# This script reads a surfaced-based plot3d file as typically
# outputted by aerosurf. It then creates a b-spline surfaces for each
# surface patch.
timeA = time.time()
aircraft = pyGeo.pyGeo('plot3d',file_name='../input/full_aircraft.xyz')

aircraft.calcEdgeConnectivity(1e-6,1e-6)
timeB = time.time()
print 'Edge Calc Time:',timeB-timeA
#aircraft.writeEdgeConnectivity('aircraft.con')
#aircraft.readEdgeConnectivity('aircraft.con')
sys.exit(0)
aircraft.propagateKnotVectors()
aircraft.update()
aircraft.fitSurfaces()
aircraft.update()
timeA = time.time()
aircraft.writeTecplot('full_aircraft.dat')
timeB =time.time()
print 'Write time is:',timeB-timeA
print 'full time',time.time()-timeA
