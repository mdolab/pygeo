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
aircraft = pyGeo.pyGeo('plot3d',file_name='double_degen.xyz')

aircraft.calcEdgeConnectivity()
timeA = time.time()
aircraft.writeTecplot('loopy.dat')
timeB =time.time()
print 'Write time is:',timeB-timeA
print 'full time',time.time()-timeA


