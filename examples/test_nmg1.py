#!/usr/bin/python
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross, vstack, arctan2, tan

import petsc4py
petsc4py.init(sys.argv)

# =============================================================================
# Extension modules
# =============================================================================                     

# pySpline 
sys.path.append('../../pySpline/python')
import pySpline

#pyGeo
sys.path.append('../')
import pyGeo_NM as pyGeo

#geo_utils
from geo_utils import *

# ==============================================================================
# Start of Script
# ==============================================================================

#Load in the plot3D file to test:
#eo = pyGeo.pyGeo('plot3d',file_name='../input/nmg1.xyz.fmt')
#geo = pyGeo.pyGeo('plot3d',file_name='../input/loopy.xyz')
geo = pyGeo.pyGeo('plot3d',file_name='../input/full_aircraft.xyz',no_print=True)

timeA = time.time()
geo.calcEdgeConnectivity()
timeB = time.time()
print 'Edge Calc Time:',timeB-timeA

geo.printEdgeConnectivity()
geo.writeEdgeConnectivity('nmg1.con')
geo.readEdgeConnectivity('nmg1.con')
geo.printEdgeConnectivity()

for i in xrange(len(geo.l_index)):
    print geo.l_index[i]
geo.writeTecplot('../output/npg1.dat',orig=True)
print ' Done!'
