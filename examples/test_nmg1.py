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
import pyGeo

#geo_utils
from geo_utils import *

# ==============================================================================
# Start of Script
# ==============================================================================

#Load in the plot3D file to test:
#geo = pyGeo.pyGeo('plot3d',file_name='../input/nmg1.xyz.fmt')
#geo = pyGeo.pyGeo('plot3d',file_name='../input/loopy.xyz')
geo = pyGeo.pyGeo('plot3d',file_name='../input/full_aircraft.xyz')
# del geo.surfs[1]
# del geo.surfs[2]
# del geo.surfs[2]
# del geo.surfs[2]
# del geo.surfs[2]
# del geo.surfs[2]
# del geo.surfs[2]
# del geo.surfs[2]
# del geo.surfs[2]
# del geo.surfs[-2]
# geo.nSurf = 4

#geo.calcEdgeConnectivity()
#geo.printEdgeConnectivity()
#geo.writeEdgeConnectivity('nmg1.con')

geo.readEdgeConnectivity('long-ez.con')
geo.propagateKnotVectors()

geo.fitSurfaces()
geo.update()
# for i in xrange(len(geo.l_index)):
#     print geo.l_index[i]
geo.writeTecplot('../output/npg1.dat',orig=True,nodes=False)
print ' Done!'
