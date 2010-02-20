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

# =============================================================================
# Extension modules
# =============================================================================
from mdo_import_helper import *
exec(import_modules('pyGeo'))

# ==============================================================================
# Start of Script
# ==============================================================================

# Script to Generate a Wing Geometry
naf=3
airfoil_list = ['../../input/naca2412.dat','../../input/sd7062.dat','../../input/naca0012.dat']

chord = [1,.75,.25]
x = [0,0,0]
y = [0,0,.75]
z = [0,4,4.75]
rot_x = [0,0,-90]
rot_y = [0,0,0]
tw_aero = [0,4,0] # ie rot_z

offset = zeros((naf,2))
offset[:,0] = .25 # Offset sections by 0.25 in x

# Make the break-point vector
Nctl = 13
# Put spatial and rotations into two arrays (always the same)-------
X = zeros((naf,3))
rot = zeros((naf,3))

X[:,0] = x
X[:,1] = y
X[:,2] = z
rot[:,0] = rot_x
rot[:,1] = rot_y
rot[:,2] = tw_aero

# Create the directix spline
#curve = pySpline.curve('lms',

wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
                   scale=chord,offset=offset,Xsec=X,rot=rot,
                   Nctl=Nctl)
wing.writeTecplot('./as_wing.dat',size=0.001)
#wing.setSymmetry('xy')
#wing.calcEdgeConnectivity(1e-6,1e-6)
# #wing.writeEdgeConnectivity('as_wing.con')
# wing.readEdgeConnectivity('as_wing.con')
# wing.propagateKnotVectors()
# wing.fitSurfaces(nIter=2000,constr_tol=1e-8,opt_tol=1e-6)
# wing.writeTecplot('./as_wing.dat')
# wing.writeIGES('./as_wing.igs')

