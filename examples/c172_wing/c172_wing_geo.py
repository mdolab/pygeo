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
sys.path.append('../../../pySpline/python')

#cfd-csm pre (Optional)
sys.path.append('../../../../pyHF/pycfd-csm/python/')

#pyGeo
sys.path.append('../../')

#pyLayout
sys.path.append('../../../pyLayout/')

# pyOpt
sys.path.append('../../../../pyACDT/pyACDT/Optimization/pyOpt')

# pySnopt
sys.path.append('../../../../pyACDT/pyACDT/Optimization/pyOpt/pySNOPT')

import pyGeo
import pyLayout

# ==============================================================================
# Start of Script
# ==============================================================================

# Script to Generate a Wing Geometry

naf=3
airfoil_list = ['../../input/naca2412.dat','../../input/naca2412.dat','../../input/naca2412.dat']

chord = [1.67,1.67,1.18]
x = [0,0,.125*1.18]
y = [0,0,0]
z = [0,2.5,10.58/2]
rot_x = [0,0,0]
rot_y = [0,0,0]
tw_aero = [0,0,0] # ie rot_z

offset = zeros((naf,2))

# Make the break-point vector
breaks = [1]
nsections = [4,4]# Length breaks + 1
Nctlu = 13
end_type = 'rounded'
                               
# Put spatial and rotations into two arrays (always the same)-------
X = zeros((naf,3))
rot = zeros((naf,3))

X[:,0] = x
X[:,1] = y
X[:,2] = z
rot[:,0] = rot_x
rot[:,1] = rot_y
rot[:,2] = tw_aero

wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,\
                   file_type='xfoil',scale=chord,offset=offset, \
                   Xsec=X,rot=rot,end_type=end_type, breaks=breaks,end_scale=1.05,
                   nsections=nsections,fit_type='lms', Nctlu=Nctlu,Nfoil=45)
wing.setSymmetry('xy')
wing.calcEdgeConnectivity(1e-6,1e-6)
#wing.writeEdgeConnectivity('c172.con')
wing.readEdgeConnectivity('c172.con')
wing.propagateKnotVectors()
wing.fitSurfaces(nIter=2000,constr_tol=1e-8,opt_tol=1e-6)
wing.writeTecplot('./c172_geo.dat')
wing.writeIGES('./c172.igs')

