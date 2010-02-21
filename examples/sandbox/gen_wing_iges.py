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

#cfd-csm pre (Optional)
sys.path.append('../../pyHF/pycfd-csm/python/')

#pyGeo
sys.path.append('../')
import pyGeo2 as pyGeo

# ==============================================================================
# Start of Script
# ==============================================================================

# Wing Information - Create a Geometry Object from cross sections

naf = 2
airfoil_list = ['naca0012.dat','naca0012.dat']
chord = [1,0.4]
x = [0,.75]
y = [0,.1]
z = [0,4]
rot_x = [0,0]
rot_y = [0,0]
tw_aero = [0,-2] # ie rot_z

offset = zeros((naf,2))
offset[:,0] = .25 # Offset sections by 0.25 in x
offset[-1,0] = 0
# Make the break-point vector
Nctlu = 9
#  a new surface on the end 

# Put spatial and rotations into two arrays (always the same)-------
X = zeros((naf,3))
rot = zeros((naf,3))
X[:,0] = x
X[:,1] = y
X[:,2] = z
rot[:,0] = rot_x
rot[:,1] = rot_y
rot[:,2] = tw_aero
# ------------------------------------------------------------------
    
# Procedure for Using pyGEO

# Step 1: Run the folloiwng Commands: (Uncomment between -------)
# ---------------------------------------------------------------------
#Note: u direction is chordwise, v direction is span-wise
wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
                   file_type='xfoil',scale=chord,offset=offset, 
                   Xsec=X,rot=rot,fit_type='lms',Nctlu=Nctlu,Nfoil=45)

wing.calcEdgeConnectivity(1e-6,1e-6)
wing.propagateKnotVectors()
wing.update()
wing.writeIGES('geometry/wing.igs')

print 'Done'
