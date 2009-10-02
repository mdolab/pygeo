#!/usr/bin/python
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, zeros, ones, array

import petsc4py
petsc4py.init(sys.argv)

# =============================================================================
# Extension modules
# =============================================================================

# pySpline 
sys.path.append('../../pySpline/python')

#pyGeo
sys.path.append('../')
import pyGeo_NM as pyGeo

# ==============================================================================
# Start of Script
# ==============================================================================

# Wing Information - Create a Geometry Object from cross sections

naf=2
Nctlu = 13
airfoil_list = ['../input/naca0012.dat','../input/naca0012.dat']
chord = [1,.5]
x = [0,.25]
y = [0,0]
z = [0,4]
rot_x = [0,0]
rot_y = [0,30]
rot_z = [0,-15]

offset = zeros((naf,2))
nsections = [5]
# Make the break-point vector

                               
# Put spatial and rotations into two arrays (always the same)-------
X = zeros((naf,3))
rot = zeros((naf,3))

X[:,0] = x
X[:,1] = y
X[:,2] = z
rot[:,0] = rot_x
rot[:,1] = rot_y
rot[:,2] = rot_z
# ------------------------------------------------------------------
    
# Procedure for Using pyGEO

# Step 1: Run the folloiwng Commands: (Uncomment between -------)
# ---------------------------------------------------------------------
#Note: u direction is chordwise, v direction is span-wise
# wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
#                    file_type='xfoil',scale=chord,offset=offset, 
#                    nsections=nsections, Xsec=X,rot=rot,
#                    fit_type='lms',Nctlu=Nctlu,Nfoil=45)

# wing.calcEdgeConnectivity(1e-6,1e-6)
# wing.writeEdgeConnectivity('wing_fit_test.con')
# wing.printEdgeConnectivity()
# wing.propagateKnotVectors()
# wing.writeTecplot('../output/wing_fit_test.dat')
# wing.writeIGES('../input/wing_fit_test.igs')
# print 'Done Step 1'

# Step 2: Run the folloiwng Commands: (Uncomment between -------)
# ---------------------------------------------------------------------

wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
                   file_type='xfoil',scale=chord,offset=offset, 
                   nsections=nsections, Xsec=X,rot=rot,
                   fit_type='lms',Nctlu=Nctlu,Nfoil=45)

wing.readEdgeConnectivity('wing_fit_test.con')
wing.printEdgeConnectivity()
wing.propagateKnotVectors()

wing.fitSurfaces()
wing.update()
wing.writeTecplot('../output/wing_fit_test.dat',size=.006,orig=True)

print 'Done Step 2'
