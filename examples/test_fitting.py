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

# pyOpt
sys.path.append('../../../pyACDT/pyACDT/Optimization/pyOpt')

# pySnopt
sys.path.append('../../../pyACDT/pyACDT/Optimization/pyOpt/pySNOPT')

#pyGeo
sys.path.append('../')
import pyGeo_NM as pyGeo

# ==============================================================================
# Start of Script
# ==============================================================================

# Wing Information - Create a Geometry Object from cross sections

naf=2
Nctlu = 13
airfoil_list = ['../input/naca2412.dat','../input/naca2412.dat']
airfoil_list = ['../input/af15-16.inp','../input/af15-16.inp']
chord = [1,1]
x = [0,0]
y = [0,0]
z = [0,4]
rot_x = [0,0]
rot_y = [0,0]
rot_z = [0,0]

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
                   file_type='precomp',scale=chord,offset=offset, 
                   nsections=nsections, Xsec=X,rot=rot,
                   fit_type='lms',Nctlu=Nctlu,Nfoil=45)

wing.readEdgeConnectivity('wing_fit_test.con')
wing.printEdgeConnectivity()
wing.propagateKnotVectors()
timeA = time.time()
wing.fitSurfaces2()
wing.update()
timeB = time.time()
wing.writeTecplot('../output/wing_fit_test.dat',orig=True)

print 'Done Step 2'
print 'fit time:',timeB-timeA
