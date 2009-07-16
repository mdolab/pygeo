#!/usr/bin/python
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross, vstack

# =============================================================================
# Extension modules
# =============================================================================
from matplotlib.pylab import *

# pySpline
sys.path.append('../pySpline/python')

#pyGeo
import pyGeo

# Wing Information

naf=8
airfoil_list = ['af15-16.inp','af15-16.inp','af15-16.inp','af15-16.inp','af15-16.inp','af15-16.inp','af15-16.inp','pinch.inp']
chord = [1.25,1,.8,.65,.65,0.65,.65,.65]
x = [1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25]
y = [0,0.1,0.2,0.4,.405,.55,.6,1.2]
z = [0,2,4,6,6.05,6.2,6.2,6.2]
rot_x = [0,0,0,0,0,-90,-90,-90]
rot_y = [0,0,0,0,0,0,0,0]
tw_aero = [-4,0,4,4.5,4.5,0,0,0] # ie rot_z
ref_axis1 = pyGeo.ref_axis(x,y,z,rot_x,rot_y,tw_aero)
offset = zeros((naf,2))
offset[:,0] = .25 #1/4 chord

# Make the break-point vector
breaks = [3,6] #zero based
Nctlv = [4,4,4] # Length breaks + 1

# Procedure for Using pyGEO

# Step 1: Run the folloiwng Commands: (Uncomment between -------)
# ---------------------------------------------------------------------
# wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis1,breaks=breaks,fit_type='lms',Nctlu = 13,Nctlv=Nctlv)
# wing.calcEdgeConnectivity(1e-2,1e-2)
# wing.writeEdgeConnectivity('wing.con') wing.writeTecplot('wing.dat')
# print 'Done Step 1' sys.exit(0)
# ----------------------------------------------------------------------
# Now: -> Load wing.dat to check connectivity information and modifiy
# wing.con file to correct any connectivity info and set
# continuity. Re-run step 1 until all connectivity information and
# continutity information is correct.

# Step 2: -> Run the following Commands (Uncomment between --------)
# After step 1 we can load connectivity information from file,
# propagate the knot vectors, stitch the edges, and then fit the
# entire surfaces with continuity constraints.  This output is then
# saved as an igs file which is the archive format storage format we
# are using for bspline surfaces

# ----------------------------------------------------------------------
# wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis1,breaks=breaks,fit_type='lms',Nctlu = 13,Nctlv=Nctlv)
# wing.readEdgeConnectivity('wing.con')
# wing.propagateKnotVectors()
# wing.stitchEdges()
# #wing.fitSurfaces()
# wing.writeTecplot('wing.dat')
# wing.writeIGES('wing.igs')
# print 'Done Step 2'
# sys.exit(0)
# ----------------------------------------------------------------------

# Step 3: -> After step 2 we now have two files we need, the stored
# iges file as well as the connectivity file. The IGES file which has
# been generated can be used to generate a 3D CFD mesh in ICEM.  Now
# to load a geometry for an optimization run we simply load the IGES
# file as well as the connectivity file and we are good to go.

# ----------------------------------------------------------------------

wing = pyGeo.pyGeo('iges',file_name='wing.igs')
wing.readEdgeConnectivity('wing.con')
wing.stitchEdges() # Just to be sure
print 'Done Step 3'

# ----------------------------------------------------------------------

# Step 4: Now the rest of the code is up to the user. The user only
# needs to run the commands in part 3 to fully define the geometry of
# interest

print 'Attaching Ref Axis...'
wing.setRefAxis([0,1,2,3,4,5],ref_axis1)

wing.writeTecplot('wing.dat',ref_axis1)

print 'modifiying ref axis:'

ref_axis1.x[:,2] *= 1.2
ref_axis1.x[:,0] += 3*ref_axis1.xs.s
ref_axis1.x[:,1] += 0.4*ref_axis1.xs.s**2

wing.update(ref_axis1)
wing.stitchEdges() # Just to be sure
wing.writeTecplot('wing2.dat',ref_axis1)

