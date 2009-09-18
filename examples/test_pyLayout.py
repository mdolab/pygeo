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

#cfd-csm pre (Optional)
sys.path.append('../../../pyHF/pycfd-csm/python/')

#pyGeo
sys.path.append('../')
import pyGeo

#pyLayout
sys.path.append('../../pyLayout/')
import pyLayout

#Design Variable Functions
from dv_funcs import *

# ==============================================================================
# Start of Script
# ==============================================================================


# Wing Information - Create a Geometry Object from cross sections

naf=3
airfoil_list = ['../input/naca0012.dat','../input/naca0012.dat',
                '../input/pinch_xfoil.dat']
chord = [1,1,.50]
x = [0,.1,0]
y = [0,0,0]
z = [0,3.94,4]
rot_x = [0,0,0]
rot_y = [0,0,0]
tw_aero = [0,0,0] # ie rot_z

offset = zeros((naf,2))
offset[:,0] = .25 # Offset sections by 0.25 in x
offset[-1,0] = 0
# Make the break-point vector
breaks = [1] #zero based (Must NOT contain 0 or index of last value)
cont = [1] # vector of length breaks: 0 for c0 continuity 1 for c1 continutiy
nsections = [10,10]# Length breaks + 1
section_spacing = [linspace(0,1,10),linspace(0,1,10)]
Nctlu = 9
end_type = 'pinch'
# 'pinch' or 'flat' or 'rounded' -> flat and rounded result in a
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
#Note: u direction is chordwise, v direction is span-wise
# wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,\
#                        file_type='xfoil',scale=chord,offset=offset, \
#                        Xsec=X,rot=rot,breaks=breaks,cont=cont,end_type=end_type,\
#                        nsections=nsections,fit_type='lms', Nctlu=Nctlu,Nfoil=45)

# wing.calcEdgeConnectivity(1e-6,1e-6)
# wing.propagateKnotVectors()
# wing.writeTecplot('../output/wing.dat')
# wing.writeIGES('../input/wing.igs')
# ------------------------------------------------------------------

# Load in the split plot3d file
# wing = pyGeo.pyGeo('plot3d',file_name='../input/wing.xyz.fmt')
# wing.calcEdgeConnectivity(1e-6,1e-6)
# wing.writeEdgeConnectivity('wing_split.con')
# wing.propagateKnotVectors()
# wing.writeIGES('../input/wing_split.igs')

wing = pyGeo.pyGeo('iges',file_name='../input/wing_split.igs')
wing.readEdgeConnectivity('wing_split.con')
wing.writeTecplot('../output/wing.dat',
                  labels=True,ref_axis=True,directions=True)


# Create the empty pyLayout Object

MAX_SPARS = 3
Nsection = 1
wing_box = pyLayout.Layout(wing,Nsection,MAX_SPARS)

# ---------- Create the First Domain -------------

MAX_RIBS = 10
le_list = array([[0,0,0],[0,0,3.94]])
te_list = array([[.60,0,0],[.6,0,3.94]])

domain = pyLayout.domain(le_list,te_list)

# ---------- OPTIONAL SPECIFIC RIB DISTIRBUTION -----------
rib_pos = zeros((MAX_RIBS,3))
spline = pySpline.linear_spline(task='interpolate',k=2,X=X[0:2])
rib_pos = spline.getValueV(linspace(0,1,MAX_RIBS))

rib_dir = zeros((MAX_RIBS,3))
rib_dir[:] = [1,0,0]
rib_dir[6] = [1,.25,0]
# -----------------------------------------------------------

rib_blank = ones(MAX_RIBS)
spar_blank = ones(MAX_SPARS)
rib_blank[5] = 0
surfs = [1,0,2,3]
spar_con = [0,-1,1]

def1 = pyLayout.struct_def(MAX_RIBS,MAX_SPARS,domain,surfs,spar_con,
                           rib_blank=rib_blank,rib_pos=rib_pos,rib_dir=rib_dir,
                           spar_blank=spar_blank)
                           

wing_box.addSection(def1)
wing_box.finalize()
