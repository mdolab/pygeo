#!/usr/bin/python
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross, vstack, arctan2, tan, loadtxt,\
    lexsort,savetxt,append

import petsc4py
petsc4py.init(sys.argv)

# =============================================================================
# Extension modules
# =============================================================================

# pySpline 
sys.path.append('../../../pySpline/python')
import pySpline

#cfd-csm pre (Optional)
sys.path.append('../../../../pyHF/pycfd-csm/python/')

#pyGeo
sys.path.append('../../')
import pyGeo_NM as pyGeo


#pyLayout
sys.path.append('../../../pyLayout/')
import pyLayout_NM as pyLayout

#Design Variable Functions
sys.path.append('../')
from dv_funcs import *
# ==============================================================================
# Start of Script
# ==============================================================================

# Wing Information - Create a Geometry Object from cross sections

naf=22
n0012 = '../../input/naca0012.dat'

# Use the digitize it data for the planform:
le = array(loadtxt('bwb_le.out'))
te = array(loadtxt('bwb_te.out'))
front_up = array(loadtxt('bwb_front_up.out'))
front_low = array(loadtxt('bwb_front_low.out'))

# Now make a ONE-DIMENSIONAL spline for each of the le and trailing edes
le_spline = pySpline.linear_spline(task='lms',k=4,X=le[:,1],s=le[:,0],Nctl=20)
te_spline = pySpline.linear_spline(task='lms',k=4,X=te[:,1],s=te[:,0],Nctl=20)
up_spline = pySpline.linear_spline(task='lms',k=4,X=front_up[:,1],s=front_up[:,0],Nctl=20)
low_spline = pySpline.linear_spline(task='lms',k=4,X=front_low[:,1],s=front_low[:,0],Nctl=20)

# Generate consistent equally spaced spline data
span = linspace(0,138,naf-2)
le = le_spline.getValueV(span)
te = te_spline.getValueV(span)
up = up_spline.getValueV(span)
low = low_spline.getValueV(span)

chord = te-le
x = le
z = span
mid_y = (up[0]+low[0])/2.0
y = -(up+low)/2 + mid_y

# This will be usedful later as the mid surface value
mid_y_spline = pySpline.linear_spline(task='lms',k=4,X=y,s=span,Nctl=20)


# Estimate t/c ratio
tc = (low-up)/chord # Array division
points0 = loadtxt(n0012)
airfoil_list = []
points = zeros(points0.shape)
points[:,0] = points0[:,0]
for i in xrange(naf-2):
    scale_y = tc/.12
    points[:,1] = points0[:,1]*scale_y[i]
    savetxt('../../input/autogen_input/%d.dat'%(i),points, fmt="%12.6G")
    airfoil_list.append('../../input/autogen_input/%d.dat'%(i))
# end for

airfoil_list.append(airfoil_list[-1])
airfoil_list.append(airfoil_list[-1])

# Now append two extra cross sections for the winglet
chord = append(chord,[chord[-1]*.90,.3*chord[-1]])
x=append(x,[x[-1] + 2,x[-1] + 25])
z=append(z,[140,144.5])
y=append(y,[y[-1] + 1.5,y[-1] + 15])

rot_x = zeros(naf)
rot_x[-1] = -80
rot_x[-2] = -80
rot_y = zeros(naf)
rot_z = zeros(naf)
offset = zeros((naf,2))
Nctlu = 11
                      
# Make the break-point vector
breaks = [19,20]
cont = [0,0] # vector of length breaks: 0 for c0 continuity 1 for c1 continutiy
nsections = [20,4,4] # length of breaks +1

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
bwb = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
                  file_type='xfoil',scale=chord,offset=offset, 
                  breaks=breaks,cont=cont,nsections=nsections,
                  end_type='rounded',end_scale = 1,
                  Xsec=X,rot=rot,fit_type='lms',Nctlu=Nctlu,Nfoil=45)


bwb.calcEdgeConnectivity(1e-6,1e-6)
bwb.writeEdgeConnectivity('bwb.con')
bwb.propagateKnotVectors()

bwb.writeTecplot('../../output/bwb.dat',orig=True)
bwb.writeIGES('../../input/bwb.igs')


print '---------------------------'
print 'Attaching Reference Axis...'
print '---------------------------'

# End-Type ref_axis attachments
ref_axis_x = X
rot = zeros(X.shape)
bwb.addRefAxis([0,1,2,3,4,5,6,7],X,rot)
print ' ** Adding Global Design Variables **'
bwb.addGeoDVGlobal('span',1,0.5,2.0,span_extension)
bwb.calcCtlDeriv()
print 'Done Ref Axis Adding!'






print '---------------------------'
print '      pyLayout Setup' 
print '---------------------------'
 
MAX_SPARS = 5  # This is the same for each spanwise section
Nsection = 1
wing_box = pyLayout.Layout(bwb,Nsection,MAX_SPARS)

# ---------- Create the First Domain -------------

# Load in the digitized data
# Use the digitize it data for the planform:
le_spar_list = array(loadtxt('bwb_le_spar.out'))
te_spar_list = array(loadtxt('bwb_te_spar.out'))
rib_list     = array(loadtxt('bwb_rib_list.out'))

# Now make a ONE-DIMENSIONAL spline for each of the le and trailing edes
le_spar_spline = pySpline.linear_spline(task='interpolate',k=2,X=le_spar_list[:,1],s=le_spar_list[:,0])
te_spar_spline = pySpline.linear_spline(task='interpolate',k=2,X=te_spar_list[:,1],s=te_spar_list[:,0])

MAX_RIBS = len(rib_list)

le_list = zeros((MAX_RIBS,3)) # Make them the same as the Rib list...not necessary but should work
te_list = zeros((MAX_RIBS,3)) # Make them the same as the Rib list...not necessary but should work

span_coord = rib_list[:,0]
le_list[:,2] = span_coord # Use the same spanwise cordinates (z-axis)
le_list[:,0] = le_spar_spline.getValueV(span_coord)
le_list[:,1] = mid_y_spline.getValueV(span_coord)

te_list[:,2] = span_coord
te_list[:,0] = te_spar_spline.getValueV(span_coord)
te_list[:,1] = mid_y_spline.getValueV(span_coord)

domain = pyLayout.domain(le_list,te_list)

# # ---------- OPTIONAL SPECIFIC RIB DISTIRBUTION -----------

#Comput the physical rib position
rib_pos = zeros((MAX_RIBS,3))
rib_basis = rib_list[:,0] # This is the basis
rib_pos[:,2] = rib_basis
rib_pos[:,1] = mid_y_spline.getValueV(rib_basis)
rib_pos[:,0] = rib_list[:,1]

rib_dir = zeros((MAX_RIBS,3))
rib_dir[:] = [1,0,0]

# # -----------------------------------------------------------

rib_blank = ones((MAX_RIBS,MAX_SPARS-1))
spar_blank = ones((MAX_SPARS,MAX_RIBS-1))

# Spacing Parameters for Elements
span_space = 1*ones(MAX_RIBS-1)
rib_space  = 1*ones(MAX_SPARS+1) # Note the +1
v_space    = 1

surfs = [[0],[1]] #Upper surfs for LE to TE then Lower Surfs from LE to TE
spar_con = [0,0,0,0,0]

timeA = time.time()

def1 = pyLayout.struct_def(MAX_RIBS,MAX_SPARS,domain,surfs,spar_con,
                           rib_blank=rib_blank,rib_pos=rib_pos,rib_dir=rib_dir,
                           spar_blank=spar_blank,
                           span_space = span_space,rib_space=rib_space,v_space=v_space)
                           
wing_box.addSection(def1)
wing_box.writeTecplot('../../output/bwb_layout.dat')
wing_box.finalize()
