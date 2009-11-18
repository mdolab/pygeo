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
#pyPSG:
sys.path.append(os.path.abspath('../../../pySpline/python'))
sys.path.append(os.path.abspath('../../')) # pyGeo & geo_utils
sys.path.append(os.path.abspath('../../../pyLayout/'))

#cfd-csm pre (Optional)
sys.path.append('../../../../pyHF/pycfd-csm/python/')

# pyOpt
sys.path.append('../../../../pyACDT/pyACDT/Optimization/pyOpt')

# pySnopt
sys.path.append('../../../../pyACDT/pyACDT/Optimization/pyOpt/pySNOPT')

# pySpline 
import pySpline

#pyGeo
import pyGeo

#pyLayout
import pyLayout

#Design Variable Functions
sys.path.append('../')
from dv_funcs import *

#Matplotlib
try:
    from matplotlib.pylab import plot,show
except:
    print 'Matploptlib could not be imported'
# end if

# ==============================================================================
# Start of Script
# ==============================================================================

# Run the geometry data script
execfile('bwb_data.py')

# ------------------------------------------------------------------
# Procedure for Using pyGEO

# Step 1: Run the folloiwng Commands: (Uncomment between -------)
# ---------------------------------------------------------------------
#Note: u direction is chordwise, v direction is span-wise
bwb = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
                  file_type='xfoil',scale=chord,offset=offset, 
                  breaks=breaks,cont=cont,nsections=nsections,
                  end_type='rounded',end_scale =1,
                  Xsec=X,rot=rot,fit_type='lms',Nctlu=Nctlu,Nfoil=45)

bwb.setSymmetry('xy')
##bwb.calcEdgeConnectivity(1e-6,1e-6)
##bwb.writeEdgeConnectivity('bwb.con')
bwb.readEdgeConnectivity('bwb.con')
bwb.propagateKnotVectors()

#bwb.fitSurfaces(nIter=100,constr_tol=1e-8,opt_tol=1e-6)
bwb.coef*=SCALE
bwb.update()
bwb.writeIGES('../../input/bwb_constr1.igs')

# bwb.fitSurfaces3(nIter=150,constr_tol=1e-8,opt_tol=1e-6)
# bwb.writeIGES('../../input/bwb_constr2.igs')
# #bwb.writeTecplot('../../output/bwb.dat',orig=True,nodes=True)

#sys.exit(0)

# Read the IGES file
bwb = pyGeo.pyGeo('iges',file_name='../../input/bwb_constr2.igs')
bwb.readEdgeConnectivity('bwb.con')
bwb.writeTecplot('../../output/bwb.dat',orig=True,nodes=True)
#sys.exit(0)

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
 
MAX_SPARS = 2  # This is the same for each spanwise section
Nsection = 4
wing_box = pyLayout.Layout(bwb,Nsection,MAX_SPARS)

# Create the full set of rib data for the main body/wing section (out
# as far as winglet blend)

# Load in the digitized data
# Use the digitize it data for the planform:
le_spar_list = array(loadtxt('bwb_le_spar.out'))
te_spar_list = array(loadtxt('bwb_te_spar.out'))
rib_list     = array(loadtxt('bwb_rib_list.out'))

# Now make a ONE-DIMENSIONAL spline for each of the le and trailing edes
le_spar_spline = pySpline.linear_spline(task='interpolate',k=2,X=le_spar_list[:,1],s=le_spar_list[:,0])
te_spar_spline = pySpline.linear_spline(task='interpolate',k=2,X=te_spar_list[:,1],s=te_spar_list[:,0])


# ---------- Create the First Domain -- First 8 ribts

MAX_RIBS1 = 5

le_list = zeros((MAX_RIBS1,3)) # Make them the same as the Rib list...not necessary but should work
te_list = zeros((MAX_RIBS1,3)) # Make them the same as the Rib list...not necessary but should work

span_coord = rib_list[0:MAX_RIBS1,0]
le_list[:,2] = span_coord # Use the same spanwise cordinates (z-axis)
le_list[:,0] = le_spar_spline.getValueV(span_coord)
le_list[:,1] = mid_y_spline.getValueV(span_coord)

te_list[:,2] = span_coord
te_list[:,0] = te_spar_spline.getValueV(span_coord)
te_list[:,1] = mid_y_spline.getValueV(span_coord)

le_list/=SCALE
te_list/=SCALE
domain1 = pyLayout.domain(le_list.copy(),te_list.copy())


# # ---------- OPTIONAL SPECIFIC RIB DISTIRBUTION -----------

#Comput the physical rib position
rib_pos = zeros((MAX_RIBS1,3))
rib_basis = rib_list[0:MAX_RIBS1,0] # This is the basis
rib_pos[:,2] = rib_basis
rib_pos[:,1] = mid_y_spline.getValueV(rib_basis)
rib_pos[:,0] = rib_list[0:MAX_RIBS1,1]
rib_pos/=SCALE
rib_dir = zeros((MAX_RIBS1,3))
rib_dir[:] = [1,0,0]

# # -----------------------------------------------------------

rib_blank = ones((MAX_RIBS1,MAX_SPARS-1))
spar_blank = ones((MAX_SPARS,MAX_RIBS1-1))

# Spacing Parameters for Elements
span_space = 1*ones(MAX_RIBS1-1)
span_space[0] = 2
span_space[1] = 2
span_space[2] = 4

rib_space  = 3*ones(MAX_SPARS+1) # Note the +1
v_space    = 3

surfs = [[0],[1]] #Upper surfs for LE to TE then Lower Surfs from LE to TE
spar_con = [0,0]

def1 = pyLayout.struct_def(MAX_RIBS1,MAX_SPARS,domain1,surfs,spar_con,
                           rib_blank=rib_blank,rib_pos=rib_pos,rib_dir=rib_dir,
                           spar_blank=spar_blank,
                           span_space = span_space,rib_space=rib_space,v_space=v_space)
                           
wing_box.addSection(def1)

# # ---------- Create the Second Domain -- Last First 19 ribs

MAX_RIBS2 = 22

le_list = zeros((MAX_RIBS2,3)) # Make them the same as the Rib list...not necessary but should work
te_list = zeros((MAX_RIBS2,3)) # Make them the same as the Rib list...not necessary but should work

span_coord = rib_list[MAX_RIBS1-1:MAX_RIBS1+MAX_RIBS2,0]

le_list[:,2] = span_coord # Use the same spanwise cordinates (z-axis)
le_list[:,0] = le_spar_spline.getValueV(span_coord)
le_list[:,1] = mid_y_spline.getValueV(span_coord)

te_list[:,2] = span_coord
te_list[:,0] = te_spar_spline.getValueV(span_coord)
te_list[:,1] = mid_y_spline.getValueV(span_coord)

le_list/=SCALE
te_list/=SCALE

domain2 = pyLayout.domain(le_list,te_list)

# # ---------- OPTIONAL SPECIFIC RIB DISTIRBUTION -----------

#Comput the physical rib position
rib_pos = zeros((MAX_RIBS2,3))
rib_basis = rib_list[MAX_RIBS1-1:MAX_RIBS1+MAX_RIBS2,0] # This is the basis
rib_pos[:,2] = rib_basis
rib_pos[:,1] = mid_y_spline.getValueV(rib_basis)
rib_pos[:,0] = rib_list[MAX_RIBS1-1:MAX_RIBS1+MAX_RIBS2,1]
rib_pos/=SCALE
rib_dir = zeros((MAX_RIBS2,3))
rib_dir[:] = [1,0,0]

# # -----------------------------------------------------------

rib_blank = ones((MAX_RIBS2,MAX_SPARS-1))
spar_blank = ones((MAX_SPARS,MAX_RIBS2-1))

# Spacing Parameters for Elements
span_space = 1*ones(MAX_RIBS2-1)
rib_space  = 3*ones(MAX_SPARS+1) # Note the +1
v_space    = 3

surfs = [[2],[3]] #Upper surfs for LE to TE then Lower Surfs from LE to TE
spar_con = [2,2]

def2 = pyLayout.struct_def(MAX_RIBS2,MAX_SPARS,domain2,surfs,spar_con,
                           rib_blank=rib_blank,rib_pos=rib_pos,rib_dir=rib_dir,
                           spar_blank=spar_blank,
                           span_space = span_space,rib_space=rib_space,v_space=v_space)
                           
wing_box.addSection(def2)


# # ---------- Create the Third Domain Wing Blend

MAX_RIBS3 = 3

le_list3 = zeros((MAX_RIBS3,3)) # Make them the same as the Rib list...not necessary but should work
te_list3 = zeros((MAX_RIBS3,3)) # Make them the same as the Rib list...not necessary but should work

le_list3[0] = le_list[-1]
te_list3[0] = te_list[-1]
le_list3[1] = bwb.surfs[4].getValue(0,0.5)+[.025,0,0]
te_list3[1] = bwb.surfs[4].getValue(1,0.5)-[.025,0,0]
le_list3[2] = bwb.surfs[4].getValue(0,1) + [.025,0,0]
te_list3[2] = bwb.surfs[4].getValue(1,1) - [.025,0,0]

domain3 = pyLayout.domain(le_list3,te_list3)

# # ---------- OPTIONAL SPECIFIC RIB DISTIRBUTION -----------

#Comput the physical rib position
rib_pos = zeros((MAX_RIBS3,3))
rib_pos = 0.5*(le_list3+te_list3)
rib_dir = zeros((MAX_RIBS3,3))
rib_dir[:] = [1,0,0]
print 'rib_pos:',rib_pos
# # -----------------------------------------------------------

rib_blank = ones((MAX_RIBS3,MAX_SPARS-1))
spar_blank = ones((MAX_SPARS,MAX_RIBS3-1))

# Spacing Parameters for Elements
span_space = 1*ones(MAX_RIBS3-1)
rib_space  = 3*ones(MAX_SPARS+1) # Note the +1
v_space    = 3

surfs = [[4],[5]] #Upper surfs for LE to TE then Lower Surfs from LE to TE
spar_con = [4,4]

def3 = pyLayout.struct_def(MAX_RIBS3,MAX_SPARS,domain3,surfs,spar_con,
                           rib_blank=rib_blank,rib_pos=rib_pos,rib_dir=rib_dir,
                           spar_blank=spar_blank,
                           span_space = span_space,rib_space=rib_space,v_space=v_space)
                           
wing_box.addSection(def3)


# # ---------- Create the Fourth Domain Winglet

MAX_RIBS4 = 8

le_list4 = zeros((MAX_RIBS4,3)) # Make them the same as the Rib list...not necessary but should work
te_list4 = zeros((MAX_RIBS4,3)) # Make them the same as the Rib list...not necessary but should work

le_list4[:] = bwb.surfs[6].getValueV(zeros(MAX_RIBS4),linspace(0,1,MAX_RIBS4)) + [.0085,0,0]
te_list4[:] = bwb.surfs[6].getValueV(ones(MAX_RIBS4),linspace(0,1,MAX_RIBS4)) - [.005,0,0]
domain4 = pyLayout.domain(le_list4,te_list4)

# # ---------- OPTIONAL SPECIFIC RIB DISTIRBUTION -----------

#Comput the physical rib position
rib_pos = zeros((MAX_RIBS4,3))
rib_pos = 0.5*(le_list4+te_list4)
rib_dir = zeros((MAX_RIBS4,3))
rib_dir[:] = [1,0,0]
# # -----------------------------------------------------------

rib_blank = ones((MAX_RIBS4,MAX_SPARS-1))
spar_blank = ones((MAX_SPARS,MAX_RIBS4-1))

# Spacing Parameters for Elements
span_space = 1*ones(MAX_RIBS4-1)
rib_space  = 3*ones(MAX_SPARS+1) # Note the +1
v_space    = 3

surfs = [[6],[7]] #Upper surfs for LE to TE then Lower Surfs from LE to TE
spar_con = [6,6]

def4 = pyLayout.struct_def(MAX_RIBS4,MAX_SPARS,domain4,surfs,spar_con,
                           rib_blank=rib_blank,rib_pos=rib_pos,rib_dir=rib_dir,
                           spar_blank=spar_blank,
                           span_space = span_space,rib_space=rib_space,v_space=v_space)
                           
wing_box.addSection(def4)


wing_box.writeTecplot('../../output/bwb_layout.dat')
wing_box.finalize()
