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

# pyOpt
sys.path.append('../../../../pyACDT/pyACDT/Optimization/pyOpt')

# pySnopt
sys.path.append('../../../../pyACDT/pyACDT/Optimization/pyOpt/pySNOPT')

#pyGeo
sys.path.append('../../')
import pyGeo_NM as pyGeo

#pyLayout
sys.path.append('../../../pyLayout/')
import pyLayout_NM as pyLayout

#Matplotlib
try:
    from matplotlib.pylab import plot,show
except:
    print 'Matploptlib could not be imported'
# end if

# ==============================================================================
# Start of Script
# ==============================================================================

# Blade Information - Create a Geometry Object from cross sections

# ------------- Taken directly from precomp example ---------------
naf = 16
bl_length = 21.15
chord = [0.6640,0.6640,0.6640,0.6640,0.6640,0.6640,1.0950,1.6800,\
                                1.5390,1.2540,0.9900,0.7900,0.6100,0.4550,0.4540,0.4530]
sloc = [0.0000,0.0236,0.0273,0.0333,0.0393,0.0397,0.1141,\
                                0.2184,0.3226,0.4268,0.5310,0.6352,0.7395,0.8437,0.9479,1.0000]
le_loc = [0.5000,0.5000,0.5000,0.5000,0.5000,0.5000,0.3300,\
                                 0.2500,0.2500,0.2500,0.2500,0.2500,0.2500,0.2500,0.2500,0.2500]
tw_aero = [0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,20.3900,16.0200,11.6500,\
                              6.9600,1.9800,-1.8800,-3.3700,-3.4100,-3.4500,-3.4700]
airfoil_list = ['af1-6.inp','af1-6.inp','af1-6.inp','af1-6.inp','af1-6.inp','af1-6.inp',\
                    'af-07.inp','af8-9.inp','af8-9.inp','af-10.inp','af-11.inp','af-12.inp','af-13.inp',\
                    'af-14.inp','af15-16.inp','af15-16.inp']
# -------------------------------------------------------------------

offset = zeros((naf,2))
offset[:,0] = sloc

X = zeros((naf,3))
rot = zeros((naf,3))
X[:,0] = zeros(naf)
X[:,1] = zeros(naf)
X[:,2] = sloc
X[:,2]*=bl_length
rot[:,0] = zeros(naf)
rot[:,1] = zeros(naf)
rot[:,2] = tw_aero

Nctlu = 15
# ------------------------------------------------------------------
# Procedure for Using pyGEO

# Step 1: Run the folloiwng Commands: (Uncomment between -------)
# ---------------------------------------------------------------------
#Note: u direction is chordwise, v direction is span-wise
# blade = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
#                     file_type='precomp',scale=chord,offset=offset, 
#                     end_type='rounded',end_scale =1,
#                     Xsec=X,rot=rot,fit_type='lms',Nctlu=Nctlu,Nfoil=45)
# blade.setSymmetry('xy')
# #blade.calcEdgeConnectivity(1e-6,1e-6)
# #blade.writeEdgeConnectivity('blade.con')
# #sys.exit(0)
# blade.readEdgeConnectivity('blade.con')
# blade.propagateKnotVectors()

# blade.fitSurfaces(nIter=100,constr_tol=1e-7,opt_tol=1e-6)
# blade.writeIGES('./blade.igs')
# blade.writeTecplot('./blade.dat',orig=True,nodes=True)


# Read the IGES file
blade = pyGeo.pyGeo('iges',file_name='blade.igs')
blade.readEdgeConnectivity('blade.con')
blade.writeTecplot('./blade.dat',orig=True,nodes=True)

print '---------------------------'
print 'Attaching Reference Axis...'
print '---------------------------'

ref_axis_x = X
rot = zeros(X.shape)
blade.addRefAxis([0,1,2,3],X,rot)
print 'Done Ref Axis Adding!'

print '---------------------------'
print 'Adding Design Variables' 
print '---------------------------'

def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
    ref_axis[0].x[:,2] = ref_axis[0].x0[:,2] * val
    return ref_axis

print ' ** Adding Global Design Variables **'
blade.addGeoDVGlobal('span',1,0.5,2.0,span_extension)
blade.calcCtlDeriv()
sys.exit(0)
print '---------------------------'
print '      pyLayout Setup' 
print '---------------------------'
 
MAX_SPARS = 2  # This is the same for each spanwise section
Nsection = 4
wing_box = pyLayout.Layout(blade,Nsection,MAX_SPARS)

# Create the full set of rib data for the main body/wing section (out
# as far as winglet blend)

# Load in the digitized data
# Use the digitize it data for the planform:
le_spar_list = array(loadtxt('blade_le_spar.out'))
te_spar_list = array(loadtxt('blade_te_spar.out'))
rib_list     = array(loadtxt('blade_rib_list.out'))

# Now make a ONE-DIMENSIONAL spline for each of the le and trailing edes
le_spar_spline = pySpline.linear_spline(task='interpolate',k=2,X=le_spar_list[:,1],s=le_spar_list[:,0])
te_spar_spline = pySpline.linear_spline(task='interpolate',k=2,X=te_spar_list[:,1],s=te_spar_list[:,0])


# ---------- Create the First Domain -- First 8 ribts

MAX_RIBS1 = 8

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

MAX_RIBS2 = 19

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
le_list3[1] = blade.surfs[4].getValue(0,0.5)+[.025,0,0]
te_list3[1] = blade.surfs[4].getValue(1,0.5)-[.025,0,0]
le_list3[2] = blade.surfs[4].getValue(0,1) + [.025,0,0]
te_list3[2] = blade.surfs[4].getValue(1,1) - [.025,0,0]

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

le_list4[:] = blade.surfs[6].getValueV(zeros(MAX_RIBS4),linspace(0,1,MAX_RIBS4)) + [.0085,0,0]
te_list4[:] = blade.surfs[6].getValueV(ones(MAX_RIBS4),linspace(0,1,MAX_RIBS4)) - [.005,0,0]
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


wing_box.writeTecplot('../../output/_layout.dat')
wing_box.finalize()
