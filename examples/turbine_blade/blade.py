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
chord = array([0.6640,0.6640,0.6640,0.6640,0.6640,0.6640,1.0950,1.6800,\
               1.5390,1.2540,0.9900,0.7900,0.6100,0.4550,0.4540,0.4530])
sloc = array([0.0000,0.0236,0.0273,0.0333,0.0393,0.0397,0.1141,\
              0.2184,0.3226,0.4268,0.5310,0.6352,0.7395,0.8437,0.9479,1.0000])
le_loc = array([0.5000,0.5000,0.5000,0.5000,0.5000,0.5000,0.3300,\
                0.2500,0.2500,0.2500,0.2500,0.2500,0.2500,0.2500,0.2500,0.2500])
tw_aero = array([0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,20.3900,16.0200,11.6500,\
           6.9600,1.9800,-1.8800,-3.3700,-3.4100,-3.4500,-3.4700])
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

print '---------------------------'
print '      pyLayout Setup' 
print '---------------------------'
 
MAX_SPARS = 2  # This is the same for each spanwise section
Nsection = 1
wing_box = pyLayout.Layout(blade,Nsection,MAX_SPARS)

# ---------- Create the First Domain -- First 8 ribts

MAX_RIBS_1 = naf

le_list = array([[0.06,0,0],[-.24,0,5.5],[-.3,0,21.15]])
te_list = array([[.54,0,0],[.45,0,5.5],[-.14,0,21.15]])
domain1 = pyLayout.domain(le_list.copy(),te_list.copy())

# # ---------- OPTIONAL SPECIFIC RIB DISTIRBUTION -----------

#Compute the physical rib position
rib_pos = linspace(0,bl_length,naf)
rib_dir = zeros((MAX_RIBS_1,3))
rib_dir[:] = [1,0,0]

# # -----------------------------------------------------------

rib_blank = zeros((MAX_RIBS_1,MAX_SPARS-1))
spar_blank = ones((MAX_SPARS,MAX_RIBS_1-1))

# Spacing Parameters for Elements
span_space = 3*ones(MAX_RIBS_1-1)
rib_space  = 2*ones(MAX_SPARS+1) # Note the +1
v_space    = 2

surfs = [[0],[1]] #Upper surfs for LE to TE then Lower Surfs from LE to TE
spar_con = [0,0]

def1 = pyLayout.struct_def(MAX_RIBS_1,MAX_SPARS,domain1,surfs,spar_con,
                           rib_blank=rib_blank,rib_pos=rib_pos,rib_dir=rib_dir,
                           spar_blank=spar_blank,
                           span_space = span_space,rib_space=rib_space,v_space=v_space)
                           
wing_box.addSection(def1)
wing_box.writeTecplot('./blade_layout.dat')
wing_box.finalize()
