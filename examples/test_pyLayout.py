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

# pyOpt
sys.path.append('../../../pyACDT/pyACDT/Optimization/pyOpt')

# pySnopt
sys.path.append('../../../pyACDT/pyACDT/Optimization/pyOpt/pySNOPT')

#pyGeo
sys.path.append('../')
import pyGeo_NM as pyGeo

#geo_utils
from geo_utils import *

#pyLayout
sys.path.append('../../pyLayout/')
import pyLayout_NM as pyLayout

#Design Variable Functions
from dv_funcs import *

# ==============================================================================
# Start of Script
# ==============================================================================

# Wing Information - Create a Geometry Object from cross sections

naf=2
airfoil_list = ['../input/naca0012.dat','../input/naca0012.dat']
chord = [1,1]
x = [0,0]
y = [0,0]
z = [0,3.94]
rot_x = [0,0]
rot_y = [0,0]
tw_aero = [0,0]

offset = zeros((naf,2))
offset[:,0] = .25 # Offset sections by 0.25 in x
section_spacing = [linspace(0,1,10)]
nsections = [10]
# Make the break-point vector
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
# ------------------------------------------------------------------
# #Note: u direction is chordwise, v direction is span-wise
wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,\
                   file_type='xfoil',scale=chord,offset=offset, \
                   Xsec=X,rot=rot,end_type=end_type,nsections=nsections,
                   section_spacing=section_spacing,
                   fit_type='lms', Nctlu=Nctlu,Nfoil=45)

#wing.calcEdgeConnectivity(1e-6,1e-6)
#wing.writeEdgeConnectivity('test_layout.con')
wing.readEdgeConnectivity('test_layout.con')
wing.propagateKnotVectors()
wing.fitSurfaces()
wing.writeTecplot('../output/wing.dat',labels=True,ref_axis=True,directions=True,nodes=True)
wing.writeIGES('../input/wing.igs')
# ------------------------------------------------------------------

#Load in the split plot3d file
# wing = pyGeo.pyGeo('plot3d',file_name='../input/wing.xyz.fmt')
# wing.calcEdgeConnectivity(1e-6,1e-6)
# wing.propagateKnotVectors()


print '---------------------------'
print 'Attaching Reference Axis...'
print '---------------------------'

wing.addRefAxis([0,1,2,3],X[0:2,:],rot[0:2,:],nrefsecs=nsections[0],\
                    spacing=section_spacing[0])
print 'Done Ref Axis Adding!'

print ' ** Adding Global Design Variables **'
wing.addGeoDVGlobal('span',1,0.5,2.0,span_extension)
wing.calcCtlDeriv()

print '---------------------------'
print '      pyLayout Setup' 
print '---------------------------'
 
MAX_SPARS = 5  # This is the same for each spanwise section
Nsection = 1
wing_box = pyLayout.Layout(wing,Nsection,MAX_SPARS)

# ---------- Create the First Domain -------------

MAX_RIBS = 8
le_list = array([[-.10,0,0],[-.10,0,3.94]])
te_list = array([[.60,0,0],[.6,0,3.94]])

domain = pyLayout.domain(le_list,te_list)

# # ---------- OPTIONAL SPECIFIC RIB DISTIRBUTION -----------
rib_pos = zeros((MAX_RIBS,3))
spline = pySpline.linear_spline(task='interpolate',k=2,X=X[0:2])
rib_pos = spline.getValueV(linspace(0,1,MAX_RIBS))

rib_dir = zeros((MAX_RIBS,3))
rib_dir[:] = [1,0,0] # Note a rib-dir MUST be specified 
# # -----------------------------------------------------------

rib_blank = ones((MAX_RIBS,MAX_SPARS-1))
spar_blank = ones((MAX_SPARS,MAX_RIBS-1))
span_space = 2*ones(MAX_RIBS-1)
rib_space  = 2*ones(MAX_SPARS+1) # Note the +1
v_space    = 2

surfs = [[0],[1]] #Upper surfs for LE to TE then Lower Surfs from LE to TE
spar_con = [0,0,0,0,0]

timeA = time.time()
def1 = pyLayout.struct_def(MAX_RIBS,MAX_SPARS,domain,surfs,spar_con,
                           rib_blank=rib_blank,rib_pos=rib_pos,rib_dir=rib_dir,
                           spar_blank=spar_blank,
                           span_space = span_space,rib_space=rib_space,v_space=v_space)
                           
wing_box.addSection(def1)
wing_box.writeTecplot('../output/layout.dat')
wing_box.finalize()

print 'Time is:',time.time()-timeA
