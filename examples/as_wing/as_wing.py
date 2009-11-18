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
    lexsort, savetxt, append, zeros_like

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

# geo_utils
from geo_utils import *

# pySpline 
import pySpline

#pyGeo
import pyGeo

#pyLayout
import pyLayout

# TACS
from pyTACS import TACS
import tacs_utils
from tacs_utils import TriPan

try:
    import mpi4py
    from mpi4py import MPI
except:
    MPI = None
# end try

# ==============================================================================
# Start of Script
# ==============================================================================

# We will assume all the geometry pre-processing has been
# completed...all we have left for geometry import is .igs file and .con file

# Read the IGES file
as_wing = pyGeo.pyGeo('iges',file_name='./as_wing.igs')
as_wing.readEdgeConnectivity('as_wing.con')
as_wing.setSymmetry('xy')

mpiPrint('---------------------------')
mpiPrint('Attaching Reference Axis...')
mpiPrint('---------------------------')

# Add a single reference axis:

ref_axis_x = array([[0,0,0],[0,0,4]])
rot = zeros_like(ref_axis_x)
as_wing.addRefAxis([0,1,2,3],ref_axis_x,rot,nrefsecs=5)
mpiPrint('Done Ref Axis Adding!')

mpiPrint('------------------------------')
mpiPrint('Adding Design Variables...')
mpiPrint('------------------------------')
ntwist = 5
def twist(vals,ref_axis):
    '''Twist'''
    ref_axis[0].rot[:,2] = vals
    return ref_axis

mpiPrint(' ** Adding Global Design Variables **')
as_wing.addGeoDVGlobal('twist',zeros(ntwist),-10,10,twist)
as_wing.calcCtlDeriv()

mpiPrint('---------------------------')
mpiPrint('      TriPan Setup')
mpiPrint('---------------------------')

fname = 'as_wing_coarse.dtx'
Xtip   = [ .558,0,4.09]
Xroot = [1,0,0]

use_tripan_file = False
if use_tripan_file:
   pass
else:
    tp = TriPan.getTriPanDefaults()
    tp['Nu'] = 50
    tp['Nv'] = 50
    tp['time_dependent'] = 1
    tp['n_downstream'] = 144
    panel = TriPan.TriPan( fname, 'datex', outFile='as_wing_coarse.tripan',
                           wakeInfo = [[Xtip,Xroot]],reverseNormals=False,
                           wakeOutFile='as_wing_coarse.wake',
                           tp_params=tp)
# end if

mpiPrint(' * * Associating as_wing geometry * *')
panel.associateGeometry( as_wing )
mpiPrint(' * * Updating surface points * *')
panel.updatePoints()

# Set up the trajectory for an unsteady simualtion Note for the
# pyGeo-Aerosurf coordinate system...+x free strem, +y up, +z out left
# wing, the folloiwng rotation angles coorspond to the usual yaw-pitch-roll convention:
# Rot[0] = Roll Angle
# Rot[1] = Yaw Angle
# Rot[2] = Pitch Angle
#
def getBodyRotation(rot):
    '''
    Return the tranformation to rotate the body coordinates based on
    the roll-pitch-roll angles given in rot. Note: rot is the rotations
    about x-y-z as given in the pyGeo/Aerosurf z-out-the-wing
    orientation. However, the are applied in the roll-pitch-yaw fashion:
    i.e. x-z-y
    '''
    Cyaw = array( [ [ cos(rot[1]), 0.0, sin(rot[1]) ],
                          [ 0.0, 1.0, 0.0 ],
                          [ -sin(rot[1]), 0.0, cos(rot[1]) ] ] )

    Cpitch = array( [ [ cos(rot[2]), - sin(rot[2]), 0.0 ],
                          [ sin(rot[2]),   cos(rot[2]), 0.0 ],
                          [ 0.0, 0.0, 1.0 ] ] )

    Croll = array( [ [ 1.0, 0.0, 0.0], 
                           [ 0.0, cos(rot[0]),  -sin(rot[0]) ],
                           [ 0.0, sin(rot[0]),  cos(rot[0])] ] )

    return dot(Cyaw,dot(Cpitch,Croll))

# ----- Pitching Motion ---------
# U = 10
# omega = 10
# def getPoint( t ):
#     '''Return the positon, velocity, angular velocity and rotation matrix for the given time'''
    
#     R = zeros(3)
#     R[0] = -U*t

#     V = zeros(3)
#     V[0] = -U

#     rot = zeros(3)
#     rot[2]  =  -(5*pi/180)*sin(omega*t)
#     Cbe = getBodyRotation(rot)

#     Omega = zeros(3)
#     Omega[2] = -(5*pi/180)*cos(omega*t)*omega # Pitch Angle
    
#     return R, V, Omega, Cbe.flatten()

#----- Plunging Motion --------
U = 1.56
h0 = .019
omega = 4.26*2*pi
c = 1
mpiPrint('Reduced Frequency: %f'%(omega*c/(2*U)))

def getPoint( t ):
    '''Return the positon, velocity, angular velocity and rotation matrix for the given time'''
    
    R = zeros(3)
    R[0] = -U*t
    R[1] = h0*sin(omega*t)

    V = zeros(3)
    V[0] = -U
    V[1] = omega*h0*cos(omega*t)

    rot = zeros(3)
    Cbe = getBodyRotation(rot)

    Omega = zeros(3)
    
    return R, V, Omega, Cbe.flatten()

steps = 144
#time_max = 24*pi/omega
#dt = time_max/(steps-1)
dt = (2/U)/steps
mpiPrint('Max Sim Time: %f'%(dt*steps))

panel.timeSolve( getPoint, steps=steps, dt=dt, init_dist=0.05 )

sys.exit(0)

mpiPrint('---------------------------')
mpiPrint('      pyLayout Setup'       )
mpiPrint('---------------------------')
 
MAX_SPARS = 2  # This is the same for each spanwise section
Nsection = 4
wing_box = pyLayout.Layout(as_wing,Nsection,MAX_SPARS)

# Create the full set of rib data for the main body/wing section (out
# as far as winglet blend)

# Load in the digitized data
# Use the digitize it data for the planform:
le_spar_list = array(loadtxt('as_wing_le_spar.out'))
te_spar_list = array(loadtxt('as_wing_te_spar.out'))
rib_list     = array(loadtxt('as_wing_rib_list.out'))

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

domain1 = pyLayout.domain(le_list.copy(),te_list.copy())

# # ---------- OPTIONAL SPECIFIC RIB DISTIRBUTION -----------

#Comput the physical rib position
rib_pos = zeros((MAX_RIBS1,3))
rib_basis = rib_list[0:MAX_RIBS1,0] # This is the basis
rib_pos[:,2] = rib_basis
rib_pos[:,1] = mid_y_spline.getValueV(rib_basis)
rib_pos[:,0] = rib_list[0:MAX_RIBS1,1]
rib_pos
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

rib_space  = 2*ones(MAX_SPARS+1) # Note the +1
v_space    = 2

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

domain2 = pyLayout.domain(le_list,te_list)

# # ---------- OPTIONAL SPECIFIC RIB DISTIRBUTION -----------

#Comput the physical rib position
rib_pos = zeros((MAX_RIBS2,3))
rib_basis = rib_list[MAX_RIBS1-1:MAX_RIBS1+MAX_RIBS2,0] # This is the basis
rib_pos[:,2] = rib_basis
rib_pos[:,1] = mid_y_spline.getValueV(rib_basis)
rib_pos[:,0] = rib_list[MAX_RIBS1-1:MAX_RIBS1+MAX_RIBS2,1]

rib_dir = zeros((MAX_RIBS2,3))
rib_dir[:] = [1,0,0]

# # -----------------------------------------------------------

rib_blank = ones((MAX_RIBS2,MAX_SPARS-1))
spar_blank = ones((MAX_SPARS,MAX_RIBS2-1))

# Spacing Parameters for Elements
span_space = 1*ones(MAX_RIBS2-1)
rib_space  = 2*ones(MAX_SPARS+1) # Note the +1
v_space    = 2

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
le_list3[1] = as_wing.surfs[4].getValue(0,0.5)+[2.5,0,0]
te_list3[1] = as_wing.surfs[4].getValue(1,0.5)-[2.5,0,0]
le_list3[2] = as_wing.surfs[4].getValue(0,1) + [2.5,0,0]
te_list3[2] = as_wing.surfs[4].getValue(1,1) - [2.5,0,0]

domain3 = pyLayout.domain(le_list3,te_list3)

# # ---------- OPTIONAL SPECIFIC RIB DISTIRBUTION -----------

#Comput the physical rib position
rib_pos = zeros((MAX_RIBS3,3))
rib_pos = 0.5*(le_list3+te_list3)
rib_dir = zeros((MAX_RIBS3,3))
rib_dir[:] = [1,0,0]

# # -----------------------------------------------------------

rib_blank = ones((MAX_RIBS3,MAX_SPARS-1))
spar_blank = ones((MAX_SPARS,MAX_RIBS3-1))

# Spacing Parameters for Elements
span_space = 1*ones(MAX_RIBS3-1)
rib_space  = 2*ones(MAX_SPARS+1) # Note the +1
v_space    = 2

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

le_list4[:] = as_wing.surfs[6].getValueV(zeros(MAX_RIBS4),linspace(0,1,MAX_RIBS4)) + [.85,0,0]
te_list4[:] = as_wing.surfs[6].getValueV(ones(MAX_RIBS4),linspace(0,1,MAX_RIBS4)) - [.5,0,0]
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
rib_space  = 2*ones(MAX_SPARS+1) # Note the +1
v_space    = 2

surfs = [[6],[7]] #Upper surfs for LE to TE then Lower Surfs from LE to TE
spar_con = [6,6]

def4 = pyLayout.struct_def(MAX_RIBS4,MAX_SPARS,domain4,surfs,spar_con,
                           rib_blank=rib_blank,rib_pos=rib_pos,rib_dir=rib_dir,
                           spar_blank=spar_blank,
                           span_space = span_space,rib_space=rib_space,v_space=v_space)
                           
wing_box.addSection(def4)

wing_box.writeTecplot('./as_wing_layout.dat')
wing_box.finalize()
