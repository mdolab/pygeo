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

# Run the geometry data script
execfile('bwb_data.py')

# We will assume all the geometry pre-processing has been
# completed...all we have left for geometry import is .igs file and .con file

# Read the IGES file
bwb = pyGeo.pyGeo('iges',file_name='../../input/bwb_constr1.igs')
bwb.readEdgeConnectivity('bwb.con')
bwb.setSymmetry('xy')

mpiPrint('---------------------------')
mpiPrint('Attaching Reference Axis...')
mpiPrint('---------------------------')

# Add a single reference axis:
ref_axis_x = array([[.43,0,0],[.9648,.025,.9525]])
rot = zeros_like(ref_axis_x)
bwb.addRefAxis([0,1,2,3,4,5,6,7],ref_axis_x,rot,nrefsecs=5)
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
bwb.addGeoDVGlobal('twist',zeros(ntwist),-10,10,twist)
bwb.calcCtlDeriv()
bwb.writeTecplot('bwb.dat',ref_axis=True,links=True)

mpiPrint('---------------------------')
mpiPrint('      TriPan Setup')
mpiPrint('---------------------------')

fname = 'bwb_coarse_no_tip.dtx'
Xtip   = [ 164.925,18.7373,144.5]
Xbase  = [ 150.290,5.22604,140]
Xbase2 = [ 149.686,3.72889,138]
Xroot = [145.422,0,0]

use_tripan_file = False
if use_tripan_file:
    fname = 'bwb_coarse_no_tip.tripan'
    wakeFile = 'bwb_coarse_no_tip.wake'
    tp = TriPan.getTriPanDefaults()
    tp['Nu'] = 50
    tp['Nv'] = 50
    panel = TriPan.TriPan( fname, 'tripan', wakeFile=wakeFile,tp_params=tp)
else:
    tp = TriPan.getTriPanDefaults()
    tp['Nu'] = 50
    tp['Nv'] = 50
    panel = TriPan.TriPan( fname, 'datex', outFile='bwb_coarse_no_tip.tripan',
                           wakeInfo = [[Xtip,Xbase],[Xbase,Xbase2],[Xbase2,Xroot]],
                           reverseNormals=False, wakeOutFile='bwb_coarse_no_tip.wake',
                           tp_params=tp)
# end if

mpiPrint(' * * Associating bwb geometry * *')
panel.associateGeometry( bwb )
mpiPrint(' * * Updating surface points * *')
panel.updatePoints()


# Set up the trajectory for an unsteady simualtion Note for the
# pyGeo-Aerosurf coordinate system...+x free strem, +y up, +z out left
# wing, the folloiwng rotation angles coorspond to the usual yaw-pitch-roll convention:
# Rot[0] = Roll Angle
# Rot[1] = Yaw Angle
# Rot[2] = Pitch Angle
#
# The rotations are applied in the usualy yaw-pitch-roll sequence

Vinf = 10
omega = .1
def PosRot(t):
    '''Calulate the position and rotation of the body as a function of time'''
    X = zeros(3)
    Rot = zeros(3)
    X[0] = -Vinf*t
    X[1] = 0.0
    X[2] = 0.0

    Rot[0] = 0.0 # Roll Angle
    Rot[1] = 0.0 # Yaw Angle
    Rot[2] = 0.0 # Pitch Angle

    return X,Rot

def PosRot_dot(t):
    '''Caclualte the time derivative of the position and rotation of
    the body as a function of time'''
    Xdot = zeros(3)
    Rotdot = zeros(3)

    Xdot[0] = -Vinf
    Xdot[1] = 0.0
    Xdot[2] = 0.0

    Rotdot[0] = 0.0
    Rotdot[1] = 0.0
    Rotdot[2] = 0.0

    return Xdot,Rotdot

a0 = 0 # Constant angle of attack
omega = .1 # rad/sec
R     = 10 # meters
# Vinf = 1 = omega*R

# def PosRot(t):
#     '''Calulate the position and rotation of the body as a function of time'''
#     X = zeros(3)
#     Rot = zeros(3)
#     X[0] = -R*cos(-omega*t + pi/2)
#     X[1] = -R*sin(-omega*t + pi/2)
#     X[2] = 0.0

#     Rot[0] = 0.0 
#     Rot[1] = 0.0
#     Rot[2] = omega*t

#     return X,Rot

# def PosRot_dot(t):
#     '''Caclualte the time derivative of the position and rotation of
#     the body as a function of time'''
#     Xdot = zeros(3)
#     Rotdot = zeros(3)

#     Xdot[0] = -omega*R*sin(-omega*t + pi/2)
#     Xdot[1] = omega*R*cos(-omega*t + pi/2)
#     Xdot[2] = 0.0

#     Rotdot[0] = 0.0
#     Rotdot[1] = 0.0
#     Rotdot[2] = omega

#     return Xdot,Rotdot

U = 10
def PosRot(t):
    '''Calulate the position and rotation of the body as a function of time'''
    X = zeros(3)
    Rot = zeros(3)
    X[0] = -U*t
    X[1] = 0.0
    X[2] = 0.0

    Rot[0] = 0.0 
    Rot[1] = 0.0
    Rot[2] = -(5*pi/180)*sin(omega*t)

    return X,Rot

def PosRot_dot(t):
    '''Caclualte the time derivative of the position and rotation of
    the body as a function of time'''
    Xdot = zeros(3)
    Rotdot = zeros(3)

    Xdot[0] = -U
    Xdot[1] = 0.0
    Xdot[2] = 0.0

    Rotdot[0] = 0.0 # Roll Angle
    Rotdot[1] = 0.0 # Yaw Angle
    Rotdot[2] = -(5*pi/180)*cos(omega*t)*omega # Pitch Angle

    return Xdot,Rotdot



panel.setTrajectory(PosRot,PosRot_dot)

# Set up a forward flight condition (Steady)
# alpha = 0.0
# beta = 0.0
# Cwb = TriPan.getWindFrame( alpha, beta )
# Cbe = TriPan.getForwardFrame()
# C = dot( Cwb, Cbe )

# VInf = 100.0
# OmegaInf = 0.0
# Ry = 0.0
# Rz = 0.0
# panel.triPanel.setVelocity( VInf, OmegaInf, Ry, Rz, C.flatten() )

#panel.linearSolve()
panel.unsteadySim(2*pi*10,36)

# panel.triPanel.writeLiftDistribution( 'lift_distribution.dat', panel.computeLiftDirection(), linspace(0.01, 0.9, 75) )

# panel.triPanel.writeSectionalCp( 'cp_distZ=1.dat', 0.25 )
# panel.triPanel.writeSectionalCp( 'cp_distZ=2.dat', 0.5 )
# panel.triPanel.writeSectionalCp( 'cp_distZ=3.dat', 0.75 )

panel.writeSolution(outfile='bwb_tripan_sol.dat',wake_outfile='bwb_wake.dat')

mpiPrint('---------------------------')
mpiPrint('      pyLayout Setup'       )
mpiPrint('---------------------------')
 
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
le_list3[1] = bwb.surfs[4].getValue(0,0.5)+[2.5,0,0]
te_list3[1] = bwb.surfs[4].getValue(1,0.5)-[2.5,0,0]
le_list3[2] = bwb.surfs[4].getValue(0,1) + [2.5,0,0]
te_list3[2] = bwb.surfs[4].getValue(1,1) - [2.5,0,0]

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

le_list4[:] = bwb.surfs[6].getValueV(zeros(MAX_RIBS4),linspace(0,1,MAX_RIBS4)) + [.85,0,0]
te_list4[:] = bwb.surfs[6].getValueV(ones(MAX_RIBS4),linspace(0,1,MAX_RIBS4)) - [.5,0,0]
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

wing_box.writeTecplot('./bwb_layout.dat')
wing_box.finalize()
