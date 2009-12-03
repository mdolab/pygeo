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
from tacs_utils import TACSAnalysis 

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
c172 = pyGeo.pyGeo('iges',file_name='./c172.igs')
c172.readEdgeConnectivity('c172.con')
c172.setSymmetry('xy')
mpiPrint('---------------------------')
mpiPrint('Attaching Reference Axis...')
mpiPrint('---------------------------')

# Add a single reference axis:

ref_axis_x = array([[0,0,0],[0,0,5.4]])
rot = zeros_like(ref_axis_x)
c172.addRefAxis([0,1,2,3,4,5],ref_axis_x,rot,nrefsecs=5)
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
c172.addGeoDVGlobal('twist',zeros(ntwist),-10,10,twist)
c172.calcCtlDeriv()


mpiPrint('---------------------------')
mpiPrint('      pyLayout Setup'       )
mpiPrint('---------------------------')
 
MAX_SPARS =2  # This is the same for each spanwise section
Nsection = 2
wing_box = pyLayout.Layout(c172,Nsection,MAX_SPARS)

# # ---------- Create the First Domain -- First 7 ribs

MAX_RIBS = 7

le_list = array([[.2*1.67,0,0],[.2*1.67,0,2.5]])
te_list = array([[.75*1.67,0,0],[.75*1.67,0,2.5]])

domain1 = pyLayout.domain(le_list.copy(),te_list.copy())

# Spacing Parameters for Elements
span_space = 4*ones(MAX_RIBS-1)
rib_space  = 6*ones(MAX_SPARS+1) # Note the +1
rib_space[0] = 4
rib_space[2] = 4
v_space    = 4

rib_blank = ones((MAX_RIBS,MAX_SPARS-1))
rib_blank[0,:] = 0

surfs = [[0],[1]] #Upper surfs for LE to TE then Lower Surfs from LE to TE
spar_con = [0,0]

def1 = pyLayout.struct_def(MAX_RIBS,MAX_SPARS,domain1,surfs,spar_con,
                           rib_pos_para=linspace(0,1,MAX_RIBS),spar_pos_para=linspace(0,1,MAX_SPARS),
                           rib_blank = rib_blank,
                           span_space = span_space,rib_space=rib_space,v_space=v_space)
                           
wing_box.addSection(def1)

# # ---------- Create the Second Domain -- Last First 19 ribs

MAX_RIBS = 7

le_list = array([[.2*1.67,0,2.5],[.2*1.67,0,10.58/2]])
te_list = array([[.75*1.67,0,2.5],[.75*1.67,0,10.58/2]])

domain2 = pyLayout.domain(le_list,te_list)

# Spacing Parameters for Elements
span_space = 4*ones(MAX_RIBS-1)
rib_space  = 6*ones(MAX_SPARS+1) # Note the +1
rib_space[0] = 4
rib_space[2] = 4
v_space    = 4

surfs = [[2],[3]] #Upper surfs for LE to TE then Lower Surfs from LE to TE
spar_con = [2,2]
spar_blank = ones((MAX_SPARS,MAX_RIBS-1))
#spar_blank[1,-3:] = 0
def2 = pyLayout.struct_def(MAX_RIBS,MAX_SPARS,domain2,surfs,spar_con,
                           rib_pos_para=linspace(0,1,MAX_RIBS),spar_pos_para=linspace(0,1,MAX_SPARS),
                           spar_blank = spar_blank,
                           span_space = span_space,rib_space=rib_space,v_space=v_space)
                           
wing_box.addSection(def2)

# Create the surfaces shell objects that are not part of a struct domain:

#wing_box.addSurface(surfID=1,Nu=4,Nv=3)
#wing_box.addSurface(surfID=0,Nu=4,Nv=3)
#wing_box.addSurface(surfID=2,Nu=4,Nv=3)
#wing_box.addSurface(surfID=3,Nu=4,Nv=3)
# wing_box.addSurface(surfID=4,Nu=2,Nv=2)
# wing_box.addSurface(surfID=5,Nu=2,Nv=2)

wing_box.writeTecplot('./c172_layout.dat')
tacs = wing_box.finalize()

paramSet = TACSAnalysis.ParameterSet()
    
if MPI.COMM_WORLD.size > 1:
    # Global preconditioner options
    # -----------------------------
    restart = 120
    paramSet.kspType = TACS.TACS_PETScAssembler_Real.GMRES
    paramSet.pcType  = TACS.TACS_PETScAssembler_Real.ASM
    paramSet.asmOverlap = 1 ## Adjust this parameter
    
    paramSet.kspRestart  = restart
    paramSet.kspMaxIters = 3 * restart
    paramSet.kspTol      = 1e-8 
    
    paramSet.usePETScRCM  = 0
    paramSet.pcLevFill    = 5 ## and this one too
    paramSet.pcFillFactor = 8.0
    paramSet.pcShift      = 1.0e-3        
    
    # Set the options for the sub-preconditioner
    # ------------------------------------------       
    paramSet.subKspTol      = 0.01 # Sub tolerance
    paramSet.subKspType     = TACS.TACS_PETScAssembler_Real.PCONLY
    paramSet.subPcType      = TACS.TACS_PETScAssembler_Real.ILU
    paramSet.subKspMaxIters = 1
# end

tacsAnalysis = TACSAnalysis.TACSAnalysis( tacs, paramSet, './layout' )
tacsAnalysis.solve( tf = 0.1 )

tacsAnalysis.writeFile()




