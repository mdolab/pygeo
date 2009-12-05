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
from pyTACS import elements

import tacs_utils
from tacs_utils import TriPan
from tacs_utils import TACSAnalysis 

from petsc4py import PETSc

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

reduced_example = True

MAX_RIBS = 7
if reduced_example:
    MAX_RIBS = 2

le_list = array([[.2*1.67,0,0],[.2*1.67,0,2.5]])
te_list = array([[.75*1.67,0,0],[.75*1.67,0,2.5]])

domain1 = pyLayout.domain(le_list.copy(),te_list.copy())

# Spacing Parameters for Elements
span_space = 8*ones(MAX_RIBS-1)
rib_space  = 8*ones(MAX_SPARS+1) # Note the +1
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
                           span_space = span_space,rib_space=rib_space,v_space=v_space, t = 0.05 )
                           
wing_box.addSection(def1)

# # ---------- Create the Second Domain -- Last First 19 ribs

MAX_RIBS = 7
if reduced_example:
    MAX_RIBS = 2

le_list = array([[.2*1.67,0,2.5],[.2*1.67,0,10.58/2]])
te_list = array([[.75*1.67,0,2.5],[.75*1.67,0,10.58/2]])

domain2 = pyLayout.domain(le_list,te_list)

# Spacing Parameters for Elements
span_space = 8*ones(MAX_RIBS-1)
rib_space  = 8*ones(MAX_SPARS+1) # Note the +1
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
                           span_space = span_space,rib_space=rib_space,v_space=v_space, t = 0.05 )
                           
wing_box.addSection(def2)

# Create the surfaces shell objects that are not part of a struct domain:

#wing_box.addSurface(surfID=1,Nu=4,Nv=3)
#wing_box.addSurface(surfID=0,Nu=4,Nv=3)
#wing_box.addSurface(surfID=2,Nu=4,Nv=3)
#wing_box.addSurface(surfID=3,Nu=4,Nv=3)
# wing_box.addSurface(surfID=4,Nu=2,Nv=2)
# wing_box.addSurface(surfID=5,Nu=2,Nv=2)

wing_box.writeTecplot('./c172_layout.dat')

pre_con = 'mg'

if pre_con == 'asm':
    nmesh = 1
    structure, tacs = wing_box.finalize( nlevels = nmesh )
    mesh = structure.getMesh()

    tacs_assembler = TACS.TACS_PETScAssembler_Real( tacs[0] )

    mat   = PETSc.Mat()
    pcmat = PETSc.Mat()
    res = PETSc.Vec()
    vec = PETSc.Vec()

    tacs_assembler.createMat( mat )
    tacs_assembler.createMat( pcmat )
    tacs_assembler.createVec( vec )
    tacs_assembler.createVec( res )

    loadCase =0
    res.zeroEntries()        
    tacs_assembler.addDirectLoads( loadCase, res )
    res.scale(-1.0)
    tacs_assembler.assembleRes( loadCase, res )

    paramSet = TACSAnalysis.ParameterSet()
    
    if MPI != None and MPI.COMM_WORLD.size > 1:
        # Global preconditioner options
        # -----------------------------
        paramSet.kspType = TACS.TACS_PETScAssembler_Real.GMRES
        paramSet.pcType  = TACS.TACS_PETScAssembler_Real.ASM
        paramSet.asmOverlap = 0 ## Adjust this parameter
        
        paramSet.kspRestart  = 80
        paramSet.kspMaxIters = 240
        paramSet.kspTol      = 1e-8 
        
        paramSet.usePETScRCM  = 0
        paramSet.pcLevFill    = 2 ## and this one too
        paramSet.pcFillFactor = 6.0
        paramSet.pcShift      = 1.0e-3        
        
        # Set the options for the sub-preconditioner
        # ------------------------------------------       
        paramSet.subKspTol      = 0.01 # Sub tolerance
        paramSet.subKspType     = TACS.TACS_PETScAssembler_Real.PCONLY
        paramSet.subPcType      = TACS.TACS_PETScAssembler_Real.ILU
        paramSet.subRichardsonScale = 1.0
        paramSet.subKspMaxIters = 1
    else:
        paramSet.kspType = TACS.TACS_PETScAssembler_Real.GMRES
        paramSet.pcType  = TACS.TACS_PETScAssembler_Real.ILU

        paramSet.kspRestart  = 80
        paramSet.kspMaxIters = 240
        paramSet.kspTol      = 1e-8

        paramSet.pcLevFill    = 2 ## and this one too
        paramSet.pcFillFactor = 4.0
    # end

    paramSet.setFromOptions( tacs_assembler )

    ksp = PETSc.KSP()
    ksp.create()

    tf = 0.1
    tacs_assembler.assemblePCMat( loadCase, mat, tf, pcmat )
    
    ksp.setOperators( mat, pcmat, PETSc.Mat.Structure.SAME_NONZERO_PATTERN )
    tacs_assembler.setKSPFromOptions( ksp )

    ksp.view()
    ksp.setMonitor( PETSc.KSP.Monitor() )
    ksp.solve( res, vec )
    vec.scale( -1.0 )
    
    loadCase = 0
    tacs_assembler.setVariables( loadCase, vec )
    tacs_assembler.writeTecplotFile( loadCase, 'asm_p' + str(PETSc.COMM_WORLD.rank) )
    
elif pre_con == 'mg':

    nmesh = 3
    structure, tacs = wing_box.finalize( nlevels = nmesh )
    mesh = structure.getMesh()

    # Build the TACS models
    tacs_assembler = []
    for k in xrange(nmesh):
        tacs_assembler.append( TACS.TACS_PETScAssembler_Real( tacs[k] ) )
    # end

    # Set up for multi-grid
    paramSet = TACSAnalysis.ParameterSet()

    asmOver = [ 0, 1, 2 ]
    pcFill = [ 1, 2, 3 ]

    for k in xrange(nmesh):
        if MPI != None and MPI.COMM_WORLD.size > 1:
            # Global preconditioner options
            # -----------------------------
            paramSet.kspType = TACS.TACS_PETScAssembler_Real.GMRES
            paramSet.pcType  = TACS.TACS_PETScAssembler_Real.ASM
            paramSet.richardsonScale = 0.1
            paramSet.asmOverlap = asmOver[k] ## Adjust this parameter

            paramSet.kspRestart  = 10
            paramSet.kspMaxIters = 10
            paramSet.kspTol      = 1e-8 

            paramSet.usePETScRCM  = 0
            paramSet.pcLevFill    = pcFill[k] ## and this one too
            paramSet.pcFillFactor = 5.0
            paramSet.pcShift      = 1.0e-3        

            # Set the options for the sub-preconditioner
            # ------------------------------------------       
            paramSet.subKspTol      = 0.01 # Sub tolerance
            paramSet.subKspType     = TACS.TACS_PETScAssembler_Real.RICHARDSON
            paramSet.subPcType      = TACS.TACS_PETScAssembler_Real.ILU
            paramSet.subRichardsonScale = 1.0
            paramSet.subKspMaxIters = 1

            if k == nmesh-1:
                paramSet.kspType = TACS.TACS_PETScAssembler_Real.GMRES
                paramSet.kspRestart = 20
                paramSet.kspMaxIters = 20
            # end
        else:
            paramSet.kspType = TACS.TACS_PETScAssembler_Real.GMRES
            paramSet.pcType  = TACS.TACS_PETScAssembler_Real.ILU
            
            if k == 0:
                paramSet.pcLevFill = 1
                paramSet.pcFillFactor = 3.0
            elif k == 1:
                paramSet.pcLevFill = 2
                paramSet.pcFillFactor = 4.0
            elif k == 2:
                paramSet.pcLevFill = 3
                paramSet.pcFillFactor = 4.0                
            # end

            paramSet.kspRestart  = 5
            paramSet.kspMaxIters = 5
            paramSet.kspTol      = 1e-8             

            if k == nmesh-1:
                paramSet.kspType = TACS.TACS_PETScAssembler_Real.GMRES
                paramSet.kspRestart = 15
                paramSet.kspMaxIters = 15
            # end

            paramSet.usePETScRCM  = 0
            paramSet.pcShift      = 1.0e-3                    
         # end

        paramSet.setFromOptions( tacs_assembler[k] )
    # end

    # Create the vectors
    res = PETSc.Vec()
    vec = PETSc.Vec()

    tacs_assembler[0].createVec( vec )
    tacs_assembler[0].createVec( res )

    loadCase =0
    res.zeroEntries()        
    tacs_assembler[0].addDirectLoads( loadCase, res )
    res.scale(-1.0)
    tacs_assembler[0].assembleRes( loadCase, res )

    # Construct the interpolation/restriction operators
    # -------------------------------------------------
    Interp = []
    Restrict = []

    for k in xrange(nmesh-1):
        Interp.append( PETSc.Mat() )
        Restrict.append( PETSc.Mat() )

        mesh.setUpInterpolant( Interp[k], Restrict[k], tacs[k], k, tacs[k+1], k+1 )
    # end

    # Create the TACS_MG object
    mg = TACS.TACS_MG_Real( nmesh, TACS.TACS_MG_Real.W_CYCLE )
    tf = [ 0.001, 0.001, 0.001 ]

    mat   = PETSc.Mat()
    pcmat = PETSc.Mat()
    tacs_assembler[0].createMat( mat )
    tacs_assembler[0].createMat( pcmat )  

    tacs_assembler[0].assemblePCMat( loadCase, mat, tf[0], pcmat )

    #
    
    for k in xrange(nmesh):
        if k == 0:
            mg.setLevel( tacs[k], tacs_assembler[k], tf[k], Interp[k], Restrict[k], mat, pcmat )
        elif k == nmesh-1:
            mg.setLevel( tacs[k], tacs_assembler[k], tf[k], Interp[0], Restrict[0] )
        else:
            mg.setLevel( tacs[k], tacs_assembler[k], tf[k], Interp[k], Restrict[k] )
        # end
    # end

    ksp = PETSc.KSP()
    ksp.create()
    ksp.setOperators( mat, pcmat, PETSc.Mat.Structure.SAME_NONZERO_PATTERN )
    mg.setUpKSP( loadCase, ksp )
    ksp.setType( PETSc.KSP.Type.FGMRES )
    ksp.setPCSide( PETSc.PC.Side.RIGHT )
    ksp.setFromOptions()

    ksp.view()

    norm = res.norm()
    mpiPrint( 'The norm of the residual ' + str(norm) )

    ksp.setMonitor( PETSc.KSP.Monitor() )
    ksp.solve( res, vec )
    vec.scale( -1.0 )

    # tacs_assembler[0].setVariables( loadCase, vec )
    mg.setVariables( loadCase, vec )

    # res.zeroEntries()        
    # tacs_assembler[0].addDirectLoads( loadCase, res )
    # res.scale(-1.0)
    # tacs_assembler[0].assembleRes( loadCase, res )
    # mg.setVariables( loadCase, res )

    for k in xrange(nmesh):
        loadCase = 0
        tacs_assembler[k].writeTecplotFile( loadCase, 'level' + str(k) + '_p' + str(PETSc.COMM_WORLD.rank) )
    # end
