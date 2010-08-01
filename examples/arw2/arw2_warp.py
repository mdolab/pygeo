# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross,shape,alltrue,loadtxt,max,min

# =============================================================================
# Extension modules
# =============================================================================

from mdo_import_helper import *
import warnings
warnings.filterwarnings("ignore")
import petsc4py
from petsc4py import PETSc

exec(import_modules('pyGeo','pyBlock','pySpline','geo_utils','mpi4py','pyLayout2'))
exec(import_modules('pyAero_problem','pyAero_reference','pyAero_geometry'))
exec(import_modules('pyAeroStruct'))
if 'complex' in sys.argv:
    exec(import_modules('pyAero_flow_C'))
    exec(import_modules('pySUMB_C','MultiblockMesh_C'))
    exec(import_modules('TACS_cs','functions_cs','constitutive_cs','elements_cs'))
    TACS = TACS_cs
    functions = functions_cs
    constitutive = constitutive_cs
    elements = elements_cs
    SUMB = SUMB_C
    MBMesh = MBMesh_C
    complex = True
else:
    # Real Imports
    exec(import_modules('pyAero_flow'))
    exec(import_modules('pySUMB','MultiblockMesh'))
    exec(import_modules('TACS','functions','constitutive','elements'))
    complex = False
# end if

mpiPrint('------- ARW2 Example ----------------')

# ================================================================
#                   Grid File
grid_file = 'arw2_debug2'
#grid_file = 'arw2_700k'
#
# ================================================================
#        Set the number of processors for Aero and Structures
npAero = 3
npStruct =1
comm,flags,cumGroups = createGroups([npAero,npStruct],noSplit=False)
aeroID = 0
structID = 1
#
# ================================================================
#               Setup the Free-Form Deformation Volume
FFD = pyBlock.pyBlock('plot3d',file_name='arw2_ffd.fmt',file_type='ascii',order='f')
FFD.doConnectivity('ffd_connectivity.con')
FFD.fitGlobal()
FFD.coef *= 0.0254 # Just a hack
FFD._updateVolumeCoef()
# ================================================================
#               Set Options for each solver
#

aeroOptions={'reinitialize':False,'CFL':1.75,
             'L2Convergence':1e-11,'L2ConvergenceRel':1e-1,
             'MGCycle':'2v','MetricConversion':1.0,'sol_restart':'no',
             'printIterations':False,'printSolTime':False,'writeSolution':False,
             'Dissipation Coefficients':[0.25,0.015625],
             'Dissipation Scaling Exponent':0.87,
             'Approx PC': 'yes',
             'Adjoint solver type':'GMRES',
             'adjoint relative tolerance':1e-8,
             'adjoint absolute tolerance':1e-16,
             'adjoint max iterations': 500,
             'adjoint restart iteration' : 100,
             'adjoint monitor step': 20,
             'dissipation lumping parameter':6.0,
             'Preconditioner Side':'LEFT',
             'Matrix Ordering': 'NestedDissection',
             'Global Preconditioner Type': 'Additive Schwartz',
             'Local Preconditioner Type' : 'ILU',
             'ILU Fill Levels': 2,
             'ASM Overlap':6
             }            

structOptions={'PCFillLevel':5,
               'PCFillRatio':6.0,
               'msub':10,
               'subSpaceSize':140,
               'nRestarts':15,
               'flexible':1,
               'L2Convergence':1e-11,
               'L2ConvergenceRel':1e-2,
               'useMonitor':False,
               'monitorFrequency':1,
               'filename':'wing_box',
               'restart':False}

mdOptions = {'relTol':1e-9,
             'writeIterationVolSolutionCFD':False,
             'writeIterationSurfSolutionCFD':False,
             'writeIterationSolutionFEA':False,
             'writeVolSolutionCFD':False,
             'writeSurfSolutionCFD':True,
             'writeSolutionFEA':True,
             'writeMDConvergence':True,
             'MDConvergenceFile':'mdconverg.txt',
             'beta0':0.5,
             'CFDIter':50,
             }

meshOptions = {
    'warpType':'solid',
    'solidWarpType':'n', # or 'topo'
    'n':3,
    'sym':'xy'
    }

# Setup Aero-Solver and multiblock Mesh

if flags[aeroID] == True:
    flow = Flow(name='Base Case',mach=0.8,alpha=6.0,beta=0.0,liftIndex=2)
    ref = Reference('Baseline Reference',1.0,1.0,1.0) #area,span,chord 
    geom = Geometry
    aeroProblem = AeroProblem(name='AeroStruct Test',geom=geom,flow_set=flow,ref_set=ref)
    CFDsolver = SUMB(comm=comm,init_petsc=False,options=aeroOptions)
    mesh = MBMesh(grid_file,comm,meshOptions=meshOptions)
    mesh.addFamilyGroup("wing",['wing_le','wing_up','wing_low','wing_tip'])
    mesh.addFamilyGroup("all")
    mesh.warpMesh()
    CFDsolver.initialize(aeroProblem,'steady',grid_file)
    CFDsolver.interface.Mesh.initializeExternalWarping(mesh.getGridDOF())
    structure = None
# end if

if flags[structID]:
    execfile('setup_structure.py')
    aeroProblem = None
    CFDsolver = None
    mesh = None
    mass = structure.evalFunction(mass_func)
    #ks   = structure.evalFunction(ks_func)
    
    print 'Mass is:',mass
    #print 'KS is:',ks

# end if

AS = AeroStruct(MPI.COMM_WORLD,comm,flags,aeroOptions=aeroOptions, 
                structOptions=structOptions,mdOptions=mdOptions,complex=complex)

AS.initialize(aeroProblem,CFDsolver,structure,mesh,'wing')
AS.solve(nMDiterations=1)
#AS.initializeCoupledAdjoint()
#AS.solveCoupledAdjoint('cl')

