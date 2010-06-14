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

exec(import_modules('pyGeo','pyBlock','pySpline','geo_utils','mpi4py','pyLayout2'))
exec(import_modules('pyAero_problem','pyAero_flow','pyAero_reference','pyAero_geometry'))
exec(import_modules('pySUMB'))
exec(import_modules('TACS','functions','constitutive','elements'))
exec(import_modules('MultiblockMesh'))
exec(import_modules('pyAeroStruct'))
from petsc4py import PETSc
mpiPrint('------- ARW2 Example ----------------')

# ================================================================
#                   Grid File
grid_file = 'arw2_debug_nofam'
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

aeroOptions={'reinitialize':False,'CFL':1.6,'L2Convergence':1e-8,
             'L2ConvergenceRel':1e-1,
             'MGCycle':'3w','MetricConversion':1.0,'sol_restart':'no',
             'printIterations':False,'printSolTime':False,'writeSolution':False}

structOptions={'test':1}
# Setup Aero-Solver and multiblock Mesh

if flags[aeroID] == True:

    flow = Flow(name='Base Case',mach=0.8,alpha=2.0,beta=0.0,liftIndex=2)
    ref = Reference('Baseline Reference',1.0,1.0,1.0) #area,span,chord 
    geom = Geometry
    aeroProblem = AeroProblem(name='AeroStruct Test',geom=geom,flow_set=flow,ref_set=ref)
    CFDsolver = SUMB(comm=comm,init_petsc=False)

    mesh = MultiBlockMesh.MultiBlockMesh(grid_file,comm)
    mesh.initializeWarping()
    mesh.addFamilyGroup("wing",['wing_le','wing_up','wing_low','wing_tip'])
    mesh.addFamilyGroup("all")

    CFDsolver.initialize(aeroProblem,'steady',grid_file,solver_options=aeroOptions)
    CFDsolver.interface.Mesh.initializeExternalWarping(mesh.getGridDOF())

    structure = None

# end if

if flags[structID]:
    execfile('setup_structure.py')
    aeroProblem = None
    CFDsolver = None
    mesh = None
 
# # end if

AS = AeroStruct(MPI.COMM_WORLD,comm,flags,aeroOptions=aeroOptions, \
                structOptions=structOptions)

AS.initialize(aeroProblem,CFDsolver,structure,mesh,'wing')
timeA = time.time()
AS.solve(nMDiterations=15)
print 'MDA time:',time.time()-timeA
sys.exit(0)

