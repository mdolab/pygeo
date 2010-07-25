# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross,shape,alltrue,loadtxt,max,min,\
    take, put

# =============================================================================
# Extension modules
# =============================================================================

from mdo_import_helper import *
import warnings
warnings.filterwarnings("ignore")

# Import PETSc so it is initialized on ALL Procs
import petsc4py 
petsc4py.init(args=sys.argv)

# MDO_Lab Imports
exec(import_modules('pyGeo','pyBlock','pySpline','geo_utils','mpi4py','pyLayout2'))
exec(import_modules('RefAxis'))
exec(import_modules('pyAero_problem','pyAero_reference','pyAero_geometry'))
exec(import_modules('pyAeroStruct'))
exec(import_modules('pyOpt_Optimization','pySNOPT_MPI'))
     
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

mpiPrint('------- Sweep Example ----------------')

# ================================================================
#                   Grid File
grid_file = 'sweep_debug'
#grid_file = 'sweep_400k'
#
# ================================================================
#        Set the number of processors for Aero and Structures
npAero = MPI.COMM_WORLD.size -1 
npStruct =1
comm,flags,cumGroups = createGroups([npAero,npStruct],noSplit=False)
aeroID = 0
structID = 1
#comm = MPI.COMM_WORLD
#flags = [True]
# ================================================================
#               Set Options for each solver
#
aeroOptions={'reinitialize':False,'CFL':.80,
             'L2Convergence':1e-6,'L2ConvergenceRel':1e-6,
             'MGCycle':'4w','MetricConversion':1.0,'sol_restart':'no',
             'printIterations':True,'printSolTime':False,'writeSolution':False,
             'Dissipation Coefficients':[0.30,0.020],
             'Dissipation Scaling Exponent':0.125,
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
             'ASM Overlap':6,
             'Reference Pressure':24540.0,
             'Reference Density':.38857
             }            

structOptions={'PCFillLevel':7,
               'PCFillRatio':9.5,
               'msub':10,
               'subSpaceSize':140,
               'nRestarts':15,
               'flexible':1,
               'L2Convergence':1e-11,
               'L2ConvergenceRel':1e-3,
               'useMonitor':False,
               'monitorFrequency':1,
               'filename':'wing_box',
               'restart':False}

mdOptions = {'relTol':1e-5,
             'writeIterationVolSolutionCFD':False,
             'writeIterationSurfSolutionCFD':False,
             'writeIterationSolutionFEA':False,
             'writeVolSolutionCFD':False,
             'writeSurfSolutionCFD':True,
             'writeSolutionFEA':False,
             'writeMDConvergence':False,
             'MDConvergenceFile':'mdconverg.txt',
             'beta0':0.25,
             'CFDIter':750,
             }

meshOptions = {'warpType':'solid',
               'solidWarpType':'n', # or 'topo'
               'n':3,
               'sym':'xy'
               }

# Snopt Opt Options
optOptions = {'Major feasibility tolerance':1.0e-3,
              'Major optimality tolerance':1.0e-3, 
              'Minor feasibility tolerance':1.0e-2,
              'Verify level':0,			
              'Scale tolerance':0.9,		
              'Major iterations limit':50,
              'Minor iterations limit':500,
              'Major step limit':1.0,
              'Derivative level':0,	     
              'Nonderivative linesearch':None,
              'Function precision':1.0e-5,
              'Difference interval':.003, 	# Function precision^(1/2)
              'Central difference interval':.02# Function precision^(1/3)
}

# ================================================================
#               Setup the Free-Form Deformation Volume
FFD = pyBlock.pyBlock('plot3d',file_name='ffd_linear.fmt',file_type='ascii',order='f')
FFD.doConnectivity('ffd_connectivity.con')
FFD.fitGlobal()
FFD.coef *= 2.33 
FFD._updateVolumeCoef()

# Setup curves for ref_axis
z = array([0,5.1])*2.33
y = array([0,0])*2.33
x = array([.5,.5])*2.33
c1 = pySpline.curve(x=x,y=y,z=z,k=2)
ref_axis = RefAxis.RefAxis([c1],FFD.coef,rot_type=5)

# ================================================================
# Set Design Variable Functions

def sweep(val,ref_axis):
    
    # Do a proper rotation sweep, but still shear the sections
    
    # Extract coef for 0-th ref axis
    C = zeros((len(ref_axis.topo.l_index[0]),3))
    C[:,0] = take(ref_axis.coef[:,0],ref_axis.topo.l_index[0])
    C[:,1] = take(ref_axis.coef[:,1],ref_axis.topo.l_index[0])
    C[:,2] = take(ref_axis.coef[:,2],ref_axis.topo.l_index[0])

    M = rotyM(val)
    C[1] = C[0] + dot(M,C[1]-C[0])

    # Now set the y-rotations
    ref_axis.rot_y[0].coef[1] = val

    # Scale the Root
    ref_axis.scale_x[0].coef[0] = ref_axis.scale_x0[0].coef[0] / cos(val*pi/180)


    # Reset the coef
    put(ref_axis.coef[:,0],ref_axis.topo.l_index[0],C[:,0])
    put(ref_axis.coef[:,1],ref_axis.topo.l_index[0],C[:,1])
    put(ref_axis.coef[:,2],ref_axis.topo.l_index[0],C[:,2])
    
    return 

def twist(val,ref_axis):

    ref_axis.rot_z[0].coef[1] = val
    
    return

# Fifth: Add design variables (global) to the ref axis object

ref_axis.addGeoDVGlobal('sweep',0,0,40,sweep)
ref_axis.addGeoDVGlobal('twist',0,-10,10,twist)
idg = ref_axis.DV_namesGlobal #NOTE: This is constant (idg -> id global

# =====================================================
#        Setup Aero-Solver and multiblock Mesh

if flags[aeroID] == True:
    flow = Flow(name='Base Case',mach=0.8,alpha=0,beta=0.0,liftIndex=2)
    ref = Reference('Baseline Reference',5.0,5.0,1.0) #area,span,chord 
    geom = Geometry
    aeroProblem = AeroProblem(name='AeroStruct Test',geom=geom,flow_set=flow,ref_set=ref)
    CFDsolver = SUMB(comm=comm,init_petsc=False,options=aeroOptions)
    mesh = MBMesh(grid_file,comm,meshOptions=meshOptions)
    mesh.addFamilyGroup("wing")
    mesh.warpMesh()
    CFDsolver.initialize(aeroProblem,'steady',grid_file)
    CFDsolver.interface.Mesh.initializeExternalWarping(mesh.getGridDOF())
    structure = None

    coords = mesh.getSurfaceCoordinatesLocal('wing')
    FFD.embedVolume(coords)
    
# end if

# =====================================================
#                Setup Struct Solver

if flags[structID]:
    execfile('setup_structure.py') # structure comes out of this
    aeroProblem = None
    CFDsolver = None
    mesh = None
    mass = structure.evalFunction(mass_func)
    ks   = structure.evalFunction(ks_func)

    Nnodes = structure.getNumVariables()
    coords = empty(3*Nnodes,'d')
    structure.getNodes(coords)
    coords = coords.reshape((Nnodes,3))
    FFD.embedVolume(coords)
# end if


# =====================================================
#            Setup Aero-Struct Solver

AS = AeroStruct(MPI.COMM_WORLD,comm,flags,aeroOptions=aeroOptions, 
                structOptions=structOptions,mdOptions=mdOptions,complex=complex)

AS.initialize(aeroProblem,CFDsolver,structure,mesh,'wing')
# =====================================================
#                Objective Function

def fun_obj(x):

    # =====================================================
    #        Set DV's

    # Sclale DV's back to range
    alpha = x[0]*6-3
    sweep = x[1]*45 + 0
    twist = x[2]*20 - 10

    if MPI.COMM_WORLD.rank == 0:
        print 'DVs are:'
        print 'alpha:',alpha
        print 'sweep:',sweep
        print 'twist:',twist
    # Set up the Flow and aeroProblem again 
    flow = Flow(name='Base Case',mach=0.8,alpha=alpha,beta=0.0,liftIndex=2)
    ref = Reference('Baseline Reference',5.0,5.0,1.0) #area,span,chord 
    geom = Geometry
    AS.aero_problem = AeroProblem(name='AeroStruct Test',geom=geom,flow_set=flow,ref_set=ref)

    # Set Sweep and Twist
    ref_axis.DV_listGlobal[idg['sweep']].value = sweep
    ref_axis.DV_listGlobal[idg['twist']].value = twist

    # Update FFD
    FFD.coef = ref_axis.update()
    FFD._updateVolumeCoef()

    # Update CFD Grid
    if flags[aeroID] == True:
        mesh.setSurfaceCoordinatesLocal('wing',FFD.getVolumePoints(0))
        mesh.warpMesh()
        CFDsolver.interface.Mesh.setGrid(mesh.getGrid()) 

    # Update Structure Grid
    if flags[structID] == True:
        structure.setNodes(FFD.getVolumePoints(0).flatten())

    # Solve the aerostructural Problem
    try:
        Force = AS.solve(nMDiterations=1)
        M = rotzM(-x[0])
        Force = dot(M,Force)
    
        Lift = Force[1]
        Drag = Force[0]

        f_obj = Drag/1000
        f_con = array([Lift])/95870.0
    
        fail = 0
    except:
        fail = 1
        f_obj = 0.0
        f_con = [0.0]


    if MPI.COMM_WORLD.rank == 0:
        print 'x',x
        print 'f_obj:',f_obj
        print 'f_con:',f_con




    return f_obj,f_con,fail



# =====================================================
# Set-up Optimization Problem

opt_prob = Optimization('Swept Wing Optimization',fun_obj)

# Add variables
opt_prob.addVar('alpha',value=0.,lower=0.,upper=1.)
opt_prob.addVar('sweep',value=.1,lower=0.,upper=1.)
opt_prob.addVar('twist',value=.5,lower=0.,upper=1.)

# Add Constraints
opt_prob.addCon('lift', type='i', lower=1.0, upper=10.0)

# Add Objective 
opt_prob.addObj('Drag')

# Make Instance of Optimizer
snopt = SNOPT(options=optOptions)

# Run optimization
snopt(opt_prob)
#fun_obj([0,.1,0.5])
#fun_obj([0,.1,0.5])
