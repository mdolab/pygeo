# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time, datetime

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross,shape,alltrue,loadtxt,max,min,\
    take, put,log

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
#                   INPUT INFORMATION
grid_file = 'sweep_cgrid_debug'
output_directory = '.'
problem_name = 'sweep_opt'

# ================================================================
#                   Create Working Directoy
problem_dir = output_directory + '/' + problem_name + '_'+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

if MPI.COMM_WORLD.rank == 0:
    os.mkdir(problem_dir)
print problem_dir
#
# ================================================================
#        Set the number oBf processors for Aero and Structures
npAero = MPI.COMM_WORLD.size -1 
npStruct =1
comm,flags,cumGroups = createGroups([npAero,npStruct],noSplit=False)
aeroID = 0
structID = 1

# ================================================================
#               Set Options for each solver
#
aeroOptions={'reinitialize':False,'CFL':1.0, # 1.5 for large grid
             'L2Convergence':1e-7,'L2ConvergenceRel':1e-1,
             'MGCycle':'3w','MetricConversion':1.0,'sol_restart':'no',
             'printIterations':False,'printSolTime':False,'writeSolution':False,
             'Dissipation Coefficients':[0.25,0.0156], 
             'Dissipation Scaling Exponent':0.3, # .25 for large grid
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
               'restart':False}

mdOptions = {'relTol':1e-3,
             'writeIterationVolSolutionCFD':False,
             'writeIterationSurfSolutionCFD':False,
             'writeIterationSolutionFEA':False,
             'writeVolSolutionCFD':True,
             'writeSurfSolutionCFD':True,
             'writeSolutionFEA':True,
             'saveIterations':True, 
             'writeMDConvergence':True,
             'MDConvergenceFile':'mdconverg.txt',
             'beta0':.5,
             'CFDIter':500,
             'outputDir':problem_dir,
             'problemName':problem_name
             }

meshOptions = {'warpType':'solid',
               'solidWarpType':'n', # or 'topo'
               'n':4,
               'sym':'xy'
               }

# Snopt Opt Options
optOptions = {'Major feasibility tolerance':1.0e-3,
              'Major optimality tolerance':1.00e-3, 
              'Verify level':-1,			
              'Major iterations limit':50,
              'Minor iterations limit':500,
              'Major step limit':.1,
              'Derivative level':0,	     
              'Nonderivative linesearch':None,
              'Function precision':1.0e-5,
              'Print file':problem_dir+'/SNOPT_print.out',
              'Summary file':problem_dir+'/SNOPT_summary.out',
              'Problem Type':'Maximize'
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
Area_ref = 5.0*2.33*2.33
Span_ref = 5.0*2.33
Chord_ref = 1.0*2.33
# ================================================================
# Set Design Variable Functions

def sweep(val,ref_axis):
    
    # Do a proper rotation sweep, but still shear the sections
    
    # Extract coef for 0-th ref axis
    C = zeros((len(ref_axis.topo.l_index[0]),3))
    C[:,0] = take(ref_axis.coef0[:,0],ref_axis.topo.l_index[0])
    C[:,1] = take(ref_axis.coef0[:,1],ref_axis.topo.l_index[0])
    C[:,2] = take(ref_axis.coef0[:,2],ref_axis.topo.l_index[0])

    d0 = norm(C[1]-C[0])
    C[1][0] += d0*tan(val*pi/180)
    
    #print 'C[1]:',C[1]
    #M = rotyM(val)
    #C[1] = C[0] + dot(M,C[1]-C[0])

    # Now set the y-rotations
    #ref_axis.rot_y[0].coef[1] = val

    # Scale the Root
    #ref_axis.scale_x[0].coef[0] = ref_axis.scale_x0[0].coef[0] / cos(val*pi/180)


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
    ref = Reference('Baseline Reference',Area_ref,Span_ref,Chord_ref) #area,span,chord 
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
    alpha = x[0]*20-10
    sweep = x[1]*45
    twist = x[2]*10 - 5
    thick1 = .04
    thick2 = x[3]
    r = MPI.COMM_WORLD.rank
    print 'DVs are:'
    print 'x',r,x
    print 'alpha:',r,alpha
    print 'sweep:',r,sweep
    print 'twist:',r,twist
    print 'thick1:',r,thick1
    print 'thick2:',r,thick2

    # Set Sweep and Twist
    ref_axis.DV_listGlobal[idg['sweep']].value = sweep
    ref_axis.DV_listGlobal[idg['twist']].value = twist

    # Update FFD
    FFD.coef = ref_axis.update()
    FFD._updateVolumeCoef()

    # Update Structures
    if flags[structID]:
        structure.setDesignVars([thick1,thick2])
        structure.setNodes(FFD.getVolumePoints(0).flatten())
    # Update Aero
    if flags[aeroID]: 
        # Set up the Flow and aeroProblem again 
        flow = Flow(name='Base Case',mach=0.8,alpha=alpha,beta=0.0,liftIndex=2)
        ref = Reference('Baseline Reference',Area_ref,Span_ref,Chord_ref) #area,span,chord 
        geom = Geometry
        AS.aero_problem = AeroProblem(name='AeroStruct Test',geom=geom,flow_set=flow,ref_set=ref)

        mesh.setSurfaceCoordinatesLocal('wing',FFD.getVolumePoints(0))
        mesh.warpMesh()
        mesh.writeVolumeGrid('fuck.cgns')
        CFDsolver.interface.Mesh.setGrid(mesh.getGrid()) 
    # end if

    # Solve the aerostructural Problem
    Force,fail = AS.solve(nMDiterations=3)

    if fail:
        fail = 1
        f_obj = 0.0
        f_con = [0.0,0.0]

        return f_obj,f_con,fail
    # end try

    if flags[structID]: #bcast mass and ks 
        mass = structure.evalFunction(mass_func)
        ks   = structure.evalFunction(ks_func)
    else:
        mass = None
        ks = None
    # end if

    mass = MPI.COMM_WORLD.bcast(mass,root=AS.sroot)
    ks   = MPI.COMM_WORLD.bcast(ks,root=AS.sroot)
    
    M = rotzM(-alpha)
    Force = dot(M,Force)
    
    Lift = Force[1]
    Drag = Force[0]
    f_con = zeros([2],'d')

    #f_obj = (Lift/Drag)*log(W1/W2)
    # OEW = 11791 kg
    # MTOW = 19,500 kg
    # Wing Weight = 2358 kg (nominal 20% of OEW)
    # Sturct Weight = 9432 kg
    # Passenger = 5245 kg
    # Fuel = 2464 kg

    # W1 = MTOW = OEW + Pssenger + Fuel
    # W1 = MTOW = Struct_weight + wing_weight + Passenger + Fuel
    W1 = 2358 + mass*2 + 5245 + 2464

    # W2 = Final Weight = Struct_weight + wing_weight + Passenger
    W2 = 2358 + mass*2 + 5245
    
    f_obj = Lift/Drag*log(W1/W2)

    # Lift Constraint:
    # Lift = Weight = MTOW = W1 = Stuct_wiehgt + wing_weight +passenger + fuel
    # Scale my original MTOW
    f_con[0] = ((2*Lift/9.81) - (9432 + mass*2 + 5245 + 2464) )/19500
    f_con[1] = ks

    #f_obj = Drag
    # f_obj = Drag/1000
    #f_con = array([Lift])/95870.0
   
    fail = 0

    if MPI.COMM_WORLD.rank == 0:
        print 'mass:',mass
        print 'f_obj:',f_obj
        print 'f_con:',f_con

    return f_obj,f_con,fail

# =====================================================
# Set-up Optimization Problem

opt_prob = Optimization('Swept Wing Optimization',fun_obj)

# Add variables
opt_prob.addVar('alpha',value=0.453098,lower=0.,upper=1.)
opt_prob.addVar('sweep',value=.25,lower=0.,upper=1.)
opt_prob.addVar('twist',value=.5,lower=0.,upper=1.)
#opt_prob.addVar('thick1',value=.05,lower=.005,upper=1)
opt_prob.addVar('thick2',value=0.01,lower=0.005,upper=1)
# Add Constraints
opt_prob.addCon('lift', type='e', value=0.0)
opt_prob.addCon('KS',type='i',lower=0.0,upper=10.0)
# Add Objective 
opt_prob.addObj('Range')

# Make Instance of Optimizer
#snopt = SNOPT(options=optOptions)

# Run optimization
#print opt_prob
#snopt(opt_prob)
#r = MPI.COMM_WORLD.rank
#print 'rank:',r,opt_prob

fun_obj([.6,.8,0.8,.005])
fun_obj([.7,.8,0.8,.005])
fun_obj([.8,.8,0.8,.005])

#fun_obj([.501,0,.00500])
#fun_obj([.50,0.001,.00500])
#fun_obj([.50,0.000,.00510])
# if MPI.COMM_WORLD.rank == 0:
#     f = open('/scratch/kenway/alpha_sweep.txt','w')
#     f.write('Lift           Drag               KS\n')
# for i in xrange(10):
#     alpha_dv = i/10.0
#     f_obj,f_con,fail = fun_obj([alpha_dv,0.0,.08,.02])
#     if MPI.COMM_WORLD.rank == 0:
#         f.write('%20.15e %20.15e %20.15e \n'%(f_con[0],f_obj,f_con[1]))
#     # end if
# # end for
# if MPI.COMM_WORLD.rank == 0:
#     f.close()
                
