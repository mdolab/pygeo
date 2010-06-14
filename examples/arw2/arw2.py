# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross,shape,alltrue,loadtxt

# =============================================================================
# Extension modules
# =============================================================================

from mdo_import_helper import *
import warnings
warnings.filterwarnings("ignore")

exec(import_modules('pyGeo','pyBlock','pySpline','geo_utils','mpi4py','pyLayout2'))
exec(import_modules('pyAero_problem','pyAero_flow','pyAero_reference','pyAero_geometry'))
exec(import_modules('pySUMB','pyDummyMapping'))
exec(import_modules('TACS','functions','constitutive','elements'))

mpiPrint('------- ARW2 Example ----------------')

# ================================================================
#        Set the number of processors for Aero and Structures
nProc_aero   = 3
nProc_struct = 1
split = True
# ================================================================
#                     Setup mpi groups
if split:
    nProc_total  = nProc_aero + nProc_struct
    if MPI.COMM_WORLD.size < (nProc_aero + nProc_struct):
        mpiPrint('Error: This script must be run with at least %d processors.'%(nProc_aero+nProc_struct))
        sys.exit(0)
    # end if

    if MPI.COMM_WORLD.rank < nProc_aero:
        member_key = 0
        is_aero = True
        is_struct = False
    elif MPI.COMM_WORLD.rank >= nProc_aero and MPI.COMM_WORLD.rank < nProc_total:
        member_key = 1
        is_aero = False
        is_struct = True
    else:
        member_key = 2 # Idle processors
    # end if
    comm = MPI.COMM_WORLD.Split(member_key)
    aroot = 0 
    sroot = nProc_aero
else:
    comm = MPI.COMM_WORLD
    is_aero   = True
    is_struct = True
# end if
#
# ================================================================
#               Setup the Free-Form Deformation Volume
FFD = pyBlock.pyBlock('plot3d',file_name='arw2_ffd.fmt',file_type='ascii',order='f')
FFD.doConnectivity('ffd_connectivity.con')
FFD.fitGlobal()
FFD.coef *= 0.0254 # Just a cha
FFD._updateVolumeCoef()
FFD.writeTecplot('ffd.dat')

# ================================================================
# Setup Aero-Solver
if is_aero == True:
    flow = Flow(name='Base Case',mach=0.8,alpha=2.0,beta=0.0,liftIndex=2)
    ref = Reference('Baseline Reference',1.0,1.0,1.0) #area,span,chord 
    geom = Geometry
    test_case = AeroProblem(name='Simple Test',geom=geom,flow_set=flow,ref_set=ref)
    solver = SUMB(comm=comm,init_petsc=True)
else:
    MPI.COMM_WORLD.Barrier()
# end if

# ================================================================
#                   Embed the aero-coordinates 
#

wing_families = ['wing_le','wing_low','wing_up','wing_tip']
#wing_families = ['default']
if is_aero:
    solver_options={'reinitialize':True,'CFL':1.6,'L2Convergence':1e-3,
                    'MGCycle':'3w','MetricConversion':1.0,'sol_restart':'no'}

    try:
        solver(test_case,niterations=1000,grid_file='arw2_debug_fam',solver_options=solver_options)
    except:
        pass
    # end try
    cfd_surface_points = solver.interface.Mesh.getSurfaceCoordinatesLocal(wing_families)
    cfd_surface_points0 = cfd_surface_points.copy()
    cfd_forces = solver.interface.Mesh.getSurfaceForcesLocal(wing_families)
    FFD.embedVolume(cfd_surface_points)
    solver.interface.initializeADjoint()
    pts = cfd_surface_points
    f = open('mdpts%d.dat'%(comm.rank),'w')
    f.write ('VARIABLES = "X", "Y","Z","Fx","Fy","Fz"\n')
    f.write('Zone T=\"%s\" I=%d\n'%('cfdforces',len(pts)))
    f.write('DATAPACKING=POINT\n')
    for i in xrange(len(pts)):
        f.write('%f %f %f %f %f %f\n'%(pts[i,0],pts[i,1],pts[i,2],cfd_forces[i,0],cfd_forces[i,1],cfd_forces[i,2]))
    # end for
    f.close()

# ================================================================
#         Use pyLayout to setup the structure and then embed
#

if is_struct:

    geo_surface = pyGeo.pyGeo('plot3d',file_name='arw2_surface.xyz',file_type='ascii',order='f')
    geo_surface.doConnectivity('arw2.con')
    geo_surface.fitGlobal()
    # Now setup the structure
    mpiPrint('---------------------------',comm=comm)
    mpiPrint('      pyLayout Setup' ,comm=comm)
    mpiPrint('---------------------------',comm=comm)

    wing_box = pyLayout2.Layout(geo_surface,[67,26],scale=.0254,comm=comm,
                                surf_list=[4,6,8,12,10,17,28,30,31,26,29,27])

    le_list = array([[15.794,59,12.5],[65.7,59,113.5]])
    te_list = array([[24.7,59,12.5],[70.4,59,113.5]])

    nrib = 18
    nspar = 2
    domain = pyLayout2.domain(le_list,te_list,k=2)

    rib_space = [3,2,2]
    span_space = 2*ones(nrib-1,'intc')
    v_space = 2

    struct_def1 = pyLayout2.struct_def(
        nrib,nspar,domain=domain,t=1.0,
        rib_space=rib_space,span_space=span_space,v_space=v_space)


    wing_box.addSection(struct_def1)
    structure = wing_box.finalize2()
    structure.writeTecplotFile(0,'wing_box_tacs')
# end if

# Make sure everything is caught up

MPI.COMM_WORLD.Barrier()

# ================================================================
#               Setup Load Transfer Object


if (is_aero):
    aero_member_flag = 1
    transfer = TACS.LDTransfer(MPI.COMM_WORLD, sroot, aroot, comm, aero_member_flag, cfd_surface_points.flatten())
    
if (is_struct): 
    aero_member_flag = 0
    transfer = TACS.LDTransfer(MPI.COMM_WORLD, sroot, aroot, comm, aero_member_flag, [], structure,'tacs_file_name.dat')

nstruct,naero = transfer.getSizes()

if (is_struct):
    cfd_forces = empty((0,3),dtype='d')
    cfd_surface_points = empty((0,3),dtype='d')
# end if

struct_forces = empty((nstruct,3),dtype='d')
struct_u      = empty((nstruct,3),dtype='d')

MPI.COMM_WORLD.Barrier()
print 'myid,cfd,struct', MPI.COMM_WORLD.rank,cfd_forces.shape,struct_forces.shape
transfer.reverse3D(cfd_forces, struct_forces)

if (is_struct):
    rhs = structure.createVec()
    x   = structure.createVec()
    K   = structure.createMat()

    structure.assembleMat(0,K,rhs)
    structure.addAeroForces3D( struct_forces, rhs, transfer )

    mpiPrint('The magic initial norm is %10.5e'%(rhs.norm()),comm=comm)

    PC = TACS.ApproximateSchur(K,4,8.0,10)
    PC.factor()
    KSM = TACS.GMRES(K,PC,120,15,1)
    KSM.setTolerances(1e-12,1e-30)
    KSM.solve(rhs,x)
    x.scale(-1.0)
    structure.setVariables(0 , x )
    
    structure.assembleRes(0,rhs)
    structure.addAeroForces3D( struct_forces, rhs, transfer )
    mpiPrint('The magic norm is %10.5e'%(rhs.norm()),comm=comm)

    structure.setDisplacements3D(x, struct_u, transfer )
    structure.writeTecplotFile(0,'arw2_struct')
    print struct_u
# end if

MPI.COMM_WORLD.Barrier()
sys.exit(0)
transfer.forward3D(struct_u,cfd_surface_points)

if (is_aero):
    print cfd_surface_points
    print 'aerocomm.rank,points:',comm.rank,cfd_surface_points.shape
    solver.interface.Mesh.setSurfaceCoordinatesLocal(cfd_surface_points+cfd_surface_points0,wing_families)
    solver.interface.Mesh.warpMeshSolid(n=3,sym='xy')
    #solver.interface.Mesh.warpMesh()
    solver.interface.Mesh.WriteMeshFile('arw2_debug_warped.cgns')
    #solver(test_case,niterations=1000)

sys.exit(0)
# # ================================================================
# #               Setup Some design Variables

FFD.coef[FFD.topo.l_index[6][0,0,-1]][1] += .01
FFD.coef[FFD.topo.l_index[6][0,1,-1]][1] += .01
FFD.coef[FFD.topo.l_index[6][1,0,-1]][1] += .01
FFD.coef[FFD.topo.l_index[6][1,1,-1]][1] += .01
FFD._updateVolumeCoef()
# FFD.writeTecplot('ffd_mod.dat')

if is_aero:
    #solver.interface.initializeADjoint()
    solver.interface.Mesh.setSurfaceCoordinatesLocal(FFD.getVolumePoints(0),wing_families)
    #solver.interface.Mesh.warpMeshSolid(n=3,sym='xy')
    solver.interface.Mesh.warpMesh()
    solver.interface.Mesh.WriteMeshFile('arw2_debug_warped.cgns')
    solver(test_case,niterations=1000)
# # end if



#     timeA = time.time()
#     rhs = structure.createVec()
#     x   = structure.createVec()
#     K   = structure.createMat()
#     F   = structure.createVec()

#     F.set(1.0)
#     F.applyBCs()

#     structure.assembleMat(0,K,rhs)
#     rhs.axpy(-1.0,F)


#     mpiPrint('The magic initial norm is %10.5e'%(rhs.norm()))

#     PC = TACS.ApproximateSchur(K,4,8.0,10)
#     PC.factor()
#     KSM = TACS.GMRES(K,PC,120,15,1)
#     KSM.setTolerances(1e-12,1e-30)
#     KSM.solve(rhs,x)
    
#     x.scale(-1.0)

#     structure.setVariables(0,x)
#     structure.assembleRes(0,rhs)
#     rhs.axpy(-1.0,F)

#     mass = functions.StructuralMass(structure)
#     m = structure.evalFunction( mass )
#     mpiPrint('Mass is %f'%(m),comm=comm)

#     mpiPrint('The magic norm is %10.5e'%(rhs.norm()),comm=comm)

#     rank = comm.rank
#     structure.writeTecplotFile(0,'tacs_sol_%d'%(rank))
#     structure.writeTecplotFile(rhs,'tacs_rhs_%d'%(rank))

#     print 'time:',time.time()-timeA

# #cfd_surface_points = solver.interface.Mesh.GetGlobalSurfaceCoordinates()

print 'exiting rank:',comm.rank
sys.exit(0)
