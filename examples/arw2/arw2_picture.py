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
nProc_aero   = 4
nProc_struct = 1
split = False
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

#wing_families = ['wing_le','wing_low','wing_up','wing_tip']
#wing_families = ['default']
if is_aero:
    solver_options={'reinitialize':True,'CFL':1.5,'L2Convergence':1e-4,
                    'MGCycle':'3w','MetricConversion':1.0,'sol_restart':'no'}

    try:
        solver(test_case,niterations=1200,grid_file='arw2_400k',solver_options=solver_options)
    except:
        pass
    # end try
#     cfd_surface_points = solver.interface.Mesh.getSurfaceCoordinatesLocal(wing_families)
#     cfd_surface_points0 = cfd_surface_points.copy()
#     cfd_forces = solver.interface.Mesh.getSurfaceForcesLocal(wing_families)
#     FFD.embedVolume(cfd_surface_points)
#     solver.interface.initializeADjoint()
#     pts = cfd_surface_points
#     f = open('mdpts%d.dat'%(comm.rank),'w')
#     f.write ('VARIABLES = "X", "Y","Z","Fx","Fy","Fz"\n')
#     f.write('Zone T=\"%s\" I=%d\n'%('cfdforces',len(pts)))
#     f.write('DATAPACKING=POINT\n')
#     for i in xrange(len(pts)):
#         f.write('%f %f %f %f %f %f\n'%(pts[i,0],pts[i,1],pts[i,2],cfd_forces[i,0],cfd_forces[i,1],cfd_forces[i,2]))
#     # end for
#     f.close()

