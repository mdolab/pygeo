# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross,shape,alltrue

# =============================================================================
# Extension modules
# =============================================================================

from mdo_import_helper import *
exec(import_modules('pyGeo','pyBlock','pySpline','geo_utils','mpi4py'))
exec(import_modules('pyAero_problem','pyAero_flow','pyAero_reference','pyAero_geometry'))
exec(import_modules('pySUMB','pyDummyMapping'))

# volume = pyBlock.pyBlock('cgns',file_name='warp_test.cgns')
# volume.doConnectivity('warp_test.con')
# volume.fitGlobal()
# volume.writeTecplot('warp_test_vol.dat',edge_labels=True,orig=False,coef=False)

mpiPrint('------- Warp Test Example ---------')

mpiPrint('\nGenerating Surface Geometry')
surface = pyGeo.pyGeo('plot3d',file_name='warp_test_surf.xyz',file_type='ascii',order='f')
surface.doConnectivity('warp_test_surf.con')
surface.fitGlobal()
surface.writeTecplot('warp_test_update.dat',tecio=True,coef=True,orig=False,surf_labels=True,directions=True)

mpiPrint('\nInitializating Flow Problem')

flow = Flow(name='Base Case',mach=0.5,alpha=2.0,beta=0.0,liftIndex=2)
ref = Reference('Baseline Reference',1.0,1.0,1.0)
geom = Geometry
test_case = AeroProblem(name='Simple Test',geom=geom,flow_set=flow,ref_set=ref)
solver = SUMB()
solver_options={'reinitialize':True,\
                'CFL':1.5,\
                'L2Convergence':1.e-10,\
                'MGCycle':'sg',\
                'MetricConversion':1.0,\
                'Discretization':'Central plus scalar dissipation',\
                'sol_restart':'no',
                'solveADjoint':'no',\
                'set Monitor':'Yes',\
                'Approx PC': 'no',\
                'Adjoint solver type': 'GMRES',\
                'adjoint relative tolerance':1e-10,\
                'adjoint absolute tolerance':1e-16,\
                'adjoint max iterations': 500,\
                'adjoint restart iteration' : 80,\
                'adjoint monitor step': 10,\
                'dissipation lumping parameter':6,\
                'Preconditioner Side': 'LEFT',\
                'Matrix Ordering': 'NestedDissection',\
                'Global Preconditioner Type': 'Additive Schwartz',\
                'Local Preconditioner Type' : 'ILU',\
                'ILU Fill Levels': 2,\
                'ASM Overlap' : 5,\
                'TS Stability': 'no'}

solver(test_case,niterations=1,grid_file='warp_test',solver_options=solver_options)

cfd_surface_points = solver.interface.Mesh.GetGlobalSurfaceCoordinates()
surface.attachSurface(cfd_surface_points[:,:,0])


#--------------  Do an ad-hoc modification --------------
#surface.coef[surface.topo.l_index[2][:,:]] += [.400,.200,0]
#surface.coef[surface.topo.l_index[1][-1,:]] += [-.05,.002525,0]
#surface.coef[surface.topo.l_index[0][-1,:]] += [.05,.000062525,0]

# # # Pull out LE
for j in xrange(surface.surfs[4].Nctlv):
    surface.coef[surface.topo.l_index[4][1,j]] += [-.08103,0,0]
    surface.coef[surface.topo.l_index[4][2,j]] += [-.08103,0,0]

# # Shrink TE
for i in xrange(surface.surfs[2].Nctlu):
    for j in xrange(surface.surfs[2].Nctlv):
        surface.coef[surface.topo.l_index[2][i,j]][1] *= .05

# # Pull Up Upper surface
for j in xrange(surface.surfs[0].Nctlv):
    surface.coef[surface.topo.l_index[0][1,j]] += [0,.05,0]
    surface.coef[surface.topo.l_index[0][2,j]] += [0,-.02,0]

# # Pull Down Lower surface
for j in xrange(surface.surfs[3].Nctlv):
    surface.coef[surface.topo.l_index[3][1,j]] -= [0,.05,0]
    surface.coef[surface.topo.l_index[3][2,j]] -= [0,.03,0]

# Displace the whole thing up and forware
surface.coef[:] += [-1.1,1.1,0]

for i in xrange(len(surface.coef)):
    surface.coef[i]= rotzV(surface.coef[i],-10*pi/180)

surface._updateSurfaceCoef()
surface.writeTecplot('block_update.dat',coef=False,tecio=True,directions=True)

# ---------------------------------------------------------

# Now do the solid warp

solver.interface.Mesh.SetGlobalSurfaceCoordinates(surface.getSurfacePoints(0).transpose(),True)
timeA = time.time()
solver.interface.Mesh.warpMeshSolid(topo='warp_test_FEA.con')
#solver.interface.Mesh.warpMeshSolid(n=2)
print 'Time is:',time.time()-timeA
#solver.interface.Mesh.warpMesh()

solver.interface.Mesh.WriteMeshFile('warp_test_new.cgns')



