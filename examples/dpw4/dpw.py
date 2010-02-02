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
exec(import_modules('pyGeo'))

# Load the plot3d xyz file
aircraft = pyGeo.pyGeo('plot3d',file_name='./geo_input/dpw.xyz',no_print=False)
#Compute and save the connectivity
aircraft.doEdgeConnectivity('./geo_input/dpw.con')

# Write A tecplot file
aircraft.writeTecplot('./geo_output/dpw.dat',orig=True,directions=True,
                      surf_labels=True,edge_labels=True,node_labels=True)
# Write an iges file for reload 
aircraft.writeIGES('./geo_input/dpw.igs')

# #Re-load the above saved iges file
# aircraft = pyGeo.pyGeo('iges',file_name='./geo_input/dpw.igs',no_print=False)
# #Load the edge connectivity
# aircraft.doEdgeConnectivity('./geo_input/dpw.con')
# sys.exit(0)

# ------- Now We will attach a set of surface points ---------

# We have a file with a set of surface points from a triangular surface
# mesh from icem

coordinates = aircraft.getCoordinatesFromFile('./geo_input/points')

# Now we can attach the surface -- Only available with cfd-csm-pre
dist,patchID,uv = aircraft.attachSurface(coordinates)
# We can also save these points to a file for future reference
aircraft.writeAttachedSurface('./geo_input/attached_surface',patchID,uv)

# And we can read them back in
patchID,uv = aircraft.readAttachedSurface('./geo_input/attached_surface')



# ------- Now we will add a reference axis --------
# End-Type ref_axis attachments
nsec = 3
x = [1147,1314,1804]
y = [119,427,1156]
z = [176,181,264]
rot_x = [0,0,0]
rot_y = [0,0,0]
rot_z = [0,0,0] 

# Add a single reference axis
aircraft.addRefAxis([2,3,4,5,8,9,10,11],x=x,y=y,z=z,
                    rot_x=rot_x,rot_y=rot_y,rot_z=rot_z)


# --------- Define Design Variable Functions Here -----------

def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
    ref_axis[0].x[:,1] = ref_axis[0].x0[:,1] * val
    return ref_axis

mpiPrint(' ** Adding Global Design Variables **')
aircraft.addGeoDVGlobal('span',1,0.5,2.0,span_extension)

idg = aircraft.DV_namesGlobal #NOTE: This is constant (idg -> id global
aircraft.DV_listGlobal[idg['span']].value = 1
aircraft.update()
aircraft.writeTecplot('./geo_output/dpw.dat',surf_labels=True,orig=True)







# # Create a coef0 list for all surfaces:aircraft.writeTecplot('./geo_output/dpw.dat',surf_labels=True)
# init_coef = []
# coef_temp = []
# for isurf in xrange(aircraft.nSurf):
#     init_coef.append(copy.deepcopy(aircraft.surfs[isurf].coef)) # Pluck out coef before update
#     coef_temp.append(zeros(shape(aircraft.surfs[isurf].coef)))


# # Start of mesh warping testing
# # Pluck out a section of coefficients
# # Now parameterize it:

# surfaces_to_warp = [0,1,6,7,12,13]
# surfaces_to_warp = [0,12]#,13]#0,1,6,7,12,13]
# timeA = time.time()

# for isurf in surfaces_to_warp:
#     # Check ONLY the edges
#     coef0 = copy.deepcopy(init_coef[isurf])
#     coef  = aircraft.surfs[isurf].coef

#     print 'isurf:',isurf
#     Nu = aircraft.surfs[isurf].Nctlu
#     Nv = aircraft.surfs[isurf].Nctlv

#     S = parameterizeFace(Nu,Nv,coef0)
#     dface = zeros((Nu,Nv,2))
       
#     # Spline the edges:
#     # Create the matrix of spline objects for all the lines:
#     u_lines = []
#     v_lines = []
#     for j in xrange(Nv):
#         # Make the u-lines:
#         u_lines.append(pySpline.linear_spline('interpolate',X=coef0[:,j],k=2))
#     for i in xrange(Nu):
#         v_lines.append(pySpline.linear_spline('interpolate',X=coef0[i,:],k=2))
       

#     # Now we can get the all the parameteric dface peturbation on the edges
#     timeB = time.time()
#     # Edge 0 and Edge 1
#     counter = 0
#     for i in xrange(Nu):
#         if not alltrue(coef[i,0]==coef0[i,0]):
#            #  print 'here0'
#             u,D,converged,update = u_lines[0].projectPoint(coef[i,0])
#             v,D,converged,update = v_lines[i].projectPoint(coef[i,0])
#             dface[i,0] = [u-S[i,0,0],v-S[i,0,1]]
#             counter += 1

#         if not alltrue(coef[i,-1]==coef0[i,-1]):
#            # print 'here1'
#             u,D,converged,update = u_lines[-1].projectPoint(coef[i,-1])
#             v,D,converged,update = v_lines[i].projectPoint(coef[i,-1])
#             dface[i,-1] = [u-S[i,-1,0],v-S[i,-1,1]]
#             counter += 1
#     # end for

#     # Edge 2 and Edge 3
#     for j in xrange(Nv):
#         if not alltrue(coef[0,j]==coef0[0,j]):
#             #print 'here2'
            
#             u,D,converged,update = u_lines[j].projectPoint(coef[0,j])
#             v,D,converged,update = v_lines[0].projectPoint(coef[0,j])
           
#             dface[0,j] = [u-S[0,j,0],v-S[0,j,1]]
#             counter += 1
#         if not alltrue(coef[-1,j]==coef0[-1,j]):
#             #print 'here3'
#             u,D,converged,update = u_lines[j].projectPoint(coef[-1,j])
#             v,D,converged,update = v_lines[-1].projectPoint(coef[-1,j])
#             dface[-1,j] = [u-S[-1,j,0],v-S[-1,j,1]]
#             counter += 1
#     # end for
#    # print 'counter:',counter


#     # Now interpolate the rest of dface with the parametric algebraic
#     # warping algorithim
#     timeC = time.time()
#     dface = warp_face(Nu,Nv,S,dface)
#     timeD = time.time()
#     # Now we know the 'parametric movements' we can make splines for each 'line' and move 'em accordingly
#     for i in xrange(Nu):
#         for j in xrange(Nv):
#             # Physical Change caused by du and dv
#             dx_u = u_lines[j].getValue(S[i,j,0]+dface[i,j,0]) - coef0[i,j]
#             dx_v = v_lines[i].getValue(S[i,j,1]+dface[i,j,1]) - coef0[i,j]
#             coef[i,j] = coef0[i,j] + dx_u + dx_v
#         # end for
#     # end for
            
#     coef_temp[isurf] = coef
#     timeE = time.time()
# # end for (isurf loop)


# print 'Init Time:',timeB-timeA
# print 'Project Time:',timeC-timeB
# print 'Warp Time:',timeD-timeC
# print 'Set time:',timeE-timeD

#  # # Force set those coef in the global list:
# for isurf in surfaces_to_warp:
#     l_list = aircraft.l_index[isurf]
#     for i in xrange(Nu):
#         for j in xrange(Nv):
#             aircraft.coef[l_list[i,j]] = coef_temp[isurf][i,j]
#         # end for
#     # end for
# aircraft._updateSurfaceCoef()

# aircraft.writeTecplot('../output/dpw.dat',edges=True,directions=True,
#                       labels=True,links=True)#,size=5)
