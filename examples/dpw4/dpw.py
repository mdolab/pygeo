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
# aircraft = pyGeo.pyGeo('plot3d',file_name='./geo_input/dpw.xyz',no_print=False)
# #Compute and save the connectivity
# aircraft.doEdgeConnectivity('./geo_input/dpw2.con')
# # Write an iges file so we can load it back in after
# aircraft.writeIGES('./geo_input/dpw.igs')

# # #Re-load the above saved iges file
aircraft = pyGeo.pyGeo('iges',file_name='./geo_input/dpw.igs',no_print=False)
#Load the edge connectivity
aircraft.doEdgeConnectivity('./geo_input/dpw2.con')

# ------- Now We will attach a set of surface points ---------
# We have a file with a set of surface points from a triangular surface
# mesh from icem

coordinates = aircraft.getCoordinatesFromFile('./geo_input/points')

# Now we can attach the surface -- Only available with cfd-csm-pre
aircraft.attachSurface(coordinates)
# We can also save these points to a file for future reference
aircraft.writeAttachedSurface('./geo_input/attached_surface',0)

# And we can read them back in
aircraft.readAttachedSurface('./geo_input/attached_surface') 
#(Now we have two (identical) attached surfaces

# ------- Now we will add a reference axis --------
nsec = 3
x = array([1147+75,1314+50,1827])
y = [119,427,1148.]
z = [150,176,264.]

# Add reference axis
aircraft.addRefAxis([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,76,77,78,79,41,45,51,43],x=x,y=y,z=z,rot_type=3) #Surface list then x,y,z
aircraft.writeTecplot('./geo_output/dpw.dat',orig=True,directions=True,
                      surf_labels=True,edge_labels=False,node_labels=False,
                      links=True,ref_axis=True)

# rot_type = 1,2,3,4,5 or 6 -> Specifies the rotation order
# 1 -> x-y-z
# 2 -> x-z-y
# 3 -> y-z-x  -> This example (right body x-streamwise y-out wing z-up)
# 4 -> y-x-z
# 5 -> z-x-y  -> Default aerosurf (Left body x-streamwise y-up z-out wing)
# 6 -> z-y-x

# --------- Define Design Variable Functions Here -----------

def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
    ref_axis[0].x[:,1] = ref_axis[0].x0[:,1] * val
    return ref_axis

def outer_sweep(val,ref_axis):
    '''Single design variable for outer section sweep'''
    ref_axis[0].x[2,0] = ref_axis[0].x0[2,0] + val
    return ref_axis

def outer_twist(val,ref_axis):
    ref_axis[0].rot_y[2] = val
    return ref_axis

def outer_dihedral(val,ref_axis):
    ref_axis[0].x[2,2] = ref_axis[0].x0[2,2] + val
    return ref_axis

def tip_chord(val,ref_axis):
    ref_axis[0].scale[2] = ref_axis[0].scale0[2]*val
    return ref_axis

mpiPrint(' ** Adding Global Design Variables **')
aircraft.addGeoDVGlobal('span',1,0.5,2.0,span_extension)
aircraft.addGeoDVGlobal('outer_sweep',0,-100,100.0,outer_sweep)
aircraft.addGeoDVGlobal('outer_twist',0,-10,10.0,outer_twist)
aircraft.addGeoDVGlobal('outer_dihedral',0,-50,50.0,outer_dihedral)
aircraft.addGeoDVGlobal('tip_chord',1,.75,1.25,tip_chord)
idg = aircraft.DV_namesGlobal #NOTE: This is constant (idg -> id global
aircraft.DV_listGlobal[idg['span']].value = 1.2
aircraft.DV_listGlobal[idg['outer_sweep']].value = -25
aircraft.DV_listGlobal[idg['outer_twist']].value = -2
aircraft.DV_listGlobal[idg['outer_dihedral']].value = -20
aircraft.DV_listGlobal[idg['tip_chord']].value = 1.25
aircraft.update()

aircraft.writeTecplot('./geo_output/dpw_mod.dat',orig=True,directions=True,
                      surf_labels=True,edge_labels=True,node_labels=True,
                      links=True,ref_axis=True)

# ---------- Now compute the Surface Derivatives ---------------
aircraft.computeSurfaceDerivative(index=0)

# # ------------------ Optional Derivative Check with FD ------
# # Check the coefficients derivative
# aircraft.DV_listGlobal[idg['span']].value = 1.2 
# aircraft.update()
# coef0 = aircraft.coef.flatten().copy()
# aircraft.DV_listGlobal[idg['span']].value = 1.2 + 1e-5
# aircraft.update()
# coef = aircraft.coef.flatten().copy()
# dcoefdx = (coef-coef0)/1e-5
# # Dump out
# g = open('coef_fd.out','w')
# f = open('coef_an.out','w')
# for i in xrange(len(aircraft.coef)*3):
#     f.write('%f\n'%(aircraft.dCoefdX[i,0]))
#     g.write('%f\n'%(dcoefdx[i]))
# f.close()
# g.close()

# # Check the pt derivative
# aircraft.DV_listGlobal[idg['outer_twist']].value = 10
# aircraft.update()
# pts0 = aircraft.getSurfacePoints(0).flatten()
# aircraft.DV_listGlobal[idg['outer_twist']].value = 10 + 1e-6
# aircraft.update()
# pts = aircraft.getSurfacePoints(0).flatten()
# dptdx = (pts-pts0)/1e-6

# # Dump out
# g = open('pt_fd.out','w')
# f = open('pt_an.out','w')
# for i in xrange(len(dptdx)):
#     f.write('%f\n'%(aircraft.attached_surfaces[0].dPtdX[i,2]))
#     g.write('%f\n'%(dptdx[i]))
# f.close()
# g.close()
# #Use xxdiff to verify they are identical 

# sys.exit(0)


