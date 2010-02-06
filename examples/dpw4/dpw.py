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
nsec = 3
x = array([1147+75,1314+50,1804.+25])
y = [119,427,1156.]
z = [150,176,264.]
rot_x = [0,0,0.]
rot_y = [0,0,0.]
rot_z = [0,0,0.] 

# Add a single reference axis
aircraft.addRefAxis([2,3,4,5,8,9,10,11,16,17],x=x,y=y,z=z,
                    rot_x=rot_x,rot_y=rot_y,rot_z=rot_z)

aircraft.writeTecplot('./geo_output/dpw.dat',orig=True,directions=True,
                      surf_labels=True,edge_labels=True,node_labels=True,
                      ref_axis=True,links=True)

# --------- Define Design Variable Functions Here -----------

def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
    ref_axis[0].x[:,1] = ref_axis[0].x0[:,1] * val
    return ref_axis

def outer_sweep(val,ref_axis):
    '''Single design variable for outer section sweep'''
    ref_axis[0].x[2,0] = ref_axis[0].x0[2,0] + val
    return ref_axis

mpiPrint(' ** Adding Global Design Variables **')
aircraft.addGeoDVGlobal('span',1,0.5,2.0,span_extension)
aircraft.addGeoDVGlobal('outer_sweep',0,-100,100.0,outer_sweep)

idg = aircraft.DV_namesGlobal #NOTE: This is constant (idg -> id global
aircraft.DV_listGlobal[idg['span']].value = 1.2
aircraft.DV_listGlobal[idg['outer_sweep']].value = 100
aircraft.update()

aircraft.writeTecplot('./geo_output/dpw.dat',orig=True,directions=True,
                      surf_labels=True,edge_labels=True,node_labels=True,
                      links=True,ref_axis=True)

# ---------- Now compute the Surface Derivatives ---------------

aircraft._calcdPtdCoef(patchID,uv)



