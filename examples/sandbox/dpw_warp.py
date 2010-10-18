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
exec(import_modules('pyGeo','pySpline'))

# Load the plot3d xyz file
# aircraft = pyGeo.pyGeo('plot3d',file_name='../dpw4/geo_input/dpw.xyz',no_print=False)
# # # #Compute and save the connectivity
# aircraft.doEdgeConnectivity('dpw2.con')
# # # # Write an iges file so we can load it back in after
# aircraft.writeIGES('dpw.igs')

# # #Re-load the above saved iges file
aircraft = pyGeo.pyGeo('iges',file_name='dpw.igs',no_print=False)
#Load the edge connectivity
aircraft.doEdgeConnectivity('dpw2.con')
aircraft.writeTecplot('dpw.dat',coef=False,orig=False)
wing_surfs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,76,77,78,79,41,45,51,43]

xmin,xmax = aircraft.getBounds(surfs=wing_surfs)
vol = pySpline.trilinear_volume(xmin,xmax)
vol.writeTecplot('volume.dat',coef=False)


# Load in the volume stuff
dv_volume = pyGeo.pyBlock('plot3d',file_name='./dv_blocking/dv_volume.fmt')
#dv_surfs  = pyGeo.pyGeo('plot3d',file_name='./dv_blocking/dv_surfs.fmt')

dv_volume.writeTecplot('./dv_volume.dat',coef=False,vol_labels=True)
mpiPrint('Embedding Geo')
dv_volume.embedGeo(aircraft)

dv_volume.vols[5].coef[1,1,1] += [0,0,50]
dv_volume.vols[7].coef[1,1,0] += [0,0,50]
dv_volume.vols[8].coef[0,1,0] += [0,0,50]
dv_volume.vols[1].coef[0,1,1] += [0,0,50]
dv_volume.vols[2].coef[0,0,1] += [0,0,50]

dv_volume.updateGeo(aircraft)
dv_volume.writeTecplot('./dv_volume_mod.dat',coef=False)
aircraft.writeTecplot('dpw_mod.dat',coef=False,orig=False)
