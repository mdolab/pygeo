# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross

# =============================================================================
# Extension modules
# =============================================================================

# pySpline 
sys.path.append('../pySpline/python')

#cfd-csm pre (Optional)
sys.path.append('../../pyHF/pycfd-csm/python/')

#pyGeo
import pyGeo2 as pyGeo

# This script reads a surfaced-based plot3d file as typically
# outputted by aerosurf. It then creates a b-spline surfaces for each
# # surface patch.
# timeA = time.time()
# aircraft = pyGeo.pyGeo('plot3d',file_name='fuse2.xyz')
# aircraft.calcEdgeConnectivity()
# aircraft.writeEdgeConnectivity('dpw.con')
# #aircraft.readEdgeConnectivity('aircraft.con')
# aircraft.propagateKnotVectors()
# aircraft.fitSurfaces()
# timeA = time.time()

# aircraft.writeTecplot('dpw.dat',edges=True)
# timeB =time.time()
# print 'Write time is:',timeB-timeA
# print 'full time',time.time()-timeA

# aircraft.writeIGES('dpw.igs')
# sys.exit(0)

aircraft = pyGeo.pyGeo('iges',file_name='dpw.igs')
aircraft.readEdgeConnectivity('dpw.con')
#aircraft.writeTecplot('dpw.dat',edges=True)



# End-Type ref_axis attachments
naf = 3
x = [1147,1314,1804]
y = [119,427,1156]
z = [176,181,264]
rot_x = [0,0,0]
rot_y = [0,0,0]
rot_z = [0,0,0] # ie rot_z
X = zeros((naf,3))
rot = zeros((naf,3))
X[:,0] = x
X[:,1] = y
X[:,2] = z
rot[:,0] = rot_x
rot[:,1] = rot_y
rot[:,2] = rot_z

aircraft.addRefAxis([2,3,8,9],X[0:2,:],rot[0:2,:],nrefsecs=6)
aircraft.addRefAxis([4,5,10,11,16,17],X[1:3,:],rot[1:3,:],nrefsecs=6)
aircraft.addRefAxisCon(0,1,'end') # Innter Wing and Outer Wing ('Attach ra1 to ra0 with type 'end')


def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
                       
#    ref_axis[0].x[:,1] = ref_axis[0].x0[:,1]*1.5

    ref_axis[0].rot[:,1] = val

    return ref_axis


print ' ** Adding Global Design Variables **'
aircraft.addGeoDVGlobal('span',1,0.5,2.0,span_extension)

idg = aircraft.DV_namesGlobal #NOTE: This is constant (idg -> id global
print 'idg',idg

aircraft.DV_listGlobal[idg['span']].value =8

aircraft.update()
aircraft.checkCoef()
aircraft.writeTecplot('dpw.dat',edges=True,links=True)
