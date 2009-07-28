#!/usr/bin/python
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross, vstack, arctan2, tan

# =============================================================================
# Extension modules
# =============================================================================
#from matplotlib.pylab import *

# pySpline
sys.path.append('../pySpline/python')

#cfd-csm pre
sys.path.append('../../pyHF/pycfd-csm/python/')

#pyGeo
import pyGeo2 as pyGeo

# Wing Information

def c_atan2(x,y):
    a=real(x)
    b=imag(x)
    c=real(y)
    d=imag(y)
    return complex(arctan2(a,c),(c*b-a*d)/(a**2+c**2))


naf=3
airfoil_list = ['af15-16.inp','af15-16.inp','pinch.inp']
chord = [1.25,1.25,1.25]
x = [1.25,1.25,1.25]
y = [0,0,0]
z = [0,5,6]
rot_x = [0,0,0]
rot_y = [0,0,0]
tw_aero = [0,0,0] # ie rot_z
X = zeros((naf,3))
rot = zeros((naf,3))

offset = zeros((naf,2))
offset[:,0] = .25 # Offset sections by 0.25 in x

# Make the break-point vector
breaks = [1] #zero based (Must NOT contain 0 or index of last value)
nsections = [26,5]# Length breaks + 1
section_spacing = []

s1 = hstack([linspace(0,.25,4),linspace(0.26,0.3,5),linspace(0.35,0.7,8),\
                 linspace(0.71,0.75,5),linspace(0.76,1,4)])

section_spacing.append(s1)
section_spacing.append(linspace(0,1,5))

    #section_spacing.append(1-linspace(1,0,nsections[i])**2)
# Put spatial and rotations into two arrays
X[:,0] = x
X[:,1] = y
X[:,2] = z
rot[:,0] = rot_x
rot[:,1] = rot_y
rot[:,2] = tw_aero
         
Nctlu = 26

# Procedure for Using pyGEO

# Step 1: Run the folloiwng Commands: (Uncomment between -------)
# ---------------------------------------------------------------------
#Note: u direction is chordwise, v direction is span-wise
# wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,\
#                    offset=offset, Xsec=X,rot=rot,breaks=breaks,\
#                    nsections=nsections,section_spacing=section_spacing,\
#                    fit_type='lms', Nctlu=Nctlu,Nfoil=20)

# wing.calcEdgeConnectivity(1e-2,1e-2)
# wing.writeEdgeConnectivity('wing.con')
# wing.stitchEdges()
# wing.writeTecplot('wing.dat',write_ref_axis=False,write_links=False)
# print 'Done Step 1'
# sys.exit(0)
# ----------------------------------------------------------------------
# 0: -> Load wing.dat to check connectivity information and modifiy
# wing.con file to correct any connectivity info and set
# continuity. Re-run step 1 until all connectivity information and
# continutity information is correct.

# Step 2: -> Run the following Commands (Uncomment between --------)
# After step 1 we can load connectivity information from file,
# propagate the knot vectors, stitch the edges, and then fit the
# entire surfaces with continuity constraints.  This output is then
# saved as an igs file which is the archive format storage format we
# are using for bspline surfaces

# ----------------------------------------------------------------------
# wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,\
#                        offset=offset, Xsec=X,rot=rot,breaks=breaks,\
#                        nsections=nsections,section_spacing=section_spacing,\
#                        fit_type='lms', Nctlu=Nctlu,Nfoil=20)

# wing.readEdgeConnectivity('wing.con')
# wing.propagateKnotVectors()
# wing.stitchEdges()
# #wing.fitSurfaces()
# wing.writeTecplot('wing.dat')
# wing.writeIGES('wing.igs')
# print 'Done Step 2'
# sys.exit(0)
# ----------------------------------------------------------------------

# Step 3: -> After step 2 we now have two files we need, the stored
# iges file as well as the connectivity file. The IGES file which has
# been generated can be used to generate a 3D CFD mesh in ICEM.  Now
# to load a geometry for an optimization run we simply load the IGES
# file as well as the connectivity file and we are good to go.

# ----------------------------------------------------------------------

wing = pyGeo.pyGeo('iges',file_name='wing.igs')
wing.readEdgeConnectivity('wing.con')
wing.stitchEdges() # Just to be sure
print 'Done Step 3'

# ----------------------------------------------------------------------

# Step 4: Now the rest of the code is up to the user. The user only
# needs to run the commands in step 3 to fully define the geometry of
# interest
print '----------------------'
print 'Attaching Ref Axis...'
print '----------------------'

# Full ref_axis attachments
surf_sec = ['[:,:]','[:,:]']
wing.addRefAxis([0,1],X[0:2,:],rot[0:2,:],nrefsecs=nsections[0],\
                    spacing=section_spacing[0],surf_sec = surf_sec )
wing.addRefAxis([2,3],X[1:3,:],rot[1:3,:],nrefsecs=nsections[1],\
                    spacing=section_spacing[0],surf_sec = surf_sec )

#Flap-Type ref_axis attachment
X = array([[2.,0,1.5],[2,0,3.5]])
rot = array([[0,0,0],[0,0,0]])
surf_sec = ['[15:,6:19]','[0:5,6:19]']
wing.addRefAxis([0,1],X,rot,surf_sec = surf_sec )

# Now we specify How the ref axis move together
wing.addRefAxisCon(0,1,'end') # Wing and corner
wing.addRefAxisCon(0,2,'full') # flap
wing.writeTecplot('wing.dat',write_ref_axis=True,write_links=True)

# --------------------------------------
# Define Design Variable functions here:
# --------------------------------------
def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
    #print 'span'
    ref_axis[0].x[:,2] = ref_axis[0].x0[:,2] * val
    return ref_axis

def twist(val,ref_axis):
    '''Twist'''
    #print 'twist'
    ref_axis[0].rot[:,2] = ref_axis[0].rot0[:,2] + ref_axis[0].s*val
    ref_axis[1].rot[:,2] = ref_axis[0].rot[-1,2]

    return ref_axis

def sweep(val,ref_axis):
    '''Sweep the wing'''
    # Interpret the val as an ANGLE
    angle = val*pi/180
    dz = ref_axis[0].x[-1,2] - ref_axis[0].x[0,2]
    dx = dz*tan(angle)
    ref_axis[0].x[:,0] =  ref_axis[0].x0[:,0] +  dx * ref_axis[0].s

    dz = ref_axis[1].x[-1,2] - ref_axis[1].x[0,2]
    dx = dz*tan(angle)
    ref_axis[1].x[:,0] =  ref_axis[1].x0[:,0] +  dx * ref_axis[1].s

    return ref_axis

def flap(val,ref_axis):
    ref_axis[0].rot[:,2] = val

    return ref_axis

# ------------------------------------------
#         Name, value, lower,upper,function, ref_axis_id -> must be a list
# # Add global Design Variables FIRST
wing.addGeoDVGlobal('span',1,0.5,2.0,span_extension)
wing.addGeoDVGlobal('twist',0,-20,20,twist)
wing.addGeoDVGlobal('sweep',0,-20,20,sweep)
wing.addGeoDVGlobal('flap',0,-20,20,flap)

# # Add sets of local Design Variables SECOND
wing.addGeoDVLocal('surface1',-0.1,0.1,0)
wing.addGeoDVLocal('surface2',-0.1,0.1,1)
wing.addGeoDVLocal('surface3',-0.1,0.1,2)
wing.addGeoDVLocal('surface4',-0.1,0.1,3)

# # Get the dictionary to use names for referecing 
idg = wing.DV_namesGlobal #NOTE: This is constant (idg -> id global)
idl = wing.DV_namesLocal  #NOTE: This is constant (idl -> id local)

print 'idg',idg
print 'idl',idl
# # Change the DV's -> Normally this is done from the Optimizer
wing.DV_listGlobal[idg['span']].value = .95
wing.DV_listGlobal[idg['flap']].value = -10
wing.DV_listGlobal[idg['twist']].value = -6
wing.DV_listGlobal[idg['sweep']].value = 15

wing.DV_listLocal[idl['surface1']].value[7,7] = .14
# wing.DV_listLocal[idl['surface2']].value[5,5] = .14
# wing.DV_listLocal[idl['surface3']].value[3,3] = .14
# wing.DV_listLocal[idl['surface4']].value[2,2] = .14

timeA = time.time()
wing.update()
timeB = time.time()
wing.calcCtlDeriv()
timeC = time.time()

print 'Update Time:',timeB-timeA
print 'Derivative Time:',timeC-timeB

# Now generate the jacobian


wing.writeTecplot('wing2.dat',write_ref_axis=True,write_links=True)
wing.writeIGES('wing_mod.igs')
