#!/usr/bin/python
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross, vstack, arctan2

# =============================================================================
# Extension modules
# =============================================================================
#from matplotlib.pylab import *

# pySpline
sys.path.append('../pySpline/python')

#cfd-csm pre
sys.path.append('../../pyHF/pycfd-csm/python/')

#pyGeo
import pyGeo

# Wing Information

def c_atan2(x,y):
    a=x.real
    b=x.imag
    c=y.real
    d=y.imag
    return complex(arctan2(a,c),(c*b-a*d)/(a**2+c**2))


naf=5
airfoil_list = ['af15-16.inp','af15-16.inp','af15-16.inp','af15-16.inp','pinch.inp']
chord = [1.25,.65,.65,.65,.65]
x = [1.25,1.25,1.25,1.25,1.25]
y = [0,0.4,.6,1.2,1.4]
z = [0,6,6.2,6.2,6.2]
rot_x = [0,0,-90,-90,-90]
rot_y = [0,0,0,0,0]
tw_aero = [-4,4,0,0,0] # ie rot_z
X = zeros((naf,3))
rot = zeros((naf,3))

offset = zeros((naf,2))
offset[:,0] = .25 # Offset sections by 0.25 in x

# Make the break-point vector
breaks = [1,2] #zero based (Must NOT contain 0 or index of last value)
nsections = [20,20,20] # Length breaks + 1
section_spacing = []
for i in xrange(len(nsections)):
    section_spacing.append( 0.5*(1-cos(linspace(0,pi,nsections[i]))))
    #section_spacing.append(1-linspace(1,0,nsections[i])**2)
# Put spatial and rotations into two arrays
X[:,0] = x
X[:,1] = y
X[:,2] = z
rot[:,0] = rot_x
rot[:,1] = rot_y
rot[:,2] = tw_aero
         
Nctlu = 17

# Procedure for Using pyGEO

# Step 1: Run the folloiwng Commands: (Uncomment between -------)
# ---------------------------------------------------------------------
#Note: u direction is chordwise, v direction is span-wise
# wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,\
#                    ref_axis=ref_axis,fit_type='lms',breaks=breaks,Nctlu = Nctlu,Nctlv=Nctlv,ctlv_spacing=ctlv_spacing)
wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,\
                   Xsec=X,rot=rot,breaks=breaks,nsections=nsections,section_spacing=section_spacing,fit_type='lms',Nctlu=Nctlu,Nfoil=40)

wing.calcEdgeConnectivity(1e-2,1e-2)
wing.writeEdgeConnectivity('wing.con')
wing.stitchEdges()
wing.writeTecplot('wing.dat',write_ref_axis=True,write_links=True)
print 'Done Step 1'

#sys.exit(0)
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
# wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis,breaks=breaks,fit_type='lms',Nctlu = 13,Nctlv=Nctlv)
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

# wing = pyGeo.pyGeo('iges',file_name='wing.igs')
# wing.readEdgeConnectivity('wing.con')
# wing.stitchEdges() # Just to be sure
# print 'Done Step 3'

# ----------------------------------------------------------------------

# Step 4: Now the rest of the code is up to the user. The user only
# needs to run the commands in step 3 to fully define the geometry of
# interest

#print 'Attaching Ref Axis...'
#wing.setRefAxis([0,1,2,3,4,5],ref_axis,sections=[[0,1],[2,3],[4,5]])
#wing.writeTecplot('wing.dat',write_ref_axis=True,write_links=True)

# --------------------------------------
# Define Design Variable functions here:
# --------------------------------------
def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
    #print 'span'
    ref_axis.x[:,2] = ref_axis.x0[:,2] * val
    return ref_axis

def winglet_extension(val,ref_axis):
    '''extend the winglet'''
    #print 'winglet'
    ref_axis.x[:,1] = ref_axis.x0[:,1]*val
    return ref_axis

def twist(val,ref_axis):
    '''Twist'''
    #print 'twist'
    ref_axis.rot[:,2] = ref_axis.rot0[:,2] + ref_axis.s*val
    return ref_axis

def sweep(val,ref_axis):
    '''Sweep the wing'''
    ref_axis.x[:,0] =  ref_axis.x0[:,0] +  val * ref_axis.s
    angle = c_atan2(val,ref_axis.x[-1,2])*180/pi
    ref_axis.rot[:,1] = ref_axis.s * angle

    return ref_axis

def set_chord(val,ref_axis):
    '''Set the scales (and thus chords) on the wing'''
    #print 'chord'
    ref_axis[0].scale[:] = val
    ref_axis[1].scale[:] = val[-1]
    ref_axis[2].scale[:] = val[-1]

    return ref_axis
# ------------------------------------------
#                        Name, value, lower,upper,function, ref_axis_id -> must be a list
# Add global Design Variables FIRST
wing.addGeoDVGlobal('span',1,0.5,2.0,span_extension,[0])
wing.addGeoDVGlobal('winglet',1,0.5,2.0,winglet_extension,[2])
wing.addGeoDVGlobal('twist',0,-20,20,twist,[0])
wing.addGeoDVGlobal('sweep',0,-20,20,sweep,[0])
wing.addGeoDVGlobal('chord',ones(13),0.1,2,set_chord,[0,1,2])

# Add sets of local Design Variables SECOND
wing.addGeoDVLocal('surface1',-0.1,0.1,0)
wing.addGeoDVLocal('surface2',-0.1,0.1,1)
wing.addGeoDVLocal('surface3',-0.1,0.1,2)
wing.addGeoDVLocal('surface4',-0.1,0.1,3)
wing.addGeoDVLocal('surface5',-0.1,0.1,4)
wing.addGeoDVLocal('surface6',-0.1,0.1,5)

# Get the dictionary to use names for referecing 
idg = wing.DV_namesGlobal #NOTE: This is constant (idg -> id global)
idl = wing.DV_namesLocal  #NOTE: This is constant (idl -> id local)

print 'idg',idg
print 'idl',idl
# Change the DV's -> Normally this is done from the Optimizer
wing.DV_listGlobal[idg['span']].value = 1.5
wing.DV_listGlobal[idg['twist']].value = -5
wing.DV_listGlobal[idg['sweep']].value = 2 + 1.0e-40j
wing.DV_listGlobal[idg['chord']].value = linspace(4,1.25,20)
wing.DV_listGlobal[idg['winglet']].value = .5
wing.DV_listLocal[idl['surface1']].value[5,5] = .14
wing.DV_listLocal[idl['surface2']].value[5,5] = .24
wing.DV_listLocal[idl['surface3']].value[5,5] = .34
wing.DV_listLocal[idl['surface4']].value[5,5] = .44
wing.DV_listLocal[idl['surface5']].value[5,5] = .54
wing.DV_listLocal[idl['surface6']].value[5,5] = .64
print wing.DV_listLocal[idl['surface4']].Nctlu
print wing.DV_listLocal[idl['surface4']].Nctlv

timeA = time.time()
wing.update()
timeB = time.time()

print 'Coeffs:'
print wing.surfs[0].coef
#print wing.surfs[1].coef
print 'update time is :',timeB-timeA

wing.writeTecplot('wing2.dat',write_ref_axis=True,write_links=True)
wing.writeIGES('wing.igs')
