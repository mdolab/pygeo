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

# pySpline 
sys.path.append('../pySpline/python')

#cfd-csm pre (Optional)
sys.path.append('../../pyHF/pycfd-csm/python/')

#pyGeo
import pyGeo2 as pyGeo

# ==============================================================================
# Start of Script
# ==============================================================================

# Wing Information - Create a Geometry Object from cross sections

naf=3
airfoil_list = ['naca0012.dat','naca0012.dat','pinch_xfoil.dat']
chord = [1,1,1]
x = [0,0,0]
y = [0,0,0]
z = [0,4,4.125]
rot_x = [0,0,0]
rot_y = [0,0,0]
tw_aero = [0,0,0] # ie rot_z

offset = zeros((naf,2))
offset[:,0] = .25 # Offset sections by 0.25 in x

# Make the break-point vector
breaks = [1] #zero based (Must NOT contain 0 or index of last value)
cont = [1] # vector of length breaks: 0 for c0 continuity 1 for c1 continutiy
nsections = [10,8]# Length breaks + 1
section_spacing = [linspace(0,1,10),linspace(0,1,5)]
Nctlu = 13
end_type = 'pinch'
# 'pinch' or 'flat' or 'rounded' -> flat and rounded result in a
#  a new surface on the end 
                               
# Put spatial and rotations into two arrays (always the same)-------
X = zeros((naf,3))
rot = zeros((naf,3))

X[:,0] = x
X[:,1] = y
X[:,2] = z
rot[:,0] = rot_x
rot[:,1] = rot_y
rot[:,2] = tw_aero
# ------------------------------------------------------------------
    
# Procedure for Using pyGEO

# Step 1: Run the folloiwng Commands: (Uncomment between -------)
# ---------------------------------------------------------------------
#Note: u direction is chordwise, v direction is span-wise
wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,\
                       file_type='xfoil',scale=chord,offset=offset, \
                       Xsec=X,rot=rot,breaks=breaks,cont=cont,end_type=end_type,\
                       nsections=nsections,fit_type='lms', Nctlu=Nctlu,Nfoil=45)

wing.calcEdgeConnectivity(1e-2,1e-2)
wing.writeEdgeConnectivity('wing.con')
#wing.propagateKnotVectors()

#print (1-wing.surfs[2].tu)[::-1].copy()
#print wing.surfs[3].tu
#wing.surfs[3].tu = (1-wing.surfs[2].tu)[::-1].copy()
#wing.stitchEdges()
wing.writeTecplot('wing.dat',edges=True)
print 'Done Step 1'
sys.exit(0)
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
#Call the finalize command after we have set the connections
wing.finalize()
wing.stitchEdges() # Just to be sure
print 'Done Step 3'

# ----------------------------------------------------------------------

# Step 4: Now the rest of the code is up to the user. The user only
# needs to run the commands in step 3 to fully define the geometry of
# interest
print '---------------------------'
print 'Attaching Reference Axis...'
print '---------------------------'

# End-Type ref_axis attachments
# Note No surf_sec requried for the entire surface
wing.addRefAxis([0,1],X[0:2,:],rot[0:2,:],nrefsecs=nsections[0],\
                    spacing=section_spacing[0])
wing.addRefAxis([2,3],X[1:3,:],rot[1:3,:],nrefsecs=nsections[1],\
                    spacing=section_spacing[0])

#Flap-Type (full) ref_axis attachment
X = array([[2.,0,1.5],[2,0,3.5]]) # hinge Line
rot = array([[0,0,0],[0,0,0]])        
us = [15,None]
ue = [None,5]
vs = [6,6]
ve = [19,19]

wing.addRefAxis([0,1],X,rot,us=us,ue=ue,vs=vs,ve=ve)

# Now we specify How the ref axis move together
wing.addRefAxisCon(0,1,'end') # Wing and corner
wing.addRefAxisCon(0,2,'full') # flap

# Write out the surface
wing.writeTecplot('wing.dat',ref_axis=True,links=True)

# --------------------------------------
# Define Design Variable functions here:
# --------------------------------------
def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
#    print 'span',val           
    ref_axis[0].x[:,2] = ref_axis[0].x0[:,2] * val
    return ref_axis

def twist(val,ref_axis):
    '''Twist'''
#    print 'twist',val
    ref_axis[0].rot[:,2] = ref_axis[0].rot0[:,2] + ref_axis[0].s*val
    ref_axis[1].rot[:,2] = ref_axis[0].rot[-1,2]
    return ref_axis

def sweep(val,ref_axis):
    '''Sweep the wing'''
    # Interpret the val as an ANGLE
#    print 'sweep',val
    angle = val*pi/180
    dz = ref_axis[0].x[-1,2] - ref_axis[0].x[0,2]
    dx = dz*tan(angle)
    ref_axis[0].x[:,0] =  ref_axis[0].x0[:,0] +  dx * ref_axis[0].s

    dz = ref_axis[1].x[-1,2] - ref_axis[1].x[0,2]
    dx = dz*tan(angle)
    ref_axis[1].x[:,0] =  ref_axis[1].x0[:,0] +  dx * ref_axis[1].s

    return ref_axis

def flap(val,ref_axis):
#    print 'flap:',val
    ref_axis[2].rot[:,2] = val

    return ref_axis

# ------------------------------------------
#         Name, value, lower,upper,function, ref_axis_id -> must be a list
# # Add global Design Variables FIRST
wing.addGeoDVGlobal('span',1,0.5,2.0,span_extension)
wing.addGeoDVGlobal('twist',0,-20,20,twist)
wing.addGeoDVGlobal('sweep',0,-20,20,sweep)
wing.addGeoDVGlobal('flap',0,-20,20,flap)

# # Add sets of local Design Variables SECOND
wing.addGeoDVLocal('surface1',-0.1,0.1,surf=0,us=10,ue=15,vs=10,ve=15)
wing.addGeoDVLocal('surface2',-0.1,0.1,surf=1)
wing.addGeoDVLocal('surface3',-0.1,0.1,surf=2)
wing.addGeoDVLocal('surface4',-0.1,0.1,surf=3)

# # Get the dictionary to use names for referecing 
idg = wing.DV_namesGlobal #NOTE: This is constant (idg -> id global)
idl = wing.DV_namesLocal  #NOTE: This is constant (idl -> id local)

print 'idg',idg
print 'idl',idl

# # Change the DV's -> Normally this is done from the Optimizer
wing.DV_listGlobal[idg['span']].value = 1
wing.DV_listGlobal[idg['twist']].value = 0
wing.DV_listGlobal[idg['sweep']].value = 0
wing.DV_listGlobal[idg['flap']].value = 0

#wing.DV_listLocal[idl['surface1']].value[0,0] = 0.0
# wing.DV_listLocal[idl['surface2']].value[5,5] = .14
# wing.DV_listLocal[idl['surface3']].value[3,3] = .14
# wing.DV_listLocal[idl['surface4']].value[2,2] = .14

patchID,uv = wing.attachSurface() #Attach the surface BEFORE any update
wing.calcSurfaceDerivative(patchID,uv) 

wing.update()
timeA = time.time()
coef_list1 = wing.calcCtlDeriv() # Answer shows up in C
timeB = time.time()
print 'Derivative Time:',timeB-timeA

dx = 1e-5
coef0 = wing.returncoef()
coordinates0 = wing.getSurfacePoints(patchID,uv)

wing.DV_listGlobal[idg['span']].value = 1 + dx
wing.update()
coefdx = wing.returncoef()
coordinatesdx = wing.getSurfacePoints(patchID,uv)

# # Get The full vector 

dx1 = wing.C[:,0]
dx2 = (coordinatesdx-coordinates0)/(dx)

#dx1 = wing.J1[:,0]
#dx2 = (coefdx-coef0)/dx

f1 = open('dx1','w')
f2 = open('dx2','w')
print 'sizes:',len(dx1),len(dx2)
for i in xrange(len(dx1)):
    if abs(dx1[i]) < 1e-12:
        f1.write('0.0 \n')
    else:
        f1.write('%15g \n'%(dx1[i]))

    if abs(dx2[i]) < 1e-12:
        f2.write('0.0 \n')
    else: 
        f2.write('%15g \n'%(dx2[i]))
f1.close()
f2.close()

wing.writeTecplot('wing2.dat',ref_axis=True,links=True)
#wing.writeIGES('wing_mod.igs')
