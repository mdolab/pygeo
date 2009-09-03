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

import petsc4py
petsc4py.init(sys.argv)

# =============================================================================
# Extension modules
# =============================================================================

# pySpline 
sys.path.append('../pySpline/python')

#cfd-csm pre (Optional)
sys.path.append('../../pyHF/pycfd-csm/python/')

#pyGeo
import pyGeo2 as pyGeo

#pyLayout
sys.path.append('../pyLayout/')
import pyLayout

#Design Variable Functions
from dv_funcs import *

# ==============================================================================
# Start of Script
# ==============================================================================

# Wing Information - Create a Geometry Object from cross sections

naf=2
airfoil_list = ['naca0012.dat','naca0012.dat']
chord = [1,1]
x = [0,0]
y = [0,0]
z = [0,4]
rot_x = [0,0]
rot_y = [0,0]
tw_aero = [0,0] # ie rot_z

offset = zeros((naf,2))
offset[:,0] = .25 # Offset sections by 0.25 in x
# Make the break-point vector
Nctlu = 9
end_type = 'flat'
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
                       Xsec=X,rot=rot,end_type=end_type,fit_type='lms',\
                       Nctlu=Nctlu,Nfoil=45)

wing.calcEdgeConnectivity(1e-6,1e-6)
wing.writeEdgeConnectivity('degen_wing.con')
#sys.exit(0)
#wing.readEdgeConnectivity('degen_wing.con')
#sys.exit(0)
wing.propagateKnotVectors()
wing.writeTecplot('degen_wing.dat',edges=True)
wing.writeIGES('degen_wing.igs')
print 'Done Step 1'
print wing.surfs[0].tu
print wing.surfs[1].tu
print wing.surfs[2].tu
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
wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,\
                       file_type='precomp',scale=chord,offset=offset, \
                       Xsec=X,rot=rot,breaks=breaks,cont=cont,end_type=end_type,\
                       nsections=nsections,fit_type='lms', Nctlu=Nctlu,Nfoil=45)

wing.readEdgeConnectivity('wing.con')
wing.propagateKnotVectors()
timeA = time.time()
wing.fitSurfaces()
timeB = time.time()
print 'Fitting Time:',timeB-timeA
wing.update()
wing.writeTecplot('wing.dat')
wing.writeIGES('wing.igs')
print 'Done Step 2'
time.sleep(10)
sys.exit(0)
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
print 'Done Step 3'


# ----------------------------------------------------------------------

# Step 4: Now the rest of the code is up to the user. The user only
# needs to run the commands in step 3 to fully define the geometry of
# interest
print '---------------------------'
print 'Attaching Reference Axis...'
print '---------------------------'

# End-Type ref_axis attachments
wing.addRefAxis([0,1,2,3],X[0:2,:],rot[0:2,:],nrefsecs=nsections[0],\
                    spacing=section_spacing[0])
wing.addRefAxis([2,3],X[1:3,:],rot[1:3,:],nrefsecs=nsections[1],\
                    spacing=section_spacing[0])

#Flap-Type (full) ref_axis attachment
X = array([[.4,0,2],[.4,0,3]]) # hinge Line
rot = array([[0,0,0],[0,0,0]])        

# 
flap_box = pyGeo.point_select('y',pt1=[0.4,0,1.5],pt2=[1.1,0,3.5])
wing.addRefAxis([0,1],X,rot,point_select=flap_box)

print 'Done Ref Axis Adding!'

# Now we specify How the ref axis move together
wing.addRefAxisCon(0,1,'end') # Wing and cap ('Attach ra1 to ra0 with type 'end')
wing.addRefAxisCon(0,2,'full') # flap ('Attach ra2 to ra0 with type 'full')

# Write out the surface
wing.writeTecplot('wing.dat',ref_axis=True,links=True)



# ------------------------------------------
#         Name, value, lower,upper,function,
# Add global Design Variables FIRST
print ' ** Adding Global Design Variables **'
wing.addGeoDVGlobal('span',1,0.5,2.0,span_extension)
wing.addGeoDVGlobal('twist',0,-20,20,twist)
wing.addGeoDVGlobal('sweep',0,-20,20,sweep)
wing.addGeoDVGlobal('flap',0,-20,20,flap)

# Add Normal Design Variables SECOND
print ' ** Adding Normal Design Variables **'
wing.addGeoDVNormal('norm_surf0',-0.1,0.1,surf=0,overwrite=True)
a_few = pyGeo.point_select('list',coef=[[5,6],[8,5],[9,2]])
wing.addGeoDVNormal('norm_surf1',-0.1,0.1,surf=1,overwrite=True,\
                        point_select=a_few)

# Add Local Design Variables THIRD
print ' ** Adding Local Design Variables **'
wing.addGeoDVLocal('local_surf2',-0.1,0.1,surf=2)
wing.addGeoDVLocal('local_surf3',-0.1,0.1,surf=3)

# # Get the dictionary to use names for referecing 
idg = wing.DV_namesGlobal #NOTE: This is constant (idg -> id global)
idn = wing.DV_namesNormal #NOTE: This is constant (idn -> id normal)
idl = wing.DV_namesLocal  #NOTE: This is constant (idl -> id local)

print 'idg',idg
print 'idn',idn
print 'idl',idl

# -------------- Attach the discrete coorsponding surface ----------------

coors = wing.getCoordinatesFromFile('naca0012.dtx')
dist,patchID,uv = wing.attachSurface(coors) #Attach the surface BEFORE any update
wing.calcSurfaceDerivative(patchID,uv)

print 'About to do update..'
wing.update()
wing.writeTecplot('wing2.dat',ref_axis=True,links=True)
print 'Done Update:'

timeA = time.time()
wing.calcCtlDeriv() 
print 'Derivative Time:',time.time()-timeA

sys.exit(0)

# # print 'Testing pyLayout'
# # L = pyLayout.Layout(wing,'input.py')
# # print 'back in script'
# # #print 'tacs is:',tacs

# # sys.exit(0)
# # #Test code for pyLayout



# dx = 1.0e-5

# coef0 = wing.coef.astype('d')
# coordinates0 = copy.deepcopy(wing.getSurfacePoints(patchID,uv))

# wing.DV_listGlobal[idg['span']].value = 1 
# # wing.DV_listGlobal[idg['twist']].value = 0
# # wing.DV_listGlobal[idg['sweep']].value = .0 
# # wing.DV_listGlobal[idg['flap']].value = 0.0
# # wing.DV_listLocal[idl['surface1']].value[0]= 0+dx



wing.update()
wing.checkCoef()
wing.writeTecplot('wing2.dat',ref_axis=True,links=True)
print wing.ref_axis[0].rot
sys.exit(0)







coefdx = wing.coef.astype('d')
coordinatesdx = copy.deepcopy(wing.getSurfacePoints(patchID,uv))

# # Get The full vector 
dx1 = wing.C[:,3]
dx2 = (coordinatesdx-coordinates0)/(dx)

#dx1 = wing.J1[:,54]
#dx2 = ((coefdx-coef0)/dx).flatten()

f1 = open('dx1','w')
f2 = open('dx2','w')
print 'sizes:',len(dx1),len(dx2)
for i in xrange(len(dx1)):
    if not dx1[i] == 0:
        f1.write('%20.16f \n'%(dx1[i]))

    if not dx2[i] == 0:
        f2.write('%20.16f \n'%(dx2[i]))
        
f1.close()
f2.close()

#wing.writeTecplot('wing2.dat',ref_axis=True,links=True)
#wing.writeIGES('wing_mod.igs')

#tacs_geo, tacs_surfs = wing.createTACSGeo()
