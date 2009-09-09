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
# aircraft = pyGeo.pyGeo('plot3d',file_name='dpw.xyz')
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

coef0 = copy.deepcopy(aircraft.surfs[12].coef)
def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
                       
#    ref_axis[0].x[:,1] = ref_axis[0].x0[:,1]*1.5

    ref_axis[0].rot[:,1] = val
    ref_axis[1].rot[:,1] = val
    return ref_axis


print ' ** Adding Global Design Variables **'
aircraft.addGeoDVGlobal('span',1,0.5,2.0,span_extension)

idg = aircraft.DV_namesGlobal #NOTE: This is constant (idg -> id global
print 'idg',idg

aircraft.DV_listGlobal[idg['span']].value =5

aircraft.update()
#aircraft.checkCoef()
aircraft.writeTecplot('dpw.dat',edges=True,links=True)



# Start of mesh warping testing



coef = copy.deepcopy(aircraft.surfs[12].coef)
Nu = coef.shape[0]
Nv = coef.shape[1]


file_name = 'warp_test.dat'
f = open(file_name ,'w')
f.write ('VARIABLES = "X", "Y","Z"\n')
f.write('Zone T=%s I=%d J=%d\n'%('coef',Nu,Nv))
f.write('DATAPACKING=POINT\n')
for j in xrange(Nv):
    for i in xrange(Nu):
        f.write('%f %f %f \n'%(coef0[i,j,0],coef0[i,j,1],coef0[i,j,2]))
    # end for
# end for

# Pluck out a section of coefficients
# Now parameterize it:

S = zeros([Nu,Nv,2])

# The low ends should be set
# Find DelI and DelJ
def delI(i,j,vals):
    return sqrt( ( vals[i,j,0]-vals[i-1,j,0]) ** 2 + \
                  (vals[i,j,1]-vals[i-1,j,1]) ** 2 + \
                  (vals[i,j,2]-vals[i-1,j,2]) ** 2)

def delJ(i,j,vals):
    return sqrt( ( vals[i,j,0]-vals[i,j-1,0]) ** 2 + \
                  (vals[i,j,1]-vals[i,j-1,1]) ** 2 + \
                  (vals[i,j,2]-vals[i,j-1,2]) ** 2)


for i in xrange(1,Nu):
    for j in xrange(1,Nv):
        S[i,j,0] = S[i-1,j  ,0] + delI(i,j,coef)
        S[i,j,1] = S[i  ,j-1,1] + delJ(i,j,coef)

for i in xrange(1,Nu):
    S[i,0,0] = S[i-1,0,0] + delI(i,0,coef)
for j in xrange(1,Nv):
    S[0,j,1] = S[0,j-1,1] + delJ(0,j,coef)


# Do a no-check normalization
for i in xrange(Nu):
    for j in xrange(Nv):
        S[i,j,0] /= S[-1,j,0]
        S[i,j,1] /= S[i,-1,1]

dface = zeros((Nu,Nv,3))

# Set up corner perturbations:

dface[0,0] = coef[0,0]-coef0[0,0]
dface[0,-1] = coef[0,-1]-coef0[0,-1]
dface[-1,0] = coef[-1,0]-coef0[-1,0]
dface[-1,-1] = coef[-1,-1]-coef0[-1,-1]

# Edge 0
for i in xrange(1,Nu):
    j = 0
    WTK2 = S[i,j,0]
    WTK1 = 1.0-WTK2
    
    dface[i,j] = WTK1 * dface[0,j] + WTK2 * dface[-1,j]

# Edge 1
for i in xrange(1,Nu):
    j = -1
    WTK2 = S[i,j,0]
    WTK1 = 1.0-WTK2
    dface[i,j] = WTK1 * dface[0,j] + WTK2 * dface[-1,j]

# Edge 1
for j in xrange(1,Nv):
    i=0
    WTK2 = S[i,j,1]
    WTK1 = 1.0-WTK2
    dface[i,j] = WTK1 * dface[i,0] + WTK2 * dface[i,-1]

# Edge 1
for j in xrange(1,Nv):
    i=-1
    WTK2 = S[i,j,1]
    WTK1 = 1.0-WTK2
    dface[i,j] = WTK1 * dface[i,0] + WTK2 * dface[i,-1]

eps = 1.0e-14

for i in xrange(1,Nu-1):
    for j in xrange(1,Nv-1):
        WTI2 = S[i,j,0]
        WTI1 = 1.0-WTI2
        WTJ2 = S[i,j,1]
        WTJ1 = 1.0-WTJ2
        deli = WTI1 * dface[0,j,0] + WTI2 * dface[-1,j,0]
        delj = WTJ1 * dface[i,0,0] + WTJ2 * dface[i,-1,0]
       
        dface[i,j,0] = (abs(deli)*deli + abs(delj)*delj)/  \
            max( ( abs (deli) + abs(delj),eps))
            

        deli = WTI1 * dface[0,j,1] + WTI2 * dface[-1,j,1]
        delj = WTJ1 * dface[i,0,1] + WTJ2 * dface[i,-1,1]
        
        dface[i,j,1] = (abs(deli)*deli + abs(delj)*delj)/ \
            max( ( abs (deli) + abs(delj),eps))

        deli = WTI1 * dface[0,j,2] + WTI2 * dface[-1,j,2]
        delj = WTJ1 * dface[i,0,2] + WTJ2 * dface[i,-1,2]
        
        dface[i,j,2] = (abs(deli)*deli + abs(delj)*delj)/  \
            max( ( abs (deli) + abs(delj),eps))
    # end for
# end for

# That was for the corners. Now do it for the edges

# Now subtract off the edge peturbations:

#Edge 0 adn Edge 1
for i in xrange(1,Nu-1):
    dface[i,0] = coef[i,0]-coef0[i,0]-dface[i,0]
    dface[i,-1] = coef[i,-1]-coef0[i,-1]-dface[i,-1]
    coef[i,0] -= dface[i,0]
    coef[i,-1] -= dface[i,-1]
#Edge 2 adn Edge 3
for j in xrange(1,Nv-1):
    dface[0,j] = coef[0,j]-coef0[0,j]-dface[0,j]
    dface[-1,j] = coef[-1,j]-coef0[-1,j]-dface[-1,j]
    coef[0,j] -= dface[0,j]
    coef[-1,j] -= dface[-1,j]

# Now do the inside peturbations again

for i in xrange(1,Nu-1):
    for j in range(1,Nv-1):
        WTI2 = S[i,j,0]
        WTI1 = 1.0-WTI2
        WTJ2 = S[i,j,1]
        WTJ1 = 1.0-WTJ2

        deli = WTI1 * dface[0,j,0] + WTI2 * dface[-1,j,0]
        delj = WTJ1 * dface[i,0,0] + WTJ2 * dface[i,-1,0]

        coef[i,j,0] = coef[i,j,0] + (coef[i,j,0] - coef0[i,j,0]) - dface[i,j,0] - deli -delj

        deli = WTI1 * dface[0,j,1] + WTI2 * dface[-1,j,1]
        delj = WTJ1 * dface[i,0,1] + WTJ2 * dface[i,-1,1]

        coef[i,j,1] = coef[i,j,1] + (coef[i,j,1] - coef0[i,j,1]) - dface[i,j,1] - deli - delj

        deli = WTI1 * dface[0,j,2] + WTI2 * dface[-1,j,2]
        delj = WTJ1 * dface[i,0,2] + WTJ2 * dface[i,-1,2]

        coef[i,j,2] = coef[i,j,2] + (coef[i,j,2] - coef0[i,j,2]) - dface[i,j,2] - deli - delj



f.write('Zone T=%s I=%d J=%d\n'%('coef_warp',Nu,Nv))
f.write('DATAPACKING=POINT\n')
for j in xrange(Nv):
    for i in xrange(Nu):
        f.write('%f %f %f \n'%(coef[i,j,0],coef[i,j,1],coef[i,j,2]))
    # end for
# end for
print 'done'


# Force set those coef in the global list:

l_list = aircraft.l_index[12]
for i in xrange(Nu):
    for j in xrange(Nv):
        aircraft.coef[l_list[i,j]] = coef[i,j]

aircraft._updateSurfaceCoef()
aircraft.writeTecplot('dpw.dat',edges=True,links=True)
