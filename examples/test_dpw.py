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

# pySpline 
sys.path.append('../../pySpline/python')
import pySpline

#cfd-csm pre (Optional)
sys.path.append('../../../../pyHF/pycfd-csm/python/')

#pyGeo
sys.path.append('../')
import pyGeo
from geo_utils import *
# This script reads a surfaced-based plot3d file as typically
# outputted by aerosurf. It then creates a b-spline surfaces for each
# surface patch.
# timeA = time.time()
# aircraft = pyGeo.pyGeo('plot3d',file_name='../input/dpw.xyz')
# aircraft.calcEdgeConnectivity()
# aircraft.writeEdgeConnectivity('dpw.con')
# aircraft.propagateKnotVectors()
# aircraft.fitSurfaces()
# timeA = time.time()

# aircraft.writeTecplot('../output/dpw.dat',edges=True)
# timeB =time.time()
# print 'Write time is:',timeB-timeA
# print 'full time',time.time()-timeA

# aircraft.writeIGES('../input/dpw.igs')
# sys.exit(0)

aircraft = pyGeo.pyGeo('iges',file_name='../input/dpw.igs',no_print=False)
aircraft.readEdgeConnectivity('dpw.con')

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

# Add l_surfs:
#aircraft.l_surfs.append([2,3,9,8]) # Inner Wing Panels
#aircraft.l_surfs.append([4,5,10,11]) # Outer Wing Panels

aircraft.addRefAxis([2,3,8,9],X[0:2,:],rot[0:2,:],nrefsecs=6)
aircraft.addRefAxis([4,5,10,11,16,17],X[1:3,:],rot[1:3,:],nrefsecs=6)
aircraft.addRefAxisCon(0,1,'end') # Innter Wing and Outer Wing ('Attach ra1 to ra0 with type 'end')

def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
    #ref_axis[0].rot[:,1] = val
    #ref_axis[1].rot[:,1] = val
    #ref_axis[0].scale[:] = 1.2
    #ref_axis[0].x[:,2] += 30
   
    return ref_axis


print ' ** Adding Global Design Variables **'
aircraft.addGeoDVGlobal('span',1,0.5,2.0,span_extension)

idg = aircraft.DV_namesGlobal #NOTE: This is constant (idg -> id global
print 'idg',idg

aircraft.DV_listGlobal[idg['span']].value =0
# Create a coef0 list for all surfaces:
init_coef = []
coef_temp = []
for isurf in xrange(aircraft.nSurf):
    init_coef.append(copy.deepcopy(aircraft.surfs[isurf].coef)) # Pluck out coef before update
    coef_temp.append(zeros(shape(aircraft.surfs[isurf].coef)))

aircraft.update()


# Start of mesh warping testing
# Pluck out a section of coefficients
# Now parameterize it:

surfaces_to_warp = [0,1,6,7,12,13]
surfaces_to_warp = [0,12]#,13]#0,1,6,7,12,13]
timeA = time.time()

for isurf in surfaces_to_warp:
    # Check ONLY the edges
    coef0 = copy.deepcopy(init_coef[isurf])
    coef  = aircraft.surfs[isurf].coef

    print 'isurf:',isurf
    Nu = aircraft.surfs[isurf].Nctlu
    Nv = aircraft.surfs[isurf].Nctlv

    S = parameterizeFace(Nu,Nv,coef0)
    dface = zeros((Nu,Nv,2))
       
    # Spline the edges:
    # Create the matrix of spline objects for all the lines:
    u_lines = []
    v_lines = []
    for j in xrange(Nv):
        # Make the u-lines:
        u_lines.append(pySpline.linear_spline('interpolate',X=coef0[:,j],k=2))
    for i in xrange(Nu):
        v_lines.append(pySpline.linear_spline('interpolate',X=coef0[i,:],k=2))
       

    # Now we can get the all the parameteric dface peturbation on the edges
    timeB = time.time()
    # Edge 0 and Edge 1
    counter = 0
    for i in xrange(Nu):
        if not alltrue(coef[i,0]==coef0[i,0]):
           #  print 'here0'
            u,D,converged,update = u_lines[0].projectPoint(coef[i,0])
            v,D,converged,update = v_lines[i].projectPoint(coef[i,0])
            dface[i,0] = [u-S[i,0,0],v-S[i,0,1]]
            counter += 1

        if not alltrue(coef[i,-1]==coef0[i,-1]):
           # print 'here1'
            u,D,converged,update = u_lines[-1].projectPoint(coef[i,-1])
            v,D,converged,update = v_lines[i].projectPoint(coef[i,-1])
            dface[i,-1] = [u-S[i,-1,0],v-S[i,-1,1]]
            counter += 1
    # end for

    # Edge 2 and Edge 3
    for j in xrange(Nv):
        if not alltrue(coef[0,j]==coef0[0,j]):
            #print 'here2'
            
            u,D,converged,update = u_lines[j].projectPoint(coef[0,j])
            v,D,converged,update = v_lines[0].projectPoint(coef[0,j])
           
            dface[0,j] = [u-S[0,j,0],v-S[0,j,1]]
            counter += 1
        if not alltrue(coef[-1,j]==coef0[-1,j]):
            #print 'here3'
            u,D,converged,update = u_lines[j].projectPoint(coef[-1,j])
            v,D,converged,update = v_lines[-1].projectPoint(coef[-1,j])
            dface[-1,j] = [u-S[-1,j,0],v-S[-1,j,1]]
            counter += 1
    # end for
   # print 'counter:',counter


    # Now interpolate the rest of dface with the parametric algebraic
    # warping algorithim
    timeC = time.time()
    dface = warp_face(Nu,Nv,S,dface)
    timeD = time.time()
    # Now we know the 'parametric movements' we can make splines for each 'line' and move 'em accordingly
    for i in xrange(Nu):
        for j in xrange(Nv):
            # Physical Change caused by du and dv
            dx_u = u_lines[j].getValue(S[i,j,0]+dface[i,j,0]) - coef0[i,j]
            dx_v = v_lines[i].getValue(S[i,j,1]+dface[i,j,1]) - coef0[i,j]
            coef[i,j] = coef0[i,j] + dx_u + dx_v
        # end for
    # end for
            
    coef_temp[isurf] = coef
    timeE = time.time()
# end for (isurf loop)


print 'Init Time:',timeB-timeA
print 'Project Time:',timeC-timeB
print 'Warp Time:',timeD-timeC
print 'Set time:',timeE-timeD

 # # Force set those coef in the global list:
for isurf in surfaces_to_warp:
    l_list = aircraft.l_index[isurf]
    for i in xrange(Nu):
        for j in xrange(Nv):
            aircraft.coef[l_list[i,j]] = coef_temp[isurf][i,j]
        # end for
    # end for
aircraft._updateSurfaceCoef()

aircraft.writeTecplot('../output/dpw.dat',edges=True,directions=True,
                      labels=True,links=True)#,size=5)
