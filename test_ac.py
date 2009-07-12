# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross, vstack

# =============================================================================
# Extension modules
# =============================================================================
from matplotlib.pylab import *
#pyOPT
sys.path.append(os.path.abspath('../../../../pyACDT/pyACDT/Optimization/pyOpt/'))

# pySpline
sys.path.append('../pySpline/python')

#pyOPT
sys.path.append(os.path.abspath('../../pyACDT/pyACDT/Optimization/pyOpt/'))

#pySNOPT
sys.path.append(os.path.abspath('../../pyACDT/pyACDT/Optimization/pyOpt/pySNOPT'))

#pyGeo
import pyGeo

# # Wind Turbine Blade

# naf = 10
# airfoil_list = ['af1-6.inp','af-07.inp','af8-9.inp','af8-9.inp','af-10.inp',\
#                     'af-11.inp','af-12.inp', 'af-14.inp',\
#                     'af15-16.inp','af15-16.inp']

# chord = array([.6440,1.0950,1.6800,\
#          1.5390,1.2540,0.9900,0.7900,0.4550,0.4540,0.4530])

# sloc = array([0.0000,0.1141,\
#         0.2184,0.3226,0.4268,0.5310,0.6352,0.8437,0.9479,1.0000])*10

# tw_aero = array([0.0,20.3900,16.0200,11.6500,\
#                 6.9600,1.9800,-1.8800,-3.4100,-3.4500,-3.4700])

# ref_axis = pyGeo.ref_axis(2.5*ones(naf),zeros(naf),sloc,zeros(naf),zeros(naf),tw_aero)

# le_loc = array([0.5000,0.3300,\
#           0.2500,0.2500,0.2500,0.2500,0.2500,0.2500,0.2500,0.2500])

# offset = zeros([naf,2])
# offset[:,0] = le_loc
# blade  = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis,fit_type='lms',Nctlu=13,Nctlv=naf)

# #blade.writeTecplot('wing.dat')
# #blade.writeIGES('wing.igs')

# blade.stitchPatches(1e-3,1e-2) #node_tol,edge_tol

# # Fuselage

# naf = 10
# airfoil_list = ['af1-6.inp','af1-6.inp','af1-6.inp','af1-6.inp','af1-6.inp',
#                 'af1-6.inp','af1-6.inp','af1-6.inp','af1-6.inp','af1-6.inp']

# chord = array([.1,.4,.45,.46,.45,.3,.15,.1,.08,.07])
# sloc  = linspace(0,5,naf)
# plot(sloc,chord)
# show()
# ref_axis = pyGeo.ref_axis(sloc,zeros(naf),zeros(naf),0*ones(naf),90*ones(naf),90*ones(naf))

# offset = zeros([naf,2])
# offset[:,0] = 0.5

# fuse  = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis,fit_type='lms',Nctlu=13,Nctlv=naf)
# fuse.writeTecplot('fuse.dat')
# fuse.writeIGES('fuse.igs')

# Wing
naf=4
airfoil_list = ['af15-16.inp','af15-16.inp','af15-16.inp','pinch.inp']
chord = [1.25,1,.8,.65]
tw_aero = [-4,0,4,4.5]
ref_axis = pyGeo.ref_axis([1.25,1.25,1.25,1.25],[0,0.1,0.2,0.4],[0,2,4,6],[00,00,00,0],[0,0,0,0],tw_aero)
offset = zeros((4,2))
offset[:,0] = .25 #1/4 chord
wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis,fit_type='lms',Nctlu = 5,Nctlv= naf)

#Corner
naf=4
airfoil_list = ['af15-16.inp','af15-16.inp','af15-16.inp','af15-16.inp']
chord = [.65,.65,.65,.65]

ref_axis = pyGeo.ref_axis([1.25,1.25,1.25,1.25],[0.4,.405,.55,.6],[6,6.05,6.20,6.20],[0,0,-90,-90],[0,0,0,0],[4.5,4.5,0,0])
offset = zeros((4,2))
offset[:,0] = .25 #1/4 chord
corner = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis,fit_type='lms',Nctlu = 13,Nctlv= naf)

#Winglet
naf=2
airfoil_list = ['af15-16.inp','pinch.inp']
chord = [.65,.65]

ref_axis = pyGeo.ref_axis([1.25,1.25],[.6,1.2],[6.2,6.2],[-90,-90],[0,0],[0,0])
offset = zeros((2,2))
offset[:,0] = .25 #1/4 chord

winglet = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis,fit_type='lms',Nctlu = 13,Nctlv= naf)

# # Now add everything to the wing:
# wing.addGeoObject(corner)
# del corner
# wing.addGeoObject(winglet)
# del winglet

 #wing.calcEdgeConnectivity(1e-2,1e-2)

# wing.loadEdgeConnectivity('test.con')
# wing.propagateKnotVectors()
# wing.stitchEdges()
# wing.writeTecplot('wing.dat')
# wing.writeIGES('wing.igs')

# for i in xrange(wing.nPatch):
#     print wing.surfs[i].master_edge


wing.calcEdgeConnectivity(1e-2,1e-2)
sys.exit(0)
#wing.writeEdgeConnectivity('test2.con')
wing.loadEdgeConnectivity('test2.con')
wing.propagateKnotVectors()
wing.stitchEdges()
wing.fitSurfaces()
wing.writeTecplot('wing.dat')

for i in xrange(wing.nPatch):
    print i,wing.surfs[i].master_edge,wing.surfs[i].Nu,wing.surfs[i].Nv,wing.surfs[i].Nu_free*wing.surfs[i].Nv_free


# wing = pyGeo.pyGeo('iges',file_name='wing.igs')
# wing.loadEdgeConnectivity('test.con')
# wing.propagateKnotVectors()
# #Qwing.stitchEdges()
# wing.writeTecplot('wing2.dat')


