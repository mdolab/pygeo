#!/usr/bin/python
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, zeros, ones, array

import petsc4py
petsc4py.init(sys.argv)

# =============================================================================
# Extension modules
# =============================================================================

# pySpline 
sys.path.append('../../pySpline/python')

# pyOpt
sys.path.append('../../../pyACDT/pyACDT/Optimization/pyOpt')

# pySnopt
sys.path.append('../../../pyACDT/pyACDT/Optimization/pyOpt/pySNOPT')

#pyGeo
sys.path.append('../')
import pyGeo_NM as pyGeo

# ==============================================================================
# Start of Script
# ==============================================================================

# Wing Information - Create a Geometry Object from cross sections

naf=2
Nctlu = 13
airfoil_list = ['../input/naca0012.dat','../input/naca0012.dat']
#airfoil_list = ['../input/af15-16.inp','../input/af15-16.inp']
chord = [1,.51]
x = [0,0]
y = [0,0]
z = [0,4]
rot_x = [0,0]
rot_y = [0,0]
rot_z = [0,0]

offset = zeros((naf,2))
offset[:,0] = .25
nsections = [18]
# Make the break-point vector

                               
# Put spatial and rotations into two arrays (always the same)-------
X = zeros((naf,3))
rot = zeros((naf,3))

X[:,0] = x
X[:,1] = y
X[:,2] = z
rot[:,0] = rot_x
rot[:,1] = rot_y
rot[:,2] = rot_z
# ------------------------------------------------------------------
    
# Procedure for Using pyGEO

# Step 1: Run the folloiwng Commands: (Uncomment between -------)
# ---------------------------------------------------------------------
#Note: u direction is chordwise, v direction is span-wise
# wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
#                    file_type='xfoil',scale=chord,offset=offset, 
#                    nsections=nsections, Xsec=X,rot=rot,
#                    fit_type='lms',Nctlu=Nctlu,Nfoil=45)

# wing.calcEdgeConnectivity(1e-6,1e-6)
# wing.writeEdgeConnectivity('wing_fit_test.con')
# wing.printEdgeConnectivity()
# wing.propagateKnotVectors()
# wing.writeTecplot('../output/wing_fit_test.dat')
# wing.writeIGES('../input/wing_fit_test.igs')
# print 'Done Step 1'

# Step 2: Run the folloiwng Commands: (Uncomment between -------)
# ---------------------------------------------------------------------

wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
                   file_type='xfoil',scale=chord,offset=offset, 
                   nsections=nsections, Xsec=X,rot=rot,end_type='rounded',
                   fit_type='lms',Nctlu=Nctlu,Nfoil=45)
wing.setSymmetry('xy')
#wing.calcEdgeConnectivity(1e-6,1e-6)
#wing.writeEdgeConnectivity('wing_fit_test.con')
#sys.exit(0)
wing.readEdgeConnectivity('wing_fit_test.con')
wing.printEdgeConnectivity()
wing.propagateKnotVectors()
wing.fitSurfaces3()
wing.writeTecplot('../output/wing_fit_test.dat',orig=True)
sys.exit(0)

Nctlu = wing.surfs[0].Nctlu
Nctlv = wing.surfs[0].Nctlv

gpts = wing.surfs[0].getGrevillePoints(2)
print 'gpts:',gpts
print 'knots:',wing.surfs[0].tv

pt = 4

du,dv = wing.surfs[0].getDerivative(0,gpts[pt])

ku = wing.surfs[0].ku

ileftu, mflagu = wing.surfs[0].pyspline.intrv(wing.surfs[0].tv,gpts[pt],1)
if mflagu == 0: # Its Inside so everything is ok
    u_list = [ileftu-ku,ileftu-ku+1,ileftu-ku+2,ileftu-ku+3]
if mflagu == 1: # Its at the right end so just need last two
    u_list = [ileftu-ku-2,ileftu-ku-1]
    
print 'u_list:',u_list

for i in xrange(Nctlu):
    temp = zeros((Nctlv,3))
    temp2 = zeros((Nctlv,3))
    for j in xrange(Nctlv):
        for ii in xrange(3):
            wing.surfs[0].coef[i,j,ii] += 1e-6
            dupx,dvpx = wing.surfs[0].getDerivative(0,gpts[pt])
            dudx,dvdx = wing.surfs[0].calcDerivativeDeriv(0,gpts[pt],i,j)
            temp[j,ii] = (dupx[ii]-du[ii])/1e-6
            temp2[j,ii] = dudx
            wing.surfs[0].coef[i,j,ii] -= 1e-6
    # end for
    print temp[:,0]
    print temp2[:,0]
print 'Done Step 2'
