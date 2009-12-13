#!/usr/bin/python
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross, vstack, arctan2, tan, loadtxt,\
    lexsort,savetxt,append

import petsc4py
petsc4py.init(sys.argv)

# =============================================================================
# Extension modules
# =============================================================================
#pyPSG:
sys.path.append(os.path.abspath('../../../pySpline/python'))
sys.path.append(os.path.abspath('../../')) # pyGeo & geo_utils
sys.path.append(os.path.abspath('../../../pyLayout/'))

import pySpline
import pyGeo

bwb = pyGeo.pyGeo('iges',file_name='../../input/bwb_constr2.igs')
bwb.readEdgeConnectivity('bwb.con')
# Make the dummy default surface
naf=5
airfoil_list = ['../../input/naca0012.dat','../../input/naca0012.dat',
                '../../input/naca0012.dat','../../input/naca0012.dat',
                '../../input/naca0012.dat']
chord = [1,1,1,1,1]
x = [0,0,0,0,0]
y = [0,0,0,0,0]
z = [0,.25,.5,.75,1]
rot_x = [0,0,0,0,0]
rot_y = [0,0,0,0,0]
tw_aero = [0,0,0,0,0] # ie rot_z

offset = zeros((naf,2))
# Make the break-point vector
breaks = [1,2,3] # Station where surfaces are to be split; zero based (Must NOT contain 0 or index of last value (naf-1))
nsections = [25,25,10,10]
section_spacing = [linspace(0,1,10),linspace(0,1,10)]
Nctlu = 11
end_type = 'rounded'
                               
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
                       Xsec=X,rot=rot,breaks=breaks,end_type=end_type,\
                       nsections=nsections,fit_type='lms', Nctlu=Nctlu,Nfoil=45)

wing.calcEdgeConnectivity(1e-6,1e-6)
wing.propagateKnotVectors()
#wing.writeTecplot('./start.dat')


# Get the delta's for the coefficients
N = 100
delta = (bwb.coef-wing.coef)/(N-1.0)

for i in xrange(N):
    file_name = './movie_data/increment_%3d.dat'%(i)
    if i>0:
        wing.coef += delta
        wing.update()
    # end if
    print 'Writing file %d'%(i)
    wing.writeTecplot(file_name)
# end if
