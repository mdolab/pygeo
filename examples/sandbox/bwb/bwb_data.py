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
    lexsort, savetxt, append, zeros_like

# =============================================================================
# Extension modules
# =============================================================================

# pySpline 
sys.path.append('../../../pySpline/python')
import pySpline
# This script generates the original digitized data for the bwb planform
# Wing Information - Create a Geometry Object from cross sections

naf=22
n0012 = '../../input/naca0012.dat'

# Use the digitize it data for the planform:
le = array(loadtxt('bwb_le.out'))
te = array(loadtxt('bwb_te.out'))
front_up = array(loadtxt('bwb_front_up.out'))
front_low = array(loadtxt('bwb_front_low.out'))

le[0,:] = 0
te[0,0] = 0
front_up[0,0] = 0
front_low[0,0] = 0

# Now make a ONE-DIMENSIONAL spline for each of the le and trailing edes
le_spline = pySpline.linear_spline(task='lms',k=4,X=le[:,1],s=le[:,0],Nctl=20)
te_spline = pySpline.linear_spline(task='lms',k=4,X=te[:,1],s=te[:,0],Nctl=20)
up_spline = pySpline.linear_spline(task='lms',k=4,X=front_up[:,1],s=front_up[:,0],Nctl=20)
low_spline = pySpline.linear_spline(task='lms',k=4,X=front_low[:,1],s=front_low[:,0],Nctl=20)

# Generate consistent equally spaced spline data
span = linspace(0,138,naf-2)
span = hstack([linspace(0,8,5),linspace(10,138,naf-7)])
le = le_spline.getValueV(span)
te = te_spline.getValueV(span)
up = up_spline.getValueV(span)
low = low_spline.getValueV(span)

# plot(span,-le,'ko')
# show()
# sys.exit(0)

chord = te-le
x = le
z = span
mid_y = (up[0]+low[0])/2.0
y = -(up+low)/2 + mid_y

# This will be usedful later as the mid surface value
mid_y_spline = pySpline.linear_spline(task='lms',k=4,X=y,s=span,Nctl=20)


# Estimate t/c ratio
tc = (low-up)/chord # Array division
points0 = loadtxt(n0012)
airfoil_list = []
points = zeros(points0.shape)
points[:,0] = points0[:,0]
for i in xrange(naf-2):
    scale_y = tc/.12
    points[:,1] = points0[:,1]*scale_y[i]
    savetxt('../../input/autogen_input/%d.dat'%(i),points, fmt="%12.6G")
    airfoil_list.append('../../input/autogen_input/%d.dat'%(i))
# end for

airfoil_list.append(airfoil_list[-1])
airfoil_list.append(airfoil_list[-1])

# Now append two extra cross sections for the winglet
chord = append(chord,[chord[-1]*.90,.3*chord[-1]])
x=append(x,[x[-1] + 2,x[-1] + 25])
z=append(z,[140,144.5])
y=append(y,[y[-1] + 1.5,y[-1] + 15])

rot_x = zeros(naf)
rot_x[-1] = -80
rot_x[-2] = -80
rot_y = zeros(naf)
rot_z = zeros(naf)
offset = zeros((naf,2))
Nctlu = 11
                      
# Make the break-point vector
breaks = [10,19,20]
cont = [0,-1,1] # vector of length breaks: 0 for c0 continuity 1 for c1 continutiy
nsections = [25,25,10,10] # length of breaks +1

# Put spatial and rotations into two arrays (always the same)-------
X = zeros((naf,3))
rot = zeros((naf,3))

SCALE = chord[0]
#SCALE = 1 #do NOT Scale
X[:,0] = x/SCALE
X[:,1] = y/SCALE
X[:,2] = z/SCALE
rot[:,0] = rot_x
rot[:,1] = rot_y
rot[:,2] = rot_z
chord/=SCALE
