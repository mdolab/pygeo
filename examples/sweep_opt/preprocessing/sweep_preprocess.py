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
    savetxt,append

# =============================================================================
# Extension modules
# =============================================================================
from mdo_import_helper import *
exec(import_modules('pyGeo','pySpline'))

# ==============================================================================
# Start of Script
# ==============================================================================

# Script to Generate a Wing Geometry


airfoil_list = ['foil.dat',
                'foil.dat']

naf=len(airfoil_list)

Nctl = 13
chord = array([1,1])

x = [0,0]
y = [0,0]
z = [0,5]
rot_x = zeros(naf)
rot_y = zeros(naf)
rot_z = zeros(naf)

offset = zeros((naf,2))

wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
                   scale=chord,offset=offset,x=x,y=y,z=z,
                   rot_x=rot_x,rot_y=rot_y,rot_z=rot_z,Nctl=Nctl,
                   k_span=2,con_file='sweep.con',tip='rounded')
wing.setSymmetry('xy')
wing.writeIGES('sweep.igs')
wing.writeTecplot('sweep.dat')
