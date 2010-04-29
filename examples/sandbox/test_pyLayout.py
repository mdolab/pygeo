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
from mdo_import_helper import *
exec(import_modules('pyGeo','pyLayout2'))

# Wing Information - Create a Geometry Object from cross sections
naf=2
airfoil_list = ['naca2412.dat','naca2412.dat']
chord = [1,1]
x = [0,0]
y = [0,0]
z = [0,2]
rot_x = [0,0]
rot_y = [0,0]
rot_z = [0,0]

offset = zeros((naf,2))
offset[:,0] = 0 # Offset sections by 0.25 in x
Nctl = 11

# ------------------------------------------------------------------
#Note: u direction is chordwise, v direction is span-wise
wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
                  scale=chord,offset=offset,x=x,y=y,z=z,
                  rot_x=rot_x,rot_y=rot_y,rot_z=rot_z,
                  Nctl=Nctl)

wing.writeTecplot('wing.dat',edge_labels=True)

print '---------------------------'
print '      pyLayout Setup' 
print '---------------------------'

wing_box = pyLayout2.Layout(wing,[2])

le_list = array([[.15,0,0],[.30,0,1],[.45,0,2]])
te_list = array([[.80,0,0],[.60,0,1],[.75,0,2]])

nrib = 3
nspar = 2
domain = pyLayout2.domain(le_list,te_list,k_te=3,k_le=3)
rib_space = [1,1,2]
span_space = [2,2]
v_space = 2
struct_def1 = pyLayout2.struct_def(\
    nrib,nspar,domain=domain,t=0.01,
    rib_space=rib_space,span_space=span_space,v_space=v_space)

wing_box.addSection(struct_def1)
wing_box.writeTecplot('def1.dat')
wing_box.finalize2()
