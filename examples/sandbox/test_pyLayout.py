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

# ==============================================================================
# Start of Script
# ==============================================================================

# Wing Information - Create a Geometry Object from cross sections

naf=2
airfoil_list = ['naca0012.dat','naca0012.dat']#,'naca0012.dat']
chord = [1,1]#75,5]
x = [0,0]#,0]
y = [0,0]#,0]
z = [0,2]#,2]
rot_x = [0,0]#,0]
rot_y = [0,0]#,0]
rot_z = [0,0]#,0]

offset = zeros((naf,2))
offset[:,0] = 0 # Offset sections by 0.25 in x
# Make the break-point vector
Nctl = 11

# ------------------------------------------------------------------
#Note: u direction is chordwise, v direction is span-wise
# wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
#                   scale=chord,offset=offset,x=x,y=y,z=z,
#                   rot_x=rot_x,rot_y=rot_y,rot_z=rot_z,
#                   Nctl=Nctl)
# wing.writeIGES('wing.igs')
wing = pyGeo.pyGeo('iges',file_name='wing.igs')
wing.doEdgeConnectivity('default.con')
wing.writeTecplot('./wing.dat',edge_labels=True)

# ------------------------------------------------------------------
print '---------------------------'
print 'Attaching Reference Axis...'
print '---------------------------'

wing.addRefAxis([0,1],x=x,y=y,z=z,rot_x=rot_x,rot_y=rot_y,rot_z=rot_z)
print 'Done Ref Axis Adding!'

print '---------------------------'
print '      pyLayout Setup' 
print '---------------------------'

nsection = 1
wing_box = pyLayout2.Layout(wing,nsection,te_list=[2])

le_list = array([[.25,0,0],[.25,0,2]])
te_list = array([[.75,0,0],[.75,0,2]])

domain1 = pyLayout2.domain(le_list,te_list)
rib_space = [1,1,1,1]
span_space = [1]
X = zeros((2,3,3))
struct_def1 = pyLayout2.struct_def(2,3,domain1,ref_axis=wing.ref_axis[0],
                                   rib_space=rib_space,span_space=span_space)


wing_box.addSection(struct_def1)
wing_box.writeTecplot('def1.dat')
wing_box.finalize2()
