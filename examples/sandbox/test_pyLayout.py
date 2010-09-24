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
naf=3
airfoil_list = ['naca2412.dat','naca2412.dat','naca2412.dat']
chord = [1.3,.7,.45]
x = [-.4,.175,1.250]
y = [0,0,.3]
z = [0,1.5,4]
rot_x = [0,0,8]
rot_y = [0,0,0]
rot_z = [0,2,5]

offset = zeros((naf,2))
offset[:,0] = 0 # Offset sections by 0.25 in x
Nctl = 11

# ------------------------------------------------------------------
#Note: u direction is chordwise, v direction is span-wise
wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
                  scale=chord,offset=offset,x=x,y=y,z=z,
                  rot_x=rot_x,rot_y=rot_y,rot_z=rot_z,
                  Nctl=Nctl,k_span=2)

wing.writeTecplot('wing.dat',edge_labels=True)

print '---------------------------'
print '      pyLayout Setup' 
print '---------------------------'

wing_box = pyLayout2.Layout(wing,[2])

le_list = array([[-.2,0.0,.03],[1.30,0.0,3.9]])
te_list = array([[.3,0.0,.03],[1.60,0.0,3.9]])

nrib = 13
nspar = 4
domain = pyLayout2.domain(le_list,te_list,k=2)
rib_space = 4*ones(nspar+1,'intc')
span_space = 4*ones(nrib-1,'intc')
v_space = 5
rib_blank = ones((nrib,nspar-1))
rib_pos_para = linspace(0,1,nrib)

rib_pos_para[4] = 0.37

rib_blank[2,1] = 0
timeA = time.time()
struct_def1 = pyLayout2.struct_def(\
    nrib,nspar,domain=domain,t=0.01,rib_pos_para=rib_pos_para,
    rib_space=rib_space,span_space=span_space,v_space=v_space,rib_blank=rib_blank)

wing_box.addSection(struct_def1)
wing_box.writeTecplot('def1.dat')
structure = wing_box.finalize2()
print 'time:',time.time()-timeA
