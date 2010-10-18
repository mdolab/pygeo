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
exec(import_modules('pyGeo'))

# ==============================================================================
# Start of Script
# ==============================================================================

# Script to Generate a Wing Geometry

naf=3
airfoil_list = ['./geo_input/naca2412.dat','./geo_input/naca2412.dat','./geo_input/naca2412.dat']
Nctl = 13
chord = [1.67,1.67,1.18]
x = [0,0,.125*1.18]
y = [0,0,0]
z = [0,2.5,10.58/2]
rot_x = [0,0,0]
rot_y = [0,0,0]
rot_z = [0,0,0] 

offset = zeros((naf,2))

wing = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,
                   scale=chord,offset=offset,x=x,y=y,z=z,
                   rot_x=rot_x,rot_y=rot_y,rot_z=rot_z,Nctl=Nctl,
                   k_span=2,con_file='./geo_input/c172.con')

wing.setSymmetry('xy')
wing.writeTecplot('./geo_output/c172_geo.dat')
wing.writeIGES('./geo_input/c172.igs')

# Add the reference axis
nsec = 3
x = x + 0.25*array(chord)

# Add reference axis
wing.addRefAxis([0,1],x=x,y=y,z=z,rot_type=3) #Surface list then x,y,z
wing.writeTecplot('./geo_output/c172_geo.dat',orig=True,directions=True,
                      surf_labels=True,edge_labels=False,node_labels=False,
                      links=True,ref_axis=True)


def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
    ref_axis[0].x[:,1] = ref_axis[0].x0[:,1] * val
    return ref_axis

def outer_sweep(val,ref_axis):
    '''Single design variable for outer section sweep'''
    ref_axis[0].x[2,0] = ref_axis[0].x0[2,0] + val
    return ref_axis

def outer_twist(val,ref_axis):
    ref_axis[0].rot_y[2] = val
    return ref_axis

def outer_dihedral(val,ref_axis):
    ref_axis[0].x[2,2] = ref_axis[0].x0[2,2] + val
    return ref_axis

def tip_chord(val,ref_axis):
    ref_axis[0].scale[2] = ref_axis[0].scale0[2]*val
    return ref_axis

mpiPrint(' ** Adding Global Design Variables **')
wing.addGeoDVGlobal('span',1,0.5,2.0,span_extension)
wing.addGeoDVGlobal('outer_sweep',0,-1,1.0,outer_sweep)
wing.addGeoDVGlobal('outer_twist',0,-10,10.0,outer_twist)
wing.addGeoDVGlobal('outer_dihedral',0,-5,5.0,outer_dihedral)
wing.addGeoDVGlobal('tip_chord',1,.75,1.25,tip_chord)
idg = wing.DV_namesGlobal #NOTE: This is constant (idg -> id global
wing.DV_listGlobal[idg['span']].value = 1.2
wing.DV_listGlobal[idg['outer_sweep']].value = .2
wing.DV_listGlobal[idg['outer_twist']].value = -2
wing.DV_listGlobal[idg['outer_dihedral']].value = .3
wing.DV_listGlobal[idg['tip_chord']].value = .9
wing.update()

wing.writeTecplot('./geo_output/c172_geo_mod.dat',orig=True,directions=True,
                      surf_labels=True,edge_labels=True,node_labels=True,
                      links=True,ref_axis=True)
