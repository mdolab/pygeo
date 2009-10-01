#!/usr/bin/python
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

# pySpline
sys.path.append('../../pySpline/python')

#pyGeo
sys.path.append('../')
import pyGeo_NM as pyGeo
from geo_utils import *

#ACDT Geometry
sys.path.append('../../../pyACDT/pyACDT/Geometry/')
from pyGeometry_object import GeoObject
from pyGeometry_liftingsurface import LiftingSurface
from pyGeometry_bodysurface import BodySurface
from pyGeometry_system import System
from pyGeometry_aircraft import Aircraft
from pyGeometry_airfoil import Airfoil
# ------------------------------------------
#          Boeing 737 Test Example
# ------------------------------------------

input = {
    'Name':'B737',
    '_components':{
        0:BodySurface({
                'Name':'Fuselage',
                'Type':'Closed',
                'xLoc':0.0,'yLoc':0.0,'zLoc':0.0,
                'xRot':0.0,'yRot':0.0,'zRot':0.0,
                '_components':{
                    0:{'Name':'Cockpit','Length':20.0,'fwd_Radius':0.0,'fwd_lShape':1.0,'aft_Radius':6.0,'aft_lShape':1.0,'_components':{0:{'BodyCockpit':{'a1':0.7,'c1':0.8,'a2':1.0,'b2':0.8}}}},
                    1:{'Name':'Cabin'  ,'Length':55.0,'fwd_Radius':6.0,'fwd_lShape':2.0,'aft_Radius':6.0,'aft_lShape':2.0,'_components':{0:{'BodyFairing':{'rad_Loc':'Low','fwd_Loc':0.3,'aft_Loc':0.7,'hRatio':1.0,'vRatio':1.2,'rShape':0.7,'lShape':2.0,'alpha':10.0}}}},
                    2:{'Name':'Tail'   ,'Length':30.0,'fwd_Radius':6.0,'fwd_lShape':1.0,'aft_Radius':1.0,'aft_Ratio':5.0,'aft_rShape':0.2,'aft_lShape':1.0,'aft_vOffset':3.5,'dySlope':0.5,'dzSlope':0.5,},
                    }
                }),
        1:LiftingSurface({
                'Name':'Wing',
                'Symmetry':True,
                'xrLE':35.00,'yrLE':0.0,'zrLE':-5.0,
                'xRot':0.0,'yRot':0.0,'zRot':0.0,
                '_components':{
                    0:{'Name':'Internal Segment','Type':'internal','Area':100.0,'Span':02.5,'Taper':0.7,'SweepLE':25.0,'Dihedral':0.0 ,'xc_offset':0.0,'root_Incidence': 1.0,'root_Thickness':0.14,'root_Airfoil_type':'naca5','root_Airfoil_ID':'230xx','tip_Incidence': 1.0,'tip_Thickness':0.14,'tip_Airfoil_type':'naca5','tip_Airfoil_ID':'230xx'},
                    1:{'Name':'Inner Segment'   ,'Type':'external','Area':220.0,'Span':15.0,'Taper':0.6,'SweepLE':25.0,'Dihedral':3.0 ,'xc_offset':0.0,'root_Incidence': 1.0,'root_Thickness':0.14,'root_Airfoil_type':'naca5','root_Airfoil_ID':'230xx','tip_Incidence':-2.0,'tip_Thickness':0.12,'tip_Airfoil_type':'naca5','tip_Airfoil_ID':'230xx'},
                    2:{'Name':'Outer Segment'   ,'Type':'external','Area':220.0,'Span':30.0,'Taper':0.4,'SweepLE':30.0,'Dihedral':6.0 ,'xc_offset':0.0,'root_Incidence':-2.0,'root_Thickness':0.12,'root_Airfoil_type':'naca5','root_Airfoil_ID':'230xx','tip_Incidence':-3.0,'tip_Thickness':0.11,'tip_Airfoil_type':'naca5','tip_Airfoil_ID':'230xx'},
                    3:{'Name':'Wingtip Segment' ,'Type':'wingtip' ,'Area':006.0,'Span':01.0,'Taper':0.7,'SweepLE':60.0,'Dihedral':6.0 ,'xc_offset':0.0,'root_Incidence':-3.0,'root_Thickness':0.11,'root_Airfoil_type':'naca5','root_Airfoil_ID':'230xx','tip_Incidence': 0.0,'tip_Thickness':0.10,'tip_Airfoil_type':'naca5','tip_Airfoil_ID':'230xx'},
                    4:{'Name':'Winglet Segment' ,'Type':'winglet' ,'Area':025.0,'Span':05.0,'Taper':0.5,'SweepLE':45.0,'Dihedral':90.0,'xc_offset':0.0,'root_Incidence': 0.0,'root_Thickness':0.10,'root_Airfoil_type':'naca5','root_Airfoil_ID':'230xx','tip_Incidence': 0.0,'tip_Thickness':0.00,'tip_Airfoil_type':'naca5','tip_Airfoil_ID':'230xx'},
                    },
                }),
        2:LiftingSurface({
                'Name':'Horizontal Tail',
                'Symmetry':True,
                'xrLE':93.45,'yrLE':0.0,'zrLE':3.50,
                'xRot':0.0,'yRot':0.0,'zRot':0.0,
                '_components':{
					0:{'Name':'External Segment','Type':'external','Area':168.5,'Span':20.8306,'Taper':0.26,'SweepLE':35.0,'Dihedral':7.0 ,'xc_offset':0.0,'root_Incidence':0.0,'root_Thickness':0.12,'root_Airfoil_type':'naca4','root_Airfoil_ID':'00xx','tip_Incidence':0.0,'tip_Thickness':0.00,'tip_Airfoil_type':'naca4','tip_Airfoil_ID':'00xx'},
                                        },
                }),
        3:LiftingSurface({
                'Name':'Vertical Tail',
                'Symmetry':False,
                'xrLE':84.85,'yrLE':0.0,'zrLE':3.45,
                'xRot':0.0,'yRot':0.0,'zRot':0.0,
                '_components':{
                    0:{'Name':'External Segment','Type':'external','Area':248.97,'Span':19.70,'Taper':0.31,'SweepLE':40.0,'Dihedral':90.0 ,'xc_offset':0.0,'root_Incidence':0.0,'root_Thickness':0.12,'root_Airfoil_type':'naca4','root_Airfoil_ID':'00xx','tip_Incidence':0.0,'tip_Thickness':0.00,'tip_Airfoil_type':'naca4','tip_Airfoil_ID':'00xx'},
                    }
                }),
        },
   }

# # ------------------------------------------
# #             MDA Wing Example
# # ------------------------------------------

# input = {
#     'Name':'MDA_Wing',
#     '_components':{
#     0:LiftingSurface({
#     'Name':'Wing',
#     'Symmetry':False,
#     'xrLE':0.00,'yrLE':0.0,'zrLE':0.0,
#     'xRot':0.0,'yRot':0.0,'zRot':0.0,
#     '_components':{
#     0:{'Name':'External Segment','Type':'internal','Area':3.9825,'Span':4.5,'Taper':0.77,'SweepLE':10.0,'Dihedral':0.0 ,'xc_offset':0.0,'root_Incidence': 0.0,'root_Thickness':0.12,'root_Airfoil_type':'naca4','root_Airfoil_ID':'00xx','tip_Incidence': 0.0,'tip_Thickness':0.12,'tip_Airfoil_type':'naca4','tip_Airfoil_ID':'00xx'},
#     1:{'Name':'External Segment','Type':'external','Area':0.38,'Span':0.5,'Taper':0.974,'SweepLE':10.0,'Dihedral':0.0 ,'xc_offset':0.0,'root_Incidence': 0.0,'root_Thickness':0.12,'root_Airfoil_type':'naca4','root_Airfoil_ID':'00xx','tip_Incidence': 0.0,'tip_Thickness':0.00,'tip_Airfoil_type':'naca4','tip_Airfoil_ID':'00xx'}}}),
#     }
#     }

geo_objects = createGeometryFromACDT(input,LiftingSurface,BodySurface,Airfoil,pyGeo)

print 'We have %d geo objects'%(len(geo_objects))

for i in xrange(len(geo_objects)):

    geo_objects[i].writeTecplot('../output/ACDT_output/%d.dat'%(i))

