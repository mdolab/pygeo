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
    'Name':'Rutan Long Ez',
    '_components':{
        0:BodySurface({
            'Name':'Body',
            'Type':'Closed',
            'xLoc':-0.333,'yLoc':0.0,'zLoc':0.0,
            'xRot':0.0,'yRot':0.0,'zRot':0.0,
            '_components':{
                0:{'Name':'Section 0','Length':04.33,'fwd_Radius':0.00,'fwd_Ratio':1.000,'fwd_rShape':0.5,'fwd_lShape':1.6,'fwd_vOffset':0.0,'aft_Radius':0.986,'aft_Ratio':1.000,'aft_rShape':0.5,'aft_lShape':1.0,'aft_vOffset':0.5,'dySlope':1.0,'dzSlope':0.70},
                1:{'Name':'Section 1','Length':06.5,'fwd_Radius':0.986,'fwd_Ratio':1.000,'fwd_rShape':0.5,'fwd_lShape':1.0,'fwd_vOffset': 0.0,'aft_Radius':0.986,'aft_Ratio':1.0,'aft_rShape':0.5,'aft_lShape':0.5,'aft_vOffset': 0.0},
                2:{'Name':'Section 2','Length':3.0,'fwd_Radius':0.986,'fwd_Ratio':1.000,'fwd_rShape':0.5,'fwd_lShape':1.0,'fwd_vOffset': 0.00,'aft_Radius':0.020,'aft_Ratio':1.0,'aft_rShape':0.50,'aft_lShape':1.0,'aft_vOffset': 0.30,'dySlope':0.5,'dzSlope':0.35},
            }
        }),
        1:LiftingSurface({
            'Name':'Wing',
            'Symmetry':True,
            'xrLE':2.23,'yrLE':0.0,'zrLE':0.8166,
            'xRot':0.0,'yRot':0.0,'zRot':0.0,
            '_components':{
                0:{'Name':'Segment 1','Type':'external','Area':17.5,'Span':2,'Taper':0.57,'SweepLE':65.0,'Dihedral':-1.5,'xc_offset':0.0,'root_Incidence':0.5,'root_Thickness':0.07,'root_Airfoil_type':'naca4','root_Airfoil_ID':'00xx','tip_Incidence':0.50,'tip_Thickness':0.10,'tip_Airfoil_type':'datafile','tip_Airfoil_ID':'e1230.dat'},
                1:{'Name':'Segment 1','Type':'external','Area':14.8779,'Span':3.0,'Taper':0.5454,'SweepLE':50.0,'Dihedral':-1.5,'xc_offset':0.0,'root_Incidence':0.5,'root_Thickness':0.10,'root_Airfoil_type':'datafile','root_Airfoil_ID':'e1230.dat','tip_Incidence':0.50,'tip_Thickness':0.16,'tip_Airfoil_type':'datafile','tip_Airfoil_ID':'e1230.dat'},
                2:{'Name':'Segment 1','Type':'external','Area':22.84,'Span':8.579,'Taper':0.5,'SweepLE':25.0,'Dihedral':-1.5,'xc_offset':0.0,'root_Incidence':0.5,'root_Thickness':0.16,'root_Airfoil_type':'datafile','root_Airfoil_ID':'e1230.dat','tip_Incidence':0.50,'tip_Thickness':0.16,'tip_Airfoil_type':'datafile','tip_Airfoil_ID':'e1230.dat'},
                3:{'Name':'Segment 1','Type':'winglet','Area':6.5,'Span':4.14,'Taper':0.45,'SweepLE':25.0,'Dihedral':89,'xc_offset':0.0,'root_Incidence':0.5,'root_Thickness':0.16,'root_Airfoil_type':'datafile','root_Airfoil_ID':'e1230.dat','tip_Incidence':0.50,'tip_Thickness':0.0,'tip_Airfoil_type':'naca4','tip_Airfoil_ID':'00xx'},
            },
        }),
        2:LiftingSurface({
            'Name':'Canard',
            'Symmetry':True,
            'xrLE':1.525,'yrLE':0.0,'zrLE':0.94,
            'xRot':0.0,'yRot':0.0,'zRot':0.0,
            '_components':{
                0:{'Name':'Segment 1','Type':'external','Area':7.0,'Span':5.91,'Taper':1.0,'SweepLE':0.0,'Dihedral':-0.15,'xc_offset':0.0,'root_Incidence':1.0,'root_Thickness':0.19,'root_Airfoil_type':'datafile','root_Airfoil_ID':'gu255118.dat','tip_Incidence':1.0,'tip_Thickness':0.19,'tip_Airfoil_type':'datafile','tip_Airfoil_ID':'gu255118.dat'},
            },
        }),
    },
}
full_aircraft = pyGeo.pyGeo('acdt_geo',input,LiftingSurface,BodySurface,Airfoil)
full_aircraft.writeTecplot('acdt_aircraft.dat',orig=True)
full_aircraft.writeIGES('../input/acdt_aircraft.igs')
