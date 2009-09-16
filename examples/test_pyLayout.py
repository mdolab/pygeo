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

import petsc4py
petsc4py.init(sys.argv)

# =============================================================================
# Extension modules
# =============================================================================

# pySpline 
sys.path.append('../../pySpline/python')

#cfd-csm pre (Optional)
sys.path.append('../../../pyHF/pycfd-csm/python/')

#pyGeo
sys.path.append('../')
import pyGeo

#pyLayout
sys.path.append('../../pyLayout/')
import pyLayout

#Design Variable Functions
from dv_funcs import *

# ==============================================================================
# Start of Script
# ==============================================================================


# create the domain:

le_list = array([[0,1,0],[2,.8,0]])
te_list = array([[0,0,0],[1.8,.25,0]])

domain = pyLayout.domain(le_list,te_list)

