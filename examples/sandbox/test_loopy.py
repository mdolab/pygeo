# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string, pdb, copy, time

# =============================================================================
# External Python modules
# =============================================================================
from numpy import linspace, cos, pi, hstack, zeros, ones, sqrt, imag, interp, \
    array, real, reshape, meshgrid, dot, cross

# =============================================================================
# Extension modules
# =============================================================================

# pySpline 
sys.path.append('../../pySpline/python')

#pyGeo
sys.path.append('../')
import pyGeo

# This script reads a surfaced-based plot3d file as typically
# outputted by aerosurf. It then creates a b-spline surfaces for each
# surface patch.
geo = pyGeo.pyGeo('plot3d',file_name='../input/loopy.xyz')
geo.calcEdgeConnectivity()
geo.writeTecplot('../output/loopy.dat',node_labels=True,edge_labels=True,surf_labels=True)
