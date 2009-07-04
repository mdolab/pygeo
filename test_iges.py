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

#pyOPT
sys.path.append(os.path.abspath('../../../../pyACDT/pyACDT/Optimization/pyOpt/'))

# pySpline
sys.path.append('../pySpline/python')

#pyOPT
sys.path.append(os.path.abspath('../../pyACDT/pyACDT/Optimization/pyOpt/'))

#pySNOPT
sys.path.append(os.path.abspath('../../pyACDT/pyACDT/Optimization/pyOpt/pySNOPT'))

#pyGeo
import pyGeo

aircraft  = pyGeo.pyGeo('iges',file_name='DPW4_wb_no_tail_v03.igs')
aircraft.writeTecplot('DPW.dat')
#aircraft.stitchPatches(1e-1,1e-1)

#print 'knot vectors:'

# for i in xrange(4):
#     print 'surf %d'%(i)
#     print 'tu:',aircraft.surfs[i].tu
# #    print 'ceofs:',aircraft.surfs[i].coef

# print 
# print
# for i in xrange(4):
#     print 'surf %d'%(i)
#     print 'tv:',aircraft.surfs[i].tv

