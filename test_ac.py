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

naf = 10
airfoil_list = ['af1-6.inp','af-07.inp','af8-9.inp','af8-9.inp','af-10.inp',\
                    'af-11.inp','af-12.inp', 'af-14.inp',\
                    'af15-16.inp','af15-16.inp']

chord = array([.6440,1.0950,1.6800,\
         1.5390,1.2540,0.9900,0.7900,0.4550,0.4540,0.4530])

sloc = array([0.0000,0.1141,\
        0.2184,0.3226,0.4268,0.5310,0.6352,0.8437,0.9479,1.0000])*10

tw_aero = array([0.0,20.3900,16.0200,11.6500,\
                6.9600,1.9800,-1.8800,-3.4100,-3.4500,-3.4700])

ref_axis = pyGeo.ref_axis(zeros(naf),zeros(naf),sloc,zeros(naf),zeros(naf),tw_aero)

le_loc = array([0.5000,0.3300,\
          0.2500,0.2500,0.2500,0.2500,0.2500,0.2500,0.2500,0.2500])

offset = zeros([naf,2])
offset[:,0] = le_loc
blade  = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis,fit_type='lms',Nctlu=13,Nctlv=naf)

#blade.writeTecplot('wing.dat')
#blade.writeIGES('wing.igs')

blade.stitchPatches(1e-3,1e-2) #node_tol,edge_tol



# airfoil_list = ['af15-16.inp','af15-16.inp','af15-16.inp']
# chord = [1,1,1]
# tw_aero = [0,0,0]
# ref_axis = pyGeo.ref_axis([0,0,0],[0,0,0],[0,2,4],[00,00,00],[0,0,0],[0,0,0])
# offset = zeros((3,2))
# vstab = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis)
# #vstab.writeTecplot('straight_test.dat')
# vstab.writeIGES('straight_test.igs')
