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
import pyspline
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
blade  = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis)

blade.writeTecplot('wing.dat')
blade.writeIGES('wing.igs')

# print blade.surfs[0].coef[0,:,2]
# print blade.surfs[1].coef[-1,:,2]


#Test the knots
Nctl = 20
k=4
N = 20
x = 0.5*(1-cos(linspace(0,pi,N)))
x = linspace(0,1,N)**3
print 'x:',x

t = pyspline.knots(x,Nctl,4)
t2 = zeros(Nctl+k)
t2[0:k] = 0
t2[-5:-1] = 1
t2[k-1:Nctl+1]= (0.5*(1-cos(linspace(0,pi,Nctl-k+2))))
t2[k-1:Nctl+1] = linspace(0,1,Nctl-k+2)**3

t3 = pyspline.bknot(x,k)

print 't(mine):',t
print 't(linspace method):',t2
print 't(bknot):',t3

plot(x)
plot(linspace(0,N,Nctl+k),t)
plot(t3)
show()

for i in xrange(N):
    print x[i]

print

for i in xrange(Nctl+k):
    print t[i]

print

for i in xrange(Nctl+k):
    print t2[i]
print 


for i in xrange(Nctl+k):
    print t3[i]
print 


# airfoil_list = ['af15-16.inp','af15-16.inp','pinch.inp']
# chord = [1,.75,0.5]
# tw_aero = [0,0,0]
# ref_axis = pyGeo.ref_axis([2,2.25,2.5],[0,2.5,5],[0,0,0],[90,90,90],[0,15,30],[0,0,0])
# offset = zeros((3,2))
# vstab = pyGeo.pyGeo('lifting_surface',xsections=airfoil_list,scale=chord,offset=offset,ref_axis=ref_axis)
# vstab.writeTecplot('vstab.dat')
# vstab.writeIGES('vstab.igs')
