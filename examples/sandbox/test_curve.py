# This is a test script to test the functionality of the 
# pySpline curve class

from numpy import *
import sys,time
from mdo_import_helper import *
exec(import_modules('pySpline'))

# Projection Tests
print 'Projection Tests'
x = [0,2,3,5]
y = [-2,5,3,0]
z = [0,2,0,-1]
curve1 = pySpline.curve(x=x,y=y,z=z,k=4)
curve1.writeTecplot('curve1.plt')

sys.exit(0)


curve1.writeTecplot('curve1.dat',size=.1)
vals = curve1(curve1.gpts)
f = open('gpts.dat','w')
f.write ('VARIABLES = "X", "Y","Z"\n')
f.write('Zone T=gpts I=%d \n'%(curve1.Nctl))
for i in xrange(curve1.Nctl):
    f.write('%f %f %f\n'%(vals[i,0],vals[i,1],vals[i,2]))
f.close()

x0 = curve1.coef
s,D = curve1.projectPoint(x0)
vals = curve1(s)
# Output the data
f = open('projections.dat','w')
f.write ('VARIABLES = "X", "Y","Z"\n')
for i in xrange(len(x0)):
    f.write('Zone T=curve%d_proj I=2 \n'%(i))
    f.write('DATAPACKING=POINT\n')
    f.write('%f %f %f\n'%(x0[i,0],x0[i,1],x0[i,2]))
    val = curve1(s[i])
    f.write('%f %f %f\n'%(vals[i,0],vals[i,1],vals[i,2]))
# end for
