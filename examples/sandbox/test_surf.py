# This is a test script to test the functionality of the 
# pySpline surface

from numpy import *
import sys,time
from mdo_import_helper import *
exec(import_modules('pySpline'))

nu = 20
nv = 20
u = linspace(0,4,nu)
v = linspace(0,4,nv)
[V,U] = meshgrid(v,u)
Z = cos(U)*sin(V)

surf = pySpline.surface(x=U,y=V,z=Z,ku=4,kv=4,Nctlu=5,Nctlv=5)
surf.writeTecplot('surface.dat')

# Test the project point Algorithim
x0 = [[0,0,0],
      [0,0,1],
      [0,1,0],
      [0,1,1],
      [1,0,0],
      [1,0,1],
      [1,1,0],
      [1,1,1]]
x0 = array(x0).astype('d')
x0 += 0.5

x0 = zeros((10000,3))
x0[:,0] =  1
timeA = time.time()
u,v,D = surf.projectPoint(x0,u=0.25*ones(len(x0)),v=zeros(len(x0)))
timeB = time.time()

for i in xrange(len(x0)):
    u,v,D = surf.projectPoint(x0[i],u=0.25,v=0.0)
# end for

timeC = time.time()
print 'Time 1 is :',timeB-timeA
print 'Time 2 is :',timeC-timeB
sys.exit(0)
val = surf(u,v)
# Output the data
f = open('projections2.dat','w')
f.write ('VARIABLES = "X", "Y","Z"\n')
for i in xrange(len(x0)):
    f.write('Zone T=surf_proj_pt%d I=2 \n'%(i))
    f.write('DATAPACKING=POINT\n')
    f.write('%f %f %f\n'%(x0[i,0],x0[i,1],x0[i,2]))
    f.write('%f %f %f\n'%(val[i,0],val[i,1],val[i,2]))
# end for

# Test the project surface-curve Algorithim

x = [0,1,2]
y = [4,3,2]
z = [-3,1,3]

curve = pySpline.curve(k=3,x=x,y=y,z=z)
curve.writeTecplot('curve3.dat',size=.2)
u,v,s,D = surf.projectCurve(curve)
val1 = surf(u,v)
val2 = curve(s)

f.write('Zone T=surf_proj_curve I=2 \n')
f.write('DATAPACKING=POINT\n')
f.write('%f %f %f\n'%(val1[0],val1[1],val1[2]))
f.write('%f %f %f\n'%(val2[0],val2[1],val2[2]))

f.close()
