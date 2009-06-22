#!/usr/local/bin/python
'''
pyGeo

pyGeo performs the routine task of reading cross sectional information 
about a wind turbine blade or aircraft wing and producing the data 
necessary to create a surface with pySpline. 

Copyright (c) 2009 by G. Kenway
All rights reserved. Not to be used for commercial purposes.
Revision: 1.0   $Date: 26/05/2009$


Developers:
-----------
- Gaetan Kenway (GKK)

History
-------
	v. 1.0 - Initial Class Creation (GKK, 2009)
'''

__version__ = '$Revision: $'


# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, string

# =============================================================================
# External Python modules
# =============================================================================
import numpy
from numpy import sin,cos,linspace,pi,zeros,where,hstack,mat,array,transpose,\
    vstack,max,dot,sqrt,append

# =============================================================================
# Extension modules
# =============================================================================

sys.path.append(os.path.abspath('../pySpline/python'))
import pySpline

# =============================================================================
# pyGeo class
# =============================================================================
class pyGeo():
	
    '''
    Geo object class
    '''

    def __init__(self,ref_axis,le_loc,chord,twist,rot_x,rot_y,af_list,N=15):

        
        '''Create an instance of the geometry object. Input is through simple
        arrays. Most error checking is left to the user making an instance of
        this class. 
        
        Input: 
        
        ref_axis, array, size(naf,3): List of points in space defining the 
        the reference  axis for the geometry. Units are consistent with the
        remainder of the geometry. For wind turbine blades, this
        will be the pitch axis. For aircraft this will usually be the 1/4 
        chord. The twist is defined about this axis. 

        le_loc, array, size(naf) : Location of the LE point ahead of the 
        reference axis. Typically this will be 1/4. United scaled by chord

        chord, array, size(naf) : Chord lengths in consistent units
        
        twist, array, size(naf) : Twist of each section about the reference
        axis. Given in units of deg. 

        af_list, list, size(naf) : List of the filenames  to be read for each
        cross section.

        Output:

        x,y,z, array, size(naf,2N-1) : Coordinates of the surface for input to 
        pySpline.

        u, array, size(naf): parametric (spanwise) u coordinates for 
        input to pySpline 
        
        v, array, size(2*N-1): parametric (chordwise) v coordinates for 
        input to pySpline'''

        # Save the data to the class
        assert (len(ref_axis)==len(le_loc) == len(chord) \
            == len(twist) == len(af_list)),\
            "All the input data must contain the same number of records"

        naf = len(chord)
        self.naf = naf
        self.ref_axis = ref_axis
        self.ref_axis_refernece = copy.deepcopy(ref_axis)
     
        self.le_loc   = le_loc
        self.chord    = chord
        self.twist    = twist
        self.rot_x    = rot_x
        self.rot_y    = rot_y
        self.af_list  = af_list
        self.N        = N
        self.DVlist   = {}
        self.sloc = zeros(naf)
        

        for i in xrange(naf-1):
            self.sloc[i+1] = self.sloc[i] +sqrt( (ref_axis[i+1,0]-ref_axis[i,0])**2 + \
                                                 (ref_axis[i+1,1]-ref_axis[i,1])**2 +
                                                 (ref_axis[i+1,2]-ref_axis[i,2])**2  )
        #Normalize
        self.sloc/=self.sloc[-1]
      
        # This is the standard cosine distribution in x (one sided)
        s_interp = 0.5*(1-cos(linspace(0,pi,N)))
        self.s = s_interp
        
        X = zeros([2,N,naf,3])
        for i in xrange(naf):

            X_u,Y_u,X_l,Y_l = self.__load_af(af_list[i],N)

            X[0,:,i,0] = (X_u-le_loc[i])*chord[i]
            X[0,:,i,1] = Y_u*chord[i]
            X[0,:,i,2] = 0
            
            X[1,:,i,0] = (X_l-le_loc[i])*chord[i]
            X[1,:,i,1] = Y_l*chord[i]
            X[1,:,i,2] = 0
            
            for j in xrange(N):
                for isurf in xrange(2):
                    X[isurf,j,i,:] = self.__rotz(X[isurf,j,i,:],twist[i]*pi/180) # Twist Rotation
                    X[isurf,j,i,:] = self.__rotx(X[isurf,j,i,:],rot_x[i]*pi/180) # Dihediral Rotation
                    X[isurf,j,i,:] = self.__roty(X[isurf,j,i,:],rot_y[i]*pi/180) # Sweep Rotation
            #end for
           
            # Finally translate according to axis:
            X[:,:,i,:] += ref_axis[i,:]
        #end for
        
        self.X = X
        return

    def createSurface(self):

        '''Create the splined surface based on the input geometry'''

        print 'creating surfaces:'
        u = zeros([2,self.N])
        v = zeros([2,self.naf])

        u[:] = self.s
        v[:] = (self.ref_axis[:,2])/max(self.ref_axis[:,2])#*2-1
               
        print 'creating surfaces...'
        self.surf = pySpline.spline(2,u,v,self.X,fit_type='interpolate',ku=4,kv=2)
        #self.surf = pySpline.spline(2,u,v,self.X,fit_type='lms',Nctlu=13,Nctlv=10,ku=4,kv=4)
       
        return

    def __load_af(self,filename,N=35):
        ''' Load the airfoil file from precomp format'''

        # Interpolation Format
        s_interp = 0.5*(1-cos(linspace(0,pi,N)))

        f = open(filename,'r')

        aux = string.split(f.readline())
        npts = int(aux[0]) 
        
        xnodes = zeros(npts)
        ynodes = zeros(npts)

        f.readline()
        f.readline()
        f.readline()

        for i in xrange(npts):
            aux = string.split(f.readline())
            xnodes[i] = float(aux[0])
            ynodes[i] = float(aux[1])
        # end for
        f.close()

        # -------------
        # Upper Surfce
        # -------------

        # Find the trailing edge point
        index = where(xnodes == 1)
        te_index = index[0]
        n_upper = te_index+1   # number of nodes on upper surface
        n_lower = int(npts-te_index)+1 # nodes on lower surface

        # upper Surface Nodes
        x_u = xnodes[0:n_upper]
        y_u = ynodes[0:n_upper]

        # Now determine the upper surface 's' parameter

        s = zeros(n_upper)
        for j in xrange(n_upper-1):
            s[j+1] = s[j] + sqrt((x_u[j+1]-x_u[j])**2 + (y_u[j+1]-y_u[j])**2)
        # end for
        s = s/s[-1] #Normalize s

        # linearly interpolate to find the points at the positions we want
        X_u = numpy.interp(s_interp,s,x_u)
        Y_u = numpy.interp(s_interp,s,y_u)

        # -------------
        # Lower Surface
        # -------------
        x_l = xnodes[te_index:npts]
        y_l = ynodes[te_index:npts]
        x_l = hstack([x_l,0])
        y_l = hstack([y_l,0])

        # Now determine the lower surface 's' parameter

        s = zeros(n_lower)
        for j in xrange(n_lower-1):
            s[j+1] = s[j] + sqrt((x_l[j+1]-x_l[j])**2 + (y_l[j+1]-y_l[j])**2)
        # end for
        s = s/s[-1] #Normalize s

        # linearly interpolate to find the points at the positions we want
        X_l = numpy.interp(s_interp,s,x_l)
        Y_l = numpy.interp(s_interp,s,y_l)

        return X_u,Y_u,X_l,Y_l

    def __rotx(self,x,theta):
        ''' Rotate a set of airfoil coodinates in the local x frame'''
        M = [[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]]
        return dot(M,x)

    def __roty(self,x,theta):
        '''Rotate a set of airfoil coordiantes in the local y frame'''
        M = [[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]
        return dot(M,x)

    def __rotz(self,x,theta):
        '''Roate a set of airfoil coordinates in the local z frame'''
        'rotatez:'
        M = [[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]]
        return dot(M,x)

  
    def getRotations(self,s):
        '''Return a (linearly) interpolated list of the twist, xrot and
        y-rotations at a span-wise position s'''
        
        twist = numpy.interp([s],self.sloc,self.twist)
        rot_x = numpy.interp([s],self.sloc,self.rot_x)
        rot_y = numpy.interp([s],self.sloc,self.rot_y)

        return twist[0],rot_x[0],rot_y[0]

    def getLocalVector(self,s,x):
        '''Return the vector x, rotated by the twist, rot_x, rot_y as 
        linearly interpolated at span-wise position s. 
        For example getLocalVecotr(0.5,[1,0,0]) will return the vector 
        defining the local section direction at a span position of 0.5.'''

        twist,rot_x,rot_y = self.getRotations(s)
        x = self.__rotz(x,twist*pi/180) # Twist Rotation
        x = self.__rotx(x,rot_x*pi/180) # Dihedral Rotation
        x = self.__roty(x,rot_y*pi/180) # Sweep Rotation

        return x

    def getLocalChord(self,s):
        '''Return the linearly interpolated chord at span-wise postiion s'''

        return numpy.interp([s],self.sloc,self.chord)[0]
        
    def getLocalLe_loc(self,s):
        return numpy.interp([s],self.sloc,self.le_loc)[0]


    def getRefPt(self,s):
        '''Return the linearly interpolated reference location at a span-wise
        position s. '''
        
        x = zeros(3);
        x[2] = s*self.L
        x[0] = numpy.interp([s],self.sloc,self.ref_axis[:,0])[0]
        x[1] = numpy.interp([s],self.sloc,self.ref_axis[:,1])[0]

        return x

    def addVar(self,dv_name,value,mapping,lower=0,upper=1):

        '''Add a single (scalar) variable to the dv list. '''

        if dv_name in self.DVlist.keys():
            print 'Error: dv_name is already in the list of keys. Please use a unique deisgn variable name'
            sys.exit(0)
        # end if
        
        self.DVlist[dv_name] = geoDV(dv_name,value,mapping,lower,upper)
        
        
        return
    
    def updateDV(self):
        '''Update the B-spline control points from the Design Varibales'''
        print 'self.sloc',self.sloc
        for key in self.DVlist.keys():
            self.DVlist[key].applyValue(self.surf,self.sloc)


class geoDV(object):
     
    def __init__(self,dv_name,value,DVmapping,lower,upper):
        
        '''Create a geometic desing variable with specified mapping

        Input:
        
        dv_name: Design variable name. Should be unique. Can be used
        to set pyOpt variables directly

        DVmapping: One or more mappings which relate to this design
        variable

        lower: Lower bound for the variable. Again for setting in
        pyOpt

        upper: Upper bound for the variable. '''


        self.name = dv_name
        self.value = value
        self.lower = lower
        self.upper = upper
        self.DVmapping = DVmapping

        return

    def applyValue(self,surf,s):
        '''Set the actual variable. Surf is the spline surface.'''
        
        self.DVmapping.apply(surf,s,self.value)
        

        return


class DVmapping(object):

    def __init__(self,sec_start,sec_end,apply_to,formula):

        '''Create a generic mapping to apply to a set of b-spline control
        points.

        Input:
        
        sec_start: j index (spanwise) where mapping function starts
        sec_end : j index (spanwise) where mapping function
        ends. Python-based negative indexing is allowed. eg. -1 is the last element

        apply_to: literal reference to select what planform variable
        the mapping applies to. Valid litteral string are:
                'x'  -> X-coordinate of the reference axis
                'y'  -> Y-coordinate of the reference axis
                'z'  -> Z-coordinate of the reference axis
                'twist' -> rotation about the z-axis
                'x-rot' -> rotation about the x-axis
                'y-rot' -> rotation about the x-axis

        formula: is a string which contains a python expression for
        the mapping. The value of the mapping is assigned as
        'val'. Distance along the surface is specified as 's'. 

        For example, for a linear shearing sweep, the formula would be 's*val'
        '''
        self.sec_start = sec_start
        self.sec_end   = sec_end
        self.apply_to  = apply_to
        self.formula   = formula

        return

    def apply(self,surf,s,val):
        '''apply mapping to surface'''

        if self.apply_to == 'x':
#             print 'ceofs'
#             print surf.coef[0,0,self.sec_start:self.sec_end,0]
#             print surf.coef[0,0,:,0]
#             print 'formula'
#             print eval(self.formula)

            surf.coef[:,:,:,0] += eval(self.formula)

#            surf.coef[:,:,self.sec_start:self.sec_end,0]+= eval(self.formula)
#             print 'formula:',self.formula
#             print 's:',s
#             print 'val:',val
            print eval(self.formula).shape
            print 'done x'

#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'



