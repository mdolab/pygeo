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

    def __init__(self,L,ref_axis,le_loc,chord,twist,rot_x,rot_y,af_list,N=10):

        
        '''Create an instance of the geometry object. Input is through simple
        arrays. Most error checking is left to the user making an instance of
        this class. 
        
        Input: 
        
        L, scalar : Characteristic length of surface. Span for wings, 
        blade length for wind turbines

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
        self.L        = L
        self.ref_axis = ref_axis
        self.sloc     = ref_axis[:,2]/L
        self.le_loc   = le_loc
        self.chord    = chord
        self.twist    = twist
        self.rot_x    = rot_x
        self.rot_y    = rot_y
        self.af_list  = af_list
        self.N        = N

        assert (len(ref_axis)==len(le_loc) == len(chord) \
            == len(twist) == len(af_list)),\
            "All the input data must contain the same number of records"

        naf = len(chord)
        self.naf = naf
        # N is the number of points on each surface

        # This section reads in the coordinates, finds the te and reorders
        # the points so that it starts at the upper surface of the te 
        # goes around the le and ends at the lower surface te.

        # This is the standard cosine distribution in x (one sided)
        s_interp = 0.5*(1-cos(linspace(0,pi,N)))
        #s_interp[1] = .001
        #s_interp = linspace(0,1,N)**3

        self.s = s_interp
        
        X = zeros([2,N,naf,3])

#        surf_u = zeros((naf,N,3)) #Mast list of points (upper)
#        surf_l = zeros((naf,N,3)) #Mast list of points (lower)

        n_nodes = zeros((naf,2),int) #number of nodes read in 
        
        for i in xrange(naf):
            n_temp,temp_x,temp_y = self.__load_af(af_list[i])
            
            # -------------
            # Upper Surfce
            # -------------

            # Find the trailing edge point
            index = where(temp_x == 1)
            te_index = index[0][0]
            n_nodes[i,0] = te_index+1   # number of nodes on upper surface
            n_nodes[i,1] = int(n_temp-te_index) # nodes on lower surface
    
            # upper Surface Nodes
            x_u = temp_x[0:n_nodes[i,0]]
            y_u = temp_y[0:n_nodes[i,0]]
            
            # Now determine the upper surface 's' parameter

            s = zeros(n_nodes[i,0])
            for j in xrange(n_nodes[i,0]-1):
                s[j+1] = s[j] + sqrt((x_u[j+1]-x_u[j])**2 + (y_u[j+1]-y_u[j])**2)
            # end for
            s = s/s[-1] #Normalize s

            # linearly interpolate to find the points at the positions we want
            x_interp_u = numpy.interp(s_interp,s,x_u)
            y_interp_u = numpy.interp(s_interp,s,y_u)
    
            
            # -------------
            # Lower Surface
            # -------------
            n_nodes[i,0] = te_index+1   # number of nodes on upper surface
            n_nodes[i,1] = n_temp-te_index+1 # number of nodes on lower surface
            
            x_l = temp_x[te_index:n_temp]
            y_l = temp_y[te_index:n_temp]
            x_l = hstack([x_l,0])
            y_l = hstack([y_l,0])

            # Reverse coordinates for lower spline
            #x_l = x_l[::-1]
            #y_l = y_l[::-1]

            # Now determine the lower surface 's' parameter

            s = zeros(n_nodes[i,1])
            for j in xrange(n_nodes[i,1]-1):
                s[j+1] = s[j] + sqrt((x_l[j+1]-x_l[j])**2 + (y_l[j+1]-y_l[j])**2)
            # end for
            s = s/s[-1] #Normalize s

            # linearly interpolate to find the points at the positions we want
            x_interp_l = numpy.interp(s_interp,s,x_l)
            y_interp_l = numpy.interp(s_interp,s,y_l)

            # --------------------------
            # Final Coordinate Positions
            # --------------------------

            X[0,:,i,0] = (x_interp_u-le_loc[i])*chord[i]
            X[0,:,i,1] = y_interp_u*chord[i]
            X[0,:,i,2] = 0
            
            X[1,:,i,0] = (x_interp_l-le_loc[i])*chord[i]
            X[1,:,i,1] = y_interp_l*chord[i]
            X[1,:,i,2] = 0
            
            for j in xrange(N):
                for isurf in xrange(2):
                    X[isurf,j,i,:] = self.__rotz(X[isurf,j,i,:],twist[i]*pi/180) # Twist Rotation
                    X[isurf,j,i,:] = self.__rotx(X[isurf,j,i,:],rot_x[i]*pi/180) # Dihediral Rotation
                    X[isurf,j,i,:] = self.__roty(X[isurf,j,i,:],rot_y[i]*pi/180) # Sweep Rotation
            #end for
            X[:,:,i,:] += ref_axis[i,:]
            # Finally translate according to axis:
        #end for
        
        self.X = X
        
        return



#     def __init__(self,L,ref_axis,le_loc,chord,twist,rot_x,rot_y,af_list,N=10):

        
#         '''Create an instance of the geometry object. Input is through simple
#         arrays. Most error checking is left to the user making an instance of
#         this class. 
        
#         Input: 
        
#         L, scalar : Characteristic length of surface. Span for wings, 
#         blade length for wind turbines

#         ref_axis, array, size(naf,3): List of points in space defining the 
#         the reference  axis for the geometry. Units are consistent with the
#         remainder of the geometry. For wind turbine blades, this
#         will be the pitch axis. For aircraft this will usually be the 1/4 
#         chord. The twist is defined about this axis. 

#         le_loc, array, size(naf) : Location of the LE point ahead of the 
#         reference axis. Typically this will be 1/4. United scaled by chord

#         chord, array, size(naf) : Chord lengths in consistent units
        
#         twist, array, size(naf) : Twist of each section about the reference
#         axis. Given in units of deg. 

#         af_list, list, size(naf) : List of the filenames  to be read for each
#         cross section.

#         Output:

#         x,y,z, array, size(naf,2N-1) : Coordinates of the surface for input to 
#         pySpline.

#         u, array, size(naf): parametric (spanwise) u coordinates for 
#         input to pySpline 
        
#         v, array, size(2*N-1): parametric (chordwise) v coordinates for 
#         input to pySpline'''

#         # Save the data to the class
#         self.L        = L
#         self.ref_axis = ref_axis
#         self.sloc     = ref_axis[:,2]/L
#         self.le_loc   = le_loc
#         self.chord    = chord
#         self.twist    = twist
#         self.rot_x    = rot_x
#         self.rot_y    = rot_y
#         self.af_list  = af_list
#         self.surface  = None
#         self.N        = N

#         assert (len(ref_axis)==len(le_loc) == len(chord) \
#             == len(twist) == len(af_list)),\
#             "All the input data must contain the same number of records"

#         naf = len(chord)
#         self.naf = naf
#         # N is the number of points on each surface

#         # This section reads in the coordinates, finds the te and reorders
#         # the points so that it starts at the upper surface of the te 
#         # goes around the le and ends at the lower surface te.

#         # This is the standard cosine distribution in x (one sided)
#         x_interp = 0.5*(1-cos(linspace(0,pi,N)))
#         x = zeros((naf,N*2-1,3)) #Mast list of points

#         n_nodes = zeros((naf,2),int) #number of nodes read in 
        
#         for i in xrange(naf):
#             n_temp,temp_x,temp_y = self.__load_af(af_list[i])
            
#             # Find the trailing edge point
#             index = where(temp_x == 1)
#             te_index = index[0][0]
#             n_nodes[i,0] = te_index+1   # number of nodes on upper surface
#             n_nodes[i,1] = int(n_temp-te_index) # nodes on lower surface
    
#             # upper Surface
#             x_u = temp_x[0:n_nodes[i,0]]
#             y_u = temp_y[0:n_nodes[i,0]]
            
#             # linearly interpolate to find the points at the positions we want
#             y_interp_u = numpy.interp(x_interp,x_u,y_u)
    
#             # Reverse upper coordinates to start at te
#             y_interp_u = y_interp_u[::-1]
#             x_interp_u = x_interp[::-1]
    
#             # lower Surface
    
#             n_nodes[i,0] = te_index+1   # number of nodes on upper surface
#             n_nodes[i,1] = n_temp-te_index # number of nodes on lower surface
            
#             x_l = temp_x[te_index:n_temp]
#             y_l = temp_y[te_index:n_temp]

#             # Reverse coordinates for lower spline
#             x_l = x_l[::-1]
#             y_l = y_l[::-1]

#             x_interp_l = x_interp[1:] #DO NOT want 0,0 in lower surface

#             # Interpolate
#             y_interp_l = numpy.interp(x_interp_l,x_l,y_l)


#             x_cor_full = hstack([x_interp_u,x_interp_l])-le_loc[i]
#             y_cor_full = hstack([y_interp_u,y_interp_l])

#             #Determine Final Coordinate Position
            
#             x[i,:,0] = x_cor_full*chord[i]
#             x[i,:,1] = y_cor_full*chord[i]
#             x[i,:,2] = 0
            
#             for j in xrange(2*N-1):
#                 x[i,j,:] = self.__rotz(x[i,j,:],twist[i]*pi/180) # Twist Rotation
#                 x[i,j,:] = self.__rotx(x[i,j,:],rot_x[i]*pi/180) # Dihediral Rotation
#                 x[i,j,:] = self.__roty(x[i,j,:],rot_y[i]*pi/180) # Sweep Rotation
#             #end for

#             # Finally translate according to axis:

#             x[i,:,:] += ref_axis[i,:]

#         #end for
        
#         self.x = x #The coordinates 
#         #print x

    def createSurface(self):

        '''Create the splined surface based on the input geometry'''

        print 'creating surface:'
        u = zeros([2,self.N])
        v = zeros([2,self.naf])

        u[:] = self.s
        v[:] = (self.ref_axis[:,2])/max(self.ref_axis[:,2])#*2-1

        print 'u:',u
        print 'v:',v
        
        print 'creating surfaces...'
        self.surf1 = pySpline.spline(2,u,v,self.X,fit_type='interpolate')
        self.surf2 = pySpline.spline(2,u,v,self.X,fit_type='lms',Nctlu=13,Nctlv=9)
        self.surf1.writeTecplot('output.dat')
        self.surf2.writeTecplot('output2.dat')
        sys.exit(0)
        return

    def joinSurfaces(self):
        # self.surf_l.cx[0:self.surf_u.nctlu] = self.surf_u.cx[0:self.surf_u.nctlu]
#         self.surf_l.cy[0:self.surf_u.nctlu] = self.surf_u.cy[0:self.surf_u.nctlu]
#         self.surf_l.cz[0:self.surf_u.nctlu] = self.surf_u.cz[0:self.surf_u.nctlu]
#         self.surf_l.cx[::self.surf_u.nctlv] =   self.surf_u.cx[::self.surf_u.nctlv]
#         self.surf_l.cy[::self.surf_u.nctlv] =   self.surf_u.cy[::self.surf_u.nctlv]
#         self.surf_l.cz[::self.surf_u.nctlv] =   self.surf_u.cz[::self.surf_u.nctlv]
        return

    def __load_af(self,filename):
        ''' Load the airfoil file from precomp format'''
        f = open(filename,'r')

        new_aux = string.split(f.readline())
        naf = int(new_aux[0]) 
        
        xnodes = zeros(naf)
        ynodes = zeros(naf)

        f.readline()
        f.readline()
        f.readline()

        for i in xrange(naf):

            new_aux = string.split(f.readline())

            xnodes[i] = float(new_aux[0])
            ynodes[i] = float(new_aux[1])
            
        f.close()
        return naf,xnodes, ynodes

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


    
#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'



