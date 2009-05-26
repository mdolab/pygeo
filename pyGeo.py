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
    vstack,max

# =============================================================================
# Extension modules
# =============================================================================

sys.path.append('../pySpline/python')
import pySpline

# =============================================================================
# pyGeo class
# =============================================================================
class pyGeo():
	
    '''
    Geo object class
    '''

    def __init__(self,ref_axis,le_loc,chord,twist,rot_x,rot_y,af_list,N=10):

        
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
        self.ref_axis = ref_axis
        self.le_loc   = le_loc
        self.chord    = chord
        self.twist    = twist
        self.af_list  = af_list
        self.surface  = None
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
        x_interp = 0.5*(1-cos(linspace(0,pi,N)))
        x = zeros((naf,N*2-1,3)) #Mast list of points

        n_nodes = zeros((naf,2),int) #number of nodes read in 
        
        for i in xrange(naf):
            n_temp,temp_x,temp_y = self._load_af(af_list[i])
            
            # Find the trailing edge point
            index = where(temp_x == 1)
            te_index = index[0][0]
            n_nodes[i,0] = te_index+1   # number of nodes on upper surface
            n_nodes[i,1] = int(n_temp-te_index) # nodes on lower surface
    
            # upper Surface
            x_u = temp_x[0:n_nodes[i,0]]
            y_u = temp_y[0:n_nodes[i,0]]
            
            # linearly interpolate to find the points at the positions we want
            y_interp_u = numpy.interp(x_interp,x_u,y_u)
    
            # Reverse upper coordinates to start at te
            y_interp_u = y_interp_u[::-1]
            x_interp_u = x_interp[::-1]
    
            # lower Surface
    
            n_nodes[i,0] = te_index+1   # number of nodes on upper surface
            n_nodes[i,1] = n_temp-te_index # number of nodes on lower surface
            
            x_l = temp_x[te_index:n_temp]
            y_l = temp_y[te_index:n_temp]

            # Reverse coordinates for lower spline
            x_l = x_l[::-1]
            y_l = y_l[::-1]

            x_interp_l = x_interp[1:] #DO NOT want 0,0 in lower surface

            # Interpolate
            y_interp_l = numpy.interp(x_interp_l,x_l,y_l)


            x_cor_full = hstack([x_interp_u,x_interp_l])-le_loc[i]
            y_cor_full = hstack([y_interp_u,y_interp_l])

            #Determine Final Coordinate Position
            
            x[i,:,0] = x_cor_full*chord[i]
            x[i,:,1] = y_cor_full*chord[i]
            x[i,:,2] = 0
            
            x[i] = self._rotz(x[i],twist[i]*pi/180) # Twist Rotation
            x[i] = self._rotx(x[i],rot_x[i]*pi/180) # Dihedral Rotation
            x[i] = self._roty(x[i],rot_y[i]*pi/180) # Twist Rotation

            # Finally translate according to axis:

            x[i,:,:] += ref_axis[i,:]

        #end for
        
        self.x = x #The coordinates 
        #print x

    def createSurface(self):

        '''Create the splined surface based on the input geometry'''

        u = (self.ref_axis[:,2])/max(self.ref_axis[:,2])*2-1
        v = linspace(-1,1,2*self.N-1)
        self.surface = pySpline.spline(u,v,self.x[:,:,0],self.x[:,:,1],self.x[:,:,2])
        return

    def _load_af(self,filename):
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

    def _rotx(self,x,theta):
        ''' Rotate a set of airfoil coodinates in the local x frame'''
        M = mat([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]])

        for i in xrange(len(x)):
            x[i,:] = (M*mat(x[i,:]).T).T

        return x


    def _roty(self,x,theta):
        '''Rotate a set of airfoil coordiantes in the local y frame'''
        M = mat([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])

        for i in xrange(len(x)):
            x[i,:] = (M*mat(x[i,:]).T).T

        return x

    def _rotz(self,x,theta):
        '''Roate a set of airfoil coordinates in the local z frame'''
        M = mat([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])

        for i in xrange(len(x)):
            x[i,:] = (M*mat(x[i,:]).T).T

        return x

    def writeSurfaceTecplot(self,nu,nv,filename):

        '''Write the surface to a tecplot file'''

        u_plot = nu
        v_plot = nv

        u = linspace(-1,1,u_plot)
        v = linspace(-1,1,v_plot)
        # Start tecplot output

        nodes_total = u_plot*v_plot
        elements_total = (u_plot-1)*(v_plot-1) 

        f = open(filename,'w')
        points = zeros((u_plot,v_plot,3),float) # section, nodes, [x,y,z]

        f.write ('\"Blade Data\"\n')
        f.write ('VARIABLES = "X", "Y","Z"\n')
        f.write('Zone N=%d, E=%d\n'%(nodes_total,elements_total))
        f.write('DATAPACKING=POINT,ZONETYPE=FEQUADRILATERAL\n')
        for i in xrange(u_plot):
            for j in xrange(v_plot):
                points[i,j,:] = self.surface.getValue(u[i],v[j])
            # end for
        # end for

        # The next step will be to output all the x-y-z Data
        for i in xrange(u_plot):
            for j in xrange(v_plot):
                f.write("%.5g %.5g %.5g\n"%(points[i,j,0],points[i,j,1],points[i,j,2]))
            # end for
        # end for
                                 
        # now write out the connectivity
        for i in xrange(u_plot-1):
            for j in xrange(v_plot-1):
                f.write( '%d %d %d %d\n'%(i*v_plot + (j+1), i*v_plot+(j+2), \
                                              (i+1)*v_plot+(j+2),(i+1)*v_plot + (j+1)))
            # end for
        # end for

        # Also dump out the control points
        f.write('Zone I=%d, J=%d\n'%(self.naf,self.N*2-1))
        f.write('DATAPACKING=POINT\n')
        for j in xrange(2*self.N-1):
            for i in xrange(self.naf):
                f.write("%.5g %.5g %.5g \n"%(self.surface.bcoef_x[i,j],\
                                           self.surface.bcoef_y[i,j],\
                                           self.surface.bcoef_z[i,j]))
            # end for
        # end for 
        f.close()
        
        return

    
#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'



