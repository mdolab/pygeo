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

# =============================================================================
# Extension modules
# =============================================================================

# =============================================================================
# pyGeo class
# =============================================================================
class pyGeo():
	
    '''
    Geo object class
    '''

    def __init__(self,ref_axis,le_loc,chord,twist,af_list):

        
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

        print 'ref axis:',ref_axis
        print 'le_loc:',le_loc
        print 'chord:',chord
        print 'twist:',twist
        print 'af_list:',af_list

        # Save the data to the class
        self.ref_axis = ref_axis
        self.le_loc   = le_loc
        self.chord    = chord
        self.twist    = twist
        self.af_list  = af_list

        assert (len(ref_axis)==len(le_loc) == len(chord) \
            == len(twist) == len(af_list)),\
            "All the input data must contain the same number of records"

        naf = len(chord)
        N = 10 #This should be an input

        # This section reads in the coordinates, find the te and reorders
        # the points so that it starts at the upper surface of the te 
        # goes around the le and ends at the lower surface te.

        # This is the standard cosine distribution in x (one sided)
        x_interp = 0.5*(1-cos(linspace(0,pi,N)))
        x = zeros((naf,N*2-1))
        y = zeros((naf,N*2-1))
        z = zeros((naf,N*2-1))

        n_nodes = zeros((naf,2),int) #number of nodes read in 
        
        for i in xrange(naf):
            n_temp,temp_x,temp_y = self.load_af(airfoil_list[i])
            
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


            x_cor_full = hstack([x_interp_u,x_interp_l])
            y_cor_full = hstack([y_interp_u,y_interp_l])
    
            # Finally Set the Coordinates
#             x[i,:] = x_cor_full*chord[i] - le_loc[i]*chord[i]
#             y[i,:] = y_cor_full*chord[i] 
#             z[i,:] = sloc[i]*bl_length
        









    def _load_af(filename):
        ''' Load the airfoil file in precomp format'''
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

  
#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'



