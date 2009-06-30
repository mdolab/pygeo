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
import os, sys, string, copy, pdb, time

# =============================================================================
# External Python modules
# =============================================================================
import numpy
from numpy import sin, cos, linspace, pi, zeros, where, hstack, mat, array, \
    transpose, vstack, max, dot, sqrt, append, mod

# =============================================================================
# Extension modules
# =============================================================================

import pySpline2

# =============================================================================
# pyGeo class
# =============================================================================
class pyGeo():
	
    '''
    Geo object class
    '''

    def __init__(self,init_type,*args, **kwargs):
        
        '''Create an instance of the geometry object. Input is through simple
        arrays. Most error checking is left to the user making an instance of
        this class. 
        
        Input: 
        
        init_type, string: a key word defining how this geo object
        will be defined. Valid Options/keyword argmuents are:

        'plot3d',file_name = 'file_name.xyz' : Load in a plot3D
        surface patches and use them to create splined surfaces
 

        'iges',file_name = 'file_name.iges': Load the surface patches
        from an iges file to create splined surfaes.

        
        'lifting_surface',ref_axis=ref_axis,le_loc=le_loc,scale=chord,xsec_list=af_list,
        Create a lifting surface along a reference axis ref_axis,from
        airfoils in af_list. Translate by le_loc and scale by
        chord. Required Parameters are:

            ref_axis, type ref_axis: A Reference axis object to position cross sections

            le_loc, array, size(naf) : Location of the LE point ahead of the 
            reference axis. Typically this will be 1/4. 

            scale, array, size(naf) : Chord lengths in consistent units
        
            xsec_list, list, size(naf) : List of the filenames  to be read for each
            cross section.
        
        Optional Parameters are:

            N=<integer> : the number coordinates to use from the airfoil files
            fit_type = 'lms' or 'interpolate': Least-mean squares or interpolate fitting type
            ku = spline order in the u direction (chord-wise for lifting surfaces
            kv = spline order in the v direction (span-wise for lifting surfaces
'''
        print 'pyGeo init_type is %s:'%(init_type)

        if init_type == 'plot3d':
            assert 'file_name' in kwargs,'file_name must be specified as file_name=\'filename\' for plot3d init_type'
            self._loadPlot3D(kwargs['file_name'],args,kwargs)

        elif init_type == 'iges':
            assert 'file_name' in kwargs,'file_name must be specified as file_name=\'filename\' for iges init_type'
            self._loadIges(kwargs['file_name'],args,kwargs)

        elif init_type == 'lifting_surface':
            self._init_lifting_surface(*args,**kwargs)
            
        else:
            print 'Unknown init type. Valid Init types are \'plot3d\', \'iges\' and \'lifting_surface\''
            sys.exit(0)

        return


    def _init_lifting_surface(self,*args,**kwargs):

        assert 'xsections' in kwargs and 'scale' in kwargs \
               and 'offset' in kwargs and 'ref_axis' in kwargs,\
               '\'xsections\', \'offset\',\'scale\' and \'ref_axis\' must be specified as kwargs'

        if 'fit_type' in kwargs:
            fit_type = kwargs['fit_type']
        else:
            fit_type = 'interpolate'

        xsections = kwargs['xsections']
        scale     = kwargs['scale']
        offset    = kwargs['offset']
        ref_axis  = kwargs['ref_axis']

        assert len(xsections)==len(scale)==offset.shape[0]==ref_axis.N,\
               'The length of input data is inconsistent. xsections,scale,offset.shape[0] and ref_axis.N must all have the same size'

        naf = len(xsections)
        N = 30
        X = zeros([2,N,naf,3]) #We will get two surfaces
        for i in xrange(naf):

            X_u,Y_u,X_l,Y_l = self._load_af(xsections[i],N)

            X[0,:,i,0] = (X_u-offset[i,0])*scale[i]
            X[0,:,i,1] = (Y_u-offset[i,1])*scale[i]
            X[0,:,i,2] = 0
            
            X[1,:,i,0] = (X_l-offset[i,0])*scale[i]
            X[1,:,i,1] = (Y_l-offset[i,1])*scale[i]
            X[1,:,i,2] = 0
            
            for j in xrange(N):
                for isurf in xrange(2):
                    X[isurf,j,i,:] = self.__rotz(X[isurf,j,i,:],ref_axis.rot[i,2]*pi/180) # Twist Rotation
                    X[isurf,j,i,:] = self.__rotx(X[isurf,j,i,:],ref_axis.rot[i,0]*pi/180) # Dihediral Rotation
                    X[isurf,j,i,:] = self.__roty(X[isurf,j,i,:],ref_axis.rot[i,1]*pi/180) # Sweep Rotation


            # Finally translate according to axis:
            X[:,:,i,:] += ref_axis.x[i,:]
        # end for
        self.surfs = []
        self.surfs.append(pySpline2.surf_spline(fit_type,ku=4,kv=4,X=X[0],*args,**kwargs))
        self.surfs.append(pySpline2.surf_spline(fit_type,ku=4,kv=4,X=X[1],*args,**kwargs))
        self.nPatch = 2

    def _loadPlot3D(self,file_name,*args,**kwargs):

        '''Load a plot3D file and create the splines to go with each patch'''
        
        print 'Loading plot3D file: %s ...'%(file_name)

        f = open(file_name,'r')

        # First load the number of patches
        nPatch = int(f.readline())
        
        print 'nPatch = %d'%(nPatch)

        patchSizes = zeros(nPatch*3,'intc')

        # We can do 24 sizes per line 
        nHeaderLines = 3*nPatch / 24 
        if 3*nPatch% 24 != 0: nHeaderLines += 1

        counter = 0

        for nline in xrange(nHeaderLines):
            aux = string.split(f.readline())
            for i in xrange(len(aux)):
                patchSizes[counter] = int(aux[i])
                counter += 1

        patchSizes = patchSizes.reshape([nPatch,3])

        assert patchSizes[:,2].all() == 1, \
            'Error: Plot 3d does not contain only surface patches. The third index (k) MUST be 1.'

        # Total points
        nPts = 0
        for i in xrange(nPatch):
            nPts += patchSizes[i,0]*patchSizes[i,1]*patchSizes[i,2]

        print 'Number of Surface Points = %d'%(nPts)

        nDataLines = int(nPts*3/6)
        if nPts*3%6 !=0:  nDataLines += 1

        dataTemp = zeros([nPts*3])
        counter = 0
     
        for i in xrange(nDataLines):
            aux = string.split(f.readline())
            for j in xrange(len(aux)):
                dataTemp[counter] = float(aux[j])
                counter += 1
            # end for
        # end for
        
        f.close() # Done with the file

        # Post Processing
        patches = []
        counter = 0

        for ipatch in xrange(nPatch):
            patches.append(zeros([patchSizes[ipatch,0],patchSizes[ipatch,1],3]))
            for idim in xrange(3):
                for j in xrange(patchSizes[ipatch,1]):
                    for i in xrange(patchSizes[ipatch,0]):
                        patches[ipatch][i,j,idim] = dataTemp[counter]
                        counter += 1
                    # end for
                # end for
            # end for
        # end for

        # Now create a list of spline objects:
        surfs = []
        for ipatch in xrange(nPatch):
            surfs.append(pySpline2.surf_spline(task='interpolate',X=patches[ipatch],ku=4,kv=4))
            #surfs.append(pySpline2.surf_spline(task='lms',X=patches[ipatch],ku=4,kv=4,Nctlu=9,Nctlv=9))
        
        self.surfs = surfs
        self.nPatch = nPatch
        return


    def _loadIges(self,file_name,*args,**kwargs):

        '''Load a Iges file and create the splines to go with each patch'''
        print 'file_name',file_name
        f = open(file_name,'r')
        file = []
        for line in f:
            line = line.replace(';',',')  #This is a bit of a hack...
            file.append(line)
        f.close()
        
        start_lines   = int((file[-1][1:8]))
        general_lines = int((file[-1][9:16]))
        directory_lines = int((file[-1][17:24]))
        parameter_lines = int((file[-1][25:32]))

        print start_lines,general_lines,directory_lines,parameter_lines
        
        # Now we know how many lines we have to deal 

        dir_offset  = start_lines + general_lines
        para_offset = dir_offset + directory_lines

        surf_list = []
        for i in xrange(directory_lines/2): #Directory lines is ALWAYS a multiple of 2
            if int(file[2*i + dir_offset][0:8]) == 128:
                start = int(file[2*i + dir_offset][8:16])
                num_lines = int(file[2*i + 1 + dir_offset][24:32])
                surf_list.append([start,num_lines])
            # end if
        # end for
        self.nPatch = len(surf_list)
        
        print 'Found %d surfaces in Iges File.'%(self.nPatch)

        surfs = [];
        print surf_list
        weight = []
        for ipatch in xrange(self.nPatch):  # Loop over our patches
            data = []
            # Create a list of all data
            para_offset = surf_list[ipatch][0]+dir_offset+directory_lines-1 #-1 is for conversion from 1 based (iges) to python

            for i in xrange(surf_list[ipatch][1]):
                aux = string.split(file[i+para_offset][0:69],',')
                for j in xrange(len(aux)-1):
                    data.append(float(aux[j]))
                # end for
            # end for
            
            # Now we extract what we need
            Nctlu = int(data[1]+1)
            Nctlv = int(data[2]+1)
            ku    = int(data[3]+1)
            kv    = int(data[4]+1)
            
            counter = 10
            tu = data[counter:counter+Nctlu+ku]
            counter += (Nctlu + ku)
            
            tv = data[counter:counter+Nctlv+kv]
            counter += (Nctlv + kv)
            
            weights = data[counter:counter+Nctlu*Nctlv]
            weights = array(weights)
            if weights.all() != 1:
                print 'WARNING: Not all weight in B-spline surface are 1. A NURBS surface CANNOT be replicated exactly'
            counter += Nctlu*Nctlv

            coef = zeros([Nctlu,Nctlv,3])
            for j in xrange(Nctlv):
                for i in xrange(Nctlu):
                    coef[i,j,:] = data[counter:counter +3]
                    counter+=3

            # Last we need the ranges
            range = zeros(4)
           
            range[0] = data[counter    ]
            range[1] = data[counter + 1]
            range[2] = data[counter + 2]
            range[3] = data[counter + 3]

            surfs.append(pySpline2.surf_spline(task='create',ku=ku,kv=kv,tu=tu,tv=tv,coef=coef,range=range))
        # end for
        self.surfs = surfs

        return 

    def _load_af(self,filename,N=35):
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

    def stitchPatches(self):

        '''This function attempts to automatically determine the connectivity
        between the pataches and then uses that connectivity to force
        the control points to be coincidient at corners/along edges'''

        #First we need the list of nodes and edges

        nodes = []
        edges = []
        #self.nPatch = 2
        
        for ipatch in xrange(self.nPatch):
            patch = self.surfs[ipatch]
            # Go Counter clockwise for patch i             #Nominally:
            n1 = patch.getValue(patch.range[0],patch.range[2])
            n2 = patch.getValue(patch.range[1],patch.range[2])
            n3 = patch.getValue(patch.range[1],patch.range[3])
            n4 = patch.getValue(patch.range[0],patch.range[3])
            nodes.append(n1)
            nodes.append(n2)
            nodes.append(n3)
            nodes.append(n4)

        # end for
        N = len(nodes)
        n_con = []
        counter = -1
        # Exhaustive search for connections
        tol = 1e-3
        timeA = time.time()
        for i in xrange(N):
            temp = array([],'int')
            for j in xrange(i+1,N):

                dist = self._e_dist(nodes[i],nodes[j])
                if dist< tol:
                    #pdb.set_trace()
                    #print 'counter:',counter
                    # i and j are connected
                    ifound = False
                    jfound = False
                    for l in xrange(len(n_con)):
                        if i in n_con[l] and j in n_con[l]:
                            ifound = True
                            jfound = True
                        if i in n_con[l]:
                            ifound = True
                        if j in n_con[l]:
                            jfound = True
                    # end for
                    #print 'founds:',ifound,jfound
                    if not(ifound) and not(jfound):
                        n_con.append([i,j])
                        counter += 1
                    if ifound and not(jfound):
                        n_con[counter].append(j)
                    if jfound and not(ifound):
                        n_con[counter].append(i)
                # end if
            # end for
        # end for

        print 'time:',time.time()-timeA
        for i in xrange(len(n_con)):
            print n_con[i]

        # Now we know which nodes are connected. Now we can be
        # smart....we can figure out which faces are attached to which
        # group of nodes and exhaustively test the edges on those faces

        print 'figuring edges'
        N = len(edges)
        e_con = []
        counter = -1
        
        tol = 1e-2

        # We implictly know we have #patches * 4 Edges (some may be
        # degenerate however)

        #Loop over faces
        timeA = time.time()
        for ipatch in xrange(self.nPatch):
            # Test this patch against the rest
            for jpatch in xrange(ipatch+1,self.nPatch):
                
                # Patch i, edge 0
                for i in xrange(4):
                    for j in xrange(4):
                        coinc,rev_flag=self._test_edge(self.surfs[ipatch],self.surfs[jpatch],i,j)
                        if coinc:
                            #print 'We have a coincidient edge'
                            e_con.append([[ipatch,i],[jpatch,j],rev_flag])

        #end

        
        print 'edge time:',time.time()-timeA
        for i in xrange(len(e_con)):
            print e_con[i]
        return


    def _test_edge(self,surf1,surf2,i,j):
        '''Test edge i on surf1 with edge j on surf2'''
        tol = 1e-2
        val1_beg = surf1.getValueEdge(i,0)
        val1_end = surf1.getValueEdge(i,1)

        val2_beg = surf2.getValueEdge(j,0)
        val2_end = surf2.getValueEdge(j,1)

        #Three things can happen:
        coinc = False
        rev_flag = False
        
        # Beginning and End match (same sense)
        if self._e_dist(val1_beg,val2_beg) < tol and \
               self._e_dist(val1_end,val2_end) < tol:
            # End points are the same, now check the midpoint
            mid1 = surf1.getValueEdge(i,0.5)
            mid2 = surf2.getValueEdge(j,0.5)

            if self._e_dist(mid1,mid2) < tol:
                coinc = True
            else:
                coinc = False
                #print 'end points but not middle'
        
            rev_flag = False

        # Beginning and End match (opposite sense)
        elif self._e_dist(val1_beg,val2_end) < tol and \
               self._e_dist(val1_end,val2_beg) < tol:
         
            mid1 = surf1.getValueEdge(i,0.5)
            mid2 = surf2.getValueEdge(j,0.5)

            if self._e_dist(mid1,mid2) < tol:
                coinc = True
            else:
                coinc = False
                #print 'end points but not middle',mid1,mid2,self._e_dist(mid1,mid2)

            rev_flag = True
        # If nothing else
        else:
            coinc = False

        return coinc,rev_flag
                
        
        

    def _e_dist(self,x1,x2):
        '''Get the eculidean distance between two points'''
        return sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 + (x1[2]-x2[2])**2)
    

 
    def writeTecplot(self,file_name):
        '''Write the surface patches to Tecplot'''
        f = open(file_name,'w')
        f.write ('VARIABLES = "X", "Y","Z"\n')
        for ipatch in xrange(self.nPatch):
            print 'Outputing patch %d'%(ipatch)
            self.surfs[ipatch].writeTecplot(handle=f)
            
        f.close()
        return

    def writeIGES(self,file_name):
        '''write the surface patches to IGES format'''
        f = open(file_name,'w')

        #Note: Eventually we may want to put the CORRECT Data here
        f.write('                                                                        S      1\n')
        f.write('1H,,1H;,7H128-000,11H128-000.IGS,9H{unknown},9H{unknown},16,6,15,13,15, G      1\n')
        f.write('7H128-000,1.,1,4HINCH,8,0.016,15H19970830.165254,0.0001,0.,             G      2\n')
        f.write('21Hdennette@wiz-worx.com,23HLegacy PDD AP Committee,11,3,               G      3\n')
        f.write('13H920717.080000,23HMIL-PRF-28000B0,CLASS 1;                            G      4\n')
        
        Dcount = 1;
        Pcount = 1;

        for ipatch in xrange(self.nPatch):
            Pcount,Dcount =self.surfs[ipatch].writeIGES_directory(f,Dcount,Pcount)

        Pcount = 1
        counter = 1

        for ipatch in xrange(self.nPatch):
            Pcount,counter = self.surfs[ipatch].writeIGES_parameters(f,Pcount,counter)

        # Write the terminate statment
        f.write('S%7dG%7dD%7dP%7d%40sT%7s\n'%(1,4,Dcount-1,counter-1,' ',' '))
        f.close()

        return


class ref_axis(object):

    def __init__(self,x,y,z,rot_x,rot_y,rot_z):

        ''' Create a generic reference axis. This object bascally defines a
        set of points in space (x,y,z) each with three rotations
        associated with it. The purpose of the ref_axis is to link
        groups of b-spline controls points together such that
        high-level planform-type variables can be used as design
        variables
        
        Input:

        x: list of x-coordinates of axis
        y: list of y-coordinates of axis
        z: list of z-coordinates of axis

        rot_x: list of x-axis rotations
        rot_y: list of y-axis rotations
        rot_z: list of z-axis rotations

        Note: Rotations are performed in the order: Z-Y-X
        '''

        assert len(x)==len(y)==len(z)==len(rot_x)==len(rot_y)==len(rot_z),\
            'The length of x,y,z,rot_z,rot_y,rot_x must all be the same'

        self.N = len(x)
        self.x = zeros([self.N,3])
        self.x[:,0] = x
        self.x[:,1] = y
        self.x[:,2] = z
        
        self.rot = zeros([self.N,3])
        self.rot[:,0] = rot_x
        self.rot[:,1] = rot_y
        self.rot[:,2] = rot_z

        self.x0 = copy.deepcopy(self.x)
        self.rot0 = copy.deepcopy(self.rot)
        self.sloc = zeros(self.N)
        self.updateSloc()


    def updateSloc(self):
        
        for i in xrange(self.N-1):
            self.sloc[i+1] = self.sloc[i] +sqrt( (self.x[i+1,0]-self.x[i,0])**2 + \
                                                 (self.x[i+1,1]-self.x[i,1])**2 +
                                                 (self.x[i+1,2]-self.x[i,2])**2  )
        #Normalize
        self.sloc/=self.sloc[-1]

#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'





# Old stuff possibly useful
 
#     def getRotations(self,s):
#         '''Return a (linearly) interpolated list of the twist, xrot and
#         y-rotations at a span-wise position s'''
        
#         twist = numpy.interp([s],self.sloc,self.twist)
#         rot_x = numpy.interp([s],self.sloc,self.rot_x)
#         rot_y = numpy.interp([s],self.sloc,self.rot_y)

#         return twist[0],rot_x[0],rot_y[0]

#     def getLocalVector(self,s,x):
#         '''Return the vector x, rotated by the twist, rot_x, rot_y as 
#         linearly interpolated at span-wise position s. 
#         For example getLocalVecotr(0.5,[1,0,0]) will return the vector 
#         defining the local section direction at a span position of 0.5.'''

#         twist,rot_x,rot_y = self.getRotations(s)
#         x = self.__rotz(x,twist*pi/180) # Twist Rotation
#         x = self.__rotx(x,rot_x*pi/180) # Dihedral Rotation
#         x = self.__roty(x,rot_y*pi/180) # Sweep Rotation

#         return x

#     def getLocalChord(self,s):
#         '''Return the linearly interpolated chord at span-wise postiion s'''

#         return numpy.interp([s],self.sloc,self.chord)[0]
        
#     def getLocalLe_loc(self,s):
#         return numpy.interp([s],self.sloc,self.le_loc)[0]


#     def getRefPt(self,s):
#         '''Return the linearly interpolated reference location at a span-wise
#         position s. '''
        
#         x = zeros(3);
#         x[2] = s*self.L
#         x[0] = numpy.interp([s],self.sloc,self.ref_axis[:,0])[0]
#         x[1] = numpy.interp([s],self.sloc,self.ref_axis[:,1])[0]

#         return x


#     def createAssociations(self):
#         '''Create the associated links between control pt sections and the
#         reference axis'''

#         assert self.ref_axis_reference.shape[0] == self.surf.Nctlv,\
#             'Must have the same number of control points in v (span-wise) as spanwise-stations'
#         #self.ctl_deltas = 


#     def addVar(self,dv_name,value,mapping,lower=0,upper=1):

#         '''Add a single (scalar) variable to the dv list. '''

#         if dv_name in self.DVlist.keys():
#             print 'Error: dv_name is already in the list of keys. Please use a unique deisgn variable name'
#             sys.exit(0)
#         # end if
        
#         self.DVlist[dv_name] = geoDV(dv_name,value,mapping,lower,upper)
        
        
#         return
    
#     def updateDV(self):
#         '''Update the B-spline control points from the Design Varibales'''

#         for key in self.DVlist.keys():
#             self.DVlist[key].applyValue(self.surf,self.ref_axis.sloc)

#         return





# class geoDV(object):
     
#     def __init__(self,dv_name,value,DVmapping,lower,upper):
        
#         '''Create a geometic desing variable with specified mapping

#         Input:
        
#         dv_name: Design variable name. Should be unique. Can be used
#         to set pyOpt variables directly

#         DVmapping: One or more mappings which relate to this design
#         variable

#         lower: Lower bound for the variable. Again for setting in
#         pyOpt

#         upper: Upper bound for the variable. '''

#         self.name = dv_name
#         self.value = value
#         self.lower = lower
#         self.upper = upper
#         self.DVmapping = DVmapping

#         return

#     def applyValue(self,surf,s):
#         '''Set the actual variable. Surf is the spline surface.'''
        
#         self.DVmapping.apply(surf,s,self.value)
        

#         return



# class DVmapping(object):

#     def __init__(self,sec_start,sec_end,apply_to,formula):

#         '''Create a generic mapping to apply to a set of b-spline control
#         points.

#         Input:
        
#         sec_start: j index (spanwise) where mapping function starts
#         sec_end : j index (spanwise) where mapping function
#         ends. Python-based negative indexing is allowed. eg. -1 is the last element

#         apply_to: literal reference to select what planform variable
#         the mapping applies to. Valid litteral string are:
#                 \'x\'  -> X-coordinate of the reference axis
#                 \'y\'  -> Y-coordinate of the reference axis
#                 \'z\'  -> Z-coordinate of the reference axis
#                 \'twist\' -> rotation about the z-axis
#                 \'x-rot\' -> rotation about the x-axis
#                 \'y-rot\' -> rotation about the x-axis

#         formula: is a string which contains a python expression for
#         the mapping. The value of the mapping is assigned as
#         \'val\'. Distance along the surface is specified as \'s\'. 

#         For example, for a linear shearing sweep, the formula would be \'s*val\'
#         '''
#         self.sec_start = sec_start
#         self.sec_end   = sec_end
#         self.apply_to  = apply_to
#         self.formula   = formula

#         return

#     def apply(self,surf,s,val):
#         '''apply mapping to surface'''

#         if self.apply_to == 'x':
# #             print 'ceofs'
# #             print surf.coef[0,0,self.sec_start:self.sec_end,0]
# #             print surf.coef[0,0,:,0]
# #             print 'formula'
# #             print eval(self.formula)

#             surf.coef[:,:,:,0] += eval(self.formula)

# #            surf.coef[:,:,self.sec_start:self.sec_end,0]+= eval(self.formula)
# #             print 'formula:',self.formula
# #             print 's:',s
# #             print 'val:',val
#             print eval(self.formula).shape
#             print 'done x'
