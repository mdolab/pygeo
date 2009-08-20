#!/usr/local/bin/python
'''
pyGeo

pyGeo is a (fairly) complete geometry surfacing engine. It performs
multiple functions including producing surfaces from cross sections,
fitting groups of surfaces with continutity constraints and has
built-in design variable handling. The actual b-spline surfaces are of
the pySpline surf_spline type. See the individual functions for
additional information

Copyright (c) 2009 by G. Kenway
All rights reserved. Not to be used for commercial purposes.
Revision: 1.0   $Date: 26/05/2009$
s

Developers:
-----------
- Gaetan Kenway (GKK)
- Graeme Kennedy (GJK)

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

from numpy import sin, cos, linspace, pi, zeros, where, hstack, mat, array, \
    transpose, vstack, max, dot, sqrt, append, mod, ones, interp, meshgrid, \
    real, imag, dstack, floor

from numpy.linalg import lstsq,inv
#from scipy import io #Only used for debugging

try:
    import petsc4py
    # petsc4py.init(sys.argv)
    from petsc4py import PETSc
    
    USE_PETSC = True
    #USE_PETSC = False
    print 'PETSc4py is available. Least Square Solutions will be performed \
with PETSC'
except:
    print 'PETSc4py is not available. Least Square Solutions will be performed\
with LAPACK (Numpy Least Squares)'
    USE_PETSC = False

# =============================================================================
# Extension modules
# =============================================================================

# pySpline Utilities
import pySpline

try:
    import csm_pre
    USE_CSM_PRE = True
    print 'CSM_PRE is available. Surface associations can be performed'
except:
    print 'CSM_PRE is not available. Surface associations cannot be performed'
    USE_CSM_PRE = False

from geo_utils import *


# =============================================================================
# pyGeo class
# =============================================================================
class pyGeo():
	
    '''
    Geo object class
    '''

    def __init__(self,init_type,*args, **kwargs):
        
        '''Create an instance of the geometry object. The initialization type,
        init_type, specifies what type of initialization will be
        used. There are currently 4 initialization types: plot3d,
        iges, lifting_surface and acdt_geo

        
        Input: 
        
        init_type, string: a key word defining how this geo object
        will be defined. Valid Options/keyword argmuents are:

        'plot3d',file_name = 'file_name.xyz' : Load in a plot3D
        surface patches and use them to create splined surfaces
 

        'iges',file_name = 'file_name.igs': Load the surface patches
        from an iges file to create splined surfaes.

        
        'lifting_surface',xsections=airfoil_list,scale=chord,offset=offset 
         Xsec=X,rot=rot

         Mandatory Arguments:
              
              xsections: List of the cross section coordinate files
              scsale   : List of the scaling factor for cross sections
              offset   : List of x-y offset to apply BEFORE scaling
              Xsec     : List of spatial coordinates as to the placement of 
                         cross sections
              rot      : List of x-y-z rotations to apply to cross sections

        Optional Arguments:

              breaks   : List of ZERO-BASED index locations where to break 
                         the wing into separate surfaces
              nsections: List of length breaks+1 which specifies the number
                         of control points in that section
              section_spacing : List of lenght breaks + 1 containing lists of
                         length nections which specifiy the spanwise spacing 
                         of control points
              fit_type : strig of either 'lms' or 'interpolate'. Used to 
                         initialize the surface patches
              Nctlu    : Number of control points in the chord-wise direction
              Nfoil    : Common number of data points extracted from cross
                         section file. Points are linearly interpolated to 
                         match this value
        
        'acdt_geo',acdt_geo=object : Load in a pyGeometry object and
        use the aircraft components to create surfaces.
        '''
        
        print ' '
        print '------------------------------------------------'
        print 'pyGeo Initialization Type is: %s'%(init_type)
        print '------------------------------------------------'

        self.ref_axis = []
        self.ref_axis_con = []
        self.DV_listGlobal = []
        self.DV_listLocal  = []
        self.DV_namesGlobal = {}
        self.DV_namesLocal  = {}
        self.J = None
        self.J1 = None
        self.J2 = None
        self.con = None
        if init_type == 'plot3d':
            assert 'file_name' in kwargs,'file_name must be specified as \
file_name=\'filename\' for plot3d init_type'
            self._readPlot3D(kwargs['file_name'],args,kwargs)

        elif init_type == 'iges':
            assert 'file_name' in kwargs,'file_name must be specified as \
file_name=\'filename\' for iges init_type'
            self._readIges(kwargs['file_name'],args,kwargs)

        elif init_type == 'lifting_surface':
            self._init_lifting_surface(*args,**kwargs)

        elif init_type == 'acdt_geo':
            self._init_acdt_geo(*args,**kwargs)
        else:
            print 'Unknown init type. Valid Init types are \'plot3d\', \
\'iges\',\'lifting_surface\' and \'acdt\''
            sys.exit(0)

        return

# ----------------------------------------------------------------------------
#               Initialization Type Functions
# ----------------------------------------------------------------------------

    def _readPlot3D(self,file_name,*args,**kwargs):

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
            'Error: Plot 3d does not contain only surface patches.\
 The third index (k) MUST be 1.'

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
            surfs.append(pySpline.surf_spline(task='lms',X=patches[ipatch],\
                                                  ku=4,kv=4,Nctlu=13,Nctlv=13))

        self.surfs = surfs
        self.nPatch = nPatch
        return

    def _readIges(self,file_name,*args,**kwargs):

        '''Load a Iges file and create the splines to go with each patch'''
        print 'file_name is: %s'%(file_name)
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

        #print start_lines,general_lines,directory_lines,parameter_lines
        
        # Now we know how many lines we have to deal 

        dir_offset  = start_lines + general_lines
        para_offset = dir_offset + directory_lines

        surf_list = []
        # Directory lines is ALWAYS a multiple of 2
        for i in xrange(directory_lines/2): 
            if int(file[2*i + dir_offset][0:8]) == 128:
                start = int(file[2*i + dir_offset][8:16])
                num_lines = int(file[2*i + 1 + dir_offset][24:32])
                surf_list.append([start,num_lines])
            # end if
        # end for
        self.nPatch = len(surf_list)
        
        print 'Found %d surfaces in Iges File.'%(self.nPatch)

        self.surfs = [];
        #print surf_list
        weight = []
        for ipatch in xrange(self.nPatch):  # Loop over our patches
            data = []
            # Create a list of all data
            # -1 is for conversion from 1 based (iges) to python
            para_offset = surf_list[ipatch][0]+dir_offset+directory_lines-1 

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
                print 'WARNING: Not all weight in B-spline surface are 1.\
 A NURBS surface CANNOT be replicated exactly'
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

            self.surfs.append(pySpline.surf_spline(\
                    task='create',ku=ku,kv=kv,tu=tu,tv=tv,coef=coef,\
                        range=range))
        # end for

        return 
  
    def _init_lifting_surface(self,*args,**kwargs):

        assert 'xsections' in kwargs and 'scale' in kwargs \
               and 'offset' in kwargs and 'Xsec' in kwargs and 'rot' in kwargs,\
               '\'xsections\', \'offset\',\'scale\' and \'X\'  and \'rot\'\
 must be specified as kwargs'

        xsections = kwargs['xsections']
        scale     = kwargs['scale']
        offset    = kwargs['offset']
        Xsec      = kwargs['Xsec']
        rot       = kwargs['rot']

        if not len(xsections)==len(scale)==offset.shape[0]:
            print 'The length of input data is inconsistent. xsections,scale,\
offset.shape[0], Xsec, rot, must all have the same size'
            print 'xsections:',len(xsections)
            print 'scale:',len(scale)
            print 'offset:',offset.shape[0]
            print 'Xsec:',Xsec.shape[0]
            print 'rot:',rot.shape[0]
            sys.exit(1)


        if 'fit_type' in kwargs:
            fit_type = kwargs['fit_type']
        else:
            fit_type = 'interpolate'
        # end if

        if 'file_type' in kwargs:
            file_type = kwargs['file_type']
        else:
            file_type = 'xfoil'
        # end if

        if 'end_type' in kwargs:
            end_type = kwargs['end_type']
        else:
            end_type = 'flat'
        # end if

        if 'breaks' in kwargs:
            print 'we have breaks'
            breaks = kwargs['breaks']
            nBreaks = len(breaks)
            
            if 'nsections' in kwargs:
                nsections = kwargs['nsections']
            else: # Figure out how many sections are in each break
                nsections = zeros(nBreaks +1,'int' )
                counter = 0 
                for i in xrange(nBreaks):
                    nsections[i] = breaks[i] - counter + 1
                    counter = breaks[i]
                # end for
                nsections[-1] = len(xsections) - counter
                nsections[-1] 
            # end if

            if 'section_spacing' in kwargs:
                section_spacing = kwargs['section_spacing']
            else:
                # Generate the section spacing
                section_spacing = []
                for i in xrange(len(nsections)):
                    section_spacing.append(linspace(0,1,nsections[i]))
                # end for
            # end if

            if 'cont' in kwargs:
                cont = kwargs['cont']
            else:
                cont = []
                for i in xrange(nBreaks):
                    cont.append[0] # Default is c0 contintity
                # end for
            # end if 

        # end if
     
        naf = len(xsections)
        if 'Nfoil' in kwargs:
            N = kwargs['Nfoil']
        else:
            N = 35
        # end if
        
        # ------------------------------------------------------
        # Generate the coordinates for the sections we are given 
        # ------------------------------------------------------
        X = zeros([2,N,naf,3]) #We will get two surfaces
        for i in xrange(naf):

            X_u,Y_u,X_l,Y_l = read_af(xsections[i],file_type,N)

            X[0,:,i,0] = (X_u-offset[i,0])*scale[i]
            X[0,:,i,1] = (Y_u-offset[i,1])*scale[i]
            X[0,:,i,2] = 0
            
            X[1,:,i,0] = (X_l-offset[i,0])*scale[i]
            X[1,:,i,1] = (Y_l-offset[i,1])*scale[i]
            X[1,:,i,2] = 0
            
            for j in xrange(N):
                for isurf in xrange(2):
                    # Twist Rotation
                    X[isurf,j,i,:] = rotzV(X[isurf,j,i,:],rot[i,2]*pi/180)
                    # Dihediral Rotation
                    X[isurf,j,i,:] = rotxV(X[isurf,j,i,:],rot[i,0]*pi/180)
                    # Sweep Rotation
                    X[isurf,j,i,:] = rotyV(X[isurf,j,i,:],rot[i,1]*pi/180)


            # Finally translate according to 
            X[:,:,i,:] += Xsec[i,:]
        # end for

        # ---------------------------------------------------------------------
        # Now, we interpolate them IF we have breaks 
        # ---------------------------------------------------------------------

        self.surfs = []

        if breaks:

            tot_sec = sum(nsections)-nBreaks
            Xnew    = zeros([2,N,tot_sec,3])
            Xsecnew = zeros((tot_sec,3))
            rotnew  = zeros((tot_sec,3))
            start   = 0
            start2  = 0

            for i in xrange(nBreaks+1):
                # We have to interpolate the sectional data 
                if i == nBreaks:
                    end = naf
                else:
                    end  = breaks[i]+1
                #end if

                end2 = start2 + nsections[i]

                # We need to figure out what derivative constraints are
                # required
                
                
                # Create a chord line representation

                Xchord_line = array([X[0,0,start],X[0,-1,start]])
                chord_line = pySpline.linear_spline(task='interpolate',X=Xchord_line,k=2)

                for j in xrange(N): # This is for the Data points

                    if i == nBreaks:

                        # Interpolate across each point in the spanwise direction
                        # Take a finite difference to get dv and normalize
                        dv = (X[0,j,start] - X[0,j,start-1])
                        dv /= sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])

                        # Now project the vector between sucessive
                        # airfoil points onto this vector                        
                        V = X[0,j,end-1]-X[0,j,start]
                        dx1 = dot(dv,V) * dv

                        # For the second vector, project the point
                        # onto the chord line of the previous section

                        # D is the vector we want
                        s,D,converged,updated = \
                            chord_line.projectPoint(X[0,j,end-1])
                        dx2 = V-D
 
                        # Now generate the line and extract the points we want
                        temp_spline = pySpline.linear_spline(\
                            task='interpolate',X=X[0,j,start:end,:],k=4,\
                                dx1=dx1,dx2=dx2)
                   
                        Xnew[0,j,start2:end2,:] = \
                            temp_spline.getValueV(section_spacing[i])

                        # Interpolate across each point in the spanwise direction
                        
                        dv = (X[1,j,start]-X[1,j,start-1])
                        dv /= sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])
                        V = X[0,j,end-1]-X[0,j,start]
                        dist = dv * dot(dv,V)

                        # D is the vector we want
                        s,D,converged,updated = \
                            chord_line.projectPoint(X[1,j,end-1])
                        # We already have the 'V' vector
                        V = X[1,j,end-1]-X[1,j,start]
                        dx2 = V-D

                        temp_spline = pySpline.linear_spline(\
                            task='interpolate',X=X[1,j,start:end,:],k=4,\
                                dx1=dx1,dx2=dx2)
                        Xnew[1,j,start2:end2,:] = \
                            temp_spline.getValueV(section_spacing[i])

                    else:
                            
                        temp_spline = pySpline.linear_spline(\
                            task='interpolate',X=X[0,j,start:end,:],k=2)
                        Xnew[0,j,start2:end2,:] = \
                            temp_spline.getValueV(section_spacing[i])

                        temp_spline = pySpline.linear_spline(\
                            task='interpolate',X=X[1,j,start:end,:],k=2)
                        Xnew[1,j,start2:end2,:] = \
                            temp_spline.getValueV(section_spacing[i])
                    # end if
                # end for

                # Now we can generate and append the surfaces
                print 'generating surface'
                self.surfs.append(pySpline.surf_spline(\
                        fit_type,ku=4,kv=4,X=Xnew[0,:,start2:end2,:],\
                            Nctlv=nsections[i],*args,**kwargs))
                self.surfs.append(pySpline.surf_spline(\
                        fit_type,ku=4,kv=4,X=Xnew[1,:,start2:end2,:],\
                            Nctlv=nsections[i],*args,**kwargs))

                start = end-1
                start2 = end2-1
            # end for
            self.nPatch = len(self.surfs)


        else:  #No breaks
            
            self.surfs.append(pySpline.surf_spline(fit_type,ku=4,kv=4,X=X[0],\
                                                       *args,**kwargs))
            self.surfs.append(pySpline.surf_spline(fit_type,ku=4,kv=4,X=X[1],\
                                                       *args,**kwargs))
            self.nPatch = 2

            # Create the Reference Axis:
            self.surfs[0].associateRefAxis(cur_ref_axis)
            self.surfs[1].associateRefAxis(cur_ref_axis)
        # end if


    def _init_acdt_geo(self,*args,**kwargs):

        assert 'acdt_geo' in kwargs,\
            'key word argument \'acdt_geo\' Must be specified for \
init_acdt_geo type. The user must pass an instance of a pyGeometry aircraft'

        if 'fit_type' in kwargs:
            fit_type = kwargs['fit_type']
        else:
            fit_type = 'interpolate'

        acg = kwargs['acdt_geo']
        Components = acg._components
        ncomp = len(Components)
        counter = 0
        self.surfs = []
        # Write Aircraft Components
        for comp1 in xrange(ncomp):
            ncomp2 = len(Components[comp1])
            for comp2 in xrange(ncomp2):
                counter += 1
                [m,n] = Components[comp1]._components[comp2].Surface_x.shape
                X = zeros((m,n,3))
                X[:,:,0] = Components[comp1]._components[comp2].Surface_x
                X[:,:,1] = Components[comp1]._components[comp2].Surface_y
                X[:,:,2] = Components[comp1]._components[comp2].Surface_z
                self.surfs.append(pySpline.surf_spline(\
                        fit_type,ku=4,kv=4,X=X,*args,**kwargs))
            # end for
        # end for

        self.nPatch = len(self.surfs)
		
# ----------------------------------------------------------------------
#                      Edge Connection Information Functions
# ----------------------------------------------------------------------    

    def calcEdgeConnectivity(self,node_tol=1e-4,edge_tol=1e-4):

        '''This function attempts to automatically determine the connectivity
        between the pataches'''
        if not self.con == None:
            print 'Warning edge connectivity will be overwritten. \
 Enter 1 to continue, 0 to quit.'
            ans = raw_input()
            if ans == '0':
                return
            # end if
        # end if

        print  ' '
        print 'Attempting to Determine Edge Connectivity'

        e_con = []
        
        #Loop over faces
        timeA = time.time()
        for ipatch in xrange(self.nPatch):
            # Test this patch against the rest
            for jpatch in xrange(ipatch+1,self.nPatch):
                for i in xrange(4):
                    for j in xrange(4):
                        coinc,dir_flag=test_edge(\
                            self.surfs[ipatch],self.surfs[jpatch],i,j,edge_tol)
                        cont_flag = 0 # By Default only C0 continuity
                        if coinc:
                            #print 'We have a coincidient edge'
                            e_con.append([[ipatch,i],[jpatch,j],cont_flag,\
                                              dir_flag,-1])
                            # end if
                    # end for
                # end for
            # end for
        # end for
      

        # That calculates JUST the actual edge connectivity, i.e. The
        # Type 1's. We now have to set the remaining edges to type 0

        # Dump ALL Edges into a list...their position determines if it
        # is a connected edge or mirror edge
        edge_list = []
        for i in xrange(len(e_con)):
            edge_list.append(e_con[i][0])
            edge_list.append(e_con[i][1])

        mirrored_edges = []
        for i in xrange(self.nPatch):
            for j in xrange(4):

                if not([i,j] in edge_list):
                    mirrored_edges.append([[i,j],-1])
                    edge_list.append([i,j])
                # end if
            # end for
        # end for 

        # Now we know the connected edges and the mirrored edges.  The
        # last thing we need is the driving group
        # information...basically how does the spacing on an edge
        # propagate across the connections

        nJoined  = len(e_con)*2        # Number of joined edges
        nMirror  = len(mirrored_edges) # Number of free or mirrored edges
        # concenate the two lists --- FULL list of edge info
        edges = e_con+mirrored_edges   # Full list of edge connections
        dg_counter = -1

        for i in xrange(len(edges)):
            found_new_edge = False
            if edges[i][-1] == -1: # it hasn't been assigned a driving group yet
                dg_counter += 1
                edges[i][-1] = dg_counter
                found_new_edge = True
            # end if
            
            if found_new_edge:
                # We have assigned a new edge...now we must propagate it

                # Always set the other side of the patch of the edge we 
                # are dealing with 

                # This is confusing...need to explain better

                flip_edge = flipEdge(edges[i][0][1])
                flip_index,order = self._getConIndex(\
                    edge_list,[edges[i][0][0],flip_edge],nJoined,nMirror)
                #print 'flip_index:',flip_index
                edges[flip_index][-1] = dg_counter
                
                # Now we have to propagate along faces connected to both 
                # sides of edge

                if len(edges[i]) == 2: # We have a mirrored edge:
                    pass # Nothing to do since we are at a mirrored edge
                else:
                    
                    cont2 = True
                    index = i
                    # This means we are starting with the patch coorsponding
                    # to the second edge entry
                    order = 1 
                    while cont2: 
                        # Check the edge OPPOSITE the joined edge and keep going

                        # Joined face/edge to edge i
                        cur_face = edges[index][order][0] 
                        cur_edge = edges[index][order][1]
                        # Get the opposite edge
                        cur_edge = flipEdge(cur_edge) 
                        
                        # Find where that face/edge is
                        new_index,order = self._getConIndex(\
                            edge_list,[cur_face,cur_edge],nJoined,nMirror)

                        # if it has already been set
                        if not edges[new_index][-1] == -1: 
                            break 
                        # Order is the first or second listing on the
                        # edge. What we want to take the other face
                        # connection
                        
                        if order == 0:
                            order =1
                        else:
                            order = 0
                        
                        # Set this to current counter
                        edges[new_index][-1] = dg_counter 
                                                
                        # If this new edge is a mirrored edge stop
                        if len(edges[new_index]) == 2: 
                            cont2 = False
                        else:                          # Else we keep going
                            index = new_index
                        # end if
                    # end while
                # end if

                # Do along the First direction

                if len(edges[flip_index]) == 2: # We have a mirrored edge:
                    pass # Nothing to do since we are at a mirrored edge
                else:
                    cont2 = True
                    index = flip_index
                    #This means we proceeed with the patch coorsponding
                    # to the first edge entry
                    order = 1
                    # Check the edge OPPOSITE the joined edge and keep going
                    while cont2: 

                        # Joined face/edge to edge i
                        cur_face = edges[index][order][0] 
                        cur_edge = edges[index][order][1]
                        cur_edge = flipEdge(cur_edge)
                        new_index,order = self._getConIndex(\
                            edge_list,[cur_face,cur_edge],nJoined,nMirror)

                        # if it has already been set
                        if not edges[new_index][-1] == -1:
                            break 
                        # Set this to current counter
                        edges[new_index][-1] = dg_counter 
                        if order == 0:
                            order =1
                        else:
                            order = 0             

                        # If this edge is a mirrored edge stop
                        if len(edges[new_index]) == 2: 
                            cont2 = False
                        else:                          # Else we keep going
                            index = new_index

                    # end while
                # end if
        # end for
        # Now we can FINALLY set edge objects....creating strings
        # is a bit clunky but it works
        self.con = []
        for i in xrange(len(edges)):
            if i < nJoined/2: #Joined Edges
                init_string = '%3d        |%3d     %3d    |%3d   | %3d  |  %3d\
       | %3d  | %3d           | %3d  | %3d     %3d      |\n'\
                    %(i,edges[i][0][0],edges[i][0][1],1,7,edges[i][2],\
                          edges[i][3],edges[i][4],10,edges[i][1][0],\
                          edges[i][1][1])
            else: # Mirror Edges
                init_string = '%3d        |%3d     %3d    |%3d   | %3d  |  %3d\
       | %3d  | %3d           | %3d  | %3d     %3d      |\n'\
                    %(i,edges[i][0][0],edges[i][0][1],0,3,-1,1,edges[i][1]\
                          ,10,-1,-1)
            # end if

            temp = init_string.replace('|',' ')  #Get rid of the bars
            self.con.append(edge(temp))
        # end for
        
        # Set the edge connection info in the surfaces themselves
        self._setEdgeConnectivity()
        # Finally Print Connection Info
        self.printEdgeConnectivity()
        print 'Time for Edge Calculation:',time.time()-timeA
        return

    
    def _setEdgeConnectivity(self):
        '''Internal function to set edge_con and master_edge flags in
        surfaces'''
        if self.con == None:
            print 'Error: No edge connectivity is set yet. Either run \
 calcEdgeConnectivity or load in a .con file'
            sys.exit(1)
        # end if

        # Set the edge info
       
        for i in xrange(len(self.con)):

            self.surfs[self.con[i].f1].edge_con[self.con[i].e1] = \
                [self.con[i].f2,self.con[i].e2]
            self.surfs[self.con[i].f1].master_edge[self.con[i].e1] = True
            self.surfs[self.con[i].f1].dir[self.con[i].e1] = self.con[i].dir
            self.surfs[self.con[i].f1].edge_type[self.con[i].e1] = \
                self.con[i].type

            if self.con[i].type == 1:
                
                self.surfs[self.con[i].f2].edge_con[self.con[i].e2] = \
                    [self.con[i].f1,self.con[i].e1]
                self.surfs[self.con[i].f2].master_edge[self.con[i].e2] = False
                self.surfs[self.con[i].f2].dir[self.con[i].e2] = self.con[i].dir
                self.surfs[self.con[i].f2].edge_type[self.con[i].e2] = \
                    self.con[i].type
            # end if
        # end for

        # Set the node info
        for i in xrange(len(self.con)):
        #for i in xrange(3):

            f1 = self.con[i].f1
            e1 = self.con[i].e1

            n1,n2 = getNodesFromEdge(e1)

            #print 'face1 %d, edge %d, nodes %d and %d'%(f1,e1,n1,n2)

            if self.con[i].dir == 1:
                n1_master = n1
                n2_master = n2
            else:
                n1_master = n2
                n2_master = n1

            # if we haven't set this node yet, set them as a master
            if self.surfs[f1].master_node[n1] == None:
                self.surfs[f1].master_node[n1] = True
            # end if
            
            if self.surfs[f1].master_node[n2] == None:
                self.surfs[f1].master_node[n2] = True
            # end if

            # If there are two edges connected (type 1)
            if self.con[i].type == 1: 
                f2 = self.con[i].f2
                e2 = self.con[i].e2
                n1,n2 = getNodesFromEdge(e2)

                self.surfs[f2].master_node[n1] = False
                self.surfs[f2].master_node[n2] = False
                
                #print 'face2 %d, edge %d, nodes %d and %d'%(f2,e2,n1,n2)
                

                # Set the driven edge nodes to False ONLY if they are not 
                # already set
                # We want to set this node to f1,n1_master iff f1,n1_master
                # is a master node

                if self.surfs[f1].node_con[n1_master] == []:
                    self.surfs[f2].node_con[n1] = [f1,n1_master]
                    cont = False

                else:
                    cur_face = self.surfs[f1].node_con[n1_master][0]
                    cur_node = self.surfs[f1].node_con[n1_master][1]
                    cont = True
                    #print 'doing loop 1:'
                    while cont:

                        #print 'cur_face,cur_node:',cur_face,cur_node

                        if self.surfs[cur_face].node_con[cur_node] == []:
                            self.surfs[f2].node_con[n1] = [cur_face,cur_node]
                            cont = False
                            #print 'done loop 1'
                        else:
                            cur_face = self.surfs[cur_face].\
                                node_con[cur_node][0]
                            cur_node = self.surfs[cur_face].\
                                node_con[cur_node][1]
                        # end if
                    # end while
                # end if
                
               
                if self.surfs[f1].node_con[n2_master] == []:
                    self.surfs[f2].node_con[n2] = [f1,n2_master]
                else:
                    cont = True
                    cur_face = self.surfs[f1].node_con[n2_master][0]
                    cur_node = self.surfs[f1].node_con[n2_master][1]
                    #print 'doing loop 2:'
                    
                    while cont:

                        #print 'cur_face,cur_node:',cur_face,cur_node

                        if self.surfs[cur_face].node_con[cur_node] == []:
                            self.surfs[f2].node_con[n2] = [cur_face,cur_node]
                            cont = False
                            #print 'done loop 2'
                            #print 
                        else:
                            cur_face= self.surfs[cur_face].node_con[cur_node][0]
                            cur_node= self.surfs[cur_face].node_con[cur_node][1]
                        # end if
                    # end while
                # end if
            # end if
        # end for

        # Last thing we need to do is back propagate the slave nodes
        # to the master nodes...currently they are just []

        for ipatch in xrange(self.nPatch):
            for j in xrange(4):
                if self.surfs[ipatch].node_con[j] == []:
                    # Loop over to find matches
                    for jpatch in xrange(self.nPatch):
                        for k in xrange(4):
                            if self.surfs[jpatch].node_con[k] == [ipatch,j]:
                                self.surfs[ipatch].node_con[j].append([jpatch,k])
                            # end if
                        # end for
                    # end for
                # end if
            # end for
        # end for
        
        return

    def printEdgeConnectivity(self):

        '''Print the Edge Connectivity'''
        print ' '
        print 'Connection | Face    Edge  | Type | dof  | Continutiy | Dir?\
 | Driving Group | Nctl | Face    Edge     |'
        for i in xrange(len(self.con)):
            self.con[i].write_info(i,sys.stdout)
        # end for
        print ' '

        return

    def writeEdgeConnectivity(self,file_name):

        '''Write the current edge connectivity to a file'''

        f = open(file_name ,'w')
        f.write('Connection | Face    Edge  | Type | dof  | Continutiy | Dir?\
 | Driving Group | Nctl | Face    Edge     |\n')
        for i in xrange(len(self.con)):
            self.con[i].write_info(i,f)
        # end for
        f.close()
        return
        

    def readEdgeConnectivity(self,file_name):

        '''Load the current edge connectivity from a file'''
        if not self.con == None:
            print 'Warning edge connectivity will be overwritten. Enter 1\
 to continue, 0 to quit.'
            ans = raw_input()
            if ans == '0':
                return
            # end if
        # end if
        self.con = []
        f = open(file_name,'r')
        file = []
        for line in f:
            line = line.replace('|',' ')  #Get rid of the bars
            file.append(line)
        f.close()
        
        for i in range(1,len(file)):
            # Test for blank lines here'
            self.con.append(edge(file[i]))
        # end for

        self.printEdgeConnectivity()
        self._setEdgeConnectivity()

        return
    
    def _getConIndex(self,edges,edge,nJoined,nMirrored):

        i = edges.index(edge)
        
        if i < nJoined:
            return int(floor(i/2.0)), mod(i,2)  #integer division
        else:
            return int(floor(nJoined/2.0)) + i-nJoined,0 #integer division

    def propagateKnotVectors(self):

        '''This function propogates the knots according to the design groups'''

        # Read number of design group
        nGroups = 0;
        dg = []
        dg_counter = -1
        for i in xrange(len(self.con)):
            if self.con[i].dg > dg_counter:
                dg.append([self.con[i]])
                dg_counter += 1
            else :
                dg[self.con[i].dg].append(self.con[i])

        nGroup =len(dg) 
        
        for i in xrange(nGroup):
            
            # Check to see if ANY of the edges have reversed
            # orientation. If so, then we have to make sure we have a
            # symmetric knot vector
            symmetric = False
            for j in xrange(len(dg[i])):
                cur_edge = dg[i][j]
                if cur_edge.dir == -1:
                    symmetric = True
                    break
                # end if
            # end for

            # Take first edge entry
            first_edge = dg[i][0]
            if first_edge.e1 ==0 or first_edge.e1 == 1:  #Is it a u or a v?
                knot_vec = self.surfs[first_edge.f1].tu.copy()
            else:
                knot_vec = self.surfs[first_edge.f1].tv.copy()
            # end if

            if symmetric: # We have to symmetrize the vector:
                if mod(len(knot_vec),2) == 1: #its odd
                    mid = (len(knot_vec) -1)/2
                    beg1 = knot_vec[0:mid]
                    beg2 = (1-knot_vec[mid+1:])[::-1]
                    # Average
                    beg = 0.5*(beg1+beg2)
                    knot_vec[0:mid] = beg
                    knot_vec[mid+1:] = (1-beg)[::-1]
                else: # its even
                    mid = len(knot_vec)/2
                    beg1 = knot_vec[0:mid]
                    beg2 = (1-knot_vec[mid:])[::-1]
                    beg = 0.5*(beg1+beg2)
                    knot_vec[0:mid] = beg
                    knot_vec[mid:] = (1-beg)[::-1]
                # end if
            # end if

            # Next copy them to the rest of the face/edge combinations

            for j in xrange(len(dg[i])): # Loop over the edges with same dg's
                cur_edge = dg[i][j] # current edge class
                face1 = cur_edge.f1
                edge1 = cur_edge.e1

                if edge1 == 0 or edge1 == 1:
                    self.surfs[face1].tu = knot_vec
                else:
                    self.surfs[face1].tv = knot_vec
                # end if
                # A connected edge do th other side as well
                if cur_edge.type == 1: 
                    face2 = cur_edge.f2
                    edge2 = cur_edge.e2
                    
                    if edge2 == 0 or edge2 == 1:
                        self.surfs[face2].tu = knot_vec
                    else:
                        self.surfs[face2].tv = knot_vec
                    # end if
                # end if
            # end for
        # end for


        print 'recomputing surfaces...'
        for ipatch in xrange(self.nPatch):
            self.surfs[ipatch].recompute()




        return

    def stitchEdges(self):
        
        '''Actually join the edges'''

        for i in xrange(len(self.con)):
            if self.con[i].type == 1:
                f1 = self.con[i].f1
                e1 = self.con[i].e1
                f2 = self.con[i].f2
                e2 = self.con[i].e2

                coef = self.surfs[f1].getCoefEdge(e1).copy()
                if self.con[i].dir == -1:
                    coef = coef[::-1]
                # end if
                self.surfs[f2].setCoefEdge(e2,coef)
            # end if
        # end for
        return

# ----------------------------------------------------------------------
#                        Surface Fitting Functions
# ----------------------------------------------------------------------

    def fitSurfaces(self):
        '''This function does a lms fit on all the surfaces respecting
        the stitched edges as well as the continuity constraints'''

        # Make sure number of free points are calculated

        for ipatch in xrange(self.nPatch):
            self.surfs[ipatch]._calcNFree()
        # end for

        # Size of new jacobian and positions of block starts
        self.M = [0]
        self.N = [0]
        for ipatch in xrange(0,self.nPatch):
            self.M.append(self.M[ipatch] + self.surfs[ipatch].Nu_free*\
                              self.surfs[ipatch].Nv_free)
            self.N.append(self.N[ipatch] + self.surfs[ipatch].Nctlu_free*\
                              self.surfs[ipatch].Nctlv_free)
        # end for
        print 'M,N:',self.M,self.N

        self._initJacobian()

        #Do Loop to fill up the matrix
        col_counter = -1
        print 'Generating Matrix...'
        for ipatch in xrange(self.nPatch):
            #print 'Patch %d'%(ipatch)
            for j in xrange(self.surfs[ipatch].Nctlv):
                per_don =((j+0.0)/self.surfs[ipatch].Nctlv) # Percent Done
                #print 'done %4.2f'%(per_don)

                for i in xrange(self.surfs[ipatch].Nctlu):
                    pt_type,edge_info,node_info = \
                        self.surfs[ipatch].checkCtl(i,j) 

                    if pt_type == 0: # Its a driving node
                        col_counter += 1
                        self._setCol(self.surfs[ipatch]._calcCtlDeriv(i,j),\
                                         self.M[ipatch],col_counter)

                        # Now check for nodes/edges

                        # Its on a master edge driving another control point
                        if edge_info: 

                            # Unpack edge info
                            face  = edge_info[0][0]
                            edge  = edge_info[0][1]
                            index = edge_info[1]
                            direction = edge_info[2]
                            edge_type = edge_info[3]

                            if edge_type == 1:
                                self._setCol(self.surfs[face]._calcCtlDerivEdge\
                                                 (edge,index,direction),\
                                                 self.M[face],col_counter)

                        # Its on a corner driving (potentially)
                        # multiplie control points
                        if node_info: 
                            # Loop over the number of affected nodes
                            for k in xrange(len(node_info)): 
                                face = node_info[k][0]
                                node = node_info[k][1]
                                self._setCol(self.surfs[face].\
                                                 _calcCtlDerivNode(node),\
                                                 self.M[face],col_counter)
                            # end for
                        # end if
                    # end if
                # end for
            # end for
        # end for

        # Set the RHS
        print 'Done Matrix...'
        self._setRHS()
        # Now Solve
        self._solve()
      
        return

    def _initJacobian(self):
        
        '''Initialize the Jacobian either with PETSc or with Numpy for use
with LAPACK'''
        if USE_PETSC:
            self.J = PETSc.Mat()
            # Approximate Number of non zero entries per row:
            nz = self.surfs[0].Nctl_free
            for i in xrange(1,self.nPatch):
                if self.surfs[i].Nctl_free > nz:
                    nz = self.surfs[i].Nctl_free
                # end if
            # end for
            self.J.createAIJ([self.M[-1],self.N[-1]],nnz=nz)
        else:
            self.J = zeros([self.M[-1],self.N[-1]])
        # end if

    def _setCol(self,vec,i,j):
        '''Set a column vector, vec, at position i,j'''
        # Note: These are currently the same...
        # There is probably a more efficient way to set in PETSc
        if USE_PETSC:
            self.J[i:i+len(vec),j] = vec 
            #self.J.setValues(len(vec),i,1,j,vec)
        else:
            self.J[i:i+len(vec),j] = vec 
        # end if
        return 

        
    def _setRHS(self):
        '''Set the RHS Vector'''
        #NOTE: GET CROP DATA IS NOT WORKING! FIX ME!
        
        
        self.RHS = self.nPatch*[]
        for idim in xrange(3):
            if USE_PETSC:
                self.RHS.append(PETSc.Vec())
                self.RHS[idim].createSeq(self.M[-1])
            else:
                self.RHS.append(zeros(self.M[-1]))
            # end if 
            for ipatch in xrange(self.nPatch):
                temp = self.surfs[ipatch]._getCropData(\
                    self.surfs[ipatch].X[:,:,idim]).flatten()
                self.RHS[idim][self.M[ipatch]:self.M[ipatch] + len(temp)] = temp
            # end for
        # end for

    def _solve(self):
        '''Solve for the control points'''
        print 'in solve...'
        self.coef = zeros((self.N[-1],3))
        if USE_PETSC:
            self.J.assemblyBegin()
            self.J.assemblyEnd()

                        
            ksp = PETSc.KSP()
            ksp.create(PETSc.COMM_WORLD)
            ksp.getPC().setType('none')
            ksp.setType('lsqr')
           
            def monitor(ksp, its, rnorm):
                if mod(its,50) == 0:
                    print its,rnorm

            ksp.setMonitor(monitor)
            #ksp.setMonitor(ksp.Monitor())
            ksp.setTolerances(rtol=1e-15, atol=1e-15, divtol=100, max_it=500)

            X = PETSc.Vec()
            X.createSeq(self.N[-1])

            ksp.setOperators(self.J)
            
            for idim in xrange(3):
                print 'solving %d'%(idim)
                ksp.solve(self.RHS[idim], X) 
                for i in xrange(self.N[-1]):
                    self.coef[i,idim] = X.getValue(i)
                # end if
            # end for
        else:
            for idim in xrange(3):
                X = lstsq(self.J,self.RHS[idim])
                self.coef[:,idim] = X[0]
                print 'residual norm:',X[1]
            # end for
        # end if

        data_save = {'COEF':self.coef}
        #io.savemat('coef_lapack.mat',data_save)

        return

# ----------------------------------------------------------------------
#                Reference Axis Handling
# ----------------------------------------------------------------------

    def addRefAxis(self,surf_ids,X,rot,nrefsecs=None,spacing=None,\
                       us=None,ue=None,vs=None,ve=None):
            '''Add surf_ids surfacs to a new reference axis defined by X and
             rot with nsection values'''

            print 'adding ref axis...'
            # A couple of things can happen here: 
            # 1. nsections < len(X)
            #    -> We do a LMS fit on the ref axis
            # 2. nsection == len(X)
            #    -> We can make the ref axis as is
            # 3. nsection < len(X)
            #    -> We reinterpolate before making the ref axis

            if nrefsecs == None:
                nrefsecs = X.shape[0]

            if nrefsecs < X.shape[0]:
                # Do the lms fit
                x = pySpline.linear_spline(task='lms',X=X,\
                                                  Nctl=nrefsecs,k=2)
                s = x.s
                rotxs = pySpline.linear_spline(task='lms',s=s,X=rot[:,0],\
                                                   Nctl=nrefsecs,k=2)
                rotys = pySpline.linear_spline(task='lms',s=s,X=rot[:,1],\
                                                   Nctl=nrefsecs,k=2)
                rotzs = pySpline.linear_spline(task='lms',s=s,X=rot[:,2],\
                                                   Nctl=nrefsecs,k=2)

                if not spacing == None:
                    spacing = linspace(0,1,nrefsecs)
                    
                Xnew = x.getValue(spacing)
                rotnew = zeros((nrefsecs,3))
                rotnew[:,0] = rotxs.getValueV(spacing)
                rotnew[:,1] = rotys.getValueV(spacing)
                rotnew[:,2] = rotzs.getValueV(spacing)

                
            elif nrefsecs == X.shape[0]:
                Xnew = X
                rotnew = rot

            else: #nrefsecs > X.shape
                # Do the interpolate fit
                x = pySpline.linear_spline(task='interpolate',X=X,k=2)
                s = x.s
                rotxs = pySpline.linear_spline(\
                    task='interpolate',s=s,X=rot[:,0],Nctl=nrefsecs,k=2)
                rotys = pySpline.linear_spline(\
                    task='interpolate',s=s,X=rot[:,1],Nctl=nrefsecs,k=2)
                rotzs = pySpline.linear_spline(\
                    task='interpolate',s=s,X=rot[:,2],Nctl=nrefsecs,k=2)

                if not spacing == None:
                    spacing = linspace(0,1,nrefsecs)
                    
                Xnew = x.getValueV(spacing)
                rotnew = zeros((nrefsecs,3))
                rotnew[:,0] = rotxs.getValueV(spacing)
                rotnew[:,1] = rotys.getValueV(spacing)
                rotnew[:,2] = rotzs.getValueV(spacing)
                
            # end if

            # create the ref axis:

            ra = ref_axis(Xnew,rotnew)
          
            # Check if the start/end specifiers are NOT given at all
            
            if us == None: # No us was given
                us = []
                for i in xrange(len(surf_ids)):
                    us.append(0)
                # end for
            # end if

            if ue == None: # No ue was given
                ue = []
                for i in xrange(len(surf_ids)):
                    ue.append(self.surfs[surf_ids[i]].Nctlu_free)
                # end for 
            # end if
            
            if vs == None: # No vs was given
                vs = []
                for i in xrange(len(surf_ids)):
                    vs.append(0)
                # end for
            # end if

            if ve == None: # No ve was given
                ve =[]
                for i in xrange(len(surf_ids)):
                    ve.append(self.surfs[surf_ids[i]].Nctlv_free)
                # end for
            # end if

            # Now go back back over them and replace any Nones with
            # the appropriate value

            for i in xrange(len(surf_ids)):

                if us[i] == None:
                    us[i] = 0
                # end if

                if ue[i] == None:
                    ue[i] = self.surfs[surf_ids[i]].Nctlu_free
                # end if

                if vs[i] == None:
                    vs[i] = 0
                # end if

                if ve[i] == None:
                    ve[i] = self.surfs[surf_ids[i]].Nctlv_free
                # end if
            # end for
                
            for ii in xrange(len(surf_ids)):
                ipatch = surf_ids[ii]
                ra.surf_ids.append(ipatch)
                ra.surf_sec.append([slice(us[ii],ue[ii]),slice(vs[ii],ve[ii])])
                # Now Section out the part of the (free control
                # points) we actually want to connect
                crop_coef = self.surfs[ipatch].getFreeCtl()\
                    [slice(us[ii],ue[ii]),slice(vs[ii],ve[ii])]

                # Get the direction of the ref axis on the surface
                dir,max_s,min_s = \
                    self.surfs[ipatch].getRefAxisDir(ra,crop_coef)
                
                Nctlu = crop_coef.shape[0]
                Nctlv = crop_coef.shape[1]
                ra.surf_sizes.append([Nctlu,Nctlv])
                ra.links_x.append(zeros((Nctlu,Nctlv,3)))
                ra.links_s.append(zeros((Nctlu,Nctlv)))
                ra.surf_dir.append(dir)
                if dir == 1:
                    #print 'along v:'
                    for j in xrange(Nctlv):
                        # Create a line (k=2 spline) for the control
                        # points along U
                        ctl_line = pySpline.linear_spline(\
                            'lms',X=crop_coef[:,j],Nctl=2,k=2)
                        # Now find the minimum distance between
                        # midpoint and the ref_axis
                        s,D,conv,eps = ra.xs.projectPoint(\
                            ctl_line.getValue(0.5))
                        if s > max_s:
                            s = max_s
                        if s < min_s:
                            s = min_s
                        # Now loop over the control points to set links
                        base_point = ra.xs.getValue(s)
                        M = ra.getRotMatrixGlobalToLocal(s)

                        for i in xrange(Nctlu):
                            D = crop_coef[i,j] - base_point
                            D = dot(M,D) #Rotate to local frame
                            ra.links_s[ii][i,j] = s
                            ra.links_x[ii][i,j,:] = D
                        # end for
                     # end for
                else:
                    #print 'along u:'
                    for i in xrange(Nctlu):
                        # Create a line (k=2 spline) for the control
                        # points along U
                        ctl_line = pySpline.linear_spline(\
                            'lms',X=crop_coef[i,:],Nctl=2,k=2)
                        # Now find the minimum distance between
                        # midpoint and the ref_axis
                        s,D,conv,eps = ra.xs.projectPoint(\
                            ctl_line.getValue(0.5))
                        if s > max_s:
                            s = max_s
                        if s < min_s:
                            s = min_s
                        # Now loop over the control points to set links
                        base_point = ra.xs.getValue(s)
                        M = ra.getRotMatrixGlobalToLocal(s)

                        for j in xrange(Nctlv):
                            D = crop_coef[i,j] - base_point
                            D = dot(M,D) #Rotate to local frame
                            ra.links_s[ii][i,j] = s
                            ra.links_x[ii][i,j,:] = D
                        # end for
                     # end for
                # end if
            # end for


            self.ref_axis.append(ra)
            
    def addRefAxisCon(self,axis1,axis2,con_type):
        '''Add a reference axis connection to the connection list'''
        
        # Attach axis2 to axis1 
        # Find out the POSITION and DISTANCE on
        # axis1 that axis2 will be attached
        
        s,D,converged,update = self.ref_axis[axis1].xs.projectPoint(\
            self.ref_axis[axis2].xs.getValue(0))

        M = self.ref_axis[axis1].getRotMatrixGlobalToLocal(s)
        D = dot(M,D)

        self.ref_axis[axis2].base_point_s = s
        self.ref_axis[axis2].base_point_D = D
        self.ref_axis[axis2].con_type = con_type
        if con_type == 'full':
            assert self.ref_axis[axis2].N == 2, 'Full reference axis connection \
is only available for reference axis with 2 points. A typical usage is for \
a hinge line'
            
            s,D,converged,update = self.ref_axis[axis1].xs.projectPoint(\
                self.ref_axis[axis2].xs.getValue(1))

            M = self.ref_axis[axis1].getRotMatrixGlobalToLocal(s)
            D = dot(M,D)

            self.ref_axis[axis2].end_point_s = s
            self.ref_axis[axis2].end_point_D = D
            
        # end if
            
        self.ref_axis_con.append([axis1,axis2,con_type])

        return



# ----------------------------------------------------------------------
#                Update and Derivative Functions
# ----------------------------------------------------------------------

    def finalize(self):

        '''The finalize command must be run before the geometry can be fully
        surface fitted or used with design variables. No further
        changes in the edge connectivity is allowed. Two commands are
        run for each surface: calcNfree and getFreeIndex. These
        commands calculate the number and size of free control points
        and calculates the slice string which is used to extract/set
        those control points'''

        for ipatch in xrange(self.nPatch):
            self.surfs[ipatch]._getFreeIndex()
            self.surfs[ipatch]._calcNFree()
        # end for

        # Also calculate the global index for each control point
        current_index = 0
        for ipatch in xrange(self.nPatch):
            current_index = self.surfs[ipatch]._assignGlobalIndex(current_index)
        # end for
            
        # Now back propogate the masters to the slaves:

        for ipatch in xrange(self.nPatch):
            Nctlu = self.surfs[ipatch].Nctlu
            Nctlv = self.surfs[ipatch].Nctlv
            
            for i in xrange(Nctlu):
                for j in xrange(Nctlv):
                    pt_type,edge_info,node_info=self.surfs[ipatch].checkCtl(i,j)
                    
                    if pt_type == 0:
                        global_index = self.surfs[ipatch].globalCtlIndex[i,j]

                        if edge_info:
                            # Unpack edge info
                            face  = edge_info[0][0]
                            edge  = edge_info[0][1]
                            index = edge_info[1]
                            direction = edge_info[2]
                            edge_type = edge_info[3]
                    
                            if edge_type == 1: # We have another attached ctl
                                self.surfs[face].setCoefIndexEdge(\
                                    edge,global_index,index,direction)
                            # end if
                        # end if

                        if node_info: 
                            # Loop over the number of affected nodes
                            for k in xrange(len(node_info)): 
                                face = node_info[k][0]
                                node = node_info[k][1]
                                self.surfs[face].setCoefIndexCorner(\
                                    node,global_index)
                            # end for
                        # end if
                    # end if
                # end for
            # end for
        # end for

        return

    def update(self):
        '''update the entire pyGeo Object'''

        # First, update the reference axis info from the design variables
        #timeA = time.time()
        for i in xrange(len(self.DV_listGlobal)):
            # Call the each design variable with the ref axis list
            self.ref_axis = self.DV_listGlobal[i](self.ref_axis)
        # end for

        # Second, update the end_point base_point on the ref_axis:
        #timeB = time.time()
        for i in xrange(len(self.ref_axis_con)):
            axis1 = self.ref_axis_con[i][0]
            axis2 = self.ref_axis_con[i][1]

            self.ref_axis[axis1].update()
            s = self.ref_axis[axis2].base_point_s
            D = self.ref_axis[axis2].base_point_D
            M = self.ref_axis[axis1].getRotMatrixLocalToGloabl(s)
            D = dot(M,D)

            X0 = self.ref_axis[axis1].xs.getValue(s)

            self.ref_axis[axis2].base_point = X0 + \
                D*self.ref_axis[axis1].scales(s)

            if self.ref_axis[axis2].con_type == 'full':
                s = self.ref_axis[axis2].end_point_s
                D = self.ref_axis[axis2].end_point_D
                M = self.ref_axis[axis1].getRotMatrixLocalToGloabl(s)
                D = dot(M,D)
                
                X0 = self.ref_axis[axis1].xs.getValue(s)

                self.ref_axis[axis2].end_point = X0 +\
                    D*self.ref_axis[axis1].scales(s)
            # end if

            self.ref_axis[axis2].update()
        # end for
        #timeC = time.time()

        # Third, update the design variables
        for r in xrange(len(self.ref_axis)):
            for ii in xrange(len(self.ref_axis[r].surf_ids)):
                ipatch = self.ref_axis[r].surf_ids[ii]
                Nctlu = self.ref_axis[r].surf_sizes[ii][0]
                Nctlv = self.ref_axis[r].surf_sizes[ii][1]
                dir = self.ref_axis[r].surf_dir[ii]
                s_pos = self.ref_axis[r].links_s[ii]
                links = self.ref_axis[r].links_x[ii]
                # Data from the ref_axis:
                s = self.ref_axis[r].s    # parameter for ref axis
                t = self.ref_axis[r].xs.t # common knot vector for ref axis
                x = self.ref_axis[r].xs.coef
                rot   = zeros((self.ref_axis[r].N,3),'d')
                rot[:,0] = self.ref_axis[r].rotxs.coef
                rot[:,1] = self.ref_axis[r].rotys.coef
                rot[:,2] = self.ref_axis[r].rotzs.coef
                scales   = self.ref_axis[r].scales.coef
                #print 'caling getcoef'
                coef = pySpline.pyspline.getcoef(\
                    dir,s,t,x,rot,scales,s_pos,links)
                
                # Update the section of free control points
                self.surfs[ipatch].setFreeCtlSection(\
                    coef,self.ref_axis[r].surf_sec[ii][0],\
                        self.ref_axis[r].surf_sec[ii][1])
            # end for
        # end for

        #timeD = time.time()

        # fourth update the Local coordinates

        for i in xrange(len(self.DV_listLocal)):
            self.surfs[self.DV_listLocal[i].surface_id] = \
                self.DV_listLocal[i](self.surfs[self.DV_listLocal[i].surface_id])
        # end for
        #timeE = time.time()
        # Fifth, run the stitch surfaces command to enforce master dv's
        self.stitchEdges()
        #timeF = time.time()

#         print 'time1:',timeB-timeA
#         print 'time2:',timeC-timeB
#         print 'time3:',timeD-timeC
#         print 'time4:',timeE-timeD
#         print 'time5:',timeF-timeE
        return
         
    def returncoef(self):
        '''Temp function to get the list of (compressed) coefficientzs for testing with fd'''
        coef = array([])
        for ipatch in xrange(self.nPatch):
            temp = self.surfs[ipatch].getFreeCtl().flatten()
            coef = hstack([coef,temp])
        # end for
            
        return coef
         
    def returncoef2(self):
        '''Temp function to get the list of (compressed) coefficientzs for testing with fd'''
        coef = []
        for ipatch in xrange(self.nPatch):
            coef.append(self.surfs[ipatch].getFreeCtl().copy())
        # end for
            
        return coef

    def getSurfacePoints(self,patchID,uv):

        '''Temp function to get ALL surface Points'''

        N = len(patchID)
        coordinates = zeros((N,3))
        for i in xrange(N):
            coordinates[i] = self.surfs[patchID[i]].getValue(uv[i][0],uv[i][1])

        return coordinates.flatten()


    def calcCtlDeriv(self):

        '''This function runs the complex step method over the design variable
        and generates a (sparse) jacobian of the control pt
        derivatives wrt to the design variables'''

        # Initialize the jacobian

        if not self.J1: # Not initialized
            # Calculate the size Ncoef_free x Ndesign Variables
            
            M = [0]
            for i in xrange(self.nPatch):
                self.surfs[i]._calcNFree()
                M.append(M[-1]+self.surfs[i].Nctl_free)
            # end if

            Nctl = M[-1]
            print 'M:',M
            # Calculate the Number of Design Variables:
            

            N = [0]
            for i in xrange(len(self.DV_listGlobal)): #Global Variables
                N.append(N[-1]+self.DV_listGlobal[i].nVal)
            # end for
            
            NdvGlobal = N[-1]
            Ndv = N[-1]
            for i in xrange(len(self.DV_listLocal)): # Local Variables
                Ndv += self.DV_listLocal[i].nVal
            # end for
                
            NdvLocal = Ndv-NdvGlobal

            print 'NdvGlobal:',NdvGlobal
            print 'Nctl:',Nctl
            print 'Ndv:',Ndv
            # We know the row filling factor: Its (exactly) nGlobal + 3
             
            if USE_PETSC:
                self.J1 = PETSc.Mat()
                self.J1.createAIJ([Nctl*3,Ndv],nnz=NdvGlobal+3)
            else:
                self.J1 = zeros((Nctl*3,Ndv))
            # end if
        # end if 
   
        # This next section of code is basically the update() function
        h = 1.0e-40j
        col_counter = 0
        for idv in xrange(len(self.DV_listGlobal)): # This is the Master CS Loop
            nVal = self.DV_listGlobal[idv].nVal

            for jj in xrange(nVal):
                if nVal == 1:
                    self.DV_listGlobal[idv].value += h
                else:
                    self.DV_listGlobal[idv].value[jj] += h
                # end if

                coef = []
                # Create a list of the free coefficients for EACH surface
                for ipatch in xrange(self.nPatch):
                    coef.append(zeros((self.surfs[ipatch].Nctlu_free,\
                                           self.surfs[ipatch].Nctlv_free,3),'D'))

                # -----------COPY OF UPDATE--------------
                 # First, update the reference axis info from the design variables

                for i in xrange(len(self.DV_listGlobal)):
                    # Call the each design variable with the ref axis list
                    self.ref_axis = self.DV_listGlobal[i](self.ref_axis)
                # end for

                # Second, update the end_point base_point on the ref_axis:

                for i in xrange(len(self.ref_axis_con)):
                    axis1 = self.ref_axis_con[i][0]
                    axis2 = self.ref_axis_con[i][1]
                    self.ref_axis[axis1].update()
                    s = self.ref_axis[axis2].base_point_s
                    D = self.ref_axis[axis2].base_point_D
                    R = self.ref_axis[axis1].getRotMatrixLocalToGloabl(s)
                    D = dot(R,D)

                    X0 = self.ref_axis[axis1].xs.getValue(s)
                    self.ref_axis[axis2].base_point = X0 + \
                        D*self.ref_axis[axis1].scales(s)

                    if self.ref_axis[axis2].con_type  == 'full':
                        s = self.ref_axis[axis2].end_point_s
                        D = self.ref_axis[axis2].end_point_D
                        R = self.ref_axis[axis1].getRotMatrixLocalToGloabl(s)
                        D = dot(R,D)

                        X0 = self.ref_axis[axis1].xs.getValue(s)
                        self.ref_axis[axis2].end_point = X0 +\
                            D*self.ref_axis[axis1].scales(s)
                    # end if
                        
                    self.ref_axis[axis2].update()
                # end for
                timeC = time.time()

                # -------END COPY OF UPDATE--------------

                # Third, update the design variables
                for r in xrange(len(self.ref_axis)):
                    for ii in xrange(len(self.ref_axis[r].surf_ids)):
                        
                        ipatch = self.ref_axis[r].surf_ids[ii]
                        Nctlu = self.ref_axis[r].surf_sizes[ii][0]
                        Nctlv = self.ref_axis[r].surf_sizes[ii][1]
                        dir = self.ref_axis[r].surf_dir[ii]
                        s_pos = self.ref_axis[r].links_s[ii]
                        links = self.ref_axis[r].links_x[ii]
                        # Data from the ref_axis:
                        s = self.ref_axis[r].s    # parameter for ref axis
                        t = self.ref_axis[r].xs.t # common knot vector 
                        x = self.ref_axis[r].xs.coef
                        rot   = dstack((self.ref_axis[r].rotxs.coef,\
                                            self.ref_axis[r].rotys.coef, \
                                            self.ref_axis[r].rotzs.coef))
                        scales   = self.ref_axis[r].scales.coef

                        # Just the coefficient affected by this ref_axis
                     
                        # Fortran Call Template
                        #coef = getcoef(dir,s,t,x,rot,scale,s_pos,links,\
                            #[nref,nx,ny])

                        coef_temp = pySpline.pyspline_cs.getcoef(\
                            dir,s,t,x,rot,scales,s_pos,links)
                        
                        # TOTAL size of the coefficients on the patch
                        su = self.ref_axis[r].surf_sec[ii][0]
                        sv = self.ref_axis[r].surf_sec[ii][1]
                        coef[ipatch][su,sv] = coef_temp

                    # end for
                # end for
                # Fourth 
                for idv2 in xrange(len(self.DV_listLocal)):

                    ipatch = self.DV_listLocal[idv2].surface_id

                    slice_u = self.DV_listLocal[idv2].slice_u
                    slice_v = self.DV_listLocal[idv2].slice_v
                    value = self.DV_listLocal[idv2].value

                    coef[ipatch] = self.surfs[ipatch].\
                        updateSurfacePointsDeriv(coef[ipatch],value,\
                                                     slice_u,slice_v)
                # end for

                # Now set the column in J1
                for ipatch in xrange(self.nPatch):
                    self.J1[M[ipatch]*3:M[ipatch+1]*3,col_counter] =       \
                        imag(coef[ipatch].flatten())/1e-40
                # Increment Column Counter
                col_counter += 1
                #print 'col_counter:',col_counter

                # Reset Design Variable Peturbation
                if nVal == 1:
                    self.DV_listGlobal[idv].value -= h
                else:
                    self.DV_listGlobal[idv].value[jj] -= h
                # end if
            # end for
        # end for
        
        # The next step is go to over all the LOCAL variables,
        # compute the surface normal and 

        for idv in xrange(len(self.DV_listLocal)): # This is the Master CS Loop
            
            nVal = self.DV_listLocal[idv].nVal
            ipatch = self.DV_listLocal[idv].surface_id
            
            ipatch = self.DV_listLocal[idv].surface_id
            normals = pySpline.pyspline.getctlnormals(\
                self.surfs[ipatch].coef,self.surfs[ipatch].tu,\
                    self.surfs[ipatch].tv,self.surfs[ipatch].ku, \
                    self.surfs[ipatch].kv)

            # NOTE: Normals are the FULL size of the surface

            us = self.DV_listLocal[idv].slice_u.start
            vs = self.DV_listLocal[idv].slice_v.start

            Nctlu_free = self.surfs[ipatch].Nctlu_free
            Nctlv_free = self.surfs[ipatch].Nctlv_free

            for i in xrange(self.DV_listLocal[idv].Nctlu):
                for j in xrange(self.DV_listLocal[idv].Nctlv):
                    # Calculate the actual position
                    index = M[ipatch]*3 + ((i+us)*Nctlv_free + j+vs)*3
                    self.J1[index:index+3,col_counter] = -normals[i+us,j+vs]
                    #print 'col_counter',col_counter,index,index+3
                    col_counter += 1
                # end for
            # end for

            # Now set the normal in the correct location

        if USE_PETSC:
            self.J1.assemblyBegin()
            self.J1.assemblyEnd()
        # end if 
       
        # Print the first column to a file

        # Now Do the Try the matrix multiplication
        if USE_PETSC:
            self.C = PETSc.Mat()
            self.J2.matMult(self.J1,result=self.C)
        else:
            self.C = dot(self.J2,self.J1)
        # end if

        return coef

    def addGeoObject(self,geo_obj):

        '''Concentate two pyGeo objects into one'''

        for  i in xrange(geo_obj.nPatch):
            self.surfs.append(geo_obj.surfs[i])

        # end for
        self.nPatch += geo_obj.nPatch
        self.con = None
        self.ref_axis = []
        self.DV_listGlobal = []
        self.DV_listLocal = []
        self.DV_namesGlobal = {}
        self.DV_namesLocal = {}

        print 'Warning: edge connections,reference_axis and design variables \
 have been reset'
        return 


    def addGeoDVLocal(self,dv_name,lower,upper,surf=None,us=None,ue=None,\
                          vs=None,ve=None):

        '''Add a local design variable group.  For Local Design variables the
        reference us,ue,vs,ve are given wrt the full surface since the
        user will generally know the full size of the surface but not
        where it has been clipped due to master/slave edges'''

        if surf == None:
            print 'Error: A surface must be specified with surf = <surf_id>'
            sys.exit(1)

        # First get the full and free size of the surface we are dealing with
        Nctlu = self.surfs[surf].Nctlu
        Nctlv = self.surfs[surf].Nctlv
        Nctlu_free = self.surfs[surf].Nctlu_free
        Nctlv_free = self.surfs[surf].Nctlv_free

        if us == None:
            us = 0

        if ue == None or ue > Nctlu_free:
            ue = Nctlu_free

        if vs == None:
            vs = 0
            
        if ve == None or ve > Nctlv_free:
            ve = Nctlv_free

        slice_u = slice(us,ue)
        slice_v = slice(vs,ve)
        
        self.DV_listLocal.append(geoDVLocal(dv_name,lower,upper,\
                                                surf,slice_u,slice_v))
        
        self.DV_namesLocal[dv_name] = len(self.DV_listLocal)-1
        
        return


    def addGeoDVGlobal(self,dv_name,value,lower,upper,function):
        '''Add a global design variable'''
        self.DV_listGlobal.append(geoDVGlobal(\
                dv_name,value,lower,upper,function))
        self.DV_namesGlobal[dv_name]=len(self.DV_listGlobal)-1
        return 

# ----------------------------------------------------------------------
#                   Surface Writing Output Functions
# ----------------------------------------------------------------------

    def writeTecplot(self,file_name,surfs=True,edges=False,\
                         ref_axis=True,links=False):
        '''Write the pyGeo Object to Tecplot'''

        # ---------------------------
        #    Write out the surfaces
        # ---------------------------
        
        if surfs == True:
            f = open(file_name,'w')
            f.write ('VARIABLES = "X", "Y","Z"\n')
            print ' '
            print 'Writing Tecplot file: %s '%(file_name)
            sys.stdout.write('Outputting Patch: ')
            for ipatch in xrange(self.nPatch):
                sys.stdout.write('%d '%(ipatch))
                self.surfs[ipatch].writeTecplotSurface(handle=f,size=0.03)

        # ---------------------------
        #    Write out the edges
        # ---------------------------

        # We also want to output edge continuity for visualization
        if self.con and edges==True:
            counter = 1
            for i in xrange(len(self.con)): #Output Simple Edges (no continuity)
                if self.con[i].cont == 0 and self.con[i].type == 1:
                    surf = self.con[i].f1
                    edge = self.con[i].e1
                    zone_name = 'simple_edge%d'%(counter)
                    counter += 1
                    self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
                # end if
            # end for

            for i in xrange(len(self.con)): #Output Continuity edges
                if self.con[i].cont == 1 and self.con[i].type == 1:
                    surf = self.con[i].f1
                    edge = self.con[i].e1
                    zone_name = 'continuity_edge%d'%(counter)
                    counter += 1
                    self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
                # end if
            # end for

            for i in xrange(len(self.con)): #Output Mirror (free) edges
                if self.con[i].type == 0: #output the edge
                    surf = self.con[i].f1
                    edge = self.con[i].e1
                    zone_name = 'mirror_edge%d'%(counter)
                    counter += 1
                    self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
                # end if
            # end for
        # end if

        # ---------------------------------
        #    Write out Ref Axis
        # ---------------------------------

        if len(self.ref_axis)>0 and ref_axis==True:
            for r in xrange(len(self.ref_axis)):
                axis_name = 'ref_axis%d'%(r)
                self.ref_axis[r].writeTecplotAxis(f,axis_name)
            # end for
        # end if

        # ---------------------------------
        #    Write out Links
        # ---------------------------------

        if len(self.ref_axis)>0 and links==True:
            for r in xrange(len(self.ref_axis)):
                self.ref_axis[r].writeTecplotLinks(f,self.surfs)
            # end for
        # end if
              
        f.close()
        sys.stdout.write('\n')
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
            Pcount,Dcount =self.surfs[ipatch].writeIGES_directory(\
                f,Dcount,Pcount)

        Pcount  = 1
        counter = 1

        for ipatch in xrange(self.nPatch):
            Pcount,counter = self.surfs[ipatch].writeIGES_parameters(\
                f,Pcount,counter)

        # Write the terminate statment
        f.write('S%7dG%7dD%7dP%7d%40sT%7s\n'%(1,4,Dcount-1,counter-1,' ',' '))
        f.close()

        return

    # ----------------------------------------------------------------------
    #                              Utility Functions 
    # ----------------------------------------------------------------------

    def coordinatesFromFile(self,file_name):
        '''Get a list of coordinates from a file - useful for testing'''

        f = open(file_name,'r')
        coordinates = []
        for line in f:
            aux = string.split(line)
            coordinates.append([float(aux[0]),float(aux[1]),float(aux[2])])
        # end for
        f.close()
        coordinates = transpose(array(coordinates))

        return coordinates
    
    def attachSurface(self,coordinates,patch_list=None,Nu=20,Nv=20):

        '''Attach a list of surface points to either all the pyGeo surfaces
        of a subset of the list of surfaces provided by patch_list.

        Arguments:
             coordinates   :  a 3 by nSurf numpy array
             patch_list    :  list of patches to locate next to nodes,
                              None means all patches will be used
             Nu,Nv         :  parameters that control the temporary
                              discretization of each surface        
             
        Returns:
             dist          :  distance between mesh location and point
             patchID       :  patch on which each u,v coordinate is defined
             uv            :  u,v coordinates in a 2 by nSurf array.

        Modified by GJK to include a search on a subset of surfaces.
        This is useful for associating points in a mesh where points may
        lie on the edges between surfaces. Often, these points must be used
        twice on two different surfaces for load/displacement transfer.        
        '''
        print ''
        print 'Attaching a discrete surface to the Geometry Object...'

        if patch_list == None:
            patch_list = range(self.nPatch)
        # end
    
        nSurf = coordinates.shape[1]
        # Now make the 'FE' Grid from the sufaces.

        # Global 'N' Parameter
        patches = len(patch_list)
        
        nelem    = patches * (Nu-1)*(Nv-1)
        nnode    = patches * Nu *Nv
        conn     = zeros((4,nelem),int)
        xyz      = zeros((3,nnode))
        elemtype = 4*ones(nelem) # All Quads
        
        counter = 0
        for n in xrange(patches):
            ipatch = patch_list[n]
            
            u = linspace(self.surfs[ipatch].range[0],\
                             self.surfs[ipatch].range[1],Nu)
            v = linspace(self.surfs[ipatch].range[0],\
                             self.surfs[ipatch].range[1],Nv)
            [U,V] = meshgrid(u,v)

            temp = self.surfs[ipatch].getValueM(U,V)
            for idim in xrange(self.surfs[ipatch].nDim):
                xyz[idim,n*Nu*Nv:(n+1)*Nu*Nv]= \
                    temp[:,:,idim].flatten()
            # end for

            # Now do connectivity info
           
            for j in xrange(Nv-1):
                for i in xrange(Nu-1):
                    conn[0,counter] = Nu*Nv*n + (j  )*Nu + i     + 1
                    conn[1,counter] = Nu*Nv*n + (j  )*Nu + i + 1 + 1 
                    conn[2,counter] = Nu*Nv*n + (j+1)*Nu + i + 1 + 1
                    conn[3,counter] = Nu*Nv*n + (j+1)*Nu + i     + 1
                    counter += 1

                # end for
            # end for
        # end for

        # Now run the csm_pre command 
        print 'Running CSM_PRE...'
        [dist,nearest_elem,uvw,base_coord,weightt,weightr] = \
            csm_pre.csm_pre(coordinates,xyz,conn,elemtype)
        print 'Done CSM_PRE...'
        # All we need from this is the nearest_elem array and the uvw array

        # First we back out what patch nearest_elem belongs to:
        patchID = (nearest_elem-1) / ((Nu-1)*(Nv-1))  # Integer Division

        # Now go back through and adjust the patchID to the element list
        for i in xrange(nSurf):
            patchID[i] = patch_list[patchID[i]]
        # end

        # Next we need to figure out what is the actual UV coordinate 
        # on the given surface

        uv = zeros((nSurf,2))
        
        for i in xrange(nSurf):

            # Local Element
            local_elem = (nearest_elem[i]-1) - patchID[i]*(Nu-1)*(Nv-1)
            #print local_elem
            # Find out what its row/column index is

            #row = int(floor(local_elem / (Nu-1.0)))  # Integer Division
            row = local_elem / (Nu-1)
            col = mod(local_elem,(Nu-1)) 

            #print nearest_elem[i],local_elem,row,col

            if uvw[0,i] > 1:
                u_local = 1
            elif uvw[0,i] < 0:
                u_local = 0
            else:
                u_local = uvw[0,i]

            if uvw[1,i] > 1:
                v_local = 1
            elif uvw[1,i] < 0:
                v_local = 0
            else:
                v_local = uvw[1,i]

            uv[i,0] =  u_local/(Nu-1)+ col/(Nu-1.0)
            uv[i,1] =  v_local/(Nv-1)+ row/(Nv-1.0)

        # end for

        # Release the tree - otherwise fortran will get upset
        csm_pre.release_adt()

        # THERE IS AN ERROR IN PROJECT POINT (I THINK)
       #  # Check to see how far the coordinate and the surface point is:
#         counter1 = 0
#         counter2 = 0 
#         tol = 1e-20
#         for i in xrange(nSurf):
#             D = coordinates[:,i] - \
#                 self.surfs[patchID[i]].getValue(uv[i,0],uv[i,1])
#             D = sqrt(dot(D,D))

#             if D > tol:
#                 print 'uv before:',uv[i,0],uv[i,1],coordinates[:,i],self.surfs[patchID[i]].getValue(uv[i,0],uv[i,1])
#                 counter1 += 1
#                 uv[i,0],uv[i,1],D2,converged =\
#                     self.surfs[patchID[i]].projectPoint(\
#                     coordinates[:,i],u0=uv[i,0],v0=uv[i,1])
#                 print 'uv after:',uv[i,0],uv[i,1],self.surfs[patchID[i]].getValue(uv[i,0],uv[i,1])
#                 if sqrt(dot(D2,D2)) > tol:
#                     counter2 += 1
#                 # end if
# #             # end if
#         # end for

#         print '%d Points were worse than %f before.'%(counter1,tol)
#         print '%d Points were worse than %f after.'%(counter2,tol)
        print 'Done Surface Attachment'

        return dist,patchID,uv
        

    def calcSurfaceDerivative(self,patchID,uv):
        '''Calculate the (fixed) surface derivative of a discrete set of ponits'''

        print 'Calculating Surface Derivative for %d Points...'%(len(patchID))
        timeA = time.time()
        if not self.J2: # Not initialized
            # Calculate the size Ncoef_free x Ndesign Variables
            
            M = len(patchID)

            N = [0]
            for i in xrange(self.nPatch):
                N.append(N[-1]+self.surfs[i].Nctl_free)
            # end if

            Nctl = N[-1]

            # We know the row filling factor: Its (no more) than ku*kv
            # control points for each control point. Since we don't
            # use more than k=4 we will set at 16
             
            if USE_PETSC:
                self.J2 = PETSc.Mat()
                self.J2.createAIJ([M*3,N[-1]*3],nnz=16*3)
            else:
                self.J2 = zeros((M*3,N[-1]*3))
            # end if
        # end if 
                
        for i in xrange(len(patchID)):
            #print 'patch id:',i
            ku = self.surfs[patchID[i]].ku
            kv = self.surfs[patchID[i]].kv
            Nctlu = self.surfs[patchID[i]].Nctlu
            Nctlv = self.surfs[patchID[i]].Nctlv

            ileftu, mflagu = self.surfs[patchID[i]].pyspline.intrv(\
                self.surfs[patchID[i]].tu,uv[i][0],1)
            ileftv, mflagv = self.surfs[patchID[i]].pyspline.intrv(\
                self.surfs[patchID[i]].tv,uv[i][1],1)

            if mflagu == 0: # Its Inside so everything is ok
                u_list = [ileftu-ku,ileftu-ku+1,ileftu-ku+2,ileftu-ku+3]
            if mflagu == 1: # Its at the right end so just need last one
                u_list = [ileftu-ku-1]

            if mflagv == 0: # Its Inside so everything is ok
                v_list = [ileftv-kv,ileftv-kv+1,ileftv-kv+2,ileftv-kv+3]
            if mflagv == 1: # Its at the right end so just need last one
                v_list = [ileftv-kv-1]

            for ii in xrange(len(u_list)):
                for jj in xrange(len(v_list)):

                    x = self.surfs[patchID[i]].calcPtDeriv(\
                        uv[i][0],uv[i][1],u_list[ii],v_list[jj])

                    index = 3*self.surfs[patchID[i]].globalCtlIndex[\
                        u_list[ii],v_list[jj]]
                    self.J2[3*i  ,index  ] = x
                    self.J2[3*i+1,index+1] = x
                    self.J2[3*i+2,index+2] = x

            # end for

        # end for

        # Assemble the (Constant) J2
        if USE_PETSC:
            self.J2.assemblyBegin()
            self.J2.assemblyEnd()
        # end if

        print 'Finished Surface Derivative in %5.3f seconds'%(time.time()-timeA)

        return
   

class edge(object):

    '''A class for working with patch edges'''

    def __init__(self,init_string):
        
        '''Create an edge based on the init string as read from a file'''
#Example: Two surface wing with pinched tip
#0          1       2       3      4      5            6      7                 8      9      10   

#Conection | Face    Edge  | Type | dof  | Continutiy | Dir? | Driving Group | Nctl | Face    Edge	|	 
#0	   | 0	     0	   | 0	  | 3    | -	      | -    | 0       	     | 10   | -       -      	|
#1	   | 0	     1	   | 1    | -    | 0	      | -1   | 0	     | 10   | 1       1  	|
#2	   | 0	     2	   | 1    | -    | 0	      | 1    | 1	     | 4    | 1       3	        |
#3	   | 0	     3	   | 1    | -    | 1	      | 1    | 1	     | 4    | 1       2	        |
#4	   | 1	     0	   | 0    | 3    | -	      | -    | 0	     | 10   | -       - 	|

        aux = string.split(init_string)

        self.type = int(aux[3])
        self.f1 = int(aux[1])
        self.e1 = int(aux[2])
        self.dg = int(aux[7])
        self.Nctl = int(aux[8])

        if self.type == 0: # Symmetry constraint
            self.dof = int(aux[4])
            self.cont = -1
            self.dir  = 1
            self.f2 = -1
            self.e2 = -1 
            
        else: # Stitch Edge
            self.dof  = 7 #Default all dof move together
            self.cont = int(aux[5])
            self.dir  = int(aux[6])
            self.f2   = int(aux[9])
            self.e2   = int(aux[10])
        # end if

        # Note Conection number is not necessary (only dummy character)
            
        return

    def write_info(self,i,handle):

        handle.write('%3d        |%3d     %3d    |%3d   | %3d  |  %3d       \
| %3d  | %3d           | %3d  | %3d     %3d      |\n'\
              %(i,self.f1,self.e1,self.type,self.dof,self.cont,self.dir,\
                    self.dg,self.Nctl,self.f2,self.e2))
        
        return


class ref_axis(object):

    def __init__(self,X,rot,*args,**kwargs):

        ''' Create a generic reference axis. This object bascally defines a
        set of points in space (x,y,z) each with three rotations
        associated with it. The purpose of the ref_axis is to link
        groups of b-spline controls points together such that
        high-level planform-type variables can be used as design
        variables
        
        Input:

        X: array of size N,3: Contains the x-y-z coodinates of the axis
        rot: array of size N,3: Contains the rotations of the axis

        Note: Rotations are performed in the order: Z-Y-X
        '''

        self.surf_ids = []
        self.links_s = []
        self.links_x = []
        self.surf_sizes = []
        self.surf_sec = []
        self.surf_dir = []
        self.con_type = None
        if not  X.shape == rot.shape:
            print 'The shape of X and rot must be the same'
            print 'X:',X.shape
            print 'rot:',rot.shape
            sys.exit(1)

        # Note: Ref_axis data is ALWAYS Complex. 
        X = X.astype('D')
        rot = rot.astype('D')
        self.N = X.shape[0]

        self.base_point = X[0,:]
        
        self.base_point_s = None
        self.base_point_D = None

        self.end_point   = X[-1,:]
        self.end_point_s = None
        self.end_point_D = None

        # Values are stored wrt the base point
        self.x = X-self.base_point
        self.rot = rot
        self.scale = ones(self.N,'D')

        # Deep copy the x,rot and scale for design variable reference
        self.x0 = copy.deepcopy(self.x)
        self.rot0 = copy.deepcopy(self.rot)
        self.scale0 = copy.deepcopy(self.scale)

        # Create an interpolating spline for the spatial part and use
        # its  basis for the rotatinoal part
        
        self.xs = pySpline.linear_spline(\
            task='interpolate',X=self.base_point+self.x,k=2,complex=True)
        self.s = self.xs.s

        self.rotxs = pySpline.linear_spline(\
            task='interpolate',X=self.rot[:,0],k=2,s=self.s,complex=True)
        self.rotys = pySpline.linear_spline(\
            task='interpolate',X=self.rot[:,1],k=2,s=self.s,complex=True)
        self.rotzs = pySpline.linear_spline(\
            task='interpolate',X=self.rot[:,2],k=2,s=self.s,complex=True)

        self.scales = pySpline.linear_spline(\
            task='interpolate',X=self.scale,k=2,s=self.s,complex=True)

    def update(self):
        
        self.xs.coef = self.base_point+self.x
        self.rotxs.coef = self.rot[:,0]
        self.rotys.coef = self.rot[:,1]
        self.rotzs.coef = self.rot[:,2]

        self.scales.coef = self.scale

        if self.con_type == 'full':
            self.xs.coef[-1,:] = self.end_point
        # end if
        
        return
       
    def writeTecplotAxis(self,handle,axis_name):
        '''Write the ref axis to the open file handle'''
        N = len(self.s)
        handle.write('Zone T=%s I=%d\n'%(axis_name,N))
        values = self.xs.getValueV(self.s)
        for i in xrange(N):
            handle.write('%f %f %f \n'%(values[i,0],values[i,1],values[i,2]))
        # end for

        return

    def writeTecplotLinks(self,handle,surfs):
        '''Write out the surface links. Note surfs is the pyGeo 
        list of surfaces'''

        for ii in xrange(len(self.surf_ids)):
            ipatch = self.surf_ids[ii]

            Nctlu = self.surf_sizes[ii][0]
            Nctlv = self.surf_sizes[ii][1]
            num_vectors = Nctlu*Nctlv
            coords = zeros((2*num_vectors,3))
            icoord = 0
            counter = 0
            crop_coef = surfs[ipatch].getFreeCtl()[self.surf_sec[ii][0]\
                                                     ,self.surf_sec[ii][1]]
            for j in xrange(Nctlv):
                for i in xrange(Nctlu):                    
                    x0 = self.xs.getValue(self.links_s[ii][i,j])
                    coords[icoord    ,:] = x0
                    coords[icoord + 1,:] = crop_coef[i,j,:]
                    counter += 1
                    icoord  += 2
                # end for
            # end for

            icoord = 0
            conn = zeros((num_vectors,2))
            for ivector  in xrange(num_vectors):
                conn[ivector,:] = icoord, icoord+1
                icoord += 2
            # end for

            handle.write('Zone N= %d ,E= %d\n'%(2*num_vectors, num_vectors) )
            handle.write('DATAPACKING=BLOCK, ZONETYPE = FELINESEG\n')

            for n in xrange(3):
                for i in  range(2*num_vectors):
                    handle.write('%f\n'%(coords[i,n]))
                # end for
            # end for

            for i in range(num_vectors):
                handle.write('%d %d \n'%(conn[i,0]+1,conn[i,1]+1))
            # end for
        # end for

        return

    def getRotMatrixGlobalToLocal(self,s):
        
        '''Return the rotation matrix to convert vector from global to
        local frames'''
        return     dot(rotyM(self.rotys(s)),dot(rotxM(self.rotxs(s)),\
                                                    rotzM(self.rotzs(s))))
    
    def getRotMatrixLocalToGloabl(self,s):
        
        '''Return the rotation matrix to convert vector from global to
        local frames'''
        return inv(dot(rotyM(self.rotys(s)),dot(rotxM(self.rotxs(s)),\
                                                    rotzM(self.rotzs(s)))))
    
class geoDVGlobal(object):
     
    def __init__(self,dv_name,value,lower,upper,function):
        
        '''Create a geometric design variable (or design variable group)

        Input:
        
        dv_name: Design variable name. Should be unique. Can be used
        to set pyOpt variables directly

        value: Value of Design Variable
        
        lower: Lower bound for the variable. Again for setting in
        pyOpt

        upper: Upper bound for the variable. '''

        self.name = dv_name
        self.value = value
        if isinstance(value, int):
            self.nVal = 1
        else:
            self.nVal = len(value)

        self.lower = lower
        self.upper = upper
        self.function = function
        return

    def __call__(self,ref_axis):

        '''When the object is called, actually apply the function'''
        # Run the user-supplied function

        return self.function(self.value,ref_axis)
        

class geoDVLocal(object):
     
    def __init__(self,dv_name,lower,upper,surface_id,slice_u,slice_v):
        
        '''Create a set of gemoetric design variables whcih change the shape
        of a surface patch, surface_id

        Input:
        
        dv_name: Design variable name. Should be unique. Can be used
        to set pyOpt variables directly

        lower: Lower bound for the variable. Again for setting in
        pyOpt

        upper: Upper bound for the variable.

        Note: Value is NOT specified, value will ALWAYS be initialized to 0

        '''

        self.Nctlu = slice_u.stop-slice_u.start
        self.Nctlv = slice_v.stop-slice_v.start
        self.nVal = self.Nctlu*self.Nctlv
        self.slice_u = slice_u
        self.slice_v = slice_v                                 
    
        self.value = zeros((self.Nctlu,self.Nctlv),'D')
        self.name = dv_name
        self.lower = lower
        self.upper = upper
        self.surface_id = surface_id
        return

    def __call__(self,surface):

        '''When the object is called, apply the design variable values to the
        surface'''
        #call the surface with the values
        surface.updateSurfacePoints(self.value,self.slice_u,self.slice_v)
        return surface

#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'

