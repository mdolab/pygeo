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
    real, imag, dstack, floor, size, reshape

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
    version = petsc4py.__version__
    vals = string.split(version,'.')
    PETSC_MAJOR_VERSION = int(vals[0])
    PETSC_MINOR_VERSION = int(vals[1])
    PETSC_UPDATE        = int(vals[2])
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

        self.ref_axis       = []
        self.ref_axis_con   = []
        self.DV_listGlobal  = []
        self.DV_listNormal  = []
        self.DV_listLocal   = []
        self.DV_namesGlobal = {}
        self.DV_namesNormal = {}
        self.DV_namesLocal  = {}
        self.petsc_coef = None # Global vector of PETSc coefficients
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
        nSurf = int(f.readline())
        
        print 'nSurf = %d'%(nSurf)

        patchSizes = zeros(nSurf*3,'intc')

        # We can do 24 sizes per line 
        nHeaderLines = 3*nSurf / 24 
        if 3*nSurf% 24 != 0: nHeaderLines += 1

        counter = 0

        for nline in xrange(nHeaderLines):
            aux = string.split(f.readline())
            for i in xrange(len(aux)):
                patchSizes[counter] = int(aux[i])
                counter += 1

        patchSizes = patchSizes.reshape([nSurf,3])

        assert patchSizes[:,2].all() == 1, \
            'Error: Plot 3d does not contain only surface patches.\
 The third index (k) MUST be 1.'

        # Total points
        nPts = 0
        for i in xrange(nSurf):
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

        for isurf in xrange(nSurf):
            patches.append(zeros([patchSizes[isurf,0],patchSizes[isurf,1],3]))
            for idim in xrange(3):
                for j in xrange(patchSizes[isurf,1]):
                    for i in xrange(patchSizes[isurf,0]):
                        patches[isurf][i,j,idim] = dataTemp[counter]
                        counter += 1
                    # end for
                # end for
            # end for
        # end for

        # Now create a list of spline objects:
        surfs = []
        for isurf in xrange(nSurf):
            surfs.append(pySpline.surf_spline(task='lms',X=patches[isurf],\
                                                  ku=4,kv=4,Nctlu=13,Nctlv=13))

        self.surfs = surfs
        self.nSurf = nSurf
        return

    def _readIges(self,file_name,*args,**kwargs):

        '''Load a Iges file and create the splines to go with each patch'''
        print 'File Name is: %s'%(file_name)
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
        self.nSurf = len(surf_list)
        
        print 'Found %d surfaces in Iges File.'%(self.nSurf)

        self.surfs = [];
        #print surf_list
        weight = []
        for isurf in xrange(self.nSurf):  # Loop over our patches
            data = []
            # Create a list of all data
            # -1 is for conversion from 1 based (iges) to python
            para_offset = surf_list[isurf][0]+dir_offset+directory_lines-1 

            for i in xrange(surf_list[isurf][1]):
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

        else:
            breaks = None
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
            self.nSurf = len(self.surfs)


        else:  #No breaks
            Nctlv = naf
            self.surfs.append(pySpline.surf_spline(fit_type,ku=4,kv=4,X=X[0],\
                                                       Nctlv=Nctlv,*args,**kwargs))
            self.surfs.append(pySpline.surf_spline(fit_type,ku=4,kv=4,X=X[1],\
                                                       Nctlv=Nctlv,*args,**kwargs))
            self.nSurf = 2

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

        self.nSurf = len(self.surfs)
		
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
        for isurf in xrange(self.nSurf):
            # Test this patch against the rest
            for i in xrange(4):
                for jsurf in xrange(isurf+1,self.nSurf):

                    for j in xrange(4):
                        coinc,dir_flag=test_edge(\
                            self.surfs[isurf],self.surfs[jsurf],i,j,edge_tol)
                        cont_flag = 0 # By Default only C0 continuity
                        if coinc:
                            #print 'We have a coincidient edge'
                            e_con.append([[isurf,i],[jsurf,j],cont_flag,\
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
        for i in xrange(self.nSurf):
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
        # Finally Print Connection Info
        self.printEdgeConnectivity()
        print 'going to set edge connectivity'

        self._setEdgeConnectivity()
        print 'Time for Edge Calculation:',time.time()-timeA
        return

    def _findConIndex(self,isurf,edge=None):
        '''Find the index of the entry in the edge list for isurf and edge'''
        for i in xrange(len(self.con)):
            if self.con[i].f1 == isurf and self.con[i].e1 == edge:
                return i,True
            # end if
            if self.con[i].f2 == isurf and self.con[i].e2 == edge:
                return i,False
            # end if
        # end if
        print 'Error: Edge was not found in the edge list. EVERY edge MUST \
appear in the edge con list'
        sys.exit(1)
        return 

    def _setEdgeConnectivity(self):
        '''Internal function to calculate the globalCtlIndex for each surface'''
        if self.con == None:
            print 'Error: No edge connectivity is set yet. Either run \
 calcEdgeConnectivity or load in a .con file'
            sys.exit(1)
        # end if

        # This function will set the globalCtlIndex for each surface

        # Call the new function

        sizes = []
        for isurf in xrange(self.nSurf):
            sizes.append([self.surfs[isurf].Nctlu,self.surfs[isurf].Nctlv])
        # end for
        total,g_index,l_index = self.calcGlobalNumbering(sizes)
        self.global_coef = g_index
        self.Ncoef = total  #Total Number of Free Coefficients
              
        for isurf in xrange(self.nSurf):
            self.surfs[isurf].globalCtlIndex = l_index[isurf]
        # end for
        
        self.coef = []
        # Now Fill up the self.coef list:
        for ii in xrange(len(self.global_coef)):
            isurf = self.global_coef[ii][0][0]
            i = self.global_coef[ii][0][1]
            j = self.global_coef[ii][0][2]
            self.coef.append( self.surfs[isurf].coef[i,j])
        # end for
            
        # Finally turn self.coef into a complex array
        self.coef = array(self.coef,'D')

        # Create a PETSc vector of the global knots
        if USE_PETSC:
            self.petsc_coef = PETSc.Vec()
            self.petsc_coef.createSeq(3*self.Ncoef)
            self.petsc_coef[:] = self.coef.flatten().astype('d')
        # end
        return

    def calcGlobalNumbering(self,sizes,surface_list=None):
        '''Internal function to calculate the globalCtlIndex for each surface'''
        if self.con == None:
            print 'Error: No edge connectivity is set yet. Either run \
 calcEdgeConnectivity or load in a .con file'
            sys.exit(1)
        # end if

        def getIndexEdge(isurf,edge,index,dir):
            '''Get the global index value from edge,index,dir information'''
            if edge == 0:
                if dir == 1:
                    return l_index[isurf][index,0]
                else:
                    return l_index[isurf][Nu-1-index,0]
                # end if
            elif edge == 1:
                if dir == 1:
                    return l_index[isurf][index,Nv-1]
                else:
                    return l_index[isurf][Nu-1-index,Nv-1]
                # end if
            elif edge == 2:
                if dir == 1:
                    return l_index[isurf][0,index]
                else:
                    return l_index[isurf][0,Nv-1-index]
                # end if
            elif edge == 3:
                if dir == 1:
                    return l_index[isurf][Nu-1,index]
                else:
                    return l_index[isurf][Nu-1,Nv-1-index]
                # end if
            # end if

            return
             
        # Check if sizes is the same length as the number of surfaces
        #assert len(sizes) == self.nSurf,'The length of sizes must be the same as the \
        #number of surface points'

        # This function will calculate a global index ordering for a
        # set of surfaces with the supplied connectivity
      
        def add_master(counter):
            '''Add a master control point'''
            l_index[isurf][i,j] = counter
            counter =counter + 1
            g_index.append([[isurf,i,j]])
            return counter

        def add_slave(icon,index):
            current_index = getIndexEdge(self.con[icon].f1, self.con[icon].e1, index,\
                                 self.con[icon].dir)
            g_index[current_index].append([isurf,i,j])
            l_index[isurf][i,j] = current_index
            return
        
        counter = 0
        g_index = []
        l_index = []

        if not surface_list: # W
            surface_list = range(0,self.nSurf)            
        
        for ii in xrange(len(surface_list)):
            isurf = surface_list[ii]
            Nu = sizes[isurf][0]
            Nv = sizes[isurf][1]
            l_index.append(zeros((Nu,Nv),'intc'))
            for i in xrange(Nu):
                for j in xrange(Nv):
                    # This is the basic "internal" control type
                    if i > 0 and i < Nu -1 and j > 0 and j < Nv -1:
                        counter = add_master(counter)

                    # There are 8 other possibilites now: Each of 4
                    # edges and 4 corners. Do the edges first
                    else:
                        if i > 0 and i < Nu-1 and j == 0:       # Edge 0
                            icon, master = self._findConIndex(isurf,edge=0)
                            if master:
                                counter = add_master(counter)
                            else:
                                add_slave(icon,i)
                            # end if
                      
                        elif i > 0 and i < Nu-1 and j == Nv-1: # Edge 1
                            icon, master = self._findConIndex(isurf,edge=1)
                            if master:
                                counter = add_master(counter)
                            else:
                                add_slave(icon,i)
                            # end if
                   
                        elif i == 0 and j > 0 and j < Nv -1:      # Edge 2
                            icon, master = self._findConIndex(isurf,edge=2)
                            if master:
                                counter = add_master(counter)
                            else:
                                add_slave(icon,j)
                            # end if

                        elif i == Nu-1 and j > 0 and j < Nv-1: # Edge 3
                            icon, master = self._findConIndex(isurf,edge=3)
                            if master:
                                counter = add_master(counter)
                            else:
                                add_slave(icon,j)
                            # end if

                        elif i == 0 and j == 0:             # Node 0
                            icon1,master1 = self._findConIndex(isurf,edge=0)
                            icon2,master2 = self._findConIndex(isurf,edge=2)
                            if master1 and master2:
                                counter = add_master(counter)
                            else:
                                if master1 == False:
                                    add_slave(icon1,0)
                                else:
                                    add_slave(icon2,0)
                                # end if
                            # end if

                        elif i == Nu-1 and j == 0:       # Node 1
                            icon1,master1 = self._findConIndex(isurf,edge=0)
                            icon2,master2 = self._findConIndex(isurf,edge=3)
                            if master1 and master2:
                                counter = add_master(counter)
                            else:
                                if master1 == False:
                                    add_slave(icon1,Nu-1)
                                else:
                                    add_slave(icon2,0)
                                # end if 
                            # end if
                        
                        elif i == 0 and j == Nv-1:       # Node 2
                            icon1,master1 = self._findConIndex(isurf,edge=1)
                            icon2,master2 = self._findConIndex(isurf,edge=2)
                            if master1 and master2:
                                counter = add_master(counter)
                            else:
                                if master1 == False:
                                    add_slave(icon1,0)
                                else:
                                    add_slave(icon2,Nv-1)
                                # end if 
                            # end if

                        elif i == Nu-1 and j == Nv-1: # Node 3
                            icon1,master1 = self._findConIndex(isurf,edge=1)
                            icon2,master2 = self._findConIndex(isurf,edge=3)
                            if master1 and master2:
                                counter = add_master(counter)
                            else:
                                if master1 == False:
                                    add_slave(icon1,Nu-1)
                                else:
                                    add_slave(icon2,Nv-1)
                                # end if 
                            # end if
                        # end if (edges and nodes)
                    # end if (middle or not)
                # end for  (j loop - Nv)
            # end for (i loop - Nu)
        # end for (isurf loop)
     
        return counter,g_index,l_index

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

        self._sortEdgeConnectivity() # Edge Connections MUST be sorted
        self.printEdgeConnectivity()
        self._setEdgeConnectivity()

        return
    

    def _sortEdgeConnectivity(self):
        # Go through edges and give each a unique id (uid). Connectd
        # edges are first and then free ones
        uid_max = 0
        for i in xrange(len(self.con)):
            if self.con[i].type == 0:
                pass
            else:
                uid = self.con[i].f1*4 + self.con[i].e1
                if uid>uid_max:
                    uid_max = uid

                self.con[i].uid = uid
            # end if
        # end for
        uid_max += 1
        for i in xrange(len(self.con)):
            if self.con[i].type == 0:
                self.con[i].uid = uid_max + self.con[i].f1*4 + self.con[i].e1
            # end if
        # end for

        # Now sort them in place 
        self.con.sort()
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
        for isurf in xrange(self.nSurf):
            self.surfs[isurf].recompute()
        # end for

        return

    def checkCoef(self):
        '''Check all surface coefficients for consistency'''
        for isurf in xrange(self.nSurf):
            counter = self.surfs[isurf].checkCoef()
            if counter > 0:
                print '%d control points on surface %d'%(counter,isurf)
        # end for


# ----------------------------------------------------------------------
#                        Surface Fitting Functions
# # ----------------------------------------------------------------------

#     def fitSurfaces(self):
#         '''This function does a lms fit on all the surfaces respecting
#         the stitched edges as well as the continuity constraints'''

#         # Make sure number of free points are calculated

#         for isurf in xrange(self.nSurf):
#             self.surfs[isurf]._calcNFree()
#         # end for

#         # Size of new jacobian and positions of block starts
#         self.M = [0]
#         self.N = [0]
#         for isurf in xrange(0,self.nSurf):
#             self.M.append(self.M[isurf] + self.surfs[isurf].Nu_free*\
#                               self.surfs[isurf].Nv_free)
#             self.N.append(self.N[isurf] + self.surfs[isurf].Nctlu_free*\
#                               self.surfs[isurf].Nctlv_free)
#         # end for
#         print 'M,N:',self.M,self.N

#         self._initJacobian()

#         #Do Loop to fill up the matrix
#         col_counter = -1
#         print 'Generating Matrix...'
#         for isurf in xrange(self.nSurf):
#             #print 'Patch %d'%(isurf)
#             for j in xrange(self.surfs[isurf].Nctlv):
#                 per_don =((j+0.0)/self.surfs[isurf].Nctlv) # Percent Done
#                 #print 'done %4.2f'%(per_don)

#                 for i in xrange(self.surfs[isurf].Nctlu):
#                     pt_type,edge_info,node_info = \
#                         self.surfs[isurf].checkCtl(i,j) 

#                     if pt_type == 0: # Its a driving node
#                         col_counter += 1
#                         self._setCol(self.surfs[isurf]._calcCtlDeriv(i,j),\
#                                          self.M[isurf],col_counter)

#                         # Now check for nodes/edges

#                         # Its on a master edge driving another control point
#                         if edge_info: 

#                             # Unpack edge info
#                             face  = edge_info[0][0]
#                             edge  = edge_info[0][1]
#                             index = edge_info[1]
#                             direction = edge_info[2]
#                             edge_type = edge_info[3]

#                             if edge_type == 1:
#                                 self._setCol(self.surfs[face]._calcCtlDerivEdge\
#                                                  (edge,index,direction),\
#                                                  self.M[face],col_counter)

#                         # Its on a corner driving (potentially)
#                         # multiplie control points
#                         if node_info: 
#                             # Loop over the number of affected nodes
#                             for k in xrange(len(node_info)): 
#                                 face = node_info[k][0]
#                                 node = node_info[k][1]
#                                 self._setCol(self.surfs[face].\
#                                                  _calcCtlDerivNode(node),\
#                                                  self.M[face],col_counter)
#                             # end for
#                         # end if
#                     # end if
#                 # end for
#             # end for
#         # end for

#         # Set the RHS
#         print 'Done Matrix...'
#         self._setRHS()
#         # Now Solve
#         self._solve()
      
#         return

#     def _initJacobian(self):
        
#         '''Initialize the Jacobian either with PETSc or with Numpy for use
# with LAPACK'''
#         if USE_PETSC:
#             self.J = PETSc.Mat()
#             # Approximate Number of non zero entries per row:
#             nz = self.surfs[0].Nctl_free
#             for i in xrange(1,self.nSurf):
#                 if self.surfs[i].Nctl_free > nz:
#                     nz = self.surfs[i].Nctl_free
#                 # end if
#             # end for
#             self.J.createAIJ([self.M[-1],self.N[-1]],nnz=nz)
#         else:
#             self.J = zeros([self.M[-1],self.N[-1]])
#         # end if

#     def _setCol(self,vec,i,j):
#         '''Set a column vector, vec, at position i,j'''
#         # Note: These are currently the same...
#         # There is probably a more efficient way to set in PETSc
#         if USE_PETSC:
#             self.J[i:i+len(vec),j] = vec 
#             #self.J.setValues(len(vec),i,1,j,vec)
#         else:
#             self.J[i:i+len(vec),j] = vec 
#         # end if
#         return 

        
#     def _setRHS(self):
#         '''Set the RHS Vector'''
#         #NOTE: GET CROP DATA IS NOT WORKING! FIX ME!
        
        
#         self.RHS = self.nSurf*[]
#         for idim in xrange(3):
#             if USE_PETSC:
#                 self.RHS.append(PETSc.Vec())
#                 self.RHS[idim].createSeq(self.M[-1])
#             else:
#                 self.RHS.append(zeros(self.M[-1]))
#             # end if 
#             for isurf in xrange(self.nSurf):
#                 temp = self.surfs[isurf]._getCropData(\
#                     self.surfs[isurf].X[:,:,idim]).flatten()
#                 self.RHS[idim][self.M[isurf]:self.M[isurf] + len(temp)] = temp
#             # end for
#         # end for

 #    def _solve(self):
#         '''Solve for the control points'''
#         print 'in solve...'
#         self.coef = zeros((self.N[-1],3))
#         if USE_PETSC:
#             self.J.assemblyBegin()
#             self.J.assemblyEnd()

                        
#             ksp = PETSc.KSP()
#             ksp.create(PETSc.COMM_WORLD)
#             ksp.getPC().setType('none')
#             ksp.setType('lsqr')
           
#             def monitor(ksp, its, rnorm):
#                 if mod(its,50) == 0:
#                     print its,rnorm

#             ksp.setMonitor(monitor)
#             #ksp.setMonitor(ksp.Monitor())
#             ksp.setTolerances(rtol=1e-15, atol=1e-15, divtol=100, max_it=500)

#             X = PETSc.Vec()
#             X.createSeq(self.N[-1])

#             ksp.setOperators(self.J)
            
#             for idim in xrange(3):
#                 print 'solving %d'%(idim)
#                 ksp.solve(self.RHS[idim], X) 
#                 for i in xrange(self.N[-1]):
#                     self.coef[i,idim] = X.getValue(i)
#                 # end if
#             # end for
#         else:
#             for idim in xrange(3):
#                 X = lstsq(self.J,self.RHS[idim])
#                 self.coef[:,idim] = X[0]
#                 print 'residual norm:',X[1]
#             # end for
#         # end if

#         data_save = {'COEF':self.coef}
#         #io.savemat('coef_lapack.mat',data_save)

#         return

# ----------------------------------------------------------------------
#                Reference Axis Handling
# ----------------------------------------------------------------------

    def addRefAxis(self,surf_ids,X,rot,nrefsecs=None,spacing=None,\
                       point_select=None):
            '''Add surf_ids surfacs to a new reference axis defined by X and
             rot with nsection values'''

            print 'Adding ref axis...'
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

            coef_list = []
            if point_select == None: # It is was not defined -> Assume full surface
                for isurf in surf_ids:
                    for i in xrange(self.surfs[isurf].Nctlu):
                        for j in xrange(self.surfs[isurf].Nctlv):
                            coef_list.append(self.surfs[isurf].globalCtlIndex[i,j])
                        # end for
                    # end for
                # end for
            # end if

            else:   # We have a point selection class passed in
                for isurf in surf_ids:
                    coef_list = point_select.getControlPoints(\
                        self.surfs[isurf],coef_list)
                # end for
            # end if

            # Now parse out duplicates and sort
            coef_list = unique(coef_list)
            coef_list.sort()
            N = len(coef_list)
            for i in xrange(len(coef_list)):
                isurf = self.global_coef[coef_list[i]][0][0]
                s,D,conv,eps = ra.xs.projectPoint(self.coef[coef_list[i]])

                base_point = ra.xs.getValue(s)
                D = self.coef[coef_list[i]] - base_point
                M = ra.getRotMatrixGlobalToLocal(s)
                D = dot(M,D) #Rotate to local frame
                ra.links_s.append(s)
                ra.links_x.append(D)
            # end for
            ra.coef_list = coef_list
            ra.surf_ids  = surf_ids
            
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
a flap hinge line'
            
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

    def _updateCoef(self,local=True):
        '''update the entire pyGeo Object'''
        
        # First, update the reference axis info from the design variables
        for i in xrange(len(self.DV_listGlobal)):
            # Call the each design variable with the ref axis list
            self.ref_axis = self.DV_listGlobal[i](self.ref_axis)
        # end for

        # Second, update the end_point base_point on the ref_axis:
        
        if len(self.ref_axis_con)> 0:
            for i in xrange(len(self.ref_axis_con)):
                axis1 = self.ref_axis_con[i][0]
                axis2 = self.ref_axis_con[i][1]

                self.ref_axis[axis1].update()
                s = self.ref_axis[axis2].base_point_s
                D = self.ref_axis[axis2].base_point_D
                M = self.ref_axis[axis1].getRotMatrixLocalToGlobal(s)
                D = dot(M,D)

                X0 = self.ref_axis[axis1].xs.getValue(s)

                self.ref_axis[axis2].base_point = X0 + \
                    D*self.ref_axis[axis1].scales(s)

                if self.ref_axis[axis2].con_type == 'full':
                    s = self.ref_axis[axis2].end_point_s
                    D = self.ref_axis[axis2].end_point_D
                    M = self.ref_axis[axis1].getRotMatrixLocalToGlobal(s)
                    D = dot(M,D)

                    X0 = self.ref_axis[axis1].xs.getValue(s)

                    self.ref_axis[axis2].end_point = X0 +\
                        D*self.ref_axis[axis1].scales(s)
                # end if
                self.ref_axis[axis2].update()
        else:
            for r in xrange(len(self.ref_axis)):
                self.ref_axis[r].update()
            # end for
       
        # Third, update the coefficients (from global DV changes)
        for r in xrange(len(self.ref_axis)):
            # Call the fortran function
            ra = self.ref_axis[r]
            rot = zeros((len(ra.xs.s),3),'D')
            rot[:,0] = ra.rotxs.coef
            rot[:,1] = ra.rotys.coef
            rot[:,2] = ra.rotzs.coef
                        
            #coef = getcoef(type,s_pos,links,coef,indicies,s,t,x,rot,scale)
            if ra.con_type == 'full':
                self.coef = pySpline.pyspline_cs.getcomplexcoef(\
                1,ra.links_s,ra.links_x,self.coef,ra.coef_list,\
                        ra.xs.s,ra.xs.t,ra.xs.coef,rot,ra.scales.coef)
            else:
                self.coef = pySpline.pyspline_cs.getcomplexcoef(\
                    0,ra.links_s,ra.links_x,self.coef,ra.coef_list,\
                        ra.xs.s,ra.xs.t,ra.xs.coef,rot,ra.scales.coef)
            # end if
#---------------- PYTHON IMPLEMENTATION  ------------------
           #  for i in xrange(len(ra.links_s)):
#                 base_point = ra.xs.getValue(ra.links_s[i])
#                 D = ra.links_x[i]
#                 M = ra.getRotMatrixLocalToGlobal(ra.links_s[i])
#                 D = dot(M,D)
#                 coef[ra.coef_list[i]] = base_point + D*ra.scales(s)
#             # end for
# ---------------------------------------------------------
        # end for

        # fourth, update the coefficients (from local DV changes)
        if local:
            for i in xrange(len(self.DV_listLocal)):
                surface = self.surfs[self.DV_listLocal[i].surface_id]
                self.coef = self.DV_listLocal[i](surface,self.coef)
            # end for
        # end if

        return
         
    def update(self):
        '''Run the update coefficients command and then set the control
        points'''
        self._updateCoef(local=True)

        # Update the values in PETSc
        if USE_PETSC:
            self.petsc_coef[:] = self.coef.flatten().astype('d')
        # end
            
        self._updateSurfaceCoef()
        return

    def _updateSurfaceCoef(self):
        '''Copy the pyGeo list of control points back to the surfaces'''
        for ii in xrange(len(self.coef)):
            for jj in xrange(len(self.global_coef[ii])):
                isurf = self.global_coef[ii][jj][0]
                i     = self.global_coef[ii][jj][1]
                j     = self.global_coef[ii][jj][2]
                self.surfs[isurf].coef[i,j] = self.coef[ii].astype('d')
            # end for
        # end for

        return

    def getSizes( self ):
        '''
        Get the sizes:
        - The number of global design variables
        - The number of local design variables
        - The number of control points
        '''
        
        # Initialize the jacobian
        # Calculate the size Ncoef x Ndesign Variables
        Nctl = len(self.coef)

        # Calculate the Number of Design Variables:
        N = 0
        for i in xrange(len(self.DV_listGlobal)): #Global Variables
            if self.DV_listGlobal[i].useit:
                N += self.DV_listGlobal[i].nVal
            # end if
        # end for
            
        NdvGlobal = N
        Ndv = N
        for i in xrange(len(self.DV_listLocal)): # Local Variables
            Ndv += self.DV_listLocal[i].nVal
        # end for
                
        NdvLocal = Ndv-NdvGlobal

        # print 'NdvGlobal:',NdvGlobal
        # print 'Ndv:',Ndv
        # print 'Nctl:',Nctl

        return NdvGlobal, NdvLocal, Nctl


    def _initJ1( self ):
        '''
        Allocate the space for J1 and perform some setup        
        '''

        NdvGlobal, NdvLocal, Nctl = self.getSizes()
        Ndv = NdvGlobal + NdvLocal
        
        if USE_PETSC:
            J1 = PETSc.Mat()
            
            # We know the row filling factor: Its (exactly) nGlobal + 3            
            if PETSC_MAJOR_VERSION == 1:
                J1.createAIJ([Nctl*3,Ndv],nnz=NdvGlobal+3,comm=PETSc.COMM_SELF)
            elif PETSC_MAJOR_VERSION == 0:
                J1.createSeqAIJ([Nctl*3,Ndv],nz=NdvGlobal+3)
            else:
                print 'Error: PETSC_MAJOR_VERSION = %d is not supported'%(PETSC_MAJOR_VERSION)
                sys.exit(1)
            # end if
        else:
            J1 = zeros((Nctl*3,Ndv))
        # end if

        return J1
        

    def calcCtlDeriv(self):

        '''This function runs the complex step method over the design variable
        and generates a (sparse) jacobian of the control pt
        derivatives wrt to the design variables'''

        if not self.J1:
            self.J1 = self._initJ1()
        # end
   
        h = 1.0e-40j
        col_counter = 0
        for idv in xrange(len(self.DV_listGlobal)): # This is the Master CS Loop
            if self.DV_listGlobal[idv].useit:
                nVal = self.DV_listGlobal[idv].nVal

                for jj in xrange(nVal):
                    if nVal == 1:
                        self.DV_listGlobal[idv].value += h
                    else:
                        self.DV_listGlobal[idv].value[jj] += h
                    # end if

                    # Now get the updated coefficients and set the column
                    self._updateCoef(local=False)
                    self.J1[:,col_counter] = imag(self.coef.flatten())/1e-40
                    col_counter += 1    # Increment Column Counter

                    # Reset Design Variable Peturbation
                    if nVal == 1:
                        self.DV_listGlobal[idv].value -= h
                    else:
                        self.DV_listGlobal[idv].value[jj] -= h
                    # end if
                # end for (nval loop)
            # end if (useit)
        # end for (outer design variable loop)
        
        # The next step is go to over all the LOCAL variables,
        # compute the surface normal
        print 'col counter before local dv',col_counter
        for idv in xrange(len(self.DV_listLocal)): # This is the Master CS Loop
            surface = self.surfs[self.DV_listLocal[idv].surface_id]
            normals = self.DV_listLocal[idv].getNormals(\
                surface,self.coef.astype('d'))

            # Normals is the length of local dv on this surface
            for i in xrange(self.DV_listLocal[idv].nVal):
                index = 3*self.DV_listLocal[idv].coef_list[i]
                self.J1[index:index+3,col_counter] = normals[i,:]
                col_counter += 1
            # end for
        # end for
            
        if USE_PETSC:
            self.J1.assemblyBegin()
            self.J1.assemblyEnd()
        # end if 

        # Now Do the Try the matrix multiplication
        if USE_PETSC:
            if self.J2:
                if not self.C:
                    self.C = PETSc.Mat()
                # end
                self.J2.matMult(self.J1,result=self.C)
            # end
        else:
            if self.J2:
                self.C = dot(self.J2,self.J1)
            # end
        # end if

        return 

    def getSurfacePoints(self,patchID,uv):

        '''Function to return ALL surface points'''

        N = len(patchID)
        coordinates = zeros((N,3))
        for i in xrange(N):
            coordinates[i] = self.surfs[patchID[i]].getValue(uv[i][0],uv[i][1])

        return coordinates.flatten()

    def addGeoDVNormal(self,dv_name,lower,upper,surf=None,point_select=None,\
                           overwrite=False):

        '''Add a normal local design variable group.'''

        if surf == None:
            print 'Error: A surface must be specified with surf = <surf_id>'
            sys.exit(1)
        # end if

        coef_list = []
        if point_select == None:
            counter = 0
            # Assume all control points on surface are to be used
            for i in xrange(self.surfs[surf].Nctlu):
                for j in xrange(self.surfs[surf].Nctlv):
                    coef_list.append(self.surfs[surf].globalCtlIndex[i,j])
                # end for
            # end for
        else:
            # Use the bounding box to find the appropriate indicies
            coef_list = point_select.getControlPoints(\
                self.surfs[surf],coef_list)
        # end if
        
        # Now, we have the list of the conrol points that we would
        # LIKE to add to this dv group. However, some may already be
        # specified in other normal of local dv groups. 

        if overwrite:
            # Loop over ALL normal and local group and force them to
            # remove all dv in coef_list

            for idv in xrange(len(self.DV_listNormal)):
                self.DV_listNormal[idv].removeCoef(coef_list)
            # end for
            
            for idv in xrange(len(self.DV_listLocal)):
                self.DV_listLocal[idv].removeCoef(coef_list)
        else:
            # We need to (possibly) remove coef from THIS list since
            # they already exist on other dvlocals or dvnormals
           
            new_list = copy.copy(coef_list)
            for i in xrange(len(coef_list)):

                for idv in xrange(len(self.DV_listNormal)):
                    if coef_list[i] in self.DV_listNormal[idv].coef_list:
                        new_list.remove(coef_list[i])
                    # end if
                # end for
                for idv in xrange(len(self.DV_listLocal)):
                    if coef_list[i] in self.DV_listLocal[idv].coef_list:
                        new_list.remove(coef_list[i])
                    # end if
                # end for
            # end for
            coef_list = new_list
        # end if

        self.DV_listNormal.append(geoDVNormal(\
                dv_name,lower,upper,surf,coef_list,self.global_coef))
        self.DV_namesNormal[dv_name] = len(self.DV_listLocal)-1
        
        return



    def addGeoDVLocal(self,dv_name,lower,upper,surf=None,point_select=None,\
                          overwrite=False):

        '''Add a general local design variable group.'''

        if surf == None:
            print 'Error: A surface must be specified with surf = <surf_id>'
            sys.exit(1)
        # end if

        coef_list = []
        if point_select == None:
            counter = 0
            # Assume all control points on surface are to be used
            for i in xrange(self.surfs[surf].Nctlu):
                for j in xrange(self.surfs[surf].Nctlv):
                    coef_list.append(self.surfs[surf].globalCtlIndex[i,j])
                # end for
            # end for
        else:
            # Use the bounding box to find the appropriate indicies
            coef_list = point_select.getControlPoints(\
                self.surfs[surf],coef_list)
        # end if
        
        # Now, we have the list of the conrol points that we would
        # LIKE to add to this dv group. However, some may already be
        # specified in other normal of local dv groups. 

        if overwrite:
            # Loop over ALL normal and local group and force them to
            # remove all dv in coef_list

            for idv in xrange(len(self.DV_listNormal)):
                self.DV_listNormal[idv].removeCoef(coef_list)
            # end for
            
            for idv in xrange(len(self.DV_listLocal)):
                self.DV_listLocal[idv].removeCoef(coef_list)
        else:
            # We need to (possibly) remove coef from THIS list since
            # they already exist on other dvlocals or dvnormals
           
            new_list = copy.copy(coef_list)
            for i in xrange(len(coef_list)):

                for idv in xrange(len(self.DV_listNormal)):
                    if coef_list[i] in self.DV_listNormal[idv].coef_list:
                        new_list.remove(coef_list[i])
                    # end if
                # end for
                for idv in xrange(len(self.DV_listLocal)):
                    if coef_list[i] in self.DV_listLocal[idv].coef_list:
                        new_list.remove(coef_list[i])
                    # end if
                # end for
            # end for
            coef_list = new_list
        # end if

        self.DV_listLocal.append(geoDVLocal(\
                dv_name,lower,upper,surf,coef_list,self.global_coef))
        self.DV_namesLocal[dv_name] = len(self.DV_listLocal)-1
        
        return


    def addGeoDVGlobal(self,dv_name,value,lower,upper,function,useit=True):
        '''Add a global design variable'''
        self.DV_listGlobal.append(geoDVGlobal(\
                dv_name,value,lower,upper,function,useit))
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
            sys.stdout.write('Outputting Surface: ')
            for isurf in xrange(self.nSurf):
                sys.stdout.write('%d '%(isurf))
                self.surfs[isurf].writeTecplotSurface(handle=f,size=0.03)

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
                self.writeTecplotLinks(f,self.ref_axis[r])
            # end for
        # end if
              
        f.close()
        sys.stdout.write('\n')
        return


    def writeTecplotLinks(self,handle,ref_axis):
        '''Write out the surface links. '''

        num_vectors = len(ref_axis.links_s)
        coords = zeros((2*num_vectors,3))
        icoord = 0
    
        for i in xrange(len(ref_axis.links_s)):
            coords[icoord    ,:] = ref_axis.xs.getValue(ref_axis.links_s[i])
            coords[icoord +1 ,:] = self.coef[ref_axis.coef_list[i]]
            icoord += 2
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

        return


    def writeIGES(self,file_name):
        '''write the surfaces to IGES format'''
        f = open(file_name,'w')

        #Note: Eventually we may want to put the CORRECT Data here
        f.write('                                                                        S      1\n')
        f.write('1H,,1H;,7H128-000,11H128-000.IGS,9H{unknown},9H{unknown},16,6,15,13,15, G      1\n')
        f.write('7H128-000,1.,1,4HINCH,8,0.016,15H19970830.165254,0.0001,0.,             G      2\n')
        f.write('21Hdennette@wiz-worx.com,23HLegacy PDD AP Committee,11,3,               G      3\n')
        f.write('13H920717.080000,23HMIL-PRF-28000B0,CLASS 1;                            G      4\n')
        
        Dcount = 1;
        Pcount = 1;

        for isurf in xrange(self.nSurf):
            Pcount,Dcount =self.surfs[isurf].writeIGES_directory(\
                f,Dcount,Pcount)

        Pcount  = 1
        counter = 1

        for isurf in xrange(self.nSurf):
            Pcount,counter = self.surfs[isurf].writeIGES_parameters(\
                f,Pcount,counter)

        # Write the terminate statment
        f.write('S%7dG%7dD%7dP%7d%40sT%7s\n'%(1,4,Dcount-1,counter-1,' ',' '))
        f.close()

        return

    # ----------------------------------------------------------------------
    #                              Utility Functions 
    # ----------------------------------------------------------------------

    def getCoordinatesFromFile(self,file_name):
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
    
    def attachSurface(self,coordinates,patch_list=None,Nu=20,Nv=20,force_domain=True):

        '''Attach a list of surface points to either all the pyGeo surfaces
        of a subset of the list of surfaces provided by patch_list.

        Arguments:
             coordinates   :  a 3 by nPts numpy array
             patch_list    :  list of patches to locate next to nodes,
                              None means all patches will be used
             Nu,Nv         :  parameters that control the temporary
                              discretization of each surface        
             
        Returns:
             dist          :  distance between mesh location and point
             patchID       :  patch on which each u,v coordinate is defined
             uv            :  u,v coordinates in a 2 by nPts array.

        Modified by GJK to include a search on a subset of surfaces.
        This is useful for associating points in a mesh where points may
        lie on the edges between surfaces. Often, these points must be used
        twice on two different surfaces for load/displacement transfer.        
        '''
        print ''
        print 'Attaching a discrete surface to the Geometry Object...'

        if patch_list == None:
            patch_list = range(self.nSurf)
        # end

        nPts = coordinates.shape[1]
        
        # Now make the 'FE' Grid from the sufaces.
        patches = len(patch_list)
        
        nelem    = patches * (Nu-1)*(Nv-1)
        nnode    = patches * Nu *Nv
        conn     = zeros((4,nelem),int)
        xyz      = zeros((3,nnode))
        elemtype = 4*ones(nelem) # All Quads
        
        counter = 0
        for n in xrange(patches):
            isurf = patch_list[n]
            
            u = linspace(self.surfs[isurf].range[0],\
                             self.surfs[isurf].range[1],Nu)
            v = linspace(self.surfs[isurf].range[0],\
                             self.surfs[isurf].range[1],Nv)
            [U,V] = meshgrid(u,v)

            temp = self.surfs[isurf].getValueM(U,V)
            for idim in xrange(self.surfs[isurf].nDim):
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

        # Next we need to figure out what is the actual UV coordinate 
        # on the given surface

        uv = zeros((nPts,2))
        
        for i in xrange(nPts):

            # Local Element
            local_elem = (nearest_elem[i]-1) - patchID[i]*(Nu-1)*(Nv-1)
            #print local_elem
            # Find out what its row/column index is

            #row = int(floor(local_elem / (Nu-1.0)))  # Integer Division
            row = local_elem / (Nu-1)
            col = mod(local_elem,(Nu-1)) 

            #print nearest_elem[i],local_elem,row,col

            u_local = uvw[0,i]
            v_local = uvw[1,i]

            if ( force_domain ):
                if u_local > 1.0:
                    u_local = 1.0
                elif u_local < 0.0:
                    u_local = 0.0
                # end

                if v_local > 1.0:
                    v_local = 1.0
                elif v_local < 0.0:
                    v_local = 0.0
                # end
            # end
            
            uv[i,0] =  u_local/(Nu-1)+ col/(Nu-1.0)
            uv[i,1] =  v_local/(Nv-1)+ row/(Nv-1.0)

        # end for

        # Now go back through and adjust the patchID to the element list
        for i in xrange(nPts):
            patchID[i] = patch_list[patchID[i]]
        # end

        # Release the tree - otherwise fortran will get upset
        csm_pre.release_adt()
        print 'Done Surface Attachment'

        return dist,patchID,uv

    def _initJ2( self, M, Nctl ):
        
        # We know the row filling factor: Its (no more) than ku*kv
        # control points for each control point. Since we don't
        # use more than k=4 we will set at 16
        
        if USE_PETSC:
            J2 = PETSc.Mat()
            if PETSC_MAJOR_VERSION == 1:
                J2.createAIJ([M*3,Nctl*3],nnz=16*3,comm=PETSc.COMM_SELF)
            elif PETSC_MAJOR_VERSION == 0:
                J2.createSeqAIJ([M*3,Nctl*3],nz=16*3)
            else:
                print 'Error: PETSC_MAJOR_VERSION = %d is not supported'%(PETSC_MAJOR_VERSION)
                sys.exit(1)
            # end if

        else:
            J2 = zeros((M*3,Nctl*3))
        # end if

        return J2
        

    def calcSurfaceDerivative(self,patchID,uv,indices=None,J2=None):
        '''Calculate the (fixed) surface derivative of a discrete set of ponits'''

        print 'Calculating Surface Derivative for %d Points...'%(len(patchID))
        timeA = time.time()

        # If no matrix is provided, use self.J2
        if not J2:
            J2 = self.J2
        # end

        if indices == None:
            indices = numpy.arange(len(patchID))#,numpy.int)
        # end

        if not J2:
            # Calculate the size Ncoef_free x Ndesign Variables            
            M = len(patchID)
            Nctl = self.Ncoef

            J2 = self._initJ2( M, Nctl )
        # end                     
                
        for i in xrange(len(patchID)):
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

                    index = 3*self.surfs[patchID[i]].globalCtlIndex[ \
                        u_list[ii],v_list[jj]]
                    J2[3*indices[i]  ,index  ] = x
                    J2[3*indices[i]+1,index+1] = x
                    J2[3*indices[i]+2,index+2] = x
                # end for
            # end for
        # end for 
        
        # Assemble the (Constant) J2
        if USE_PETSC:
            J2.assemblyBegin()
            J2.assemblyEnd()
        # end if

        self.J2 = J2 # Make sure we're dealing with the same matrix

        print 'Finished Surface Derivative in %5.3f seconds'%(time.time()-timeA)

        return

    def createTACSGeo(self):
        '''
        Create the spline classes for use within TACS
        '''

        try:
            from pyTACS import elements as elems
        except:
            print 'Could not import TACS. Cannot create TACS splines.'
            return
        # end

        if USE_PETSC == False:
            print 'Must have PETSc to create TACS splines.'
            return
        # end

        # Calculate the Number of global design variables
        N = 0
        for i in xrange(len(self.DV_listGlobal)): #Global Variables
            if self.DV_listGlobal[i].useit:
                N += self.DV_listGlobal[i].nVal
            # end if
        # end for

        gdvs = numpy.arange(N,dtype=numpy.intc)
      
        global_geo = elems.GlobalGeo( gdvs, self.petsc_coef, self.J1 )
      
        # For each segment, number the local variables
        localDVs = []
        for local in self.DV_listLocal:
            localDVs.append( numpy.arange(N,N+local.nVal,dtype=numpy.intc) )
            N += local.nVal
        # end

        # Create the list of local dvs for each surface patch
        surfDVs = []
        for i in xrange(self.nSurf):
            surfDVs.append(None)
        # end
        
        for i in xrange(len(self.DV_listLocal)):
            sid = self.DV_listLocal[i].surface_id
            if ( surfDVs[sid] == None ):
                surfDVs[sid] = localDVs[i]
            else:
                numpy.hstack( surfDVs[sid], localDVs[i] )
            # end
        # end

        # Go through and add local objects for each design variable
        def convert( s, ldvs ):
            if ldvs == None:
                ldvs = []
            # end
            
            return elems.SplineGeo( int(s.ku), int(s.kv),
                                    s.tu.astype('d'), s.tv.astype('d'),
                                    s.coef[:,:,0].astype('d'),
                                    s.coef[:,:,1].astype('d'),
                                    s.coef[:,:,2].astype('d'),
                                    global_geo, ldvs, s.globalCtlIndex.astype('intc') )
        # end

        tacs_surfs = []
        for i in xrange(self.nSurf):
            tacs_surfs.append( convert( self.surfs[i], surfDVs[i] ) )
        # end
     
        return global_geo, tacs_surfs
   

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

        # Note Conection nubmer is not necessary (only dummy character)
        self.uid = None
        return

    def write_info(self,i,handle):

        handle.write('%3d        |%3d     %3d    |%3d   | %3d  |  %3d       \
| %3d  | %3d           | %3d  | %3d     %3d      |\n'\
              %(i,self.f1,self.e1,self.type,self.dof,self.cont,self.dir,\
                    self.dg,self.Nctl,self.f2,self.e2))
        
        return
    def __cmp__(self, other):
        return cmp(self.uid,other.uid)


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

        self.links_s = []
        self.links_x = []
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

    def getRotMatrixGlobalToLocal(self,s):
        
        '''Return the rotation matrix to convert vector from global to
        local frames'''
        return     dot(rotyM(self.rotys(s)),dot(rotxM(self.rotxs(s)),\
                                                    rotzM(self.rotzs(s))))
    
    def getRotMatrixLocalToGlobal(self,s):
        
        '''Return the rotation matrix to convert vector from global to
        local frames'''
        return inv(dot(rotyM(self.rotys(s)),dot(rotxM(self.rotxs(s)),\
                                                    rotzM(self.rotzs(s)))))
    
class geoDVGlobal(object):
     
    def __init__(self,dv_name,value,lower,upper,function,useit):
        
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
        self.useit=useit
        return

    def __call__(self,ref_axis):

        '''When the object is called, actually apply the function'''
        # Run the user-supplied function
        return self.function(self.value,ref_axis)
        

class geoDVNormal(object):
     
    def __init__(self,dv_name,lower,upper,surface_id,coef_list,global_coef):
        
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

        self.nVal = len(coef_list)
        self.value = zeros(self.nVal,'D')
        self.name = dv_name
        self.lower = lower
        self.upper = upper
        self.surface_id = surface_id
        self.coef_list = coef_list
        
        # We also need to know what local surface i,j index is for
        # each point in the coef_list since we need to know the
        # position on the surface to get the normal. That's why we
        # passed in the global_coef list so we can figure it out
        
        self.local_coef_index = zeros((self.nVal,2),'intc')
        
        for icoef in xrange(self.nVal):
            current_point = global_coef[coef_list[icoef]]
            # Since the local DV only have driving control points, the
            # i,j index coorsponding to the first entryin the
            # global_coef list is the one we want
            self.local_coef_index[icoef,:] = global_coef[coef_list[icoef]][0][1:3]
        # end for
        return

    def __call__(self,surface,coef):

        '''When the object is called, apply the design variable values to the
        surface'''

        coef = pySpline.pyspline_cs.updatesurfacepoints(\
            coef,self.local_coef_index,self.coef_list,self.value,\
                surface.globalCtlIndex,surface.tu,surface.tv,surface.ku,surface.kv)

        return coef

    def getNormals(self,surf,coef):
        normals = pySpline.pyspline_real.getctlnormals(\
            coef,self.local_coef_index,self.coef_list,\
                surf.globalCtlIndex,surf.tu,surf.tv,surf.ku,surf.kv)
        return normals

    def removeCoef(self,rm_list):
        '''Remove coefficient from this dv if its in rm_list'''
        for i in xrange(len(rm_list)):
            if rm_list[i] in self.coef_list:
                index = self.coef_list.index(rm_list[i])
                del self.coef_list[index]
                delete(self.local_coef_index,index)
                delete(self.value,index)
                self.nVal -= 1
            # end if
        # end for

        return



class geoDVLocal(object):
     
    def __init__(self,dv_name,lower,upper,surface_id,coef_list,global_coef):
        
        '''Create a set of gemoetric design variables whcih change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.

        Input:
        
        dv_name: Design variable name. Should be unique. Can be used
        to set pyOpt variables directly

        lower: Lower bound for the variable. Again for setting in
        pyOpt

        upper: Upper bound for the variable.

        surface_id: Surface this set of design variables belongs to

        coef_list: The indicies on the surface used for these dvs

        global_coef: The pyGeo global_design variable linkinng list to
        determine if a design variable is free of driven
        
        Note: Value is NOT specified, value will ALWAYS be initialized to 0

        '''

        self.nVal = len(coef_list)
        self.value = zeros((self.nVal,3),'D')
        self.name = dv_name
        self.lower = lower
        self.upper = upper
        self.surface_id = surface_id
        self.coef_list = coef_list
        
        # We also need to know what local surface i,j index is for
        # each point in the coef_list since we need to know the
        # position on the surface to get the normal. That's why we
        # passed in the global_coef list so we can figure it out
        
        self.local_coef_index = zeros((self.nVal,2),'intc')
        
        for icoef in xrange(self.nVal):
            self.local_coef_index[icoef,:] = global_coef[coef_list[icoef]][0][1:3]
        # end for
        return

    def __call__(self,surface,coef):

        '''When the object is called, apply the design variable values to the
        surface'''
        pass
      
        return coef

    def removeCoef(self,rm_list):
        '''Remove coefficient from this dv if its in rm_list'''
        for i in xrange(len(rm_list)):
            if rm_list[i] in self.coef_list:
                index = self.coef_list.index(rm_list[i])
                del self.coef_list[index]
                delete(self.local_coef_index,index)
                delete(self.value,index)
                self.nVal -= 1
            # end if
        # end for
   
class point_select(object):

    def __init__(self,type,*args,**kwargs):

        '''Initialize a control point selection class. There are several ways
        to initialize this class depending on the 'type' qualifier:

        Inputs:
        
        type: string which inidicates the initialization type:
        
        'x': Define two corners (pt1=,pt2=) on a plane parallel to the
        x=0 plane

        'y': Define two corners (pt1=,pt2=) on a plane parallel to the
        y=0 plane

        'z': Define two corners (pt1=,pt2=) on a plane parallel to the
        z=0 plane

        'quad': Define FOUR corners (pt1=,pt2=,pt3=,pt4=) in a
        COUNTER-CLOCKWISE orientation 

        'slice': Define a grided region using two slice parameters:
        slice_u= and slice_v are used as inputs

        'list': Simply use a list of control point indidicies to
        use. Use coef = [[i1,j1],[i2,j2],[i3,j3]] format

        '''
        
        if type == 'x' or type == 'y' or type == 'z':
            assert 'pt1' in kwargs and 'pt2' in kwargs,'Error:, two points \
must be specified with initialization type x,y, or z. Points are specified \
with kwargs pt1=[x1,y1,z1],pt2=[x2,y2,z2]'

        elif type == 'quad':
            assert 'pt1' in kwargs and 'pt2' in kwargs and 'pt3' in kwargs \
                and 'pt4' in kwargs,'Error:, four points \
must be specified with initialization type quad. Points are specified \
with kwargs pt1=[x1,y1,z1],pt2=[x2,y2,z2],pt3=[x3,y3,z3],pt4=[x4,y4,z4]'
            
        elif type == 'slice':
            assert 'slice_u'  in kwargs and 'slice_v' in kwargs,'Error: two \
python slice objects must be specified with slice_u=slice1, slice_v=slice_2 \
for slice type initialization'

        elif type == 'list':
            assert 'coef' in kwargs,'Error: a coefficient list must be \
speficied in the following format: coef = [[i1,j1],[i2,j2],[i3,j3]]'
        else:
            print 'Error: type must be one of: x,y,z,quad,slice or list'
            sys.exit(1)
        # end if

        if type == 'x' or type == 'y' or type =='z' or type == 'quad':
            corners = zeros([4,3])
            if type == 'x':
                corners[0] = kwargs['pt1']

                corners[1][1] = kwargs['pt2'][1]
                corners[1][2] = kwargs['pt1'][2]

                corners[2][1] = kwargs['pt1'][1]
                corners[2][2] = kwargs['pt2'][2]

                corners[3] = kwargs['pt2']

                corners[:,0] = 0.5*(kwargs['pt1'][0] + kwargs['pt2'][0])

            elif type == 'y':
                corners[0] = kwargs['pt1']

                corners[1][0] = kwargs['pt2'][0]
                corners[1][2] = kwargs['pt1'][2]

                corners[2][0] = kwargs['pt1'][0]
                corners[2][2] = kwargs['pt2'][2]

                corners[3] = kwargs['pt2']

                corners[:,1] = 0.5*(kwargs['pt1'][1] + kwargs['pt2'][1])

            elif type == 'z':
                corners[0] = kwargs['pt1']

                corners[1][0] = kwargs['pt2'][0]
                corners[1][1] = kwargs['pt1'][1]

                corners[2][0] = kwargs['pt1'][0]
                corners[2][1] = kwargs['pt2'][1]

                corners[3] = kwargs['pt2']

                corners[:,2] = 0.5*(kwargs['pt1'][2] + kwargs['pt2'][2])

            elif type == 'quad':
                corners[0] = kwargs['pt1']
                corners[1] = kwargs['pt2']
                corners[2] = kwargs['pt4'] # Note the switch here from CC orientation
                corners[3] = kwargs['pt3']
            # end if

            X = reshape(corners,[2,2,3])

            self.box=pySpline.surf_spline(task='lms',ku=2,kv=2,\
                                              Nctlu=2,Nctlv=2,X=X)

        elif type == 'slice':
            self.slice_u = kwargs['slice_u']
            self.slice_v = kwargs['slice_v']
        elif type == 'list':
            self.coef_list = kwargs['coef']
        # end if

        self.type = type

        return


    def getControlPoints(self,surface,coef_list):

        '''Take is a pySpline surface, and a (possibly non-empty) coef_list
        and add to the coef_list the global index of the control point
        on the surface that can be projected onto the box'''
        
        if self.type=='x'or self.type=='y' or self.type=='z' or self.type=='quad':

            for i in xrange(surface.Nctlu):
                for j in xrange(surface.Nctlv):
                    u0,v0,D,converged = self.box.projectPoint(surface.coef[i,j])
                    if u0 > 0 and u0 < 1 and v0 > 0 and v0 < 1: # Its Inside
                        coef_list.append(surface.globalCtlIndex[i,j])
                    #end if
                # end for
            # end for
        elif self.type == 'slice':

            for i in self.slice_u:
                for j in self.slice_v:
                    coef_list.append(surface.globalCtlIndex[i,j])
                # end for
            # end for
        elif self.type == 'list':
            for i in xrange(len(self.coef_list)):
                coef_list.append(surface.globalCtlIndex[self.coef_list[i][0],\
                                                            self.coef_list[i][1]])
            # end for
        # end if

        return coef_list

#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'

