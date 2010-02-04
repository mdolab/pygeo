'''
pyGeo

pyGeo is a (fairly) complete geometry surfacing engine. It performs
multiple functions including producing surfaces from cross sections,
fitting groups of surfaces with continutity constraints and has
built-in design variable handling. The actual b-spline surfaces are of
the pySpline surface type. See the individual functions for
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
    real, imag, dstack, floor, size, reshape, arange,alltrue,cross,average

from numpy.linalg import lstsq,inv,norm

# =============================================================================
# Extension modules
# =============================================================================

from mdo_import_helper import *
exec(import_modules('geo_utils','pySpline','csm_pre','mpi4py'))

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
        
        # First thing to do is to check if we want totally silent
        # operation i.e. no print statments
        if 'no_print' in kwargs:
            self.NO_PRINT = kwargs['no_print']
        else:
            self.NO_PRINT = False
        # end if
        self.init_type = init_type
        mpiPrint(' ',self.NO_PRINT)
        mpiPrint('------------------------------------------------',self.NO_PRINT)
        mpiPrint('pyGeo Initialization Type is: %s'%(init_type),self.NO_PRINT)
        mpiPrint('------------------------------------------------',self.NO_PRINT)

        #------------------- pyGeo Class Atributes -----------------

        self.ref_axis       = [] # Reference Axis list
        self.ref_axis_con   = [] # Reference Axis connection list
        self.DV_listGlobal  = [] # Global Design Variable List
        self.DV_listNormal  = [] # Normal Design Variable List
        self.DV_listLocal   = [] # Local Design Variable List
        self.DV_namesGlobal = {} # Names of Global Design Variables
        self.DV_namesNormal = {} # Names of Normal Design Variables
        self.DV_namesLocal  = {} # Names of Local Design Variables
        self.petsc_coef = None   # Global vector of PETSc coefficients
        self.dCoefdx  = None     # Derivative of control points wrt
                                 # design variables
        self.dPtdCoef = None     # Derivate of surface points wrt
                                 # control points
        self.dPtdx    = None     # Multiplication of above matricies,
                                 # derivative of surface points wrt
                                 # design variables
        self.topo = None         # The topology of the surfaces
        self.surfs = []          # The list of surface (pySpline surf)
                                 # objects
        self.nSurf = None        # The total number of surfaces
        self.coef  = None        # The global (reduced) set of control
                                 # points
        self.sym   = None        # Symmetry type. 'xy','yz','xz'
        self.sym_normal = None   # Normal consistent with symmetry type
        self.l_surfs = []        # Logical Surfaces: List of list of
                                 # surfaces that can be thought of as
                                 # connected.

        # --------------------------------------------------------------

        if init_type == 'plot3d':
            self._readPlot3D(*args,**kwargs)

        elif init_type == 'iges':
            assert 'file_name' in kwargs,'file_name must be specified as \
file_name=\'filename\' for iges init_type'
            self._readIges(kwargs['file_name'])

        elif init_type == 'lifting_surface':
            self._init_lifting_surface(*args,**kwargs)

        elif init_type == 'acdt_geo':
            self._init_acdt_geo(*args,**kwargs)
        elif init_type == 'create':
            # Don't do anything 
            pass
        else:
            mpiPrint('Unknown init type. Valid Init types are \'plot3d\', \
\'iges\',\'lifting_surface\' and \'acdt_geo\'')
            sys.exit(0)

        return

# ----------------------------------------------------------------------------
#               Initialization Type Functions
# ----------------------------------------------------------------------------

    def _readPlot3D(self,*args,**kwargs):

        '''Load a plot3D file and create the splines to go with each patch'''
        assert 'file_name' in kwargs,'file_name must be specified for plot3d'
        file_name = kwargs['file_name']
        mpiPrint('Loading plot3D file: %s ...'%(file_name),self.NO_PRINT)

        f = open(file_name,'r')
        nSurf = int(f.readline())         # First load the number of patches

        mpiPrint('nSurf = %d'%(nSurf),self.NO_PRINT)

        patchSizes = readNValues(f,nSurf*3,'int')
        patchSizes = patchSizes.reshape([nSurf,3])

        assert patchSizes[:,2].all() == 1, \
            'Error: Plot 3d does not contain only surface patches.\
 The third index (k) MUST be 1.'

        # Total points
        nPts = 0
        for i in xrange(nSurf):
            nPts += patchSizes[i,0]*patchSizes[i,1]

            
        mpiPrint('Number of Surface Points = %d'%(nPts),self.NO_PRINT)

        dataTemp = readNValues(f,3*nPts,'float')
        
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
            surfs.append(pySpline.surface('lms',X=patches[isurf],\
                                              ku=4,kv=4,Nctlu=8,Nctlv=8,\
                                              no_print=self.NO_PRINT))
            
        self.surfs = surfs
        self.nSurf = nSurf
        return

    def _readIges(self,file_name,*args,**kwargs):

        '''Load a Iges file and create the splines to go with each patch'''
        mpiPrint('File Name is: %s'%(file_name),self.NO_PRINT)
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

        mpiPrint('Found %d surfaces in Iges File.'%(self.nSurf),self.NO_PRINT)

        self.surfs = [];

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

            self.surfs.append(pySpline.surface(\
                    'create',ku=ku,kv=kv,tu=tu,tv=tv,coef=coef,\
                        range=range,no_print=self.NO_PRINT))
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


        if 'breaks' in kwargs:
            breaks = kwargs['breaks']
            nBreaks = len(breaks)
        else:
            nBreaks = 0
        # end if
            
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
        # end if

        if 'section_spacing' in kwargs:
            section_spacing = kwargs['section_spacing']
        else:
            # Generate the section spacing -> linear default
            section_spacing = []
            for i in xrange(len(nsections)):
                section_spacing.append(linspace(0,1,nsections[i]))
            # end for
        # end if

        if 'cont' in kwargs:
            cont = kwargs['cont']
        else:
            cont = [0]*nBreaks # Default is c0 contintity
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
                    # Twist Rotation (z-Rotation)
                    X[isurf,j,i,:] = rotzV(X[isurf,j,i,:],rot[i,2]*pi/180)
                    # Dihediral Rotation (x-Rotation)
                    X[isurf,j,i,:] = rotxV(X[isurf,j,i,:],rot[i,0]*pi/180)
                    # Sweep Rotation (y-Rotation)
                    X[isurf,j,i,:] = rotyV(X[isurf,j,i,:],rot[i,1]*pi/180)
                # end ofr
            # end for

            # Finally translate according to  positions specified
            X[:,:,i,:] += Xsec[i,:]
        # end for

        # ---------------------------------------------------------------------
        # Now, we interpolate them IF we have breaks 
        # ---------------------------------------------------------------------

        self.surfs = []

        if nBreaks>0:
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
                chord_line = pySpline.curve('interpolate',X=Xchord_line,k=2)

                for j in xrange(N): # This is for the Data points

                    if i > 0 and cont[i-1] == 2: # Do a continuity join (from both sides)
                        #print 'cont join FIX ME'
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
                        temp_spline = pySpline.curve(\
                            'interpolate',X=X[0,j,start:end,:],k=4,\
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

                        temp_spline = pySpline.curve(\
                            'interpolate',X=X[1,j,start:end,:],k=4,\
                                dx1=dx1,dx2=dx2)
                        Xnew[1,j,start2:end2,:] = \
                            temp_spline.getValueV(section_spacing[i])

                    else:
                            
                        temp_spline = pySpline.curve(\
                            'interpolate',X=X[0,j,start:end,:],k=2)
                        Xnew[0,j,start2:end2,:] = \
                            temp_spline.getValueV(section_spacing[i])

                        temp_spline = pySpline.curve(\
                            'interpolate',X=X[1,j,start:end,:],k=2)
                        Xnew[1,j,start2:end2,:] = \
                            temp_spline.getValueV(section_spacing[i])
                    # end if
                # end for

                # Now we can generate and append the surfaces

                self.surfs.append(pySpline.surface(\
                        fit_type,ku=4,kv=4,X=Xnew[0,:,start2:end2,:].copy(),\
                            Nctlv=nsections[i],no_print=self.NO_PRINT,*args,**kwargs))
                self.surfs.append(pySpline.surface(\
                        fit_type,ku=4,kv=4,X=Xnew[1,:,start2:end2,:].copy(),\
                            Nctlv=nsections[i],no_print=self.NO_PRINT,*args,**kwargs))

                start = end-1
                start2 = end2-1
            # end for
        
            # Second loop for 1-sided continuity constraints
            start   = breaks[0]
            start2  = nsections[0]
            for i in xrange(1,nBreaks+1): # The first (mirror plane can't happen)
               
                # We have to interpolate the sectional data 
                if i == nBreaks:
                    end = naf
                else:
                    end  = breaks[i]+1
                #end if

                end2 = start2 + nsections[i]
        
                if cont[i-1] == -1 and cont[i] == 1: # We have what we're looking for
                    for j in xrange(N): # This is for the Data points
                        dx1 = (X[0,j,start] - X[0,j,start-1])
                        dx1 /= sqrt(dx1[0]*dx1[0] + dx1[1]*dx1[1] + dx1[2]*dx1[2])

                        dx2 = (X[0,j,end] - X[0,j,end-1])
                        dx2 /= sqrt(dx2[0]*dx2[0] + dx2[1]*dx2[1] + dx2[2]*dx2[2])

                        dx1/= 50
                        dx2/=50

                        # Now generate the line and extract the points we want
                        temp_spline = pySpline.curve(\
                            'interpolate',X=X[0,j,start:end,:],k=4,\
                                dx1=dx1,dx2=dx2)
                        Xnew[0,j,start2:end2,:] =  temp_spline.getValueV(section_spacing[i])

                        dx1 = (X[1,j,start] - X[1,j,start-1])
                        dx1 /= sqrt(dx1[0]*dx1[0] + dx1[1]*dx1[1] + dx1[2]*dx1[2])

                        dx2 = (X[1,j,end] - X[1,j,end-1])
                        dx2 /= sqrt(dx2[0]*dx2[0] + dx2[1]*dx2[1] + dx2[2]*dx2[2])
                        dx1/= 50
                        dx2/=50

                        # Now generate the line and extract the points we want
                        temp_spline = pySpline.curve(\
                            'interpolate',X=X[1,j,start:end,:],k=4,\
                                dx1=dx1,dx2=dx2)
                        Xnew[1,j,start2:end2,:] =  temp_spline.getValueV(section_spacing[i])
                    # end if
        
                    # Now we need to REPLACE
                        
                    self.surfs[2*i] = pySpline.surface(\
                        fit_type,ku=4,kv=4,X=Xnew[0,:,start2:end2,:],\
                            Nctlv=nsections[i],no_print=self.NO_PRINT,*args,**kwargs)
                    self.surfs[2*i+1] = pySpline.surface(\
                        fit_type,ku=4,kv=4,X=Xnew[1,:,start2:end2,:],\
                            Nctlv=nsections[i],no_print=self.NO_PRINT,*args,**kwargs)
                # end if
                start = end-1
                start2 = end2-1
            # end for
        
        else:  #No breaks
            tot_sec = sum(nsections)
            Xnew    = zeros([2,N,tot_sec,3])
            Xsecnew = zeros((tot_sec,3))
            rotnew  = zeros((tot_sec,3))

            for j in xrange(N):
                temp_spline = pySpline.curve('interpolate',X=X[0,j,:,:],k=2)
                Xnew[0,j,:,:] = temp_spline.getValueV(section_spacing[0])
                temp_spline = pySpline.curve('interpolate',X=X[1,j,:,:],k=2)
                Xnew[1,j,:,:] = temp_spline.getValueV(section_spacing[0])
            # end for
            Nctlv = nsections
            self.surfs.append(pySpline.surface(fit_type,ku=4,kv=4,X=Xnew[0],Nctlv=nsections[0],
                                                   no_print=self.NO_PRINT,*args,**kwargs))
            self.surfs.append(pySpline.surface(fit_type,ku=4,kv=4,X=Xnew[1],Nctlv=nsections[0],
                                                   no_print=self.NO_PRINT,*args,**kwargs))
        # end if

        if 'end_type' in kwargs: # The user has specified automatic tip completition
            end_type = kwargs['end_type']

            assert end_type in ['rounded','flat'],'Error: end_type must be one of \'rounded\' or \
\'flat\'. Rounded will result in a non-degenerate geometry while flat type will result in a single \
double degenerate patch at the tip'


            if end_type == 'flat':
            
                spacing = 10
                v = linspace(0,1,spacing)
                X2 = zeros((N,spacing,3))
                for j in xrange(1,N-1):
                    # Create a linear spline 
                    x1 = X[0,j,-1]
                    x2 = X[1,N-j-1,-1]

                    temp = pySpline.curve('interpolate',\
                                                      k=2,X=array([x1,x2]))
                    X2[j,:,:] = temp.getValueV(v)

                # end for
                X2[0,:] = X[0,0,-1]
                X2[-1,:] = X[1,0,-1]
                
                self.surfs.append(pySpline.surface('lms',ku=4,kv=4,\
                                                           X=X2,Nctlv=spacing,\
                                                           *args,**kwargs))
            elif end_type == 'rounded':
                if 'end_scale' in kwargs:
                    end_scale = kwargs['end_scale']
                else:
                    end_scale = 1
                # This code uses *some* huristic measures but generally works fairly well
                # Generate a "pinch" airfoil from the last one given

                # First determine the maximum thickness of the airfoil, since this will 
                # determine how far we need to offset it
                dist_max = 0

                for j in xrange(N):
                    dist = e_dist(X[0,j,-1],X[1,N-j-1,-1])
                    if dist > dist_max:
                        dist_max = dist
                    # end if
                # end for

                # Determine the data for the pinch section
                # Front
                n =  (X[0,0,-1] - X[0,0,-2])
                n_front = n/sqrt(dot(n,n)) #Normalize

                # Back
                n =  (X[0,-1,-1] - X[0,-1,-2])
                n_back = n/sqrt(dot(n,n))
            
                # Create a chord line representation of the end section
                Xchord_line = array([X[0,0,-1],X[0,-1,-1]])
                end_chord_line = pySpline.curve('interpolate',X=Xchord_line,k=2)

                # Create a chord line representation of the tip line
                Xchord_line = array([X[0,0,-1] + dist_max*n_front*end_scale,
                                     X[0,-1,-1] + dist_max*n_back*end_scale])
                
                chord_line = pySpline.curve('interpolate',X=Xchord_line,k=2)
                Xchord_line =chord_line.getValueV([0.10,1.0])
                chord_line = pySpline.curve('interpolate',X=Xchord_line,k=2)
                tip_line = chord_line.getValueV(linspace(0,1,N))
                # Now Get the front, back top and bottom guide curves
                #-------- Front
                
                X_input = array([X[0,0,-1],tip_line[0]])
                dv = (X[0,0,-1] - X[0,0,-2])
                dv /= sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])
                V = tip_line[0]-X[0,0,-1]
                D = end_chord_line.getValue(0.1)-chord_line.getValue(0)
                dx1 = dot(dv,V) * dv * end_scale
                dx2 = D+V
       
                front_spline = pySpline.curve('interpolate',X=X_input,
                                                             k=4,dx1=dx1,dx2=dx2)

                #-------- Back
                X_input = array([X[0,-1,-1],tip_line[-1]])
                dv = (X[0,-1,-1] - X[0,-1,-2])
                dv /= sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])
                V = tip_line[-1]-X[0,-1,-1]
                D = end_chord_line.getValue(1)-chord_line.getValue(1)
                dx1 = dot(dv,V) * dv * end_scale
                dx2 = D+V
                end_spline = pySpline.curve('interpolate',X=X_input,
                                                    k=4,dx1=dx1,dx2=dx2)

                #-------- Top
                X_input = array([X[0,N/2,-1],tip_line[N/2]])
                dv = (X[0,N/2,-1] - X[0,N/2,-2])
                dv /= sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])
                V = tip_line[N/2]-X[0,N/2,-1]
                D = end_chord_line.getValue(.5)-chord_line.getValue(.5)
                dx1 = dot(dv,V) * dv * end_scale
                dx2 = D+V
                top_spline = pySpline.curve('interpolate',X=X_input,
                                                             k=4,dx1=dx1,dx2=dx2)

                #-------- Bottom
                X_input = array([X[1,N/2,-1],tip_line[N/2]])
                dv = (X[1,N/2,-1] - X[1,N/2,-2])
                dv /= sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])
                V = tip_line[N/2]-X[1,N/2,-1]
                D = end_chord_line.getValue(.5)-chord_line.getValue(.5)
                dx1 = dot(dv,V) * dv * end_scale
                dx2 = D+V
                bottom_spline = pySpline.curve('interpolate',X=X_input,
                                                             k=4,dx1=dx1,dx2=dx2)

                Ndata = 15

                chords = []
                thicknesses = []
                chord0 = e_dist(front_spline(0),end_spline(0))
                thickness0 = e_dist(top_spline(0),bottom_spline(0))
                for i in xrange(Ndata):
                    chords.append(e_dist(front_spline(i/(Ndata-1.0)),end_spline(i/(Ndata-1.0)))/chord0)
                    thicknesses.append(e_dist(top_spline(i/(Ndata-1.0)),bottom_spline(i/(Ndata-1.0)))/thickness0)
                # end for

                # Re-read the last airfoil section data
                X_u,Y_u,X_l,Y_l = read_af(xsections[-1],file_type,N)
                
                Xnew = zeros((2,N,Ndata,3))
        
                for i in xrange(1,Ndata):

                    if i == Ndata-1:
                        Xnew[0,:,i,0] = (X_u)*scale[-1]*chords[i]
                        Xnew[0,:,i,1] = (Y_u*thicknesses[i])*scale[-1]*chords[i]
                        Xnew[0,:,i,2] = 0
            
                        Xnew[1,:,i,0] = (X_u[::-1])*scale[-1]*chords[i]
                        Xnew[1,:,i,1] = (Y_l*thicknesses[i])*scale[-1]*chords[i]
                        Xnew[1,:,i,2] = 0

                    else:

                        Xnew[0,:,i,0] = (X_u)*scale[-1]*chords[i]
                        Xnew[0,:,i,1] = (Y_u*thicknesses[i])*scale[-1]*chords[i]
                        Xnew[0,:,i,2] = 0
                        
                        Xnew[1,:,i,0] = (X_l)*scale[-1]*chords[i]
                        Xnew[1,:,i,1] = (Y_l*thicknesses[i])*scale[-1]*chords[i]
                        Xnew[1,:,i,2] = 0

                    for j in xrange(N):
                        for isurf in xrange(2):
                            # Twist Rotation (z-Rotation)
                            Xnew[isurf,j,i,:] = rotzV(Xnew[isurf,j,i,:],rot[-1,2]*pi/180)
                            # Dihediral Rotation (x-Rotation)
                            Xnew[isurf,j,i,:] = rotxV(Xnew[isurf,j,i,:],rot[-1,0]*pi/180)
                            # Sweep Rotation (y-Rotation)
                            Xnew[isurf,j,i,:] = rotyV(Xnew[isurf,j,i,:],rot[-1,1]*pi/180)
                        # end for
                    # end for
                    Xnew[:,:,i,:] += front_spline(i/(Ndata-1.0))
                    
                # end for
                Xnew[:,:,0,:] = X[:,:,-1,:]
                # Xnew[:,:,-1,:] = tip_line
                self.surfs.append(pySpline.surface('lms',ku=4,kv=4,X=Xnew[0],
                                                       Nctlv=4, *args,**kwargs))
                
                self.surfs.append(pySpline.surface('lms',ku=4,kv=4,X=Xnew[1],
                                                       Nctlv=4, *args,**kwargs))

        self.nSurf = len(self.surfs) # And last but not least
        return

    def _init_acdt_geo(self,ac,LiftingSurface,BodySurface,*args,**kwargs):
        '''Create a list of pyGeo objects coorsponding to the pyACDT geometry specified in ac'''

        dtor = pi/180
        self.nSurf = 0
        geo_objects = []

        for i in xrange(len(ac)):
            print 'Processing Component: %s'%(ac[i].Name)
            # Determine Type -> Lifting Surface or Body Surface
            if isinstance(ac[i],BodySurface):
                nSubComp = len(ac[i])
                for j in xrange(nSubComp):
                    [m,n] = ac[i][j].Surface_x.shape
                    print 'm,n:',m,n
                    N = (n+1)/2
                    X = zeros((m,N,3))
                    print N
                    X[:,:,0] = ac[i][j].Surface_x[:,N-1:]
                    X[:,:,1] = ac[i][j].Surface_y[:,N-1:]
                    X[:,:,2] = ac[i][j].Surface_z[:,N-1:]

                    self.surfs.append(pySpline.surface('interpolate',ku=4,kv=4,X=X))
                    self.nSurf += 1
                # end for (subcomp)

            elif isinstance(ac[i],LiftingSurface):
                nSubComp = len(ac[i])

                for j in xrange(nSubComp):
                    [m,n] = ac[i][j].Surface_x.shape
                    N=  (n-1)/2
                    X = zeros((2,m,N+1,3))
                    X[0,:,:,0] = ac[i][j].Surface_x[:,0:N+1]
                    X[0,:,:,1] = ac[i][j].Surface_y[:,0:N+1]
                    X[0,:,:,2] = ac[i][j].Surface_z[:,0:N+1]

                    X[1,:,:,0] = ac[i][j].Surface_x[:,N:]
                    X[1,:,:,1] = ac[i][j].Surface_y[:,N:]
                    X[1,:,:,2] = ac[i][j].Surface_z[:,N:]

                    self.surfs.append(pySpline.surface('interpolate',ku=4,kv=4,X=X[0]))
                    self.surfs.append(pySpline.surface('interpolate',ku=4,kv=4,X=X[1]))
                    self.nSurf += 2

                # end for (sub Comp)
            # end if (lifting/body type)
        # end if (Comp Loop)

    def setSymmetry(self,sym_type):
        '''Set the symmetry flag and symmetry normal for this geometry object'''
        if sym_type == 'xy':
            self.sym = 'xy'
            self.sym_normal = [0,0,1]
        elif sym_type == 'yz':
            self.sym = 'yz'
            self.sym_normal = [1,0,0]
        elif sym_type == 'xz':
            self.sym = 'xz'
            self.sym_normal = [0,1,0]
        else:
            print 'Error: Symmetry must be specified as \'xy\', \'yz\' or \'xz\''
            sys.exit(1)
        # end if

        return
	
# ----------------------------------------------------------------------
#                     Topology Information Functions
# ----------------------------------------------------------------------    

    def _calcEdgeConnectivity(self,node_tol,edge_tol):

        '''This function attempts to automatically determine the connectivity
        between the pataches'''
        
        # Calculate the 4 corners and 4 midpoints for each surface

        coords = zeros((self.nSurf,8,3))
        for isurf in xrange(self.nSurf):
            beg,mid,end = self.surfs[isurf].getOrigValuesEdge(0)
            coords[isurf][0] = beg
            coords[isurf][1] = end
            coords[isurf][4] = mid
            beg,mid,end = self.surfs[isurf].getOrigValuesEdge(1)
            coords[isurf][2] = beg
            coords[isurf][3] = end
            coords[isurf][5] = mid
            beg,mid,end = self.surfs[isurf].getOrigValuesEdge(2)
            coords[isurf][6] = mid
            beg,mid,end = self.surfs[isurf].getOrigValuesEdge(3)
            coords[isurf][7] = mid
        # end for

        self.topo = Topology(coords=coords,node_tol=node_tol,edge_tol=edge_tol)
        return
   
    def _setEdgeConnectivity(self):
        '''Internal function to set the global/local numbering'''
     
        # Call the calcGlobalNumbering function
        sizes = []
        for isurf in xrange(self.nSurf):
            sizes.append([self.surfs[isurf].Nctlu,self.surfs[isurf].Nctlv])
        # end for

        self.topo.calcGlobalNumbering(sizes)
        self.Ncoef = self.topo.counter
        
        self.coef = []
        # Now Fill up the self.coef list:
        for ii in xrange(len(self.topo.g_index)):
            isurf = self.topo.g_index[ii][0][0]
            i = self.topo.g_index[ii][0][1]
            j = self.topo.g_index[ii][0][2]
            self.coef.append( self.surfs[isurf].coef[i,j])
        # end for
            
#         # Finally turn self.coef into a complex array
#         self.coef = array(self.coef,'D')

    def printEdgeConnectivity(self,node_link=None,edge_list=None,edge_link=None,edge_dir=None):
        self.topo.printEdgeConnectivity()
        return

    def doEdgeConnectivity(self,file_name,node_tol=1e-4,edge_tol=1e-4):
        '''This is the ONLY function that is available to the user for
        edge connectivity functionality pyGeo automatically creates
        the con file from scratch if the specified con file doesn't
        exists, and if it does, read it.'''

        if os.path.isfile(file_name):
            mpiPrint('Reading Edge Connectivity File: %s'%(file_name),self.NO_PRINT)
            self.topo = Topology(file=file_name)
            self._setEdgeConnectivity()
            if not self.init_type == 'iges':
                self.propagateKnotVectors()
            # end if
        else:
            self._calcEdgeConnectivity(node_tol,edge_tol)
            self.topo.writeEdgeConnectivity(file_name)
            self._setEdgeConnectivity()
            if not self.init_type == 'iges':
                self.propagateKnotVectors()
            # end if
        # end if

    def _calcnDG(self):
        '''Calculate the number of design groups'''
        nDG = -1
        for i in xrange(len(self.edge_list)):
            if self.edge_list[i].dg > nDG:
                nDG = self.edge_list[i].dg
            # end if
        # end for
        nDG += 1
        self.nDG = nDG
        
    def propagateKnotVectors(self):

        # First get the number of design groups
        nDG = -1
        ncoef = []
        for i in xrange(self.topo.nEdge):
            if self.topo.edges[i].dg > nDG:
                nDG = self.topo.edges[i].dg
                ncoef.append(self.topo.edges[i].Nctl)
            # end if
        # end for

        nDG += 1
	for isurf in xrange(self.nSurf):
            dg_u = self.topo.edges[self.topo.edge_link[isurf][0]].dg
            dg_v = self.topo.edges[self.topo.edge_link[isurf][2]].dg
            self.surfs[isurf].Nctlu = ncoef[dg_u]
            self.surfs[isurf].Nctlv = ncoef[dg_v]
            if self.surfs[isurf].ku < self.surfs[isurf].Nctlu:
                if self.surfs[isurf].Nctlu > 4:
	            self.surfs[isurf].ku = 4
                else:
                    self.surfs[isurf].ku = self.surfs[isurf].Nctlu
		# endif
            # end if
            if self.surfs[isurf].kv < self.surfs[isurf].Nctlv:
		if self.surfs[isurf].Nctlv > 4:
                    self.surfs[isurf].kv = 4
                else:
                    self.surfs[isurf].kv = self.surfs[isurf].Nctlv
                # end if
            # end if

            self.surfs[isurf]._calcKnots()
            # Now loop over the number of design groups, accumulate all
            # the knot vectors that coorspond to this dg, then merge them all
        # end for

        for idg in xrange(nDG):
            sym = False
            knot_vectors = []
            for isurf in xrange(self.nSurf):
                # Check edge 0 and edge 2
                if self.topo.edges[self.topo.edge_link[isurf][0]].dg == idg:
                    if self.topo.edge_dir[isurf][0] == -1 or self.topo.edge_dir[isurf][1] == -1:
                        sym = True
                    # end if
                    knot_vectors.append(self.surfs[isurf].tu)
                # end if
                if self.topo.edges[self.topo.edge_link[isurf][2]].dg == idg:
                    if self.topo.edge_dir[isurf][2] == -1 or self.topo.edge_dir[isurf][3] == -1:
                        sym = True
                    # end if
                    knot_vectors.append(self.surfs[isurf].tv)
                # end if
            # end for

            # Now blend all the knot vectors
            new_knot_vec = blendKnotVectors(knot_vectors,sym)

            # And reset them all
            for isurf in xrange(self.nSurf):
                # Check edge 0 and edge 2
                if self.topo.edges[self.topo.edge_link[isurf][0]].dg == idg:
                    self.surfs[isurf].tu = new_knot_vec.copy()
                # end if
                if self.topo.edges[self.topo.edge_link[isurf][2]].dg == idg:
                    self.surfs[isurf].tv = new_knot_vec.copy()
                # end if
            # end for
        # end for
       
        mpiPrint('Recomputing surfaces...',self.NO_PRINT)

        for isurf in xrange(self.nSurf):
            self.surfs[isurf].recompute()
        # end for

        # Update the coefficients on the local surfaces
        self._setEdgeConnectivity()
        self.update()

        return

    def getSurfaceFromEdge(self,edge):
        '''Determine the surfaces and their edge_link index that points to edge iedge'''
        surfaces = []
        for isurf in xrange(self.nSurf):
            for iedge in xrange(4):
                if self.topo.edge_link[isurf][iedge] == edge:
                    surfaces.append([isurf,iedge])
                # end if
            # end for
        # end for

        return surfaces
# # ----------------------------------------------------------------------
# #                        Surface Fitting Functions 2
# # ----------------------------------------------------------------------

#     def fitSurfaces3(self,nIter=40,constr_tol=1e-6,opt_tol=1e-3):
#         if USE_SNOPT == False:
#             print 'Error: pyOpt_Optimization and/or pySNOPT were not imported correctly \
# These modules must be imported for global surfaces fitting to take place.'
#             sys.exit(1)
#         # end if

#         time0 = time.time()
                    
#         self.ndv = 3*len(self.coef)
#         sizes = [ [self.surfs[isurf].Nu,self.surfs[isurf].Nv] for isurf in xrange(self.nSurf)]
        
#         # Get the Global number of the original data
#         nPts, g_index,l_index = self.calcGlobalNumbering(sizes)
#         self._initJacobian(nPts,self.ndv,g_index)

#         if USE_PETSC:
#             self.rhs = PETSc.Vec().createSeq(nPts*3) # surface points
#             self.temp = PETSc.Vec().createSeq(nPts*3) # A temporary vector for residual calc
#             self.X_PETSC = PETSc.Vec().createSeq(self.ndv) # PETSc version of X
#             self.gobj_PETSC =  PETSc.Vec().createSeq(self.ndv) # PETSc version of objective derivative
#             X = zeros(self.ndv) # X for initial guess
#         else:
#             self.rhs = zeros(nPts*3)
#             X = zeros(self.ndv)
#         # end if 

#         # Fill up the 'X' with the the current coefficients (optimization start point)
#         for icoef in xrange(len(self.coef)):
#             X[3*icoef:3*icoef+3] = self.coef[icoef]
#         # end for

#         # Now Fill up the RHS point list
#         for ii in xrange(len(g_index)):
#             isurf = g_index[ii][0][0]
#             i = g_index[ii][0][1]
#             j = g_index[ii][0][2]
#             self.rhs[3*ii:3*ii+3] = self.surfs[isurf].X[i,j]
#         # end for

#         # Now determine the number of constraints
#         self.ncon = 0
#         for iedge in xrange(len(self.topo.edges)):
#             if self.topo.edges[iedge].cont == 1: # We have continuity
#                 self.ncon += 3*self.topo.edges[iedge].Nctl-2
#             # end if
#         # end for

#         inf = 1e20 # Define a value for infinity
        
#         lower_bound = -inf*ones(self.ndv)
#         upper_bound =  inf*ones(self.ndv)

#         for iedge in xrange(len(self.topo.edges)):
#             surfaces = self.getSurfaceFromEdge(iedge)
#             if len(surfaces) == 1: # Its a mirror edge
#                 surf0 = surfaces[0][0] # First surface on this edge
#                 edge0 = surfaces[0][1] # Edge of surface on this edge     
#                 for i in xrange(self.topo.edges[iedge].Nctl):
#                     if edge0 == 0:
#                         index = self.l_index[surf0][i,0]
#                     elif edge0 == 1:
#                         index = self.l_index[surf0][i,-1]
#                     elif edge0 == 2:
#                         index = self.l_index[surf0][0,i]
#                     elif edge0 == 3:
#                         index = self.l_index[surf0][-1,i]
#                     # end if

#                     if self.sym == 'xy':
#                         lower_bound[3*index+2] = 0
#                         upper_bound[3*index+2] = 0
#                     if self.sym == 'yz':
#                         lower_bound[3*index+0] = 0
#                         upper_bound[3*index+0] = 0
#                     if self.sym == 'xz':
#                         lower_bound[3*index+1] = 0
#                         upper_bound[3*index+1] = 0
#                     # end if
#                 # end for
#             # end if
#         # end for

#         locA,indA = self._computeSparsityPattern3()

#         mpiPrint('------------- Fitting Surfaces Globally ------------------')
#         mpiPrint('nPts (Number of Surface Points):',nPts)
#         mpiPrint('nDV (Degrees of Freedom):',self.ndv)
#         mpiPrint('nCon (Constraints):',self.ncon)
        
#         # Setup Optimization Probelm
#         opt_prob = Optimization('Constrained LMS Fitting',self._objcon2)
#         opt_prob.addVarGroup('x',self.ndv,'c',value=X,lower=lower_bound,upper=upper_bound)
#         opt_prob.addConGroup('cont_constr',self.ncon,'i',lower=0.0,upper=0.0)
#         opt_prob.addObj('RMS Error')
#         opt = SNOPT()

#         opt.setOption('Verify level',-1)
#         opt.setOption('Nonderivative linesearch')
#         opt.setOption('Major step limit',1e-4)
#         opt.setOption('Major optimality tolerance', opt_tol)
#         opt.setOption('Major feasibility tolerance',constr_tol)
#         opt.setOption('Major iterations limit',nIter)
#         opt.setOption('New superbasics limit',250)
#         opt.setOption('Minor iterations limit',1500)
#         #opt.setOption('QPSolver','CG')
#         opt(opt_prob,self._sens3, sparse=[indA,locA],
#             logHistory='SNOPT_history',
#             loadHistory='SNOPT_history') # Run the actual problem
        
#         # Reset the coefficients after the optimization is done
#         for icoef in xrange(len(self.coef)):
#             self.coef[icoef][0] = opt_prob._solutions[0]._variables[3*icoef + 0].value
#             self.coef[icoef][1] = opt_prob._solutions[0]._variables[3*icoef + 1].value
#             self.coef[icoef][2] = opt_prob._solutions[0]._variables[3*icoef + 2].value
#         # end for

#         # Update the entire object with new coefficients
#         self.update()

#         # Delete the self. values we don't need anymore
#         del self.ndv,self.ncon
#         del self.rhs,self.X_PETSC,self.gobj_PETSC,self.temp

#         return


#     def _sens3(self,x,f_obj,f_con,*args,**kwargs):
#         '''Sensitivity function for Fitting Optimization'''
#         time0 = time.time()
#         # ----------- Objective Derivative ----------------
#         if USE_PETSC:
#             self.X_PETSC.setValues(arange(0,self.ndv).astype('intc'),x)
#             self.J(self.X_PETSC,self.temp)
#             self.J.multTranspose(self.temp-self.rhs,self.gobj_PETSC)
#             g_obj = array(self.gobj_PETSC.getValues(arange(self.ndv).astype('intc')))
#             self.temp = self.temp-self.rhs
#             self.temp.abs()
#             print 'Objective: %f, Max Error %f:'%(f_obj,self.temp.max()[1])
#         else:
#             g_obj = dot((dot(self.J,x)-self.rhs),self.J)
#         # end if
#         # ----------- Constraint Derivative ---------------
#         g_con = []
#         counter = 0
#         for iedge in xrange(len(self.topo.edges)):
#             if self.topo.edges[iedge].cont == 1: # We have a continuity edge
#                 # Now get the two surfaces for this edge:
#                 surfaces = self.getSurfaceFromEdge(iedge)
#                 surf0 = surfaces[0][0] # First surface on this edge
#                 edge0 = surfaces[0][1] # Edge of surface on this edge     
#                 # Get the greville points for this edge
#                 gpts = self.surfs[surf0].getGrevillePoints(edge0)
                
#                 for i in xrange(len(gpts)):
#                     du,dv1 = self._getDerivativeOnEdge(surf0,edge0,gpts[i])
#                     dd_i0,ddu,ddv_v0 = self._getDerivativeDerivativeOnEdge(
#                         surf0,edge0,gpts[i])

#                     if len(surfaces) == 1: 
#                         dv2 = self.sym_normal
#                         ddv_v1 = zeros(8)
#                     else:
#                         surf1 = surfaces[1][0] # Second surface on this edge
#                         edge1 = surfaces[1][1] # Edge of second surface on this edge
#                         du,dv2 = self._getDerivativeOnEdge(surf1,edge1,gpts[i])
#                         dd_i1,ddu,ddv_v1 = self._getDerivativeDerivativeOnEdge(surf1,edge1,gpts[i])
#                     # end if

#                     dS = zeros(9)

#                     dS[0] =  (dv1[1]*dv2[2]-dv1[2]*dv2[1]) # dSdu
#                     dS[1] = -(dv1[0]*dv2[2]-dv1[2]*dv2[0])
#                     dS[2] =  (dv1[0]*dv2[1]-dv1[1]*dv2[0])

#                     dS[3] = -du[1]*dv2[2] + du[2]*dv2[1] # dSdv1
#                     dS[4] =  du[0]*dv2[2] - du[2]*dv2[0]
#                     dS[5] = -du[0]*dv2[1] + du[1]*dv2[0]

#                     dS[6] =  du[1]*dv1[2] - du[2]*dv1[1] # dSdv2
#                     dS[7] = -du[0]*dv1[2] + du[2]*dv1[0]
#                     dS[8] =  du[0]*dv1[1] - du[1]*dv1[0]

#                     for ii in xrange(8): # all the return values from the derivative-derivative
#                         # function have a length of 8
#                         if mod(ii,2) == 0:
#                             g_con.append(dS[0]*ddu[ii] + dS[3]*ddv_v0[ii] + dS[6]*ddv_v1[ii])
#                             g_con.append(dS[1]*ddu[ii] + dS[4]*ddv_v0[ii] + dS[7]*ddv_v1[ii])
#                             g_con.append(dS[2]*ddu[ii] + dS[5]*ddv_v0[ii] + dS[8]*ddv_v1[ii])
#                         else:
#                             g_con.append(dS[0]*ddu[ii] + dS[3]*ddv_v0[ii])
#                             g_con.append(dS[1]*ddu[ii] + dS[4]*ddv_v0[ii])
#                             g_con.append(dS[2]*ddu[ii] + dS[5]*ddv_v0[ii])
#                             if len(surfaces) != 1:
#                                 if dd_i1[ii]  == dd_i0[ii]:
#                                     g_con[-3] += dS[0]*ddu[ii] + dS[6]*ddv_v1[ii]
#                                     g_con[-2] += dS[1]*ddu[ii] + dS[7]*ddv_v1[ii]
#                                     g_con[-1] += dS[2]*ddu[ii] + dS[8]*ddv_v1[ii]
#                                 else:
#                                     g_con.append(dS[0]*ddu[ii] + dS[6]*ddv_v1[ii])
#                                     g_con.append(dS[1]*ddu[ii] + dS[7]*ddv_v1[ii])
#                                     g_con.append(dS[2]*ddu[ii] + dS[8]*ddv_v1[ii])
#                                 # end if
#                             # end if
#                         # end if
#                     # end for
#                 # end for (gpts loop)
#             # end if (cont edge)
#         # end for (edge listloop)
#         Bp,Bi,new_gcon = convertCSRtoCSC_one(self.ncon,self.ndv,self.loc,self.index,g_con)
#         print 'Sens Time:',time.time()-time0
#         return g_obj,new_gcon,0

#     def _computeSparsityPattern3(self):
#         '''Compute the sparsity for geometric constraints'''
#         Ai = [] # Index 
#         Ap = [1] # Pointer
#         for iedge in xrange(len(self.topo.edges)):
#             if self.topo.edges[iedge].cont == 1: # We have a continuity edge
#                 # Now get the two surfaces for this edge:
#                 surfaces = self.getSurfaceFromEdge(iedge)
#                 surf0 = surfaces[0][0] # First surface on this edge
#                 edge0 = surfaces[0][1] # Edge of surface on this edge     
#                 gpts = self.surfs[surf0].getGrevillePoints(edge0) # Greville Points
                
#                 for i in xrange(len(gpts)):
#                     dd_i0,ddu,ddv_v0 = self._getDerivativeDerivativeOnEdge(surf0,edge0,gpts[i])

#                     if len(surfaces) == 1: 
#                         pass
#                     else:
#                         surf1 = surfaces[1][0] # Second surface on this edge
#                         edge1 = surfaces[1][1] # Edge of second surface on this edge
#                         dd_i1,ddu,ddv_v1 = self._getDerivativeDerivativeOnEdge(surf1,edge1,gpts[i])
#                     # end if       
#                     temp = []
#                     for ii in xrange(8): 
#                         if mod(ii,2) == 0:
#                             temp.append(dd_i0[ii]*3  )
#                             temp.append(dd_i0[ii]*3+1)
#                             temp.append(dd_i0[ii]*3+2)
#                         else:
#                             temp.append(dd_i0[ii]*3)
#                             temp.append(dd_i0[ii]*3+1)
#                             temp.append(dd_i0[ii]*3+2)
#                             if len(surfaces) != 1:
#                                 # This is where we have to check if its already in
#                                 if dd_i1[ii] != dd_i0[ii]:
#                                     temp.append(dd_i1[ii]*3  )
#                                     temp.append(dd_i1[ii]*3+1)
#                                     temp.append(dd_i1[ii]*3+2)
#                                 # end if
#                             # end if
#                        # end if
#                     # end for
#                     Ai.extend(temp)
#                     Ap.append(Ap[-1]+len(temp))
#                     # end if
#                 # end for
#             # end if (cont edge)
#         # end for (edge listloop)
                    
#         self.loc = Ap
#         self.index = Ai
#         Bp,Bi,Bx = convertCSRtoCSC_one(self.ncon,self.ndv,Ap,Ai,zeros(len(Ai)))

#         return Bp,Bi


# # ----------------------------------------------------------------------
# #                        Surface Fitting Functions 2
# # ----------------------------------------------------------------------

#     def fitSurfaces2(self,nIter=40,constr_tol=1e-6,opt_tol=1e-3):
#         time0 = time.time()
                    
#         self.ndv = 3*len(self.coef)
#         sizes = [ [self.surfs[isurf].Nu,self.surfs[isurf].Nv] for isurf in xrange(self.nSurf)]
        
#         # Get the Global number of the original data
#         nPts, g_index,l_index = self.calcGlobalNumbering(sizes)
#         self._initJacobian(nPts,self.ndv,g_index)

#         if USE_PETSC:
#             self.rhs = PETSc.Vec().createSeq(nPts*3) # surface points
#             self.temp = PETSc.Vec().createSeq(nPts*3) # A temporary vector for residual calc
#             self.X_PETSC = PETSc.Vec().createSeq(self.ndv) # PETSc version of X
#             self.gobj_PETSC =  PETSc.Vec().createSeq(self.ndv) # PETSc version of objective derivative
#             X = zeros(self.ndv) # X for initial guess
#         else:
#             self.rhs = zeros(nPts*3)
#             X = zeros(self.ndv)
#         # end if 

#         # Fill up the 'X' with the the current coefficients (optimization start point)
#         for icoef in xrange(len(self.coef)):
#             X[3*icoef:3*icoef+3] = self.coef[icoef]
#         # end for

#         # Now Fill up the RHS point list
#         for ii in xrange(len(g_index)):
#             isurf = g_index[ii][0][0]
#             i = g_index[ii][0][1]
#             j = g_index[ii][0][2]
#             self.rhs[3*ii:3*ii+3] = self.surfs[isurf].X[i,j]
#         # end for

#         # Now determine the number of constraints
#         self.ncon = 0
#         for iedge in xrange(len(self.topo.edges)):
#             if self.topo.edges[iedge].cont == 1: # We have continuity
#                 self.ncon += self.topo.edges[iedge].Nctl
#             # end if
#         # end for

#         locA,indA = self._computeSparsityPattern2()

#         if not self.NO_PRINT:
#             print '------------- Fitting Surfaces Globally ------------------'
#             print 'nPts (Number of Surface Points):',nPts
#             print 'nDV (Degrees of Freedom):',self.ndv
#             print 'nCon (Constraints):',self.ncon
#         # end if

#         # Setup Optimization Probelm
#         opt_prob = Optimization('Constrained LMS Fitting',self._objcon2)
#         opt_prob.addVarGroup('x',self.ndv,'c',value=X)
#         opt_prob.addConGroup('cont_constr',self.ncon,'i',lower=0.0,upper=0.0)
#         opt_prob.addObj('RMS Error')
#         opt = SNOPT()

#         opt.setOption('Verify level',-1)
#         opt.setOption('Nonderivative linesearch')
#         opt.setOption('Major step limit',1e-2)
#         opt.setOption('Major optimality tolerance', opt_tol)
#         opt.setOption('Major feasibility tolerance',constr_tol)
#         opt.setOption('Major iterations limit',nIter)
#         opt.setOption('New superbasics limit',250)
#         opt.setOption('Minor iterations limit',500)

#         opt(opt_prob,self._sens2,sparse=[indA,locA]) # Run the actual problem

#         # Reset the coefficients after the optimization is done
#         for icoef in xrange(len(self.coef)):
#             self.coef[icoef][0] = opt_prob._solutions[0]._variables[3*icoef + 0].value
#             self.coef[icoef][1] = opt_prob._solutions[0]._variables[3*icoef + 1].value
#             self.coef[icoef][2] = opt_prob._solutions[0]._variables[3*icoef + 2].value
#         # end for

#         # Update the entire object with new coefficients
#         self.update()

#         # Delete the self. values we don't need anymore
#         del self.ndv,self.ncon
#         del self.rhs,self.X_PETSC,self.gobj_PETSC,self.temp

#         return

#     def _objcon2(self,x,*arg,**kwargs):
#         '''Compute the objective and the constraints'''
#         # ------------ Objective ---------
#         if USE_PETSC:
#             self.X_PETSC.setValues(arange(self.ndv).astype('intc'),x)
#             self.J.mult(self.X_PETSC,self.temp)
#             f_obj = 0.5*(self.temp-self.rhs).norm()**2
#         else:
#             f_obj = 0.5*norm(dot(self.J,x)-self.rhs)**2

#         # ---------- Constraints ---------
#         for icoef in xrange(len(self.coef)):
#             self.coef[icoef,0] = x[3*icoef+0]
#             self.coef[icoef,1] = x[3*icoef+1]
#             self.coef[icoef,2] = x[3*icoef+2]
#         # end for
#         self.update()

#         f_con = []
#         for iedge in xrange(len(self.topo.edges)):
#             if self.topo.edges[iedge].cont == 1: # We have a continuity edge
#                 # Now get the two surfaces for this edge:
#                 surfaces = self.getSurfaceFromEdge(iedge)
#                 surf0 = surfaces[0][0] # First surface on this edge
#                 edge0 = surfaces[0][1] # Edge of surface on this edge     
#                 # Get the greville points for this edge
#                 gpts = self.surfs[surf0].getGrevillePoints(edge0)

#                 for i in xrange(len(gpts)):
#                     du,dv1 = self._getDerivativeOnEdge(surf0,edge0,gpts[i])
#                     if len(surfaces) == 1: 
#                         dv2 = self.sym_normal
#                     else:
#                         surf1 = surfaces[1][0] # Second surface on this edge
#                         edge1 = surfaces[1][1] # Edge of second surface on this edge
#                         du,dv2 = self._getDerivativeOnEdge(surf1,edge1,gpts[i])
#                     # end if

#                     S = du[0]*(dv1[1]*dv2[2]-dv1[2]*dv2[1]) - \
#                         du[1]*(dv1[0]*dv2[2]-dv1[2]*dv2[0]) + \
#                         du[2]*(dv1[0]*dv2[1]-dv1[1]*dv2[0])
#                     f_con.append(S) # scalar triple product
#                 # end for (nctl on edge)
#             # end if (continuity edge)
#         # end for (edge list)
#         return f_obj,f_con,0

#     def _sens2(self,x,f_obj,f_con,*args,**kwargs):
#         '''Sensitivity function for Fitting Optimization'''
#         time0 = time.time()
#         # ----------- Objective Derivative ----------------
#         if USE_PETSC:
#             self.X_PETSC.setValues(arange(0,self.ndv).astype('intc'),x)
#             self.J(self.X_PETSC,self.temp)
#             self.J.multTranspose(self.temp-self.rhs,self.gobj_PETSC)
#             g_obj = array(self.gobj_PETSC.getValues(arange(self.ndv).astype('intc')))
#             self.temp = self.temp-self.rhs
#             self.temp.abs()
#             print 'Objective: %f, Max Error %f:'%(f_obj,self.temp.max()[1])
#         else:
#             g_obj = dot((dot(self.J,x)-self.rhs),self.J)
#         # end if
#         # ----------- Constraint Derivative ---------------
#         g_con = []
#         counter = 0
#         for iedge in xrange(len(self.topo.edges)):
#             if self.topo.edges[iedge].cont == 1: # We have a continuity edge
#                 # Now get the two surfaces for this edge:
#                 surfaces = self.getSurfaceFromEdge(iedge)
#                 surf0 = surfaces[0][0] # First surface on this edge
#                 edge0 = surfaces[0][1] # Edge of surface on this edge     
#                 # Get the greville points for this edge
#                 gpts = self.surfs[surf0].getGrevillePoints(edge0)
                
#                 for i in xrange(len(gpts)):
#                     du,dv1 = self._getDerivativeOnEdge(surf0,edge0,gpts[i])
#                     dd_i0,ddu,ddv_v0 = self._getDerivativeDerivativeOnEdge(
#                         surf0,edge0,gpts[i])

#                     if len(surfaces) == 1: 
#                         dv2 = self.sym_normal
#                         ddv_v1 = zeros(8)
#                     else:
#                         surf1 = surfaces[1][0] # Second surface on this edge
#                         edge1 = surfaces[1][1] # Edge of second surface on this edge
#                         du,dv2 = self._getDerivativeOnEdge(surf1,edge1,gpts[i])
#                         dd_i1,ddu,ddv_v1 = self._getDerivativeDerivativeOnEdge(surf1,edge1,gpts[i])
#                     # end if

#                     dS = zeros(9)

#                     dS[0] =  (dv1[1]*dv2[2]-dv1[2]*dv2[1]) # dSdu
#                     dS[1] = -(dv1[0]*dv2[2]-dv1[2]*dv2[0])
#                     dS[2] =  (dv1[0]*dv2[1]-dv1[1]*dv2[0])

#                     dS[3] = -du[1]*dv2[2] + du[2]*dv2[1] # dSdv1
#                     dS[4] =  du[0]*dv2[2] - du[2]*dv2[0]
#                     dS[5] = -du[0]*dv2[1] + du[1]*dv2[0]

#                     dS[6] =  du[1]*dv1[2] - du[2]*dv1[1] # dSdv2
#                     dS[7] = -du[0]*dv1[2] + du[2]*dv1[0]
#                     dS[8] =  du[0]*dv1[1] - du[1]*dv1[0]

#                     for ii in xrange(8): # all the return values from the derivative-derivative
#                         # function have a length of 8
#                         if mod(ii,2) == 0:
#                             g_con.append(dS[0]*ddu[ii] + dS[3]*ddv_v0[ii] + dS[6]*ddv_v1[ii])
#                             g_con.append(dS[1]*ddu[ii] + dS[4]*ddv_v0[ii] + dS[7]*ddv_v1[ii])
#                             g_con.append(dS[2]*ddu[ii] + dS[5]*ddv_v0[ii] + dS[8]*ddv_v1[ii])
#                         else:
#                             g_con.append(dS[0]*ddu[ii] + dS[3]*ddv_v0[ii])
#                             g_con.append(dS[1]*ddu[ii] + dS[4]*ddv_v0[ii])
#                             g_con.append(dS[2]*ddu[ii] + dS[5]*ddv_v0[ii])
#                             if len(surfaces) != 1:
#                                 g_con.append(dS[0]*ddu[ii] + dS[6]*ddv_v1[ii])
#                                 g_con.append(dS[1]*ddu[ii] + dS[7]*ddv_v1[ii])
#                                 g_con.append(dS[2]*ddu[ii] + dS[8]*ddv_v1[ii])
#                             # end if
#                         # end if
#                     # end for
#                 # end for (gpts loop)
#             # end if (cont edge)
#         # end for (edge listloop)
#         Bp,Bi,new_gcon = convertCSRtoCSC_one(self.ncon,self.ndv,self.loc,self.index,g_con)
#         print 'Sens Time:',time.time()-time0
#         return g_obj,new_gcon,0

#     def _computeSparsityPattern2(self):
#         '''Compute the sparsity for geometric constraints'''
#         Ai = [] # Index 
#         Ap = [1] # Pointer
#         for iedge in xrange(len(self.topo.edges)):
#             if self.topo.edges[iedge].cont == 1: # We have a continuity edge
#                 # Now get the two surfaces for this edge:
#                 surfaces = self.getSurfaceFromEdge(iedge)
#                 surf0 = surfaces[0][0] # First surface on this edge
#                 edge0 = surfaces[0][1] # Edge of surface on this edge     
#                 gpts = self.surfs[surf0].getGrevillePoints(edge0) # Greville Points
                
#                 for i in xrange(len(gpts)):
#                     dd_i0,ddu,ddv_v0 = self._getDerivativeDerivativeOnEdge(surf0,edge0,gpts[i])

#                     if len(surfaces) == 1: 
#                         pass
#                     else:
#                         surf1 = surfaces[1][0] # Second surface on this edge
#                         edge1 = surfaces[1][1] # Edge of second surface on this edge
#                         dd_i1,ddu,ddv_v1 = self._getDerivativeDerivativeOnEdge(surf1,edge1,gpts[i])
#                     # end if       
#                     for ii in xrange(8): 
#                         if mod(ii,2) == 0:
#                             Ai.append(dd_i0[ii]*3)
#                             Ai.append(dd_i0[ii]*3+1)
#                             Ai.append(dd_i0[ii]*3+2)
#                         else:
#                             Ai.append(dd_i0[ii]*3)
#                             Ai.append(dd_i0[ii]*3+1)
#                             Ai.append(dd_i0[ii]*3+2)
#                             if len(surfaces) != 1:
#                                 Ai.append(dd_i1[ii]*3  )
#                                 Ai.append(dd_i1[ii]*3+1)
#                                 Ai.append(dd_i1[ii]*3+2)
#                             # end if
#                        # end if
#                     # end for
#                     if len(surfaces) == 1:
#                         Ap.append(Ap[-1] + 24)
#                     else:
#                         Ap.append(Ap[-1] + 36)
#                     # end if
#                 # end for
#             # end if (cont edge)
#         # end for (edge listloop)
#         Ax = zeros(len(Ai))
#         self.loc = Ap
#         self.index = Ai
#         Bp,Bi,Bx = convertCSRtoCSC_one(self.ncon,self.ndv,Ap,Ai,Ax)

#         return Bp,Bi

#     def _getDerivativeDerivativeOnEdge(self,surface,edge,value):
#         '''Get the derivative of du and dv wrt all applicable control points
#         for surface surface, edge edge and at position value'''
#         # Note: In this context du is the derivative along the edge,
#         # dv is the derivative normal to the edge
        
#         dd_i = [] # Index
#         ddu_val = [] # Value
#         ddv_val = [] # Value
#         if self.edge_dir[surface,edge] == -1: # Direction Check
#             orig_value = value
#             value = 1-value
#         # end if
            
#         if edge in [0,1]:
#             t = self.surfs[surface].tu
#             k = self.surfs[surface].ku
#         else:
#             t = self.surfs[surface].tv
#             k = self.surfs[surface].kv
#         # end if

#         ileft, mflag = self.surfs[surface].pyspline.intrv(t,value,1)
#         if mflag == 0: # Its Inside so everything is ok
#             pt_list = [ileft-k,ileft-k+1,ileft-k+2,ileft-k+3]
#         # end if
#         if mflag == 1: # Its at the right end
#             pt_list = [ileft-k-4,ileft-k-3,ileft-k-2,ileft-k-1]
#         # end if
            
            
#         if self.edge_dir[surface,edge] == -1:
#             pt_list = pt_list[::-1]
#         # end if

#         if edge == 0:
#             for i in xrange(4):
#                 dd_i.append(self.l_index[surface][pt_list[i],0])
#                 ddu,ddv = self.surfs[surface].calcDerivativeDeriv(value,0,pt_list[i],0)
#                 ddu_val.append(ddu)
#                 ddv_val.append(ddv)
#                 dd_i.append(self.l_index[surface][pt_list[i],1])
#                 ddu,ddv = self.surfs[surface].calcDerivativeDeriv(value,0,pt_list[i],1)
#                 ddu_val.append(ddu)
#                 ddv_val.append(ddv)
#             # end for
#         elif edge == 1:
#             for i in xrange(4):
#                 dd_i.append(self.l_index[surface][pt_list[i],-1])
#                 ddu,ddv = self.surfs[surface].calcDerivativeDeriv(value,1,pt_list[i],-1)
#                 ddu_val.append(ddu)
#                 ddv_val.append(ddv)
#                 dd_i.append(self.l_index[surface][pt_list[i],-2])
#                 ddu,ddv = self.surfs[surface].calcDerivativeDeriv(value,1,pt_list[i],-2)
#                 ddu_val.append(ddu)
#                 ddv_val.append(ddv)
#             # end for
#         elif edge == 2:
#             for i in xrange(4):
#                 dd_i.append(self.l_index[surface][0,pt_list[i]])
#                 ddu,ddv = self.surfs[surface].calcDerivativeDeriv(0,value,0,pt_list[i])
#                 ddu_val.append(ddv)
#                 ddv_val.append(ddu)
#                 dd_i.append(self.l_index[surface][1,pt_list[i]])
#                 ddu,ddv = self.surfs[surface].calcDerivativeDeriv(0,value,1,pt_list[i])
#                 ddu_val.append(ddv)
#                 ddv_val.append(ddu)
#             # end for
#         elif edge == 3:
#             for i in xrange(4):
#                 dd_i.append(self.l_index[surface][-1,pt_list[i]])
#                 ddu,ddv = self.surfs[surface].calcDerivativeDeriv(1,value,-1,pt_list[i])
#                 ddu_val.append(ddv)
#                 ddv_val.append(ddu)
#                 dd_i.append(self.l_index[surface][-2,pt_list[i]])
#                 ddu,ddv = self.surfs[surface].calcDerivativeDeriv(1,value,-2,pt_list[i])
#                 ddu_val.append(ddv)
#                 ddv_val.append(ddu)
#             # end for
#         # end if
#         return dd_i,ddu_val,ddv_val


#     def _getDerivativeOnEdge(self,surface,edge,value):
#         '''Get the directional surface derivatives on surface suface and edge edge'''
#         # Note: In this context du is the derivative along the edge,
#         # dv is the derivative normal to the edge
#         if self.edge_dir[surface,edge] == -1:
#             value = 1-value
#         # end if
#         if edge == 0:
#             du,dv = self.surfs[surface].getDerivative(value,0)
#             return du,dv
#         elif edge == 1:
#             du,dv = self.surfs[surface].getDerivative(value,1)
#             return du,dv
#         elif edge == 2:
#             du,dv = self.surfs[surface].getDerivative(0,value)
#             return dv,du
#         elif edge == 3:
#             du,dv = self.surfs[surface].getDerivative(1,value)
#             return dv,du
#         # end if

# # ----------------------------------------------------------------------
# #                        Surface Fitting Functions
# # ----------------------------------------------------------------------

#     def fitSurfaces(self,nIter=50,constr_tol=1e-7,opt_tol=1e-4):
#         if USE_SNOPT == False:
#             print 'Error: pyOpt_Optimization and/or pySNOPT were not imported correctly \
# These modules must be imported for global surfaces fitting to take place.'
#             sys.exit(1)
#         # end if

#         time0 = time.time()
                    
#         self.ndv = 3*len(self.coef)
#         sizes = [ [self.surfs[isurf].Nu,self.surfs[isurf].Nv] for isurf in xrange(self.nSurf)]
        
#         # Get the Global number of the original data
#         self.topo.calcGlobalNumbering(sizes)
#         nPts=self.topo.counter
#         g_index=self.topo.g_index
#         l_index=self.topo.l_index
#         self._initJacobian(nPts,self.ndv,g_index)

#         if USE_PETSC:
#             self.rhs = PETSc.Vec().createSeq(nPts*3) # surface points
#             self.temp = PETSc.Vec().createSeq(nPts*3) # A temporary vector for residual calc
#             self.X_PETSC = PETSc.Vec().createSeq(self.ndv) # PETSc version of X
#             self.gobj_PETSC =  PETSc.Vec().createSeq(self.ndv) # PETSc version of objective derivative
#             X = zeros(self.ndv) # X for initial guess
#         else:
#             self.rhs = zeros(nPts*3)
#             X = zeros(self.ndv)
#         # end if 

#         # Fill up the 'X' with the the current coefficients (optimization start point)
#         for icoef in xrange(len(self.coef)):
#             X[3*icoef:3*icoef+3] = self.coef[icoef]
#         # end for

#         # Now Fill up the RHS point list
#         for ii in xrange(len(g_index)):
#             isurf = g_index[ii][0][0]
#             i = g_index[ii][0][1]
#             j = g_index[ii][0][2]
#             self.rhs[3*ii:3*ii+3] = self.surfs[isurf].X[i,j]
#         # end for

#         # Now determine the number of constraints
#         self.ncon = 0
#         for iedge in xrange(len(self.topo.edges)):
#             if self.topo.edges[iedge].cont == 1: # We have continuity
#                 self.ncon += self.topo.edges[iedge].Nctl*3
#             # end if
#         # end for
                
#         inf = 1e20 # Define a value for infinity
        
#         lower_bound = -inf*ones(self.ndv)
#         upper_bound =  inf*ones(self.ndv)

#         for iedge in xrange(len(self.topo.edges)):
#             surfaces = self.getSurfaceFromEdge(iedge)
#             if len(surfaces) == 1: # Its a mirror edge
#                 surf0 = surfaces[0][0] # First surface on this edge
#                 edge0 = surfaces[0][1] # Edge of surface on this edge     

#                 for i in xrange(self.topo.edges[iedge].Nctl):
#                     if edge0 == 0:
#                         index = self.l_index[surf0][i,0]
#                     elif edge0 == 1:
#                         index = self.l_index[surf0][i,-1]
#                     elif edge0 == 2:
#                         index = self.l_index[surf0][0,i]
#                     elif edge0 == 3:
#                         index = self.l_index[surf0][-1,i]
#                     # end if

#                     if self.sym == 'xy':
#                         lower_bound[3*index+2] = 0
#                         upper_bound[3*index+2] = 0
#                     if self.sym == 'yz':
#                         lower_bound[3*index+0] = 0
#                         upper_bound[3*index+0] = 0
#                     if self.sym == 'xz':
#                         lower_bound[3*index+1] = 0
#                         upper_bound[3*index+1] = 0
#                     # end if
#                 # end for
#             # end if
#         # end for

#         locA,indA = self._computeSparsityPattern()

#         print '------------- Fitting Surfaces Globally ------------------'
#         print 'nPts (Number of Surface Points):',nPts
#         print 'nDV (Degrees of Freedom):',self.ndv
#         print 'nCon (Constraints):',self.ncon

#         # Setup Optimization Probelm
#         opt_prob = Optimization('Constrained LMS Fitting',self._objcon)
#         opt_prob.addVarGroup('x',self.ndv,'c',value=X,lower=lower_bound,upper=upper_bound)
#         opt_prob.addConGroup('cont_constr',self.ncon,'i',lower=0.0,upper=0.0)
#         opt_prob.addObj('RMS Error')
#         opt = SNOPT()
#         opt.setOption('Verify level',-1)
#         opt.setOption('Nonderivative linesearch')
#         opt.setOption('Major step limit',1e-2)
#         opt.setOption('Major optimality tolerance', opt_tol)
#         opt.setOption('Major feasibility tolerance',constr_tol)
#         opt.setOption('Major iterations limit',nIter)
#         opt.setOption('New superbasics limit',250)
#         opt.setOption('Minor iterations limit',500)
#         opt.setOption('QPSolver','CG')
#         opt(opt_prob,self._sens,sparse=[indA,locA]) # Run the actual problem

#         # Reset the coefficients after the optimization is done
#         for icoef in xrange(len(self.coef)):
#             self.coef[icoef][0] = opt_prob._solutions[0]._variables[3*icoef + 0].value
#             self.coef[icoef][1] = opt_prob._solutions[0]._variables[3*icoef + 1].value
#             self.coef[icoef][2] = opt_prob._solutions[0]._variables[3*icoef + 2].value
#         # end for

#         # Update the entire object with new coefficients
#         self.update()

#         # Delete the self. values we don't need anymore
#         del self.ndv
#         del self.ncon
#         del self.rhs
#         del self.X_PETSC
#         del self.gobj_PETSC
#         del self.temp
#         del self.loc
#         del self.index
#         del opt_prob
#         del opt
#         print 'Fitting Time: %f seconds'%(time.time()-time0)
#         return

#     def _computeSparsityPattern(self):
#         '''Compute the sparsity pattern for the constraints. Return the SNOPT
#         defined indA,locA to be passed into the optimizer.'''

#         Ai = [] # Index 
#         Ap = [1] # Pointer
#         for iedge in xrange(len(self.topo.edges)):
#             if self.topo.edges[iedge].cont == 1: # We have a continuity edge
#                 surfaces = self.getSurfaceFromEdge(iedge)
#                 if len(surfaces) > 2: 
#                     print 'Continuity is not defined for more than 2 surfaces'
#                     sys.exit(1)
#                 # end if
#                 if len(surfaces) == 1:
#                     if self.sym == None:
#                         print 'Error: A symmetry plane must be speficied using .setSymmetry( ) \
# command in pyGeo in order to use continuity of free (i.e. mirrored) surfaces)'
#                         sys.exit(1)
#                     # end if

#                     surf0 = surfaces[0][0] # First surface on this edge
#                     edge0 = surfaces[0][1] # Edge of surface on this edge                           
#                     for i in xrange(self.topo.edges[iedge].Nctl):
#                         indA,indB = self._getTwoIndiciesOnEdge(
#                             self.l_index[surf0],i,edge0,self.edge_dir[surf0])
#                         Ai.append(3*indB+1)
#                         Ai.append(3*indB+2)
#                         Ai.append(3*indA+1)
#                         Ai.append(3*indA+2)
#                         Ap.append(Ap[-1] + 4)

#                         Ai.append(3*indB+2)
#                         Ai.append(3*indB+0)
#                         Ai.append(3*indA+2)
#                         Ai.append(3*indA+0)
#                         Ap.append(Ap[-1] + 4)

#                         Ai.append(3*indB+0)
#                         Ai.append(3*indB+1)
#                         Ai.append(3*indA+0)
#                         Ai.append(3*indA+1)
#                         Ap.append(Ap[-1] + 4)
#                      # end for
#                 else:
#                     surf0 = surfaces[0][0] # First surface on this edge
#                     edge0 = surfaces[0][1] # Edge of surface on this edge                           
#                     surf1 = surfaces[1][0] # Second surface on this edge
#                     edge1 = surfaces[1][1] # Edge of second surface on this edge

#                     for i in xrange(self.topo.edges[iedge].Nctl):
#                         indA,indB = self._getTwoIndiciesOnEdge(
#                             self.l_index[surf0],i,edge0,self.edge_dir[surf0])
#                         indA,indC = self._getTwoIndiciesOnEdge(
#                             self.l_index[surf1],i,edge1,self.edge_dir[surf1])

#                         Ai.append(3*indB+1)
#                         Ai.append(3*indB+2)
#                         Ai.append(3*indC+2)
#                         Ai.append(3*indC+1)
#                         Ai.append(3*indA+1)
#                         Ai.append(3*indA+2)
#                         Ap.append(Ap[-1] + 6)

#                         Ai.append(3*indB+2)
#                         Ai.append(3*indB+0)
#                         Ai.append(3*indC+0)
#                         Ai.append(3*indC+2)
#                         Ai.append(3*indA+2)
#                         Ai.append(3*indA+0)
#                         Ap.append(Ap[-1] + 6)

#                         Ai.append(3*indB+0)
#                         Ai.append(3*indB+1)
#                         Ai.append(3*indC+1)
#                         Ai.append(3*indC+0)
#                         Ai.append(3*indA+0)
#                         Ai.append(3*indA+1)
#                         Ap.append(Ap[-1] + 6)
#                     # end for (nctl on edge)
#                 # end if
#             # end if (continuity edge)
#         # end for (edge list)
#         Ax = zeros(len(Ai)) # Dummy Ax data
#         self.loc = Ap
#         self.index = Ai
#         Bp,Bi,Bx = convertCSRtoCSC_one(self.ncon,self.ndv,Ap,Ai,Ax)

#         return Bp,Bi

#     def _objcon(self,x,*arg,**kwargs):
#         '''Compute the objective and the constraints'''
#         # ------------ Objective ---------
#         if USE_PETSC:
#             self.X_PETSC.setValues(arange(self.ndv).astype('intc'),x)
#             self.J.mult(self.X_PETSC,self.temp)
#             f_obj = 0.5*(self.temp-self.rhs).norm()**2
          
#         else:
#             f_obj = 0.5*norm(dot(self.J,x)-self.rhs)**2
#         # ---------- Constraints ---------

#         f_con = []
#         sym = self.sym_normal
#         for iedge in xrange(len(self.topo.edges)):
#             if self.topo.edges[iedge].cont == 1: # We have a continuity edge
                
#                 # Now get the two surfaces for this edge:
#                 surfaces = self.getSurfaceFromEdge(iedge)
#                 if len(surfaces) == 1: 

#                     surf0 = surfaces[0][0] # First surface on this edge
#                     edge0 = surfaces[0][1] # Edge of surface on this edge                           
#                     for i in xrange(self.topo.edges[iedge].Nctl):
#                         indA,indB = self._getTwoIndiciesOnEdge(
#                             self.l_index[surf0],i,edge0,self.edge_dir[surf0])

#                         f_con.append(
#                             (x[3*indB + 1] - x[3*indA + 1])*sym[2]- 
#                             (x[3*indB + 2] - x[3*indA + 2])*sym[1])
                        
#                         f_con.append(
#                             (x[3*indB + 2] - x[3*indA + 2])*sym[0]-
#                             (x[3*indB + 0] - x[3*indA + 0])*sym[2])
                        
#                         f_con.append(
#                             (x[3*indB + 0] - x[3*indA + 0])*sym[1]-
#                             (x[3*indB + 1] - x[3*indA + 1])*sym[0])
#                     # end for
#                 else:
#                     surf0 = surfaces[0][0] # First surface on this edge
#                     edge0 = surfaces[0][1] # Edge of surface on this edge                           
                    
#                     surf1 = surfaces[1][0] # Second surface on this edge
#                     edge1 = surfaces[1][1] # Edge of second surface on this edge

#                     for i in xrange(self.topo.edges[iedge].Nctl):

#                         indA,indB = self._getTwoIndiciesOnEdge(
#                             self.l_index[surf0],i,edge0,self.edge_dir[surf0])

#                         indA,indC = self._getTwoIndiciesOnEdge(
#                             self.l_index[surf1],i,edge1,self.edge_dir[surf1])

#                         # indB and indC are the global indicies of the two control 
#                         # points on either side of this node on the edge (indA)

#                         f_con.append(
#                             (x[3*indB + 1] - x[3*indA + 1])*(x[3*indC + 2]-x[3*indA + 2])- 
#                             (x[3*indB + 2] - x[3*indA + 2])*(x[3*indC + 1]-x[3*indA + 1]))

#                         f_con.append(
#                             (x[3*indB + 2] - x[3*indA + 2])*(x[3*indC + 0]-x[3*indA + 0])- 
#                             (x[3*indB + 0] - x[3*indA + 0])*(x[3*indC + 2]-x[3*indA + 2]))

#                         f_con.append(
#                             (x[3*indB + 0] - x[3*indA + 0])*(x[3*indC + 1]-x[3*indA + 1])- 
#                             (x[3*indB + 1] - x[3*indA + 1])*(x[3*indC + 0]-x[3*indA + 0]))
#                     # end for (nctl on edge)
#             # end if (continuity edge)
#         # end for (edge list)

#         return f_obj,f_con,0

#     def _sens(self,x,f_obj,f_con,*args,**kwargs):
#         '''Sensitivity function for Fitting Optimization'''
#         # ----------- Objective Derivative ----------------
#         if USE_PETSC:
#             self.X_PETSC.setValues(arange(self.ndv).astype('intc'),x)
#             self.J(self.X_PETSC,self.temp)
#             self.J.multTranspose(self.temp-self.rhs,self.gobj_PETSC)
#             gobj = array(self.gobj_PETSC.getValues(arange(self.ndv).astype('intc')))
#             self.temp = self.temp-self.rhs
#             self.temp.abs()
#             print 'Objective: %f, Max Error %f:'%(f_obj,self.temp.max()[1])
#         else:
#             gobj = dot((dot(self.J,x)-self.rhs),self.J)
#         # end if
#         # ----------- Constraint Derivative ---------------

#         sym = self.sym_normal
#         gcon = []
#         for iedge in xrange(len(self.topo.edges)):
#             if self.topo.edges[iedge].cont == 1: # We have a continuity edge
#                 # Now get the two surfaces for this edge:
#                 surfaces = self.getSurfaceFromEdge(iedge)
#                 if len(surfaces) == 1: 
#                     surf0 = surfaces[0][0] # First surface on this edge
#                     edge0 = surfaces[0][1] # Edge of surface on this edge                           
#                     for i in xrange(self.topo.edges[iedge].Nctl):
#                         indA,indB = self._getTwoIndiciesOnEdge(
#                             self.l_index[surf0],i,edge0,self.edge_dir[surf0])
                   
#                         gcon.append(  sym[2] )
#                         gcon.append( -sym[1] )
#                         gcon.append( -sym[2] )
#                         gcon.append(  sym[1] )

#                         gcon.append(  sym[0] )
#                         gcon.append( -sym[2] )
#                         gcon.append( -sym[0] )
#                         gcon.append(  sym[2] )

#                         gcon.append(  sym[1] )
#                         gcon.append( -sym[0] )
#                         gcon.append( -sym[1] )
#                         gcon.append(  sym[0] )
#                 else:
#                     surf0 = surfaces[0][0] # First surface on this edge
#                     edge0 = surfaces[0][1] # Edge of surface on this edge                           
#                     surf1 = surfaces[1][0] # Second surface on this edge
#                     edge1 = surfaces[1][1] # Edge of second surface on this edge

#                     for i in xrange(self.topo.edges[iedge].Nctl):

#                         indA,indB = self._getTwoIndiciesOnEdge(
#                             self.l_index[surf0],i,edge0,self.edge_dir[surf0])

#                         indA,indC = self._getTwoIndiciesOnEdge(
#                             self.l_index[surf1],i,edge1,self.edge_dir[surf1])

#                         # indB and indC are the global indicies of the two control 
#                         # points on either side of this node on the edge (indA)

#                         gcon.append( x[3*indC + 2]-x[3*indA + 2] )
#                         gcon.append( -x[3*indC + 1]+x[3*indA + 1] )
#                         gcon.append( x[3*indB + 1]-x[3*indA + 1] )
#                         gcon.append (-x[3*indB + 2]+x[3*indA + 2] )
#                         gcon.append( -x[3*indC + 2]+x[3*indA + 2] + x[3*indB+2] - x[3*indA + 2] )
#                         gcon.append( -x[3*indB + 1]+x[3*indA + 1] + x[3*indC+1] - x[3*indA + 1] )

#                         gcon.append(  x[3*indC + 0]-x[3*indA + 0] )
#                         gcon.append( -x[3*indC + 2]+x[3*indA + 2] )
#                         gcon.append(  x[3*indB + 2]-x[3*indA + 2] )
#                         gcon.append( -x[3*indB + 0]+x[3*indA + 0] ) 
#                         gcon.append( -x[3*indC + 0]+x[3*indA + 0] + x[3*indB+0] - x[3*indA + 0] )
#                         gcon.append(-x[3*indB + 2]+x[3*indA + 2] + x[3*indC+2] - x[3*indA + 2] )

#                         gcon.append(  x[3*indC + 1]-x[3*indA + 1] )
#                         gcon.append( -x[3*indC + 0]+x[3*indA + 0] )
#                         gcon.append(  x[3*indB + 0]-x[3*indA + 0] )
#                         gcon.append( -x[3*indB + 1]+x[3*indA + 1] )
#                         gcon.append(-x[3*indC + 1]+x[3*indA + 1] + x[3*indB+1] - x[3*indA + 1] )
#                         gcon.append( -x[3*indB + 0]+x[3*indA + 0] + x[3*indC+0] - x[3*indA + 0] )

#                     # end for
#             # end if (cont edge)
#         # end for (edge listloop)
#         Bp,Bi,new_gcon = convertCSRtoCSC_one(self.ncon,self.ndv,self.loc,self.index,gcon)

#         return gobj,new_gcon,0

#     def _addJacobianValue(self,i,j,value):
#         if USE_PETSC: 
#             self.J.setValue(i,j,value,PETSc.InsertMode.ADD_VALUES)
#         else:
#             self.J[i,j] += value
#         # end if

#     def _initJacobian(self,nPts,ndv,g_index):
#         '''Initialize the Jacobian either with PETSc or with Numpy for use
#         with LAPACK and fill it up'''
#         nRows = nPts*3
#         nCols = ndv
#         if USE_PETSC:
#             self.J = PETSc.Mat()
#             # We know the row filling factor: 16*3 (4 for ku by 4 for
#             # kv and 3 spatial)
#             if PETSC_MAJOR_VERSION == 1:
#                 self.J.createAIJ([nRows,nCols],nnz=16*3,comm=PETSc.COMM_SELF)
#             elif PETSC_MAJOR_VERSION == 0:
#                 self.J.createSeqAIJ([nRows,nCols],nz=16*3)
#             else:
#                 print 'Error: PETSC_MAJOR_VERSION = %d is not supported'%(PETSC_MAJOR_VERSION)
#                 sys.exit(1)
#             # end if
#         else:
#             self.J = zeros((nRows,nCols))
#         # end if

#         # Assemble the Jacobian
#         for ii in xrange(nPts):
#             #  -------- Short Version of some data for cleaner code ---------
#             surfID = g_index[ii][0][0]
#             i      = g_index[ii][0][1]
#             j      = g_index[ii][0][2]
            
#             u = self.surfs[surfID].u[i]
#             v = self.surfs[surfID].v[j]

#             ku = self.surfs[surfID].ku
#             kv = self.surfs[surfID].kv
#             # ---------------------------------------------------------------
#             ileftu, mflagu = self.surfs[surfID].pyspline.intrv(self.surfs[surfID].tu,u,1)
#             ileftv, mflagv = self.surfs[surfID].pyspline.intrv(self.surfs[surfID].tv,v,1)

#             if mflagu == 0: # Its Inside so everything is ok
#                 u_list = [ileftu-ku,ileftu-ku+1,ileftu-ku+2,ileftu-ku+3]
#             if mflagu == 1: # Its at the right end so just need last one
#                 u_list = [ileftu-ku-1]

#             if mflagv == 0: # Its Inside so everything is ok
#                 v_list = [ileftv-kv,ileftv-kv+1,ileftv-kv+2,ileftv-kv+3]
#             if mflagv == 1: # Its at the right end so just need last one
#                 v_list = [ileftv-kv-1]

#             for iii in xrange(len(u_list)):
#                 for jjj in xrange(len(v_list)):
#                     x = self.surfs[surfID].calcPtDeriv(\
#                         u,v,u_list[iii],v_list[jjj])
#                     global_index = self.topo.l_index[surfID][u_list[iii],v_list[jjj]]
#                     self._addJacobianValue(3*ii    ,global_index*3    ,x)
#                     self._addJacobianValue(3*ii + 1,global_index*3 + 1,x)
#                     self._addJacobianValue(3*ii + 2,global_index*3 + 2,x)
#                 # end for (jjj)
#             # end for  (iii)
#         # end for (ii pt loop)
#         if USE_PETSC:
#             self.J.assemblyBegin()
#             self.J.assemblyEnd()
#         # end if

#         return 

    def _getTwoIndiciesOnEdge(self,interpolant,index,edge,edge_dir):
        '''for a given interpolat matrix, get the two values in interpolant
        that coorspond to \'index\' along \'edge\'. The direction is
        accounted for by edge_dir'''
        N = interpolant.shape[0]
        M = interpolant.shape[1]
        if edge == 0:
            if edge_dir[0] == 1:
                return interpolant[index,0],interpolant[index,1]
            else:
                return interpolant[N-index-1,0],interpolant[N-index-1,1]
        elif edge == 1:
            if edge_dir[1] == 1:
                return interpolant[index,-1],interpolant[index,-2]
            else:
                return interpolant[N-index-1,-1],interpolant[N-index-1,-2]
        elif edge == 2:
            if edge_dir[2] == 1:
                return interpolant[0,index],interpolant[1,index]
            else:
                return interpolant[0,M-index-1],interpolant[1,M-index-1]
        elif edge == 3:
            if edge_dir[3] == 1:
                return interpolant[-1,index],interpolant[-2,index]
            else:
                return interpolant[-1,M-index-1],interpolant[-2,M-index-1]


# ----------------------------------------------------------------------
#                Reference Axis Handling
# ----------------------------------------------------------------------
            
    def addRefAxis(self,surf_ids,*args,**kwargs):
        '''Add surf_ids surfacs to a new reference axis defined by X and
        rot with nsection values'''
        mpiPrint('Adding ref axis...',self.NO_PRINT)
        ra = ref_axis(surf_ids,self.surfs,self.topo,*args,**kwargs)

        self.ref_axis.append(ra)
            
   

 #    def getL_surfs_index(self,isurf):
#         '''Return the index of l_surfs for surface isurf'''
#         for i in xrange(len(self.l_surfs)):
#             for j in xrange(len(self.l_surfs[i])):
#                 if isurf == self.l_surfs[i][j]:
#                     return i
#                 # end if
#             # end for
#         # end for
        
#         return None

  
# ----------------------------------------------------------------------
#                Update and Derivative Functions
# ----------------------------------------------------------------------

    def _updateCoef(self,local=True):
        '''update the coefficents on the pyGeo update'''
        
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
       
        if local:
            # fourth, update the coefficients (from normal DV changes)        
            for i in xrange(len(self.DV_listNormal)):
                surface = self.surfs[self.DV_listNormal[i].surface_id]
                self.coef = self.DV_listNormal[i](surface,self.coef)
            # end for

            # fifth: update the coefficient from local DV changes
            
            for i in xrange(len(self.DV_listLocal)):
                self.coef = self.DV_listLocal[i](self.coef)
            # end for
        # end if

        return
         
    def update(self):
        '''Run the update coefficients command and then set the control
        points'''
        self._updateCoef(local=True)
        self._updateSurfaceCoef()
        return

    def _updateSurfaceCoef(self):
        '''Copy the pyGeo list of control points back to the surfaces'''
        
        for ii in xrange(len(self.coef)):
            for jj in xrange(len(self.topo.g_index[ii])):
                isurf = self.topo.g_index[ii][jj][0]
                i     = self.topo.g_index[ii][jj][1]
                j     = self.topo.g_index[ii][jj][2]
                self.surfs[isurf].coef[i,j] = self.coef[ii].astype('d')
            # end for
        # end for
        return

    def getSizes( self ):
        '''
        Get the sizes:
        - The number of global design variables
        - The number of normal design variables
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
        
        for i in xrange(len(self.DV_listNormal)): # Normal Variables
            N += self.DV_listLocal[i].nVal
        # end for
                
        NdvNormal = N-NdvGlobal

        for i in xrange(len(self.DV_listLocal)): # Local Variables
            N += self.DV_listLocal[i].nVal*3
        # end for
                
        NdvLocal = N-(NdvNormal+NdvGlobal)

        return NdvGlobal, NdvNormal, NdvLocal, Nctl


    def _initdCoefdx( self ):
        '''
        Allocate the space for dCoefdx and perform some setup        
        '''

        NdvGlobal, NdvNormal, NdvLocal, Nctl = self.getSizes()
        Ndv = NdvGlobal + NdvNormal + NdvLocal
        
        if USE_PETSC:
            dCoefdx = PETSc.Mat()
            
            # We know the row filling factor: Its (exactly) nGlobal + 3            
            if PETSC_MAJOR_VERSION == 1:
                dCoefdx.createAIJ([Nctl*3,Ndv],nnz=NdvGlobal+3,comm=PETSc.COMM_SELF)
            elif PETSC_MAJOR_VERSION == 0:
                dCoefdx.createSeqAIJ([Nctl*3,Ndv],nz=NdvGlobal+3)
            else:
                print 'Error: PETSC_MAJOR_VERSION = %d is not supported'%(PETSC_MAJOR_VERSION)
                sys.exit(1)
            # end if
        else:
            dCoefdx = zeros((Nctl*3,Ndv))
        # end if

        return dCoefdx
        
    def calcCtlDeriv(self):

        '''This function runs the complex step method over the design variable
        and generates a (sparse) jacobian of the control pt
        derivatives wrt to the design variables'''

        if self.dCoefdx == None:
            self.dCoefdx = self._initdCoefdx()
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
                    self.dCoefdx[:,col_counter] = imag(self.coef.flatten())/1e-40
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
        
        # The next step is go to over all the NORMAL and LOCAL variables,
        # compute the surface normal
        
        for idv in xrange(len(self.DV_listNormal)): 
            surface = self.surfs[self.DV_listNormal[idv].surface_id]
            normals = self.DV_listNormal[idv].getNormals(\
                surface,self.coef.astype('d'))

            # Normals is the length of local dv on this surface
            for i in xrange(self.DV_listNormal[idv].nVal):
                index = 3*self.DV_listNormal[idv].coef_list[i]
                self.dCoefdx[index:index+3,col_counter] = normals[i,:]
                col_counter += 1
            # end for
        # end for

        for idv in xrange(len(self.DV_listLocal)):
            for i in xrange(self.DV_listLocal[idv].nVal):
                for j in xrange(3):
                    index = 3*self.DV_listLocal[idv].coef_list[i]
                    self.dCoefdx[index+j,col_counter] = 1.0
                    col_counter += 1
                # end for
            # end for
        # end for
            
        if USE_PETSC:
            self.dCoefdx.assemblyBegin()
            self.dCoefdx.assemblyEnd()
        # end if 

        return

    def compute_dPtdx( self ):
        '''
        Compute the product of the derivative of the surface points w.r.t.
        the control points and the derivative of the control points w.r.t.
        the design variables. This gives the derivative of the surface points
        w.r.t. the design variables: a Jacobian matrix.
        '''
        
        # Now Do the Try the matrix multiplication
        
        # Now Do the matrix multiplication
        if USE_PETSC:
            if self.dPtdCoef:
                if self.dPtdx == None:
                    self.dPtdx = PETSc.Mat()
                # end
                self.dPtdCoef.matMult(self.dCoefdx,result=self.dPtdx)
            # end
        else:
            if self.dPtdCoef:
                self.dPtdx = dot(self.dPtdCoef,self.dCoefdx)
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
                    coef_list.append(self.topo.l_index[surf][i,j])
                # end for
            # end for
        else:
            # Use the point select class to get the indicies
            coef_list = point_select.getControlPoints(\
                self.surfs[surf],isurf,coef_list,l_index)
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
                dv_name,lower,upper,surf,coef_list,self.topo.g_index))
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
                    coef_list.append(self.topo.l_index[surf][i,j])
                # end for
            # end for
        else:
            # Use the bounding box to find the appropriate indicies
            coef_list = point_select.getControlPoints(\
                self.surfs[surf],isurf,coef_list,l_index)
        # end if
        
        # Now, we have the list of the conrol points that we would
        # LIKE to add to this dv group. However, some may already be
        # specified in other normal or local dv groups. 

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
                dv_name,lower,upper,surf,coef_list,self.topo.g_index))
        self.DV_namesLocal[dv_name] = len(self.DV_listLocal)-1
        
        return


    def addGeoDVGlobal(self,dv_name,value,lower,upper,function,useit=True):
        '''Add a global design variable'''
        self.DV_listGlobal.append(geoDVGlobal(\
                dv_name,value,lower,upper,function,useit))
        self.DV_namesGlobal[dv_name]=len(self.DV_listGlobal)-1
        return

    def addGeoDVs( self, opt_prob, dvNum = 0 ):
        '''Add the pyGeo variables to pyOpt'''
        
        # Add the global, normal and local design variables
        for dvList in [ self.DV_listGlobal, self.DV_listNormal, self.DV_listLocal ]:
            for n in dvList:
                if n.nVal > 1:
                    opt_prob.addVarGroup( n.name, n.nVal, 'c', value = real(n.value), lower=n.lower, upper = n.upper )
                else:
                    opt_prob.addVar( n.name, 'c', value = real(n.value), lower=n.lower, upper = n.upper )
                # end
            # end
        # end

        return        

    def setGeoDVs( self, x, dvNum = 0 ):
        '''
        Given the value of x, set all the internal design variables.
        The surface control points are not updated.
        '''

        # Set the global, normal and local design variables
        for dvList in [ self.DV_listGlobal, self.DV_listNormal, self.DV_listLocal ]:
            for n in dvList:
                if n.nVal == 1:
                    n.value = x[dvNum]
                else:
                    n.value[:] = x[dvNum:(dvNum+n.nVal)]
                # end
                dvNum += n.nVal
            # end
        # end
        return
    
# ----------------------------------------------------------------------
#                   Surface Writing Output Functions
# ----------------------------------------------------------------------

    def writeTecplot(self,file_name,orig=False,surfs=True,coef=True,
                     edges=False,ref_axis=False,links=False,
                     directions=False,surf_labels=False,edge_labels=False,size=None,
                     node_labels=False):

        '''Write the pyGeo Object to Tecplot'''

        # Open File and output header
        #if MPI.Comm.Get_rank( MPI.WORLD ) != 0:
        if MPI.Comm.Get_rank(MPI.COMM_WORLD) != 0: 
            return
        # end if
        
        mpiPrint('Writing Tecplot file: %s '%(file_name),self.NO_PRINT)

        f = open(file_name,'w')
        f.write ('VARIABLES = "X", "Y","Z"\n')

        # --------------------------------------
        #    Write out the Interpolated Surfaces
        # --------------------------------------
        
        if surfs == True:
            for isurf in xrange(self.nSurf):
                self.surfs[isurf]._writeTecplotSurface(f,size=size)

        # -------------------------------
        #    Write out the Control Points
        # -------------------------------
        
        if coef == True:
            for isurf in xrange(self.nSurf):
                self.surfs[isurf]._writeTecplotCoef(f)

        # ----------------------------------
        #    Write out the Original Data
        # ----------------------------------
        
        if orig == True:
            for isurf in xrange(self.nSurf):
                self.surfs[isurf]._writeTecplotOrigData(f)
        # ----------------------
        #    Write out the edges
        # ----------------------

        # We also want to output edge continuity for visualization
#         if self.con and edges==True:
#             counter = 1
#             for i in xrange(len(self.con)): #Output Simple Edges (no continuity)
#                 if self.con[i].cont == 0 and self.con[i].type == 1:
#                     surf = self.con[i].f1
#                     edge = self.con[i].e1
#                     zone_name = 'simple_edge%d'%(counter)
#                     counter += 1
#                     self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
#                 # end if
#             # end for

#             for i in xrange(len(self.con)): #Output Continuity edges
#                 if self.con[i].cont == 1 and self.con[i].type == 1:
#                     surf = self.con[i].f1
#                     edge = self.con[i].e1
#                     zone_name = 'continuity_edge%d'%(counter)
#                     counter += 1
#                     self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
#                 # end if
#             # end for

#             for i in xrange(len(self.con)): #Output Mirror (free) edges
#                 if self.con[i].type == 0: #output the edge
#                     surf = self.con[i].f1
#                     edge = self.con[i].e1
#                     zone_name = 'mirror_edge%d'%(counter)
#                     counter += 1
#                     self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
#                 # end if
#             # end for
#         # end if

        # ---------------------
        #    Write out Ref Axis
        # ---------------------

        if len(self.ref_axis)>0 and ref_axis==True:
            for r in xrange(len(self.ref_axis)):
                #axis_name = 'ref_axis%d'%(r)
                axis_name = 'axis_copy.dat'
                self.ref_axis[r].writeTecplot(f,axis_name)
            # end for
        # end if

        # ------------------
        #    Write out Links
        # ------------------

        if len(self.ref_axis)>0 and links==True:
            for r in xrange(len(self.ref_axis)):
                self.writeTecplotLinks(f,self.ref_axis[r])
            # end for
        # end if
              
        # -----------------------------------
        #    Write out The Surface Directions
        # -----------------------------------

        if directions == True:
            for isurf in xrange(self.nSurf):
                self.surfs[isurf]._writeDirections(f,isurf)
            # end for
        # end if

        # ---------------------------------------------
        #    Write out The Surface,Edge and Node Labels
        # ---------------------------------------------
        if surf_labels == True:
            # Split the filename off
            (dirName,fileName) = os.path.split(file_name)
            (fileBaseName, fileExtension)=os.path.splitext(fileName)
            label_filename = dirName+'/'+fileBaseName+'.surf_labels.dat'
            f2 = open(label_filename,'w')
            for isurf in xrange(self.nSurf):
                midu = floor(self.surfs[isurf].Nctlu/2)
                midv = floor(self.surfs[isurf].Nctlv/2)
                text_string = 'TEXT CS=GRID3D, X=%f,Y=%f,Z=%f,ZN=%d, T=\"S%d\"\n'%(self.surfs[isurf].coef[midu,midv,0],self.surfs[isurf].coef[midu,midv,1], self.surfs[isurf].coef[midu,midv,2],isurf+1,isurf)
                f2.write('%s'%(text_string))
            # end for 
            f2.close()
        # end if 

        if edge_labels == True:
            # Split the filename off
            (dirName,fileName) = os.path.split(file_name)
            (fileBaseName, fileExtension)=os.path.splitext(fileName)
            label_filename = dirName+'/'+fileBaseName+'edge_labels.dat'
            f2 = open(label_filename,'w')
            for iedge in xrange(self.topo.nEdge):
                surfaces =  self.topo.getSurfaceFromEdge(iedge)
                pt = self.surfs[surfaces[0][0]].edge_curves[surfaces[0][1]](0.5)
                text_string = 'TEXT CS=GRID3D X=%f,Y=%f,Z=%f,T=\"E%d\"\n'%(pt[0],pt[1],pt[2],iedge)
                f2.write('%s'%(text_string))
            # end for
            f2.close()
        # end if
        
        if node_labels == True:
            # First we need to figure out where the corners actually *are*
            n_nodes = len(unique(self.topo.node_link.flatten()))
            node_coord = zeros((n_nodes,3))
            for i in xrange(n_nodes):
                # Try to find node i
                for isurf in xrange(self.nSurf):
                    if self.topo.node_link[isurf][0] == i:
                        coordinate = self.surfs[isurf].getValueCorner(0)
                        break
                    elif self.topo.node_link[isurf][1] == i:
                        coordinate = self.surfs[isurf].getValueCorner(1)
                        break
                    elif self.topo.node_link[isurf][2] == i:
                        coordinate = self.surfs[isurf].getValueCorner(2)
                        break
                    elif self.topo.node_link[isurf][3] == i:
                        coordinate = self.surfs[isurf].getValueCorner(3)
                        break
                # end for
                node_coord[i] = coordinate
            # end for
            # Split the filename off

            (dirName,fileName) = os.path.split(file_name)
            (fileBaseName, fileExtension)=os.path.splitext(fileName)
            label_filename = dirName+'/'+fileBaseName+'.node_labels.dat'
            f2 = open(label_filename,'w')

            for i in xrange(n_nodes):
                text_string = 'TEXT CS=GRID3D, X=%f,Y=%f,Z=%f,T=\"n%d\"\n'%(
                    node_coord[i][0],node_coord[i][1],node_coord[i][2],i)
                f2.write('%s'%(text_string))
            # end for 
            f2.close()

        f.close()
        
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
        f.write('S%7dG%7dD%7dP%7d%40sT%6s1\n'%(1,4,Dcount-1,counter-1,' ',' '))
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
        
        mpiPrint('Attaching a discrete surface to the Geometry Object...',self.NO_PRINT)

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
            
            u = linspace(self.surfs[isurf].umin,self.surfs[isurf].umax,Nu)
            v = linspace(self.surfs[isurf].vmin,self.surfs[isurf].vmax,Nv)
            [U,V] = meshgrid(u,v)

            temp = self.surfs[isurf].getValue(U,V)
            for idim in xrange(self.surfs[isurf].nDim):
                xyz[idim,n*Nu*Nv:(n+1)*Nu*Nv]= temp[:,:,idim].flatten()
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
        mpiPrint('  -> Running CSM_PRE...',self.NO_PRINT)
        [dist,nearest_elem,uvw,base_coord,weightt,weightr] = \
            csm_pre.csm_pre(coordinates,xyz,conn,elemtype)

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

        # Now we can do a secondary newton search
        diff = zeros(nPts)
        for i in xrange(nPts):
            uv[i,0],uv[i,1],D = self.surfs[patchID[i]].projectPoint(coordinates[:,i],u=uv[i,0],v=uv[i,1])
            diff[i] = D[0]**2 + D[1]**2 + D[2] **2
        # Release the tree - otherwise fortran will get upset
        csm_pre.release_adt()
        mpiPrint('  -> Done Surface Attachment',self.NO_PRINT)
        mpiPrint('  -> RMS Error : %f'%(sqrt(sum(diff)/nPts)),self.NO_PRINT)
        mpiPrint('  -> Max Error : %f'%(sqrt(max(diff))),self.NO_PRINT)

        return dist,patchID,uv
  
    def writeAttachedSurface(self,file_name,patchID,uv):
        '''Write the patchID and uv coordinates for a set of points to a
        file. This allows the user to reload the points and (possibly)
        (slightly) modify the underlying geometry (but NOT topology)'''
        mpiPrint('Writing Attached Surface...',self.NO_PRINT)
        f = open(file_name,'w')
        for icoord in xrange(len(patchID)):
            f.write('%d,%20.16g,%20.16g\n'%(patchID[icoord],uv[icoord,0],uv[icoord,0]))
        # end if
        f.close()

        return

    def readAttachedSurface(self,file_name):
        '''Read the patchID and uv coordinates for a set of points from a
        file. This allows the user to reload the points and (possibly)
        (slightly) modify the underlying geometry (but NOT
        topology)'''
        mpiPrint('Reading Attached Surface...',self.NO_PRINT)
        f = open(file_name,'r')
        patchID = []
        uv = []
        for line in f:
            aux = string.split(line,',')
            patchID.append(int(aux[0]))
            uv.append([float(aux[1]),float(aux[2])])
        # end if
        f.close()
        patchID = array(patchID)
        uv = array(uv)
        
        return patchID,uv

    def _initdPtdCoef( self, M, Nctl ):
        
        # We know the row filling factor: Its (no more) than ku*kv
        # control points for each control point. Since we don't
        # use more than k=4 we will set at 16
        
        if USE_PETSC:
            dPtdCoef = PETSc.Mat()
            if PETSC_MAJOR_VERSION == 1:
                dPtdCoef.createAIJ([M*3,Nctl*3],nnz=16*3,comm=PETSc.COMM_SELF)
            elif PETSC_MAJOR_VERSION == 0:
                dPtdCoef.createSeqAIJ([M*3,Nctl*3],nz=16*3)
            else:
                print 'Error: PETSC_MAJOR_VERSION = %d is not supported'%(PETSC_MAJOR_VERSION)
                sys.exit(1)
            # end if
        else:
            dPtdCoef = zeros((M*3,Nctl*3))
        # end if

        return dPtdCoef

    def calcSurfaceDerivative(self,patchID,uv,indices=None,dPtdCoef=None):
        '''Calculate the (fixed) surface derivative of a discrete set of ponits'''

        mpiPrint('Calculating Surface Derivative for %d Points...'%(len(patchID)),self.NO_PRINT)
        timeA = time.time()
        
        if USE_PETSC:
            PETSC_INSERT_MODE = PETSc.InsertMode.ADD_VALUES
        # end if
        # If no matrix is provided, use self.dPtdCoef
        if dPtdCoef == None:
            dPtdCoef = self.dPtdCoef
        # end

        if indices == None:
            indices = arange(len(patchID),'intc')
        # end

        if dPtdCoef == None:
            # Calculate the size Ncoef_free x Ndesign Variables            
            M = len(patchID)
            Nctl = self.Ncoef

            dPtdCoef = self._initdPtdCoef( M, Nctl )
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

                    index = 3*self.topo.l_index[patchID[i]][u_list[ii],v_list[jj]]
                    if USE_PETSC:
                        dPtdCoef.setValue( 3*indices[i]  , index  ,x,PETSC_INSERT_MODE)
                        dPtdCoef.setValue( 3*indices[i]+1, index+1,x,PETSC_INSERT_MODE)
                        dPtdCoef.setValue( 3*indices[i]+2, index+2,x,PETSC_INSERT_MODE)
                    else:
                        dPtdCoef[3*indices[i]    ,index    ] += x
                        dPtdCoef[3*indices[i] + 1,index + 1] += x
                        dPtdCoef[3*indices[i] + 2,index + 2] += x
                    # end if
                # end for
            # end for
        # end for 
        
        # Assemble the (Constant) dPtdCoef
        if USE_PETSC:
            dPtdCoef.assemblyBegin()
            dPtdCoef.assemblyEnd()
        # end if

        self.dPtdCoef = dPtdCoef # Make sure we're dealing with the same matrix

        mpiPrint('Finished Surface Derivative in %5.3f seconds'%(time.time()-timeA),self.NO_PRINT)

        return

    def createTACSGeo(self,surface_list=None):
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

        if surface_list == None:
            surface_list = arange(self.nSurf)
        # end if

        # Calculate the Number of global design variables
        N = 0
        for i in xrange(len(self.DV_listGlobal)): #Global Variables
            if self.DV_listGlobal[i].useit:
                N += self.DV_listGlobal[i].nVal
            # end if
        # end for

        gdvs = arange(N,dtype='intc')
      
        global_geo = elems.GlobalGeo( gdvs, self.petsc_coef, self.dCoefdx )
      
        # For each dv object, number the normal variables
        normalDVs = []
        for normal in self.DV_listNormal:
            normalDVs.append( arange(N,N+normal.nVal,dtype='intc') )
            N += normal.nVal
        # end

        # For each dv object, number all three coordinates
        localDVs = []
        for local in self.DV_listLocal:
            localDVs.append( arange(N,N+3*local.nVal,dtype='intc') )
            N += 3*local.nVal
        # end

        # Create the list of local dvs for each surface patch
        surfDVs = []
        for i in xrange(self.nSurf):
            surfDVs.append(None)
        # end
        
        for i in xrange(len(self.DV_listNormal)):
            sid = self.DV_listNormal[i].surface_id
            if ( surfDVs[sid] == None ):
                surfDVs[sid] = normalDVs[i]
            else:
                hstack( surfDVs[sid], normalDVs[i] )
            # end
        # end

        for i in xrange(len(self.DV_listLocal)):
            sid = self.DV_listLocal[i].surface_id
            if ( surfDVs[sid] == None ):
                surfDVs[sid] = localDVs[i]
            else:
                hstack( surfDVs[sid], localDVs[i] )
            # end
        # end        

        # Go through and add local objects for each design variable
        def convert( isurf, ldvs ):
            if ldvs == None:
                ldvs = []
            # end

            return elems.SplineGeo( int(self.surfs[isurf].ku),
                                    int(self.surfs[isurf].kv),
                                    self.surfs[isurf].tu, self.surfs[isurf].tv,
                                    self.surfs[isurf].coef[:,:,0], 
                                    self.surfs[isurf].coef[:,:,1], 
                                    self.surfs[isurf].coef[:,:,2], 
                                    global_geo, ldvs, self.topo.l_index[isurf].astype('intc') )
        # end

        tacs_surfs = []
        for isurf in surface_list:
            tacs_surfs.append( convert(isurf, surfDVs[isurf] ) )
        # end
     
        return global_geo, tacs_surfs

class ref_axis(object):

    def __init__(self,surf_ids,surfs,topo,*args,**kwargs):

        ''' Create a generic reference axis. This object bascally defines a
        set of points in space (x,y,z) each with three rotations
        associated with it. The purpose of the ref_axis is to link
        groups of b-spline controls points together such that
        high-level planform-type variables can be used as design
        variables
        
        Input:
        
        The only non-keyword input is surf_ids: a list of surfaces for 
        This ref axis
        
        # The spatial data supplied as 
        x = x_coordiantes
        y = y_coordiantes
        z = z_coordiantes  OR
        X = array((N,3)) with all x-y-z coordinates

        # Rotation data is supplied as
        rot_x = rot_x
        rot_y = rot_y
        rot_z = rot_z
        
        Note: Rotations are performed in the order: Z-Y-X
        '''

        # Extract Some information from kwargs:
        if 'X' in kwargs:
            X = kwargs['X']
        else:
            X = vstack([kwargs['x'],kwargs['y'],kwargs['z']]).T
        # end if

        if 'rot' in kwargs:
            self.rot_x = kwargs['rot'][:,0]
            self.rot_y = kwargs['rot'][:,1]
            self.rot_z = kwargs['rot'][:,2]
        else:
            self.rot_x = kwargs['rot_x']
            self.rot_y = kwargs['rot_y']
            self.rot_z = kwargs['rot_z']
        # end if

        # Create the splines for the axis
 
        self.xs    = pySpline.curve('interpolate',X=X,k=2,no_print=True)
        self.rotxs = pySpline.curve('interpolate',x=self.rot_x,s=self.xs.s,k=2,no_print=True)
        self.rotys = pySpline.curve('interpolate',x=self.rot_x,s=self.xs.s,k=2,no_print=True)
        self.rotzs = pySpline.curve('interpolate',x=self.rot_x,s=self.xs.s,k=2,no_print=True)

        self.scale = ones(self.xs.Nctl)
        self.scales = pySpline.curve('interpolate',x=self.scale,s=self.xs.s,k=2,no_print=True)
        self.links_s = []
        self.links_x = []
        self.con_type = None

        self.base_point = self.xs(0)
        
        self.base_point_s = None
        self.base_point_D = None

        self.end_point   = self.xs(1)
        self.end_point_s = None
        self.end_point_D = None

        # Values are stored wrt the base point
        self.x = self.xs.coef-self.base_point

        # Deep copy the x,rot and scale for design variable reference
        self.x0 = copy.deepcopy(self.x)
        self.rotx0 = copy.deepcopy(self.rotxs.coef)
        self.roty0 = copy.deepcopy(self.rotys.coef)
        self.rotz0 = copy.deepcopy(self.rotzs.coef)
        
        self.scale0 = copy.deepcopy(self.scale)

        # Now determine what control points will be associated with this axis
        coef_list = []
        if not 'point_select' in kwargs: # No point_select->Assume full surface
            for isurf in surf_ids:
                coef_list.extend(topo.l_index[isurf].flatten())
            # end for
        # end if

        else:   # We have a point selection class passed in
            for isurf in surf_ids:
                coef_list.extend(kwargs['point_select'].getControlPoints(\
                        surfs[isurf],isurf,coef_list,topo.l_index))
            # end for
        # end if

        # Now parse out duplicates and sort
        coef_list = unique(coef_list) #unique is in geo_utils
        coef_list.sort()

        # Now we must determine how the surfaces are oriented wrt the axis
        # We also must determine (based on the design groups) how to attach the axis

        # Algorithim Description:
        # 1. Do until all surfaces accounted for:
        # 2.    -> Take the first surface and determine its orientation wrt the axis
        # 3.    -> The the perpendicualar design group attach in the same manner
        surf_ids_copy =copy.copy(surf_ids)
        reordered_coef_list = []
        global_counter = 0 
        while len(surf_ids_copy) > 0:
            isurf = surf_ids_copy.pop(0)
            dir_type = directionAlongSurface(surfs[isurf],self.xs)

            if isurf == 16:
                dir_type = 0
            
            surf_list = []
            dir_list = []
            surf_list.append(isurf)
            if dir_type in [0,1]:
                dir_list.append(0) 
                dg_parallel = topo.edges[topo.edge_link[isurf][0]].dg
            else:
                dir_list.append(1)
                dg_parallel = topo.edges[topo.edge_link[isurf][2]].dg
            # Find all other surfaces/edges with this design group
            for isurf in set(surf_ids_copy).difference(set(surf_list)):
                #print 'isurf,dg:',isurf,topo.edges[topo.edge_link[isurf][0]].dg,topo.edges[topo.edge_link[isurf][2]].dg
                if topo.edges[topo.edge_link[isurf][0]].dg == dg_parallel:
                    surf_ids_copy.remove(isurf)
                    surf_list.append(isurf)
                    dir_list.append(0)
                elif topo.edges[topo.edge_link[isurf][2]].dg == dg_parallel:
                    surf_ids_copy.remove(isurf)
                    surf_list.append(isurf)
                    dir_list.append(1)
            # end for
        # end for
            # Now we can simply attach all the surfaces in surf list according to the directions 
            
            # N is the number of parallel control points
            if dir_list[0] == 0:
                N = surfs[surf_list[0]].Nctlu
            else:
                N = surfs[surf_list[0]].Nctlv
            # end if
                
            s = zeros(N)

            for i in xrange(N):
                section_coef_list = []
                for j in xrange(len(surf_list)):
                    isurf = surf_list[j]
                    if dir_list[j] == 0:
                        section_coef_list.extend(surfs[isurf].coef[i,:])
                    else:
                        section_coef_list.extend(surfs[isurf].coef[:,i])
                    # end if
                # end if
                # Average coefficients
                pt = average(section_coef_list,axis=0)
                # This effectively averages the coefficients
                s[i],D = self.xs.projectPoint(pt)
            # end if

            # Now we can attach these with links if they are in coef_list
            for i in xrange(N):
                for j in xrange(len(surf_list)):
                    isurf = surf_list[j]
                    if dir_list[j] == 0:
                        for k in xrange(surfs[isurf].Nctlv):
                            global_index = topo.l_index[isurf][i,k]
                            if global_index in coef_list:
                                D = surfs[isurf].coef[i,k] - self.xs(s[i])
                                M = self.getRotMatrixGlobalToLocal(s[i])
                                D = dot(M,D) #Rotate to local frame
                                self.links_s.append(s[i])
                                self.links_x.append(D)
                                reordered_coef_list.append(global_index)
                            # end if
                        # end for
                    elif dir_list[j] == 1:
                        for k in xrange(surfs[isurf].Nctlu):
                            global_index = topo.l_index[isurf][k,i]
                            if global_index in coef_list:
                                D = surfs[isurf].coef[k,i] - self.xs(s[i])
                                M = self.getRotMatrixGlobalToLocal(s[i])
                                D = dot(M,D) #Rotate to local frame
                                self.links_s.append(s[i])
                                self.links_x.append(D)
                                reordered_coef_list.append(global_index)
                            # end if
                        # end for
                    # end if
                # end for
            # end for
        # end for
        self.coef_list = reordered_coef_list
        self.surf_ids  = surf_ids
        
    def update(self):
        
        self.xs.coef = self.base_point+self.x
        self.rotxs.coef = self.rot_x
        self.rotys.coef = self.rot_y
        self.rotzs.coef = self.rot_z

        self.scales.coef = self.scale

        if self.con_type == 'full':
            self.xs.coef[-1,:] = self.end_point
        # end if
        
        return
       
    def writeTecplot(self,handle,axis_name):
        '''Write the ref axis to the open file handle'''
        N = len(self.xs.s)
        handle.write('Zone T=%s I=%d\n'%(axis_name,N))
        values = self.xs.getValue(self.xs.s)
        for i in xrange(N):
            handle.write('%f %f %f \n'%(values[i,0],values[i,1],values[i,2]))
        # end for

        return

    def getRotMatrixGlobalToLocal(self,s):
        
        '''Return the rotation matrix to convert vector from global to
        local frames'''
        return     dot(rotyM(self.rotys(s)[0]),dot(rotxM(self.rotxs(s)[0]),\
                                                    rotzM(self.rotzs(s)[0])))
    
    def getRotMatrixLocalToGlobal(self,s):
        
        '''Return the rotation matrix to convert vector from global to
        local frames'''
        return inv(dot(rotyM(self.rotys(s)[0]),dot(rotxM(self.rotxs(s)[0]),\
                                                    rotzM(self.rotzs(s)[0]))))


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
                self.ref_axis[axis2].xs.getValue(1.0))

            M = self.ref_axis[axis1].getRotMatrixGlobalToLocal(s)
            D = dot(M,D)

            self.ref_axis[axis2].end_point_s = s
            self.ref_axis[axis2].end_point_D = D
            
        # end if
            
        self.ref_axis_con.append([axis1,axis2,con_type])

        return

class geoDVGlobal(object):
     
    def __init__(self,dv_name,value,lower,upper,function,useit=True):
        
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
        
        if getattr(self.value,'__iter__',False):
            self.nVal = len(value)
        else:
            self.nVal = 1
        # end

        self.lower    = lower
        self.upper    = upper
        self.function = function
        self.useit    = useit
        return

    def __call__(self,ref_axis):

        '''When the object is called, actually apply the function'''
        # Run the user-supplied function
        return self.function(self.value,ref_axis)
        

class geoDVNormal(object):
     
    def __init__(self,dv_name,lower,upper,surface_id,coef_list,topo):
        
        '''Create a set of gemoetric design variables which change the shape
        of surface, surface_id

        Input:
        
        dv_name: Design variable name. Must be unique. Can be used
        to set pyOpt variables directly

        lower: Lower bound for the variable. Again for setting in
        pyOpt

        upper: Upper bound for the variable.

        surface_id: The surface these design variables apply to 

        coef_list: The list of (global) indicies for thes design variables

        topo: The topology for the geometry object

        Note: Value is NOT specified, value will ALWAYS be initialized to 0

        '''

        self.nVal = len(coef_list)
        self.value = zeros(self.nVal,'D')
        self.name = dv_name
        self.lower = lower
        self.upper = upper
        self.surface_id = surface_id
        self.coef_list = coef_list
        self.l_index   = topo.l_index[surface_id]
        # We also need to know what local surface i,j index is for
        # each point in the coef_list since we need to know the
        # position on the surface to get the normal. That's why we
        # passed in the global_coef list so we can figure it out
        
        self.local_coef_index = zeros((self.nVal,2),'intc')
        
        for icoef in xrange(self.nVal):
            current_point = g_index[coef_list[icoef]]
            # Since the local DV only have driving control points, the
            # i,j index coorsponding to the first entryin the
            # global_coef list is the one we want
            self.local_coef_index[icoef,:] = topo.g_index[coef_list[icoef]][0][1:3]
        # end for
        return

    def __call__(self,surface,coef):

        '''When the object is called, apply the design variable values to the
        surface'''

        coef = pySpline.pyspline_cs.updatesurfacepoints(\
            coef,self.local_coef_index,self.coef_list,self.value,\
                self.l_index,surface.tu,surface.tv,surface.ku,surface.kv)

        return coef

    def getNormals(self,surf,coef):
        normals = pySpline.pyspline_real.getctlnormals(\
            coef,self.local_coef_index,self.coef_list,\
                self.l_indexs,surf.tu,surf.tv,surf.ku,surf.kv)
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
        self.value = zeros((3*self.nVal),'D')
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

    def __call__(self,coef):

        '''When the object is called, apply the design variable values to 
        coefficients'''
        
        for i in xrange(self.nVal):
            coef[self.coef_list[i]] += self.value[3*i:3*i+3]
        # end for
      
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
   

#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'

# DEPRECATED


# # ----------------------------------------------------------------------
# #                        Surface Fitting Functions
# # ----------------------------------------------------------------------

#     def fitSurfaces(self):
#         '''This function does a lms fit on all the surfaces respecting
#         the stitched edges as well as the continuity constraints'''

#         nCtl = len(self.coef)

#         sizes = []
#         for isurf in xrange(self.nSurf):
#             sizes.append([self.surfs[isurf].Nu,self.surfs[isurf].Nv])
#         # end for
        
#         # Get the Globaling number of the original data
#         nPts, g_index,l_index = self.calcGlobalNumbering(sizes)
        
#         nRows,nCols,dv_link = self._initJacobian(nPts)

#         if not self.NO_PRINT:
#             print '------------- Fitting Surfaces Globally ------------------'
#             print 'nRows (Surface Points):',nRows
#             print 'nCols (Degrees of Freedom):',nCols

#         if USE_PETSC:
#             pts = PETSc.Vec().createSeq(nRows)
#             temp= PETSc.Vec().createSeq(nRows)
#             X = PETSc.Vec().createSeq(nCols)
#             X_cur = PETSc.Vec().createSeq(nCols)
#         else:
#             pts = zeros(nRows) 
#             temp = None
#             X = zeros(nCols)
#             X_cur = zeros(nCols)
#         # end if 
      
#         # Fill up the 'X' with the best curent solution guess
#         for i in xrange(len(dv_link)):
#             if len(dv_link[i][0]) == 1: # Its regular
#                 X[dv_link[i][0][0]:dv_link[i][0][0]+3] = self.coef[i].astype('d')
#             else:
#                 X[dv_link[i][0][0]] = 0.5
#                 dv_index = dv_link[i][0][0]
#                 n1_index = dv_link[i][0][1] # node one side of constrined node
#                 n2_index = dv_link[i][0][2] # node other side of constrained node
#                 self.coef[i] = self.coef[n1_index]*(1-X[dv_index]) + X[dv_index]*self.coef[n2_index]
#             # end if
#         # end for
        
#         if USE_PETSC:
#             X.copy(X_cur)
#         else:
#             X_cur = X.copy()
#         # end if

#         # Now Fill up the RHS point list
#         for ii in xrange(len(g_index)):
#             isurf = g_index[ii][0][0]
#             i = g_index[ii][0][1]
#             j = g_index[ii][0][2]
#             pts[3*ii:3*ii+3] = self.surfs[isurf].X[i,j]
#         # end for
#         rhs = pts
#         if not self.NO_PRINT:
#             print 'LMS solving...'
#         nIter = 6
#         for iter in xrange(nIter):
#             # Assemble the Jacobian
#             nRows,nCols,dv_link = self._initJacobian(nPts)
#             for ii in xrange(nPts):
#                 surfID = g_index[ii][0][0]
#                 i      = g_index[ii][0][1]
#                 j      = g_index[ii][0][2]

#                 u = self.surfs[surfID].u[i]
#                 v = self.surfs[surfID].v[j]

#                 ku = self.surfs[surfID].ku
#                 kv = self.surfs[surfID].kv

#                 ileftu, mflagu = self.surfs[surfID].pyspline.intrv(\
#                     self.surfs[surfID].tu,u,1)
#                 ileftv, mflagv = self.surfs[surfID].pyspline.intrv(\
#                     self.surfs[surfID].tv,v,1)

#                 if mflagu == 0: # Its Inside so everything is ok
#                     u_list = [ileftu-ku,ileftu-ku+1,ileftu-ku+2,ileftu-ku+3]
#                 if mflagu == 1: # Its at the right end so just need last one
#                     u_list = [ileftu-ku-1]

#                 if mflagv == 0: # Its Inside so everything is ok
#                     v_list = [ileftv-kv,ileftv-kv+1,ileftv-kv+2,ileftv-kv+3]
#                 if mflagv == 1: # Its at the right end so just need last one
#                     v_list = [ileftv-kv-1]

#                 for iii in xrange(len(u_list)):
#                     for jjj in xrange(len(v_list)):
#                         # Should we need a += here??? I don't think so...
#                         x = self.surfs[surfID].calcPtDeriv(\
#                             u,v,u_list[iii],v_list[jjj])

#                         # X is the derivative of the physical point at parametric location u,v
#                         # by control point u_list[iii],v_list[jjj]

#                         global_index = self.l_index[surfID][u_list[iii],v_list[jjj]]
#                         if len(dv_link[global_index][0]) == 1:
#                             dv_index = dv_link[global_index][0][0]
#                             self._addJacobianValue(3*ii    ,dv_index    ,x)
#                             self._addJacobianValue(3*ii + 1,dv_index + 1,x)
#                             self._addJacobianValue(3*ii + 2,dv_index + 2,x)
#                         else: # its a constrained one
#                             dv_index = dv_link[global_index][0][0]
#                             n1_index = dv_link[global_index][0][1] # node one side of constrined node
#                             n2_index = dv_link[global_index][0][2] # node other side of constrained node
#                           #   print '1:',dv_index
#                             dv1 = dv_link[n1_index][0][0]
#                             dv2 = dv_link[n2_index][0][0]
                            
#                             dcoefds = -self.coef[n1_index] + self.coef[n2_index]
#                             self._addJacobianValue(3*ii    ,dv_index,x*dcoefds[0])
#                             self._addJacobianValue(3*ii + 1,dv_index,x*dcoefds[1])
#                             self._addJacobianValue(3*ii + 2,dv_index,x*dcoefds[2])

#                             # We also need to add the dependance of the other two nodes as well
#                             #print '1:',global_index
#                             dv_index = dv_link[n1_index][0][0]
#                             #print '2:',n1_index,dv_index
#                             self._addJacobianValue(3*ii    ,dv_index  ,(1-X[dv_index])*x)
#                             self._addJacobianValue(3*ii + 1,dv_index+1,(1-X[dv_index])*x)
#                             self._addJacobianValue(3*ii + 2,dv_index+2,(1-X[dv_index])*x)
                            
#                             dv_index = dv_link[n2_index][0][0]
#                             #print '3:',n2_index,dv_index
#                             self._addJacobianValue(3*ii    ,dv_index  ,X[dv_index]*x)
#                             self._addJacobianValue(3*ii + 1,dv_index+1,X[dv_index]*x)
#                             self._addJacobianValue(3*ii + 2,dv_index+2,X[dv_index]*x)

#                         # end if
#                     # end for
#                 # end for
#             # end for 
#             if iter == 0:
#                 if USE_PETSC:
#                     self.J.assemblyBegin()
#                     self.J.assemblyEnd()
#                     self.J.mult(X,temp)
#                     rhs = rhs - temp
#                 else:
#                     rhs -= dot(self.J,X)
#                 # end if
#             # end if
#             rhs,X,X_cur = self._solve(X,X_cur,rhs,temp,dv_link,iter)
#         # end for (iter)
#         return

#     def _addJacobianValue(self,i,j,value):
#         if USE_PETSC: 
#             self.J.setValue(i,j,value,PETSc.InsertMode.ADD_VALUES)
#         else:
#             self.J[i,j] += value
#         # end if

#     def _solve(self,X,X_cur,rhs,temp,dv_link,iter):
#         '''Solve for the control points'''
        

#         if USE_PETSC:

#             self.J.assemblyBegin()
#             self.J.assemblyEnd()

#             ksp = PETSc.KSP()
#             ksp.create(PETSc.COMM_WORLD)
#             ksp.getPC().setType('none')
#             ksp.setType('lsqr')
#             #ksp.setInitialGuessNonzero(True)

#             print 'Iteration   Residual'
#             def monitor(ksp, its, rnorm):
#                 if mod(its,50) == 0:
#                     print '%5d      %20.15g'%(its,rnorm)

#             ksp.setMonitor(monitor)
#             ksp.setTolerances(rtol=1e-15, atol=1e-15, divtol=100, max_it=250)

#             ksp.setOperators(self.J)
#             ksp.solve(rhs, X)
#             self.J.mult(X,temp)
#             rhs = rhs - temp

#         else:

#             X = lstsq(self.J,rhs)[0]
#             rhs -= dot(self.J,X)
#             print 'rms:',sqrt(dot(rhs,rhs))

#         # end if
#         scale = 1
#         X_cur = X_cur + X/scale

#         for icoef in xrange(len(self.coef)):
#             if len(dv_link[icoef][0]) == 1:
#                 dv_index = dv_link[icoef][0][0]
#                 self.coef[icoef,0] = (X_cur[dv_index + 0])
#                 self.coef[icoef,1] = (X_cur[dv_index + 1])
#                 self.coef[icoef,2] = (X_cur[dv_index + 2])
#             # end if
#         for icoef in xrange(len(self.coef)):
#             if len(dv_link[icoef][0]) != 1:
#                 dv_index = dv_link[icoef][0][0]
#                 n1_index = dv_link[icoef][0][1] # node one side of constrined node
#                 n2_index = dv_link[icoef][0][2] # node other side of constrained node
#                 dv1 = dv_link[n1_index][0][0]
#                 dv2 = dv_link[n2_index][0][0]
#                 #print 'Value1:',X_cur[dv_index]

#                 update0 = X[dv_index]/scale
#                 value = update0
#                 for i in xrange(25):
#                     if abs(value) > 0.1:
#                         value /= 2
#                     else:
#                         break
                
#                 # end for
#                 # We've already added update---but we really want to add value instread
#                 #print 'update0,value:',update0,value
#                 X_cur[dv_index] = X_cur[dv_index] - update0 +value
#                 value = X_cur[dv_index]
#                 #value = .5
#                 #X_cur[dv_index] = .5
#                 print 'Value2:',X_cur[dv_index]
                
#                 self.coef[icoef] = (1-value)*self.coef[n1_index] + value*(self.coef[n2_index])
              
#             # end if
#         # end for

#         return rhs,X,X_cur

#     def _initJacobian(self,Npt):
        
#         '''Initialize the Jacobian either with PETSc or with Numpy for use
#         with LAPACK'''
        
#         dv_link = [-1]*len(self.coef)
#         dv_counter = 0
#         for isurf in xrange(self.nSurf):
#             Nctlu = self.surfs[isurf].Nctlu
#             Nctlv = self.surfs[isurf].Nctlv
#             for i in xrange(Nctlu):
#                 for j in xrange(Nctlv):
#                     type,edge,node,index = indexPosition(i,j,Nctlu,Nctlv)
#                     if type == 0: # Interior
#                         dv_link[self.l_index[isurf][i,j]] = [[dv_counter]]
#                         dv_counter += 3
#                     elif type == 1: # Edge
#                         if dv_link[self.l_index[isurf][i,j]] ==-1: # Its isn't set yet
#                             # Now determine if its on a continuity edge
#                             if self.edge_list[self.edge_link[isurf][edge]].cont == 1: #its continuous
#                                 iedge = self.edge_link[isurf][edge] # index of edge of interest
#                                 surfaces = self.getSurfaceFromEdge(iedge) # Two surfaces we want

#                                 surf0 = surfaces[0][0] # First surface on this edge
#                                 edge0 = surfaces[0][1] # Edge of surface on this edge                           
#                                 surf1 = surfaces[1][0] # Second surface on this edge
#                                 edge1 = surfaces[1][1] # Edge of second surface on this edge

#                                 tindA,indB = self._getTwoIndiciesOnEdge(
#                                     self.l_index[surf0],index,edge0,self.edge_dir[surf0])

#                                 tindA,indC = self._getTwoIndiciesOnEdge(
#                                     self.l_index[surf1],index,edge1,self.edge_dir[surf1])

#                                 # indB and indC are the global indicies of the two control 
#                                 # points on either side of this node on the edge

#                                 dv_link[self.l_index[isurf][i,j]] = [[dv_counter,indB,indC]]
#                                 dv_counter += 1
#                             else: # Just add normally
#                                 dv_link[self.l_index[isurf][i,j]] = [[dv_counter]]
#                                 dv_counter += 3
#                             # end if
#                         # end if
#                     elif type == 2: # Corner
#                         if dv_link[self.l_index[isurf][i,j]] == -1: # Its not set yet
#                             # Check both possible edges
#                             edge1,edge2,index1,index2 = edgesFromNodeIndex(node,Nctlu,Nctlv)
#                             edges= [edge1,edge2]
#                             indices = [index1,index2]
#                             dv_link[self.l_index[isurf][i,j]] = []
#                             for ii in xrange(2):
#                                 if self.edge_list[self.edge_link[isurf][edges[ii]]].cont == 1:
#                                     iedge = self.edge_link[isurf][edges[ii]] # index of edge of interest
#                                     surfaces = self.getSurfaceFromEdge(iedge) # Two surfaces we want
#                                     surf0 = surfaces[0][0] # First surface on this edge
#                                     edge0 = surfaces[0][1] # Edge of surface on this edge                           
#                                     surf1 = surfaces[1][0] # Second surface on this edge
#                                     edge1 = surfaces[1][1] # Edge of second surface on this edge
                                    
#                                     tindA,indB = self._getTwoIndiciesOnEdge(
#                                         self.l_index[surf0],indices[ii],edge0,self.edge_dir[surf0])

#                                     tindA,indC = self._getTwoIndiciesOnEdge(
#                                         self.l_index[surf1],indices[ii],edge1,self.edge_dir[surf1])

#                                     # indB and indC are the global indicies of the two control 
#                                     # points on either side of this node on the edge
#                                     dv_link[self.l_index[isurf][i,j]].append([dv_counter,indB,indC])
#                                     dv_counter += 1

#                                 # end if
#                             # end for
#                             # If its STILL not set there's no continutiy
#                             if dv_link[self.l_index[isurf][i,j]] == []: # Need this check again
#                                 dv_link[self.l_index[isurf][i,j]] = [[dv_counter]]
#                                 dv_counter += 3
#                             # end if
#                     # end if (pt type)
#                 # end for (Nctlv loop)
#             # end for (Nctlu loop)
#         # end for (isurf looop)
                                
#         nRows = Npt*3
#         nCols = dv_counter

#         if USE_PETSC:
#             self.J = PETSc.Mat()
#             # We know the row filling factor: 16*3 (4 for ku by 4 for
#             # kv and 3 spatial)
#             if PETSC_MAJOR_VERSION == 1:
#                 self.J.createAIJ([nRows,nCols],nnz=16*3,comm=PETSc.COMM_SELF)
#             elif PETSC_MAJOR_VERSION == 0:
#                 self.J.createSeqAIJ([nRows,nCols],nz=16*3)
#             else:
#                 print 'Error: PETSC_MAJOR_VERSION = %d is not supported'%(PETSC_MAJOR_VERSION)
#                 sys.exit(1)
#             # end if
#         else:
#             self.J = zeros((nRows,nCols))
#         # end if
#         return nRows,nCols,dv_link
   #                  gcon[counter  ,3*indB + 1] =  x[3*indC + 2]-x[3*indA + 2]
#                     gcon[counter  ,3*indB + 2] = -x[3*indC + 1]+x[3*indA + 1]
#                     gcon[counter  ,3*indC + 2] =  x[3*indB + 1]-x[3*indA + 1]
#                     gcon[counter  ,3*indC + 1] = -x[3*indB + 2]+x[3*indA + 2]
#                     gcon[counter  ,3*indA + 1] = -x[3*indC + 2]+x[3*indA + 2] + x[3*indB+2] - x[3*indA + 2]
#                     gcon[counter  ,3*indA + 2] = -x[3*indB + 1]+x[3*indA + 1] + x[3*indC+1] - x[3*indA + 1]

#                     gcon[counter+1,3*indB + 2] =  x[3*indC + 0]-x[3*indA + 0]
#                     gcon[counter+1,3*indB + 0] = -x[3*indC + 2]+x[3*indA + 2]
#                     gcon[counter+1,3*indC + 0] =  x[3*indB + 2]-x[3*indA + 2]
#                     gcon[counter+1,3*indC + 2] = -x[3*indB + 0]+x[3*indA + 0]
#                     gcon[counter+1,3*indA + 2] = -x[3*indC + 0]+x[3*indA + 0] + x[3*indB+0] - x[3*indA + 0]
#                     gcon[counter+1,3*indA + 0] = -x[3*indB + 2]+x[3*indA + 2] + x[3*indC+2] - x[3*indA + 2]
                    
#                     gcon[counter+2,3*indB + 0] =  x[3*indC + 1]-x[3*indA + 1]
#                     gcon[counter+2,3*indB + 1] = -x[3*indC + 0]+x[3*indA + 0]
#                     gcon[counter+2,3*indC + 1] =  x[3*indB + 0]-x[3*indA + 0]
#                     gcon[counter+2,3*indC + 0] = -x[3*indB + 1]+x[3*indA + 1]
#                     gcon[counter+2,3*indA + 0] = -x[3*indC + 1]+x[3*indA + 1] + x[3*indB+1] - x[3*indA + 1]
#                     gcon[counter+2,3*indA + 1] = -x[3*indB + 0]+x[3*indA + 0] + x[3*indC+0] - x[3*indA + 0]


# BACK UP


#     def _sens3(self,x,f_obj,f_con,*args,**kwargs):
#         '''Sensitivity function for Fitting Optimization'''
#         time0 = time.time()
#         # ----------- Objective Derivative ----------------
#         if USE_PETSC:
#             self.X_PETSC.setValues(arange(0,self.ndv),x)
#             self.J(self.X_PETSC,self.temp)
#             self.J.multTranspose(self.temp-self.rhs,self.gobj_PETSC)
#             g_obj = array(self.gobj_PETSC.getValues(arange(self.ndv)))
#             self.temp = self.temp-self.rhs
#             self.temp.abs()
#             print 'Objective: %f, Max Error %f:'%(f_obj,self.temp.max()[1])
#         else:
#             g_obj = dot((dot(self.J,x)-self.rhs),self.J)
#         # end if
#         # ----------- Constraint Derivative ---------------

#         g_con = []
#         g_con = zeros((self.ncon,self.ndv))
#         counter = 0
#         for iedge in xrange(len(self.edge_list)):
#             if self.edge_list[iedge].cont == 1: # We have a continuity edge
#                 # Now get the two surfaces for this edge:
#                 surfaces = self.getSurfaceFromEdge(iedge)
#                 surf0 = surfaces[0][0] # First surface on this edge
#                 edge0 = surfaces[0][1] # Edge of surface on this edge     
#                 # Get the greville points for this edge
#                 gpts = self.surfs[surf0].getGrevillePoints(edge0)
                
#                 # Get ALL control points that could affect the constraints
#                 pt_list = array([],'intc')
#                 if   edge0 == 0:
#                     pt_list = append(pt_list,self.l_index[surf0][:,0:2].flatten())
#                 elif edge0 == 1:
#                     pt_list = append(pt_list,self.l_index[surf0][:,-2:].flatten())
#                 elif edge0 == 2:
#                     pt_list = append( pt_list,self.l_index[surf0][0:2,:].flatten())
#                 elif edge0 == 3:
#                     pt_list = append(pt_list,self.l_index[surf0][-2:,:].flatten())
#                 # end if

#                 if len(surfaces) != 1:

#                     surf1 = surfaces[1][0] # First surface on this edge
#                     edge1 = surfaces[1][1] # Edge of surface on this edge 
                 
#                     if   edge1 == 0:
#                         pt_list = append(pt_list,self.l_index[surf1][:,1])
#                     elif edge1 == 1:
#                         pt_list = append(pt_list,self.l_index[surf1][:,-2])
#                     elif edge1 == 2:
#                         pt_list = append(pt_list,self.l_index[surf1][1,:])
#                     elif edge1 == 3:
#                         pt_list = append(pt_list,self.l_index[surf1][-2,:])
#                     # end if
                  
#                 # end if

#                 # Unique-ify the list
#                 pt_list = unique(pt_list)
#                 pt_list.sort()
#                 for i in xrange(len(gpts)):
#                     S0 = f_con[counter]
                    
#                     for icoef in xrange(len(pt_list)):
#                         index = pt_list[icoef]
#                         for ii in xrange(3):

#                             self.coef[index,ii] += 1e-7
                      
#                             for jj in xrange(len(self.g_index[index])): # Set the coefficient
#                                 isurf = self.g_index[index][jj][0]
#                                 iii     = self.g_index[index][jj][1]
#                                 jjj     = self.g_index[index][jj][2]
#                                 self.surfs[isurf].coef[iii,jjj] = self.coef[index].astype('d')
#                             # end for
                          
#                             du,dv1 = self._getDerivativeOnEdge(surf0,edge0,gpts[i])
#                             if len(surfaces) == 1:
#                                 dv2 = self.sym_normal
#                             else:
#                                 surf1 = surfaces[1][0] # Second surface on this edge
#                                 edge1 = surfaces[1][1] # Edge of second surface on this edge
#                                 du,dv2 = self._getDerivativeOnEdge(surf1,edge1,gpts[i])
#                             # end if
                         
#                             S = du[0]*(dv1[1]*dv2[2]-dv1[2]*dv2[1]) - \
#                                 du[1]*(dv1[0]*dv2[2]-dv1[2]*dv2[0]) + \
#                                 du[2]*(dv1[0]*dv2[1]-dv1[1]*dv2[0])
#                             #print S
#                             g_con[counter,3*pt_list[icoef] + ii] = (S-S0)/ 1e-7
#                             self.coef[pt_list[icoef],ii] -= 1e-7

#                             for jj in xrange(len(self.g_index[index])): # Reset
#                                 isurf = self.g_index[index][jj][0]
#                                 iii     = self.g_index[index][jj][1]
#                                 jjj     = self.g_index[index][jj][2]
#                                 self.surfs[isurf].coef[iii,jjj] = self.coef[index].astype('d')
#                             # end for
#                         # end for (3 loop)
#                     # end for (pt_list loop)
#                     counter += 1
#                 # end for (gpts)
#             # end if (cont edge)
#         # end for (edge listloop)
#         #Bp,Bi,new_gcon = convertCSRtoCSC_one(self.ncon,self.ndv,self.loc,self.index,g_con)
#         print 'Sens Time:',time.time()-time0
#         return g_obj,g_con,0



# Backup

#                 Xchord_line =chord_line.getValue([0.10,1.0])
#                 chord_line = pySpline.curve('interpolate',X=Xchord_line,k=2)
#                 tip_line = chord_line.getValue(linspace(0,1,N))
#                 #print 'tip_line:',tip_line
#                 for ii in xrange(2): # up/low side loop
#                     Xnew = zeros((N,25,3))
#                     for j in xrange(N): # This is for the Data points
#                         # Interpolate across each point in the spanwise direction
#                         # Take a finite difference to get dv and normalize
#                         dv = (X[ii,j,-1] - X[ii,j,-2])
#                         dv /= sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])

#                         # Now project the vector between sucessive
#                         # airfoil points onto this vector                        
#                         if ii == 0:
#                             V = tip_line[j]-X[ii,j,-1]
#                             D = end_chord_line.getValue(0.1+.9*(j/(N-1.0)))-chord_line.getValue(j/(N-1.0))
#                             X_input = array([X[ii,j,-1],tip_line[j]])
#                         else:
#                             V = tip_line[N-j-1]-X[ii,j,-1]
#                             D = end_chord_line.getValue(0.1+0.9*(N-j-1)/(N-1.0))-chord_line.getValue((N-j-1)/(N-1.0))
#                             X_input = array([X[ii,j,-1],tip_line[N-j-1]])
#                         # end if

#                         dx1 = dot(dv,V) * dv * end_scale
                        
#                         dx2 = D+V
#                         print 'D,V:',D,V
#                         print 'dx2:',dx2
#                         temp_spline = pySpline.curve('interpolate',X=X_input,
#                                                              k=4,dx1=dx1,dx2=dx2)

#                         Xnew[j] =  temp_spline.getValue(linspace(0,1,25))
                        
#                     # end for
