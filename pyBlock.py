'''
pyBlock

pyBlcok is a (fairly) complete volume geometry engine. It performs
multiple functions including fitting volumes and mesh warping volumes. 
The actual b-spline volumes are of the pySpline volume type. See the individual
functions for additional information

Copyright (c) 2010 by G. Kenway
All rights reserved. Not to be used for commercial purposes.
Revision: 1.0   $Date: 11/03/2010$


Developers:
-----------
- Gaetan Kenway (GKK)

History
-------
	v. 1.0 - Initial Class Creation (GKK, 2010)
'''

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

try:
    from scipy import sparse,io
    from scipy.sparse.linalg.dsolve import factorized
    from scipy.sparse.linalg import bicgstab,gmres
    USE_SCIPY_SPARSE = True
except:
    USE_SCIPY_SPARSE = False
    print 'There was an error importing scipy scparse tools'

# =============================================================================
# Extension modules
# =============================================================================
from mdo_import_helper import *
exec(import_modules('geo_utils','pySpline','mpi4py'))

# =============================================================================
# pyBlock class
# =============================================================================
class pyBlock():
	
    def __init__(self,init_type,*args, **kwargs):
        
        '''Create an instance of the geometry object. The initialization type,
        init_type, specifies what type of initialization will be
        used. There are currently 4 initialization types: plot3d,
        iges, lifting_surface and acdt_geo

        
        Input: 
        
        init_type, string: a key word defining how this geo object
        will be defined. Valid Options/keyword argmuents are:

        'plot3d',file_name = 'file_name.xyz' : Load in a plot3D
        surface patches and use them to create splined volumes
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
        mpiPrint('pyBlock Initialization Type is: %s'%(init_type),self.NO_PRINT)
        mpiPrint('------------------------------------------------',self.NO_PRINT)

        #------------------- pyVol Class Atributes -----------------
        self.topo = None         # The topology of the volumes/surface
        self.vols = []           # The list of volumes (pySpline volume)
        self.nVol = None         # The total number of volumessurfaces
        self.coef  = None        # The global (reduced) set of control pts
        self.embeded_volumes = []
        # --------------------------------------------------------------

        if init_type == 'plot3d':
            self._readPlot3D(*args,**kwargs)
        elif init_type == 'cgns':
            self._readCGNS(*args,**kwargs)
        elif init_type == 'bvol':
            self._readBVol(*args,**kwargs)
        elif init_type == 'create':
            pass
        else:
            mpiPrint('init_type must be one of plot3d,cgns, or bvol.')
            sys.exit(0)
        return
# ----------------------------------------------------------------------
#                     Initialization Types
# ----------------------------------------------------------------------    

    def _readPlot3D(self,*args,**kwargs):

        '''Load a plot3D file and create the splines to go with each patch'''
        assert 'file_name' in kwargs,'file_name must be specified for plot3d'
        assert 'file_type' in kwargs,'file_type must be specified as binary or ascii'
        assert 'order'     in kwargs,'order must be specified as \'f\' or \'c\''
        file_name = kwargs['file_name']        
        file_type = kwargs['file_type']
        order     = kwargs['order']
        mpiPrint(' ',self.NO_PRINT)
        if file_type == 'ascii':
            mpiPrint('Loading ascii plot3D file: %s ...'%(file_name),self.NO_PRINT)
            binary = False
            f = open(file_name,'r')
        else:
            mpiPrint('Loading binary plot3D file: %s ...'%(file_name),self.NO_PRINT)
            binary = True
            f = open(file_name,'rb')
        # end if
        if binary:
            itype = readNValues(f,1,'int',binary)[0]
            nVol = readNValues(f,1,'int',binary)[0]
            itype = readNValues(f,1,'int',binary)[0] # Need these
            itype = readNValues(f,1,'int',binary)[0] # Need these
            sizes   = readNValues(f,nVol*3,'int',binary).reshape((nVol,3))
        else:
            nVol = readNValues(f,1,'int',binary)[0]
            sizes   = readNValues(f,nVol*3,'int',binary).reshape((nVol,3))
        # end if

        mpiPrint(' -> nVol = %d'%(nVol),self.NO_PRINT)
        blocks = []
        for i in xrange(nVol):
            cur_size = sizes[i,0]*sizes[i,1]*sizes[i,2]
            blocks.append(zeros([sizes[i,0],sizes[i,1],sizes[i,2],3]))
            for idim in xrange(3):
                blocks[-1][:,:,:,idim] = readNValues(f,cur_size,'float',binary).reshape((sizes[i,0],sizes[i,1],sizes[i,2]),order=order)
            # end for
        # end for

        f.close()

        # Now create a list of spline volume objects:
        vols = []
        # Note This doesn't actually fit the volumes...just produces
        # the parameterization and knot vectors

        for ivol in xrange(nVol):
            vols.append(pySpline.volume(X=blocks[ivol],ku=2,kv=2,kw=2,\
                                            Nctlu=3,Nctlv=3,Nctlw=3,\
                                            no_print=self.NO_PRINT,
                                        recompute=False))
        self.vols = vols
        self.nVol = len(vols)
        
        return

    def _readBVol(self,*args,**kwargs):
        '''Read a bvol file and produce the volumes'''
        assert 'file_name' in kwargs,'file_name must be specified for plot3d'
        assert 'file_type' in kwargs,'file_type must be specified as binary or ascii'
        file_name = kwargs['file_name']        
        file_type = kwargs['file_type']
        mpiPrint(' ',self.NO_PRINT)
        if file_type == 'ascii':
            mpiPrint('Loading ascii bvol file: %s ...'%(file_name),self.NO_PRINT)
            binary = False
            f = open(file_name,'r')
        else:
            mpiPrint('Loading binary bvol file: %s ...'%(file_name),self.NO_PRINT)
            binary = True
            f = open(file_name,'rb')
        # end for

        self.nVol = readNValues(f,1,'int',binary)
        mpiPrint(' -> nVol = %d'%(self.nVol),self.NO_PRINT)
        self.vols = []
        for ivol in xrange(self.nVol):
            inits = readNValues(f,6,'int',binary) # This is
                                                  # nctlu,nctlv,nctlw,
                                                  # ku,kv,kw
            tu  = readNValues(f,inits[0]+inits[3],'float',binary)
            tv  = readNValues(f,inits[1]+inits[4],'float',binary)
            tw  = readNValues(f,inits[2]+inits[5],'float',binary)
            coef= readNValues(f,inits[0]*inits[1]*inits[2]*3,'float',binary).reshape([inits[0],inits[1],inits[2],3])
            self.vols.append(pySpline.volume(\
                    Nctlu=inits[0],Nctlv=inits[1],Nctlw=inits[2],ku=inits[3],
                    kv=inits[4],kw=inits[5],tu=tu,tv=tv,tw=tw,coef=coef))
        # end for
            
    def _readCGNS(self,*args,**kwargs):
        '''Load a CGNS file and create the spline to go with each patch'''
        assert 'file_name' in kwargs,'file_name must be specified for CGNS'
        file_name = kwargs['file_name']
        import pyspline

        mpiPrint(' ',self.NO_PRINT)
        mpiPrint('Loading CGNS file: %s ...'%(file_name),self.NO_PRINT)
        cg,nzones = pyspline.open_cgns(file_name)
        mpiPrint(' -> nVol = %d'%(nzones),self.NO_PRINT)

        blocks = []
        BCs = []
        for i in xrange(nzones):
            zoneshape = pyspline.read_cgns_zone_shape(cg,i+1)
            X,faceBCs = pyspline.read_cgns_zone(cg,i+1,zoneshape[0],zoneshape[1],zoneshape[2])
            blocks.append(X)
            BCs.append(faceBCs)
        # end for

        pyspline.close_cgns(cg)

        # Now create a list of spline volume objects:
        vols = []
        # Note This doesn't actually fit the volumes...just produces
        # the parameterization and knot vectors

        for ivol in xrange(nzones):
            vols.append(pySpline.volume(X=blocks[ivol],ku=2,kv=2,kw=2,
                                        Nctlu=4,Nctlv=4,Nctlw=4,
                                        no_print=self.NO_PRINT,
                                        recompute=False,faceBCs=BCs[ivol]))
        self.vols = vols
        self.nVol = len(vols)
    
        return

    def fitGlobal(self):
        mpiPrint(' ',self.NO_PRINT)
        mpiPrint('Global Fitting',self.NO_PRINT)
        nCtl = self.topo.nGlobal
        mpiPrint(' -> Copying Topology',self.NO_PRINT)
        orig_topo = copy.deepcopy(self.topo)
        
        mpiPrint(' -> Creating global numbering',self.NO_PRINT)
        sizes = []
        for ivol in xrange(self.nVol):
            sizes.append([self.vols[ivol].Nu,self.vols[ivol].Nv,self.vols[ivol].Nw])
        # end for
        
        # Get the Globaling number of the original data
        orig_topo.calcGlobalNumbering(sizes) 
        N = orig_topo.nGlobal
        mpiPrint(' -> Creating global point list',self.NO_PRINT)
        pts = zeros((N,3))
        for ii in xrange(N):
            pts[ii] = self.vols[orig_topo.g_index[ii][0][0]].X[orig_topo.g_index[ii][0][1],
                                                               orig_topo.g_index[ii][0][2],
                                                               orig_topo.g_index[ii][0][3]]
        # end for

        # Get the maximum k (ku,kv,kw for each vol)
        kmax = 2
        for ivol in xrange(self.nVol):
            if self.vols[ivol].ku > kmax:
                kmax = self.vols[ivol].ku
            if self.vols[ivol].kv > kmax:
                kmax = self.vols[ivol].kv
            if self.vols[ivol].kw > kmax:
                kmax = self.vols[ivol].kw
            # end if
        # end for
        nnz = N*kmax*kmax*kmax
        vals = zeros(nnz)
        row_ptr = [0]
        col_ind = zeros(nnz,'intc')
        mpiPrint(' -> Calculating Jacobian',self.NO_PRINT)
        for ii in xrange(N):
            ivol = orig_topo.g_index[ii][0][0]
            i = orig_topo.g_index[ii][0][1]
            j = orig_topo.g_index[ii][0][2]
            k = orig_topo.g_index[ii][0][3]

            u = self.vols[ivol].U[i,j,k]
            v = self.vols[ivol].V[i,j,k]
            w = self.vols[ivol].W[i,j,k]

            vals,col_ind = self.vols[ivol]._getBasisPt(
                u,v,w,vals,row_ptr[ii],col_ind,self.topo.l_index[ivol])

            kinc = self.vols[ivol].ku*self.vols[ivol].kv*self.vols[ivol].kw
            row_ptr.append(row_ptr[-1] + kinc)
        # end for

        # Now we can crop out any additional values in col_ptr and vals
        vals    = vals[:row_ptr[-1]]
        col_ind = col_ind[:row_ptr[-1]]
        # Now make a sparse matrix

        NN = sparse.csr_matrix((vals,col_ind,row_ptr))
        mpiPrint(' -> Multiplying N^T * N',self.NO_PRINT)
        NNT = NN.T
        NTN = NNT*NN
        mpiPrint(' -> Factorizing...',self.NO_PRINT)
        solve = factorized(NTN)
        mpiPrint(' -> Back Solving...',self.NO_PRINT)
        self.coef = zeros((nCtl,3))
        for idim in xrange(3):
            self.coef[:,idim] = solve(NNT*pts[:,idim])
        # end for

        mpiPrint(' -> Setting Volume Coefficients...',self.NO_PRINT)
        self._updateVolumeCoef()


# ----------------------------------------------------------------------
#                     Topology Information Functions
# ----------------------------------------------------------------------    

    def doConnectivity(self,file_name,node_tol=1e-4,edge_tol=1e-4):
        '''
        This is the only public edge connectivity function. 
        If file_name exists it loads the file OR it calculates the connectivity
        and saves to that file.
        Required:
            file_name: filename for con file
        Optional:
            node_tol: The tolerance for identical nodes
            edge_tol: The tolerance for midpoints of edge being identical
        Returns:
            None
            '''
        if os.path.isfile(file_name):
            mpiPrint(' ',self.NO_PRINT)
            mpiPrint('Reading Connectivity File: %s'%(file_name),self.NO_PRINT)
            self.topo = BlockTopology(file=file_name)
            if self.init_type != 'bvol':
                self._propagateKnotVectors()
            # end if
        else:
            mpiPrint(' ',self.NO_PRINT)
            self._calcConnectivity(node_tol,edge_tol)
            self._propagateKnotVectors()
            mpiPrint('Writing Connectivity File: %s'%(file_name),self.NO_PRINT)
            self.topo.writeConnectivity(file_name)
        # end if

        sizes = []
        for ivol in xrange(self.nVol):
            sizes.append([self.vols[ivol].Nctlu,self.vols[ivol].Nctlv,
                              self.vols[ivol].Nctlw])
        self.topo.calcGlobalNumbering(sizes)
        return 

    def _calcConnectivity(self,node_tol,edge_tol):
        # Determine the blocking connectivity

        # Compute the corners
        corners = zeros((self.nVol,8,3))
        for ivol in xrange(self.nVol):
            for icorner in xrange(8):
                corners[ivol,icorner] = self.vols[ivol].getOrigValueCorner(icorner)
            # end for
        # end for
        self.topo = BlockTopology(corners)
        sizes = []
        for ivol in xrange(self.nVol):
            sizes.append([self.vols[ivol].Nctlu,self.vols[ivol].Nctlv,
                          self.vols[ivol].Nctlw])
        self.topo.calcGlobalNumbering(sizes)


    def printConnectivity(self):
        '''
        Print the connectivity to the screen
        Required:
            None
        Returns:
            None
            '''
        self.topo.printEdgeConnectivity()
        return
  
    def _propagateKnotVectors(self):
        ''' Propage the knot vectors to make consistent'''
     
        nDG = -1
        ncoef = []
        for i in xrange(self.topo.nEdge):
            if self.topo.edges[i].dg > nDG:
                nDG = self.topo.edges[i].dg
                ncoef.append(self.topo.edges[i].Nctl)
            # end if
        # end for
        nDG += 1
        
    	for ivol in xrange(self.nVol):
            dg_u = self.topo.edges[self.topo.edge_link[ivol][0]].dg
            dg_v = self.topo.edges[self.topo.edge_link[ivol][2]].dg
            dg_w = self.topo.edges[self.topo.edge_link[ivol][8]].dg
            self.vols[ivol].Nctlu = ncoef[dg_u]
            self.vols[ivol].Nctlv = ncoef[dg_v]
            self.vols[ivol].Nctlw = ncoef[dg_w]
            if self.vols[ivol].ku < self.vols[ivol].Nctlu:
                if self.vols[ivol].Nctlu > 4:
	            self.vols[ivol].ku = 4
                else:
                    self.vols[ivol].ku = self.vols[ivol].Nctlu
		# endif
            # end if
            if self.vols[ivol].kv < self.vols[ivol].Nctlv:
		if self.vols[ivol].Nctlv > 4:
                    self.vols[ivol].kv = 4
                else:
                    self.vols[ivol].kv = self.vols[ivol].Nctlv
                # end if
            # end if

            if self.vols[ivol].kw < self.vols[ivol].Nctlw:
		if self.vols[ivol].Nctlw > 4:
                    self.vols[ivol].kw = 4
                else:
                    self.vols[ivol].kw = self.vols[ivol].Nctlw
                # end if
            # end if

            self.vols[ivol]._calcKnots()
            # Now loop over the number of design groups, accumulate all
            # the knot vectors that coorspond to this dg, then merge them all
        # end for
        
        for idg in xrange(nDG):
            #print '---------------- DG %d ------------'%(idg)
            knot_vectors = []
            flip = []
            for ivol in xrange(self.nVol):
                for iedge in xrange(12):
                    if self.topo.edges[self.topo.edge_link[ivol][iedge]].dg == idg:
                        if self.topo.edge_dir[ivol][iedge] == -1:
                            flip.append(True)
                        else:
                            flip.append(False)
                        # end if
                        if iedge in [0,1,4,5]:
                            knot_vec = self.vols[ivol].tu
                        elif iedge in [2,3,6,7]:
                            knot_vec = self.vols[ivol].tv
                        elif iedge in [8,9,10,11]:
                            knot_vec = self.vols[ivol].tw
                        # end if

                        if flip[-1]:
                            knot_vectors.append((1-knot_vec)[::-1].copy())
                        else:
                            knot_vectors.append(knot_vec)
                        # end if
                    # end if
                # end for
            # end for
           
            # Now blend all the knot vectors
            new_knot_vec = blendKnotVectors(knot_vectors,False)
            new_knot_vec_flip = (1-new_knot_vec)[::-1]
            # And reset them all
            counter = 0
            for ivol in xrange(self.nVol):
                for iedge in xrange(12):
                    if self.topo.edges[self.topo.edge_link[ivol][iedge]].dg == idg:
                        if iedge in [0,1,4,5]:
                            if flip[counter] == True:
                                self.vols[ivol].tu = new_knot_vec_flip.copy()
                            else:
                                self.vols[ivol].tu = new_knot_vec.copy()
                            # end if
                        elif iedge in [2,3,6,7]:
                            if flip[counter] == True:
                                self.vols[ivol].tv = new_knot_vec_flip.copy()
                            else:
                                self.vols[ivol].tv = new_knot_vec.copy()
                            # end if
                        elif iedge in [8,9,10,11]:
                            if flip[counter] == True:
                                self.vols[ivol].tw = new_knot_vec_flip.copy()
                            else:
                                self.vols[ivol].tw = new_knot_vec.copy()
                            # end if
                        # end if
                        counter += 1
                    # end if
                # end for
                self.vols[ivol]._setCoefSize()
            # end for
        # end for (dg loop)

        return    


    def reOrderIndices(self):
        '''This funcion takes the order from self.topo and reorders
        them accordint to the Boundary Condition types in each volume
        class. The sole purpose of this is to facilitate the
        structural mesh warping algorithim for matrix assembly.'''

        # We want the global indicies ordered according to:
        #[ interior unonstrainted nodes]
        #[ constrained surface nodes   ]
        #[ constrained fixed nodes     ]

        # First determine the number in each category
        
        pt_type = zeros(self.topo.nGlobal,'intc')
        for ii in xrange(self.topo.nGlobal):
            
            for jj in xrange(len(self.topo.g_index[ii])):
                ivol = self.topo.g_index[ii][jj][0]
                i    = self.topo.g_index[ii][jj][1]
                j    = self.topo.g_index[ii][jj][2]
                k    = self.topo.g_index[ii][jj][3]

                N = self.topo.l_index[ivol].shape[0]
                M = self.topo.l_index[ivol].shape[0]
                L = self.topo.l_index[ivol].shape[0]
                type,number,index1,index2 = \
                    indexPosition3D(i,j,k,N,M,L)
                
                checkFaces = []
                if type == 0:
                    pt_type[ii] = 0
                else: # Face
                    if type == 1: # Face
                        checkFaces.append(number)
                    elif type == 2: # Edge
                        if number in [0,1,2,3]:
                            checkFaces.append(0)
                        if number in [4,5,6,7]:
                            checkFaces.append(1)
                        if number in [2,6,8,10]:
                            checkFaces.append(2)
                        if number in [3,7,9,11]:
                            checkFaces.append(3)
                        if number in [0,4,8,9]:
                            checkFaces.append(4)
                        if number in [1,5,10,11]:
                            checkFaces.append(5)
                    elif type == 3: # Corner
                        if number == 0:
                            checkFaces.extend([0,2,4])
                        elif number == 1:
                            checkFaces.extend([0,3,4])
                        elif number == 2:
                            checkFaces.extend([0,2,5])
                        elif number == 3:
                            checkFaces.extend([0,3,5])
                        elif number == 4:
                            checkFaces.extend([1,2,4])
                        elif number == 5:
                            checkFaces.extend([1,3,4])
                        elif number == 6:
                            checkFaces.extend([1,2,5])
                        elif number == 7:
                            checkFaces.extend([1,3,5])
                    # end if

                    for iii in xrange(len(checkFaces)):
                        iface = checkFaces[iii]
                        if self.vols[ivol].faceBCs[checkFaces[iii]] == 1:
                            pt_type[ii] = 1
                        # end if
                        if self.vols[ivol].faceBCs[checkFaces[iii]] == 2:
                            pt_type[ii] = 2
                        # end if
                    # end for
                # end if
            # end for
        # end for
                            
        print 'Points on surface:'
        surface_points = 0
        boundary_points = 0
        for i in xrange(len(pt_type)):
            if pt_type[i] == 1:
                surface_points += 1
            if pt_type[i] == 2:
                boundary_points += 1
            # end if
        # end for
        free_points = self.topo.nGlobal - (boundary_points+surface_points)
        mpiPrint('Total Nodes: %d'%(self.topo.nGlobal))
        mpiPrint('Surface Nodes: %d'%(surface_points))
        mpiPrint('Boundary Nodes: %d'%(boundary_points))

        # Now attempt to do the reorganization
        # We can split the the g_index up into three parts:
        #  -> pt_type 0 
        #  -> pt_type 1
        #  -> pt_type 2
        g_index_free = [ [] for i in xrange(free_points)]
        g_index_surface = [ [] for i in xrange(surface_points)]
        g_index_boundary = [ [] for i in xrange(boundary_points)]

        # Make a single pass through, puck out each list in
        # g_index[i], add it to the correct sub_list, AND update each
        # of the entries in the local index to point back to the
        # correct place IN THE NEW,RE-ASSEMBLED g_index

        freeCount = 0
        surfaceCount = 0
        boundaryCount = 0
        for ii in xrange(self.topo.nGlobal):
            if pt_type[ii] == 0:
                g_index_free[freeCount] = self.topo.g_index[ii]
                
                for jj in xrange(len(self.topo.g_index[ii])):
                    ivol = self.topo.g_index[ii][jj][0]
                    i    = self.topo.g_index[ii][jj][1]
                    j    = self.topo.g_index[ii][jj][2]
                    k    = self.topo.g_index[ii][jj][3]
                
                    self.topo.l_index[ivol][i,j,k] = 0 + freeCount
                # end for

                freeCount += 1
                                    
            elif pt_type[ii] == 1:
                g_index_surface[surfaceCount] = self.topo.g_index[ii]

                for jj in xrange(len(self.topo.g_index[ii])):
                    ivol = self.topo.g_index[ii][jj][0]
                    i    = self.topo.g_index[ii][jj][1]
                    j    = self.topo.g_index[ii][jj][2]
                    k    = self.topo.g_index[ii][jj][3]
                
                    self.topo.l_index[ivol][i,j,k] = free_points + surfaceCount
                # end for

                surfaceCount += 1

            elif pt_type[ii] == 2:
                g_index_boundary[boundaryCount] = self.topo.g_index[ii]

                for jj in xrange(len(self.topo.g_index[ii])):
                    ivol = self.topo.g_index[ii][jj][0]
                    i    = self.topo.g_index[ii][jj][1]
                    j    = self.topo.g_index[ii][jj][2]
                    k    = self.topo.g_index[ii][jj][3]
                
                    self.topo.l_index[ivol][i,j,k] = \
                        free_points + surface_points + boundaryCount
                # end for

                boundaryCount += 1
            # end if
        # end for

        # Finally, concentenate g_index back together
        print len(g_index_free),len(g_index_surface),len(g_index_boundary)
        g_index_free.extend(g_index_surface)
        g_index_free.extend(g_index_boundary)
        self.topo.g_index = g_index_free
        return free_points,surface_points,boundary_points
        
# ----------------------------------------------------------------------
#                        Output Functions
# ----------------------------------------------------------------------    

    def writeTecplot(self,file_name,vols=True,coef=True,orig=False,
                     vol_labels=False,tecio=False):

        '''Write the pyGeo Object to Tecplot dat file
        Required:
            file_name: The filename for the output file
        Optional:
            vols: boolean, write the interpolated volumes
            coef: boolean, write the control points
            vol_labels: boolean, write the surface labels
            '''

        # Open File and output header
        
        f = pySpline.openTecplot(file_name,3,tecio=tecio)

        # --------------------------------------
        #    Write out the Interpolated Surfaces
        # --------------------------------------
        
        if vols == True:
            for ivol in xrange(self.nVol):
                self.vols[ivol]._writeTecplotVolume(f)

        # --------------------------------------
        #    Write out the Original Grid
        # --------------------------------------
        
        if orig == True:
            for ivol in xrange(self.nVol):
                self.vols[ivol]._writeTecplotOrigData(f)

        # -------------------------------
        #    Write out the Control Points
        # -------------------------------
        
        if coef == True:
            for ivol in xrange(self.nVol):
                self.vols[ivol]._writeTecplotCoef(f)

        # ---------------------------------------------
        #    Write out The Volume Labels
        # ---------------------------------------------
        if vol_labels == True:
            # Split the filename off
            (dirName,fileName) = os.path.split(file_name)
            (fileBaseName, fileExtension)=os.path.splitext(fileName)
            label_filename = dirName+'./'+fileBaseName+'.vol_labels.dat'
            f2 = open(label_filename,'w')
            for ivol in xrange(self.nVol):
                midu = floor(self.vols[ivol].Nctlu/2)
                midv = floor(self.vols[ivol].Nctlv/2)
                midw = floor(self.vols[ivol].Nctlw/2)
                text_string = 'TEXT CS=GRID3D, X=%f,Y=%f,Z=%f, T=\"V%d\"\n'%(self.vols[ivol].coef[midu,midv,midw,0],self.vols[ivol].coef[midu,midv,midw,1], self.vols[ivol].coef[midu,midv,midw,2],ivol)
                f2.write('%s'%(text_string))
            # end for 
            f2.close()
        # end if 
        pySpline.closeTecplot(f)
        return

    def writeBvol(self,file_name,binary=False):
        '''Write the pyBlock volumes to a file. This is the equilivent
        of the iges file for the surface version. 
        '''
        if binary:
            f = open(file_name,'wb')
            array(self.nVol).tofile(f,sep="")
        else:
            f = open(file_name,'w')
            f.write('%d\n'%(self.nVol))
        # end for
        for ivol in xrange(self.nVol):
            self.vols[ivol]._writeBvol(f,binary)
        # end for
        

    def writePlot3d(self,file_name,binary=False):
        '''Write the grid to a plot3d file'''

        sizes = []
        for ivol in xrange(self.nVol):
            sizes.append(self.vols[ivol].Nu)
            sizes.append(self.vols[ivol].Nv)
            sizes.append(self.vols[ivol].Nw)
        # end for
        
        if binary:
            f = open(file_name,'wb')
            array(self.nVol).tofile(f,sep="")
            array(sizes).tofile(f,sep="")
            for ivol in xrange(self.nVol):
                vals = self.vols[ivol](self.vols[ivol].U,self.vols[ivol].V,
                                       self.vols[ivol].W)
                vals[:,:,:,0].flatten(1).tofile(f,sep="")
                vals[:,:,:,1].flatten(1).tofile(f,sep="")
                vals[:,:,:,2].flatten(1).tofile(f,sep="")
            
            
        else:
            f = open(file_name,'w')
            f.write('%d\n'%(self.nVol))
            array(sizes).tofile(f,sep=" ")
            f.write('\n')
            for ivol in xrange(self.nVol):
                vals = self.vols[ivol](self.vols[ivol].U,self.vols[ivol].V,
                                       self.vols[ivol].W)
                vals[:,:,:,0].flatten(1).tofile(f,sep="\n")
                f.write('\n')
                vals[:,:,:,1].flatten(1).tofile(f,sep="\n")
                f.write('\n')
                vals[:,:,:,2].flatten(1).tofile(f,sep="\n")
                f.write('\n')
                

        # end for
        f.close()
        

# ----------------------------------------------------------------------
#               Update Functions
# ----------------------------------------------------------------------    
    def _updateVolumeCoef(self):
        '''Copy the pyBlock list of control points back to the volumes'''
        for ii in xrange(len(self.coef)):
            for jj in xrange(len(self.topo.g_index[ii])):
                ivol  = self.topo.g_index[ii][jj][0]
                i     = self.topo.g_index[ii][jj][1]
                j     = self.topo.g_index[ii][jj][2]
                k     = self.topo.g_index[ii][jj][3]
                self.vols[ivol].coef[i,j,k] = self.coef[ii].astype('d')
            # end for
        # end for
        return

# ----------------------------------------------------------------------
#               External Struct Solve Functions
# ----------------------------------------------------------------------    

    def writeFEAPCorners(self,file_name):
        # Make sure sizes are 2
        sizes = []
        for ivol in xrange(self.nVol):
            sizes.append([2,2,2])
        # end for
        E = 1.0
        nu = 0.3
        self.topo.calcGlobalNumbering(sizes)
        
        numnp  = self.topo.nNode #number of nodal points
        numel  = self.nVol # number of elements
        nummat = self.nVol  #one material type
        ndm    = 3  #dimension of mesh
        ndf    = 3  #number of dof per node
        nen    = 8  #number of nodes per element

        f = open(file_name,'w')
        f.write("FEAP * * Solid Element Element Example\n")
        #f.write("NOPRINT\n")
        f.write("%d %d %d %d %d %d\n"%(numnp,numel,1,ndm,ndf,nen))

        f.write("\n")

        #for ivol in xrange(self.nVol):
        f.write("MATErial %d\n"%(1))
        f.write("SOLID\n")
        f.write("ELAStic ISOtripoic ")
        f.write("%f %f\n"%(E,nu))

        f.write("\n")

        f.write("COORdinate ALL\n")
        for icoord in xrange(self.topo.nNode):
            ivol = self.topo.g_index[icoord][0][0]
            i = self.topo.g_index[icoord][0][1]
            j = self.topo.g_index[icoord][0][2]
            k = self.topo.g_index[icoord][0][3]
            pt = self.vols[ivol].X[i,j,k]
            f.write("%d 0 %f %f %f \n"%(icoord+1,pt[0],pt[1],pt[2]))
            
        f.write("\n")

        f.write("ELEMents\n") # Use vol_con here
        for ivol in xrange(self.nVol):
            f.write("%d 1 %d %d %d %d %d %d %d %d %d \n"
                    %(ivol+1,1,
                      self.topo.vol_con[ivol][0]+1,
                      self.topo.vol_con[ivol][1]+1,
                      self.topo.vol_con[ivol][3]+1,
                      self.topo.vol_con[ivol][2]+1,
                      self.topo.vol_con[ivol][4]+1,
                      self.topo.vol_con[ivol][5]+1,
                      self.topo.vol_con[ivol][7]+1,
                      self.topo.vol_con[ivol][6]+1))
        f.write("\n")
        f.write("BOUNdary restraints\n")

        f.write('%d 0 1 1 1 \n'%(1))
        f.write('%d 0 1 1 1 \n'%(2))
        f.write('%d 0 1 1 1 \n'%(3))
        
    
        f.write("\n")
        
        f.write("FORCe\n")

        f.write("%d 0 0 0 %f \n"%(10,10.00))
        f.write("\n")

        f.write("END\n")
        #f.write("NOPRINT\n")
        f.write("BATCh\n")
        f.write("TANGent\n")
        f.write("FORM\n")
        f.write("SOLV\n")
        f.write("PRINT\n")
        f.write("DISPlacement all\n")
        f.write("END\n")
        f.write("STOP\n")

        f.close() #close file

    def updateFEAP(file_name):
        f = open(file_name)
        counter = 0
        new_pts = zeros((self.topo.nNode,3))
        for line in f:
            if counter >= 82 and mod(counter,2) == 0:
                aux = string.split(line)
                #new_pts[i,0] = 


    def writeFEAP(self,file_name):

        nu = 0.0
        numnp  = len(self.topo.g_index) #number of nodal points
        numel = 0
        for ivol in xrange(self.nVol):
            numel += (self.vols[ivol].Nctlu-1)*(self.vols[ivol].Nctlv-1)*(self.vols[ivol].Nctlw-1)
        nummat = numel  #one material type for each elem
        ndm    = 3  #dimension of mesh
        ndf    = 3  #number of dof per node
        nen    = 8  #number of nodes per element

        f = open(file_name,'w')
        f.write("FEAP * * Solid Element Element Example\n")
        f.write("%d %d %d %d %d %d\n"%(numnp,numel,1,ndm,ndf,nen))
        f.write("NOPRINT\n")
        f.write("\n")
        counter = 1
        Es = []
        g = open('bad_vols.dat','w')
        g.write ('VARIABLES = "X", "Y","Z"\n')
        for ivol in xrange(self.nVol):
            Nctlu = self.vols[ivol].Nctlu
            Nctlv = self.vols[ivol].Nctlv
            Nctlw = self.vols[ivol].Nctlw
            for i in xrange(Nctlu-1):
                for j in xrange(Nctlv-1):
                    for k in xrange(Nctlw-1):
                        points = [self.vols[ivol].coef[i,j,k  ],self.vols[ivol].coef[i+1,j  ,k],
                                  self.vols[ivol].coef[i,j+1,k],self.vols[ivol].coef[i+1,j+1,k],
                                  self.vols[ivol].coef[i,j,k+1],self.vols[ivol].coef[i+1,j,k+1],
                                  self.vols[ivol].coef[i,j+1,k+1],self.vols[ivol].coef[i+1,j+1,k+1]]
                        vol = volume_hexa(points)
                        if vol < 0:
                            name = '%d,%d,%d,%d'%(ivol,i,j,k)
                            g.write('Zone T=\"%s\" I=%d J=%d K=%d\n'%(name,2,2,2))
                            g.write('DATAPACKING=POINT\n')
                            for ii in xrange(8):
                                g.write('%f %f %f\n'%(points[ii][0],points[ii][1],points[ii][2]))
                            # end for
                        # end if
                        f.write("MATErial %d\n"%(counter))
                        f.write("SOLID\n")
                        f.write("ELAStic ISOtripoic ")
                        f.write("%f %f\n"%(1/vol,nu))
                        Es.append(1.0/vol)
                        counter += 1
                        f.write("\n")
                    # end for
                # end for
            # end for
        # end for
        f.write("\n")
        print 'Max E: %g, Min E:%g, Ratio: %g'%(max(Es),min(Es),max(Es)/min(Es))
        f.write("COORdinate ALL\n")
        for ii in xrange(len(self.topo.g_index)):
            ivol  = self.topo.g_index[ii][0][0]
            i     = self.topo.g_index[ii][0][1]
            j     = self.topo.g_index[ii][0][2]
            k     = self.topo.g_index[ii][0][3]
            pt = self.vols[ivol].coef[i,j,k]
            f.write("%d 0 %f %f %f \n"%(ii+1,pt[0],pt[1],pt[2]))
            
        f.write("\n")

        f.write("ELEMents\n") 
        counter = 1
        for ivol in xrange(self.nVol):
            for i in xrange(self.vols[ivol].Nctlu-1):
                for j in xrange(self.vols[ivol].Nctlv-1):
                    for k in xrange(self.vols[ivol].Nctlw-1):
                        f.write("%d 1 %d %d %d %d %d %d %d %d %d \n"
                                %(counter,counter,
                                  self.topo.l_index[ivol][i,j,k]+1,
                                  self.topo.l_index[ivol][i+1,j,k]+1,
                                  self.topo.l_index[ivol][i+1,j+1,k]+1,
                                  self.topo.l_index[ivol][i,j+1,k]+1,
                                  self.topo.l_index[ivol][i,j,k+1]+1,
                                  self.topo.l_index[ivol][i+1,j,k+1]+1,
                                  self.topo.l_index[ivol][i+1,j+1,k+1]+1,
                                  self.topo.l_index[ivol][i,j+1,k+1]+1))
                        counter += 1
        f.write("\n")
        f.write("BOUNdary restraints\n")

        for i in xrange(50):
            f.write('%d 0 1 1 1 \n'%(i))
        
    
        f.write("\n")
        f.write("FORCe\n")
        f.write("%d 0 0 0 %f \n"%(100,10.00))
        f.write("\n")

        f.write("END\n")
        f.write("BATCh\n")
        f.write("TANGent\n")
        f.write("FORM\n")
        f.write("SOLV\n")
        f.write("DISPlacement all\n")
        f.write("END\n")
        f.write("STOP\n")

        f.close() #close file


# ----------------------------------------------------------------------
#             Embeded Geometry Functions
# ----------------------------------------------------------------------    

    def embedVolume(self,coordinates,volume_list=None):
        '''Embed a set of coordinates into volume in volume_list'''
        volID = []
        u = []
        v = []
        w = []
        
        for i in xrange(len(coordinates)):
            ivol,u0,v0,w0,D0 = self.projectPoint(coordinates[i])
            u.append(u0)
            v.append(v0)
            w.append(w0)
            volID.append(ivol)
        # end for

        self.embeded_volumes.append(embeded_volume(volID,u,v,w))


# ----------------------------------------------------------------------
#             Geometric Functions
# ----------------------------------------------------------------------    

    def projectPoint(self,x0):
        '''Project a point into any one of the volumes. Returns 
        the volID,u,v,w,D of the point in volID or closest to it.

        This is a brute force search and is NOT efficient'''

        u0,v0,w0,D0 = self.vols[0].projectPoint(x0)
        volID = 0

        for ivol in xrange(1,self.nVol):
            u,v,w,D = self.vols[ivol].projectPoint(x0)
            if norm(D)<norm(D0):
                D0 = D
                u0 = u
                v0 = v
                w0 = w
                volID = ivol
            # end if
        # end for

        return volID,u0,v0,w0,D0

    def _calcdPtdCoef(self,index):
        '''Calculate the (fixed) volume derivative of a discrete set of ponits'''
        volID = self.embeded_volumes[index].volID
        u       = self.embeded_volumes[index].u
        v       = self.embeded_volumes[index].v
        w       = self.embeded_volumes[index].w
        N       = self.embeded_volumes[index].N
        mpiPrint('Calculating Volume %d Derivative for %d Points...'%(index,len(volID)),self.NO_PRINT)

        # Get the maximum k (ku or kv for each surface)
        kmax = 2
        for ivol in xrange(self.nVol):
            if self.vols[ivol].ku > kmax:
                kmax = self.vols[ivol].ku
            if self.vols[ivol].kv > kmax:
                kmax = self.vols[ivol].kv
            if self.vols[ivol].kw > kmax:
                kmax = self.vols[ivol].kw
            # end if
        # end for
        nnz = N*kmax*kmax*kmax
        vals = zeros(nnz)
        row_ptr = [0]
        col_ind = zeros(nnz,'intc')
        for i in xrange(N):
            kinc = self.vols[volID[i]].ku*self.vols[volID[i]].kv*self.vols[volID[i]].kw
            vals,col_ind = self.vols[volID[i]]._getBasisPt(\
                u[i],v[i],w[i],vals,row_ptr[i],col_ind,self.topo.l_index[volID[i]])
            row_ptr.append(row_ptr[-1] + kinc)

        # Now we can crop out any additional values in col_ptr and vals
        vals    = vals[:row_ptr[-1]]
        col_ind = col_ind[:row_ptr[-1]]
        # Now make a sparse matrix
        self.embeded_volumes[index].dPtdCoef = sparse.csr_matrix((vals,col_ind,row_ptr),shape=[N,len(self.coef)])
        mpiPrint('  -> Finished Embeded Volume %d Derivative'%(index),self.NO_PRINT)
        
        return

  
class embeded_volume(object):

    def __init__(self,volID,u,v,w):
        '''A Container class for a set of embeded volume points
        Requres:
            voliD list of the volume iD's for the points
            uvw: list of the uvw points
            '''
        self.volID = volID
        self.u = u
        self.v = v
        self.w = w
        self.N = len(self.u)
        self.dPtdCoef = None
        self.dPtdX    = None
