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

from scipy import sparse, linsolve

# =============================================================================
# Extension modules
# =============================================================================

from mdo_import_helper import *
exec(import_modules('geo_utils','pySpline','mpi4py'))
import pyspline
# =============================================================================
# pyGeo class
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
        mpiPrint('pyBlcok Initialization Type is: %s'%(init_type),self.NO_PRINT)
        mpiPrint('------------------------------------------------',self.NO_PRINT)

        #------------------- pyVol Class Atributes -----------------
        self.topo = None         # The topology of the volumes/surface
        self.vols = []           # The list of volumes (pySpline volume)
        self.nVol = None         # The total number of volumessurfaces
        self.coef  = None        # The global (reduced) set of control pts

        # --------------------------------------------------------------

        if init_type == 'plot3d':
            self._readPlot3D(*args,**kwargs)
        elif init_type == 'cgns':
            self._readCGNS(*args,**kwargs)
        elif init_type == 'bvol':
            self._readBVol(*args,**kwargs)
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
        file_name = kwargs['file_name']        
        file_type = kwargs['file_type']

        if file_type == 'ascii':
            mpiPrint('Loading ascii plot3D file: %s ...'%(file_name),self.NO_PRINT)
            binary = False
            f = open(file_name,'r')
        else:
            mpiPrint('Loading binary plot3D file: %s ...'%(file_name),self.NO_PRINT)
            binary = True
            f = open(file_name,'rb')
        # end if
        nVol = readNValues(f,1,'int',binary)[0]
        mpiPrint('nVol = %d'%(nVol),self.NO_PRINT)
        sizes   = readNValues(f,nVol*3,'int',binary).reshape((nVol,3))
        blocks = []

        for i in xrange(nVol):
            cur_size = sizes[i,0]*sizes[i,1]*sizes[i,2]
            blocks.append(zeros([sizes[i,0],sizes[i,1],sizes[i,2],3]))
            for idim in xrange(3):
                blocks[-1][:,:,:,idim] = readNValues(f,cur_size,'float',binary).reshape((sizes[i,0],sizes[i,1],sizes[i,2]),order='F')
            # end for
        # end for

        f.close()

        # Now create a list of spline volume objects:
        vols = []
        # Note This doesn't actually fit the volumes...just produces
        # the parameterization and knot vectors
        #nVol = 10
        for ivol in xrange(nVol):
#             print '---------%d---------'%(ivol)
#             S,u,v,w = pyspline.para3d(blocks[ivol])
#             print 'dir0:'
#             print pyspline.knots_lms(S[:,0,0,0],6,4)
#             print pyspline.knots_lms(S[:,0,-1,0],6,4)
#             print pyspline.knots_lms(S[:,-1,0,0],6,4)
#             print pyspline.knots_lms(S[:,-1,-1,0],6,4)
#             print 'u:',pyspline.knots_lms(u,6,4)

#             print 'dir1:'
#             print pyspline.knots_lms(S[0,:,0,1],6,4)
#             print pyspline.knots_lms(S[0,:,-1,1],6,4)
#             print pyspline.knots_lms(S[-1,:,0,1],6,4)
#             print pyspline.knots_lms(S[-1,:,-1,1],6,4)
#             print 'v:',pyspline.knots_lms(v,6,4)

#             print 'dir1:'
#             print pyspline.knots_lms(S[0,0,:,2],6,4)
#             print pyspline.knots_lms(S[0,-1,:,2],6,4)
#             print pyspline.knots_lms(S[-1,0,:,2],6,4)
#             print pyspline.knots_lms(S[-1,-1,:,2],6,4)
#             print 'w:',pyspline.knots_lms(w,6,4)
            if ivol<=100:#3 or ivol==14:
                vols.append(pySpline.volume(X=blocks[ivol],ku=4,kv=4,kw=4,\
                                                Nctlu=6,Nctlv=6,Nctlw=6,\
                                                no_print=self.NO_PRINT))
        self.vols = vols
        self.nVol = len(vols)
        #self.nVol = nVol
        
        return

    def _readBVol(self,*args,**kwargs):
        '''Read a bvol file and produce the volumes'''
        assert 'file_name' in kwargs,'file_name must be specified for plot3d'
        assert 'file_type' in kwargs,'file_type must be specified as binary or ascii'
        file_name = kwargs['file_name']        
        file_type = kwargs['file_type']

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
            mpiPrint('Reading Connectivity File: %s'%(file_name),self.NO_PRINT)
            self.topo = BlockTopology(file=file_name)
            sizes = []
            for ivol in xrange(self.nVol):
                sizes.append([self.vols[ivol].Nctlu,self.vols[ivol].Nctlv,
                              self.vols[ivol].Nctlw])
            self.topo.calcGlobalNumbering(sizes)
            if self.init_type != 'bvol':
                self._propagateKnotVectors()
            # end if
            self._setConnectivity()
            self._updateVolumeCoef()
        else:
            self._calcConnectivity(node_tol,edge_tol)
            self._propagateKnotVectors()
            self._setConnectivity()
            self._updateVolumeCoef()
            self.topo.writeConnectivity(file_name)
        # end if
            
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


    def _setConnectivity(self):
        # Sets the numbering based on the number of control points on each edge
       
        self.coef = []
        # Now Fill up the self.coef list:
        for ii in xrange(len(self.topo.g_index)):
            cur_coef = array([0,0,0])
            for jj in xrange(len(self.topo.g_index[ii])):
                ivol = self.topo.g_index[ii][jj][0]
                i = self.topo.g_index[ii][jj][1]
                j = self.topo.g_index[ii][jj][2]
                k = self.topo.g_index[ii][jj][3]
                cur_coef+= self.vols[ivol].coef[i,j,k]
            # end for
            self.coef.append(cur_coef/len(self.topo.g_index[ii]))
        # end for


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
        # First get the number of design groups
        print 'here'
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
            print '---------------- DG %d ------------'%(idg)
            sym = False
            knot_vectors = []
            syms = []
            for ivol in xrange(self.nVol):
                dirs = array([self.topo.edge_dir[ivol][i] for i in range(12)])
                # Check edge 0,2 and 8
                if self.topo.edges[self.topo.edge_link[ivol][0]].dg == idg:
                    u_dirs = array([self.topo.edge_dir[ivol][i] for i in [0,1,4,5]])
                    if (u_dirs == -1).any():
                        sym = True
                        syms.append(True)
                        knot_vectors.append(self.vols[ivol].tu)
                    else:
                        syms.append(False)
                        knot_vectors.append(self.vols[ivol].tu)
                    # end if

                    print self.vols[ivol].tu
                # end if
                if self.topo.edges[self.topo.edge_link[ivol][2]].dg == idg:
                    v_dirs = array([self.topo.edge_dir[ivol][i] for i in [2,3,6,7]])
                    if (v_dirs == -1).any():
                        sym = True
                        syms.append(True)
                        knot_vectors.append(self.vols[ivol].tv)
                    else:
                        syms.append(False)
                        knot_vectors.append(self.vols[ivol].tv)
                    # end if
                    print self.vols[ivol].tv
                # end if
                if self.topo.edges[self.topo.edge_link[ivol][8]].dg == idg:
                    w_dirs = array([self.topo.edge_dir[ivol][i] for i in [8,9,10,11]])
                    if (w_dirs == -1).any():
                        sym = True
                        syms.append(True)
                        knot_vectors.append(self.vols[ivol].tw)
                    else:
                        syms.append(False)
                        knot_vectors.append(self.vols[ivol].tw)
                    # end if
                    print self.vols[ivol].tw
                # end if

            # end for

            # Now blend all the knot vectors
            new_knot_vec = blendKnotVectors(knot_vectors,sym)
            print 'Belneded:','Sym is:',sym
            print new_knot_vec


            # And reset them all
            for ivol in xrange(self.nVol):
                # Check edge 0 and edge 2
                if self.topo.edges[self.topo.edge_link[ivol][0]].dg == idg:
                    self.vols[ivol].tu = new_knot_vec.copy()
                if self.topo.edges[self.topo.edge_link[ivol][2]].dg == idg:
                    self.vols[ivol].tv = new_knot_vec.copy()
                if self.topo.edges[self.topo.edge_link[ivol][8]].dg == idg:
                    self.vols[ivol].tw = new_knot_vec.copy()
            # end for
        # end for
       
        mpiPrint('Recomputing volumes...',self.NO_PRINT)

        for ivol in xrange(self.nVol):
            mpiPrint('Volume %d'%(ivol))
            self.vols[ivol].recompute()
        # end for

        return    

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
        f.write("NOPRINT\n")
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
        f.write("NOPRINT\n")
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

# ----------------------------------------------------------------------
#             Embeded Geometry Functions
# ----------------------------------------------------------------------    

    def embedGeo(self,geo):
        '''Embed a pyGeo object's surfaces into the volume'''
        self.volID = []
        self.u = []
        self.v = []
        self.w = []
        
        for icoef in xrange(len(geo.coef)):
            ivol,u0,v0,w0,D0 = self.projectPoint(geo.coef[icoef])
            self.u.append(u0)
            self.v.append(v0)
            self.w.append(w0)
            self.volID.append(ivol)
        # end for

    def updateGeo(self,geo):
        for icoef in xrange(len(geo.coef)):
            geo.coef[icoef] = self.vols[self.volID[icoef]](\
                self.u[icoef],self.v[icoef],self.w[icoef])
        #end for
        geo._updateSurfaceCoef()


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
                
    
    
