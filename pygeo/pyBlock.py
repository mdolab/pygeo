# ======================================================================
#         Imports
# ======================================================================
import os, copy
import numpy
from scipy import sparse
from scipy.sparse import linalg
from pyspline import pySpline
from .geo_utils import readNValues, BlockTopology, blendKnotVectors

class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a explicitly raised exception.
    """
    def __init__(self, message):
        msg = '\n+'+'-'*78+'+'+'\n' + '| pyBlock Error: '
        i = 16
        for word in message.split():
            if len(word) + i + 1 > 78: # Finish line and start new one
                msg += ' '*(78-i)+'|\n| ' + word + ' '
                i = 1 + len(word)+1
            else:
                msg += word + ' '
                i += len(word)+1
        msg += ' '*(78-i) + '|\n' + '+'+'-'*78+'+'+'\n'
        print(msg)
        Exception.__init__(self)

class pyBlock():
    """
    pyBlock represents a collection of pySpline Volume objects.

    It performs several functions including fitting and point
    projections.  The actual b-spline volumes are of the pySpline
    Volume type.

    Parameters
    ----------
    initType : str
       Initialization type. Only 'plot3d' is currently available.

    fileName : str
       Filename of the plot3d file to be loaded. Should have a .fmt or
       .xyz extension. Also must be in ASCII format.

    FFD : bool
       Flag to indicate that this object is to be created as an FFD.
       When this is true, no fitting is performed; The coordinates in
       the plot 3d file explicitly become the control points and
       uniform (and symmetric) knot vectors are assumed
       everywhere. This ensures a seamless FFD.
       """

    def __init__(self, initType, fileName=None, FFD=False,symmPlane=None, **kwargs):

        self.initType = initType
        self.FFD = False
        self.topo = None         # The topology of the volumes/surface
        self.vols = []           # The list of volumes (pySpline volume)
        self.nVol = None         # The total number of volumessurfaces
        self.coef  = None        # The global (reduced) set of control pts
        self.embededVolumes = {}
        self.symmPlane = symmPlane

        if initType == 'plot3d':
            self._readPlot3D(fileName, FFD=FFD, **kwargs)
        elif initType == 'create':
            pass
        else:
            raise Error('initType must be one of "plot3d" or "create". \
            ("create" is only for expert debugging)')

# ----------------------------------------------------------------------
#                     Initialization Types
# ----------------------------------------------------------------------

    def _readPlot3D(self, fileName, order='f', FFD=False,symmTol=0.001):
        """ Load a plot3D file and create the splines to go with each
        patch. See the pyBlock() docstring for more information.

        Parameters
        ----------
        order : {'f','c'}
            Internal ordering of plot3d file. Generally should be 'f'
            for fortran ordering. But could be 'c'.
        """

        binary = False # Binary read no longer supported.
        f = open(fileName, 'r')
        nVol = readNValues(f, 1, 'int', False)[0]
        sizes   = readNValues(f, nVol*3, 'int', False).reshape((nVol, 3))
        blocks = []
        for i in range(nVol):
            cur_size = sizes[i, 0]*sizes[i, 1]*sizes[i, 2]
            blocks.append(numpy.zeros(
                [sizes[i, 0], sizes[i, 1], sizes[i, 2], 3]))
            for idim in range(3):
                blocks[-1][:, :, :, idim] = readNValues(
                    f, cur_size, 'float', binary).reshape(
                    (sizes[i, 0], sizes[i, 1], sizes[i, 2]), order=order)
        f.close()

        def flip(axis, coords):
            """Flip coordinates by plane defined by 'axis'"""
            if axis.lower() == 'x':
                index = 0
            elif axis.lower() == 'y':
                index = 1
            elif axis.lower() == 'z':
                index = 2
            coords[:, :, :, index] = -coords[:, :, :, index]

            # HOWEVER just doing this results in a left-handed block (if
            # the original block was right handed). So we have to also
            # reverse ONE of the indices
            coords[:, :, :, :] = coords[::-1, :, :, :]
            # dims = coords.shape
            # for k in range(dims[2]):
            #     for j in range(dims[1]):
            #         for idim in range(3):
            #             self.coords[:, j, k, idim] = self.coords[::-1, j, k, idim]

        def symmZero(axis,coords,tol):
            """ set all coords within a certain tolerance of the symm plan to be exactly 0"""

            if axis.lower() == 'x':
                index = 0
            elif axis.lower() == 'y':
                index = 1
            elif axis.lower() == 'z':
                index = 2

            dims = coords.shape
            for k in range(dims[2]):
                for j in range(dims[1]):
                    for i in range(dims[0]):
                        error = abs(coords[i,j,k,index])
                        if error <= tol:
                            coords[i,j,k,index]=0

        if self.symmPlane is not None:
            #duplicate and mirror the blocks.
            newBlocks = []
            for block in blocks:
                newBlock = copy.deepcopy(block)
                symmZero(self.symmPlane,newBlock,symmTol)
                flip(self.symmPlane,newBlock)
                newBlocks.append(newBlock)
            # now create the appended list with double the blocks
            blocks+=newBlocks
            #Extend sizes
            newSizes = numpy.zeros([nVol*2,3],'int')
            newSizes[:nVol,:] = sizes
            newSizes[nVol:,:] = sizes
            sizes = newSizes
            #increase the volume counter
            nVol*=2

        # Now create a list of spline volume objects:
        self.vols = []

        if FFD:
            self.FFD = True

            def uniformKnots(N, k):
                """ Simple function to generate N uniform knots of
                order k"""

                knots = numpy.zeros(N+k)
                knots[0:k-1] = 0.0
                knots[-k:] = 1.0
                knots[k-1:-k+1] = numpy.linspace(0, 1, N-k+2)

                return knots

            for ivol in range(nVol):
                ku = min(4, sizes[ivol, 0])
                kv = min(4, sizes[ivol, 1])
                kw = min(4, sizes[ivol, 2])

                # A uniform knot vector is ok and we won't have to
                # propagate the vectors since they are by
                # construction symmetric

                self.vols.append(pySpline.Volume(
                    ku=ku, kv=kv, kw=kw, coef=blocks[ivol],
                    tu=uniformKnots(sizes[ivol, 0], ku),
                    tv=uniformKnots(sizes[ivol, 1], kv),
                    tw=uniformKnots(sizes[ivol, 2], kw)))

                # Generate dummy original data:
                U = numpy.zeros((3, 3, 3))
                V = numpy.zeros((3, 3, 3))
                W = numpy.zeros((3, 3, 3))

                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            U[i, j, k] = float(i)/2
                            V[i, j, k] = float(j)/2
                            W[i, j, k] = float(k)/2

                # Evaluate the spline "original data"
                self.vols[-1].X = self.vols[-1](U, V, W)

                self.vols[-1].origData = True
                self.vols[-1].Nu = 3
                self.vols[-1].Nv = 3
                self.vols[-1].Nw = 3
            # end for (ivol loop)

            self.nVol = len(self.vols)
            self._calcConnectivity(1e-4, 1e-4)
            nCtl = self.topo.nGlobal
            self.coef = numpy.zeros((nCtl, 3))
            self._setVolumeCoef()

            for ivol in range(self.nVol):
                self.vols[ivol].setFaceSurfaces()
                self.vols[ivol].setEdgeCurves()

        else: # (not FFD check --- must run fitGlobal after!)
            # Note This doesn't actually fit the volumes...just produces
            # the parametrization and knot vectors
            for ivol in range(nVol):
                self.vols.append(pySpline.Volume(
                    X=blocks[ivol], ku=4, kv=4, kw=4,
                    nCtlu=4, nCtlv=4, nCtlw=4,
                    recompute=False))
            self.nVol = len(self.vols)
        # end if (FFD Check)


    def fitGlobal(self, greedyReorder=False):
        """
        Determine the set of b-spline coefficients that best fits the
        set of volumes in the global sense. This is *required* for
        non-FFD creation.

        Parameters
        ----------
        greedyReorder : bool
            Flag to compute ordering of initial mesh in a greedy
            ordering sense.
            """

        nCtl = self.topo.nGlobal
        origTopo = copy.deepcopy(self.topo)

        print(' -> Creating global numbering')
        sizes = []
        for ivol in range(self.nVol):
            sizes.append([self.vols[ivol].Nu,
                          self.vols[ivol].Nv,
                          self.vols[ivol].Nw])

        # Get the Global number of the original data
        origTopo.calcGlobalNumbering(sizes, greedyReorder=greedyReorder)
        N = origTopo.nGlobal
        print(' -> Creating global point list')
        pts = numpy.zeros((N, 3))
        for ii in range(N):
            pts[ii] = self.vols[origTopo.gIndex[ii][0][0]].X[
                origTopo.gIndex[ii][0][1],
                origTopo.gIndex[ii][0][2],
                origTopo.gIndex[ii][0][3]]

        # Get the maximum k (ku, kv, kw for each vol)
        kmax = 2
        for ivol in range(self.nVol):
            kmax = max(kmax, self.vols[ivol].ku, self.vols[ivol].kv,
                       self.vols[ivol].kw)

        nnz = N*kmax*kmax*kmax
        vals = numpy.zeros(nnz)
        rowPtr = [0]
        colInd = numpy.zeros(nnz, 'intc')
        for ii in range(N):
            ivol = origTopo.gIndex[ii][0][0]
            i = origTopo.gIndex[ii][0][1]
            j = origTopo.gIndex[ii][0][2]
            k = origTopo.gIndex[ii][0][3]

            u = self.vols[ivol].U[i, j, k]
            v = self.vols[ivol].V[i, j, k]
            w = self.vols[ivol].W[i, j, k]

            vals, colInd = self.vols[ivol].getBasisPt(
                u, v, w, vals, rowPtr[ii], colInd, self.topo.lIndex[ivol])
            kinc = self.vols[ivol].ku*self.vols[ivol].kv*self.vols[ivol].kw
            rowPtr.append(rowPtr[-1] + kinc)

        # Now we can crop out any additional values in colInd and vals
        vals = vals[:rowPtr[-1]]
        colInd = colInd[:rowPtr[-1]]

        # Now make a sparse matrix, the N, and N^T * N factors, sovle
        # and set:
        NN = sparse.csr_matrix((vals, colInd, rowPtr))
        NNT = NN.T
        NTN = NNT*NN
        solve = linalg.dsolve.factorized(NTN)
        self.coef = numpy.zeros((nCtl, 3))
        for idim in range(3):
            self.coef[:, idim] = solve(NNT*pts[:, idim])

        self._updateVolumeCoef()
        for ivol in range(self.nVol):
            self.vols[ivol].setFaceSurfaces()
            self.vols[ivol].setEdgeCurves()

# ----------------------------------------------------------------------
#                     Topology Information Functions
# ----------------------------------------------------------------------

    def doConnectivity(self, fileName=None, nodeTol=1e-4, edgeTol=1e-4,
                       greedyReorder=False):
        """
        This function is used if a separate fitting topology is
        required for non-FFD creations. The sequence of calls is given
        in the examples section.

        Parameters
        ----------
        fileName : str
            If the filename exists, read in the topology. Otherwise, write a
            an initial default topology
        nodeTol : float
            Tolerance for co-incidient nodes
        edgeTol : float
            Tolerance for co-incidient mid points of edges
        greedyReorder : bool
            Flag to reorder numbering in a greedy form.
            """

        if fileName is not None and os.path.isfile(fileName):
            print(' ')
            print('Reading Connectivity File: %s'%(fileName))
            self.topo = BlockTopology(file=fileName)
            self._propagateKnotVectors()
        else:
            print(' ')
            self._calcConnectivity(nodeTol, edgeTol)
            self._propagateKnotVectors()
            if fileName is not None:
                print('Writing Connectivity File: %s'%(fileName))
                self.topo.writeConnectivity(fileName)

        sizes = []
        for ivol in range(self.nVol):
            sizes.append([self.vols[ivol].nClu, self.vols[ivol].nCtlv,
                              self.vols[ivol].nCtlw])
        self.topo.calcGlobalNumbering(sizes, greedyReorder=greedyReorder)

    def _calcConnectivity(self, nodeTol, edgeTol):
        """Determine the blocking connectivity

        Parameters
        ----------
        nodeTol : float
            Tolerance for identical nodes
        edgeTol :float
            Tolerance for midpoint of edges to determine if they are the same
        """
        coords = numpy.zeros((self.nVol, 26, 3))

        for ivol in range(self.nVol):
            for icorner in range(8):
                coords[ivol, icorner] = \
                             self.vols[ivol].getOrigValueCorner(icorner)

            for iedge in range(12):
                coords[ivol, 8+iedge] = \
                             self.vols[ivol].getMidPointEdge(iedge)

            for iface in range(6):
                coords[ivol, 20+iface] = \
                             self.vols[ivol].getMidPointFace(iface)

        self.topo = BlockTopology(coords, nodeTol=nodeTol, edgeTol=edgeTol)
        sizes = []
        for ivol in range(self.nVol):
            sizes.append([self.vols[ivol].nCtlu, self.vols[ivol].nCtlv,
                          self.vols[ivol].nCtlw])
        self.topo.calcGlobalNumbering(sizes)

    def printConnectivity(self):
        """
        Print the connectivity information to the screen
        """
        self.topo.printConnectivity()

    def _propagateKnotVectors(self):
        """ Propagate the knot vectors to make consistent"""

        nDG = -1
        ncoef = []
        for i in range(self.topo.nEdge):
            if self.topo.edges[i].dg > nDG:
                nDG = self.topo.edges[i].dg
                ncoef.append(self.topo.edges[i].N)

        nDG += 1

        for ivol in range(self.nVol):
            dgU = self.topo.edges[self.topo.edgeLink[ivol][0]].dg
            dgV = self.topo.edges[self.topo.edgeLink[ivol][2]].dg
            dgW = self.topo.edges[self.topo.edgeLink[ivol][8]].dg
            self.vols[ivol].nCtlu = ncoef[dgU]
            self.vols[ivol].nCtlv = ncoef[dgV]
            self.vols[ivol].nCtlw = ncoef[dgW]
            if self.vols[ivol].ku < self.vols[ivol].nCtlu:
                if self.vols[ivol].nCtlu > 4:
                    self.vols[ivol].ku = 4
                else:
                    self.vols[ivol].ku = self.vols[ivol].nCtlu

            if self.vols[ivol].kv < self.vols[ivol].nCtlv:
                if self.vols[ivol].nCtlv > 4:
                    self.vols[ivol].kv = 4
                else:
                    self.vols[ivol].kv = self.vols[ivol].nCtlv

            if self.vols[ivol].kw < self.vols[ivol].nCtlw:
                if self.vols[ivol].nCtlw > 4:
                    self.vols[ivol].kw = 4
                else:
                    self.vols[ivol].kw = self.vols[ivol].nCtlw

            self.vols[ivol].calcKnots()
            # Now loop over the number of design groups, accumulate all
            # the knot vectors that correspond to this dg, then merge them all

        for idg in range(nDG):
            knotVectors = []
            flip = []
            for ivol in range(self.nVol):
                for iedge in range(12):
                    if self.topo.edges[
                        self.topo.edgeLink[ivol][iedge]].dg == idg:

                        if self.topo.edgeDir[ivol][iedge] == -1:
                            flip.append(True)
                        else:
                            flip.append(False)

                        if iedge in [0, 1, 4, 5]:
                            knotVec = self.vols[ivol].tu
                        elif iedge in [2, 3, 6, 7]:
                            knotVec = self.vols[ivol].tv
                        elif iedge in [8, 9, 10, 11]:
                            knotVec = self.vols[ivol].tw

                        if flip[-1]:
                            knotVectors.append((1-knotVec)[::-1].copy())
                        else:
                            knotVectors.append(knotVec)

            # Now blend all the knot vectors
            newKnotVec = blendKnotVectors(knotVectors, False)
            newKnotVecFlip = (1-newKnotVec)[::-1]

            # And now reset them all:
            counter = 0
            for ivol in range(self.nVol):
                for iedge in range(12):
                    if self.topo.edges[
                        self.topo.edgeLink[ivol][iedge]].dg == idg:

                        if iedge in [0, 1, 4, 5]:
                            if flip[counter]:
                                self.vols[ivol].tu = newKnotVecFlip.copy()
                            else:
                                self.vols[ivol].tu = newKnotVec.copy()

                        elif iedge in [2, 3, 6, 7]:
                            if flip[counter]:
                                self.vols[ivol].tv = newKnotVecFlip.copy()
                            else:
                                self.vols[ivol].tv = newKnotVec.copy()

                        elif iedge in [8, 9, 10, 11]:
                            if flip[counter]:
                                self.vols[ivol].tw = newKnotVecFlip.copy()
                            else:
                                self.vols[ivol].tw = newKnotVec.copy()

                        counter += 1
                    # end if (dg match)
                self.vols[ivol].setCoefSize()
            # end for (edge loop)
        # end for (dg loop)

# ----------------------------------------------------------------------
#                        Output Functions
# ----------------------------------------------------------------------
    def writeTecplot(self, fileName, vols=True, coef=True, orig=False,
                     volLabels=False, edgeLabels=False, nodeLabels=False):
        """Write a tecplot visualization of the pyBlock object.

        Parameters
        ----------
        fileName : str
            Filename of tecplot file. Should have a .dat extension

        vols : bool. Default is True
            Flag to write interpolated volumes

        coef : bool. Default is True
            Flag to write spline control points

        orig : bool. Default is True
            Flag to write original data (if it exists)

        volLabels: bool. Default is True
            Flag to write volume labels in a separate tecplot file; filename
            is derived from the supplied fileName.

        edgeLabels: bool. Default is False
            Flag to write edge labels in a separate tecplot file; filename
            is derived from the supplied fileName.

        nodeLabels: bool. Default is False
            Flag to write node labels in a separate tecplot file; filename
            is derived from the supplied fileName.
            """

        # Open File and output header
        f = pySpline.openTecplot(fileName, 3)

        if vols:
            for ivol in range(self.nVol):
                self.vols[ivol].computeData()
                pySpline.writeTecplot3D(f, 'interpolated',
                                        self.vols[ivol].data)
        if orig:
            for ivol in range(self.nVol):
                pySpline.writeTecplot3D(f, 'orig_data', self.vols[ivol].X)

        if coef:
            for ivol in range(self.nVol):
                pySpline.writeTecplot3D(f, 'control_pts', self.vols[ivol].coef)

        # ---------------------------------------------
        #    Write out labels:
        # ---------------------------------------------
        if volLabels:
            # Split the filename off
            dirName, fileName = os.path.split(fileName)
            fileBaseName, fileExtension = os.path.splitext(fileName)
            labelFilename = dirName+'./' + fileBaseName+'.vol_labels.dat'
            f2 = open(labelFilename, 'w')
            for ivol in range(self.nVol):
                midu = self.vols[ivol].nCtlu//2
                midv = self.vols[ivol].nCtlv//2
                midw = self.vols[ivol].nCtlw//2
                textString = 'TEXT CS=GRID3D, X=%f, Y=%f, Z=%f, T=\"V%d\"\n'% (
                    self.vols[ivol].coef[midu, midv, midw, 0],
                    self.vols[ivol].coef[midu, midv, midw, 1],
                    self.vols[ivol].coef[midu, midv, midw, 2], ivol)
                f2.write('%s'%(textString))
            f2.close()

        if edgeLabels:
            # Split the filename off
            dirName, fileName = os.path.split(fileName)
            fileBaseName, fileExtension = os.path.splitext(fileName)
            labelFilename = dirName+'./' + fileBaseName+'.edge_labels.dat'
            f2 = open(labelFilename, 'w')
            for ivol in range(self.nVol):
                for iedge in range(12):
                    pt = self.vols[ivol].edgeCurves[iedge](0.5)
                    edgeID = self.topo.edgeLink[ivol][iedge]
                    textString = 'TEXT CS=GRID3D X=%f, Y=%f, Z=%f, T=\"E%d\"\n'% (
                        pt[0], pt[1], pt[2], edgeID)
                    f2.write('%s'%(textString))
            f2.close()

        if nodeLabels:
            # First we need to figure out where the corners actually *are*
            nNodes = len(numpy.unique(self.topo.nodeLink.flatten()))
            nodeCoord = numpy.zeros((nNodes, 3))

            for i in range(nNodes):
                # Try to find node i
                for ivol in range(self.nVol):
                    for inode  in range(8):
                        if self.topo.nodeLink[ivol][inode] == i:
                            coordinate = self.vols[ivol].getValueCorner(inode)

                nodeCoord[i] = coordinate

            # Split the filename off
            dirName, fileName = os.path.split(fileName)
            fileBaseName, fileExtension = os.path.splitext(fileName)
            labelFilename = dirName+'./' + fileBaseName+'.node_labels.dat'
            f2 = open(labelFilename, 'w')
            for i in range(nNodes):
                textString = 'TEXT CS=GRID3D, X=%f, Y=%f, Z=%f, T=\"n%d\"\n'% (
                    nodeCoord[i][0], nodeCoord[i][1], nodeCoord[i][2], i)
                f2.write('%s'%(textString))
            f2.close()

        pySpline.closeTecplot(f)

    def writePlot3d(self, fileName):
        """Write the grid to a plot3d file. This isn't efficient as it
        used ASCII format. Only useful for quick visualizations

        Parameters
        ----------
        fileName : plot3d file name.
            Should end in .xyz
        """

        sizes = []
        for ivol in range(self.nVol):
            sizes.append(self.vols[ivol].Nu)
            sizes.append(self.vols[ivol].Nv)
            sizes.append(self.vols[ivol].Nw)

        f = open(fileName, 'w')
        f.write('%d\n'%(self.nVol))
        numpy.array(sizes).tofile(f, sep=" ")
        f.write('\n')
        for ivol in range(self.nVol):
            vals = self.vols[ivol](self.vols[ivol].U, self.vols[ivol].V,
                                   self.vols[ivol].W)
            vals[:, :, :, 0].flatten(order="F").tofile(f, sep="\n")
            f.write('\n')
            vals[:, :, :, 1].flatten(order="F").tofile(f, sep="\n")
            f.write('\n')
            vals[:, :, :, 2].flatten(order="F").tofile(f, sep="\n")
            f.write('\n')
        f.close()

    def writePlot3dCoef(self, fileName):
        """Write the *coefficients* of the volumes to a plot3d
        file.

        Parameters
        ----------
        fileName : plot3d file name.
            Should end in .fmt
        """

        sizes = []
        for ivol in range(self.nVol):
            sizes.append(self.vols[ivol].nCtlu)
            sizes.append(self.vols[ivol].nCtlv)
            sizes.append(self.vols[ivol].nCtlw)

        f = open(fileName, 'w')
        f.write('%d\n'% (self.nVol))
        numpy.array(sizes).tofile(f, sep=" ")
        f.write('\n')
        for ivol in range(self.nVol):
            vals = self.vols[ivol].coef
            vals[:, :, :, 0].flatten(order="F").tofile(f, sep="\n")
            f.write('\n')
            vals[:, :, :, 1].flatten(order="F").tofile(f, sep="\n")
            f.write('\n')
            vals[:, :, :, 2].flatten(order="F").tofile(f, sep="\n")
            f.write('\n')
        f.close()

# ----------------------------------------------------------------------
#               Update Functions
# ----------------------------------------------------------------------
    def _updateVolumeCoef(self):
        """Copy the pyBlock list of control points back to the volumes"""
        for ii in range(len(self.coef)):
            for jj in range(len(self.topo.gIndex[ii])):
                ivol  = self.topo.gIndex[ii][jj][0]
                i     = self.topo.gIndex[ii][jj][1]
                j     = self.topo.gIndex[ii][jj][2]
                k     = self.topo.gIndex[ii][jj][3]
                self.vols[ivol].coef[i, j, k] = self.coef[ii].real.astype('d')

    def _setVolumeCoef(self):
        """Set the global coefficient array self.coef from the
        coefficients on the volumes. This typically needs only to be
        called once when the object is created"""

        self.coef = numpy.zeros((self.topo.nGlobal, 3))
        for ivol in range(self.nVol):
            vol = self.vols[ivol]
            for i in range(vol.nCtlu):
                for j in range(vol.nCtlv):
                    for k in range(vol.nCtlw):
                        self.coef[self.topo.lIndex[ivol][i, j, k]] = \
                            vol.coef[i, j, k]

    def calcdPtdCoef(self, ptSetName):
        """Calculate the (fixed) derivative of a set of embedded
        points with respect to the b-spline coefficients. This
        derivative consists of the b-spline basis functions

        Parameters
        ----------
        ptSetName : str
            The name of the point set to use.
        """

        # Extract values to make the code a little easier to read:
        volID = self.embededVolumes[ptSetName].volID
        u = self.embededVolumes[ptSetName].u
        v = self.embededVolumes[ptSetName].v
        w = self.embededVolumes[ptSetName].w
        N = self.embededVolumes[ptSetName].N

        # Get the maximum k (ku or kv for each volume)
        kmax = 2
        for ivol in range(self.nVol):
            kmax = max(kmax, self.vols[ivol].ku, self.vols[ivol].kv,
                       self.vols[ivol].kw)

        # Maximum number of non-zeros in jacobian
        nnz = N*kmax*kmax*kmax
        vals = numpy.zeros(nnz)
        rowPtr = [0]
        colInd = numpy.zeros(nnz, 'intc')
        for i in range(N):
            kinc = self.vols[volID[i]].ku*\
                   self.vols[volID[i]].kv*\
                   self.vols[volID[i]].kw
            vals, colInd = self.vols[volID[i]].getBasisPt(\
                u[i], v[i], w[i], vals, rowPtr[i], colInd,
                self.topo.lIndex[volID[i]])

            rowPtr.append(rowPtr[-1] + kinc)
            if self.embededVolumes[ptSetName].mask is not None:
                if not i in self.embededVolumes[ptSetName].mask:
                    # Kill the values we just added
                    vals[rowPtr[-2]:rowPtr[-1]] = 0.0

        # Now we can crop out any additional values in colInd and vals
        vals    = vals[:rowPtr[-1]]
        colInd = colInd[:rowPtr[-1]]
        # Now make a sparse matrix iff we actually have coordinates
        if N > 0:
            self.embededVolumes[ptSetName].dPtdCoef = sparse.csr_matrix(
                (vals, colInd, rowPtr), shape=[N, len(self.coef)])

    def getAttachedPoints(self, ptSetName):
        """
        Return all the volume points for an embedded volume with name ptSetName.

        Parameters
        ----------
        ptSetName : str
            Name of a point set added with attachPoints()

        Returns
        -------
        coordinates : numpy array (Nx3)
            The coordinates of the embedded points. If a mask was used,
            only the points corresponding to the indices in mask will be
            non-zero in the array.
            """

        volID = self.embededVolumes[ptSetName].volID
        u     = self.embededVolumes[ptSetName].u
        v     = self.embededVolumes[ptSetName].v
        w     = self.embededVolumes[ptSetName].w
        N     = self.embededVolumes[ptSetName].N
        mask  = self.embededVolumes[ptSetName].mask
        coordinates = numpy.zeros((N, 3))

        # This evaluation is fast enough we don't really care about
        # only looping explictly over the mask values
        for iVol in self.embededVolumes[ptSetName].indices:
            indices =  self.embededVolumes[ptSetName].indices[iVol]
            u = self.embededVolumes[ptSetName].u[indices]
            v = self.embededVolumes[ptSetName].v[indices]
            w = self.embededVolumes[ptSetName].w[indices]
            coords = self.vols[iVol](u, v, w)
            coordinates[indices, :] = coords

        if mask is not None:
            # Explictly zero anything not in mask to ensure no-one
            # accidently uses it when they should not
            tmp = coordinates.copy() # Create copy
            coordinates[:,:] = 0.0   # Completely zero
            coordinates[mask, :] = tmp[mask, :] # Just put back the ones we wnat.

        return coordinates

# ----------------------------------------------------------------------
#             Embedded Geometry Functions
# ----------------------------------------------------------------------

    def attachPoints(self, coordinates, ptSetName, interiorOnly=False,
                     faceFreeze=None, eps=1e-12, **kwargs):
        """Embed a set of coordinates into the volumes. This is the
        main high level function that is used by DVGeometry when
        pyBlock is used as an FFD.

        Parameters
        ----------
        coordinates : array, size (N,3)
            The coordinates to embed in the object
        ptSetName : str
            The name given to this set of coordinates.
        interiorOnly : bool
            Project only points that lie fully inside the volume
        faceFreeze :
            A dictionary of lists of strings specifying which faces should be
            'frozen'. Each dictionary represents one block in the FFD.
            This is only used with child FFD's in DVGeometry.
            For example if faceFreeze =['0':['iLow'],'1':[]], then the
            plane of control points corresponding to i=0, and i=1, in block '0'
            will not be able to move in DVGeometry.
        eps : float
            Physical tolerance to which to converge Newton search
        kwargs : dict
            kwargs pass through to the actual projectPoints() function
        """

        # Project Points, if some were actually passed in:
        if coordinates is not None:
            if not interiorOnly:
                volID, u, v, w, D = self.projectPoints(
                    coordinates, checkErrors=True, eps=eps, **kwargs)
                self.embededVolumes[ptSetName] = EmbeddedVolume(volID, u, v, w)
            else:
                volID, u, v, w, D = self.projectPoints(
                    coordinates, checkErrors=False, eps=eps, **kwargs)

                mask = []
                for i in range(len(D)):
                    Dnrm = numpy.linalg.norm(D[i])
                    if Dnrm < 50*eps: # Sufficiently inside
                        mask.append(i)

                # Now that we have the mask we can create the embedded volume
                self.embededVolumes[ptSetName] = EmbeddedVolume(volID, u, v, w, mask)
        # end if (Coordinate not none check)

        return

# ----------------------------------------------------------------------
#             Geometric Functions
# ----------------------------------------------------------------------

    def projectPoints(self, x0, eps=1e-12, checkErrors=True, nIter=100):
        """Project a set of points x0, into any one of the volumes. It
        returns the the volume ID, u, v, w, D of the point in volID or
        closest to it.

        This is still *technically* a inefficient brute force search,
        but it uses some heuristics to give a much more efficient
        algorithm. Basically, we use the volume the last point was
        projected in as a 'good guess' as to what volume the current
        point falls in. This works since subsequent points are usually
        close together. This will not help for randomly distributed
        points.

        Parameters
        ----------
        x0 : array of points (Nx3 array)
            The list or array of points to use
        eps : float
            Physical tolerance to which to converge Newton search
        checkErrors : bool
            Flag to print out the error is points have not been projected
            to tolerance eps.
        nIter : int
            Maximum number of Newton iterations to perform. The
            default of 100 should be sufficient for points that
            **actually** lie inside the volume, except for
            pathological or degenerate FFD volumes
        """

        # Make sure we are dealing with a 2D "Nx3" list of points
        x0 = numpy.atleast_2d(x0)
        N = len(x0)
        volID = numpy.zeros(N, 'intc')
        u     = numpy.zeros(N)
        v     = numpy.zeros(N)
        w     = numpy.zeros(N)
        D     = 1e10*numpy.ones((N, 3))

        # Starting list is just [0, 1, 2, ..., nVol-1]
        volList = numpy.arange(self.nVol)
        u0 = 0.0
        v0 = 0.0
        w0 = 0.0

        for i in range(N):
            for j in range(self.nVol):
                iVol = volList[j]
                u0, v0, w0, D0 = self.vols[iVol].projectPoint(
                    x0[i], eps=eps, nIter=nIter)

                D0Norm = numpy.linalg.norm(D0)
                # If the new distance is less than the previous best
                # distance, set the volID, u, v, w, since this may be
                # best we can do:
                if D0Norm < numpy.linalg.norm(D[i]):
                    volID[i] = iVol

                    u[i] = u0
                    v[i] = v0
                    w[i] = w0
                    D[i] = D0.real

                # Now, if D0 is close enough to our tolerance, we can
                # exit the loop since we know we won't do any better
                if (D0Norm < eps*50):
                    break
            # end for (volume loop)

            # Shuffle the order of the volList such that the last
            # volume used (iVol or volList[j]) is at the start of the
            # list and the remainder are shuffled towards the back
            volList = numpy.hstack([iVol, volList[:j], volList[j+1:]])
        # end for (length of x0)

        # If desired check the errors and print warnings:
        if checkErrors:
            # Loop back through the points and determine which ones are
            # bad (> 50*eps) and print them to the screen:
            counter = 0
            DMax = 0.0
            DRms = 0.0
            badPts = []
            for i in range(len(x0)):
                nrm = numpy.linalg.norm(D[i])
                if nrm > DMax:
                    DMax = nrm

                DRms += nrm**2
                if nrm > eps*50:
                    counter += 1
                    badPts.append([x0[i], D[i]])

            if len(x0) > 0:
                DRms = numpy.sqrt(DRms / len(x0))
            else:
                DRms = None

            # Check to see if we have bad projections and print a warning:
            if counter > 0:
                print(' -> Warning: %d point(s) not projected to tolerance: \
                %g\n.  Max Error: %12.6g ; RMS Error: %12.6g'%(counter, eps, DMax, DRms))
                print('List of Points is: (pt, delta):')
                for i in range(len(badPts)):
                    print('[%12.5g %12.5g %12.5g] [%12.5g %12.5g %12.5g]'%(
                        badPts[i][0][0],
                        badPts[i][0][1],
                        badPts[i][0][2],
                        badPts[i][1][0],
                        badPts[i][1][1],
                        badPts[i][1][2]))

        return volID, u, v, w, D

    def getBounds(self):
        """Determine the extents of the set of volumes

        Returns
        -------
        xMin : array of length 3
            Lower corner of the bounding box
        xMax : array of length 3
            Upper corner of the bounding box
            """

        Xmin, Xmax = self.vols[0].getBounds()
        for iVol in range(1, self.nVol):
            Xmin0, Xmax0 = self.vols[iVol].getBounds()
            for iDim in range(3):
                Xmin[iDim] = min(Xmin[iDim], Xmin0[iDim])
                Xmax[iDim] = max(Xmax[iDim], Xmax0[iDim])

        return Xmin, Xmax

class EmbeddedVolume(object):
    """A Container class for a set of embedded volume points

    Parameters
    ----------
    volID : int array
        Index of the volumes this point is located in
    u, v, w, : float arrays
        Parametric locations of the coordinates in volID
    mask : array of indices
        Mask is an array of length less than N (N = len of
        u,v,w,volID). It contains only a subset of the indices to
        be used. It is used for DVGeometry's children
        implementation.
        """

    def __init__(self, volID, u, v, w, mask=None):
        self.volID = numpy.array(volID)
        self.u = numpy.array(u)
        self.v = numpy.array(v)
        self.w = numpy.array(w)
        self.N = len(self.u)
        self.indices = {}
        self.dPtdCoef = None
        self.dPtdX    = None
        self.mask     = mask

        # Get the number of unique volumes this point set requires:
        uniqueVolIDs = numpy.unique(self.volID)

        for iVol in uniqueVolIDs:
            self.indices[iVol] = numpy.where(self.volID == iVol)[0]
