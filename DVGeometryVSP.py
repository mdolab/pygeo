# ======================================================================
#         Imports
# ======================================================================
from __future__ import print_function
import copy, time, tempfile
import os
from collections import OrderedDict
import numpy
from scipy import sparse
from mpi4py import MPI
from pyspline import pySpline

class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a explicitly raised exception.
    """
    def __init__(self, message):
        msg = '\n+'+'-'*78+'+'+'\n' + '| DVGeometryVSP Error: '
        i = 22
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

class DVGeometryVSP(object):
    """A class for manipulating VSP geometry

    The purpose of the DVGeometryVSP class is to provide translation
    of the VSP geometry engine to externally supplied surfaces. This
    allows the use of VSP design variables to control the MACH
    framework.

    There are several import limitations:

    1. Since VSP is surface based only, it cannot be used to
    parameterize a geometry that doesn't lie on the surface. This
    means it cannot be used for structural analysis. It is generally
    possible use most of the constraints DVConstraints since most of
    those points lie on the surface.

    2. It cannot handle *moving* intersection. A geometry with static
    intersections is fine as long as the g

    3. It does not support complex numbers for the complex-step
    method.

    4. It does not surpport separate configurations.

    Parameters
    ----------
    vspFile : str
       filename of .vsp3 file.

    desFile : str
       filename of the design parameter file

    comm : MPI Intra Comm
       Comm on which to build operate the object. This is used to
       perform embarasisngly parallel finite differencing.

    vspScriptCmd : str
       If the vspScript command isn't in your path, supply the full path
       to the executate.

    Examples
    --------
    The general sequence of operations for using DVGeometry is as follows::
      >>> from pygeo import *
      >>> DVGeo = DVGeometryVSP("wing.vsp3", "wing.des", MPI_COMM_WORLD)
      >>> # Add a set of coordinates Xpt into the object
      >>> DVGeo.addPointSet(Xpt, 'myPoints')
      >>>

    """
    def __init__(self, vspFile, desFile, comm, vspScriptCmd=None, *args, **kwargs):
        self.points = OrderedDict()
        self.updated = {}
        self.finalized = False
        self.dtype = 'd'

        # Just store the vspFile as string
        self.vspFile = vspFile
        self.vspScriptCmd=vspScriptCmd
        if self.vspScriptCmd is None:
            self.vspScriptCmd = 'vspscript'

        # Create a secure directory in tmp for each processor to play
        # with. Put the rank at the end to potentially help with
        # debugging.
        self.mydir = tempfile.mkdtemp(suffix='-%d'%MPI.COMM_WORLD.rank)

        # Run the update. This will also set the conn on the first pass
        self.conn = None
        self.pts0, self.conn = self._getUpdatedSurface(self.desFile)
        self.pts = None # The current set of points

        # Parse and add all the DVs from the DVFile. This will be only
        # time we need this desFile so it isn't stored as a member in
        # the class.
        self.DVs = OrderedDict()
        self._parseDesignFile(desFile)

    def __del__(self):
        """ On delete we can remove the myid"""
        shutil.rmtree(self.myid)

    def _getUpdatedSurface(self, desFile):
        """Return the updated surface already converted to an unstructured
        format. Note that this routine is safe to call in an
        embarrassing parallel fashion. That is it can be called with
        different 'desFile's on different processors and they won't
        interfere with each other

        Parameters
        ----------
        desFile : str
            Filename containing the variables to use for this update.
        """

        # Create a temporary export file name.
        tmpExport = os.path.join(self.mydir, "export.x")
        cmd = "void main() {\
        ReadVSPFile(\"%s\"); \
        ReadApplyDESFile(\"%s\"); \
        ExportFile(\"%s\", 0, EXPORT_PLOT3D); \
        }" %(self.vspFile, desFile, tmpExport)

        # Now create the command file
        tmpScript = os.path.join(self.mydir, 'export.vspscript')
        f = open(tmpScript, 'w')
        f.write(cmd)
        f.close()

        # Now execute the command:
        os.system("%s -script %s\n"%(self.vspScriptCmd, tmpScript))

        # Now we load in this updated plot3d file. Compute the
        # connectivity if not already done.
        computeConn = False
        if self.conn is None:
            computeConn = True
        pts, conn = self._readPlot3DSurfFile(tmpExport, computeConn)

        return pts, conn

    def _readPlot3DSurfFile(self, fileName, computeConn=False):
        """Read a plot3d file and return the points and connectivity"""

        f = open(fileName, 'r')
        nSurf = numpy.fromfile(f, 'int', count=1, sep=' ')[0]
        sizes = numpy.fromfile(f, 'int', count=3*nSurf, sep=' ').reshape((nSurf, 3))
        nPts = 0; nElem = 0
        for i in range(nSurf):
            curSize = sizes[i, 0]*sizes[i, 1]
            nPts += curSize
            nElem += (sizes[i, 0]-1)*(sizes[i, 1]-1)

        # Generate the uncompacted point and connectivity list:
        pts = numpy.zeros((nPts, 3), dtype=self.dtype)
        if computeConn:
            conn = numpy.zeros((nElem, 4), dtype='intc')
        else:
            conn = self.conn

        nodeCount = 0
        elemCount = 0
        for iSurf in range(nSurf):
            curSize = sizes[iSurf, 0]*sizes[iSurf, 1]
            for idim in range(3):
                pts[nodeCount:nodeCount+curSize, idim] = (
                    numpy.fromfile(f, 'float', curSize, sep=' '))
            if computeConn:
                # Add in the connectivity.
                iSize = sizes[iSurf, 0]
                for j in range(sizes[iSurf, 1]-1):
                    for i in range(sizes[iSurf, 0]-1):
                        conn[elemCount, 0] = nodeCount + j*iSize + i
                        conn[elemCount, 1] = nodeCount + j*iSize + i+1
                        conn[elemCount, 2] = nodeCount + (j+1)*iSize + i+1
                        conn[elemCount, 3] = nodeCount + (j+1)*iSize + i
                        elemCount += 1
            nodeCount += curSize

        return pts, conn

    def addPointSet(self, points, ptName, **kwargs):

        """
        Add a set of coordinates to DVGeometry

        The is the main way that geometry, in the form of a coordinate
        list is given to DVGeoemtry to be manipulated.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed. These cordinates *should* all
            project into the interior of the FFD volume.
        ptName : str
            A user supplied name to associate with the set of
            coordinates. This name will need to be provided when
            updating the coordinates or when getting the derivatives
            of the coordinates.
        """

        # save this name so that we can zero out the jacobians properly
        self.ptSetNames.append(ptName)
        self.points[ptName] = True # ADFlow checks self.poitns to see
                                   # if something is added or not.

        points = numpy.array(points).real.astype('d')
        # Attach the points to the original surface using the fast ADT
        # projection code from pySpline. Note that we convert to
        # 1-based indexing for conn here.
        faceID, uv = pySpline.libspline.adtprojections(
            self.pts0.T, (self.conn+1).T, points.T)
        uv = uv.T
        faceID -= 1 # Convert back to zero-based indexing.

        # Compute the offsets by evaluating the new points.
        # eval which should be fast enough
        newPts = numpy.zeros_like(points)
        for idim in range(3):
            newPts[:, idim] = \
                              (1-uv[:, 0])*(1 - uv[:, 1]) * self.pts0[self.conn[faceID, 0], idim] + \
                              (  uv[:, 0])*(1 - uv[:, 1]) * self.pts0[self.conn[faceID, 1], idim] + \
                              (  uv[:, 0])*(    uv[:, 1]) * self.pts0[self.conn[faceID, 2], idim] + \
                              (1-uv[:, 0])*(    uv[:, 1]) * self.pts0[self.conn[faceID, 3], idim]
        offset = newPts - points

        # Create the little class with the data
        self.pointSets[ptName] = PointSet(points, faceID, uv, offset,
                                          self.conn, len(self.pts0))
        self.updated[ptName] = False

    def setDesignVars(self, dvDict):
        """
        Standard routine for setting design variables from a design
        variable dictionary.

        Parameters
        ----------
        dvDict : dict
            Dictionary of design variables. The keys of the dictionary
            must correspond to the design variable names. Any
            additional keys in the dfvdictionary are simply ignored.
            """

        for key in dvDict:
            if key in self.DVs:
                dvDict[key].value = dvDict[key]

        # When we set the design variables we will also compute the
        # new surface self.pts. Only one proc needs to do this.
        if self.comm.rank == 0:
            desFile = os.path.join(self.mydir, 'desFile')
            self._createDesginFile(desFile)
            self.pts, conn = self.getUpdatedSurface(desFile)
        self.pts = self.comm.bcast(self.pts)

        # We will also compute the jacobian so it is also up to date
        self._computeSurfJacobian()

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

    def getValues(self):
        """
        Generic routine to return the current set of design
        variables. Values are returned in a dictionary format
        that would be suitable for a subsequent call to setValues()

        Returns
        -------
        dvDict : dict
            Dictionary of design variables
        """
        dvDict = OrderedDict()
        for dvName in self.DVs:
            dvDict[dvName] = self.DVs[dvName].value

        return dvDict

    def update(self, ptSetName):
        """
        This is the main routine for returning coordinates that have
        been updated by design variables.

        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the
            given in an :func:`addPointSet()` call.
        """

        # Since we are sure that self.pts is always up to date, we can
        # just do the eval + offset.
        n      = len(self.pointSets[ptSetname].points)
        uv     = self.pointSets[ptSetName].uv
        faceID = self.pointSets[ptSetName].faceID
        offset = self.pointSets[ptSetName].offset
        newPts = numpy.zeros((n, 3))

        for idim in range(3):
            newPts[:, idim] = \
                              (1-uv[:, 0])*(1 - uv[:, 1]) * self.pts[self.conn[faceID, 0], idim] + \
                              (  uv[:, 0])*(1 - uv[:, 1]) * self.pts[self.conn[faceID, 1], idim] + \
                              (  uv[:, 0])*(    uv[:, 1]) * self.pts[self.conn[faceID, 2], idim] + \
                              (1-uv[:, 0])*(    uv[:, 1]) * self.pts[self.conn[faceID, 3], idim]
        newPts += offset

        # Finally flag this pointSet as being up to date:
        self.updated[ptSetName] = True

        return coords

    def pointSetUpToDate(self, ptSetName):
        """
        This is used externally to query if the object needs to update
        its pointset or not. Essentially what happens, is when
        update() is called with a point set, it the self.updated dict
        entry for pointSet is flagged as true. Here we just return
        that flag. When design variables are set, we then reset all
        the flags to False since, when DVs are set, nothing (in
        general) will up to date anymore.

        Parameters
        ----------
        ptSetName : str
            The name of the pointset to check.
        """
        if ptSetName in self.updated:
            return self.updated[ptSetName]
        else:
            return True

    def getNDV():
        """ Return the number of DVs"""
        return len(self.DVs)

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        """
        This function computes sensitivty information.

        Specificly, it computes the following:
        :math:`\\frac{dX_{pt}}{dX_{DV}}^T \\frac{dI}{d_{pt}}

        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)

            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once.

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place.

        Returns
        -------
        dIdxDict : dic
            The dictionary containing the derivatives, suitable for
            pyOptSparse

        Notes
        -----
        The ``child`` and ``nDVStore`` options are only used
        internally and should not be changed by the user.
        """

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = numpy.array([dIdpt])
        N = dIdpt.shape[0]

        # now that we have self.JT compute the Mat-Mat multiplication
        nDV = self._getNDV()
        dIdx_local = numpy.zeros((N, nDV), 'd')
        for i in range(N):

            # This is the final sensitivity of the
            tmp = self.pointSets[ptSetName].dPtdXsurf.T.dot(dIdpt[i,:,:].flatten())
            dIdx_local[i,:] = self.jac.T.dot(tmp)

        if comm: # If we have a comm, globaly reduce with sum
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # Now convert to dict:
        dIdxDict = {}
        i = 0
        for dvName in self.DVs:
            dIdxDict[dvName] = dIdx[:, i]
            i += 1

        return dIdxDict

    def addVariablesPyOpt(self, optProb):
        """
        Add the current set of variables to the optProb object.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which variables are added
        """

        for dvName in self.DVList:
            dv = DVList[dvName]
            optProb.addVar(dvName, 'c', value=dv.value, lower=dv.lower,
                           upper=dv.upper, scale=dv.scale)

    def printDesignVariables(self):
        """
        Print a formatted list of design variables to the screen
        """

        print ('-----------------------------------------------------------')
        print ('Component      Group          Parm           Value         ')
        print ('-----------------------------------------------------------')
        for DV in self.DVs:
            print('%15s %15s %15s %15s %15g'%(DV.component, DV.group, DV.parm, DV.value))

# ----------------------------------------------------------------------
#        THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER
# ----------------------------------------------------------------------

    def _computeSurfJacobian(self):
        """This routine comptues the jacobian of the VSP surface with respect
        to the design variables. Since this can be somewhat costly, we
        can compute it in an embarassingly parallel fashion across the
        COMM this object was created on.
        """

        jacLocal = numpy.zeros((len(self.pts0)*3, len(self.DVs)))
        ptsRef = self.pts.flatten()
        h = 1e-6
        dvKeys = list(self.DVs.keys())
        for iDV in range(len(dvKeys)):
            # I have to do this one.
            if i % self.comm.size == self.comm.rank:

                # Perturb the DV
                self.DVs[dvKeys[iDV]].value += h

                # Create the DVFile
                desFile = os.path.join(self.myid, 'desFile')
                slef._createDesginFile(desFile)

                # Get the updated points with this des file
                pts, conn = self.getUpdatedSurface(desFile)

                jacLocal[:, iDV] = (pts.flatten() - ptsRef)/h

                # Reset the DV
                self.DVs[dvKeys[iDV]].value += h

        # To get the full jacobian we can now all reduce across the
        # procs. This is pretty inefficient but easy to code.
        self.jac = self.comm.allreduce(jacLocal, op=MPI.SUM)

    def _parseDesignFile(self, fileName):

        """Parse through the given design variable file and add the DVs it
        finds to the DV List"""
        f = open(fileName, 'r')
        lines = f.readline()
        ndvs = int(lines[0])
        for i in range(ndvs):
            aux = lines[i+1].split(':')
            dvHash = aux[0]
            dvComponent = aux[1]
            dvGroup = aux[2]
            dvParm = aux[3]
            dvVal = float(aux[4])
            dvName = '%s:%s:%s'%(DV.component, DV.group, DV.parm)
            self.DVs[dvName] = vspDV(dvHash, dvComponent, dvGroup, dvParm, dvVal)

    def _createDesignFile(self, fileName):
        """Take the current set of design variables and create a .des file"""
        f = open(fileName, 'w')
        f.write(len(self.DVs))
        for dvName in self.DVs:
            DV = self.DVs[dvName]
            f.write('%s:%s:%s:%s:%20.15g\n'%(DV.hash, DV.component, DV.group, DV.parm, DV.value))
        f.close()

class vspDV(object):

    def __init__(self, hash, component, group, parm, value, lower, upper, scale)

        """Create a geometric design variable (or design variable group)
        See addGeoDVGlobal in DVGeometry class for more information
        """
        self.hash = hash
        self.component = component
        self.group = group
        self.parm = parm
        self.value = value
        self.lower = None
        self.upper = None
        if lower is not None:
            self.lower = _convertTo1D(lower, self.nVal)
        if upper is not None:
            self.upper = _convertTo1D(upper, self.nVal)
        if scale is not None:
            self.scale = _convertTo1D(scale, self.nVal)

class PointSet(object):
    def __init__(self, points, faceID, uv, offset, conn, nSurf):
        self.points = points
        self.faceID = faceID
        self.uv = uv
        self.offset = offset

        # We need to compute the (constant) dPtdXsurf. This is the
        # sensivitiy of the the points of this proc set with respect
        # to the positions on the VSP surface (Xsurf). This is
        # computed directly from the (linear) shape functions, so we
        # have all the information we need. Since this is (very)
        # sparse, we'll use a scipy sparse matrix.

        jac = sparse.lil_matrix((len(self.points)*3, nSurf*3))

        # Loop over each point:
        for i in range(len(self.points)):

            # This point depends on the the 4 nodes of its quad.
            for idim in range(3):
                jac[i*3+idim, 3*conn[self.faceID[i], 0]+idim] = (1 - uv[i, 0])*(1 - uv[i, 1])
                jac[i*3+idim, 3*conn[self.faceID[i], 1]+idim] = (    uv[i, 0])*(1 - uv[i, 1])
                jac[i*3+idim, 3*conn[self.faceID[i], 2]+idim] = (    uv[i, 0])*(    uv[i, 1])
                jac[i*3+idim, 3*conn[self.faceID[i], 3]+idim] = (1 - uv[i, 0])*(    uv[i, 1])

        # Convert to csc becuase we'll be doing a transpose product on it.
        self.dXpdXsurf = jac.tocsc()
