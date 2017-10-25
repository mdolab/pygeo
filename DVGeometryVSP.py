# ======================================================================
#         Imports
# ======================================================================
from __future__ import print_function
import copy
import tempfile
import shutil
import os
import sys
import time
from collections import OrderedDict
import numpy
from scipy import sparse
from mpi4py import MPI
from pyspline import pySpline
import vsp

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
    intersections is fine as long as the intersection doesn't move

    3. It does not support complex numbers for the complex-step
    method.

    4. It does not surpport separate configurations.

    Parameters
    ----------
    vspFile : str
       filename of .vsp3 file.

    comm : MPI Intra Comm
       Comm on which to build operate the object. This is used to
       perform embarasisngly parallel finite differencing. Defaults to
       MPI.COMM_WORLD.

    scale : float
       A global scale factor from the VSP geometry to incoming (CFD) mesh
       geometry. For example, if the VSP model is in inches, and the CFD
       in meters, scale=0.0254.

    comps : list of strings
       A list of string defining the subset of the VSP components to use when
       exporting the P3D surface files

    Examples
    --------
    The general sequence of operations for using DVGeometry is as follows:
      >>> from pygeo import *
      >>> DVGeo = DVGeometryVSP("wing.vsp3", MPI_COMM_WORLD)
      >>> # Add a set of coordinates Xpt into the object
      >>> DVGeo.addPointSet(Xpt, 'myPoints')
      >>>

    """
    def __init__(self, vspFile, comm=MPI.COMM_WORLD, scale=1.0, comps=[], debug=False):
        self.points = OrderedDict()
        self.pointSets = OrderedDict()
        self.updated = {}
        self.vspScale = scale
        self.comm = comm
        self.vspFile = vspFile
        self.debug = debug
        # Load in the VSP model
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(vspFile)

        # Setup the export group set (0) with just the sets we want.
        self.exportSet = 9

        # List of all componets returned from VSP
        allComps = vsp.FindGeoms()

        # If we were not given comps, use all of them
        if comps == []:
            comps = allComps

        # First set the export flag for exportSet to False for everyone
        for comp in allComps:
            vsp.SetSetFlag(comp, self.exportSet, False)

        for comp in comps:
            compID = vsp.FindContainer(comp, 0)
            vsp.SetSetFlag(compID, self.exportSet, True)

        # Create a directory in which we will put the temporary files
        # we need. We *should* use something like tmepfile.mkdtemp()
        # but that behaves bady on pleiades.
        tmpDir = None
        if self.comm.rank == 0:
            tmpDir = './tmpDir_%d_%s'%(MPI.COMM_WORLD.rank, time.time())
            print ('Temp dir is: %s'%tmpDir)
            if not os.path.exists(tmpDir):
                os.makedirs(tmpDir)
        self.tmpDir = self.comm.bcast(tmpDir)

        # Initial list of DVs
        self.DVs = OrderedDict()

        # Run the update. This will also set the conn on the first pass
        self.conn = None
        self.pts0 = None
        if self.comm.rank == 0:
            self.pts0, self.conn = self._getUpdatedSurface()

        self.pts0 = self.comm.bcast(self.pts0)
        self.conn = self.comm.bcast(self.conn)
        self.pts = self.pts0.copy()

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
        self.points[ptName] = True # ADFlow checks self.points to see
                                   # if something is added or not.

        points = numpy.array(points).real.astype('d')
        # Attach the points to the original surface using the fast ADT
        # projection code from pySpline. Note that we convert to
        # 1-based indexing for conn here.

        if len(points) > 0:
            faceID, uv = pySpline.libspline.adtprojections(
                self.pts0.T, (self.conn+1).T, points.T)
            uv = uv.T
            faceID -= 1 # Convert back to zero-based indexing.
        else:
            faceID = numpy.zeros((0), 'intc')
            uv = numpy.zeros((0, 2), 'intc')

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

        # Just dump in the values
        for key in dvDict:
            if key in self.DVs:
                self.DVs[key].value = dvDict[key]

        # When we set the design variables we will also compute the
        # new surface self.pts. Only one proc needs to do this.
        if self.comm.rank == 0:
            self.pts, conn = self._getUpdatedSurface()
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

    def update(self, ptSetName, config=None):
        """This is the main routine for returning coordinates that have been
        updated by design variables. Multiple configs are not
        supported.

        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the
            given in an :func:`addPointSet()` call.
        """

        # Since we are sure that self.pts is always up to date, we can
        # just do the eval + offset.
        n      = len(self.pointSets[ptSetName].points)
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
        newPts -= offset

        # Finally flag this pointSet as being up to date:
        self.updated[ptSetName] = True

        return newPts

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

    def getNDV(self):
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
        """

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = numpy.array([dIdpt])
        N = dIdpt.shape[0]

        nDV = self.getNDV()
        dIdx_local = numpy.zeros((N, nDV), 'd')
        # The following code computes the final sensitivity product:
        #
        #         T        T      T
        #  pXsurf     pXpt     pI
        #  ------   -------   ----
        #  pXdv      pXsurf    pXpt
        #
        # Where I is the objective, Xpt are the externally coordinates
        # supplied in addPointSet, Xsurf are the coordinates of the
        # VSP plot3d File and Xdv are the design variables.

        for i in range(N):
            dIdx_local[i,:] = self.jac.T.dot(
                self.pointSets[ptSetName].dPtdXsurf.T.dot(dIdpt[i,:,:].flatten()))

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

    def addVariable(self, component, group, parm, value=None,
                    lower=None, upper=None, scale=1.0, scaledStep=True,
                    dh=1e-6):
        """
        Add a design variable definition.

        Parameters
        ----------
        component : str
            Name of the VSP component
        group : str
            Name of the VSP group
        parm : str
            Name of the VSP parameter
        value : float or None
            The design variable. If this value is not supplied (None), then
            the current value in the VSP model will be queried and used
        lower : float or None
            Lower bound for the design variable. Use None for no lower bound
        upper : float or None
            Upper bound for the design variable. Use None for no upper bound
        scale : float
            Scale factor
        scaledStep : bool
            Flag to use a scaled step sized based on the initial value of the
            variable. It will remain constant thereafter.
        dh : float
            Step size. When scaledStep is True, the actual step is dh*value. Otherwise
            this actual step is used.
        """

        container_id = vsp.FindContainer(component, 0)
        if container_id == "":
            raise Error('Bad component for DV: %s'%component)

        parm_id = vsp.FindParm(container_id, parm, group)
        if parm_id == "":
            raise Error('Bad group or parm: %s %s'%(group, parm))

        # Now we know the parmID is ok. So we just get the value
        val = vsp.GetParmVal(parm_id)

        dvName = '%s:%s:%s'%(component, group, parm)

        if value is None:
            value = val

        if scaledStep:
            dh = dh * value

        self.DVs[dvName] = vspDV(parm_id, component, group, parm, value, lower,
                                 upper, scale, dh)

    def addVariablesPyOpt(self, optProb):
        """
        Add the current set of variables to the optProb object.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which variables are added
        """

        for dvName in self.DVs:
            dv = self.DVs[dvName]
            optProb.addVar(dvName, 'c', value=dv.value, lower=dv.lower,
                           upper=dv.upper, scale=dv.scale)

    def printDesignVariables(self):
        """
        Print a formatted list of design variables to the screen
        """
        print ('-'*85)
        print ('%30s%20s%20s%15s'%('Component','Group','Parm','Value'))
        print ('-'*85)
        for dvName in self.DVs:
            DV = self.DVs[dvName]
            print('%30s%20s%20s%15g'%(DV.component, DV.group, DV.parm, DV.value))

    def createDesignFile(self, fileName):
        """Take the current set of design variables and create a .des file"""
        f = open(fileName, 'w')
        f.write('%d\n'%len(self.DVs))
        for dvName in self.DVs:
            DV = self.DVs[dvName]
            f.write('%s:%s:%s:%s:%20.15g\n'%(DV.parmID, DV.component, DV.group, DV.parm, DV.value))
        f.close()

    def writePlot3D(self, fileName):
        """Write the current design to Plot3D file"""

        for dvName in self.DVs:
            DV = self.DVs[dvName]
            # We use float here since sometimes pyoptsparse will give
            # stupid numpy zero-dimensional arrays, which swig does
            # not like.
            vsp.SetParmVal(DV.parmID, float(DV.value))
        vsp.Update()

        # Write the export file.
        vsp.ExportFile(fileName, self.exportSet, vsp.EXPORT_PLOT3D)

# ----------------------------------------------------------------------
#        THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER
# ----------------------------------------------------------------------

    def _getUpdatedSurface(self, fName='export.x'):
        """Return the updated surface for the currently set design variables,
        converted to an unstructured format. Note that this routine is
        safe to call in an embarrassing parallel fashion. That is it
        can be called with different DVs set on different
        processors and they won't interfere with each other.
        """
        import time

        # Set each of the DVs. We have the parmID stored so its easy.
        for dvName in self.DVs:
            DV = self.DVs[dvName]
            # We use float here since sometimes pyoptsparse will give
            # stupid numpy zero-dimensional arrays, which swig does
            # not like.
            vsp.SetParmVal(DV.parmID, float(DV.value))
        vsp.Update()

        # Write the export file.
        fName = os.path.join(self.tmpDir, fName)
        vsp.ExportFile(fName, self.exportSet, vsp.EXPORT_PLOT3D)

        # Now we load in this updated plot3d file and compute the
        # connectivity if not already done.
        computeConn = False
        if self.conn is None:
            computeConn = True
        pts, conn = self._readPlot3DSurfFile(fName, computeConn)

        # Apply the VSP scale
        newPts = pts * self.vspScale

        return newPts, conn

    def _readPlot3DSurfFile(self, fileName, computeConn=False):
        """Read a scii plot3d file and return the points and connectivity"""

        f = open(fileName, 'r')
        nSurf = numpy.fromfile(f, 'int', count=1, sep=' ')[0]
        sizes = numpy.fromfile(f, 'int', count=3*nSurf, sep=' ').reshape((nSurf, 3))
        nPts = 0; nElem = 0
        for i in range(nSurf):
            curSize = sizes[i, 0]*sizes[i, 1]
            nPts += curSize
            nElem += (sizes[i, 0]-1)*(sizes[i, 1]-1)

        # Generate the uncompacted point and connectivity list:
        pts = numpy.zeros((nPts, 3))
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

    def __del__(self):
        # if self.comm.rank == 0 and not self.debug:
        #     if os.path.exists(self.tmpDir):
        #         shuitl.rmtree(self.tmpDir)
        pass

    def _computeSurfJacobian(self):
        """This routine comptues the jacobian of the VSP surface with respect
        to the design variables. Since this can be somewhat costly, we
        can compute it in an embarassingly parallel fashion across the
        COMM this object was created on.
        """

        ptsRef = self.pts.flatten()
        self.jac = numpy.zeros((len(self.pts0)*3, len(self.DVs)))

        dvKeys = list(self.DVs.keys())

        # Determine how many points we'll perturb
        n = 0
        for iDV in range(len(dvKeys)):
            # I have to do this one.
            if iDV % self.comm.size == self.comm.rank:
                n += 1

        localJac = numpy.zeros((len(self.pts0)*3, n))
        i = 0 # Counter on local Jac
        reqs = []
        for iDV in range(len(dvKeys)):
            # I have to do this one.
            if iDV % self.comm.size == self.comm.rank:

                # Step size for this particular DV
                dh =  self.DVs[dvKeys[iDV]].dh

                # Perturb the DV
                dvSave = self.DVs[dvKeys[iDV]].value
                self.DVs[dvKeys[iDV]].value += dh

                # Get the updated points. Note that this is the only
                # time we need to call _getUpdatedSurface in parallel,
                # so we have to supply a unique file name.
                pts, conn = self._getUpdatedSurface('dv_perturb_%d.x'%iDV)

                localJac[:, i] = (pts.flatten() - ptsRef)/dh

                # Reset the DV
                self.DVs[dvKeys[iDV]].value = dvSave

                # If we are on the root proc, set direcly, otherwise, MPI_isend
                if self.comm.rank == 0:
                    self.jac[:, iDV] = localJac[:, i]
                else:
                    self.comm.send(localJac[:, i].copy(), 0, iDV)

                i += 1

        # Root proc finishes the receives
        if self.comm.rank == 0:
            for iDV in range(len(dvKeys)):
                # This is the rank that did this DV:
                rank = iDV % self.comm.size
                if rank != 0:
                    status = MPI.Status()
                    tmp =  self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    tag = status.Get_tag()
                    # Get the tag so we know where to put it
                    self.jac[:, tag] = tmp

        # Broadcast back out to everyone. mpi4py is complete screwed
        # on pleiades so do 1 column at at time..sigh
        for iDV in range(len(dvKeys)):
            self.jac[:, iDV] = self.comm.bcast(self.jac[:, iDV])

class vspDV(object):

    def __init__(self, parmID, component, group, parm, value, lower, upper, scale, dh):
        """Inernal class for storing VSP design variable information"""
        self.parmID = parmID
        self.component = component
        self.group = group
        self.parm = parm
        self.value = value
        self.lower = lower
        self.upper = upper
        self.dh = dh
        self.scale = scale

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
        self.dPtdXsurf = jac.tocsc()
