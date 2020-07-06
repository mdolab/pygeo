# ======================================================================
#         Imports
# ======================================================================
import copy
import tempfile
import shutil
import os
import sys
import time
import numpy
from collections import OrderedDict
from scipy import sparse
from scipy.spatial import cKDTree
from mpi4py import MPI
from pyspline import pySpline
import vsp
from cgtlib import lsect

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
    def __init__(self, vspFile, comm=MPI.COMM_WORLD, scale=1.0, comps=[],
                 intersectedComps=None, debug=False):
        self.points = OrderedDict()
        self.pointSets = OrderedDict()
        self.updated = {}
        self.vspScale = scale
        self.comm = comm
        self.vspFile = vspFile
        self.debug = debug
        self.jac = None
        # Load in the VSP model
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(vspFile)

        # Setup the export group set (0) with just the sets we want.
        self.exportSet = 9

        # List of all componets returned from VSP. Note that this
        # order is important...it is the order the comps will be
        # written out in plot3d format.
        allComps = vsp.FindGeoms()

        # If we were not given comps, use all of them
        if comps == []:
            for c in allComps:
                comps.append(vsp.GetContainerName(c))

        # First set the export flag for exportSet to False for everyone
        for comp in allComps:
            vsp.SetSetFlag(comp, self.exportSet, False)

        self.exportComps = []
        for comp in allComps:
            # Check if this one is in our list:
            compName = vsp.GetContainerName(comp)
            if compName in comps:
                vsp.SetSetFlag(comp, self.exportSet, True)
                self.exportComps.append(compName)

        # Create a directory in which we will put the temporary files
        # we need. We *should* use something like tmepfile.mkdtemp()
        # but that behaves badly on pleiades.
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
        self.cumSizes = None
        self.sizes = None
        if self.comm.rank == 0:
            self.pts0, self.conn, self.cumSizes, self.sizes = self._getUpdatedSurface()

        self.pts0 = self.comm.bcast(self.pts0)
        self.conn = self.comm.bcast(self.conn)
        self.cumSizes = self.comm.bcast(self.cumSizes)
        self.sizes = self.comm.bcast(self.sizes)
        self.pts = self.pts0.copy()

        # Finally process theintersection information. We had to wait
        # until all processors have the surface information.
        self.intersectComps = []
        if intersectedComps is None:
            intersectedComps = []
        for i in range(len(intersectedComps)):
            c = intersectedComps[i]
            # Get the index of each of the two comps:
            compIndexA = self.exportComps.index(c[0])
            compIndexB = self.exportComps.index(c[1])
            direction = c[2]
            dStar = c[3]
            extraComps = []
            if len(c) == 5:
                for j in range(len(c[4])):
                    extraComps.append(self.exportComps.index(c[4][j]))

            self.intersectComps.append(CompIntersection(
                compIndexA, compIndexB, extraComps, direction, dStar,
                self.pts, self.cumSizes, self.sizes, self.tmpDir))

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
            coordinates. Thisname will need to be provided when
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

        # From the faceID we can back out what component each one is
        # connected to. This way if we have intersecting components we
        # only change the ones that are apart of the two surfaces.
        cumFaceSizes = numpy.zeros(len(self.sizes) + 1, 'intc')
        for i in range(len(self.sizes)):
            nCellI = self.sizes[i][0]-1
            nCellJ = self.sizes[i][1]-1
            cumFaceSizes[i+1] = cumFaceSizes[i] + nCellI*nCellJ
        compIDs = numpy.searchsorted(cumFaceSizes, faceID, side='right')-1

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

        # Add the points to each of the intersection curve objects if we have any
        for IC in self.intersectComps:
            IC.addPointSet(points, compIDs, ptName)

        self.updated[ptName] = False

    def setDesignVars(self, dvDict, updateJacobian=True):
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
        # new surface self.pts. Only one proc needs to do this. This
        # should be the only location where self.pts is set.
        if self.comm.rank == 0:
            self.pts, conn, cumSizes, sizes = self._getUpdatedSurface()
        self.pts = self.comm.bcast(self.pts)

        # We need to give the updated coordinates to each of the
        # intersectComps (if we have any) so they can update the new
        # intersection curve
        for IC in self.intersectComps:
            IC.setSurface(self.pts)

        # We will also compute the jacobian so it is also up to date,
        # provided we are asked for ti
        if updateJacobian:
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

        # Now compute the delta between the nominal new poitns and the
        # original points:
        delta = newPts - self.pointSets[ptSetName].points

        # Potentially modify the delta according to our intersection curves
        for IC in self.intersectComps:
            delta = IC.update(ptSetName, delta)

        # Now get the final newPts with a possibly modified delta.
        newPts = self.pointSets[ptSetName].points + delta

        # Finally flag this pointSet as being up to date:
        self.updated[ptSetName] = True

        return newPts

    def writeVSPFile(self, fileName, exportSet=0):
        """Take the current design and write a new FSP file"""
        vsp.WriteVSPFile(fileName, exportSet)

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

    def getVarNames(self):
        """
        Return a list of the design variable names. This is typically
        used when specifying a wrt= argument for pyOptSparse.

        Examples
        --------
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        return list(self.DVs.keys())


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

        # We may not have set the variables so the surf jac might not be computed.
        if self.jac is None:
            self._computeSurfJacobian()

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
        nPts = len(self.pts)

        # Extract just the single dIdpt we are working with. Make
        # a copy because we may need to modify it.

        n = 0
        for IC in self.intersectComps:
            n += len(IC.seam)

        dIdSeam = numpy.zeros((N, n*3))
        for i in range(N):
            n = 0
            for IC in self.intersectComps:
                dIdpt_i = dIdpt[i, :, :].copy()
                seamBar = IC.sens(dIdpt_i, ptSetName)
                dIdpt[i, :, :] = dIdpt_i
                dIdSeam[i, 3*n:3*(n+len(seamBar))] = seamBar.flatten()
                n += len(IC.seam)
        dIdpt = dIdpt.reshape((dIdpt.shape[0], dIdpt.shape[1]*3))

        # Now take dIdpt_i back to the plot3D surface:
        tmp = self.pointSets[ptSetName].dPtdXsurf.T.dot(dIdpt.T)

        # Now vstack the result with seamBar as that is far as the
        # forward FD jacobian went.
        tmp = numpy.vstack([tmp, dIdSeam.T])

        # Do final local jacobian transpose product back to the
        # DVs. This is a little trickier since we have a distributed
        # total surface jacobian. First we all reduce the "tmp"
        # variable which is pI/pXsurf. The "Reduce" is important
        # here...that is performance critical. We need the fast numpy
        # version. 
        
        if comm:
            tmp2 = numpy.zeros_like(tmp)
            comm.Reduce(tmp, tmp2, op=MPI.SUM)
            comm.Bcast(tmp2)
        else:
            tmp2 = tmp
        # ---------------------------------------------------------------

        # Remember the jacobian contains the surface poitns *and* the
        # the seam nodes. dIdx_compact is the final derivative for the
        # variables this proc owns. 
        dIdx_compact = self.jac.T.dot(tmp2)

        # Now scatter the dIdx_compact into dIdx_local before we all
        # reduce over the actual DVs. 
        dvKeys = list(self.DVs.keys())
        i = 0 # Counter on local Jac
        for iDV in range(len(dvKeys)):
            # I have to do this one.
            if iDV % self.comm.size == self.comm.rank:
                dIdx_local[:, iDV] = dIdx_compact[i, :]
                i += 1

        dIdx = self.comm.allreduce(dIdx_local, op=MPI.SUM)

        # Now convert to dict:
        dIdxDict = {}
        i = 0
        for dvName in self.DVs:
            dIdxDict[dvName] = numpy.array([dIdx[:, i]]).T
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

            if value == 0:
                raise Error('Initial value is exactly 0. scaledStep option cannot be used'
                            'Specify an explicit dh with scaledStep=False')

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
        pts, conn, cumSizes, sizes = self._readPlot3DSurfFile(fName, computeConn)

        # Apply the VSP scale
        newPts = pts * self.vspScale

        return newPts, conn, cumSizes, sizes

    def _readPlot3DSurfFile(self, fileName, computeConn=False):
        """Read a scii plot3d file and return the points and connectivity"""

        f = open(fileName, 'r')
        nSurf = numpy.fromfile(f, 'int', count=1, sep=' ')[0]
        sizes = numpy.fromfile(f, 'int', count=3*nSurf, sep=' ').reshape((nSurf, 3))
        nPts = 0; nElem = 0
        cumSizes = numpy.zeros(nSurf + 1, 'intc')
        for i in range(nSurf):
            curSize = sizes[i, 0]*sizes[i, 1]
            nPts += curSize
            cumSizes[i+1] = cumSizes[i] + curSize
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

        return pts, conn, cumSizes, sizes

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

        # Determine the size of the 'surface' jacobian. It actually
        # contains the surface from VSP as well as any requried
        # intersection curves.

        ptsRef = self.pts.flatten()
        nPts = len(self.pts)

        seamsRef = numpy.zeros((0, 3))
        for IC in self.intersectComps:
            seamsRef = numpy.vstack([seamsRef, IC.seam])
        nSeam = len(seamsRef)
        seamsRef = seamsRef.flatten()

        dvKeys = list(self.DVs.keys())

        # Determine how many variables we'll perturb. 
        n = 0
        for iDV in range(len(dvKeys)):
            # I have to do this one.
            if iDV % self.comm.size == self.comm.rank:
                n += 1

        # The section of the jacobian that I own:
        self.jac = numpy.zeros((3*(nPts + nSeam), n))

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
                pts, conn, cumSizes, sizes = self._getUpdatedSurface('dv_perturb_%d.x'%iDV)
                self.jac[0:nPts*3, i] = (pts.flatten() - ptsRef)/dh

                # Do any required intersections:
                seams = numpy.zeros((0, 3))
                for IC in self.intersectComps:
                    IC.setSurface(pts)
                    seams = numpy.vstack([seams, IC.seam])

                self.jac[nPts*3:, i] = (seams.flatten() - seamsRef)/dh

                # Reset the DV
                self.DVs[dvKeys[iDV]].value = dvSave

                i += 1

        # Restore the seams
        for IC in self.intersectComps:
            IC.setSurface(self.pts)


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


class CompIntersection(object):
    def __init__(self, compA, compB, extraComps, direction, dStar, pts, cumSizes, sizes, tmpDir):
        '''Class to store information required for an intersection.  The order
        of compA and compB are important: LSect will give a different
        result for compA intersecting compB than compB intersecting
        compA.

        Input
        -----
        compA , int : Index of the surface in the plot3D file
        compB , int : Index of the surface in the plot3D file
        extraComps, list : Indexes of other comps to move as well 
        direction, str: Coordiante index direction to use for
            lsect. Must be 'j' or 'k'
        dStar, real : Radius over which to attenuate the deformation
        pts , real (N, 3) : All coordinates that are being used for this model

        Internally we will store the indices and the weights of the
        points that this intersection will have to modify. In general,
        all this code is not super efficient since it's all python,
        but it should not be more of a bottleneck than VSP is itself
        in doing the export.
        '''
        self.compA = compA
        self.compB = compB
        self.extraComps = extraComps
        self.dir = direction.lower()
        self.dStar = dStar
        self.halfdStar = dStar/2.0
        self.pts = pts
        self.cumSizes = cumSizes
        self.sizes = sizes
        if direction not in ['j', 'k']:
            raise Error("Direction must be given as 'j' or 'k' for intersection")
        if direction == 'j':
            self.dir = 1
        else:
            self.dir = 2

        # First generate the initial intersection curve.
        s = self.sizes[self.compA]
        blkA = self.pts[self.cumSizes[self.compA]:self.cumSizes[self.compA+1], :].reshape(
            (s[0], s[1], s[2], 3), order='f')

        s = self.sizes[self.compB]
        blkB = self.pts[self.cumSizes[self.compB]:self.cumSizes[self.compB+1], :].reshape(
            (s[0], s[1], s[2], 3), order='f')

        nmax = max(blkB.shape[0], blkB.shape[1], blkA.shape[0], blkA.shape[1])
        seam, nOut = lsect.lsect_wrap(blkA, blkB, self.dir, nmax)
        self.seam0 = seam[0:nOut, :]
        self.seam = self.seam0.copy()
        self.ptSets = {}

    def setSurface(self, pts):
        """ This set the new udpated surface on which we need to comptue the new intersection curve"""

        s = self.sizes[self.compA]
        blkA = pts[self.cumSizes[self.compA]:self.cumSizes[self.compA+1], :].reshape(
            (s[0], s[1], s[2], 3), order='f')

        s = self.sizes[self.compB]
        blkB = pts[self.cumSizes[self.compB]:self.cumSizes[self.compB+1], :].reshape(
            (s[0], s[1], s[2], 3), order='f')

        nmax = max(blkB.shape[0], blkB.shape[1], blkA.shape[0], blkA.shape[1])
        seam, nOut = lsect.lsect_wrap(blkA, blkB, self.dir, nmax)
        self.seam = seam[0:nOut, :]

    def addPointSet(self, pts, compIDs, ptSetName):

        # Figure out which points this intersection object has to deal with.
        tree = cKDTree(self.seam0)

        # Find all the distances. This is pretty fast.
        d, index = tree.query(pts)
        indices = []
        factors = []
        for i in range(len(pts)):
            if compIDs[i] == self.compA or compIDs[i] == self.compB or compIDs[i] in self.extraComps:
                if d[i] < self.dStar:

                    # Compute the factor
                    if d[i] < self.halfdStar:
                        factor = .5*(d[i]/self.halfdStar)**3
                    else:
                        factor = .5*(2-((self.dStar - d[i])/self.halfdStar)**3)

                    # Save the index and factor
                    indices.append(i)
                    factors.append(factor)

        # Save the affected indices and the factor in the little dictionary
        self.ptSets[ptSetName] = [pts.copy(), indices, factors]

    def update(self, ptSetName, delta):

        """Update the delta in ptSetName with our correction. The delta need
        to be supplied as we will be changing it and returning them
        """
        pts     = self.ptSets[ptSetName][0]
        indices = self.ptSets[ptSetName][1]
        factors = self.ptSets[ptSetName][2]
        seamDiff = self.seam - self.seam0
        for i in range(len(factors)):

            # j is the index of the point in the full set we are
            # working with.
            j = indices[i]

            #Run the weighted interp:
            # num = numpy.zeros(3)
            # den = 0.0
            # for k in range(len(seamDiff)):
            #     rr = pts[j] - self.seam0[k]
            #     LdefoDist = 1.0/numpy.sqrt(rr[0]**2 + rr[1]**2 + rr[2]**2+1e-16)
            #     LdefoDist3 = LdefoDist**3
            #     Wi = LdefoDist3
            #     Si = seamDiff[k]
            #     num = num + Wi*Si
            #     den = den + Wi

            # interp = num / den

            # Do it vectorized
            rr = pts[j] - self.seam0
            LdefoDist = (1.0/numpy.sqrt(rr[:,0]**2 + rr[:,1]**2 + rr[:,2]**2+1e-16))
            LdefoDist3 = LdefoDist**3
            Wi = LdefoDist3
            den = numpy.sum(Wi)
            interp = numpy.zeros(3)
            for iDim in range(3):
                interp[iDim] = numpy.sum(Wi*seamDiff[:, iDim])/den

            # Now the delta is replaced by 1-factor times the weighted
            # interp of the seam * factor of the original:

            delta[j] = factors[i]*delta[j] + (1-factors[i])*interp

        return delta

    def sens(self, dIdPt, ptSetName):
        # Return the reverse accumulation of dIdpt on the seam
        # nodes. Also modifies the dIdp array accordingly.

        pts     = self.ptSets[ptSetName][0]
        indices = self.ptSets[ptSetName][1]
        factors = self.ptSets[ptSetName][2]

        seamBar = numpy.zeros_like(self.seam0)
        for i in range(len(factors)):

            # j is the index of the point in the full set we are
            # working with.
            j = indices[i]

            # This is the local seed (well the 3 seeds for the point)
            localVal = dIdPt[j]*(1 - factors[i])

            # Scale the dIdpt by the factor..dIdpt is input/output
            dIdPt[j] *= factors[i]

            # Do it vectorized
            rr = pts[j] - self.seam0
            LdefoDist = (1.0/numpy.sqrt(rr[:,0]**2 + rr[:,1]**2 + rr[:,2]**2+1e-16))
            LdefoDist3 = LdefoDist**3
            Wi = LdefoDist3
            den = numpy.sum(Wi)
            interp = numpy.zeros(3)
            for iDim in range(3):
                seamBar[:, iDim] += Wi*localVal[iDim]/den

        return seamBar
