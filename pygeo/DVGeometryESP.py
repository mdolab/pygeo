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
import numpy
from collections import OrderedDict
from scipy import sparse
from scipy.spatial import cKDTree
from mpi4py import MPI
from pyOCSM import pyOCSM
import pickle 

import os
import sys
from contextlib import contextmanager

@contextmanager
def stdout_redirected(flag, to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    if flag:
        fd = sys.stdout.fileno()
        with os.fdopen(os.dup(fd), 'w') as old_stdout:
            with open(to, 'w') as file:
                _redirect_stdout(to=file)
            try:
                yield # allow code to be run with the redirected stdout
            finally:
                _redirect_stdout(to=old_stdout) # restore stdout.
                                                # buffering and flags such as
                                                # CLOEXEC may be different
    else:
        yield

class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a explicitly raised exception.
    """
    def __init__(self, message):
        msg = '\n+'+'-'*78+'+'+'\n' + '| DVGeometryESP Error: '
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

class DVGeometryESP(object):
    """
    A class for manipulating Engineering Sketchpad (ESP) geometry
    The purpose of the DVGeometryESP class is to provide translation
    of the ESP geometry engine to externally supplied surfaces. This
    allows the use of ESP design parameters to control the MACH
    framework.
    There are several import limitations:
    1. Since ESP is surface based only, it cannot be used to
    parameterize a geometry that doesn't lie on the surface. 
    Structural members need to be directly modeled in ESP as surfaces.
    2. It cannot handle *moving* intersection. A geometry with static
    intersections is fine as long as the intersection doesn't move
    3. It does not support complex numbers for the complex-step
    method.
    4. It does not surpport separate configurations.
    Parameters
    ----------
    espFile : str
       filename of .csm file containing the parameterized CAD 
    comm : MPI Intra Comm
       Comm on which to build operate the object. This is used to
       perform embarasisngly parallel finite differencing. Defaults to
       MPI.COMM_WORLD.
    scale : float
       A global scale factor from the ESP geometry to incoming (CFD) mesh
       geometry. For example, if the ESP model is in inches, and the CFD
       in meters, scale=0.0254.
    bodies : list of strings
       A list of the names of the ESP bodies to consider. 
       They need to be on the top of the ESP body stack (i.e., visible
       in the ESP user interface when all the branches are built)
    Examples
    --------
    The general sequence of operations for using DVGeometry is as follows:
      >>> from pygeo import *
      >>> DVGeo = DVGeometryESP("wing.csm", MPI_COMM_WORLD)
      >>> # Add a set of coordinates Xpt into the object
      >>> DVGeo.addPointSet(Xpt, 'myPoints')
      >>>
    """
    def __init__(self, espFile, comm=MPI.COMM_WORLD, scale=1.0, bodies=[],
                 intersectedBodies=None, projTol=0.01, debug=False, maxproc=None, suppress_stdout=False):

        if comm.rank == 0:
            print("Initializing DVGeometryESP")
            t0 = time.time()
        self.maxproc = maxproc
        self.esp = True
        self.points = OrderedDict()
        self.pointSets = OrderedDict()
        self.updated = {}
        self.updatedJac = {}
        self.globalDVList = [] # will become a list of tuples with (DVName, localIndex) - used for finite difference load balancing
        self.suppress_stdout = suppress_stdout
        # this scales coordinates from esp to mesh geometry
        self.espScale = scale
        # and this scales coordinates from mesh to esp geometry
        self.meshScale = 1./scale
        self.projTol = projTol*self.meshScale # default input is in meters.
        self.comm = comm
        self.espFile = espFile
        self.debug = debug

        t1 = time.time()
        # read the model
        self.espModel = pyOCSM.Ocsm(self.espFile)
        pyOCSM.SetOutLevel(0)
        # build the baseline model
        if self.comm.rank == 0:
            pyOCSM.SetOutLevel(0)
        else:
            pyOCSM.SetOutLevel(0)
        self.num_branches_baseline, _, allBodyIndices = self.espModel.Build(0,200) #pick 200 as arbitrary large number of bodies to allocate
        if self.num_branches_baseline < 0:
            raise ValueError('It appears the initial build of the ESP model did not succeed')

        t2 = time.time()
        if self.comm.rank == 0:
            print('Loading the esp model took:', (t2-t1))

        # List of all bodies returned from ESP
        if not bodies:
            # no components specified, we use all
            self.bodyIndices= allBodyIndices
        else:
            raise NotImplementedError('Specifying bodies by name still needs to be ported from the new pyOCSM wrapper.')
            # we get the comps from the comps list
            self.bodyIndices = []
            for bodyIndex in allBodyIndices:
                try:
                    # TODO BB this is missing from new wrapper and needs to be reimplemented completely
                    this_body_name = self.espModel.getTopoAttr(bodyIndex, 'body')['_name']
                except KeyError:
                    # no _name attribute, make a default
                    this_body_name = 'body'+str(bodyIndex)
                if this_body_name in bodies:
                    self.bodyIndices.append(bodyIndex)
            if len(self.bodyIndices) == 0:
                raise Error('No bodies matching the provided body names were found. \
                    Check that _name attributes are set correctly in the ESP model')

        # Initial list of DVs
        self.DVs = OrderedDict()
        self.csmDesPmtrs = OrderedDict()

        # Get metadata about external design parameters in the CSM model
        pmtrIndex = 0
        pmtrsleft = True
        ocsmExternal = 500
        ocsmIllegalPmtrIndex = -262
        while pmtrsleft:
            try:
                pmtrIndex += 1
                pmtrType, numRow, numCol, pmtrName= self.espModel.GetPmtr(pmtrIndex)
                baseValue = numpy.zeros(numRow*numCol)
                for rowIdx in range(numRow):
                    for colIdx in range(numCol):
                        try:
                            baseValue[colIdx+numCol*rowIdx] = self.espModel.GetValu(pmtrIndex, rowIdx+1, colIdx+1)[0]
                        except pyOCSM.OcsmError as e:
                            if e.value == 'ILLEGAL_PTMR_INDEX':
                                # I don't think we should ever make it here if the GetPmtr check is correct
                                raise Error('Column or row index out of range in design parameter '+pmtrName)
                
                if pmtrType == ocsmExternal:
                    self.csmDesPmtrs[pmtrName] = ESPParameter(pmtrName, pmtrIndex, numRow, numCol, baseValue)
            except pyOCSM.OcsmError as e:
                if e.value == 'ILLEGAL_PMTR_INDEX':
                    pmtrsleft = False
                else:
                    raise e
        if pmtrIndex == 1:
            if comm.rank == 0:
                print('DVGeometryESP Warning: no design parameters defined in the CSM file')

        if comm.rank == 0:
            t3 = time.time()
            print("Initialized DVGeometryESP in",(t3-t0),"seconds.")


    def addPointSet(self, points, ptName, distributed=True, cache_projections=False, **kwargs):
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
        distributed : bool
            Whether the pointset is distributed (different each proc)
            or non-distributed (duplicated and identical each proc)
            Should be set to false for duplicated pointsets to avoid
            very poor parallel scaling in the derivatives routine.
        cache_projections : None or str
            The user can optionally cache the point set projections
            to save initialization time. If a filename is provided,
            the cached u v t coordinates will be saved in numpy compressed
            format ('.npz' extension should be used).
            The points will be validated to ensure that the projections
            remain within tolerance of the model and if not, the projections
            will be recreated.
        """

        # save this name so that we can zero out the jacobians properly
        self.points[ptName] = True # ADFlow checks self.points to see
                                   # if something is added or not.
        points = numpy.array(points).real.astype('d')

        # check that duplicated pointsets are actually the same length
        sizes = numpy.array(self.comm.allgather(points.shape[0]), dtype = 'intc')
        if not distributed:
            all_same_length = numpy.all(sizes == sizes[0])
            if not all_same_length:
                raise ValueError('Nondistributed pointsets must be identical on each proc, but these pointsets vary in length per proc. Lengths: ', str(sizes))
        
        # check if a cache file exists 
        cache_loaded = False
        # cache_projections=False will disable caching, but None will generate a cachefile name automatically
        if cache_projections is None:
            cache_projections = ptName+'.npz'
        
        if cache_projections:
            if os.path.isfile(cache_projections):
                cache_loaded = True
                if self.comm.rank == 0:
                    cached_pt_arrays = numpy.load(cache_projections)
                    cached_sizes = cached_pt_arrays['sizes']
                    cached_nprocs = cached_pt_arrays['nprocs']
                    cached_distrib = cached_pt_arrays['distributed']
                    bodyIDg = cached_pt_arrays['bodyID']
                    faceIDg = cached_pt_arrays['faceID']
                    edgeIDg = cached_pt_arrays['edgeID']
                    ug = cached_pt_arrays['u']
                    vg = cached_pt_arrays['v']
                    tg = cached_pt_arrays['t']
                    uvlimg = cached_pt_arrays['uvlimits']
                    tlimg = cached_pt_arrays['tlimits']
                    cached_dmax = cached_pt_arrays['dmax']
                else:
                    cached_dmax = 0.0
                    cached_sizes = 0.0
                    cached_nprocs = 0.0
                    cached_distrib = False
                cached_dmax = self.comm.bcast(cached_dmax, 0)
                cached_sizes = self.comm.bcast(cached_sizes, 0)
                cached_nprocs = self.comm.bcast(cached_nprocs, 0)
                cached_distrib = self.comm.bcast(cached_distrib, 0)

                # find easy cache invalidations:
                if distributed:
                    if not cached_distrib:
                        raise ValueError('Cached pointset file ', cache_projections, ' is invalid because the cache was saved for a nondistributed pointset')
                    if cached_nprocs != self.comm.size:
                        raise ValueError('Cached pointset file ', cache_projections, ' is invalid because the cache was saved with different num procs')
                    if not numpy.all(cached_sizes == sizes):
                        raise ValueError('Cached pointset file ', cache_projections, ' is invalid because point counts have changed on some proc')
                else:
                    if cached_distrib:
                        raise ValueError('Cached pointset file ', cache_projections, ' is invalid because the cache was saved for a distributed pointset')

                # now figure out which proc has how many points.
                    
                nptsl = points.shape[0]
                # set up recieve buffers for all procs
                faceIDl = numpy.zeros(nptsl, dtype='intc')
                bodyIDl = numpy.zeros(nptsl, dtype='intc')
                edgeIDl = numpy.zeros(nptsl, dtype='intc')
                ul = numpy.zeros(nptsl)
                vl = numpy.zeros(nptsl)
                tl = numpy.zeros(nptsl)
                uvliml = numpy.zeros((nptsl, 4))           
                tliml = numpy.zeros((nptsl, 2))                                                      
                                        
                recvbuf1 = [bodyIDl,  MPI.INT]
                recvbuf2 = [faceIDl,  MPI.INT]
                recvbuf3 = [edgeIDl,  MPI.INT]
                recvbuf4 = [ul,  MPI.DOUBLE]
                recvbuf5 = [vl,  MPI.DOUBLE]
                recvbuf6 = [tl,  MPI.DOUBLE]
                recvbuf7 = [uvliml,  MPI.DOUBLE]
                recvbuf8 = [tliml,  MPI.DOUBLE]
                if distributed:
                    # displacements for scatter
                    disp = numpy.array([numpy.sum(sizes[:i]) for i in range(self.comm.size)], dtype='intc')
                    nptsg = numpy.sum(sizes)
                    if self.comm.rank == 0:
                        sendbuf1 = [bodyIDg, sizes, disp, MPI.INT]
                        sendbuf2 = [faceIDg, sizes, disp, MPI.INT]
                        sendbuf3 = [edgeIDg, sizes, disp, MPI.INT]
                        sendbuf4 = [ug, sizes, disp, MPI.DOUBLE]
                        sendbuf5 = [vg, sizes, disp, MPI.DOUBLE]
                        sendbuf6 = [tg, sizes, disp, MPI.DOUBLE]
                        sendbuf7 = [uvlimg, sizes*4, disp*4, MPI.DOUBLE]
                        sendbuf8 = [tlimg, sizes*2, disp*2, MPI.DOUBLE]
                    else:
                        sendbuf1 = sendbuf2 = sendbuf3 = sendbuf4 = sendbuf5 = sendbuf6 = sendbuf7 = sendbuf8 = None

                    # scatter the loaded info to all procs
                    self.comm.Scatterv(sendbuf1, recvbuf1, root=0)
                    self.comm.Scatterv(sendbuf2, recvbuf2, root=0)
                    self.comm.Scatterv(sendbuf3, recvbuf3, root=0)
                    self.comm.Scatterv(sendbuf4, recvbuf4, root=0)
                    self.comm.Scatterv(sendbuf5, recvbuf5, root=0)
                    self.comm.Scatterv(sendbuf6, recvbuf6, root=0)
                    self.comm.Scatterv(sendbuf7, recvbuf7, root=0)
                    self.comm.Scatterv(sendbuf8, recvbuf8, root=0)
                else:
                    # nondistributed pointset
                    # read on proc 0 and bcast to all
                    # then check that the points are correct
                    nptsg = points.shape[0]
                    if self.comm.rank == 0:
                        faceIDl[:] = faceIDg[:]
                        bodyIDl[:] = bodyIDg[:]
                        edgeIDl[:] = edgeIDg[:]
                        ul[:] = ug[:]
                        vl[:] = vg[:]
                        tl[:] = tg[:]
                        uvliml[:,:] = uvlimg[:,:]
                        tliml[:,:] = tlimg[:,:]
                    self.comm.Bcast(recvbuf1, root=0)
                    self.comm.Bcast(recvbuf2, root=0)
                    self.comm.Bcast(recvbuf3, root=0)
                    self.comm.Bcast(recvbuf4, root=0)
                    self.comm.Bcast(recvbuf5, root=0)
                    self.comm.Bcast(recvbuf6, root=0)
                    self.comm.Bcast(recvbuf7, root=0)
                    self.comm.Bcast(recvbuf8, root=0)

                nPts = len(points)
                if nPts != ul.shape[0]:
                    raise ValueError('Cached point projections does not match length of point set')

                # go through and check that each point projects 
                # within tolerance or else invalidate the cached values
                proj_pts = self._evaluatePoints(ul, vl, tl, uvliml, tliml, bodyIDl, faceIDl, edgeIDl, nPts)
                if points.shape[0] == 0:
                    # empty pointset can occur for some distributed pointsets
                    dMax_local = 0.0
                else:
                    dMax_local = numpy.max(numpy.sqrt(numpy.sum((points-proj_pts)**2, axis=1)))
                dMax_global = self.comm.allreduce(dMax_local, op=MPI.MAX)
                

                if (dMax_global - cached_dmax) / cached_dmax >= 1e-3:
                    raise ValueError('The cached point projections appear to no longer be valid for this geometry')                
                uvl = numpy.column_stack([ul,vl])
                self.pointSets[ptName] = PointSet(points, proj_pts, bodyIDl, faceIDl, edgeIDl, uvl, tl, uvliml, tliml, distributed)
                # Set the updated flag to false because the jacobian is not up to date.
                self.updated[ptName] = False
                self.updatedJac[ptName] = False

                return dMax_global

        # we need to project each of these points onto the ESP geometry,
        # get geometry and surface IDs, u, v values, and coordinates of the projections.
        # then calculate the self.offset variable using the projected points.

        # coordinates to store the physical coords (xyz) of the projected points
        proj_pts_esp = numpy.zeros_like(points)

        npoints = len(points)
        bodyIDArray = numpy.zeros(npoints, dtype='intc')
        faceIDArray = numpy.zeros(npoints, dtype='intc')
        edgeIDArray = numpy.zeros(npoints, dtype='intc')
        uv = numpy.zeros((npoints, 2))
        t = numpy.zeros(npoints)
        uvlimArray = numpy.zeros((npoints, 4))
        tlimArray = numpy.zeros((npoints, 2))
        dists = numpy.ones((npoints), dtype='float_')*999999.0

        t1 = time.time()
        # TODO parallelize projections for nondistributed pointsets?        
        
        # we want to reject points that are far outside the trimmed surface
        # and slightly prefer surface projections to edge projections
        rejectuvtol = 1e-4
        edgetol = 1e-8
        for ptidx in range(npoints):
            truexyz = points[ptidx]
            bi_best = -1
            fi_best = -1
            ei_best = -1
            dist_best = 99999999999
            uv_best = [-1, -1]
            uvlimits_best = None
            tlimits_best = None
            t_best = -1
            for bodyIndex in self.bodyIndices:
                nEdges = self.espModel.GetBody(bodyIndex)[6]
                nFaces = self.espModel.GetBody(bodyIndex)[7]
                for edgeIndex in range(1,nEdges+1):
                    # try to match point on edges first
                    with stdout_redirected(self.suppress_stdout):
                        # get the parametric coordinate along the edge
                        ttemp = self.espModel.GetUV(bodyIndex, pyOCSM.EDGE, edgeIndex, 1, truexyz.tolist())
                        # get the xyz location of the newly projected point
                        xyztemp = numpy.array(self.espModel.GetXYZ(bodyIndex, pyOCSM.EDGE, edgeIndex, 1, ttemp))
                    dist_temp = numpy.sum((truexyz - xyztemp)**2)
                    ttemp = ttemp[0]
                    tlimits = self._getUVLimits(bodyIndex, pyOCSM.EDGE, edgeIndex)
                    if not(ttemp < tlimits[0]-rejectuvtol or ttemp > tlimits[1]+rejectuvtol):
                        if dist_temp < dist_best:
                            tlimits_best = tlimits
                            t_best = ttemp
                            bi_best = bodyIndex
                            ei_best = edgeIndex
                            dist_best = dist_temp
                            xyzbest = xyztemp.copy()

                for faceIndex in range(1,nFaces+1):
                    with stdout_redirected(self.suppress_stdout):
                        # get the projected points on the ESP surface in UV coordinates
                        uvtemp = self.espModel.GetUV(bodyIndex, pyOCSM.FACE, faceIndex, 1, truexyz.tolist())
                        # get the XYZ location of the newly projected points
                        xyztemp = numpy.array(self.espModel.GetXYZ(bodyIndex, pyOCSM.FACE, faceIndex, 1, uvtemp))
                    dist_temp = numpy.sum((truexyz - xyztemp)**2)
                    # validate u and v
                    utemp = uvtemp[0]
                    vtemp = uvtemp[1]
                    uvlimits = self._getUVLimits(bodyIndex, pyOCSM.FACE, faceIndex)
                    if not(utemp < uvlimits[0]-rejectuvtol or utemp > uvlimits[1] + rejectuvtol or vtemp < uvlimits[2]-rejectuvtol or vtemp > uvlimits[3] + rejectuvtol):
                        if dist_temp - edgetol < dist_best:
                            uvlimits_best = uvlimits
                            uv_best = [utemp, vtemp]
                            bi_best = bodyIndex
                            fi_best = faceIndex
                            # if a face match is better wipe out the best edge match
                            ei_best = -1
                            t_best = -1
                            tlimits_best = None
                            dist_best = dist_temp
                            xyzbest = xyztemp.copy()
            if dist_best == 99999999999:
                # no matches
                raise ValueError('Point had no projection within tolerance')
            faceIDArray[ptidx] = fi_best
            edgeIDArray[ptidx] = ei_best
            bodyIDArray[ptidx] = bi_best
            uvlimArray[ptidx, :] = numpy.array(uvlimits_best)
            tlimArray[ptidx, :] = numpy.array(tlimits_best)
            uv[ptidx,0] = uv_best[0]
            uv[ptidx,1] = uv_best[1]
            t[ptidx] = t_best
            dists[ptidx] = dist_best
            proj_pts_esp[ptidx,:] = xyzbest
        
        proj_pts = proj_pts_esp * self.espScale
        if points.shape[0] != 0:
            dMax = numpy.max(numpy.sqrt(numpy.sum((points-proj_pts)**2, axis=1)))
        else:
            dMax = 0.0
 
        dMax_global = self.comm.allreduce(dMax, op=MPI.MAX)
        t2 = time.time()

        if self.comm.rank == 0 or self.comm is None:
            print('Adding pointset',ptName, 'took', t2-t1, 'seconds.')
            print('Maximum distance between the added points and the ESP geometry is',dMax_global)

        # Create the little class with the data
        self.pointSets[ptName] = PointSet(points, proj_pts, bodyIDArray, faceIDArray, edgeIDArray, uv, t, uvlimArray, tlimArray, distributed)

        # Set the updated flag to false because the jacobian is not up to date.
        self.updated[ptName] = False
        self.updatedJac[ptName] = False

        if cache_projections and not cache_loaded:
            # get the global projections and save in compressed npz format
            ul = uv[:,0].copy()
            vl = uv[:,1].copy()
            if distributed:
                ug, vg, tg, faceIDg, bodyIDg, edgeIDg, uvlimitsg, tlimitsg, sizes = self._allgatherCoordinates(ul,vl,t,faceIDArray,bodyIDArray,edgeIDArray,uvlimArray,tlimArray)
            else:
                ug = ul
                vg = vl 
                tg = t
                faceIDg = faceIDArray
                bodyIDg = bodyIDArray
                edgeIDg = edgeIDArray
                uvlimitsg = uvlimArray
                tlimitsg = tlimArray
                sizes = np.array([len(ug)])
            if self.comm.rank == 0:
                numpy.savez_compressed(cache_projections, distributed=distributed, sizes=sizes, nprocs=self.comm.size, dmax=dMax_global, u=ug, v=vg, t=tg, faceID=faceIDg, bodyID=bodyIDg, edgeID=edgeIDg, uvlimits=uvlimitsg, tlimits=tlimitsg)

        return dMax_global

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
                self.DVs[key].value = dvDict[key].copy()

        # we need to update the design variables in the ESP model and rebuild
        built_successfully = self._updateESPModel()
        if not built_successfully:
            # failed geometry, return fail flag
            return built_successfully

        # update the projected coordinates
        self._updateProjectedPts()

        # We will also compute the jacobian so it is also up to date,
        # provided we are asked for it
        if updateJacobian:
            self._computeSurfJacobian()

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

        # set the jacobian flag to false
        if not updateJacobian:
            for ptName in self.pointSets:
                self.updatedJac[ptName] = False
        return built_successfully


    def writeCADFile(self, filename):
        valid_filetypes = ['brep',
                           'bstl',
                           'egads',
                           'egg',
                           'iges',
                           'igs',
                           'sens',
                           'step',
                           'stl',
                           'stp',
                           'tess',
                           'grid']
        splitfile = filename.split('.')
        file_extension = filename.split('.')[-1]
        if file_extension.lower() not in valid_filetypes:
            raise IOError('CAD filename ' + filename + ' must have a valid exension. ' +
                          'Consult the EngineeringSketchPad docs for the DUMP function')
        if self.comm.rank == 0:
            modelCopy = self.espModel.Copy()
            n_branches, _, _ = modelCopy.Info()
            modelCopy.NewBrch(n_branches,
                              modelCopy.GetCode("dump"), 
                              "<none>", 
                              0,
                              filename, "0", "0", "", "", "", "", "", "")
            modelCopy.Build(0, 0)
    
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
        """
        This is the main routine for returning coordinates that have been
        updated by design variables. Multiple configs are not
        supported.
        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the
            given in an :func:`addPointSet()` call.
        """
        
        # this returns the current projection point coordinates
        newPts = self.pointSets[ptSetName].proj_pts
        
        if not self.updated[ptSetName]:
            # get the offset between points and original projected points
            offset = self.pointSets[ptSetName].offset

            # Get the coordinates of new surface cfd points, use the same array, BUT this should
            # actually be called newPoints since it has cfd nodes now
            newPts -= offset

            # Now compute the delta between the nominal new points and the
            # original points:
            delta = newPts - self.pointSets[ptSetName].points

            # Now get the final newPts with a possibly modified delta.
            newPts = self.pointSets[ptSetName].points + delta

            # Finally flag this pointSet as being up to date:
            self.updated[ptSetName] = True

        return newPts

    def writeCSMFile(self, fileName):
        valid_filetypes = ['csm']
        if fileName.split('.')[-1] not in valid_filetypes:
            raise IOError('Must use ".csm" file extension')
        if self.comm.rank == 0:
            self.espModel.Save(fileName)
            

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
        """ 
        Return the number of DVs"""
        return len(self.globalDVList)

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
        :math:`\\frac{dI}{d_{pt}}\\frac{dX_{pt}}{dX_{DV}} 
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
        if self.pointSets[ptSetName].jac is None:
            # in this case, we updated our pts when we added our pointset,
            # therefore the reference pts are up to date.
            self._computeSurfJacobian()

        # if the jacobian for this pointset is not up to date
        # update all the points
        if not self.updatedJac[ptSetName]:
            self._computeSurfJacobian()

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = numpy.array([dIdpt])
        N = dIdpt.shape[0]

        nDV = self.getNDV()

        # The following code computes the final sensitivity product:
        #
        #        T       T
        #   pXpt     pI
        #  ------  ------
        #   pXdv    pXpt
        #
        # Where I is the objective, Xpt are the externally coordinates
        # supplied in addPointSet

        # Extract just the single dIdpt we are working with. Make
        # a copy because we may need to modify it.

        # reshape the dIdpt array from [N] * [nPt] * [3] to  [N] * [nPt*3]
        dIdpt = dIdpt.reshape((dIdpt.shape[0], dIdpt.shape[1]*3))

        # # transpose dIdpt and vstack;
        # # Now vstack the result with seamBar as that is far as the
        # # forward FD jacobian went.
        # tmp = numpy.vstack([dIdpt.T, dIdSeam.T])
        tmp = dIdpt.T

        # we also stack the pointset jacobian
        jac = self.pointSets[ptSetName].jac.copy()

        dIdxT_local = jac.T.dot(tmp)
        dIdx_local = dIdxT_local.T

        if comm:
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # Now convert to dict:
        dIdxDict = {}
        for dvName in self.DVs:
            dv = self.DVs[dvName]
            jac_start = dv.globalStartInd
            jac_end = jac_start + dv.nVal
            # dIdxDict[dvName] = numpy.array([dIdx[:, i]]).T
            dIdxDict[dvName] = dIdx[:, jac_start:jac_end]

        return dIdxDict

    def totalSensitivityProd(self, vec, ptSetName, comm=None, config=None):
        """
        This function computes sensitivty information.
        Specificly, it computes the following:
        :math:`\\frac{dX_{pt}}{dX_{DV}} \\vec'`
        Parameters
        ----------
        vec : dictionary whose keys are the design variable names, and whose
              values are the derivative seeds of the corresponding design variable.
        ptSetName : str
            The name of set of points we are dealing with
        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place.
        Returns
        -------
        xsdot : array (Nx3) -> Array with derivative seeds of the surface nodes.
        """

        # We may not have set the variables so the surf jac might not be computed.
        if self.pointSets[ptSetName].jac is None:
            # in this case, we updated our pts when we added our pointset,
            # therefore the reference pts are up to date.
            self._computeSurfJacobian()

        # if the jacobian for this pointset is not up to date
        # update all the points
        if not self.updatedJac[ptSetName]:
            self._computeSurfJacobian()

        # vector that has all the derivative seeds of the design vars
        newvec = numpy.zeros(self.getNDV())

        for dvName in self.DVs:
            dv = self.DVs[dvName]
            if dvName in vec.keys():
                jac_start = dv.globalStartInd
                jac_end = jac_start + dv.nVal
                newvec[jac_start:jac_end] = vec[dvName]

        # we need to multiply with the surface jacobian
        dPtlocal = self.pointSets[ptSetName].jac.dot(newvec)

        if comm:
            dPt = comm.allreduce(dPtlocal, op=MPI.SUM)
        else:
            dPt = dPtlocal            

        return dPt

    def addVariable(self, desmptr_name, name=None, value=None,
                    lower=None, upper=None, scale=1.0, rows=None, cols=None, dh=0.001):
        """
        Add an ESP design parameter to the DVGeo problem definition.
        The name of the parameter must match a despmtr in the .csm file
        of the CAD model.

        Array-valued desmptrs can be handled in one of four ways:
        - rows=None, cols=None (default): treats the entire array as a flat vector
        - rows=[1, 2, 3], cols=None: pick specific rows (all cols included)
        - rows=[1, 2, 3], cols=[2, 4]: pick specific rows and columns
        - rows=[1], cols=[2]: pick a specific value
        THE INDICES ARE 1-indexing (per the OpenCSM standard)!! Not 0-indexing.

        The design variable vector passed to pyOptSparse will be in row-major order:
        in other words, the vector will look like:
        [a(1, 1), a(1, 2), a(1, 3), a(2, 1), a(2, 2), a(2, 3), a(3, 1), ....]

        The value, upper, and lower bounds must all be of length len(rows)*len(cols)

        Parameters
        ----------
        desmptr_name : str
            Name of the ESP design parameter
        name : str or None
            Human-readable name for this design variable (default same as despmtr)
        value : float or None
            The design variable. If this value is not supplied (None), then
            the current value in the ESP model will be queried and used
        lower : float or None
            Lower bound for the design variable. Use None for no lower bound
        upper : float or None
            Upper bound for the design variable. Use None for no upper bound
        scale : float
            Scale factor sent to pyOptSparse and used in optimization
        rows : list or None
            Design variable row index(indices) to use. 
            Default None uses all rows
        cols : list or None
            Design variable col index(indices) to use.
            Default None uses all cols
        dh : float
            Finite difference step size (default 0.001)
        """
        # if name is none, use the desptmr name instead
        if name is not None:
            dvName = name
        else:
            dvName = desmptr_name

        if dvName in self.DVs.keys():
            raise Error('Design variable name '+dvName+' already in use.')

        # find the design parm index in ESP
        if desmptr_name not in self.csmDesPmtrs.keys():
            raise Error('User specified design parameter name "' 
                        + desmptr_name + '" which was not found in the CSM file')

        csmDesPmtr = self.csmDesPmtrs[desmptr_name]
        numRow = csmDesPmtr.numRow
        numCol = csmDesPmtr.numCol
        self._validateRowCol(rows, cols, numRow, numCol, dvName)

        if rows is None:
            rows = range(1, numRow + 1)
        if cols is None:
            cols = range(1, numCol + 1)
        # if value is None, get the current value from ESP
        if value is None:
            value = self._csmToFlat(csmDesPmtr.baseValue, rows, cols, numRow, numCol)
        else:
            # validate that it is of correct length
            if len(value) != len(rows)*len(cols):
                raise Error('User-specified DV value does not match the dimensionality' +
                            'of the ESP despmtr. Value is of length ' + str(len(value)) + 
                            ' but should be '+str(len(rows)*len(cols)))

        # check that upper and lower are correct length

        if upper is not None:
            if isinstance(upper, (float, int)):
                upper = numpy.ones((len(rows)*len(cols),))*upper
            if len(upper) != len(rows)*len(cols):
                raise Error('User-specified DV upper bound does not match the dimensionality' +
                            'of the ESP despmtr. Upper is of length ' + str(len(upper)) + 
                            ' but should be '+str(len(rows)*len(cols)))

        if lower is not None:
            if isinstance(lower, (float, int)):
                lower = numpy.ones((len(rows)*len(cols),))*lower
            if len(lower) != len(rows)*len(cols):
                raise Error('User-specified DV lower bound does not match the dimensionality' +
                            'of the ESP despmtr. lower is of length ' + str(len(lower)) + 
                            ' but should be '+str(len(rows)*len(cols)))
        nVal = len(rows)*len(cols)
        
        # add an entry in the global DV list to make finite differencing load balancing easy
        globalStartInd = len(self.globalDVList)
        for localInd in range(nVal):
            self.globalDVList.append((dvName, localInd))

        self.DVs[dvName] = espDV(csmDesPmtr, dvName, 
                                 value, lower, upper, scale, rows, cols, dh, globalStartInd)

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
            optProb.addVarGroup(dv.name, dv.nVal, 'c', value=dv.value,
                                lower=dv.lower, upper=dv.upper,
                                scale=dv.scale)

# # ----------------------------------------------------------------------
# #        THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER
# # ----------------------------------------------------------------------

    def _getUVLimits(self, ibody, seltype, iselect):
        """
        Get the limits of the parametric coords
        on an edge or face

        Inputs
        ------
        ibody : int
            Body index
        seltype : int
            pyOCSM.EDGE or pyOCSM.FACE
        iselect : int
            Index of edge or face
        
        Outputs
        -------
        uvlimits : list
            ulower, uupper, vlower, vupper or tlower, tupper
        """
        this_ego = self.espModel.GetEgo(ibody, seltype, iselect)
        _, _, _, uvlimits, _, _ = this_ego.getTopology()
        return uvlimits

    def _csmToFlat(self, value, rows, cols, numRow, numCol):
        """
        Gets a slice of a flat array based on listed row and col indices
        """
        if numRow == 1 and numCol == 1:
            # early exit for scalars
            valOut = value
        elif len(value) == len(rows)*len(cols):
            # early exit for non-sliced arrays
            valOut = value
        else:            
            valOut = numpy.zeros(len(rows)*len(cols))
            irow = 0
            for rowInd in rows:
                icol = 0
                for colInd in cols:
                    valOut[icol + irow*len(cols)] = value[(colInd-1) + numCol*(rowInd-1)]
                    icol += 1
                irow += 1
        return valOut
    
    def _validateRowCol(self, rows, cols, numRow, numCol, dvName):
        # check that all rows, cols specified are within desmptr bounds
        # check for duplicate rows, cols
        if rows is not None:
            rowArr = numpy.array(rows)
            if numpy.max(rowArr) > numRow:
                raise Error('Design variable ' + dvName + ' slice out of bounds. ' + 
                            'Design var has '+str(numRow) + ' rows and index up to ' + 
                            str(numpy.max(rowArr)) + ' was specified: ' + 
                            str(rows))
            if numpy.min(rowArr) < 1:
                raise Error('Design variable ' + dvName + ' slice out of bounds. ' + 
                            'Row index less than 1 specified: ' + str(rows))
            if len(rows) != len(set(rows)):
                # duplicates
                raise Error('Duplicate indices specified in the rows of design variable '+
                             dvName + ': '+str(rows))

        if cols is not None:
            colArr = numpy.array(cols)
            if numpy.max(colArr) > numCol:
                raise Error('Design variable ' + dvName + ' slice out of bounds. ' + 
                            'Design var has '+str(numCol) + ' cols and index up to ' + 
                            str(numpy.max(colArr)) + ' was specified: ' + 
                            str(cols))
            if numpy.min(colArr) < 1:
                raise Error('Design variable ' + dvName + ' slice out of bounds. ' + 
                            'col index less than 1 specified: ' + str(cols))
            if len(cols) != len(set(cols)):
                # duplicates
                raise Error('Duplicate indices specified in the cols of design variable '+
                             dvName + ': '+str(cols))
    
    def _updateESPModel(self):
        """
        Sets design parameters in ESP to the correct value
        then rebuilds the model.
        """
        # for each design variable in the dictionary:
        
        # loop through rows and cols setting design paramter values
        for dvName in self.DVs:
            dv = self.DVs[dvName]
            espParamIdx = dv.csmDesPmtr.pmtrIndex
            for localIdx in range(dv.nVal):
                rowIdx = localIdx // len(dv.cols)
                colIdx = localIdx % len(dv.cols)
                espRowIdx = dv.rows[rowIdx]
                espColIdx = dv.cols[colIdx]
                self.espModel.SetValuD(espParamIdx, irow=espRowIdx, icol=espColIdx, value=dv.value[localIdx])
        # finally, rebuild
        outtuple = self.espModel.Build(0,0)
        # check that the number of branches built successfully
        # matches the number when the model was first built on __init__
        # otherwise, there was an EGADS/CSM build failure at this design point
        if outtuple[0] != self.num_branches_baseline:
            return False
        else:
            # built correctly
            return True
            

    def _evaluatePoints(self, u, v, t, uvlimits0, tlimits0, bodyID, faceID, edgeID, nPts):
        points = numpy.zeros((nPts, 3))
        for ptidx in range(nPts):
            # check if on an edge or surface
            bid = bodyID[ptidx]
            fid = faceID[ptidx]
            eid = edgeID[ptidx]

            if eid != -1:
                # get the point from an edge
                # get upper and lower parametric limits of updated model
                tlim0 = tlimits0[ptidx]
                tlim = self._getUVLimits(bid, pyOCSM.EDGE, eid)
                trange0 = tlim0[1] - tlim0[0]
                trange = tlim[1] - tlim[0]
                tnew = (t[ptidx] - tlim0[0]) * trange / trange0 + tlim[0]
                points[ptidx, :] = self.espModel.GetXYZ(bid,
                                                        pyOCSM.EDGE,
                                                        eid,
                                                        1,
                                                        [tnew])
            else:
                # point from a face
                if fid == -1:
                    raise ValueError('both edge ID and face ID are unset')
                # get the upper and lower uv limits of the updated model
                uvlim0 = uvlimits0[ptidx]
                uvlim = self._getUVLimits(bid, pyOCSM.FACE, fid)
                urange0 = uvlim0[1] - uvlim0[0]
                vrange0 = uvlim0[3] - uvlim0[2]
                urange = uvlim[1] - uvlim[0]
                vrange = uvlim[3] - uvlim[2]
                # scale the input uv points according to the original uv limits
                unew = (u[ptidx] - uvlim0[0]) * urange / urange0 + uvlim[0]
                vnew = (v[ptidx] - uvlim0[2]) * vrange / vrange0 + uvlim[2]            
                points[ptidx, :] = self.espModel.GetXYZ(bid, 
                                                            pyOCSM.FACE,
                                                            fid,
                                                            1,
                                                            [unew, vnew])
        points = points * self.espScale
        return points

    def _updateProjectedPts(self):
        # for each pointset
        # run getxyz and obtain new projected_pts
        # update the proj_pts
        for pointSetName in self.pointSets:
            pointSet = self.pointSets[pointSetName]
            proj_pts = self._evaluatePoints(pointSet.u, pointSet.v, pointSet.t, pointSet.uvlimits0, pointSet.tlimits0,
                                            pointSet.bodyID, pointSet.faceID, pointSet.edgeID, pointSet.nPts)
            pointSet.proj_pts = proj_pts

    def _allgatherCoordinates(self, ul, vl, tl, faceIDl, bodyIDl, edgeIDl, uvlimitsl, tlimitsl):
        # create the arrays to receive the global info
        # now figure out which proc has how many points.
        sizes = numpy.array(self.comm.allgather(len(ul)), dtype = 'intc')
        # displacements for allgather
        disp = numpy.array([numpy.sum(sizes[:i]) for i in range(self.comm.size)], dtype='intc')
        # global number of points
        nptsg = numpy.sum(sizes)
        ug = numpy.zeros(nptsg)
        vg = numpy.zeros(nptsg)
        tg = numpy.zeros(nptsg)
        faceIDg = numpy.zeros(nptsg, dtype='intc')
        bodyIDg = numpy.zeros(nptsg, dtype='intc')
        edgeIDg = numpy.zeros(nptsg, dtype='intc')
        ulower0g = numpy.zeros(nptsg)
        uupper0g = numpy.zeros(nptsg)
        vlower0g = numpy.zeros(nptsg)
        vupper0g = numpy.zeros(nptsg)
        tlower0g = numpy.zeros(nptsg)
        tupper0g = numpy.zeros(nptsg)
        ulowerl = uvlimitsl[:,0].copy()
        uupperl = uvlimitsl[:,1].copy()
        vlowerl = uvlimitsl[:,2].copy()
        vupperl = uvlimitsl[:,3].copy()
        tlowerl = tlimitsl[:,0].copy()
        tupperl = tlimitsl[:,1].copy()

        # Now we do an allGatherv to get a long list of all pointset information
        # TODO probably can avoid breaking up the uvt limits with copies and stack below
        self.comm.Allgatherv([ul, len(ul)], [ug, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([vl, len(vl)], [vg, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([tl, len(tl)], [tg, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([faceIDl, len(faceIDl)], [faceIDg, sizes, disp, MPI.INT])
        self.comm.Allgatherv([bodyIDl, len(bodyIDl)], [bodyIDg, sizes, disp, MPI.INT])
        self.comm.Allgatherv([edgeIDl, len(edgeIDl)], [edgeIDg, sizes, disp, MPI.INT])
        self.comm.Allgatherv([ulowerl, len(ulowerl)], [ulower0g, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([uupperl, len(uupperl)], [uupper0g, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([vlowerl, len(vlowerl)], [vlower0g, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([vupperl, len(vupperl)], [vupper0g, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([tlowerl, len(tlowerl)], [tlower0g, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([tupperl, len(tupperl)], [tupper0g, sizes, disp, MPI.DOUBLE])

        uvlimitsg = numpy.column_stack((ulower0g, uupper0g, vlower0g, vupper0g))
        tlimitsg = numpy.column_stack((tlower0g, tupper0g))

        return ug, vg, tg, faceIDg, bodyIDg, edgeIDg, uvlimitsg, tlimitsg, sizes

    def _computeSurfJacobian(self, fd=True):
        """This routine comptues the jacobian of the ESP surface with respect
        to the design variables. Since our point sets are rigidly linked to
        the ESP projection points, this is all we need to calculate. The input
        pointSets is a list or dictionary of pointSets to calculate the jacobian for.
        """

        # timing stuff:
        t1 = time.time()
        tesp = 0
        teval = 0
        tcomm = 0

        # counts
        nDV = self.getNDV()
        if self.maxproc is None:
            nproc = self.comm.size
        else:
            if self.maxproc <= self.comm.size:
                nproc = self.maxproc
            else:
                nproc = self.comm.size
        rank  = self.comm.rank
        
        # arrays to collect local pointset info
        ul = numpy.zeros(0) # local u coordinates
        vl = numpy.zeros(0) # local v coordinates
        tl = numpy.zeros(0) # local t coordinates
        faceIDl = numpy.zeros(0, dtype = 'intc') # surface index
        bodyIDl = numpy.zeros(0, dtype = 'intc') # body index
        edgeIDl = numpy.zeros(0, dtype='intc') # edge index
        uvlimitsl = numpy.zeros((0,4))
        tlimitsl = numpy.zeros((0,2))
        any_ptset_nondistributed = False
        any_ptset_distributed = False
        for ptSetName in self.pointSets:
            # initialize the Jacobians
            self.pointSets[ptSetName].jac = numpy.zeros((3*self.pointSets[ptSetName].nPts, nDV))
            if self.pointSets[ptSetName].distributed:
                any_ptset_distributed = True
            else:
                any_ptset_nondistributed = True

            # first, we need to vstack all the point set info we have
            # counts of these are also important, saved in ptSet.nPts
            ul = numpy.concatenate((ul, self.pointSets[ptSetName].u))
            vl = numpy.concatenate((vl, self.pointSets[ptSetName].v))
            tl = numpy.concatenate((tl, self.pointSets[ptSetName].t))
            faceIDl = numpy.concatenate((faceIDl, self.pointSets[ptSetName].faceID))
            bodyIDl = numpy.concatenate((bodyIDl, self.pointSets[ptSetName].bodyID))
            edgeIDl = numpy.concatenate((edgeIDl, self.pointSets[ptSetName].edgeID))
            uvlimitsl = numpy.concatenate((uvlimitsl, self.pointSets[ptSetName].uvlimits0))
            tlimitsl = numpy.concatenate((tlimitsl, self.pointSets[ptSetName].tlimits0))
        if any_ptset_distributed and any_ptset_nondistributed:
            raise ValueError('Both nondistributed and distributed pointsets were added to this DVGeoESP which is not yet supported')

        if any_ptset_distributed:
            # need to get ALL the coordinates from every proc on every proc to do the parallel FD
            if self.maxproc is not None:
                raise ValueError('Max processor limit is not usable with distributed pointsets')
            # now figure out which proc has how many points.
            sizes = numpy.array(self.comm.allgather(len(ul)), dtype = 'intc')
            # displacements for allgather
            disp = numpy.array([numpy.sum(sizes[:i]) for i in range(nproc)], dtype='intc')
            # global number of points
            nptsg = numpy.sum(sizes)
            ug, vg, tg, faceIDg, bodyIDg, edgeIDg, uvlimitsg, tlimitsg, sizes = self._allgatherCoordinates(ul,vl,tl,faceIDl,bodyIDl,edgeIDl,uvlimitsl,tlimitsl)
        else:
            nptsg = len(ul)
            ug = ul
            vg = vl 
            tg = tl 
            faceIDg = faceIDl
            bodyIDg = bodyIDl
            edgeIDg = edgeIDl
            uvlimitsg = uvlimitsl 
            tlimitsg = tlimitsl
        # create a local new point array. We will use this to get the new
        # coordinates as we perturb DVs. We just need one (instead of nDV times the size)
        # because we get the new points, calculate the jacobian and save it right after
        ptsNewL = numpy.zeros(len(ul)*3)
        
        
        # we now have all the point info on all procs.
        tcomm += (time.time() - t1)

        # We need to evaluate all the points on respective procs for FD computations

        # determine how many DVs this proc will perturb.
        n = 0
        for iDV in range(self.getNDV()):
            # I have to do this one.
            if iDV % nproc == rank:
                n += 1
        if fd:
            # evaluate all the points
            pts0 = self._evaluatePoints(ug, vg, tg, uvlimitsg, tlimitsg, bodyIDg, faceIDg, edgeIDg, nptsg)
            # allocate the approriate sized numpy array for the perturbed points
            ptsNew = numpy.zeros((n, nptsg, 3))

            # perturb the DVs on different procs and compute the new point coordinates.
            i = 0 # Counter on local Jac

            for iDV in range(self.getNDV()):
                # I have to do this one.
                if iDV % nproc == rank:
                    # Get the DV object for this variable
                    dvName = self.globalDVList[iDV][0]
                    dvLocalIndex = self.globalDVList[iDV][1]
                    dvObj = self.DVs[dvName]
                    # Step size for this particular DV
                    dh =  dvObj.dh

                    # Perturb the DV
                    dvSave = dvObj.value.copy()
                    dvObj.value[dvLocalIndex] += dh

                    # update the esp model
                    t11 = time.time()
                    self._updateESPModel()
                    t12 = time.time()
                    tesp += (t12-t11)

                    t11 = time.time()
                    # evaluate the points

                    ptsNew[i,:,:] = self._evaluatePoints(ug, vg, tg, uvlimitsg, tlimitsg, bodyIDg, faceIDg, edgeIDg, nptsg)
                    t12 = time.time()
                    teval += (t12-t11)
                    # now we can calculate the jac and put it back in ptsNew
                    ptsNew[i, :, :] = (ptsNew[i, :, :] - pts0[:,:]) / dh

                    # Reset the DV
                    dvObj.value = dvSave.copy()

                    # increment the counter
                    i += 1

            # Now, we have perturbed points on each proc that perturbed a DV

            # reset the model.
            t11 = time.time()
            self._updateESPModel()
            t12 = time.time()
            tesp += (t12-t11)
        
        else:
            raise NotImplementedError('ESP analytic derivatives are not implemented')
           
        ii = 0
        # loop over the DVs and scatter the perturbed points to original procs
        for iDV in range(self.getNDV()):
            # Get the DV object for this variable
            dvName = self.globalDVList[iDV][0]
            dvLocalIndex = self.globalDVList[iDV][1]
            dvObj = self.DVs[dvName]
            # Step size for this particular DV
            dh =  dvObj.dh

            t11 = time.time()
            root_proc = iDV % nproc
            if any_ptset_distributed:
                # create the send/recv buffers for the scatter
                if root_proc == rank:
                    sendbuf = [ptsNew[ii, :, :].flatten(), sizes*3, disp*3, MPI.DOUBLE]
                else:
                    sendbuf = [numpy.zeros((0,3)), sizes*3, disp*3, MPI.DOUBLE]
                recvbuf = [ptsNewL,  MPI.DOUBLE]
                # scatter the info from the proc that perturbed this DV to all procs
                self.comm.Scatterv(sendbuf, recvbuf, root=root_proc)
            else:
                # create the send/recv buffers for the bcast
                if root_proc == rank:
                    bcastbuf = [ptsNew[ii, :, :].flatten(), MPI.DOUBLE]
                    ptsNewL[:] = ptsNew[ii, :, :].flatten()
                else:
                    bcastbuf = [ptsNewL, MPI.DOUBLE]
                # bcast the info from the proc that perturbed this DV to all procs
                self.comm.Bcast(bcastbuf, root=root_proc)
                self.comm.Barrier()

            t12 = time.time()
            tcomm += (t12-t11)

            # calculate the jacobian here for the pointsets
            offset = 0
            for ptSet in self.pointSets:
                # number of points in this pointset
                nPts = self.pointSets[ptSet].nPts

                # indices to extract correct points from the long pointset array
                ibeg = offset*3
                iend = ibeg + nPts*3

                # ptsNewL has the jacobian itself...
                self.pointSets[ptSet].jac[:, iDV] = ptsNewL[ibeg:iend].copy()

                # increment the offset
                offset += nPts

            # pertrub the local counter on this proc.
            # This loops over the DVs that this proc perturbed
            if iDV % nproc == rank:
                ii += 1

        t2 = time.time()
        if rank == 0:
            print('FD jacobian calcs with dvgeovsp took',(t2-t1),'seconds in total')
            print('updating the esp model took',tesp,'seconds')
            print('evaluating the new points took', teval,'seconds')
            print('communication took',tcomm,'seconds')

        # set the update flags
        for ptSet in self.pointSets:
            self.updatedJac[ptSet] = True

class ESPParameter(object):
    def __init__(self, pmtrName, pmtrIndex, numRow, numCol, baseValue):
        """Internal class for storing metadata about the ESP model"""
        self.pmtrName = pmtrName
        self.pmtrIndex = pmtrIndex
        self.numRow = numRow
        self.numCol = numCol
        self.baseValue = baseValue

class espDV(object):

    def __init__(self, csmDesPmtr, name, value, lower, upper, scale, rows, cols, dh, globalstartind):
        """Internal class for storing ESP design variable information"""
        self.csmDesPmtr = csmDesPmtr
        self.name = name
        self.value = numpy.array(value)
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.rows = rows
        self.cols = cols
        self.nVal = len(rows)*len(cols)
        self.dh = dh
        self.globalStartInd = globalstartind

class PointSet(object):
    def __init__(self, points, proj_pts, bodyID, faceID, edgeID, uv, t, uvlimits, tlimits, distributed):
        self.points = points
        self.proj_pts = proj_pts
        self.bodyID = bodyID
        self.faceID = faceID
        self.edgeID = edgeID
        self.u = uv[:,0]
        self.v = uv[:,1]
        self.t = t
        self.uvlimits0 = uvlimits
        self.tlimits0 = tlimits
        self.offset = self.proj_pts - self.points
        self.nPts = len(self.proj_pts)
        self.jac = None
        self.distributed = distributed
