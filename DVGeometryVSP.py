# ======================================================================
#         Imports
# ======================================================================
# from __future__ import print_function
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

# analysis error
# TODO: Ultimately, we want to avoid importing and using this if we dont have OM installed
# from openmdao.core.analysis_error import AnalysisError

# directly import the interface to the fortran APIs
# from pysurf.geometryEngines.TSurf.python import intersectionAPI, curveSearchAPI, utilitiesAPI, tsurf_tools
# generic import for all pysurf codes
# import pysurf


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
                 intersectedComps=None, symNormal = 'k', projTol=0.01, debug=False):

        if comm.rank == 0:
            print("Initializing DVGeometryVSP")
            t0 = time.time()

        self.points = OrderedDict()
        self.pointSets = OrderedDict()
        self.ptSetNames = []
        self.updated = {}
        self.updatedJac = {}
        # normal vector of the symmetry plane. We will use this when removing the duplicate intersection curves on the other side of the symmetry plane.
        self.symNorm = symNormal

        # this scales coordinates from vsp to mesh geometry
        self.vspScale = scale
        # and this scales coordinates from mesh to vsp geometry
        self.meshScale = 1./scale
        self.projTol = projTol*self.meshScale # default input is in meters.
        self.comm = comm
        self.vspFile = vspFile
        self.debug = debug

        # Clear the vsp model
        vsp.ClearVSPModel()

        t1 = time.time()
        # read the model
        vsp.ReadVSPFile(vspFile)
        t2 = time.time()
        if self.comm.rank == 0:
            print('Loading the vsp model took:', (t2-t1))

        # List of all componets returned from VSP. Note that this
        # order is important. It is the order that we use to map the
        # actual geom_id by using the geom_names
        allComps = vsp.FindGeoms()
        allNames = []
        for c in allComps:
            allNames.append(vsp.GetContainerName(c))

        if not comps:
            # no components specified, we use all
            self.allComps = allComps[:]
        else:
            # we get the vsp comp IDs from the comps list
            self.allComps = []
            for c in comps:
                self.allComps.append(allComps[allNames.index(c)])

        # we need the names and bounding boxes of components
        self.compNames = []
        self.bbox = OrderedDict()
        self.bboxuv = self._getuv()
        for c in self.allComps:
            self.compNames.append(vsp.GetContainerName(c))
            self.bbox[c] = self._getBBox(c)

        # Initial list of DVs
        self.DVs = OrderedDict()

        # Now, we need to form our own quad meshes for fast projections
        if comm.rank == 0:
            print("Building a quad mesh for fast projections.")
        self._getQuads()

        # initialize intersections
        if comm.rank == 0:
            print("Computing intersections")
        self.intersectComps = []
        if intersectedComps is None:
            intersectedComps = []
        for i in range(len(intersectedComps)):
            # we will just initialize the intersections by passing in the dictionary
            self.intersectComps.append(CompIntersection(intersectedComps[i], self))

        if comm.rank == 0:
            t3 = time.time()
            print("Initialized DVGeometry VSP in",(t3-t0),"seconds.")


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

        self.ptSetNames.append(ptName)

        # save this name so that we can zero out the jacobians properly
        self.points[ptName] = True # ADFlow checks self.points to see
                                   # if something is added or not.

        points = numpy.array(points).real.astype('d')

        # we need to project each of these points onto the VSP geometry,
        # get geometry and surface IDs, u, v values, and coordinates of the projections.
        # then calculate the self.offset variable using the projected points.

        # first, to get a good initial guess on the geometry and u,v values,
        # we can use the adt projections in pyspline
        if len(points) > 0:
            # faceID has the index of the corresponding quad element.
            # uv has the parametric u and v weights of the projected point.

            faceID, uv = pySpline.libspline.adtprojections(
                self.pts0.T, (self.conn+1).T, points.T)
            uv = uv.T
            faceID -= 1 # Convert back to zero-based indexing.
            # after this point we should have the projected points.

        else:
            faceID = numpy.zeros((0), 'intc')
            uv = numpy.zeros((0, 2), 'intc')

        # now we need to figure out which surfaces the points got projected to
        # From the faceID we can back out what component each one is
        # connected to. This way if we have intersecting components we
        # only change the ones that are apart of the two surfaces.
        cumFaceSizes = numpy.zeros(len(self.sizes) + 1, 'intc')
        for i in range(len(self.sizes)):
            nCellI = self.sizes[i][0]-1
            nCellJ = self.sizes[i][1]-1
            cumFaceSizes[i+1] = cumFaceSizes[i] + nCellI*nCellJ
        compIDs = numpy.searchsorted(cumFaceSizes, faceID, side='right')-1

        # coordinates to store the projected points
        pts = numpy.zeros_like(points)

        # npoints * 3 list containing the geomID, u and v values
        # this can be improved if we can group points that get
        # projected to the same geometry.
        npoints = len(points)
        geom = numpy.zeros(npoints, dtype='intc')
        u = numpy.zeros(npoints)
        v = numpy.zeros(npoints)

        # initialize one 3dvec for projections
        pnt = vsp.vec3d()

        # Keep track of the largest distance between cfd and vsp surfaces
        dMax = 1e-16

        t1 = time.time()
        for i in range(points.shape[0]):
            # this is the geometry our point got projected to in the adt code
            gind = compIDs[i] # index
            gid = self.allComps[gind] # ID

            # set the coordinates of the point object
            pnt.set_xyz(points[i,0] * self.meshScale, points[i,1] * self.meshScale, points[i,2] * self.meshScale)

            # first, we call the fast projection code with the initial guess

            # this is the global index of the first node of the projected element
            nodeInd = self.conn[faceID[i],0]
            # get the local index of this node
            nn = nodeInd - self.cumSizes[gind]
            # figure out the i and j indices of the first point of the element
            # we projected this point to
            ii = numpy.mod(nn, self.sizes[gind,0])
            jj = numpy.floor_divide(nn, self.sizes[gind,0])

            # calculate the global u and v change in this element
            du = self.uv[gind][0][ii+1] - self.uv[gind][0][ii]
            dv = self.uv[gind][1][jj+1] - self.uv[gind][1][jj]

            # now get this points u,v coordinates on the vsp geometry and add
            # compute the initial guess using the  tessalation data of the surface
            ug = uv[i,0]*du + self.uv[gind][0][ii]
            vg = uv[i,1]*dv + self.uv[gind][1][jj]

            # project the point
            d, u[i], v[i] = vsp.ProjPnt01Guess( gid, 0, pnt, ug, vg )
            geom[i] = gind

            # if we dont have a good projection, try projecting again to surfaces
            #  with the slow code.
            if (d > self.projTol):
                # print('Guess code failed with projection distance',d)
                # for now, we need to check for all geometries separately.
                # Just pick the one that yields the smallest d
                gind = 0
                for gid in self.allComps:

                    # only project if the point is in the bounding box of the geometry
                    if ((self.bbox[gid][0,0] < points[i,0] < self.bbox[gid][0,1]) and
                        (self.bbox[gid][1,0] < points[i,1] < self.bbox[gid][1,1]) and
                        (self.bbox[gid][2,0] < points[i,2] < self.bbox[gid][2,1])):

                        # project the point onto the VSP geometry
                        dNew, surf_indx_out, uout, vout = vsp.ProjPnt01I(gid, pnt)

                        # check if we are closer
                        if dNew < d:
                            # save this info if we found a closer projection
                            u[i] = uout
                            v[i] = vout
                            geom[i] = gind
                            d = dNew
                    gind +=1

            # check if the final d is larger than our previous largest value
            dMax = max(d, dMax)

            # We need to evaluate this pnt to get its coordinates in physical space
            pnt = vsp.CompPnt01(self.allComps[geom[i]], 0, u[i], v[i])
            pts[i,0] = pnt.x() * self.vspScale
            pts[i,1] = pnt.y() * self.vspScale
            pts[i,2] = pnt.z() * self.vspScale

        # some debug info
        dMax_global = self.comm.allreduce(dMax, op=MPI.MAX)
        t2 = time.time()

        if self.comm.rank == 0 or self.comm is None:
            print('DVGeometryVSP note:\nAdding pointset',ptName, 'took', t2-t1, 'seconds.')
            print('Maximum distance between the added points and the VSP geometry is',dMax_global)

        # Create the little class with the data
        self.pointSets[ptName] = PointSet(points, pts, geom, u, v)

        # Add the points to each of the intersection curve objects if we have any
        # lets ignore this for now. We might need to return compIDs in the future, or modify the IC # code completely
        for IC in self.intersectComps:
            IC.addPointSet(points, self.pointSets[ptName].geom, ptName)

        # Set the updated flag to false because the jacobian is not up to date.
        self.updated[ptName] = False
        self.updatedJac[ptName] = False

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

        # we just need to set the design variables in the VSP model and we are done
        self._updateVSPModel()

        # update the projected coordinates
        self._updateProjectedPts()

        # We need to give the updated coordinates to each of the
        # intersectComps (if we have any) so they can update the new
        # intersection curve
        for IC in self.intersectComps:
            IC.setSurface(self.comm)

        # We will also compute the jacobian so it is also up to date,
        # provided we are asked for it
        # if updateJacobian:
            # self._computeSurfJacobian()

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

        # set the jacobian flag to false
        for ptName in self.pointSets:
            self.updatedJac[ptName] = False

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

        # this returns the current projection point coordinates
        newPts = self.pointSets[ptSetName].pts

        # get the offset between points and pts
        offset = self.pointSets[ptSetName].offset

        # Get the coordinates of new surface cfd points, use the same array, BUT this should
        # actually be called newPoints since it has cfd nodes now
        newPts -= offset

        # Now compute the delta between the nominal new poitns and the
        # original points:
        delta = newPts - self.pointSets[ptSetName].points

        # we need to add this later on
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

        # number of projection points on this proc for this pointset
        # this can also be the number of surface points since we have equally many
        nPts = len(self.pointSets[ptSetName].pts)

        # Extract just the single dIdpt we are working with. Make
        # a copy because we may need to modify it.

        # We should keep track of the intersections that this pointset is close to. There is no point in including the intersections far from this pointset in the sensitivity calc as the derivative seeds will be just zeros there.
        ptSetICs = []
        nSeams  = 0
        for IC in self.intersectComps:
            # This checks if we have any entries in the affected indices on this point set with this intersection
            if IC.ptSets[ptSetName][1]:
                # this pointset is affected by this intersection. save this info.
                ptSetICs.append(IC)
                # this keeps the cumulative number of nodes on the seams this point set is effected by
                nSeams += len(IC.seam)

        dIdSeam = numpy.zeros((N, nSeams*3))

        # go over the dI values
        for i in range(N):
            n = 0
            for IC in ptSetICs:
                dIdpt_i = dIdpt[i, :, :].copy()
                seamBar = IC.sens(dIdpt_i, ptSetName)
                dIdpt[i, :, :] = dIdpt_i
                dIdSeam[i, 3*n:3*(n+len(seamBar))] = seamBar.flatten()
                n += len(IC.seam)

        # reshape the dIdpt array from [N] * [nPt] * [3] to  [N] * [nPt*3]
        dIdpt = dIdpt.reshape((dIdpt.shape[0], dIdpt.shape[1]*3))

        # transpose dIdpt and vstack;
        # Now vstack the result with seamBar as that is far as the
        # forward FD jacobian went.
        tmp = numpy.vstack([dIdpt.T, dIdSeam.T])

        # we also stack the pointset jacobian and the seam jacobians.
        jac = self.pointSets[ptSetName].jac.copy()
        for IC in ptSetICs:
            jac = numpy.vstack((jac, IC.jac))

        # Remember the jacobian contains the surface poitns *and* the
        # the seam nodes. dIdx_compact is the final derivative for the
        # points this proc owns.
        dIdxT_local = jac.T.dot(tmp)
        dIdx_local = dIdxT_local.T

        if comm:
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # Now convert to dict:
        dIdxDict = {}
        i = 0
        for dvName in self.DVs:
            dIdxDict[dvName] = numpy.array([dIdx[:, i]]).T
            i += 1

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

        # populate newvec
        for i, dv in enumerate(self.DVs):
            if dv in vec:
                newvec[i] = vec[dv]

        # we need to multiply with the surface jacobian
        dPt = self.pointSets[ptSetName].jac.dot(newvec)

        # and the seams
        for IC in self.intersectComps:
            dSeam = IC.jac.dot(newvec)

            # now we need to update the dPt wrt the changes in this seam
            # dPt is both input/output.
            IC.update_d(ptSetName, dPt, dSeam)

        return dPt

    def totalSensitivityTransProd(self, dIdpt, ptSetName, comm=None, config=None):
        """
        This is probably incorrect
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

        # number of projection points on this proc for this pointset
        # this can also be the number of surface points since we have equally many
        nPts = len(self.pointSets[ptSetName].pts)

        # Extract just the single dIdpt we are working with. Make
        # a copy because we may need to modify it.

        # We should keep track of the intersections that this pointset is close to. There is no point in including the intersections far from this pointset in the sensitivity calc as the derivative seeds will be just zeros there.
        ptSetICs = []
        nSeams  = 0
        for IC in self.intersectComps:
            # This checks if we have any entries in the affected indices on this point set with this intersection
            if IC.ptSets[ptSetName][1]:
                # this pointset is affected by this intersection. save this info.
                ptSetICs.append(IC)
                # this keeps the cumulative number of nodes on the seams this point set is effected by
                nSeams += len(IC.seam)

        dIdSeam = numpy.zeros((N, nSeams*3))

        # go over the dI values
        for i in range(N):
            n = 0
            for IC in ptSetICs:
                dIdpt_i = dIdpt[i, :, :].copy()
                seamBar = IC.sens(dIdpt_i, ptSetName)
                dIdpt[i, :, :] = dIdpt_i
                dIdSeam[i, 3*n:3*(n+len(seamBar))] = seamBar.flatten()
                n += len(IC.seam)

        # reshape the dIdpt array from [N] * [nPt] * [3] to  [N] * [nPt*3]
        dIdpt = dIdpt.reshape((dIdpt.shape[0], dIdpt.shape[1]*3))

        # transpose dIdpt and vstack;
        # Now vstack the result with seamBar as that is far as the
        # forward FD jacobian went.
        tmp = numpy.vstack([dIdpt.T, dIdSeam.T])

        # we also stack the pointset jacobian and the seam jacobians.
        jac = self.pointSets[ptSetName].jac.copy()
        for IC in ptSetICs:
            jac = numpy.vstack((jac, IC.jac))

        # Remember the jacobian contains the surface poitns *and* the
        # the seam nodes. dIdx_compact is the final derivative for the
        # points this proc owns.
        dIdxT_local = jac.T.dot(tmp)
        dIdx_local = dIdxT_local.T

        if comm:
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # Now convert to dict:
        dIdxDict = {}
        i = 0
        for dvName in self.DVs:
            dIdxDict[dvName] = numpy.array([dIdx[0, i]]).T
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
            raise Error('Bad group or parm: %s %s %s'%(component, group, parm))

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

        # First set the export flag for exportSet to False for everyone
        for comp in allComps:
            vsp.SetSetFlag(comp, self.exportSet, False)

        self.exportComps = []
        for comp in self.allComps:
            # Check if this one is in our list:
            compName = vsp.GetContainerName(comp)
            if compName in comps:
                vsp.SetSetFlag(comp, self.exportSet, True)
                self.exportComps.append(compName)

        # Write the export file.
        self.exportSet = 9
        vsp.ExportFile(fileName, self.exportSet, vsp.EXPORT_PLOT3D)

# ----------------------------------------------------------------------
#        THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER
# ----------------------------------------------------------------------

    def _updateVSPModel(self):
        # Set each of the DVs. We have the parmID stored so its easy.
        for dvName in self.DVs:
            DV = self.DVs[dvName]
            # We use float here since sometimes pyoptsparse will give
            # stupid numpy zero-dimensional arrays, which swig does
            # not like.
            vsp.SetParmVal(DV.parmID, float(DV.value))

        # update the model
        vsp.Update()

    def _updateProjectedPts(self):
        for ptSetName in self.pointSets:
            # get the current coordinates of projection points
            n      = len(self.pointSets[ptSetName].points)
            newPts = numpy.zeros((n, 3))

            # newPts should get the new projected coords

            # get the info
            geom = self.pointSets[ptSetName].geom
            u = self.pointSets[ptSetName].u
            v = self.pointSets[ptSetName].v

            # This can all be done with arrays if we group points wrt geometry
            for i in range(n):
                # evaluate the new projected point coordinates
                pnt = vsp.CompPnt01(self.allComps[geom[i]], 0, u[i], v[i])

                # update the coordinates
                newPts[i,:] = (pnt.x(), pnt.y(), pnt.z())

            # scale vsp coordinates to mesh coordinates, do it safely above for now
            newPts *= self.vspScale

            # set the updated coordinates
            self.pointSets[ptSetName].pts = newPts

    def _getBBox(self, comp):
        # this function computes the bounding box of the component. We add some buffer on each
        # direction because we will use this bbox to determine which components to project points
        # while adding point sets

        # initialize the array
        bbox = numpy.zeros((3,2))

        # we need to get the number of main surfaces on this geometry
        nSurf = vsp.GetNumMainSurfs(comp)
        nuv = self.bboxuv.shape[0]

        # allocate the arrays
        nodes = numpy.zeros((nSurf*nuv,3))

        # loop over the surfaces
        for iSurf in range(nSurf):
            offset = iSurf * nuv
            # evaluate the points
            ptVec = vsp.CompVecPnt01(comp, iSurf, self.bboxuv[:,0], self.bboxuv[:,1])
            # now extract the coordinates from the vec3dvec...sigh...
            for i in range(nuv):
                nodes[offset + i,:] = (ptVec[i].x(), ptVec[i].y(), ptVec[i].z())

        # get the min/max values of the coordinates
        for i in range(3):
            # this might be faster if we specify row/column major
            bbox[i,0] = nodes[:,i].min()
            bbox[i,1] = nodes[:,i].max()

        # finally scale the bounding box and return
        bbox *= self.vspScale

        # also give some offset on all directions
        bbox[:,0] -= 0.1
        bbox[:,1] += 0.1

        return bbox.copy()

    def _getuv(self):
        # we need to sample the geometry, just do uniformly now
        nu = 20
        nv = 20
        nu1 = nu+1

        # define the points on the parametric domain to sample
        ul = numpy.linspace(0, 1, nu+1)
        vl = numpy.linspace(0, 1, nv+1)
        uu, vv = numpy.meshgrid(ul, vl)
        uu = uu.flatten()
        vv = vv.flatten()

        # now create a concentrated uv array
        uv = numpy.dstack((uu,vv)).squeeze()

        return uv.copy()


    def __del__(self):
        # if self.comm.rank == 0 and not self.debug:
        #     if os.path.exists(self.tmpDir):
        #         shuitl.rmtree(self.tmpDir)
        pass

    def _computeSurfJacobian(self):
        """This routine comptues the jacobian of the VSP surface with respect
        to the design variables. Since our point sets are rigidly linked to
        the VSP projection points, this is all we need to calculate. The input
        pointSets is a list or dictionary of pointSets to calculate the jacobian for.
        """

        # timing stuff:
        t1 = time.time()
        tvsp = 0
        teval = 0
        tcomm = 0
        tint = 0

        # counts
        nDV = self.getNDV()
        dvKeys = list(self.DVs.keys())
        nproc = self.comm.size
        rank  = self.comm.rank

        # arrays to collect local pointset info
        ul = numpy.zeros(0)
        vl = numpy.zeros(0)
        gl = numpy.zeros(0, dtype = 'intc')

        # save the reference seams
        for IC in self.intersectComps:
            IC.seamRef = IC.seam.flatten()
            IC.jac = numpy.zeros((len(IC.seamRef), nDV))

        for ptSetName in self.pointSets:
            # initialize the Jacobians
            self.pointSets[ptSetName].jac = numpy.zeros((3*self.pointSets[ptSetName].nPts, nDV))

            # first, we need to vstack all the point set info we have
            # counts of these are also important, saved in ptSet.nPts
            ul = numpy.concatenate((ul, self.pointSets[ptSetName].u))
            vl = numpy.concatenate((vl, self.pointSets[ptSetName].v))
            gl = numpy.concatenate((gl, self.pointSets[ptSetName].geom))

        # now figure out which proc has how many points.
        sizes = numpy.array(self.comm.allgather(len(ul)), dtype = 'intc')
        # displacements for allgather
        disp = numpy.array([numpy.sum(sizes[:i]) for i in range(nproc)], dtype='intc')
        # global number of points
        nptsg = numpy.sum(sizes)
        # create a local new point array. We will use this to get the new
        # coordinates as we perturb DVs. We just need one (instead of nDV times the size)
        # because we get the new points, calculate the jacobian and save it right after
        ptsNewL = numpy.zeros(len(ul)*3)

        # create the arrays to receive the global info
        ug = numpy.zeros(nptsg)
        vg = numpy.zeros(nptsg)
        gg = numpy.zeros(nptsg, dtype='intc')

        # Now we do an allGatherv to get a long list of all pointset information
        self.comm.Allgatherv([ul, len(ul)], [ug, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([vl, len(vl)], [vg, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([gl, len(gl)], [gg, sizes, disp, MPI.INT])

        # we now have all the point info on all procs.
        tcomm += (time.time() - t1)

        # We need to evaluate all the points on respective procs for FD computations

        # allocate memory
        pts0 = numpy.zeros((nptsg,3))

        # evaluate the points
        for j in range(nptsg):
            pnt = vsp.CompPnt01(self.allComps[gg[j]], 0, ug[j], vg[j])
            pts0[j, :] = (pnt.x(), pnt.y(), pnt.z())

        # determine how many DVs this proc will perturb.
        n = 0
        for iDV in range(len(dvKeys)):
            # I have to do this one.
            if iDV % nproc == rank:
                n += 1

        # allocate the approriate sized numpy array for the perturbed points
        ptsNew = numpy.zeros((n, nptsg, 3))

        # perturb the DVs on different procs and compute the new point coordinates.
        i = 0 # Counter on local Jac
        reqs = []
        for iDV in range(len(dvKeys)):
            # I have to do this one.
            if iDV % nproc == rank:
                # Step size for this particular DV
                dh =  self.DVs[dvKeys[iDV]].dh

                # Perturb the DV
                dvSave = self.DVs[dvKeys[iDV]].value.copy()
                self.DVs[dvKeys[iDV]].value += dh

                # update the vsp model
                t11 = time.time()
                self._updateVSPModel()
                t12 = time.time()
                tvsp += (t12-t11)

                t11 = time.time()
                # evaluate the points
                for j in range(nptsg):
                    pnt = vsp.CompPnt01(self.allComps[gg[j]], 0, ug[j], vg[j])
                    ptsNew[i, j, :] = (pnt.x(), pnt.y(), pnt.z())
                t12 = time.time()
                teval += (t12-t11)

                # now we can calculate the jac and put it back in ptsNew
                ptsNew[i, :, :] = (ptsNew[i, :, :] - pts0[:,:]) / dh

                # Do any required intersections:
                t11 = time.time()
                for IC in self.intersectComps:
                    IC.setSurface(MPI.COMM_SELF)
                    IC.jac[:,iDV] = (IC.seam.flatten() - IC.seamRef) / dh
                t12 = time.time()
                tint += (t12-t11)

                # Reset the DV
                self.DVs[dvKeys[iDV]].value = dvSave.copy()

                # reset the model.
                # t11 = time.time()
                # self._updateVSPModel()
                # t12 = time.time()
                # tvsp += (t12-t11)

                # increment the counter
                i += 1

        # scale the points
        ptsNew *= self.vspScale

        # Now, we have perturbed points on each proc that perturbed a DV

        # reset the model.
        t11 = time.time()
        self._updateVSPModel()
        t12 = time.time()
        tvsp += (t12-t11)
        # Restore the seams
        for IC in self.intersectComps:
            IC.seam = IC.seamRef.reshape(len(IC.seam), 3)

        ii = 0
        # loop over the DVs and scatter the perturbed points to original procs
        for iDV in range(len(dvKeys)):
            # Step size for this particular DV
            dh =  self.DVs[dvKeys[iDV]].dh

            t11 = time.time()
            # create the send/recv buffers for the scatter
            if iDV % nproc == rank:
                sendbuf = [ptsNew[ii, :, :].flatten(), sizes*3, disp*3, MPI.DOUBLE]
            else:
                sendbuf = [numpy.zeros((0,3)), sizes*3, disp*3, MPI.DOUBLE]
            recvbuf = [ptsNewL,  MPI.DOUBLE]

            # scatter the info from the proc that perturbed this DV to all procs
            self.comm.Scatterv(sendbuf, recvbuf, root=(iDV % nproc))

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
                self.pointSets[ptSet].jac[0:nPts*3, iDV] = ptsNewL[ibeg:iend].copy()

                # self.pointSets[ptSet].jac[0:nPts*3, iDV] = (ptsNewL[ibeg:iend] - self.pointSets[ptSet].pts.flatten())/dh

                # increment the offset
                offset += nPts

            t11 = time.time()
            # also bcast the intersection jacobians
            for IC in self.intersectComps:

                # create send/recv buffers
                if iDV%nproc == rank:
                    # we have to copy this because we need a contiguous array for bcast
                    buf = IC.jac[:,iDV].copy()
                else:
                    # receive buffer for procs that will get the result
                    buf = numpy.zeros(len(IC.seamRef))

                # bcast the intersection jacobian directly from the proc that perturbed this DV
                self.comm.Bcast([buf, len(IC.seamRef), MPI.DOUBLE], root=iDV%nproc)

                # set the value in the procs that dont have it
                if iDV%nproc != rank:
                    IC.jac[:,iDV] = buf.copy()

                # the slower version of the same bcast code
                # IC.jac[:,iDV] = self.comm.bcast(IC.jac[:,iDV], root=i%nproc)

            # pertrub the local counter on this proc.
            # This loops over the DVs that this proc perturbed
            if iDV % nproc == rank:
                ii += 1
            t12 = time.time()
            tint += (t12-t11)


        t2 = time.time()
        if rank == 0:
            print('FD jacobian calcs with dvgeovsp took',(t2-t1),'seconds in total')
            print('updating the vsp model took',tvsp,'seconds')
            print('evaluating the new points took', teval,'seconds')
            print('computing new intersections took',tint,'seconds')
            print('communication took',tcomm,'seconds')

        # set the update flags
        for ptSet in self.pointSets:
            self.updatedJac[ptSet] = True

    def _getQuads(self):
        # here we build the quad data using the internal vsp geometry
        nSurf    = len(self.allComps)
        pts      = numpy.zeros((0,3))
        conn     = numpy.zeros((0,4), dtype='intc')
        sizes    = numpy.zeros((nSurf,2), 'intc')
        cumSizes = numpy.zeros(nSurf + 1, 'intc')
        uv       = [] # this will hold tessalation points
        offset   = 0

        gind = 0
        for geom in self.allComps:
            # get uv tessalation
            utess, wtess = vsp.GetUWTess01( geom, 0 )
            # check if these values are good, otherwise, do it yourself!

            # save these values
            uv.append([numpy.array(utess), numpy.array(wtess)])
            nu = len(utess)
            nv = len(wtess)
            nElem = (nu-1)*(nv-1)

            # get u,v combinations of nodes
            uu, vv = numpy.meshgrid(utess, wtess)
            utess = uu.flatten()
            wtess = vv.flatten()

            # get the points
            ptvec = vsp.CompVecPnt01( geom, 0, utess, wtess )

            # number of nodes for this geometry
            curSize = len(ptvec)

            # initialize coordinate and connectivity arrays
            compPts  = numpy.zeros((curSize, 3))
            compConn = numpy.zeros((nElem, 4), dtype='intc')

            # get the coordinates of the points
            for i in range(curSize):
                compPts[i,:] = (ptvec[i].x(), ptvec[i].y(), ptvec[i].z())

            # build connectivity array
            k = 0
            for j in range(nv-1):
                for i in range(nu-1):
                    compConn[k,0] = j*nu     + i
                    compConn[k,1] = j*nu     + i+1
                    compConn[k,2] = (j+1)*nu + i+1
                    compConn[k,3] = (j+1)*nu + i
                    k += 1

            # apply the offset to the connectivities
            compConn += offset

            # stack the results
            pts = numpy.vstack((pts,compPts))
            conn = numpy.vstack((conn,compConn))

            # number of u and v point count
            sizes[gind,:]    = (nu, nv)
            # cumilative number of points
            cumSizes[gind+1] = cumSizes[gind] + curSize

            # increment the offset
            offset += curSize

            # increment geometry index
            gind += 1

        # finally, scale the points and save the data
        self.pts0     = pts*self.vspScale
        self.conn     = conn
        self.sizes    = sizes
        self.cumSizes = cumSizes
        self.uv       = uv

class vspDV(object):

    def __init__(self, parmID, component, group, parm, value, lower, upper, scale, dh):
        """Inernal class for storing VSP design variable information"""
        self.parmID = parmID
        self.component = component
        self.group = group
        self.parm = parm
        self.value = numpy.array(value)
        self.lower = lower
        self.upper = upper
        self.dh = dh
        self.scale = scale

class PointSet(object):
    def __init__(self, points, pts, geom, u, v):
        self.points = points
        self.pts = pts
        self.geom = geom
        self.u = u
        self.v = v
        self.offset = self.pts - self.points
        self.nPts = len(self.pts)
        self.jac = None

# object to keep track of feature curves if the user supplies them to track intersections
class featureCurve(object):
    def __init__(self, name, conn):
        # TODO we can remove this and just add the conn to a list.
        self.name = name
        self.conn=conn

class CompIntersection(object):
    def __init__(self, intDict, dvGeo):
        '''Class to store information required for an intersection.
        Here, we use some fortran code from pySurf.

        Input
        -----
        compA: ID of the first  VSP geometry
        compB: ID of the second VSP geomery
        extraComps, list : IDs of other geometries this intersection will affect
        direction: direction of the constant parametric lines used in VSP model for the
        intersection. This can be either u or v. e.g. users should check if u or v is constant
        along the geometry (compB) that is fully intersected by the larger geometry (compA). In
        this case, compA can be the fuselage, compB can be the wing (or vertical tail), and the
        direction parameter should be the direction of the parameter that is defined as spanwise
        in the VSP model.

        dStar, real : Radius over which to attenuate the deformation

        old comment:
        Internally we will store the indices and the weights of the
        points that this intersection will have to modify. In general,
        all this code is not super efficient since it's all python,
        but it should not be more of a bottleneck than VSP is itself
        in doing the export.
        '''

        # get the communicator from dvgeo
        self.comm    = dvGeo.comm
        self.projTol = dvGeo.projTol
        self.allComps = dvGeo.allComps

        # here we grab the useful information from the dictionary,
        # or revert to defaults if no information is provided.

        # first make all dict. keys lowercase
        intDict = dict((k.lower(), v) for k,v in intDict.items())

        # names of compA and compB must be in the dictionary
        self.compA = dvGeo.allComps[dvGeo.compNames.index(intDict['compa'])]
        self.compB = dvGeo.allComps[dvGeo.compNames.index(intDict['compb'])]

        # extract extra components if any
        self.extraComps = []
        if 'extracomps' in intDict:
            for j in range(len(intDict['extracomps'])):
                self.extraComps.append(dvGeo.allComps[dvGeo.compNames.index(j)])

        self.dStar = intDict['dstar']

        self.vspScale = dvGeo.vspScale

        self.trackFeatures = True
        self.featureCurves = []
        if 'trackfeatures' in intDict:
            self.trackFeatures = intDict['trackfeatures']

        self.trackCurves = False
        if 'trackcurves' in intDict:
            self.trackCurves = intDict['trackcurves']

        self.halfdStar = self.dStar/2.0
        self.ptSets = OrderedDict()
        self.meshScale = 1./self.vspScale

        # each component may have more than 1 surface. CHECK if this is actually the case, we might consider removing this feature.
        self.nSurfA = vsp.GetNumMainSurfs(self.compA)
        self.nSurfB = vsp.GetNumMainSurfs(self.compB)
        if self.nSurfA != 1 or self.nSurfB != 1:
            raise Error('the intersected components should have a single main surface')

        # direction (u or v)
        self.dir = 1
        if 'dir' in intDict:
            self.dir = intDict['dir'].lower()

        # if we are using legacy code, just conver the previous intersection directions to u or v
        if 'j' in self.dir or 'u' in self.dir:
            self.dir = 0
        elif 'k' in self.dir or 'v' in self.dir:
            self.dir = 1
        else:
            # just default to 'v' for now, BUT we can automatically pick this based on the range of u,v values we get on our intersection curves once we calculate them. If thats the case, we can set self.dir to None then calculate it later bec. we'll need the intersection curve itself for this.
            self.dir = 1

        # similarly read the unit normal of the symmetry plane
        self.symDir = 'y'
        if 'symdir' in intDict:
            self.symDir = intDict['symdir'].lower()

        # convert to int
        if 'i' in self.symDir or 'x' in self.symDir:
            self.symDir = 0
        elif 'k' in self.symDir or 'z' in self.symDir:
            self.symDir = 2
        # default to y
        else:
            self.symDir = 1

        # feature angle
        self.featureAngle = 60.0*numpy.pi/180.0
        if 'featureangle' in intDict:
            self.featureAngle = intDict['featureangle']*numpy.pi/180.0

        # feature curve names
        self.featureCurveNames = []
        if 'featurecurves' in intDict:
            self.featureCurveNames = intDict['featurecurves']

        self.distTol = 1e-14 # Ney's default was 1e-7
        if 'disttol' in intDict:
            self.distTol = intDict['disttol']

        # we now need to triangulate the surfaces.

        # if the user has not supplied a cgns file for either of the surfaces, we need to sample
        # them ourselves
        if ('compafile' not in intDict) or ('compbfile' not in intDict):
            # if we have at least one surface we are triangulating, we need to correct for errors
            # in the intersection connectivity
            self.correctConn = True
            # check if the user prescribed the number of intervals in u and v
            if 'nu' in intDict:
                nu = intDict['nu']
            else:
                # set the default
                nu = 4*48
            # do the same for nv
            if 'nv' in intDict:
                nv = intDict['nv']
            else:
                # set the default
                nv = 4*48
        else:
            self.correctConn = False

        # surface A:
        if 'compafile' in intDict:
            # user has provided a cgns grid. Read the nodes and elements defining the surface and
            # save them. We also save the parametric positions of these nodes on the vsp geometry.
            self.nodesA, self.uvA, self.triConnA = self._readCGNSFile(self.compA, intDict['compafile'])
            self.connA = numpy.zeros((0, 4))
        else:
            # user has not provided a cgns file for this surface, uniformly sample in uv space
            self.nodesA, self.uvA, self.triConnA = self._getSurfaceTris(self.compA, self.nSurfA, nu, nv)
            self.connA = numpy.zeros((0, 4))

        # do the same for compB
        if 'compbfile' in intDict:
            self.nodesB, self.uvB, self.triConnB = self._readCGNSFile(self.compB, intDict['compbfile'])
            self.connB = numpy.zeros((0, 4))
        else:
            self.nodesB, self.uvB, self.triConnB = self._getSurfaceTris(self.compB, self.nSurfB, nu, nv)
            self.connB = numpy.zeros((0, 4))

        # only the node coordinates will be modified for the intersection calculations because we have calculated and saved all the connectivity information
        self.seam0 = self._getIntersectionSeam(self.comm , firstCall = True)
        self.seam = self.seam0.copy()

    def setSurface(self, comm):
        """ This set the new udpated surface on which we need to comptue the new intersection curve"""

        # get the updated surface coordinates
        self._getUpdatedCoords()

        self.seam = self._getIntersectionSeam(comm)

    def addPointSet(self, pts, geom, ptSetName):

        # Figure out which points this intersection object has to deal with.

        # use Ney's fortran code to project the point on curve
        # Get number of points
        nPoints = len(pts)

        # Initialize references if user provided none
        dist2 = numpy.ones(nPoints)*1e10
        xyzProj = numpy.zeros((nPoints,3))
        tanProj = numpy.zeros((nPoints,3))
        elemIDs = numpy.zeros((nPoints),dtype='int32')

        # only call the fortran code if we have at least one point
        if nPoints > 0:
            # Call fortran code
            # This will modify xyzProj, tanProj, dist2, and elemIDs if we find better projections than dist2.
            # Remember that we should adjust some indices before calling the Fortran code
            # Remember to use [:] to don't lose the pointer (elemIDs is an input/output variable)
            elemIDs[:] = elemIDs + 1 # (we need to do this separetely because Fortran will actively change elemIDs contents.
            curveMask = curveSearchAPI.curvesearchapi.mindistancecurve(pts.T,
                                                                       self.nodes0.T,
                                                                       self.conn0.T + 1,
                                                                       xyzProj.T,
                                                                       tanProj.T,
                                                                       dist2,
                                                                       elemIDs)

            # Adjust indices back to Python standards
            elemIDs[:] = elemIDs - 1

        # dist2 has the array of squared distances
        d = numpy.sqrt(dist2)

        indices = []
        factors = []
        for i in range(len(pts)):
            if self.allComps[geom[i]] == self.compA or self.allComps[geom[i]] == self.compB or self.allComps[geom[i]] in self.extraComps:
                if d[i] < self.dStar:

                    # Compute the factor
                    if d[i] < self.halfdStar:
                        factor = .5*(d[i]/self.halfdStar)**3
                    else:
                        factor = .5*(2-((self.dStar - d[i])/self.halfdStar)**3)

                    # Save the index and factor
                    indices.append(i)
                    factors.append(factor)

        # # Print the number of points that get associated with the intersection.
        # nPointGlobal = self.comm.allreduce(len(factors), op=MPI.SUM)
        # if self.comm.rank == 0:
        #     intName = vsp.GetContainerName(self.compA)+'_'+vsp.GetContainerName(self.compB)
        #     print('DVGEO VSP:\n%d points associated with intersection %s'%(nPointGlobal, intName))

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

    def update_d(self, ptSetName, dPt, dSeam):

        """forward mode differentiated version of the update routine.
        Note that dPt and dSeam are both one dimensional arrays
        """
        pts     = self.ptSets[ptSetName][0]
        indices = self.ptSets[ptSetName][1]
        factors = self.ptSets[ptSetName][2]

        # we need to reshape the arrays for simpler code
        dSeam = dSeam.reshape((len(self.seam0), 3))

        for i in range(len(factors)):
            # j is the index of the point in the full set we are
            # working with.
            j = indices[i]

            # Do it vectorized
            rr = pts[j] - self.seam0
            LdefoDist = (1.0/numpy.sqrt(rr[:,0]**2 + rr[:,1]**2 + rr[:,2]**2+1e-16))
            LdefoDist3 = LdefoDist**3
            Wi = LdefoDist3
            den = numpy.sum(Wi)
            interp_d = numpy.zeros(3)
            for iDim in range(3):
                interp_d[iDim] = numpy.sum(Wi*dSeam[:, iDim])/den

                # Now the delta is replaced by 1-factor times the weighted
                # interp of the seam * factor of the original:
                dPt[j*3 + iDim] = factors[i]*dPt[j*3 + iDim] + (1-factors[i])*interp_d[iDim]

        return

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

    def _readCGNSFile(self, comp, filename):
        # this function reads the unstructured CGNS grid in filename and returns node coordinates,
        # projected u,v parameters for nodes, and element connectivities.
        # Here, only the root proc reads the cgns file, broadcasts node and connectivity info.
        # Then, we project these nodes on the vsp geometry in parallel,
        # and gather the uv coordinates of the surface nodes.

        # save these so we dont need the objects to read the values
        nproc = self.comm.size
        rank  = self.comm.rank

        # print some info
        if rank == 0:
            print('Reading file %s'%filename)

        # only root proc reads the file
        if rank == 0:
            # use the default routine in tsurftools
            print("reading the file on root")
            t1 = time.time()
            nodes, sectionDict = tsurf_tools.getCGNSsections(filename, comm = MPI.COMM_SELF)
            t2 = time.time()
            print("read the cgns file")
            print('reading the cgns file took', (t2-t1))

            # scale  the node coordinates to be consistent with the vsp scaling
            # nodes *= self.meshScale

            triConn = numpy.zeros((0,3), dtype = numpy.int8)
            # print('Part names in triangulated cgns file for %s'%filename)
            for part in sectionDict:
                # print(part)
                if 'triaConnF' in sectionDict[part].keys():
                    #this is a surface, read the tri connectivities
                    triConn = numpy.vstack((triConn, sectionDict[part]['triaConnF']))

                # also check if any of the feature curves are in here
                for curveName in self.featureCurveNames:
                    if curveName.lower() == part:
                        # we have found a feature curve!
                        print('Found feature curve %s'%curveName)
                        # print(sectionDict[part].keys())
                        self.featureCurves.append(featureCurve(curveName, sectionDict[part]['barsConn']))
        else:
            # create these to recieve the data
            nodes = None
            triConn = None

        # each proc gets the nodes and connectivities
        # CHECK if this should be bcast or Bcast...
        nodes = self.comm.bcast(nodes, root=0)
        triConn = self.comm.bcast(triConn, root=0)

        # time the projection code
        t1 = time.time()

        # now we need to figure out the indices the procs will operate on
        # first, equally divide the number of nodes among procs
        counts = numpy.full(nproc, int(nodes.shape[0]/nproc))
        # get the remainder node count and increase the number of nodes by one for this many procs
        counts[:nodes.shape[0]%nproc] += 1

        # calculate the displacements in the full array. This will also be useful when we are
        # reading the nodes on different procs.
        disp = [numpy.sum(counts[:i]) for i in range(nproc)]

        # starting index on each proc
        iBeg = int(disp[rank])

        pnt = vsp.vec3d()
        uLocal = numpy.zeros((int(counts[rank])))
        vLocal = numpy.zeros((int(counts[rank])))
        dMaxLocal = 1e-16

        # Here, we will build a quad mesh and project the points using the adt in pyspline
        # to get a good initial guess for the fast projection code...
        # get uv tessalation
        utess, wtess = vsp.GetUWTess01( comp, 0 )
        # check if these values are good, otherwise, do it yourself!

        # save these values
        nu = len(utess)
        nv = len(wtess)
        nElem = (nu-1)*(nv-1)

        # get u,v combinations of nodes
        uu, vv = numpy.meshgrid(utess, wtess)
        uu = uu.flatten()
        vv = vv.flatten()

        # get the points
        ptvec = vsp.CompVecPnt01( comp, 0, uu, vv )

        # number of nodes for this geometry
        curSize = len(ptvec)

        # initialize coordinate and connectivity arrays
        pts  = numpy.zeros((curSize, 3))
        conn = numpy.zeros((nElem, 4), dtype='intc')

        # get the coordinates of the points
        for i in range(curSize):
            pts[i,:] = (ptvec[i].x(), ptvec[i].y(), ptvec[i].z())

        # build connectivity array
        k = 0
        for j in range(nv-1):
            for i in range(nu-1):
                conn[k,0] = j*nu     + i
                conn[k,1] = j*nu     + i+1
                conn[k,2] = (j+1)*nu + i+1
                conn[k,3] = (j+1)*nu + i
                k += 1

        # scale the points
        # pts *= self.vspScale

        # use the adt to get the initial guesses
        # faceID has the index of the corresponding quad element.
        # uv has the parametric u and v weights of the projected point.
        faceID, uv = pySpline.libspline.adtprojections(
            pts.T, (conn+1).T, nodes[iBeg:iBeg+int(counts[rank])].T)
        uv = uv.T
        faceID -= 1 # Convert back to zero-based indexing.

        for i in range(int(counts[rank])):
            # this is the global index
            ig = i+iBeg
            # set the coordinates of the point object
            pnt.set_xyz(nodes[ig,0], nodes[ig,1], nodes[ig,2])
            # pnt.set_xyz(nodes[ig,0] * self.meshScale,  nodes[ig,1] * self.meshScale, nodes[ig,2] * self.meshScale)

            # this is the global index of the first node of the projected element
            nn = conn[faceID[i],0]
            # figure out the i and j indices of the first point of the element
            # we projected this point to
            ii = numpy.mod(nn, nu)
            jj = numpy.floor_divide(nn, nu)

            # calculate the global u and v change in this element
            du = utess[ii+1] - utess[ii]
            dv = wtess[jj+1] - wtess[jj]

            # now get this points u,v coordinates on the vsp geometry and add
            # compute the initial guess using the  tessalation data of the surface
            ug = uv[i,0]*du + utess[ii]
            vg = uv[i,1]*dv + wtess[jj]

            # project the point
            d, uLocal[i], vLocal[i] = vsp.ProjPnt01Guess( comp, 0, pnt, ug, vg )

            # if we dont have a good projection, try projecting again to surfaces
            #  with the slow code.
            if (d > self.projTol):
                d, surf, uLocal[i], vLocal[i] = vsp.ProjPnt01I(comp, pnt)

            dMaxLocal = max(dMaxLocal, d)

        # print some debug info
        dMax = self.comm.allreduce(dMaxLocal, op=MPI.MAX)
        t2 = time.time()
        if rank == 0:
            print('projecting the triangulation points took ',(t2-t1))
            print('Max distance between intersection triangulation and vsp surface is',dMax)

        # we can dstack and flatten the numpy array locally
        uvLocal = numpy.dstack((uLocal,vLocal)).squeeze().flatten()

        # we send the full uvLocal array on each proc
        sendBuff = [uvLocal, len(uvLocal)]

        # we need to multiply counts and disp by 2 because we are adding u and v arrays back to back
        counts *= 2
        disp = [2*d for d in disp]
        # create the empty uv array.
        uv = numpy.zeros(nodes.shape[0]*2)
        recvBuff = [uv, counts, disp, MPI.DOUBLE]

        # actually do the exchange
        self.comm.Allgatherv(sendBuff, recvBuff)

        # re-shape the uv data
        uv.shape = nodes.shape[0], 2

        # finally, evaluate the vsp coordinates of the projections,
        # as we will directly use them instead of the coordinates of the cgns mesh
        ptVec = vsp.CompVecPnt01(comp, 0, uv[:,0], uv[:,1])
        for i in range(len(ptVec)):
            nodes[i,:] = (ptVec[i].x(), ptVec[i].y(), ptVec[i].z())
        nodes *= self.vspScale

        return nodes, uv, triConn

    def _getSurfaceTris(self, comp, nSurf, nu, nv):
        # This function will triangulate the component defined by comp
        # for book-keeping
        nu1 = nu+1

        # define the points on the parametric domain to sample
        ul = numpy.linspace(0, 1, nu+1)
        vl = numpy.linspace(0, 1, nv+1)
        uu, vv = numpy.meshgrid(ul, vl)
        uu = uu.flatten()
        vv = vv.flatten()

        # now create a concentrated uv array
        uv = numpy.dstack((uu,vv)).squeeze()

        # we can just append the nodes on different surfaces back to back?
        nNodes = len(uu)
        nodes  = numpy.zeros((nSurf*nNodes, 3))

        # allocate the connectivity array
        nTris  = 2 * nu * nv
        triConn = numpy.zeros((nSurf*nTris, 3))

        # get the connectivity matrix. We just get quad connectivities
        # as Ney's code is faster at turning them into tris. This is
        # the "same" for each surface, so we will just copy-paste
        elemCount = 0
        for j in range(nv):
            for i in range(nu):
                # triConn[elemCount, 0]   = j*nu1     + i
                # triConn[elemCount, 1]   = j*nu1     + i+1
                # triConn[elemCount, 2]   = (j+1)*nu1 + i+1
                #
                # triConn[elemCount+1, 0] = (j+1)*nu1 + i+1
                # triConn[elemCount+1, 1] = (j+1)*nu1 + i
                # triConn[elemCount+1, 2] = j*nu1     + i

                # this formulation assumes the geometry is wrapped around v,
                # so that v = 0 and v = 1 give the same curve in the physical space
                triConn[elemCount, 0]   = j*nu1     + i
                triConn[elemCount, 1]   = j*nu1     + i+1
                triConn[elemCount, 2]   = ((j+1)%nv)*nu1 + i+1

                triConn[elemCount+1, 0] = ((j+1)%nv)*nu1 + i+1
                triConn[elemCount+1, 1] = ((j+1)%nv)*nu1 + i
                triConn[elemCount+1, 2] = j*nu1     + i

                elemCount += 2

        # loop over the surfaces
        for iSurf in range(nSurf):
            # initial index of nodes for this surface
            offset = iSurf * nNodes

            # copy the connectivity matrix for the latter surfaces
            if iSurf != 0:
                # copy the same connectivity matrix by including the offset
                triConn[iSurf*nTris:(iSurf+1)*nTris,:] = triConn[0:nTris, :].copy() + offset

            # evaluate the points
            ptVec = vsp.CompVecPnt01(comp, iSurf, uv[:,0], uv[:,1])
            # now extract the coordinates from the vec3dvec...sigh...
            for i in range(nNodes):
                nodes[offset + i,:] = (ptVec[i].x(), ptVec[i].y(), ptVec[i].z())

        # adjust the indices for fortran ordering
        triConn += 1

        # scale the nodes
        nodes *= self.vspScale

        # return the coordinates and the connectivities
        return nodes.copy(), uv.copy(), triConn.copy()

    def _getUpdatedCoords(self):
        # this code returns the updated coordinates

        # first comp a
        ptVec = vsp.CompVecPnt01(self.compA, 0, self.uvA[:,0], self.uvA[:,1])
        for i in range(len(self.nodesA)):
            self.nodesA[i,:] = (ptVec[i].x(), ptVec[i].y(), ptVec[i].z())
        self.nodesA *= self.vspScale

        # then comp b
        ptVec = vsp.CompVecPnt01(self.compB, 0, self.uvB[:,0], self.uvB[:,1])
        for i in range(len(self.nodesB)):
            self.nodesB[i,:] = (ptVec[i].x(), ptVec[i].y(), ptVec[i].z())
        self.nodesB *= self.vspScale

        return

    def _getIntersectionSeam(self, comm, firstCall = False):
        # we can parallelize here. each proc gets one intersection, but needs re-structuring of some of the code.

        # this function computes the intersection curve, cleans up the data and splits the curve based on features or curves specified by the user.

        # Call Ney's code with the quad information.
        # only root does this as this is serial code
        if comm.rank == 0:
            # compute the intersection curve, in the first step we just get the array sizes to hide allocatable arrays from python
            arraySizes = intersectionAPI.intersectionapi.computeintersection(self.nodesA.T,
                                                                             self.triConnA.T,
                                                                             self.connA.T,
                                                                             self.nodesB.T,
                                                                             self.triConnB.T,
                                                                             self.connB.T,
                                                                             self.distTol,
                                                                             MPI.COMM_SELF.py2f())

            # Retrieve results from Fortran if we have an intersection
            if numpy.max(arraySizes[1:]) > 0:

                # Second Fortran call to retrieve data from the CGNS file.
                intersectionArrays = intersectionAPI.intersectionapi.retrievedata(*arraySizes)

                # We need to do actual copies, otherwise data will be overwritten if we compute another intersection.
                # We subtract one to make indices consistent with the Python 0-based indices.
                # We also need to transpose it since Python and Fortran use different orderings to store matrices in memory.
                intNodes = numpy.array(intersectionArrays[0]).T[0:-1] # last entry is 0,0,0 for some reason, CHECK THIS! Checked, still zero for a proper intersection.
                barsConn = numpy.array(intersectionArrays[1]).T - 1
                # parentTria = numpy.array(intersectionArrays[2]).T - 1

                # write this to a file to check.
                # only write the intersection data if the comm is the global comm.
                # if comm.size > 1:
                #     self.count += 1
                #     fileName = 'int_'+vsp.GetContainerName(self.compA)+'_'+vsp.GetContainerName(self.compB)+'_%d'%self.count
                #     pysurf.tecplot_interface.writeTecplotFEdata(intNodes,barsConn,fileName,fileName)
                    # pysurf.tecplot_interface.writeTecplotFEdata(intNodes * self.meshScale,barsConn,fileName,fileName)

            else:
                raise Error('DVGeometryVSP Warning: The components %s and %s do not intersect.'%(vsp.GetContainerName(self.compA), vsp.GetContainerName(self.compB)))

            # Release memory used by fortran
            intersectionAPI.intersectionapi.releasememory()
        else:
            intNodes = None
            barsConn = None

        # broadcast the arrays
        intNodes = comm.bcast(intNodes, root = 0)
        barsConn = comm.bcast(barsConn, root = 0)

        # now we need to eliminate the duplicates, order the connectivities and figure out the features

        # if self.comm.rank == 0:
            # print('barsconn', barsConn)
            # newConnUnmodified, _ = tsurf_tools.FEsort(barsConn.tolist())
            # print('newconn', newConnUnmodified)

        if self.correctConn:
            # first order within bar elements
            for i in range(len(barsConn)):
                if barsConn[i,0]>barsConn[i,1]:
                    dummy = barsConn[i,0]
                    barsConn[i,0] = barsConn[i,1]
                    barsConn[i,1] = dummy

            # now remove duplicates and point elements
            uniqueConn = []
            for x in barsConn.tolist():
                # loop over the elements and check
                found = False
                for y in uniqueConn:
                    if x[0] == y[0] and x[1] == y[1]:
                        # print('found a repeated bar element...')
                        found = True
                # we found a new element that was not in our new list
                if not found:
                    # finally check if this is a point element
                    if x[0] != x[1]:
                        uniqueConn.append(x)
                    # else:
                    #     print('found a point element...')

            # now we can sort the seam connectivities to get a continuous intersection curve
            newConn, newMap = tsurf_tools.FEsort(uniqueConn)
            # print("corrected conn:",newConn)
        else:
            newConn, newMap = tsurf_tools.FEsort(barsConn.tolist())

        # newConn might have multiple intersection curves
        if len(newConn) == 1:
            # we have a single intersection curve, just take this.
            seamConn = newConn[0].copy()
        # we might have two of the same curve on both sides of the symmetry plane, if so, get the one on the positive side
        elif len(newConn) == 2:
            # check which curve is on the positive side
            # fix the 2 to get the correct option for symmetry plane normal direction
            if intNodes[newConn[0][0][0], self.symDir] > 0.0 and intNodes[newConn[1][0][0], self.symDir] < 0.0:
                # the first array is what we need
                seamConn = newConn[0].copy()
            elif intNodes[newConn[0][0][0], self.symDir] < 0.0 and intNodes[newConn[1][0][0], self.symDir] > 0.0:
                # the second array is what we need
                seamConn = newConn[1].copy()
            else:
                # throw an error. each component pair should have one intersection on one side of the symmetry plane
                raise Error("Error at DVGeometryVSP. The intersection between two components should have a single intersection curve on each side of the symmetry plane.")
        # print(seamConn)
        # print(intNodes[seamConn][:,0])

        # Get the number of elements
        nElem = seamConn.shape[0]

        # now that we have a continuous, ordered seam connectivity in seamConn, we can try to detect features
        if firstCall or self.trackFeatures or self.trackCurves:

            # we need to track the nodes that are closest to the supplied feature curves
            if self.trackCurves:
                breakList = []
                # loop over the feature curves
                for curve in self.featureCurves:
                    # find the closest point on the intersection to this curve
                    # print('Finding closest node to', curve.name)
                    # print(curve.conn)

                    # use Ney's fortran code to project the point on curve
                    # first, we need to get a list of nodes that define the intersection
                    intNodesOrd = intNodes[seamConn[:,0]]
                    # print('ordered array of intersection nodes', intNodesOrd)

                    # Get number of points
                    nPoints = len(intNodesOrd)

                    # Initialize references if user provided none
                    dist2 = numpy.ones(nPoints)*1e10
                    xyzProj = numpy.zeros((nPoints,3))
                    tanProj = numpy.zeros((nPoints,3))
                    elemIDs = numpy.zeros((nPoints),dtype='int32')

                    # then find the closest point to the curve

                    # only call the fortran code if we have at least one point
                    if nPoints > 0:
                        # Call fortran code
                        # This will modify xyzProj, tanProj, dist2, and elemIDs if we find better projections than dist2.
                        # Remember that we should adjust some indices before calling the Fortran code
                        # Remember to use [:] to don't lose the pointer (elemIDs is an input/output variable)
                        elemIDs[:] = elemIDs + 1 # (we need to do this separetely because Fortran will actively change elemIDs contents.
                        curveMask = curveSearchAPI.curvesearchapi.mindistancecurve(intNodesOrd.T,
                                                                                self.nodesB.T,
                                                                                curve.conn.T + 1,
                                                                                xyzProj.T,
                                                                                tanProj.T,
                                                                                dist2,
                                                                                elemIDs)

                        # Adjust indices back to Python standards
                        elemIDs[:] = elemIDs - 1

                    # now, find the index of the smallest distance
                    # print('index with smallest dist' , numpy.argmin(dist2))
                    breakList.append(numpy.argmin(dist2))

            # we either track features, or this is the first call and we just want to identify the feature nodes
            else:

                # this can be done in two ways; we can either check for nodes where the seam makes a sharp turn, OR we can find the closest node we have to the user-supplied "feature curve", such as lower/upper trailing edges.

                # detect the features by angle; this is mostly copied from Ney's code.
                sharpAngle = self.featureAngle

                # Get coordinates and connectivities
                # coor = intNodes
                # barsConn = seamConn
                breakList = []

                # Get the tangent direction of the first bar element
                prevTan = intNodes[seamConn[0, 1], :] - intNodes[seamConn[0, 0], :]
                prevTan = prevTan/numpy.linalg.norm(prevTan)

                # Loop over the remaining bars to find sharp kinks
                for elemID in range(1,nElem):

                    # Compute tangent direction of the current element
                    currTan = intNodes[seamConn[elemID, 1], :] - intNodes[seamConn[elemID, 0], :]
                    currTan = currTan/numpy.linalg.norm(currTan)

                    # Compute change in direction between consecutive tangents
                    angle = numpy.arccos(numpy.min([numpy.max([prevTan.dot(currTan),-1.0]),1.0]))

                    # Check if the angle is beyond a certain threshold
                    if angle > sharpAngle:

                        # Store the current element as a break position
                        breakList.append(elemID)

                    # Now the current tangent will become the previous tangent of
                    # the next iteration

                    prevTan = currTan.copy()

                # check if the curve is periodic;
                if seamConn[0,0] == seamConn[-1, 1]:
                    # repeat for the connection between the last and first elements

                    # Compute tangent direction of the current element
                    prevTan = intNodes[seamConn[-1, 1], :] - intNodes[seamConn[-1, 0], :]
                    prevTan = prevTan/numpy.linalg.norm(prevTan)

                    # Get the tangent direction of the first bar element
                    currTan = intNodes[seamConn[0, 1], :] - intNodes[seamConn[0, 0], :]
                    currTan = currTan/numpy.linalg.norm(currTan)

                    # Compute change in direction between consecutive tangents
                    angle = numpy.arccos(numpy.min([numpy.max([prevTan.dot(currTan),-1.0]),1.0]))

                    # Check if the angle is beyond a certain threshold
                    if angle > sharpAngle:
                        # Store the current element as a break position
                        breakList.append(0)
                breakList = numpy.array(breakList)

            # print("after feature detection")
            # # these are the elements at "features"
            # print(breakList)
            # # these are the nodes of the element
            # print(seamConn[breakList])
            # # these are the feature nodes (i.e. the first nodes of feature elements)
            # print(intNodes[seamConn[breakList,0]])

            nFeature = len(breakList)

            # print('nFeatures on intersection between',vsp.GetContainerName(self.compA),'and',vsp.GetContainerName(self.compB),'is',nFeature)
            # print(breakList)
            # print(seamConn[breakList])
            # print(intNodes[seamConn[breakList,0]])

            # if this is the first call,
            if firstCall:
                # we also save the initial curve with nodes and connectivities for distance calculations
                self.conn0  = seamConn.copy()
                self.nodes0 = intNodes.copy()
                self.nFeature = nFeature
            else:
                if nFeature != self.nFeature:
                    raise AnalysisError('Number of features on the intersection curve has changed.')
                    # raise Error('Number of features on the intersection curve has changed.')

            # we can also automatically determine the intersection direction here.
            if self.dir is None:
                # just set it to w for now (0 = u, 1 = w or v)
                self.dir = 1

                # we need to project only the nodes on the intersection curve to the VSP geometry and calculate the range of u,v values we get for the intersection. If the span of v is small, then we use u for the intersection reference (e.g. u goes along a wing span-wise vs v is chord-wise. A wing-body intersection would have u in [0,1] but only a small range for v.)

            # we need to check for a few exceptions now:

            # if we have no feature nodes, we need to project the whole curve onto the VSP geometry and pick the point with lowest u or v value as our feature node

            # if we have a single feature node (the previous case will automatically go into this category), we need to set the curveSizes to contain the whole curve, from the feature node, to the end node. This is possibly a closed curve, so we just need to set the size to the whole intersection curve.

            # For all these exceptions, also modify the number of feature points

            # project the feature nodes on components to get the parametric coordinates
            # dont worry about having multiple surfaces per component now...

            # first get an ordered list of the feature points
            # this is just our breakList "list"
            featurePoints = intNodes[seamConn[breakList,0]]
            # print(featurePoints)

            # if we have a larger number of surfaces, throw an error
            if self.nSurfB > 1:
                raise Error('I have more than one surface defined for the fully intersected geometry in VSP.')

            # project these points to the VSP geometry
            # we will use compB for all of these operations
            parList = []
            dMax = 1e-16

            # initialize one 3dvec for projections
            pnt = vsp.vec3d()

            for i in range(len(featurePoints)):
                # set the coordinates of the point object
                pnt.set_x(featurePoints[i,0] * self.meshScale)
                pnt.set_y(featurePoints[i,1] * self.meshScale)
                pnt.set_z(featurePoints[i,2] * self.meshScale)
                d, surf, uout, vout = vsp.ProjPnt01I(self.compB, pnt)
                par = [uout, vout]

                # print(d, surf, uout, vout, featurePoints[i])

                parList.append(par[self.dir])

                dMax = max(d, dMax)

            # print('Maximum distance between the intersection feature nodes and the vsp surface is %1.15f'%dMax)

            # print(parList)

            # copy parlist and order it
            parListCopy = parList[:]

            parListCopy.sort()

            if not self.trackFeatures:
                # we need to track the u or v parameter of initial feature nodes
                self.parRef = parList[:]

        # or we will just project all the intersection points to the VSP geometry and get u,v data
        else:
            # get an array of nodes composed of first nodes in the intersection elements

            # initialize one 3dvec for projections
            pnt = vsp.vec3d()
            curveParList = []
            for i in range(len(seamConn)):
                point = intNodes[seamConn[i,0]]
                # set the coordinates of the point object
                pnt.set_x(point[0] * self.meshScale)
                pnt.set_y(point[1] * self.meshScale)
                pnt.set_z(point[2] * self.meshScale)
                # project onto VSP geometry
                d, surf, uout, vout = vsp.ProjPnt01I(self.compB, pnt)
                par = [uout, vout]
                # get the u or v parameter
                curveParList.append([par[self.dir]])

            # build a ckdtree for fast lookup
            parTree = cKDTree(curveParList)

            # get the closest entries of u or v parameters and add them to breaklist
            dummy = numpy.zeros((self.nFeature,1))
            dummy[:,0] = self.parRef[:]
            _, breakList = parTree.query(dummy)
            # print('breakList after the first call')
            # print(breakList)

            parList = numpy.array(curveParList)[breakList].tolist()

            # copy parlist and order it
            parListCopy = parList[:]

            parListCopy.sort()

            # we simply tracked the initial features in parametric space so we have the same number of features
            nFeature = self.nFeature


        # flip

        # to figure out if we need to flip the curve, check the first element's direction that has the first feature node. We want to go in increasing parameter direction
        point = intNodes[seamConn[breakList[parList.index(parListCopy[0])],1]]
        # print(intNodes[seamConn[breakList[parList.index(parListCopy[0])],0]])
        # print(point)

        # project this point onto the VSP geometry
        pnt.set_x(point[0] * self.meshScale)
        pnt.set_y(point[1] * self.meshScale)
        pnt.set_z(point[2] * self.meshScale)
        d, surf, uout, vout = vsp.ProjPnt01I(self.compB, pnt)
        par = [uout, vout]

        # now check if we need to flip the curve
        if par[self.dir] < parListCopy[0]:
            # print('I am flipping the intersection curve')
            # we need to reverse the order of our feature curves
            # and we will flip the elements too so keep track of this change
            breakList = numpy.mod(seamConn.shape[0] - numpy.flip(breakList, axis=0), seamConn.shape[0])

            # and we need to invert the curves themselves
            seamConn = numpy.flip(seamConn, axis = 0)
            seamConn = numpy.flip(seamConn, axis = 1)

            # and parameter list
            parList.reverse()

        # # print(parListCopy)
        # # for i in range(len(parList)):
        #     # print(parList.index(parListCopy[i]))
        #
        # # get the ordering in u or w. just do increasing order...
        #
        # print(seamConn)
        # print(breakList)
        #
        # # check if the direction is correct, if not, flip the intersection curve
        # # we can just check if the sorted and un-sorted versions are the same..
        # if parList != parListCopy:
        #     # we need to reverse the order of our feature curves
        #     # and we will flip the elements too so keep track of this change
        #     breakList = numpy.mod(seamConn.shape[0] - numpy.flip(numpy.array(breakList)), seamConn.shape[0])
        #
        #     # and we need to inver the curves themselves
        #     seamConn = numpy.flip(seamConn)
        #
        # print(seamConn)
        # print("After flipping the curve")
        # print(breakList)
        # #
        # print(seamConn[breakList])
        # #
        # print(intNodes[seamConn[breakList,0]])
        #
        # print(parList)
        # print(parListCopy)

        # we just need to roll the connectivity curve to have the first element that has the first feature node as the first one
        seamConn = numpy.roll(seamConn, -breakList[parList.index(parListCopy[0])], axis = 0)
        # print("rolled by %d"%-breakList[parList.index(parListCopy[0])])
        # print(seamConn)

        # also adjust the feature indices
        # print(breakList)
        breakList = numpy.mod(breakList - breakList[parList.index(parListCopy[0])], nElem)
        breakList.sort()
        # if 0 in breakList:
            # breakList[breakList.index(0)] = nElem
        # breakList = [nElem if a == 0 for a in breakList]
        # print(breakList)

        # orderedSeam = numpy.empty((0,2),dtype = int)
        curveSizes = []
        for i in range(nFeature-1):
            # get the elements starting from the smallest reference parameter to the next one until we are done
            curveSizes.append(numpy.mod(breakList[i+1] - breakList[i], nElem))
        # check the last curve outside the loop as it will end with no feature node
        curveSizes.append(numpy.mod(breakList[0] - breakList[-1], nElem))

        # copy the curveSizes for the first call
        if firstCall:
            self.nNodes = curveSizes[:]

        # print("After roll")
        # print(breakList)
        # # print(seamConn)
        # print(seamConn[breakList])
        # print(intNodes[seamConn[breakList,0]])
        # print(parList)
        # print(parListCopy)


        # print(parListCopy)
        # print(curveSizes)

        # the first feature node will have the smallest u or v value
        # then the rest should follow easily. We can just change the ordering of the intersection curve based on the first feature node. We also need to keep track of how many elements there are between each feature element. These will be our curveSizes

        # now loop over the curves between the feature nodes. We will remesh them separately to retain resolution between curve features, and just append the results since the features are already ordered

        curInd = 0
        seam = numpy.zeros((0,3))
        for i in range(nFeature):
            # just use the same number of points *2 for now
            nNewNodes = 2*self.nNodes[i]
            coor = intNodes
            barsConn = seamConn[curInd:curInd+curveSizes[i]]
            # print(barsConn)
            curInd += curveSizes[i]
            method = 'linear'
            spacing = 'linear'
            initialSpacing = 0.1
            finalSpacing = 0.1

            # print("remeshing curve %d"%i)
            # print(barsConn)
            #
            # for j in range(len(barsConn) - 1):
            #     if barsConn[j,1] != barsConn[j+1,0]:
            #         print(barsConn[j-1:j+1], j)

            # now re-sample the curve (try linear for now), to get N number of nodes on it spaced linearly
            # Call Fortran code. Remember to adjust transposes and indices
            newCoor, newBarsConn = utilitiesAPI.utilitiesapi.remesh(nNewNodes,
                                                                    coor.T,
                                                                    barsConn.T + 1,
                                                                    method,
                                                                    spacing,
                                                                    initialSpacing,
                                                                    finalSpacing)
            newCoor = newCoor.T
            newBarsConn = newBarsConn.T - 1

            # add these n -resampled nodes back to back in seam and return a copy of the array

            # we don't need the connectivity info for now? we just need the coords
            # print(newCoor)
            seam = numpy.vstack((seam, newCoor[:-1]))

        # We need to add the last node on the intersection if this is not a closed curve.

        # print(seam)
        return seam.copy()
