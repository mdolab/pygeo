# ======================================================================
#         Imports
# ======================================================================
from __future__ import print_function
import copy,time
try:
    from collections import OrderedDict
except ImportError:
    try:
        from ordereddict import OrderedDict
    except ImportError:
        print("Could not find any OrderedDict class. For 2.6 and earlier, "
              "use:\n pip install ordereddict")
import numpy
from scipy import sparse
from mpi4py import MPI
from pyspline import pySpline
from . import pyNetwork, pyBlock, geo_utils
from pygeo import DVGeometry
import pdb
import os

# directly import the interface to the fortran APIs
from pysurf.geometryEngines.TSurf.python import intersectionAPI, curveSearchAPI, utilitiesAPI, tsurf_tools, adtAPI
# generic import for all pysurf codes
import pysurf

class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a explicitly raised exception.
    """
    def __init__(self, message):
        msg = '\n+'+'-'*78+'+'+'\n' + '| DVGeometryMulti Error: '
        i = 19
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


class DVGeometryMulti(object):
    """
    A class for manipulating multiple components using multiple FFDs

    """

    def __init__(self, comm=MPI.COMM_WORLD, dh=1e-6):

        self.compNames = []
        self.comps = OrderedDict()
        self.DVGeoDict = OrderedDict()
        self.points = OrderedDict()
        self.comm = comm
        self.updated = {}
        self.dh = dh
        self.intersectComps = []

        # flag to keep track of IC jacobians
        self.ICJupdated = False

    def addComponent(self, comp, ffdFile, triMesh=None, scale=1.0):
        """
        Method to add components to the DVGeometryMulti object.
        Returns the DVGeo object for this component
        """

        # we need to create a new DVGeo object for this component
        DVGeo = DVGeometry(ffdFile)

        if triMesh is not None:
            # We also need to read the triMesh and save the points
            nodes, triConn, barsConn = self._readCGNSFile(triMesh)

            # scale the nodes
            nodes *= scale

            # add these points to the corresponding dvgeo
            DVGeo.addPointSet(nodes, 'triMesh')
        else:
            # the user has not provided a triangulated surface mesh for this file
            nodes = None
            triConn = None
            barsConn = None

        # we will need the bounding box information later on, so save this here
        xMin, xMax = DVGeo.FFD.getBounds()

        # initialize the component object
        self.comps[comp] = component(comp, DVGeo, nodes, triConn, barsConn, xMin, xMax)

        # add the name to the list
        self.compNames.append(comp)

        # also save the DVGeometry pointer in the dictionary we pass back
        self.DVGeoDict[comp] = DVGeo

        return DVGeo

    def addIntersection(self, compA, compB, dStarA=0.2, dStarB=0.2, featureCurves=[], distTol=1e-14, project=False, marchDir=1, includeCurves=False):
        """
        Method that defines intersections between components
        """

        # just initialize the intersection object
        self.intersectComps.append(CompIntersection(compA, compB, dStarA, dStarB, featureCurves, distTol, self, project, marchDir, includeCurves))

    def getDVGeoDict(self):
        # return DVGeo objects so that users can add design variables
        return self.DVGeoDict

    def finalizeDVs(self):
        """
        This function should be called after adding all DVGeoDVs
        """

        self.DV_listGlobal  = OrderedDict() # Global Design Variable List
        self.DV_listLocal = OrderedDict() # Local Design Variable List
        self.DV_listSectionLocal = OrderedDict() # Local Normal Design Variable List

        # we loop over all components and add the dv objects
        for comp in self.compNames:

            # get this DVGeo
            DVGeoComp = self.comps[comp].DVGeo

            # loop over the DVGeo's DV lists

            for k,v in DVGeoComp.DV_listGlobal.items():
                # change the key and add it to our dictionary...
                knew = comp + ':' + k
                self.DV_listGlobal[knew] = v

            for k,v in DVGeoComp.DV_listLocal.items():
                # change the key and add it to our dictionary...
                knew = comp + ':' + k
                self.DV_listLocal[knew] = v

            for k,v in DVGeoComp.DV_listSectionLocal.items():
                # change the key and add it to our dictionary...
                knew = comp + ':' + k
                self.DV_listSectionLocal[knew] = v

    def addPointSet(self, points, ptName, compNames=None, **kwargs):

        # if the user passes a list of compNames, we only use these comps.
        # we will still use all points to add the pointset, but by default,
        # the components not in this list will get 0 npoints. This is for
        # consistency, each dvgeo will have all pointsets and book keeping
        # becomes easier this way.

        # if compList is not provided, we use all components
        if compNames is None:
            compNames = self.compNames

        # before we do anything, we need to create surface ADTs
        # for which the user provided triangulated meshes
        # TODO Time these, we can do them once and keep the ADTs
        for comp in compNames:

            # check if we have a trimesh for this component
            if self.comps[comp].triMesh:

                # now we build the ADT
                # from ney's code:
                 # Set bounding box for new tree
                BBox = numpy.zeros((2, 3))
                useBBox = False

                # dummy connectivity data for quad elements since we have all tris
                quadConn = numpy.zeros((0,4))

                t0 = time.time()

                # Compute set of nodal normals by taking the average normal of all
                # elements surrounding the node. This allows the meshing algorithms,
                # for instance, to march in an average direction near kinks.
                nodal_normals = adtAPI.adtapi.adtcomputenodalnormals(self.comps[comp].nodes.T,
                                                                     self.comps[comp].triConn.T,
                                                                     quadConn.T)
                self.comps[comp].nodal_normals = nodal_normals.T

                # Create new tree (the tree itself is stored in Fortran level)
                adtAPI.adtapi.adtbuildsurfaceadt(self.comps[comp].nodes.T,
                                                 self.comps[comp].triConn.T,
                                                 quadConn.T, BBox.T, useBBox,
                                                 MPI.COMM_SELF.py2f(), comp)
                t1 = time.time()
                # if self.comm.rank == 0:
                #     print("Building surface ADT for component",comp,"took",t1-t0,'seconds')

        # create the pointset class
        self.points[ptName] = PointSet(points)

        for comp in self.compNames:
            # initialize the list for this component
            self.points[ptName].compMap[comp] = []
            self.points[ptName].compMapFlat[comp] = []

        # we now need to create the component mapping information
        for i in range(self.points[ptName].nPts):

            # initial flags
            inFFD = False
            proj = False
            projList = []

            # loop over components and check if this point is in a single BBox
            for comp in compNames:

                # check if inside
                xMin = self.comps[comp].xMin
                xMax = self.comps[comp].xMax
                if (xMin[0] < points[i,0] < xMax[0] and
                   xMin[1] < points[i,1] < xMax[1] and
                   xMin[2] < points[i,2] < xMax[2]):

                    # print('point',i,'is in comp',comp)
                    # add this component to the projection list
                    projList.append(comp)

                    # this point was not inside any other FFD before
                    if not inFFD:
                        inFFD  = True
                        inComp = comp
                    # this point was inside another FFD, so we need to project it...
                    else:
                        # set the projection flag
                        proj = True

            # project this point to components, we need to set inComp string
            if proj:

                # set a high initial distance
                dMin2 = 1e10

                # loop over the components
                for comp in compNames:
                    # check if this component is in the projList
                    if comp in projList:

                        # check if we have an ADT:
                        if self.comps[comp].triMesh:
                            # Initialize reference values (see explanation above)
                            numPts = 1
                            dist2 = numpy.ones(numPts)*1e10
                            xyzProj = numpy.zeros((numPts, 3))
                            normProjNotNorm = numpy.zeros((numPts, 3))

                            # Call projection function
                            _, _, _, _ = adtAPI.adtapi.adtmindistancesearch(points[i].T, comp,
                                                                            dist2, xyzProj.T,
                                                                            self.comps[comp].nodal_normals.T,
                                                                            normProjNotNorm.T)
                            # print('Distance of point',points[i],'to comp',comp,'is',numpy.sqrt(dist2))
                            # if this is closer than the previous min, take this comp
                            if dist2 < dMin2:
                                dMin2 = dist2[0]
                                inComp = comp

                        else:
                            raise Error('The point at \n(x, y, z) = (%.3f, %.3f, %.3f) \nin pointset %s is inside multiple FFDs but a triangulated mesh for component %s is not provided to determine which component owns this point.'%(points[i,0], points[i,1], points[i,1], ptName, comp))


            # this point was inside at least one FFD. If it was inside multiple,
            # we projected it before to figure out which component it should belong to
            if inFFD:
                # we can add the point index to the list of points inComp owns
                self.points[ptName].compMap[inComp].append(i)

                # also create a flattened version of the compMap
                for j in range(3):
                    self.points[ptName].compMapFlat[inComp].append(3*i + j)

            # this point is outside any FFD...
            else:
                raise Error('The point at (x, y, z) = (%.3f, %.3f, %.3f) in pointset %s is not inside any FFDs'%(points[i,0], points[i,1], points[i,1], ptName))

        # using the mapping array, add the pointsets to respective DVGeo objects
        for comp in self.compNames:
            compMap = self.points[ptName].compMap[comp]
            # print(comp,compMap)
            self.comps[comp].DVGeo.addPointSet(points[compMap], ptName)

        # loop over the intersections and add pointsets
        for IC in self.intersectComps:
            IC.addPointSet(points, ptName, self.points[ptName].compMap)

        # finally, we can deallocate the ADTs
        for comp in compNames:
            if self.comps[comp].triMesh:
                adtAPI.adtapi.adtdeallocateadts(comp)
                # print('Deallocated ADT for component',comp)

        # mark this pointset as up to date
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

        # first get the list of DVs from each comp so we can ignore extra entries
        for comp in self.compNames:
            self.comps[comp].dvDict = self.comps[comp].DVGeo.getValues()

        # loop over all dvs we get as the input
        for k,v in dvDict.items():
            # we only set dvgeomulti DVs. Then k should always have a : in it
            if ':' in k:
                # get the component name
                comp, dvName = k.split(':',1)

                # now check if this comp has this dv
                if dvName in self.comps[comp].dvDict:
                    # set the value
                    self.comps[comp].dvDict[dvName] = v

        # loop over the components and set the values
        for comp in self.compNames:
            self.comps[comp].DVGeo.setDesignVars(self.comps[comp].dvDict)

        # We need to give the updated coordinates to each of the
        # intersectComps (if we have any) so they can update the new
        # intersection curve
        for IC in self.intersectComps:
            IC.setSurface(self.comm)

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

        # also set IC Jacobians as out of date
        self.ICJupdated = False

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

        dvDict = {}
        # we need to loop over each DVGeo object and get the DVs
        for comp in self.compNames:
            dvDictComp = self.comps[comp].DVGeo.getValues()
            # we need to loop over these DVs.
            for k,v in dvDictComp.items():
                # We will add the name of the comp and a : to the full DV name
                dvName = '%s:%s'%(comp, k)
                dvDict[dvName] = v

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

        # get the new points
        newPts = numpy.zeros((self.points[ptSetName].nPts, 3))

        # we first need to update all points with their respective DVGeo objects
        for comp in self.compNames:
            ptsComp = self.comps[comp].DVGeo.update(ptSetName)

            # now save this info with the pointset mapping
            ptMap = self.points[ptSetName].compMap[comp]
            newPts[ptMap] = ptsComp

        # get the delta
        delta = newPts - self.points[ptSetName].points

        # then apply the intersection treatment
        for IC in self.intersectComps:
            delta = IC.update(ptSetName, delta)

        # now we are ready to take the delta which may be modified by the intersections
        newPts = self.points[ptSetName].points + delta

        # now, project the points that were warped back onto the trimesh
        for IC in self.intersectComps:
            if IC.projectFlag:
                newPts = IC.project(ptSetName, newPts, self.points[ptSetName].compMap)

        # set the pointset up to date
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
        # loop over components and sum number of DVs
        nDV = 0
        for comp in self.compNames:
            nDV += self.comps[comp].DVGeo.getNDV()
        return nDV

    def getVarNames(self):
        """
        Return a list of the design variable names. This is typically
        used when specifying a wrt= argument for pyOptSparse.

        Examples
        --------
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        dvNames = []
        # create a list of DVs from each comp
        for comp in self.compNames:
            # first get the list of DVs from this component
            varNames = self.comps[comp].DVGeo.getVarNames()

            for var in varNames:
                # then add the component's name to the DV name
                dvName = '%s:%s'%(comp, var)

                # finally append to the list
                dvNames.append(dvName)

        return dvNames

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        """
        This function computes sensitivty information.

        Specificly, it computes the following:
        :math:`\\frac{dX_{pt}}{dX_{DV}}^T \\frac{dI}{d_{pt}}`

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

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.


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

        # compute intersection jacobians if out of date
        if not self.ICJupdated and self.intersectComps:
            # this needs to be called with the full comm for the FD calcs to make any sense to do in parallel
            self._computeICJacobian()

        # compute the total jacobian for this pointset
        self._computeTotalJacobian(ptSetName)

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = numpy.array([dIdpt])
        N = dIdpt.shape[0]

        # do the transpose multiplication

        # get the pointset
        ptSet = self.points[ptSetName]

        # number of design variables
        nDV = ptSet.jac.shape[1]

        # We should keep track of the intersections that this pointset is close to. There is no point in including the intersections far from this pointset in the sensitivity calc as the derivative seeds will be just zeros there.
        ptSetICs = []
        nSeams  = 0
        for IC in self.intersectComps:
            # This checks if we have any entries in the affected indices on this point set with this intersection
            if IC.points[ptSetName][1]:
                # this pointset is affected by this intersection. save this info.
                ptSetICs.append(IC)
                # this keeps the cumulative number of nodes on the seams this point set is effected by
                nSeams += len(IC.seam)

        # allocate the matrix
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
        jac = self.points[ptSetName].jac.copy()
        for IC in ptSetICs:
            jac = numpy.vstack((jac, IC.jac))

        # Remember the jacobian contains the surface poitns *and* the
        # the seam nodes. dIdx_compact is the final derivative for the
        # points this proc owns.
        dIdxT_local = jac.T.dot(tmp)
        dIdx_local = dIdxT_local.T

        if comm: # If we have a comm, globaly reduce with sum
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # use respective DVGeo's convert to dict functionality
        dIdxDict = OrderedDict()
        dvOffset = 0
        for comp in self.compNames:
            DVGeo = self.comps[comp].DVGeo
            nDVComp = DVGeo.getNDV()

            # this part of the sensitivity matrix is owned by this dvgeo
            dIdxComp = DVGeo.convertSensitivityToDict(dIdx[:,dvOffset:dvOffset+nDVComp])

            # add the component names in front of the dictionary keys
            for k,v in dIdxComp.items():
                dvName = '%s:%s'%(comp,k)
                dIdxDict[dvName] = v

            # also increment the offset
            dvOffset += nDVComp

        return dIdxDict

    def addVariablesPyOpt(self, optProb, globalVars=True, localVars=True,
                          sectionlocalVars=True, ignoreVars=None, freezeVars=None, comps=None):
        """
        Add the current set of variables to the optProb object.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which variables are added

        globalVars : bool
            Flag specifying whether global variables are to be added

        localVars : bool
            Flag specifying whether local variables are to be added

        ignoreVars : list of strings
            List of design variables the user DOESN'T want to use
            as optimization variables.

        freezeVars : list of string
            List of design variables the user WANTS to add as optimization
            variables, but to have the lower and upper bounds set at the current
            variable. This effectively eliminates the variable, but it the variable
            is still part of the optimization.

        comps : list of components we want to add DVs of. If no list is passed,
            we will add all dvs of all components
        """

        # if no list was provided, we use all components
        if comps is None:
            comps = self.compNames

        # we can simply loop over all DV objects and call their respective
        # addVariablesPyOpt function with the correct prefix.
        for comp in comps:
            self.comps[comp].DVGeo.addVariablesPyOpt(optProb,
                                                     globalVars=globalVars,
                                                     localVars=localVars,
                                                     sectionlocalVars=sectionlocalVars,
                                                     ignoreVars=ignoreVars,
                                                     freezeVars=freezeVars,
                                                     prefix=comp+':')

    def getLocalIndex(self, iVol, comp):
        """ Return the local index mapping that points to the global
        coefficient list for a given volume"""
        # call this on the respective DVGeo
        DVGeo = self.comps[comp].DVGeo
        return DVGeo.FFD.topo.lIndex[iVol].copy()

# ----------------------------------------------------------------------
#        THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER
# ----------------------------------------------------------------------

    def _readCGNSFile(self, filename):
        # this function reads the unstructured CGNS grid in filename and returns
        # node coordinates and element connectivities.
        # Here, only the root proc reads the cgns file, broadcasts node and connectivity info.

        # create the featurecurve dictionary
        curveConn = OrderedDict()

        # only root proc reads the file
        if self.comm.rank == 0:
            print('Reading file %s'%filename)
            # use the default routine in tsurftools
            t1 = time.time()
            nodes, sectionDict = tsurf_tools.getCGNSsections(filename, comm = MPI.COMM_SELF)
            t2 = time.time()
            print('Finished reading the cgns file')
            print('Reading the cgns file took', (t2-t1))

            triConn = numpy.zeros((0,3), dtype = numpy.int8)
            barsConn = {}
            # print('Part names in triangulated cgns file for %s'%filename)
            for part in sectionDict:
                # print(part)
                if 'triaConnF' in sectionDict[part].keys():
                    #this is a surface, read the tri connectivities
                    triConn = numpy.vstack((triConn, sectionDict[part]['triaConnF']))

                if 'barsConn' in sectionDict[part].keys():
                    # this is a curve, save the curve connectivity
                    barsConn[part.lower()] = sectionDict[part]['barsConn']
        else:
            # create these to recieve the data
            nodes = None
            triConn = None
            barsConn = None

        # each proc gets the nodes and connectivities
        # CHECK if this should be bcast or Bcast...
        nodes = self.comm.bcast(nodes, root=0)
        triConn = self.comm.bcast(triConn, root=0)
        barsConn = self.comm.bcast(barsConn, root=0)

        return nodes, triConn, barsConn

    def _computeTotalJacobian(self, ptSetName):
        """
        This routine computes the total jacobian. It takes the jacobians
        from respective DVGeo objects and also computes the jacobians for
        the intersection seams. We then use this information in the
        totalSensitivity function.
        """

        # number of design variables
        nDV = self.getNDV()

        # allocate space for the jacobian.
        jac = numpy.zeros((self.points[ptSetName].nPts*3, nDV))

        # ptset
        ptSet = self.points[ptSetName]

        dvOffset = 0
        # we need to call computeTotalJacobian from all comps and get the jacobians for this pointset
        for comp in self.compNames:

            # number of design variables
            nDVComp = self.comps[comp].DVGeo.getNDV()

            # call the function to compute the total jacobian
            self.comps[comp].DVGeo.computeTotalJacobian(ptSetName)

            if self.comps[comp].DVGeo.JT[ptSetName] is not None:
                # we convert to dense storage.
                # this is probably not a good way to do this...
                compJ = self.comps[comp].DVGeo.JT[ptSetName].todense().T

                # loop over the entries and add one by one....
                # for i in range(len(ptSet.compMapFlat[comp])):
                #     jac[ptSet.compMapFlat[comp][i], dvOffset:dvOffset+nDVComp] = compJ[i,:]

                # or, do it (kinda) vectorized!
                jac[ptSet.compMapFlat[comp], dvOffset:dvOffset+nDVComp] = compJ[:,:]

            # increment the offset
            dvOffset += nDVComp

        # now we can save this jacobian in the pointset
        ptSet.jac = jac

    def _computeICJacobian(self):

        # loop over design variables and compute jacobians for all ICs

        # counts
        nDV = self.getNDV()
        nproc = self.comm.size
        rank  = self.comm.rank

        # save the reference seams
        for IC in self.intersectComps:
            IC.seamRef = IC.seam.flatten()
            IC.jac = numpy.zeros((len(IC.seamRef), nDV))

        # We need to evaluate all the points on respective procs for FD computations

        # determine how many DVs this proc will perturb.
        n = 0
        for iDV in range(nDV):
            # I have to do this one.
            if iDV % nproc == rank:
                n += 1

        # perturb the DVs on different procs and compute the new point coordinates.
        reqs = []
        for iDV in range(nDV):
            # I have to do this one.
            if iDV % nproc == rank:

                # TODO Use separate step sizes for different DVs?
                dh =  self.dh

                # Perturb the DV
                dvSave = self._getIthDV(iDV)
                self._setIthDV(iDV, dvSave+dh)

                # Do any required intersections:
                for IC in self.intersectComps:
                    IC.setSurface(MPI.COMM_SELF)
                    IC.jac[:,iDV] = (IC.seam.flatten() - IC.seamRef) / dh

                # Reset the DV
                self._setIthDV(iDV, dvSave)

        # Restore the seams
        for IC in self.intersectComps:
            IC.seam = IC.seamRef.reshape(len(IC.seam), 3)

        # loop over the DVs and scatter the perturbed points to original procs
        for iDV in range(nDV):

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
                # IC.jac[:,i] = self.comm.bcast(IC.jac[:,i], root=i%nproc)

        # set the flag
        self.ICJupdated = True

    def _setIthDV(self, iDV, val):
        # this function sets the design variable. The order is important, and we count every DV.

        nDVCum = 0
        # get the number of DVs from different DVGeos to figure out which DVGeo owns this DV
        for comp in self.compNames:
            # get the DVGeo object
            DVGeo = self.comps[comp].DVGeo
            # get the number of DVs on this DVGeo
            nDVComp = DVGeo.getNDV()
            # increment the cumulative number of DVs
            nDVCum += nDVComp

            # if we went past the index of the DV we are interested in
            if nDVCum > iDV:
                # this DVGeo owns the DV we want to set.
                xDV = DVGeo.getValues()

                # take back the global counter
                nDVCum -= nDVComp

                # loop over the DVs owned by this geo
                for k,v in xDV.items():
                    nDVKey = len(v)

                    # add the number of DVs for this DV key
                    nDVCum += nDVKey

                    # check if we went past the value we want to set
                    if nDVCum > iDV:
                        # take back the counter
                        nDVCum -= nDVKey

                        # this was an array...
                        if nDVKey > 1:
                            for i in range(nDVKey):
                                nDVCum += 1
                                if nDVCum > iDV:
                                    # finally...
                                    xDV[k][i] = val
                                    DVGeo.setDesignVars(xDV)
                                    return

                        # this is the value we want to set!
                        else:
                            xDV[k] = val
                            DVGeo.setDesignVars(xDV)
                            return

    def _getIthDV(self, iDV):
        # this function sets the design variable. The order is important, and we count every DV.

        nDVCum = 0
        # get the number of DVs from different DVGeos to figure out which DVGeo owns this DV
        for comp in self.compNames:
            # get the DVGeo object
            DVGeo = self.comps[comp].DVGeo
            # get the number of DVs on this DVGeo
            nDVComp = DVGeo.getNDV()
            # increment the cumulative number of DVs
            nDVCum += nDVComp

            # if we went past the index of the DV we are interested in
            if nDVCum > iDV:
                # this DVGeo owns the DV we want to set.
                xDV = DVGeo.getValues()

                # take back the global counter
                nDVCum -= nDVComp

                # loop over the DVs owned by this geo
                for k,v in xDV.items():
                    nDVKey = len(v)

                    # add the number of DVs for this DV key
                    nDVCum += nDVKey

                    # check if we went past the value we want to set
                    if nDVCum > iDV:
                        # take back the counter
                        nDVCum -= nDVKey

                        # this was an array...
                        if nDVKey > 1:
                            for i in range(nDVKey):
                                nDVCum += 1
                                if nDVCum > iDV:
                                    # finally...
                                    return xDV[k][i]

                        # this is the value we want to get!
                        else:
                            # assume single DVs come in arrays...
                            return xDV[k].copy()

class component(object):
    def __init__(self, name, DVGeo, nodes, triConn, barsConn, xMin, xMax):

        # save the info
        self.name = name
        self.DVGeo = DVGeo
        self.nodes = nodes
        self.triConn = triConn
        self.barsConn = barsConn
        self.xMin = xMin
        self.xMax = xMax

        # also a dictionary for DV names
        self.dvDict = {}

        # set a flag for triangulated meshes
        if nodes is None:
            self.triMesh = False
        else:
            self.triMesh = True

    def updateTriMesh(self):
        # update the triangulated surface mesh
        self.nodes = self.DVGeo.update('triMesh')

class PointSet(object):
    def __init__(self, points):
        self.points = points
        self.nPts = len(self.points)
        self.compMap = OrderedDict()
        self.compMapFlat = OrderedDict()

class CompIntersection(object):
    def __init__(self, compA, compB, dStarA, dStarB, featureCurves, distTol, DVGeo, project, marchDir, includeCurves):
        '''Class to store information required for an intersection.
        Here, we use some fortran code from pySurf.

        Input
        -----
        compA: ID of the first component
        compB: ID of the second component

        dStar, real : Radius over which to attenuate the deformation

        Internally we will store the indices and the weights of the
        points that this intersection will have to modify. In general,
        all this code is not super efficient since it's all python
        '''

        # same communicator with DVGeo
        self.comm = DVGeo.comm


        # names of compA and compB must be provided
        self.compA = DVGeo.comps[compA]
        self.compB = DVGeo.comps[compB]

        self.dStarA = dStarA
        self.dStarB = dStarB
        # self.halfdStar = self.dStar/2.0
        self.points = OrderedDict()

        # feature curve names
        self.featureCurveNames = featureCurves
        for i in range(len(self.featureCurveNames)):
            self.featureCurveNames[i] = self.featureCurveNames[i].lower()

        self.distTol = distTol

        # flag to determine if we want to project nodes after intersection treatment
        self.projectFlag = project

        # get the marching direction for feature curves so that we know if we need to flip them..
        self.marchDir = marchDir

        # flag to include feature curves in ID-warping
        self.incCurves = includeCurves

        # only the node coordinates will be modified for the intersection calculations because we have calculated and saved all the connectivity information
        if self.comm.rank == 0:
            print('Computing initial intersection between %s and %s'%(compA, compB))
        self.seam0 = self._getIntersectionSeam(self.comm , firstCall = True)
        self.seam = self.seam0.copy()

    def setSurface(self, comm):
        """ This set the new udpated surface on which we need to comptue the new intersection curve"""

        # get the updated surface coordinates
        self._getUpdatedCoords()

        self.seam = self._getIntersectionSeam(comm)

    def addPointSet(self, pts, ptSetName, compMap):

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
            # figure out which component this point is mapped to
            if i in compMap[self.compA.name]:
                # component A owns this
                dStar = self.dStarA
            else:
                # comp B owns this point
                dStar = self.dStarB

            # then get the halfdStar for that component
            halfdStar = dStar / 2.0

            if d[i] < dStar:

                # Compute the factor
                if d[i] < halfdStar:
                    factor = .5*(d[i]/halfdStar)**3
                else:
                    factor = .5*(2-((dStar - d[i])/halfdStar)**3)

                # Save the index and factor
                indices.append(i)
                factors.append(factor)

        # # Print the number of points that get associated with the intersection.
        # nPointGlobal = self.comm.allreduce(len(factors), op=MPI.SUM)
        # if self.comm.rank == 0:
        #     intName = vsp.GetContainerName(self.compA)+'_'+vsp.GetContainerName(self.compB)
        #     print('DVGEO VSP:\n%d points associated with intersection %s'%(nPointGlobal, intName))

        # Save the affected indices and the factor in the little dictionary
        self.points[ptSetName] = [pts.copy(), indices, factors]

    def update(self, ptSetName, delta):

        """Update the delta in ptSetName with our correction. The delta need
        to be supplied as we will be changing it and returning them
        """
        pts     = self.points[ptSetName][0]
        indices = self.points[ptSetName][1]
        factors = self.points[ptSetName][2]
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
        pts     = self.points[ptSetName][0]
        indices = self.points[ptSetName][1]
        factors = self.points[ptSetName][2]

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

        pts     = self.points[ptSetName][0]
        indices = self.points[ptSetName][1]
        factors = self.points[ptSetName][2]

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

    def project(self, ptSetName, newPts, compMap):
        # we need to build ADTs for both components if we have any components that lie on either

        flagA = False
        flagB = False

        indices = self.points[ptSetName][1]

        # maybe we can do this vectorized
        for i in range(len(indices)):

            # index of the point in pointset
            ind = indices[i]

            # check compA
            if ind in compMap[self.compA.name]:
                flagA = True

            # check compB
            if ind in compMap[self.compB.name]:
                flagB = True

            # we can terminate early if we hit both surfaces
            if flagA and flagB:
                break

        # now we know which surfaces we will use for projections.

        # build the ADT for compA
        if flagA:
            self._buildSurfaceADT(self.compA)

        # build the ADT for compB
        if flagB:
            self._buildSurfaceADT(self.compB)

        # loop over the affected points and project
        for i in range(len(indices)):
            ind = indices[i]

            # project this to compA
            if ind in compMap[self.compA.name]:
                newPoint = self._projectToSurface(newPts[ind], self.compA)
            # project this to compB
            else:
                newPoint = self._projectToSurface(newPts[ind], self.compB)

            # after we do the projections, we can get the new points and update in the newPts array.
            newPts[ind] = newPoint

        # finally, we can deallocate the ADTs
        if flagA:
            adtAPI.adtapi.adtdeallocateadts(self.compA.name)
        if flagB:
            adtAPI.adtapi.adtdeallocateadts(self.compB.name)

        return newPts

    def _buildSurfaceADT(self, comp):
        # we build an ADT using this component
        # from ney's code:
        # Set bounding box for new tree
        BBox = numpy.zeros((2, 3))
        useBBox = False

        # dummy connectivity data for quad elements since we have all tris
        quadConn = numpy.zeros((0,4))

        t0 = time.time()

        # Compute set of nodal normals by taking the average normal of all
        # elements surrounding the node. This allows the meshing algorithms,
        # for instance, to march in an average direction near kinks.
        nodal_normals = adtAPI.adtapi.adtcomputenodalnormals(comp.nodes.T,
                                                             comp.triConn.T,
                                                             quadConn.T)
        comp.nodal_normals = nodal_normals.T

        # Create new tree (the tree itself is stored in Fortran level)
        adtAPI.adtapi.adtbuildsurfaceadt(comp.nodes.T,
                                         comp.triConn.T,
                                         quadConn.T, BBox.T, useBBox,
                                         MPI.COMM_SELF.py2f(), comp.name)

    def _projectToSurface(self, point, comp):
        numPts = 1
        dist2 = numpy.ones(numPts)*1e10
        xyzProj = numpy.zeros((numPts, 3))
        normProjNotNorm = numpy.zeros((numPts, 3))

        # Call projection function
        _, _, _, _ = adtAPI.adtapi.adtmindistancesearch(point.T, comp.name,
                                                        dist2, xyzProj.T,
                                                        comp.nodal_normals.T,
                                                        normProjNotNorm.T)

        # return the projected point
        return xyzProj

    def _getUpdatedCoords(self):
        # this code returns the updated coordinates

        # first comp a
        self.compA.updateTriMesh()

        # then comp b
        self.compB.updateTriMesh()

        return

    def _getIntersectionSeam(self, comm, firstCall = False):
        # we can parallelize here. each proc gets one intersection, but needs re-structuring of some of the code.

        # this function computes the intersection curve, cleans up the data and splits the curve based on features or curves specified by the user.

        # Call Ney's code with the quad information.
        # only root does this as this is serial code
        if comm.rank == 0:
            dummyConn = numpy.zeros((0,4))
            # compute the intersection curve, in the first step we just get the array sizes to hide allocatable arrays from python
            arraySizes = intersectionAPI.intersectionapi.computeintersection(self.compA.nodes.T,
                                                                             self.compA.triConn.T,
                                                                             dummyConn.T,
                                                                             self.compB.nodes.T,
                                                                             self.compB.triConn.T,
                                                                             dummyConn.T,
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
                raise Error('DVGeometryMulti Error: The components %s and %s do not intersect.'%(self.compA.name, self.compB.name))

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

        newConn, newMap = tsurf_tools.FEsort(barsConn.tolist())

        # newConn might have multiple intersection curves
        if len(newConn) == 1:
            # we have a single intersection curve, just take this.
            seamConn = newConn[0].copy()
        else:
            raise Error('more than one intersection curve between comps %s and %s'%(self.compA.name, self.compB.name))
        # we might have two of the same curve on both sides of the symmetry plane, if so, get the one on the positive side
        # elif len(newConn) == 2:
        #     # check which curve is on the positive side
        #     # fix the 2 to get the correct option for symmetry plane normal direction
        #     if intNodes[newConn[0][0][0], self.symDir] > 0.0 and intNodes[newConn[1][0][0], self.symDir] < 0.0:
        #         # the first array is what we need
        #         seamConn = newConn[0].copy()
        #     elif intNodes[newConn[0][0][0], self.symDir] < 0.0 and intNodes[newConn[1][0][0], self.symDir] > 0.0:
        #         # the second array is what we need
        #         seamConn = newConn[1].copy()
        #     else:
        #         # throw an error. each component pair should have one intersection on one side of the symmetry plane
        #         raise Error("Error at DVGeometryVSP. The intersection between two components should have a single intersection curve on each side of the symmetry plane.")
        # print(seamConn)
        # print(intNodes[seamConn][:,0])

        # Get the number of elements
        nElem = seamConn.shape[0]

        # now that we have a continuous, ordered seam connectivity in seamConn, we can try to detect features

        # we need to track the nodes that are closest to the supplied feature curves
        breakList = []
        curveBeg = {}
        curveBegCoor = {}
        # loop over the feature curves
        for curveName in self.featureCurveNames:
            # if this is the first call, we probably need to reorder the connectivity info for the feature curves
            if firstCall:
                newConn, newMap = tsurf_tools.FEsort(self.compB.barsConn[curveName].tolist())

                if len(newConn) > 1:
                    raise Error('the curve %s generated more than one curve with FESort'%curveName)
                newConn = newConn[0]

                # we may also need to flip the curve, just do it here.
                nodesB = self.compB.nodes
                if nodesB[newConn[0][0]][self.marchDir] > nodesB[newConn[0][1]][self.marchDir]:
                    # flip on both axes
                    newConn = numpy.flip(newConn, axis=0)
                    newConn = numpy.flip(newConn, axis=1)

                self.compB.barsConn[curveName] = newConn

            curveConn = self.compB.barsConn[curveName]

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
                                                                        self.compB.nodes.T,
                                                                        curveConn.T + 1,
                                                                        xyzProj.T,
                                                                        tanProj.T,
                                                                        dist2,
                                                                        elemIDs)

                # Adjust indices back to Python standards
                elemIDs[:] = elemIDs - 1

            # now, find the index of the smallest distance
            # print('index with smallest dist' , numpy.argmin(dist2))
            breakList.append(numpy.argmin(dist2))

            # also get which element is the closest to the feature point
            curveBeg[curveName] = elemIDs[numpy.argmin(dist2)]

            # get which point on this element we projected to.
            curveBegCoor[curveName] = xyzProj[numpy.argmin(dist2)]

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

        # first get an ordered list of the feature points
        # this is just our breakList "list"
        featurePoints = intNodes[seamConn[breakList,0]]
        # print(featurePoints)

        # # if we have a larger number of surfaces, throw an error
        # if self.nSurfB > 1:
        #     raise Error('I have more than one surface defined for the fully intersected geometry in VSP.')

        # project these points to the VSP geometry
        # we will use compB for all of these operations
        # parList = []
        # dMax = 1e-16

        # # initialize one 3dvec for projections
        # pnt = vsp.vec3d()

        # for i in range(len(featurePoints)):
        #     # set the coordinates of the point object
        #     pnt.set_x(featurePoints[i,0] * self.meshScale)
        #     pnt.set_y(featurePoints[i,1] * self.meshScale)
        #     pnt.set_z(featurePoints[i,2] * self.meshScale)
        #     d, surf, uout, vout = vsp.ProjPnt01I(self.compB, pnt)
        #     par = [uout, vout]

        #     # print(d, surf, uout, vout, featurePoints[i])

        #     parList.append(par[self.dir])

        #     dMax = max(d, dMax)

        # print('Maximum distance between the intersection feature nodes and the vsp surface is %1.15f'%dMax)

        # print(parList)

        # copy parlist and order it
        # parListCopy = parList[:]

        # parListCopy.sort()

        # flip
        # we want breakList to be in increasing order...
        ii = 0
        for i in range(nFeature):
            # print(i,breakList[i], breakList[numpy.mod(i+1, nFeature)])
            # we loop over the breaklist elements and check if the element index is going up or down
            if breakList[i] < breakList[numpy.mod(i+1, nFeature)]:
                ii += 1

        # print('increase index',ii)
        # now check if we need to flip the curve
        if ii == 1: # we need at least 2 features where the element number increases...
            # print('I am flipping the intersection curve')
            # we need to reverse the order of our feature curves
            # and we will flip the elements too so keep track of this change
            # print(breakList)
            breakList = numpy.mod(seamConn.shape[0] - numpy.array(breakList), seamConn.shape[0])
            # TODO we had a bug in this line
            # breakList = numpy.mod(seamConn.shape[0] - numpy.flip(breakList, axis=0), seamConn.shape[0])
            # print(breakList)

            # and we need to invert the curves themselves
            seamConn = numpy.flip(seamConn, axis = 0)
            seamConn = numpy.flip(seamConn, axis = 1)

            # # and parameter list
            # parList.reverse()

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

        # roll so that the first breakList entry is the first node
        seamConn = numpy.roll(seamConn, -breakList[0], axis = 0)
        # print("rolled by %d"%-breakList[parList.index(parListCopy[0])])
        # print(seamConn)

        # also adjust the feature indices
        # print(breakList)
        breakList = numpy.mod(breakList - breakList[0], nElem)

        # do we need this?
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
            # print('remeshing from feature',i)
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

        # we need to re-mesh feature curves if the user wants...
        if self.incCurves:

            # we need to set up some variables
            if firstCall:
                self.nNodeFeature = {}
                self.distFeature = {}

            remeshedCurves = numpy.zeros((0,3))

            # loop over each curve, figure out what nodes get re-meshed, re-mesh, and append to seam...
            for curveName in self.featureCurveNames:

                # connectivity for this curve.
                curveConn = self.compB.barsConn[curveName]

                # we already have the element that is closest to the intersection
                elemBeg = curveBeg[curveName]

                # now lets split this element so that we get a better initial point...

                # save the original coordinate of the first point
                ptBegSave = self.compB.nodes[curveConn[elemBeg,0]]

                # and replace this with the starting point we want
                self.compB.nodes[curveConn[elemBeg,0]] = curveBegCoor[curveName]

                # print(curveName)
                # print(elemBeg)
                # print(curveConn)
                # print('curve begins at',self.compB.nodes[curveConn[0]])
                # print('curve ends   at',self.compB.nodes[curveConn[-1]])
                # print(curveConn[elemBeg])
                # print(numpy.array(curveConn))
                # print(self.compB.nodes[curveConn[elemBeg]])

                # compute the element lengths starting from elemBeg
                firstNodes  = self.compB.nodes[curveConn[elemBeg:, 0]]
                secondNodes = self.compB.nodes[curveConn[elemBeg:, 1]]
                diff = secondNodes - firstNodes
                dist2 = diff[:,0]**2 + diff[:,1]**2 + diff[:,2]**2
                elemDist = numpy.sqrt(dist2)
                # print(elemDist)

                # get the cumulative distance
                cumDist = numpy.cumsum(elemDist)
                # print(cumDist)

                if firstCall:

                    # compute the distances from curve nodes to intersection seam
                    curvePts = self.compB.nodes[curveConn[elemBeg:,0]]
                    # print(curvePts)

                    # Get number of points
                    nPoints = len(curvePts)

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
                        curveMask = curveSearchAPI.curvesearchapi.mindistancecurve(curvePts.T,
                                                                                self.nodes0.T,
                                                                                self.conn0.T + 1,
                                                                                xyzProj.T,
                                                                                tanProj.T,
                                                                                dist2,
                                                                                elemIDs)

                    dNodes = numpy.sqrt(dist2)
                    # print(dNodes)

                    # number of elements to use, subtract one to get the correct element count
                    nElem = (numpy.abs(dNodes - self.dStarB)).argmin() - 1

                    # we want to be one after the actual distance, so correct if needed
                    if dNodes[nElem] < self.dStarB:
                        nElem += 1

                    elemEnd = elemBeg + nElem

                    # print(nElem)
                    # print(dNodes[:nElem+1])

                    # print(curveConn[elemEnd])
                    # print(self.compB.nodes[curveConn[elemBeg:elemEnd+1]] )

                    # get the total curve distance from elemBeg to this element.
                    distCurve = cumDist[nElem]
                    # print(distCurve)

                    # save this distance as the remesh distance
                    self.distFeature[curveName] = distCurve

                    # also save how many nodes we have, we want 2 times this when re-meshing
                    self.nNodeFeature[curveName] = nElem

                else:
                    # figure out how many elements we need to go in this direction
                    elemEnd = (numpy.abs(cumDist - self.distFeature[curveName])).argmin()+elemBeg

                    # print('beg')
                    # print(elemBeg)
                    # print(curveConn[elemBeg])
                    # print(self.compB.nodes[curveConn[elemBeg]])

                    # print('end')
                    # print(elemEnd)
                    # print(curveConn[elemEnd])
                    # print(self.compB.nodes[curveConn[elemEnd]])

                # now, we need to modify the last node to finish exactly where we want it...
                # dError = cumDist[nElem] - self.dStarB
                # print(distCurve)
                # print(cumDist)
                # print(cumDist[nElem])
                # print(dError)

                # get the new connectivity data between the initial and final elements
                curveConnTrim = curveConn[elemBeg:elemEnd+1]

                # print(curveName)

                # print(curveConnTrim)
                # print(elemBeg, elemEnd)

                # remesh the new connectivity curve, using nNode*2 times nodes
                nNewNodes = 20*self.nNodeFeature[curveName]
                coor = self.compB.nodes
                barsConn = curveConnTrim
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

                # print('newcor',newCoor)

                # append this new curve to the featureCurve data
                remeshedCurves = numpy.vstack((remeshedCurves, newCoor))

                # finally, put the modified initial and final points back in place.
                self.compB.nodes[curveConn[elemBeg,0]] = ptBegSave

            # now we are done going over curves,
            # so we can append all the new curves to the "seam",
            # which now contains the intersection, and re-meshed feature curves
            seam = numpy.vstack((seam, remeshedCurves))
            # quit()

        # print(seam)
        return seam.copy()
