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

    def __init__(self, comps, FFDFiles, triMeshFiles, intersectedComps=[], comm=MPI.COMM_WORLD):

        t0 = time.time()

        self.compNames = comps
        self.comps = OrderedDict()
        self.DVGeoDict = OrderedDict()
        self.ptSets = OrderedDict()
        self.comm = comm
        self.updated = {}

        for (comp, filename, triMesh) in zip(comps, FFDFiles, triMeshFiles):
            # we need to create a new DVGeo object for this component
            DVGeo = DVGeometry(filename)

            if triMesh is not None:
                # We also need to read the triMesh and save the points
                nodes, triConn, barsConn = self._readCGNSFile(triMesh)

                # add these points to the corresponding dvgeo
                DVGeo.addPointSet(nodes, 'triNodes')
            else:
                # the user has not provided a triangulated surface mesh for this file
                nodes = None
                triConn = None
                barsConn = None

            # we will need the bounding box information later on, so save this here
            xMin, xMax = DVGeo.FFD.getBounds()

            # initialize the component object
            self.comps[comp] = component(comp, DVGeo, nodes, triConn, barsConn, xMin, xMax)

            # also save the DVGeometry pointer in the dictionary we pass back
            self.DVGeoDict[comp] = DVGeo

        # initialize intersections
        if comm.rank == 0:
            print("Computing intersections")
        self.intersectComps = []
        for i in range(len(intersectedComps)):
            # we will just initialize the intersections by passing in the dictionary
            self.intersectComps.append(CompIntersection(intersectedComps[i], self))

        if comm.rank == 0:
            t1 = time.time()
            print("Initialized DVGeometryMulti in",(t1-t0),"seconds.")

    def getDVGeoDict(self):
        # return DVGeo objects so that users can add design variables
        return self.DVGeoDict

    def addPointSet(self, points, ptName, **kwargs):

        # before we do anything, we need to create surface ADTs
        # for which the user provided triangulated meshes
        # TODO Time these, we can do them once and keep the ADTs
        for comp in self.compNames:

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
                                                 self.comm.py2f(), comp)
                t1 = time.time()
                # if self.comm.rank == 0:
                #     print("Building surface ADT for component",comp,"took",t1-t0,'seconds')

        # create the pointset class
        self.ptSets[ptName] = PointSet(points)

        for comp in self.compNames:
            # initialize the list for this component
            self.ptSets[ptName].compMap[comp] = []

        # we now need to create the component mapping information
        for i in range(self.ptSets[ptName].nPts):

            # initial flags
            inFFD = False
            proj = False
            projList = []

            # loop over components and check if this point is in a single BBox
            for comp in self.compNames:

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
                for comp in self.compNames:
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
                self.ptSets[ptName].compMap[inComp].append(i)

            # this point is outside any FFD...
            else:
                raise Error('The point at (x, y, z) = (%.3f, %.3f, %.3f) in pointset %s is not inside any FFDs'%(points[i,0], points[i,1], points[i,1], ptName))

        # using the mapping array, add the pointsets to respective DVGeo objects
        for comp in self.compNames:
            compMap = self.ptSets[ptName].compMap[comp]
            # print(comp,compMap)
            self.comps[comp].DVGeo.addPointSet(points[compMap], ptName)

        # loop over the intersections and add pointsets
        for IC in self.intersectComps:
            IC.addPointSet(points, ptName)

        # finally, we can deallocate the ADTs
        for comp in self.compNames:
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
            # get the component name
            comp, dvName = k.split(':',1)

            # now check if this comp has this dv
            if dvName in self.comps[comp].dvDict:
                # set the value
                self.comps[comp].dvDict[dvName] = v

        # loop over the components and set the values
        for comp in self.compNames:
            self.comps[comp].DVGeo.setDesignVars(self.comps[comp].dvDict)

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
        newPts = numpy.zeros((self.ptSets[ptSetName].nPts, 3))

        # we first need to update all points with their respective DVGeo objects
        for comp in self.compNames:
            ptsComp = self.comps[comp].DVGeo.update(ptSetName)

            # now save this info with the pointset mapping
            ptMap = self.ptSets[ptSetName].compMap[comp]
            newPts[ptMap] = ptsComp

        # get the delta
        delta = newPts - self.ptSets[ptSetName].points

        # then apply the intersection treatment
        for IC in self.intersectComps:
            delta = IC.update(ptSetName, delta)

        # now we are ready to take the delta which may be modified by the intersections
        newPts = self.ptSets[ptSetName].points + delta

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
                                                     prefix=comp.join(':'))

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

# object to keep track of feature curves if the user supplies them to track intersections
class featureCurve(object):
    def __init__(self, name, conn):
        # TODO we can remove this and just add the conn to a list.
        self.name = name
        self.conn=conn

class CompIntersection(object):
    def __init__(self, intDict, DVGeo):
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

        # here we grab the useful information from the dictionary,
        # or revert to defaults if no information is provided.

        # first make all dict. keys lowercase
        intDict = dict((k.lower(), v) for k,v in intDict.items())

        # names of compA and compB must be in the dictionary
        self.compA = DVGeo.comps[intDict['compa']]
        self.compB = DVGeo.comps[intDict['compb']]

        self.dStar = intDict['dstar']
        self.halfdStar = self.dStar/2.0
        self.ptSets = OrderedDict()

        # feature curve names
        self.featureCurveNames = intDict['featurecurves']
        for i in range(len(self.featureCurveNames)):
            self.featureCurveNames[i] = self.featureCurveNames[i].lower()

        self.distTol = 1e-14 # Ney's default was 1e-7
        if 'disttol' in intDict:
            self.distTol = intDict['disttol']

        # only the node coordinates will be modified for the intersection calculations because we have calculated and saved all the connectivity information
        self.seam0 = self._getIntersectionSeam(self.comm , firstCall = True)
        self.seam = self.seam0.copy()

    def setSurface(self, comm):
        """ This set the new udpated surface on which we need to comptue the new intersection curve"""

        # get the updated surface coordinates
        self._getUpdatedCoords()

        self.seam = self._getIntersectionSeam(comm)

    def addPointSet(self, pts, ptSetName):

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
        # loop over the feature curves
        for curveName in self.featureCurveNames:
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

        # print("after feature detection")
        # these are the elements at "features"
        # print(breakList)
        # these are the nodes of the element
        # print(seamConn[breakList])
        # these are the feature nodes (i.e. the first nodes of feature elements)
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
            breakList = numpy.mod(seamConn.shape[0] - numpy.flip(breakList, axis=0), seamConn.shape[0])
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

        # print(seam)
        return seam.copy()
