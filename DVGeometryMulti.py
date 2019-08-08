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

    def __init__(self, comps, FFDFiles, triMeshFiles, intersectedComps=None, comm=MPI.COMM_WORLD):

        self.compNames = comps
        self.comps = OrderedDict()
        self.DVGeoDict = OrderedDict()
        self.ptSets = OrderedDict()
        self.comm = comm
        self.intersectComps = []
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
                    barsConn[part] = sectionDict[part]['barsConn']
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

class PointSet(object):
    def __init__(self, points):
        self.points = points
        self.nPts = len(self.points)
        self.compMap = OrderedDict()