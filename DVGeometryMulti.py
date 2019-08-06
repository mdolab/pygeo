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
from pysurf.geometryEngines.TSurf.python import intersectionAPI, curveSearchAPI, utilitiesAPI, tsurf_tools
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
    """A class for manipulating multiple components using multiple FFDs

    """

    def __init__(self, comps, FFDFiles, triMeshFiles, intersectedComps=None, comm=MPI.COMM_WORLD):

        self.compNames = comps
        self.comps = OrderedDict()
        self.DVGeoDict = OrderedDict()

        for (comp, filename, triMesh) in zip(comps, FFDFiles, triMeshFiles):
            # we need to create a new DVGeo object for this component
            DVGeo = DVGeometry(filename)

            # We also need to read the triMesh and save the points
            nodes, triConn, barsConn = self._readCGNSFile(filename)

            # add these points to the corresponding dvgeo
            DVGeo.addPointSet(nodes, 'triNodes')

            # initialize the component object
            self.comps[comp] = component(comp, DVGeo, nodes, triConn, barsConn)

            # also save the DVGeometry pointer in the dictionary we pass back
            self.DVGeoDict[comp] = DVGeo

        # return DVGeo objects so that users can add design variables
        return self.DVGeoDict

    def addPointSet(self, points, ptName, **kwargs):

        # first figure out which points are inside which FFD.

        # we need to loop over each point and figure out if they are inside the bbox of a single FFD.

        # if they are in both FFDs, then we need to project these points to both triangulated surfaces

        # then pick the one that has a shorter distance.

        # we know that if points are outside one FFD, they must be inside the other...

        # for points in both FFDs, project to triangulated surfaces and figure out which component they correspond to

        # create the mapping array that picks components from each ptSet.

        # using the mapping array, add the pointsets to respective DVGeo objects

        # loop over the intersections and add pointsets

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

        # we first need to update all points with their respective DVGeo objects


        # then apply the intersection treatment


        # set the jacobian to be out of date

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
        if rank == 0:
            print('Reading file %s'%filename)
            # use the default routine in tsurftools
            t1 = time.time()
            nodes, sectionDict = tsurf_tools.getCGNSsections(filename, comm = MPI.COMM_SELF)
            t2 = time.time()
            print('Finished reading the cgns file')
            print('Reading the cgns file took', (t2-t1))

            triConn = numpy.zeros((0,3), dtype = numpy.int8)
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
    def __init__(self, name, DVGeo, nodes, triConn, barsConn):

        # save the info
        self.name = name
        self.DVGeo = DVGeo
        self.nodes = nodes
        self.triConn = triConn
        self.barsConn = barsConn

        # Create a dictionary for pointset information
        self.ptSets = OrderedDict()

        # also a dictionary for DV names
        self.dvDict = {}