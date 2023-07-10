# Standard Python modules
from collections import OrderedDict

# External modules
from baseclasses.utils import Error
from mpi4py import MPI
import numpy as np
from scipy import sparse

try:
    # External modules
    from pysurf import (
        adtAPI,
        adtAPI_cs,
        curveSearchAPI,
        curveSearchAPI_cs,
        intersectionAPI,
        intersectionAPI_cs,
        tecplot_interface,
        tsurf_tools,
        utilitiesAPI,
        utilitiesAPI_cs,
    )

    pysurfInstalled = True
except ImportError:
    pysurfInstalled = False


class DVGeometryMulti:
    """
    A class for manipulating multiple components using multiple FFDs
    and handling design changes near component intersections.

    Parameters
    ----------
    comm : MPI.IntraComm, optional
       The communicator associated with this geometry object.

    checkDVs : bool, optional
        Flag to check whether there are duplicate DV names in or across components.

    debug : bool, optional
        Flag to generate output useful for debugging the intersection setup.

    isComplex : bool, optional
        Flag to use complex variables for complex step verification.

    """

    def __init__(self, comm=MPI.COMM_WORLD, checkDVs=True, debug=False, isComplex=False):
        # Check to make sure pySurf is installed before initializing
        if not pysurfInstalled:
            raise ImportError("pySurf is not installed and is required to use DVGeometryMulti.")

        self.compNames = []
        self.comps = OrderedDict()
        self.DVGeoDict = OrderedDict()
        self.points = OrderedDict()
        self.comm = comm
        self.updated = {}
        self.intersectComps = []
        self.checkDVs = checkDVs
        self.debug = debug
        self.complex = isComplex

        # Set real or complex Fortran API
        if isComplex:
            self.dtype = complex
            self.adtAPI = adtAPI_cs.adtapi
        else:
            self.dtype = float
            self.adtAPI = adtAPI.adtapi

    def addComponent(self, comp, DVGeo, triMesh=None, scale=1.0, bbox=None, pointSetKwargs=None):
        """
        Method to add components to the DVGeometryMulti object.

        Parameters
        ----------
        comp : str
            The name of the component.

        DVGeo : DVGeometry
            The DVGeometry object defining the component FFD.

        triMesh : str, optional
            Path to the triangulated mesh file for this component.

        scale : float, optional
            A multiplicative scaling factor applied to the triangulated mesh coordinates.
            Useful for when the scales of the triangulated and CFD meshes do not match.

        bbox : dict, optional
            Specify a bounding box that is different from the bounds of the FFD.
            The keys can include ``xmin``, ``xmax``, ``ymin``, ``ymax``, ``zmin``, ``zmax``.
            If any of these are not provided, the FFD bound is used.

        pointSetKwargs : dict, optional
            Keyword arguments to be passed to the component addPointSet call for the triangulated mesh.

        """

        # Assign mutable defaults
        if bbox is None:
            bbox = {}
        if pointSetKwargs is None:
            pointSetKwargs = {}

        if triMesh is not None:
            # We also need to read the triMesh and save the points
            nodes, triConn, triConnStack, barsConn = self._readCGNSFile(triMesh)

            # scale the nodes
            nodes *= scale

            # add these points to the corresponding dvgeo
            DVGeo.addPointSet(nodes, "triMesh", **pointSetKwargs)
        else:
            # the user has not provided a triangulated surface mesh for this file
            nodes = None
            triConn = None
            triConnStack = None
            barsConn = None

        # we will need the bounding box information later on, so save this here
        xMin, xMax = DVGeo.FFD.getBounds()

        # also we might want to modify the bounding box if the user specified any coordinates
        if "xmin" in bbox:
            xMin[0] = bbox["xmin"]
        if "ymin" in bbox:
            xMin[1] = bbox["ymin"]
        if "zmin" in bbox:
            xMin[2] = bbox["zmin"]
        if "xmax" in bbox:
            xMax[0] = bbox["xmax"]
        if "ymax" in bbox:
            xMax[1] = bbox["ymax"]
        if "zmax" in bbox:
            xMax[2] = bbox["zmax"]

        # initialize the component object
        self.comps[comp] = component(comp, DVGeo, nodes, triConn, triConnStack, barsConn, xMin, xMax)

        # add the name to the list
        self.compNames.append(comp)

        # also save the DVGeometry pointer in the dictionary we pass back
        self.DVGeoDict[comp] = DVGeo

    def addIntersection(
        self,
        compA,
        compB,
        dStarA=0.2,
        dStarB=0.2,
        featureCurves=None,
        distTol=1e-14,
        project=False,
        marchDir=1,
        includeCurves=False,
        intDir=None,
        curveEpsDict=None,
        trackSurfaces=None,
        excludeSurfaces=None,
        remeshBwd=True,
        anisotropy=[1.0, 1.0, 1.0],
    ):
        """
        Method that defines intersections between components.

        Parameters
        ----------
        compA : str
            The name of the first component.

        compB : str
            The name of the second component.

        dStarA : float, optional
            Distance from the intersection over which the inverse-distance deformation is applied on compA.

        dStarB : float, optional
            Distance from the intersection over which the inverse-distance deformation is applied on compB.

        featureCurves : list or dict, optional
            Points on feature curves will remain on the same curve after deformations and projections.
            Feature curves can be specified as a list of curve names.
            In this case, the march direction for all curves is ``marchDir``.
            Alternatively, a dictionary can be provided.
            In this case, the keys are the curve names and the values are the march directions for each curve.
            See ``marchDir`` for the definition of march direction.

        distTol : float, optional
            Distance tolerance to merge nearby nodes in the intersection curve.

        project : bool, optional
            Flag to specify whether to project points to curves and surfaces after the deformation step.

        marchDir : int, optional
            The side of the intersection where the feature curves are remeshed.
            The sign determines the direction and the value (1, 2, 3) specifies the axis (x, y, z).
            If ``remeshBwd`` is True, the other side is also remeshed.
            In this case, the march direction only serves to define the 'free end' of the feature curve.
            If None, the entire curve is remeshed.
            This argument is only used if a list is provided for ``featureCurves``.

        includeCurves : bool, optional
            Flag to specify whether to include features curves in the inverse-distance deformation.

        intDir : int, optional
            If there are multiple intersection curves, this specifies which curve to choose.
            The sign determines the direction and the value (1, 2, 3) specifies the axis (x, y, z).
            For example, -1 specifies the intersection curve as the one that is further in the negative x-direction.

        curveEpsDict : dict, optional
            Required if using feature curves.
            The keys of the dictionary are the curve names and the values are distances.
            All points within the specified distance from the curve are considered to be on the curve.

        trackSurfaces : dict, optional
            Points on tracked surfaces will remain on the same surfaces after deformations and projections.
            The keys of the dictionary are the surface names and the values are distances.
            All points within the specified distance from the surface are considered to be on the surface.

        excludeSurfaces : dict, optional
            Points on excluded surfaces are removed from the intersection computations.
            The keys of the dictionary are the surface names and the values are distances.
            All points within the specified distance from the surface are considered to be on the surface.

        remeshBwd : bool, optional
            Flag to specify whether to remesh feature curves on the side opposite that
            which is specified by the march direction.

        anisotropy : list of float, optional
            List with three entries specifying scaling factors in the [x, y, z] directions.
            The factors multiply the [x, y, z] distances used in the curve-based deformation.
            Smaller factors in a certain direction will amplify the effect of the parts of the curve
            that lie in that direction from the points being warped.
            This tends to increase the mesh quality in one direction at the expense of other directions.
            This can be useful when the initial intersection curve is skewed.

        """

        # Assign mutable defaults
        if featureCurves is None:
            featureCurves = []
        if curveEpsDict is None:
            curveEpsDict = {}
        if trackSurfaces is None:
            trackSurfaces = {}
        if excludeSurfaces is None:
            excludeSurfaces = {}

        # just initialize the intersection object
        self.intersectComps.append(
            CompIntersection(
                compA,
                compB,
                dStarA,
                dStarB,
                featureCurves,
                distTol,
                self,
                project,
                marchDir,
                includeCurves,
                intDir,
                curveEpsDict,
                trackSurfaces,
                excludeSurfaces,
                remeshBwd,
                anisotropy,
                self.debug,
                self.dtype,
            )
        )

    def getDVGeoDict(self):
        """Return a dictionary of component DVGeo objects."""
        return self.DVGeoDict

    def addPointSet(self, points, ptName, compNames=None, comm=None, applyIC=False, **kwargs):
        """
        Add a set of coordinates to DVGeometryMulti.
        The is the main way that geometry, in the form of a coordinate list, is manipulated.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed.
            These coordinates should all be inside at least one FFD volume.
        ptName : str
            A user supplied name to associate with the set of coordinates.
            This name will need to be provided when updating the coordinates
            or when getting the derivatives of the coordinates.
        compNames : list, optional
            A list of component names that this point set should be added to.
            To ease bookkeepping, an empty point set with ptName will be added to components not in this list.
            If a list is not provided, this point set is added to all components.
        comm : MPI.IntraComm, optional
            Comm that is associated with the added point set. Does not
            work now, just added to be consistent with the API of
            other DVGeo types.
        applyIC : bool, optional
            Flag to specify whether this point set will follow the updated intersection curve(s).
            This is typically only needed for the CFD surface mesh.

        """

        # if compList is not provided, we use all components
        if compNames is None:
            compNames = self.compNames

        # before we do anything, we need to create surface ADTs
        # for which the user provided triangulated meshes
        for comp in compNames:
            # check if we have a trimesh for this component
            if self.comps[comp].triMesh:
                # Now we build the ADT using pySurf
                # Set bounding box for new tree
                BBox = np.zeros((2, 3))
                useBBox = False

                # dummy connectivity data for quad elements since we have all tris
                quadConn = np.zeros((0, 4))

                # Compute set of nodal normals by taking the average normal of all
                # elements surrounding the node. This allows the meshing algorithms,
                # for instance, to march in an average direction near kinks.
                nodal_normals = self.adtAPI.adtcomputenodalnormals(
                    self.comps[comp].nodes.T, self.comps[comp].triConnStack.T, quadConn.T
                )
                self.comps[comp].nodal_normals = nodal_normals.T

                # Create new tree (the tree itself is stored in Fortran level)
                self.adtAPI.adtbuildsurfaceadt(
                    self.comps[comp].nodes.T,
                    self.comps[comp].triConnStack.T,
                    quadConn.T,
                    BBox.T,
                    useBBox,
                    MPI.COMM_SELF.py2f(),
                    comp,
                )

        # create the pointset class
        self.points[ptName] = PointSet(points, comm=comm)

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
                # apply a small tolerance for the bounding box in case points are coincident with the FFD
                boundTol = 1e-16
                xMin = self.comps[comp].xMin
                xMax = self.comps[comp].xMax
                xMin -= np.abs(xMin * boundTol) + boundTol
                xMax += np.abs(xMax * boundTol) + boundTol

                # check if inside
                if (
                    xMin[0] < points[i, 0] < xMax[0]
                    and xMin[1] < points[i, 1] < xMax[1]
                    and xMin[2] < points[i, 2] < xMax[2]
                ):
                    # add this component to the projection list
                    projList.append(comp)

                    # this point was not inside any other FFD before
                    if not inFFD:
                        inFFD = True
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
                            dist2 = np.ones(numPts, dtype=self.dtype) * 1e10
                            xyzProj = np.zeros((numPts, 3), dtype=self.dtype)
                            normProjNotNorm = np.zeros((numPts, 3), dtype=self.dtype)

                            # Call projection function
                            _, _, _, _ = self.adtAPI.adtmindistancesearch(
                                points[i].T, comp, dist2, xyzProj.T, self.comps[comp].nodal_normals.T, normProjNotNorm.T
                            )

                            # if this is closer than the previous min, take this comp
                            if dist2 < dMin2:
                                dMin2 = dist2[0]
                                inComp = comp

                        else:
                            raise Error(
                                f"The point at (x, y, z) = ({points[i, 0]:.3f}, {points[i, 1]:.3f} {points[i, 2]:.3f})"
                                + f"in point set {ptName} is inside multiple FFDs but a triangulated mesh "
                                + f"for component {comp} is not provided to determine which component owns this point."
                            )

            # this point was inside at least one FFD. If it was inside multiple,
            # we projected it before to figure out which component it should belong to
            if inFFD:
                # we can add the point index to the list of points inComp owns
                self.points[ptName].compMap[inComp].append(i)

                # also create a flattened version of the compMap
                for j in range(3):
                    self.points[ptName].compMapFlat[inComp].append(3 * i + j)

            # this point is outside any FFD...
            else:
                raise Error(
                    f"The point at (x, y, z) = ({points[i, 0]:.3f}, {points[i, 1]:.3f} {points[i, 2]:.3f}) "
                    + f"in point set {ptName} is not inside any FFDs."
                )

        # using the mapping array, add the pointsets to respective DVGeo objects
        for comp in self.compNames:
            compMap = self.points[ptName].compMap[comp]
            self.comps[comp].DVGeo.addPointSet(points[compMap], ptName, **kwargs)

        # check if this pointset will get the IC treatment
        if applyIC:
            # loop over the intersections and add pointsets
            for IC in self.intersectComps:
                IC.addPointSet(points, ptName, self.points[ptName].compMap, comm)

        # finally, we can deallocate the ADTs
        for comp in compNames:
            if self.comps[comp].triMesh:
                self.adtAPI.adtdeallocateadts(comp)

        # mark this pointset as up to date
        self.updated[ptName] = False

    def setDesignVars(self, dvDict):
        """
        Standard routine for setting design variables from a design variable dictionary.

        Parameters
        ----------
        dvDict : dict
            Dictionary of design variables.
            The keys of the dictionary must correspond to the design variable names.
            Any additional keys in the dictionary are simply ignored.

        """

        # Check if we have duplicate DV names
        if self.checkDVs:
            dvNames = self.getVarNames()
            duplicates = len(dvNames) != len(set(dvNames))
            if duplicates:
                raise Error(
                    "There are duplicate DV names in a component or across components. "
                    "If this is intended, initialize the DVGeometryMulti class with checkDVs=False."
                )

        # loop over the components and set the values
        for comp in self.compNames:
            self.comps[comp].DVGeo.setDesignVars(dvDict)

        # We need to give the updated coordinates to each of the
        # intersectComps (if we have any) so they can update the new intersection curve
        for IC in self.intersectComps:
            IC.setSurface(self.comm)

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

    def getValues(self):
        """
        Generic routine to return the current set of design variables.
        Values are returned in a dictionary format that would be suitable for a subsequent call to setDesignVars().

        Returns
        -------
        dvDict : dict
            Dictionary of design variables.

        """

        dvDict = {}
        # we need to loop over each DVGeo object and get the DVs
        for comp in self.compNames:
            dvDictComp = self.comps[comp].DVGeo.getValues()
            # we need to loop over these DVs
            for k, v in dvDictComp.items():
                dvDict[k] = v

        return dvDict

    def update(self, ptSetName, config=None):
        """
        This is the main routine for returning coordinates that have been updated by design variables.
        Multiple configs are not supported.

        Parameters
        ----------
        ptSetName : str
            Name of point set to return.
            This must match one of those added in an :func:`addPointSet()` call.

        """

        # get the new points
        newPts = np.zeros((self.points[ptSetName].nPts, 3), dtype=self.dtype)

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
            # check if this IC is active for this ptSet
            if ptSetName in IC.points:
                delta = IC.update(ptSetName, delta)

        # now we are ready to take the delta which may be modified by the intersections
        newPts = self.points[ptSetName].points + delta

        # now, project the points that were warped back onto the trimesh
        for IC in self.intersectComps:
            if IC.projectFlag and ptSetName in IC.points:
                # new points will be modified in place using the newPts array
                IC.project(ptSetName, newPts)

        # set the pointset up to date
        self.updated[ptSetName] = True

        return newPts

    def pointSetUpToDate(self, ptSetName):
        """
        This is used externally to query if the object needs to update its point set or not.
        When update() is called with a point set, the self.updated value for pointSet is flagged as True.
        We reset all flags to False when design variables are set because nothing (in general) will up to date anymore.
        Here we just return that flag.

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
        """Return the number of DVs."""
        # Loop over components and sum the number of DVs
        nDV = 0
        for comp in self.compNames:
            nDV += self.comps[comp].DVGeo.getNDV()
        return nDV

    def getVarNames(self, pyOptSparse=False):
        """
        Return a list of the design variable names.
        This is typically used when specifying a ``wrt=`` argument for pyOptSparse.

        Examples
        --------
        >>> optProb.addCon(.....wrt=DVGeo.getVarNames())

        """
        dvNames = []
        # create a list of DVs from each comp
        for comp in self.compNames:
            # first get the list of DVs from this component
            varNames = self.comps[comp].DVGeo.getVarNames()

            # add the component DVs to the full list
            dvNames.extend(varNames)

        return dvNames

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        """
        This function computes sensitivity information.

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

        comm : MPI.IntraComm, optional
            The communicator to use to reduce the final derivative.
            If comm is None, no reduction takes place.

        config : str or list, optional
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.


        Returns
        -------
        dIdxDict : dict
            The dictionary containing the derivatives, suitable for pyOptSparse.

        Notes
        -----
        The ``child`` and ``nDVStore`` options are only used
        internally and should not be changed by the user.

        """

        # Compute the total Jacobian for this point set
        self._computeTotalJacobian(ptSetName)

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])
        N = dIdpt.shape[0]

        # create a dictionary to save total sensitivity info that might come out of the ICs
        compSensList = []

        # if we projected points for any intersection treatment,
        # we need to propagate the derivative seed of the projected points
        # back to the seeds for the initial points we get after ID-warping
        for IC in self.intersectComps:
            if IC.projectFlag and ptSetName in IC.points:
                # initialize the seed contribution to the intersection seam and feature curves from project_b
                IC.seamBarProj[ptSetName] = np.zeros((N, IC.seam0.shape[0], IC.seam0.shape[1]))

                # we pass in dIdpt and the intersection object, along with pointset information
                # the intersection object adjusts the entries corresponding to projected points
                # and passes back dIdpt in place.
                compSens = IC.project_b(ptSetName, dIdpt, comm)

                # append this to the dictionary list...
                compSensList.append(compSens)

        # do the transpose multiplication

        if self.debug:
            print(f"[{self.comm.rank}] finished project_b")

        # we need to go through all ICs bec even though some procs might not have points on the intersection,
        # communication is easier and we can reduce compSens as we compute them
        for IC in self.intersectComps:
            if ptSetName in IC.points:
                compSens = IC.sens(dIdpt, ptSetName, comm)
                # save the sensitivities from the intersection stuff
                compSensList.append(compSens)

        if self.debug:
            print(f"[{self.comm.rank}] finished IC.sens")

        # reshape the dIdpt array from [N] * [nPt] * [3] to  [N] * [nPt*3]
        dIdpt = dIdpt.reshape((dIdpt.shape[0], dIdpt.shape[1] * 3))

        # jacobian for the pointset
        jac = self.points[ptSetName].jac

        # this is the mat-vec product for the remaining seeds.
        # this only contains the effects of the FFD motion,
        # projections and intersections are handled separately in compSens
        dIdxT_local = jac.T.dot(dIdpt.T)
        dIdx_local = dIdxT_local.T

        # If we have a comm, globaly reduce with sum
        if comm:
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # use respective DVGeo's convert to dict functionality
        dIdxDict = OrderedDict()
        dvOffset = 0
        for comp in self.compNames:
            DVGeo = self.comps[comp].DVGeo
            nDVComp = DVGeo.getNDV()

            # we only do this if this component has at least one DV
            if nDVComp > 0:
                # this part of the sensitivity matrix is owned by this dvgeo
                dIdxComp = DVGeo.convertSensitivityToDict(dIdx[:, dvOffset : dvOffset + nDVComp])

                for k, v in dIdxComp.items():
                    dIdxDict[k] = v

                # also increment the offset
                dvOffset += nDVComp

        # finally, we can add the contributions from triangulated component meshes
        for compSens in compSensList:
            # loop over the items of compSens, which are guaranteed to be in dIdxDict
            for k, v in compSens.items():
                # these will bring in effects from projections and intersection computations
                dIdxDict[k] += v

        if self.debug:
            print(f"[{self.comm.rank}] finished DVGeo.totalSensitivity")

        return dIdxDict

    def addVariablesPyOpt(
        self,
        optProb,
        globalVars=True,
        localVars=True,
        sectionlocalVars=True,
        ignoreVars=None,
        freezeVars=None,
        comps=None,
    ):
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
            List of design variables the user doesn't want to use
            as optimization variables.

        freezeVars : list of string
            List of design variables the user wants to add as optimization
            variables, but to have the lower and upper bounds set at the current
            variable. This effectively eliminates the variable, but it the variable
            is still part of the optimization.

        comps : list
            List of components we want to add the DVs of.
            If no list is provided, we will add DVs from all components.

        """

        # If no list was provided, we use all components
        if comps is None:
            comps = self.compNames

        # We can simply loop over all DV objects and call their respective addVariablesPyOpt function
        for comp in comps:
            self.comps[comp].DVGeo.addVariablesPyOpt(
                optProb,
                globalVars=globalVars,
                localVars=localVars,
                sectionlocalVars=sectionlocalVars,
                ignoreVars=ignoreVars,
                freezeVars=freezeVars,
            )

    def getLocalIndex(self, iVol, comp):
        """Return the local index mapping that points to the global coefficient list for a given volume.

        Parameters
        ----------

        iVol : int
            Index specifying the FFD volume.

        comp : str
            Name of the component.

        """

        # Call this on the component DVGeo
        DVGeo = self.comps[comp].DVGeo
        return DVGeo.FFD.topo.lIndex[iVol].copy()

    # ----------------------------------------------------------------------
    #        THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER
    # ----------------------------------------------------------------------

    def _readCGNSFile(self, filename):
        # this function reads the unstructured CGNS grid in filename and returns
        # node coordinates and element connectivities.
        # Here, only the root proc reads the cgns file, broadcasts node and connectivity info.

        # only root proc reads the file
        if self.comm.rank == 0:
            print(f"Reading file {filename}")
            # use the default routine in tsurftools
            nodes, sectionDict = tsurf_tools.getCGNSsections(filename, comm=MPI.COMM_SELF)
            print("Finished reading the cgns file")

            # Convert the nodes to complex if necessary
            nodes = nodes.astype(self.dtype)

            triConn = {}
            triConnStack = np.zeros((0, 3), dtype=np.int8)
            barsConn = {}

            for part in sectionDict:
                if "triaConnF" in sectionDict[part].keys():
                    # this is a surface, read the tri connectivities
                    triConn[part.lower()] = sectionDict[part]["triaConnF"]
                    triConnStack = np.vstack((triConnStack, sectionDict[part]["triaConnF"]))

                if "barsConn" in sectionDict[part].keys():
                    # this is a curve, save the curve connectivity
                    barsConn[part.lower()] = sectionDict[part]["barsConn"]

            print(f"The {filename} mesh has {len(nodes)} nodes and {len(triConnStack)} elements.")
        else:
            # create these to recieve the data
            nodes = None
            triConn = None
            triConnStack = None
            barsConn = None

        # each proc gets the nodes and connectivities
        nodes = self.comm.bcast(nodes, root=0)
        triConn = self.comm.bcast(triConn, root=0)
        triConnStack = self.comm.bcast(triConnStack, root=0)
        barsConn = self.comm.bcast(barsConn, root=0)

        return nodes, triConn, triConnStack, barsConn

    def _computeTotalJacobian(self, ptSetName):
        """
        This routine computes the total jacobian. It takes the jacobians
        from respective DVGeo objects and also computes the jacobians for
        the intersection seams. We then use this information in the
        totalSensitivity function.

        """

        # number of design variables
        nDV = self.getNDV()

        # Initialize the Jacobian as a LIL matrix because this is convenient for indexing
        jac = sparse.lil_matrix((self.points[ptSetName].nPts * 3, nDV))

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
                # Get the component Jacobian
                compJ = self.comps[comp].DVGeo.JT[ptSetName].T

                # Set the block of the full Jacobian associated with this component
                jac[ptSet.compMapFlat[comp], dvOffset : dvOffset + nDVComp] = compJ

            # increment the offset
            dvOffset += nDVComp

        # Convert to CSR format because this is better for arithmetic
        jac = sparse.csr_matrix(jac)

        # now we can save this jacobian in the pointset
        ptSet.jac = jac


class component:
    def __init__(self, name, DVGeo, nodes, triConn, triConnStack, barsConn, xMin, xMax):
        # save the info
        self.name = name
        self.DVGeo = DVGeo
        self.nodes = nodes
        self.triConn = triConn
        self.triConnStack = triConnStack
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
        self.nodes = self.DVGeo.update("triMesh")


class PointSet:
    def __init__(self, points, comm):
        self.points = points
        self.nPts = len(self.points)
        self.compMap = OrderedDict()
        self.compMapFlat = OrderedDict()
        self.comm = comm


class CompIntersection:
    def __init__(
        self,
        compA,
        compB,
        dStarA,
        dStarB,
        featureCurves,
        distTol,
        DVGeo,
        project,
        marchDir,
        includeCurves,
        intDir,
        curveEpsDict,
        trackSurfaces,
        excludeSurfaces,
        remeshBwd,
        anisotropy,
        debug,
        dtype,
    ):
        """
        Class to store information required for an intersection.
        Here, we use some Fortran code from pySurf.
        Internally, we store the indices and weights of the points that this intersection will modify.
        This code is not super efficient because it is in Python.

        See the documentation for ``addIntersection`` in DVGeometryMulti for the API.

        """

        # same communicator with DVGeo
        self.comm = DVGeo.comm

        # define epsilon as a small value to prevent division by zero in the inverse distance computation
        self.eps = 1e-20

        # counter for outputting curves etc at each update
        self.counter = 0

        # flag that determines if we will remesh the other side of the feature curves on compB
        self.remeshBwd = remeshBwd

        # Flag for debug ouput
        self.debug = debug

        # Set real or complex Fortran APIs
        self.dtype = dtype
        if dtype == float:
            self.adtAPI = adtAPI.adtapi
            self.curveSearchAPI = curveSearchAPI.curvesearchapi
            self.intersectionAPI = intersectionAPI.intersectionapi
            self.utilitiesAPI = utilitiesAPI.utilitiesapi
            self.mpi_type = MPI.DOUBLE
        elif dtype == complex:
            self.adtAPI = adtAPI_cs.adtapi
            self.curveSearchAPI = curveSearchAPI_cs.curvesearchapi
            self.intersectionAPI = intersectionAPI_cs.intersectionapi
            self.utilitiesAPI = utilitiesAPI_cs.utilitiesapi
            self.mpi_type = MPI.C_DOUBLE_COMPLEX

        # tolerance used for each curve when mapping nodes to curves
        self.curveEpsDict = {}
        for k, v in curveEpsDict.items():
            self.curveEpsDict[k.lower()] = v

        # beginning and end element indices for each curve
        self.seamBeg = {}
        self.seamEnd = {}

        # indices of nodes to be projected to curves.
        self.curveProjIdx = {}

        # lists to track which feature curve is on which comp
        self.curvesOnA = []
        self.curvesOnB = []

        # a dict to to keep track which curves get points mapped to them
        # keys will be pointSetNames, then the values will be another dict,
        # where the keys are curve names and values will be a bool value
        self.curveProjFlag = {}

        # dicts to keep track of the coordinates and counts of points projected to curves
        self.curvePtCounts = {}
        self.curvePtCoords = {}

        # dict to keep track of the total number of points on each curve
        self.nCurvePts = {}

        # dictionary to keep the seam seeds that come from curve projections
        self.seamBarProj = {}

        # dictionaries to save the indices of points mapped to surfaces for each comp
        self.surfIdxA = {}
        self.surfIdxB = {}

        # names of compA and compB must be provided
        self.compA = DVGeo.comps[compA]
        self.compB = DVGeo.comps[compB]

        self.dStarA = dStarA
        self.dStarB = dStarB
        self.points = OrderedDict()

        # Make surface names lowercase
        self.trackSurfaces = {}
        for k, v in trackSurfaces.items():
            self.trackSurfaces[k.lower()] = v

        self.excludeSurfaces = {}
        for k, v in excludeSurfaces.items():
            if k.lower() in self.trackSurfaces:
                raise Error(f"Surface {k} cannot be in both trackSurfaces and excludeSurfaces.")
            self.excludeSurfaces[k.lower()] = v

        # Save anisotropy list
        self.anisotropy = anisotropy

        # process the feature curves

        # list to save march directions
        marchDirs = []

        # list to save curve names where we remesh all the curve
        self.remeshAll = []

        # if a list is provided, we use this and the marchdir information
        if type(featureCurves) is list:
            self.featureCurveNames = featureCurves
            for i in range(len(self.featureCurveNames)):
                self.featureCurveNames[i] = self.featureCurveNames[i].lower()
                # get one march dir per curve
                marchDirs.append(marchDir)

        else:
            # if a dict is provided, the marchdirs are the dict values
            # we save this info in lists
            self.featureCurveNames = []
            # save the curve name and march direction information
            for k, v in featureCurves.items():
                self.featureCurveNames.append(k.lower())
                marchDirs.append(v)

        # now loop over the feature curves and flip if necessary
        for ii, curveName in enumerate(self.featureCurveNames):
            # figure out which comp owns this curve...
            if curveName in self.compB.barsConn:
                curveComp = self.compB
                self.curvesOnB.append(curveName)
            elif curveName in self.compA.barsConn:
                curveComp = self.compA
                self.curvesOnA.append(curveName)
            else:
                raise Error(f"Curve {curveName} does not belong in {self.compA.name} or {self.compB.name}.")

            # sort the feature curve
            newConn, newMap = tsurf_tools.FEsort(curveComp.barsConn[curveName].tolist())

            # we only want to have a single curve
            if len(newConn) > 1:
                raise Error(f"The curve {curveName} generated more than one curve with FESort.")

            # get the connectivity
            newConn = newConn[0]

            # we may also need to flip the curve
            curveNodes = curveComp.nodes

            if marchDirs[ii] is None:
                # we remesh all of this curve
                self.remeshAll.append(curveName)
                curveComp.barsConn[curveName] = newConn
            else:
                # get the direction we want to march
                mdir = abs(marchDirs[ii]) - 1
                msign = np.sign(marchDirs[ii])

                # check if we need to flip
                if msign * curveNodes[newConn[0][0]][mdir] > msign * curveNodes[newConn[0][1]][mdir]:
                    # flip on both axes
                    newConn = np.flip(newConn, axis=0)
                    newConn = np.flip(newConn, axis=1)

                # save the new connectivity
                curveComp.barsConn[curveName] = newConn

        self.distTol = distTol

        # flag to determine if we want to project nodes after intersection treatment
        self.projectFlag = project

        # create the dictionary if we are projecting.
        if project:
            self.projData = {}
            if includeCurves:
                # dict to save the data related to projection to curves
                self.curveProjData = {}

        # flag to include feature curves in ID-warping
        self.incCurves = includeCurves

        # direction to pick if we have multiple intersection curves
        self.intDir = intDir

        # only the node coordinates will be modified for the intersection calculations because we have calculated and saved all the connectivity information
        if self.comm.rank == 0:
            print(f"Computing initial intersection between {compA} and {compB}")
        self.seam0 = self._getIntersectionSeam(self.comm, firstCall=True)
        self.seam = self.seam0.copy()

    def setSurface(self, comm):
        """This set the new udpated surface on which we need to compute the new intersection curve"""

        # get the updated surface coordinates
        self._getUpdatedCoords()

        self.seam = self._getIntersectionSeam(comm)

    def addPointSet(self, pts, ptSetName, compMap, comm):
        # Figure out which points this intersection object has to deal with

        # Use pySurf to project the point on curve
        # Get number of points
        nPoints = len(pts)

        # Initialize references if user provided none
        dist2 = np.ones(nPoints, dtype=self.dtype) * 1e10
        xyzProj = np.zeros((nPoints, 3), dtype=self.dtype)
        tanProj = np.zeros((nPoints, 3), dtype=self.dtype)
        elemIDs = np.zeros((nPoints), dtype="int32")

        # Only call the Fortran code if we have at least one point
        if nPoints > 0:
            # This will modify xyzProj, tanProj, dist2, and elemIDs if we find better projections than dist2.
            # Remember that we should adjust some indices before calling the Fortran code
            # Remember to use [:] to don't lose the pointer (elemIDs is an input/output variable)
            elemIDs[:] = (
                elemIDs + 1
            )  # (we need to do this separetely because Fortran will actively change elemIDs contents.
            self.curveSearchAPI.mindistancecurve(
                pts.T, self.nodes0.T, self.conn0.T + 1, xyzProj.T, tanProj.T, dist2, elemIDs
            )

            # Adjust indices back to Python standards
            elemIDs[:] = elemIDs - 1

        # dist2 has the array of squared distances
        d = np.sqrt(dist2)

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
                    factor = 0.5 * (d[i] / halfdStar) ** 3
                else:
                    factor = 0.5 * (2 - ((dStar - d[i]) / halfdStar) ** 3)

                # Save the index and factor
                indices.append(i)
                factors.append(factor)

        # Get all points included in the intersection computation
        intersectPts = pts[indices]
        nPoints = len(intersectPts)

        if self.projectFlag:
            # Create the dictionaries to save projection data
            self.projData[ptSetName] = {
                # We need one dictionary for each component
                "compA": {"surfaceInd": {}},
                "compB": {"surfaceInd": {}},
            }

            if nPoints > 0 and self.excludeSurfaces:
                # Associate points with the excluded surfaces
                for surface in self.excludeSurfaces:
                    surfaceEps = self.excludeSurfaces[surface]
                    self.associatePointsToSurface(intersectPts, ptSetName, surface, surfaceEps)

                # Combine the excluded indices using a set to avoid duplicates
                excludeSet = set()
                for surface in self.excludeSurfaces:
                    if surface in self.compA.triConn:
                        # Pop this surface from the saved data
                        surfaceInd = self.projData[ptSetName]["compA"]["surfaceInd"].pop(surface)
                    elif surface in self.compB.triConn:
                        surfaceInd = self.projData[ptSetName]["compB"]["surfaceInd"].pop(surface)

                    excludeSet.update(surfaceInd)

                # Invert excludeSet to get the points we want to keep
                oneToN = set(range(nPoints))
                includeSet = oneToN.difference(excludeSet)

                # Keep only the points not associated with the excluded surfaces
                indices = [indices[i] for i in includeSet]
                factors = [factors[i] for i in includeSet]

        # Save the affected indices and the factor in the little dictionary
        self.points[ptSetName] = [pts.copy(), indices, factors, comm]

        # now we need to figure out which components we are projecting to if projection is enabled
        # this can be done faster above but whatever
        if self.projectFlag:
            flagA = False
            flagB = False

            indices = self.points[ptSetName][1]

            # create the list we use to map the points to projection components
            indA = []
            indB = []

            # maybe we can do this vectorized
            for ind in indices:
                # check compA
                if ind in compMap[self.compA.name]:
                    flagA = True
                    indA.append(ind)

                # check compB
                if ind in compMap[self.compB.name]:
                    flagB = True
                    indB.append(ind)

            # Save the flags and indices
            self.projData[ptSetName]["compA"]["flag"] = flagA
            self.projData[ptSetName]["compA"]["ind"] = indA
            self.projData[ptSetName]["compB"]["flag"] = flagB
            self.projData[ptSetName]["compB"]["ind"] = indB

            # Associate points with the tracked surfaces
            for surface in self.trackSurfaces:
                surfaceEps = self.trackSurfaces[surface]
                if surface in self.compA.triConn:
                    compPoints = pts[indA]
                elif surface in self.compB.triConn:
                    compPoints = pts[indB]
                else:
                    raise Error(f"Surface {surface} was not found in {self.compA.name} or {self.compB.name}.")

                # This proc has some points to project
                if len(compPoints) > 0:
                    self.associatePointsToSurface(compPoints, ptSetName, surface, surfaceEps)

            # if we include the feature curves in the warping, we also need to project the added points to the intersection and feature curves and determine how the points map to the curves
            if self.incCurves:
                # convert the list to an array
                # we specify the dtype because numpy cannot know the type when 'indices' is empty
                indices = np.array(indices, dtype="intc")

                # get the coordinates of all points affected by this intersection
                ptsToCurves = pts[indices]

                # project these to the combined curves
                # Use pySurf to project the point on curve
                # Get number of points
                nPoints = len(ptsToCurves)

                # Initialize references if user provided none
                dist2 = np.ones(nPoints, dtype=self.dtype) * 1e10
                xyzProj = np.zeros((nPoints, 3), dtype=self.dtype)
                tanProj = np.zeros((nPoints, 3), dtype=self.dtype)
                elemIDs = np.zeros((nPoints), dtype="int32")

                # Only call the Fortran code if we have at least one point
                if nPoints > 0:
                    # This will modify xyzProj, tanProj, dist2, and elemIDs if we find better projections than dist2.
                    # Remember that we should adjust some indices before calling the Fortran code
                    # Remember to use [:] to don't lose the pointer (elemIDs is an input/output variable)
                    elemIDs[:] = elemIDs + 1
                    # (we need to do this separetely because Fortran will actively change elemIDs contents.
                    self.curveSearchAPI.mindistancecurve(
                        ptsToCurves.T, self.seam0.T, self.seamConn.T + 1, xyzProj.T, tanProj.T, dist2, elemIDs
                    )

                    # Adjust indices back to Python standards
                    elemIDs[:] = elemIDs - 1

                # dist2 has the array of squared distances
                d = np.sqrt(dist2)

                # get the names of all curves including the intersection
                allCurves = ["intersection"]
                for curveName in self.featureCurveNames:
                    allCurves.append(curveName)

                # track the points that dont get associated with any curve
                # get a full masking array with zeros
                allNodesBool = np.zeros(len(elemIDs))

                # dict to save the pt indices
                self.curveProjIdx[ptSetName] = {}
                # dict to save other data
                self.curveProjData[ptSetName] = {}

                # now loop over feature curves and use the epsilon that each curve has
                # to determine which points maps to which curve
                for curveName in allCurves:
                    # get the epsilon for this curve
                    # we map the points closer than eps to this curve
                    eps = self.curveEpsDict[curveName]

                    # also get the range of element IDs this curve owns
                    seamBeg = self.seamBeg[curveName]
                    seamEnd = self.seamEnd[curveName]

                    # this returns a bool array of indices that satisfy the conditions
                    # we check for elemIDs because we projected to all curves at once
                    curveBool = np.all([d < eps, elemIDs >= seamBeg, elemIDs < seamEnd], axis=0)

                    # get the indices of the points mapped to this element
                    idxs = np.nonzero(curveBool)

                    # save the indices. idx has the indices of the "indices" array
                    # that had the indices of points that get any intersection treatment
                    self.curveProjIdx[ptSetName][curveName] = np.array(indices[idxs])

                    if self.debug:
                        ptCoords = ptsToCurves[idxs]
                        tecplot_interface.write_tecplot_scatter(
                            f"{curveName}.plt", curveName, ["X", "Y", "Z"], ptCoords
                        )

                    # also update the masking array
                    # we will use this to figure out the indices that did not get attached to any curves
                    allNodesBool = np.any([curveBool, allNodesBool], axis=0)

                    # create an empty dict to save aux data later on
                    self.curveProjData[ptSetName][curveName] = {}

                # negate the surface mask and get indices
                surfPtIdx = np.nonzero(np.logical_not(allNodesBool))

                # figure out which of these surfNodes live only on components A and B
                allSurfIdx = np.array(indices[surfPtIdx[0]])

                # component A
                mask = np.in1d(allSurfIdx, indA, assume_unique=True)
                # this is the local indices of the points affected
                self.surfIdxA[ptSetName] = allSurfIdx[np.nonzero(mask)]

                # component B
                mask = np.in1d(allSurfIdx, indB, assume_unique=True)
                self.surfIdxB[ptSetName] = allSurfIdx[np.nonzero(mask)]

                # initialize the bool dict for this pointset
                self.curveProjFlag[ptSetName] = {}
                self.curvePtCounts[ptSetName] = {}
                self.curvePtCoords[ptSetName] = {}
                self.nCurvePts[ptSetName] = {}

                # we need to figure out if we have any points mapped to curves on comp A
                for curveName in allCurves:
                    # get the indices mapped to this curve, on this proc
                    idxs = self.curveProjIdx[ptSetName][curveName]

                    # call the utility function
                    nPtsTotal, nPtsProcs, curvePtCoords = self._commCurveProj(pts, idxs, comm)

                    # save the displacements and points
                    self.curvePtCounts[ptSetName][curveName] = nPtsProcs
                    self.curvePtCoords[ptSetName][curveName] = curvePtCoords

                    # also save the total number for convenience
                    self.nCurvePts[ptSetName][curveName] = nPtsTotal

    def update(self, ptSetName, delta):
        """Update the delta in ptSetName with our correction. The delta need
        to be supplied as we will be changing it and returning them
        """

        # original coordinates of the added pointset
        pts = self.points[ptSetName][0]
        # indices of the points that get affected by this intersection
        indices = self.points[ptSetName][1]
        # factors for each node in pointSet
        factors = self.points[ptSetName][2]

        # coordinates for the remeshed curves
        # we use the initial seam coordinates here
        coor = self.seam0
        # bar connectivity for the remeshed elements
        conn = self.seamConn
        # deltas for each point (nNode, 3) in size
        if self.seam.shape == self.seam0.shape:
            dr = self.seam - self.seam0
        else:
            # The topology has changed so we do not update the intersection
            # This will most likely break the mesh but allows
            # 1) the mesh to be output for visualization
            # 2) the optimization to continue after raising a fail flag
            if self.comm.rank == 0:
                print("The intersection topology has changed. The intersection will not be updated.")
            return delta

        # Get the two end points for the line elements
        r0 = coor[conn[:, 0]]
        r1 = coor[conn[:, 1]]

        # Get the deltas for two end points
        dr0 = dr[conn[:, 0]]
        dr1 = dr[conn[:, 1]]

        # Compute the lengths of each element in each coordinate direction
        length_x = r1[:, 0] - r0[:, 0]
        length_y = r1[:, 1] - r0[:, 1]
        length_z = r1[:, 2] - r0[:, 2]

        # Compute the 'a' coefficient
        a = (length_x) ** 2 + (length_y) ** 2 + (length_z) ** 2

        # Compute the total length of each element
        length = np.sqrt(a)

        # loop over the points that get affected
        for i in range(len(factors)):
            # j is the index of the point in the full set we are working with.
            j = indices[i]

            # coordinates of the original point
            rp = pts[j]

            # Run vectorized weighted interpolation

            # Compute the distances from the point being updated to the first end point of each element
            # The distances are scaled by the user-specified anisotropy in each direction
            dist_x = (r0[:, 0] - rp[0]) * self.anisotropy[0]
            dist_y = (r0[:, 1] - rp[1]) * self.anisotropy[1]
            dist_z = (r0[:, 2] - rp[2]) * self.anisotropy[2]

            # Compute b and c coefficients
            b = 2 * (length_x * dist_x + length_y * dist_y + length_z * dist_z)
            c = dist_x**2 + dist_y**2 + dist_z**2

            # Compute some recurring terms

            # The discriminant can be zero or negative, but it CANNOT be positive
            # This is because the quadratic that defines the distance from the line cannot have two roots
            # If the point is on the line, the quadratic will have a single root
            disc = b * b - 4 * a * c

            # Clip a + b + c might because it might be negative 1e-20 or so
            # Analytically, it cannot be negative
            sabc = np.sqrt(np.maximum(a + b + c, 0.0))
            sc = np.sqrt(c)

            # Compute denominators for the integral evaluations
            # We clip these values so that they are at max -eps to prevent them from getting a value of zero.
            # disc <= 0, sabc and sc >= 0, therefore the den1 and den2 should be <=0.
            # The clipping forces these terms to be <= -eps
            den1 = np.minimum(disc * sabc, -self.eps)
            den2 = np.minimum(disc * sc, -self.eps)

            # integral evaluations
            eval1 = (-2 * (2 * a + b) / den1 + 2 * b / den2) * length
            eval2 = ((2 * b + 4 * c) / den1 - 4 * c / den2) * length

            # denominator only gets one integral
            den = np.sum(eval1)

            # do each direction separately
            interp = np.zeros(3, dtype=self.dtype)
            for iDim in range(3):
                # numerator gets two integrals with the delta components
                num = np.sum((dr1[:, iDim] - dr0[:, iDim]) * eval2 + dr0[:, iDim] * eval1)
                # final result
                interp[iDim] = num / den

            # Now the delta is replaced by 1-factor times the weighted
            # interp of the seam * factor of the original:
            delta[j] = factors[i] * delta[j] + (1 - factors[i]) * interp

        return delta

    def sens(self, dIdPt, ptSetName, comm):
        # Return the reverse accumulation of dIdpt on the seam
        # nodes. Also modifies the dIdp array accordingly.

        # original coordinates of the added pointset
        pts = self.points[ptSetName][0]
        # indices of the points that get affected by this intersection
        indices = self.points[ptSetName][1]
        # factors for each node in pointSet
        factors = self.points[ptSetName][2]

        # coordinates for the remeshed curves
        # we use the initial seam coordinates here
        coor = self.seam0
        # bar connectivity for the remeshed elements
        conn = self.seamConn

        # Get the two end points for the line elements
        r0 = coor[conn[:, 0]]
        r1 = coor[conn[:, 1]]

        # Compute the lengths of each element in each coordinate direction
        length_x = r1[:, 0] - r0[:, 0]
        length_y = r1[:, 1] - r0[:, 1]
        length_z = r1[:, 2] - r0[:, 2]

        # Compute the 'a' coefficient
        a = (length_x) ** 2 + (length_y) ** 2 + (length_z) ** 2

        # Compute the total length of each element
        length = np.sqrt(a)

        # if we are handling more than one function,
        # seamBar will contain the seeds for each function separately
        seamBar = np.zeros((dIdPt.shape[0], self.seam0.shape[0], self.seam0.shape[1]))

        # if we have the projection flag, then we need to add the contribution to seamBar from that
        if self.projectFlag:
            seamBar += self.seamBarProj[ptSetName]

        for i in range(len(factors)):
            # j is the index of the point in the full set we are working with.
            j = indices[i]

            # coordinates of the original point
            rp = pts[j]

            # Compute the distances from the point being updated to the first end point of each element
            # The distances are scaled by the user-specified anisotropy in each direction
            dist_x = (r0[:, 0] - rp[0]) * self.anisotropy[0]
            dist_y = (r0[:, 1] - rp[1]) * self.anisotropy[1]
            dist_z = (r0[:, 2] - rp[2]) * self.anisotropy[2]

            # Compute b and c coefficients
            b = 2 * (length_x * dist_x + length_y * dist_y + length_z * dist_z)
            c = dist_x**2 + dist_y**2 + dist_z**2

            # Compute some reccurring terms
            disc = b * b - 4 * a * c
            sabc = np.sqrt(np.maximum(a + b + c, 0.0))
            sc = np.sqrt(c)

            # Compute denominators for the integral evaluations
            den1 = np.minimum(disc * sabc, -self.eps)
            den2 = np.minimum(disc * sc, -self.eps)

            # integral evaluations
            eval1 = (-2 * (2 * a + b) / den1 + 2 * b / den2) * length
            eval2 = ((2 * b + 4 * c) / den1 - 4 * c / den2) * length

            # denominator only gets one integral
            den = np.sum(eval1)

            evalDiff = eval1 - eval2

            for k in range(dIdPt.shape[0]):
                # This is the local seed (well the 3 seeds for the point)
                localVal = dIdPt[k, j, :] * (1 - factors[i])

                # Scale the dIdpt by the factor..dIdpt is input/output
                dIdPt[k, j, :] *= factors[i]

                # do each direction separately
                for iDim in range(3):
                    # seeds for the r0 point
                    seamBar[k, conn[:, 0], iDim] += localVal[iDim] * evalDiff / den

                    # seeds for the r1 point
                    seamBar[k, conn[:, 1], iDim] += localVal[iDim] * eval2 / den

        # seamBar is the bwd seeds for the intersection curve...
        # it is N,nseampt,3 in size
        # now call the reverse differentiated seam computation
        compSens = self._getIntersectionSeam_b(seamBar, comm)

        return compSens

    def project(self, ptSetName, newPts):
        # we need to build ADTs for both components if we have any components that lie on either
        # we also need to save ALL intermediate variables for gradient computations in reverse mode

        # get the comm for this point set
        comm = self.points[ptSetName][3]

        self.comm.Barrier()

        # check if we need to worry about either surface
        # we will use these flags to figure out if we need to do warping.
        # we need to do the comm for the updated curves regardless
        flagA = False
        flagB = False
        if len(self.curvesOnA) > 0:
            flagA = True
        if len(self.curvesOnB) > 0:
            flagB = True

        # do the pts on the intersection outside the loop
        nptsg = self.nCurvePts[ptSetName]["intersection"]

        # the deltas for these points are zero. they should already be on the intersection
        # also get the initial coordinates of these points. We use this during warping
        # intersection curve will be on both components
        if flagA:
            deltaA = np.zeros((nptsg, 3), dtype=self.dtype)
            curvePtCoordsA = self.curvePtCoords[ptSetName]["intersection"].copy()
        else:
            deltaA = np.zeros((0, 3), dtype=self.dtype)
            curvePtCoordsA = np.zeros((0, 3), dtype=self.dtype)

        if flagB:
            deltaB = np.zeros((nptsg, 3), dtype=self.dtype)
            curvePtCoordsB = self.curvePtCoords[ptSetName]["intersection"].copy()
        else:
            deltaB = np.zeros((0, 3), dtype=self.dtype)
            curvePtCoordsB = np.zeros((0, 3), dtype=self.dtype)

        # loop over the feature curves that we need to project
        for curveName in self.featureCurveNames:
            # get the indices of points we need to project
            idx = self.curveProjIdx[ptSetName][curveName]

            # these are the updated coordinates that will be projected to the curve
            ptsOnCurve = newPts[idx, :].copy()

            if self.debug:
                tecplot_interface.write_tecplot_scatter(
                    f"{curveName}_warped_pts.plt", "intersection", ["X", "Y", "Z"], ptsOnCurve
                )

            # conn of the current curve
            seamBeg = self.seamBeg[curveName]
            seamEnd = self.seamEnd[curveName]
            curveConn = self.seamConn[seamBeg:seamEnd]

            # Project these to the combined curves using pySurf
            # Get number of points
            nPoints = ptsOnCurve.shape[0]

            if self.debug:
                print(f"[{self.comm.rank}] curveName: {curveName}, nPoints on the fwd pass: {nPoints}")

            # Initialize references if user provided none
            dist2 = np.ones(nPoints, dtype=self.dtype) * 1e10
            xyzProj = np.zeros((nPoints, 3), dtype=self.dtype)
            tanProj = np.zeros((nPoints, 3), dtype=self.dtype)
            elemIDs = np.zeros((nPoints), dtype="int32")

            # only call the Fortran code if we have at least one point
            if nPoints > 0:
                # This will modify xyzProj, tanProj, dist2, and elemIDs if we find better projections than dist2.
                # Remember that we should adjust some indices before calling the Fortran code
                # Remember to use [:] to don't lose the pointer (elemIDs is an input/output variable)
                elemIDs[:] = (
                    elemIDs + 1
                )  # (we need to do this separetely because Fortran will actively change elemIDs contents.
                curveMask = self.curveSearchAPI.mindistancecurve(
                    ptsOnCurve.T, self.seam.T, curveConn.T + 1, xyzProj.T, tanProj.T, dist2, elemIDs
                )

                # Adjust indices back to Python standards
                elemIDs[:] = elemIDs - 1

                # we only have the curvemask if we do the projection on this proc
                self.curveProjData[ptSetName][curveName]["curveMask"] = curveMask

            # save some information for gradient comp
            self.curveProjData[ptSetName][curveName]["xyz"] = ptsOnCurve.copy()
            self.curveProjData[ptSetName][curveName]["coor"] = self.seam.copy()
            self.curveProjData[ptSetName][curveName]["barsConn"] = curveConn.copy()
            self.curveProjData[ptSetName][curveName]["xyzProj"] = xyzProj.copy()
            self.curveProjData[ptSetName][curveName]["tanProj"] = tanProj.copy()
            self.curveProjData[ptSetName][curveName]["elemIDs"] = elemIDs.copy()

            # get the delta for the points on this proc
            deltaLocal = xyzProj - ptsOnCurve

            if self.debug:
                tecplot_interface.write_tecplot_scatter(
                    f"{curveName}_projected_pts.plt", curveName, ["X", "Y", "Z"], xyzProj
                )

            # update the point coordinates on this processor.
            # we do not need to do any communication for this
            # since newPts is the final coordinates of the points we just projected
            newPts[idx] = xyzProj

            # communicate the deltas
            if comm:
                sizes = self.curvePtCounts[ptSetName][curveName]
                disp = np.array([np.sum(sizes[:i]) for i in range(comm.size)], dtype="intc")

                # save these for grad comp
                self.curveProjData[ptSetName][curveName]["sizes"] = sizes
                self.curveProjData[ptSetName][curveName]["disp"] = disp

                # sendbuf
                deltaLocal = deltaLocal.flatten()
                sendbuf = [deltaLocal, sizes[comm.rank] * 3]

                # recvbuf
                nptsg = self.nCurvePts[ptSetName][curveName]
                deltaGlobal = np.zeros(nptsg * 3, dtype=self.dtype)

                recvbuf = [deltaGlobal, sizes * 3, disp * 3, self.mpi_type]

                # do an allgatherv
                comm.Allgatherv(sendbuf, recvbuf)

                # reshape into a nptsg,3 array
                deltaGlobal = deltaGlobal.reshape((nptsg, 3))

            else:
                # we dont have a comm, so this is a "serial" pointset
                deltaGlobal = deltaLocal

                # also save the sizes and disp stuff as if we have one proc
                self.curveProjData[ptSetName][curveName]["sizes"] = self.curvePtCounts[ptSetName][curveName]
                self.curveProjData[ptSetName][curveName]["disp"] = [0]

            # we only add the deltaLocal to deltaA if this curve is on compA,
            # and we have points on compA surface
            if curveName in self.curvesOnA and flagA:
                # stack the deltas
                deltaA = np.vstack((deltaA, deltaGlobal))

                # also stack the original coordinates for warping
                curvePtCoordsNew = self.curvePtCoords[ptSetName][curveName]
                curvePtCoordsA = np.vstack((curvePtCoordsA, curvePtCoordsNew))

            # do the same for compB
            # we can use elif bec. one curve cannot be on both comps
            elif curveName in self.curvesOnB and flagB:
                # stack the deltas
                deltaB = np.vstack((deltaB, deltaGlobal))

                # also stack the original coordinates for warping
                curvePtCoordsNew = self.curvePtCoords[ptSetName][curveName]
                curvePtCoordsB = np.vstack((curvePtCoordsB, curvePtCoordsNew))

        self.comm.Barrier()

        # then, we warp all of the nodes that were affected by the intersection treatment
        # using the deltas from the previous project to curve step

        if flagA:
            self._warpSurfPts(self.points[ptSetName][0], newPts, self.surfIdxA[ptSetName], curvePtCoordsA, deltaA)

        if flagB:
            self._warpSurfPts(self.points[ptSetName][0], newPts, self.surfIdxB[ptSetName], curvePtCoordsB, deltaB)

        # save some info for the sens. computations
        self.curveProjData[ptSetName]["curvePtCoordsA"] = curvePtCoordsA
        self.curveProjData[ptSetName]["curvePtCoordsB"] = curvePtCoordsB

        # get the flags for components
        flagA = self.projData[ptSetName]["compA"]["flag"]
        flagB = self.projData[ptSetName]["compB"]["flag"]

        # Initialize component-wide projection indices
        indAComp = self.projData[ptSetName]["compA"]["ind"].copy()
        indBComp = self.projData[ptSetName]["compB"]["ind"].copy()

        # call the actual driver with the info to prevent code multiplication
        if flagA:
            # First project points on the tracked surfaces
            surfaceIndA = self.projData[ptSetName]["compA"]["surfaceInd"]
            for surface in surfaceIndA:
                surfaceInd = surfaceIndA[surface]

                # Get the subset of indices that is associated with this surface
                indA = [self.projData[ptSetName]["compA"]["ind"][i] for i in surfaceInd]

                # Remove these points from the component-wide projection indices
                for ind in indA:
                    indAComp.remove(ind)

                # get the points using the mapping
                ptsA = newPts[indA]
                # call the projection routine with the info
                # this returns the projected points and we use the same mapping to put them back in place
                newPts[indA] = self._projectToComponent(
                    ptsA, self.compA, self.projData[ptSetName][surface], surface=surface
                )

            # Project remaining points to the component as a whole
            if indAComp:
                ptsA = newPts[indAComp]
                newPts[indAComp] = self._projectToComponent(ptsA, self.compA, self.projData[ptSetName]["compA"])

        # do the same for B
        if flagB:
            surfaceIndB = self.projData[ptSetName]["compB"]["surfaceInd"]
            for surface in surfaceIndB:
                surfaceInd = surfaceIndB[surface]
                indB = [self.projData[ptSetName]["compB"]["ind"][i] for i in surfaceInd]
                for ind in indB:
                    indBComp.remove(ind)
                ptsB = newPts[indB]
                newPts[indB] = self._projectToComponent(
                    ptsB, self.compB, self.projData[ptSetName][surface], surface=surface
                )

            if indBComp:
                ptsB = newPts[indBComp]
                newPts[indBComp] = self._projectToComponent(ptsB, self.compB, self.projData[ptSetName]["compB"])

        # Store component-wide indices for derivative computation
        self.projData[ptSetName]["compA"]["indAComp"] = indAComp
        self.projData[ptSetName]["compB"]["indBComp"] = indBComp

    def project_b(self, ptSetName, dIdpt, comm):
        # call the functions to propagate ad seeds bwd
        # we need to build ADTs for both components if we have any components that lie on either
        # we also need to save ALL intermediate variables for gradient computations in reverse mode

        # number of functions we have
        N = dIdpt.shape[0]

        # get the flags for components
        flagA = self.projData[ptSetName]["compA"]["flag"]
        flagB = self.projData[ptSetName]["compB"]["flag"]

        # Initialize dictionaries to accumulate triangulated mesh sensitivities
        compSens_local = {}
        compSensA = {}
        compSensB = {}

        # call the actual driver with the info to prevent code multiplication
        if flagA:
            # Project remaining points to the component as a whole
            indAComp = self.projData[ptSetName]["compA"]["indAComp"]
            if indAComp:
                dIdptA = dIdpt[:, indAComp]
                dIdpt[:, indAComp], compSensA = self._projectToComponent_b(
                    dIdptA, self.compA, self.projData[ptSetName]["compA"]
                )

            # First project points on the tracked surfaces
            surfaceIndA = self.projData[ptSetName]["compA"]["surfaceInd"]
            for surface in surfaceIndA:
                surfaceInd = surfaceIndA[surface]

                # Get the subset of indices that is associated with this surface
                indA = [self.projData[ptSetName]["compA"]["ind"][i] for i in surfaceInd]

                # get the points using the mapping
                dIdptA = dIdpt[:, indA]
                # call the projection routine with the info
                # this returns the projected points and we use the same mapping to put them back in place
                dIdpt[:, indA], compSensA_temp = self._projectToComponent_b(
                    dIdptA, self.compA, self.projData[ptSetName][surface], surface=surface
                )

                # Accumulate triangulated mesh sensitivities
                for k, v in compSensA_temp.items():
                    try:
                        compSensA[k] += v
                    except KeyError:
                        compSensA[k] = v

            for k, v in compSensA.items():
                compSens_local[k] = v

        # set the compSens entries to all zeros on these procs
        else:
            # get the values from each DVGeo
            xA = self.compA.DVGeo.getValues()

            # loop over each entry in xA and xB and create a dummy zero gradient array for all
            for k, v in xA.items():
                # create the zero array:
                zeroSens = np.zeros((N, v.shape[0]))
                compSens_local[k] = zeroSens

        # do the same for B
        if flagB:
            indBComp = self.projData[ptSetName]["compB"]["indBComp"]
            if indBComp:
                dIdptB = dIdpt[:, indBComp]
                dIdpt[:, indBComp], compSensB = self._projectToComponent_b(
                    dIdptB, self.compB, self.projData[ptSetName]["compB"]
                )

            surfaceIndB = self.projData[ptSetName]["compB"]["surfaceInd"]
            for surface in surfaceIndB:
                surfaceInd = surfaceIndB[surface]
                indB = [self.projData[ptSetName]["compB"]["ind"][i] for i in surfaceInd]
                dIdptB = dIdpt[:, indB]
                dIdpt[:, indB], compSensB_temp = self._projectToComponent_b(
                    dIdptB, self.compB, self.projData[ptSetName][surface], surface=surface
                )

                for k, v in compSensB_temp.items():
                    try:
                        compSensB[k] += v
                    except KeyError:
                        compSensB[k] = v

            for k, v in compSensB.items():
                compSens_local[k] = v

        # set the compSens entries to all zeros on these procs
        else:
            # get the values from each DVGeo
            xB = self.compB.DVGeo.getValues()

            # loop over each entry in xA and xB and create a dummy zero gradient array for all
            for k, v in xB.items():
                # create the zero array:
                zeroSens = np.zeros((N, v.shape[0]))
                compSens_local[k] = zeroSens

        # finally sum the results across procs if we are provided with a comm
        if comm:
            compSens = {}
            # because the results are in a dictionary, we need to loop over the items and sum
            for k in compSens_local:
                compSens[k] = comm.allreduce(compSens_local[k], op=MPI.SUM)
        else:
            # we can just pass the dictionary
            compSens = compSens_local

        # now we do the warping

        # get the comm for this point set
        ptSetComm = self.points[ptSetName][3]

        # check if any processor did any warping on compA
        curvePtCoordsA = self.curveProjData[ptSetName]["curvePtCoordsA"]
        curvePtCoordsB = self.curveProjData[ptSetName]["curvePtCoordsB"]
        if ptSetComm:
            nCurvePtCoordsAG = ptSetComm.allreduce(len(curvePtCoordsA), op=MPI.MAX)
            nCurvePtCoordsBG = ptSetComm.allreduce(len(curvePtCoordsB), op=MPI.MAX)
        else:
            nCurvePtCoordsAG = len(curvePtCoordsA)
            nCurvePtCoordsBG = len(curvePtCoordsB)

        # check if we need to worry about either surface
        # we will use these flags to figure out if we need to do warping.
        # we need to do the comm for the updated curves regardless
        flagA = False
        flagB = False
        if len(self.curvesOnA) > 0:
            flagA = True
        if len(self.curvesOnB) > 0:
            flagB = True

        if ptSetComm:
            rank = ptSetComm.rank
        else:
            rank = 0

        # call the bwd warping routine
        # deltaA_b is the seed for the points projected to curves
        if flagA:
            deltaA_b_local = self._warpSurfPts_b(
                dIdpt, self.points[ptSetName][0], self.surfIdxA[ptSetName], curvePtCoordsA
            )
        else:
            deltaA_b_local = np.zeros((N, nCurvePtCoordsAG, 3))

        # do the same for comp B
        if flagB:
            deltaB_b_local = self._warpSurfPts_b(
                dIdpt, self.points[ptSetName][0], self.surfIdxB[ptSetName], curvePtCoordsB
            )
        else:
            deltaB_b_local = np.zeros((N, nCurvePtCoordsBG, 3))

        # reduce seeds for both
        if ptSetComm:
            deltaA_b = ptSetComm.allreduce(deltaA_b_local, op=MPI.SUM)
            deltaB_b = ptSetComm.allreduce(deltaB_b_local, op=MPI.SUM)
        # no comm, local is global
        else:
            deltaA_b = deltaA_b_local
            deltaB_b = deltaB_b_local

        # remove the seeds for the intersection, the disps are zero (constant) so no need to diff. those
        if flagA:
            # this just returns the seeds w/o the intersection seeds
            deltaA_b = deltaA_b[:, self.nCurvePts[ptSetName]["intersection"] :]
        if flagB:
            # this just returns the seeds w/o the intersection seeds
            deltaB_b = deltaB_b[:, self.nCurvePts[ptSetName]["intersection"] :]

        # loop over the curves
        for curveName in self.featureCurveNames:
            # sizes and displacements for this curve
            sizes = self.curveProjData[ptSetName][curveName]["sizes"]
            disp = self.curveProjData[ptSetName][curveName]["disp"]

            # we get the seeds from compA seeds
            if curveName in self.curvesOnA:
                if flagA:
                    # contribution on this proc
                    deltaBar = deltaA_b[:, disp[rank] : disp[rank] + sizes[rank]].copy()

                # this proc does not have any pts projected, so set the seed to zero
                else:
                    deltaBar = np.zeros((N, 0, 3))

                # remove the seeds for this curve from deltaA_b seeds
                deltaA_b = deltaA_b[:, disp[-1] + sizes[-1] :]

            # seeds from compB
            elif curveName in self.curvesOnB:
                if flagB:
                    # contribution on this proc
                    deltaBar = deltaB_b[:, disp[rank] : disp[rank] + sizes[rank]].copy()

                # this proc does not have any pts projected, so set the seed to zero
                else:
                    deltaBar = np.zeros((N, 0, 3))

                # remove the seeds for this curve from deltaA_b seeds
                deltaB_b = deltaB_b[:, disp[-1] + sizes[-1] :]

            else:
                print("This should not happen")

            # now we have the local seeds of projection points for all functions in xyzProjb

            # get the indices of points we need to project
            idx = self.curveProjIdx[ptSetName][curveName]
            nPoints = len(idx)

            # get some data from the fwd run
            xyz = self.curveProjData[ptSetName][curveName]["xyz"]
            coor = self.curveProjData[ptSetName][curveName]["coor"]
            barsConn = self.curveProjData[ptSetName][curveName]["barsConn"]
            xyzProj = self.curveProjData[ptSetName][curveName]["xyzProj"]
            tanProj = self.curveProjData[ptSetName][curveName]["tanProj"]
            elemIDs = self.curveProjData[ptSetName][curveName]["elemIDs"]

            # we dont use tangents so seeds are zero
            tanProjb = np.zeros_like(tanProj)

            if nPoints > 0:
                # also get the curveMask
                curveMask = self.curveProjData[ptSetName][curveName]["curveMask"]

                # run the bwd projection for everyfunction
                for k in range(N):
                    # contribution from delta
                    xyzProjb = deltaBar[k].copy()

                    # add the contribution from dIdpt for the idx points themselves
                    xyzProjb += dIdpt[k, idx]

                    # Call Fortran code (This will accumulate seeds in xyzb and self.coorb)
                    xyzb_new, coorb_new = self.curveSearchAPI.mindistancecurve_b(
                        xyz.T,
                        coor.T,
                        barsConn.T + 1,
                        xyzProj.T,
                        xyzProjb.T,
                        tanProj.T,
                        tanProjb.T,
                        elemIDs + 1,
                        curveMask,
                    )

                    # Accumulate derivatives with the correct k

                    # replace the seed in dIdpt. we subtract deltaBar here
                    # because delta was equal to the projected points minus the original points.
                    # So the delta seeds contribute with a negative sign
                    # to the seeds of the points before projection
                    dIdpt[k, idx, :] = xyzb_new.T - deltaBar[k]

                    # add the seed to the seam seed
                    self.seamBarProj[ptSetName][k, :, :] += coorb_new.T

        return compSens

    def _commCurveProj(self, pts, indices, comm):
        """
        This function will get the points, indices, and comm.
        This function is called once for each feature curve.
        The indices are the indices of points that was mapped to this curve.
        We compute how many points we have mapped to this curve globally.
        Furthermore, we compute the displacements.
        Finally, we communicate the initial coordinates of these points.
        These will later be used in the point-based warping.

        """

        # only do this fancy stuff if this is a "parallel" pointset
        if comm:
            nproc = comm.size

            # communicate the counts
            sizes = np.array(comm.allgather(len(indices)), dtype="intc")

            # total number of points
            nptsg = np.sum(sizes)

            # get the displacements
            disp = np.array([np.sum(sizes[:i]) for i in range(nproc)], dtype="intc")

            # sendbuf
            ptsLocal = pts[indices].flatten()
            sendbuf = [ptsLocal, len(indices) * 3]

            # recvbuf
            ptsGlobal = np.zeros(3 * nptsg, dtype=self.dtype)

            recvbuf = [ptsGlobal, sizes * 3, disp * 3, self.mpi_type]

            # do an allgatherv
            comm.Allgatherv(sendbuf, recvbuf)

            # reshape into a nptsg,3 array
            curvePtCoords = ptsGlobal.reshape((nptsg, 3))

        # this is a "serial" pointset, so the results are just local
        else:
            nptsg = len(indices)
            sizes = [nptsg]
            curvePtCoords = pts[indices]

        return nptsg, sizes, curvePtCoords

    def _warpSurfPts(self, pts0, ptsNew, indices, curvePtCoords, delta):
        """
        This function warps points using the displacements from curve projections.

        pts0: The original surface point coordinates.
        ptsNew: Updated surface pt coordinates. We will add the warped delta to these inplace.
        indices: Indices of the points that we will use for this operation.
        curvePtCoords: Original coordinates of points on curves.
        delta: Displacements of the points on curves after projecting them.

        """

        # Return if curvePtCoords is empty
        if not np.any(curvePtCoords):
            return

        for j in indices:
            # point coordinates with the baseline design
            # this is the point we will warp
            ptCoords = pts0[j]

            # Vectorized point-based warping
            rr = ptCoords - curvePtCoords
            LdefoDist = 1.0 / np.sqrt(rr[:, 0] ** 2 + rr[:, 1] ** 2 + rr[:, 2] ** 2 + 1e-16)
            LdefoDist3 = LdefoDist**3
            Wi = LdefoDist3
            den = np.sum(Wi)
            interp = np.zeros(3, dtype=self.dtype)
            for iDim in range(3):
                interp[iDim] = np.sum(Wi * delta[:, iDim]) / den

            # finally, update the coord in place
            ptsNew[j] = ptsNew[j] + interp

    def _warpSurfPts_b(self, dIdPt, pts0, indices, curvePtCoords):
        # seeds for delta
        deltaBar = np.zeros((dIdPt.shape[0], curvePtCoords.shape[0], 3))

        # Return zeros if curvePtCoords is empty
        if not np.any(curvePtCoords):
            return deltaBar

        for k in range(dIdPt.shape[0]):
            for j in indices:
                # point coordinates with the baseline design
                # this is the point we will warp
                ptCoords = pts0[j]

                # local seed for 3 coords
                localVal = dIdPt[k, j]

                # Vectorized point-based warping
                rr = ptCoords - curvePtCoords
                LdefoDist = 1.0 / np.sqrt(rr[:, 0] ** 2 + rr[:, 1] ** 2 + rr[:, 2] ** 2 + 1e-16)
                LdefoDist3 = LdefoDist**3
                Wi = LdefoDist3
                den = np.sum(Wi)

                for iDim in range(3):
                    deltaBar[k, :, iDim] += Wi * localVal[iDim] / den

        # return the seeds for the delta vector
        return deltaBar

    def _projectToComponent(self, pts, comp, projDict, surface=None):
        # We build an ADT for this component using pySurf
        # Set bounding box for new tree
        BBox = np.zeros((2, 3))
        useBBox = False

        # dummy connectivity data for quad elements since we have all tris
        quadConn = np.zeros((0, 4))

        if surface is not None:
            # Use the triConn for just this surface
            triConn = comp.triConn[surface]
            # Set the adtID as the surface name
            adtID = surface
        else:
            # Use the stacked triConn for the whole component
            triConn = comp.triConnStack
            # Set the adtID as the component name
            adtID = comp.name

        # Compute set of nodal normals by taking the average normal of all
        # elements surrounding the node. This allows the meshing algorithms,
        # for instance, to march in an average direction near kinks.
        nodal_normals = self.adtAPI.adtcomputenodalnormals(comp.nodes.T, triConn.T, quadConn.T)
        comp.nodal_normals = nodal_normals.T

        # Create new tree (the tree itself is stored in Fortran level)
        self.adtAPI.adtbuildsurfaceadt(
            comp.nodes.T, triConn.T, quadConn.T, BBox.T, useBBox, MPI.COMM_SELF.py2f(), adtID
        )

        # project
        numPts = pts.shape[0]
        dist2 = np.ones(numPts, dtype=self.dtype) * 1e10
        xyzProj = np.zeros((numPts, 3), dtype=self.dtype)
        normProjNotNorm = np.zeros((numPts, 3), dtype=self.dtype)

        if self.debug:
            print(f"[{self.comm.rank}] Projecting to component {comp.name}, pts.shape = {pts.shape}")

        # Call projection function
        procID, elementType, elementID, uvw = self.adtAPI.adtmindistancesearch(
            pts.T, adtID, dist2, xyzProj.T, comp.nodal_normals.T, normProjNotNorm.T
        )

        # Adjust indices and ordering
        elementID = elementID - 1
        uvw = uvw.T

        # normalize the normals
        normProj = tsurf_tools.normalize(normProjNotNorm)

        # deallocate ADT
        self.adtAPI.adtdeallocateadts(adtID)

        # save the data
        projDict["procID"] = procID.copy()
        projDict["elementType"] = elementType.copy()
        projDict["elementID"] = elementID.copy()
        projDict["uvw"] = uvw.copy()
        projDict["dist2"] = dist2.copy()
        projDict["normProjNotNorm"] = normProjNotNorm.copy()
        projDict["normProj"] = normProj.copy()

        # also save the original and projected points
        projDict["xyz"] = pts.copy()
        projDict["xyzProj"] = xyzProj.copy()

        # return projected points
        return xyzProj

    def _projectToComponent_b(self, dIdpt, comp, projDict, surface=None):
        # We build an ADT for this component using pySurf
        # Set bounding box for new tree
        BBox = np.zeros((2, 3))
        useBBox = False

        # dummy connectivity data for quad elements since we have all tris
        quadConn = np.zeros((0, 4))

        if surface is not None:
            # Use the triConn for just this surface
            triConn = comp.triConn[surface]
            # Set the adtID as the surface name
            adtID = surface
        else:
            # Use the stacked triConn for the whole component
            triConn = comp.triConnStack
            # Set the adtID as the component name
            adtID = comp.name

        # Compute set of nodal normals by taking the average normal of all
        # elements surrounding the node. This allows the meshing algorithms,
        # for instance, to march in an average direction near kinks.
        nodal_normals = self.adtAPI.adtcomputenodalnormals(comp.nodes.T, triConn.T, quadConn.T)
        comp.nodal_normals = nodal_normals.T

        # Create new tree (the tree itself is stored in Fortran level)
        self.adtAPI.adtbuildsurfaceadt(
            comp.nodes.T, triConn.T, quadConn.T, BBox.T, useBBox, MPI.COMM_SELF.py2f(), adtID
        )

        # also extract the projection data we have from the fwd pass
        procID = projDict["procID"]
        elementType = projDict["elementType"]
        elementID = projDict["elementID"]
        uvw = projDict["uvw"]
        dist2 = projDict["dist2"]
        normProjNotNorm = projDict["normProjNotNorm"]

        # get the original and projected points too
        xyz = projDict["xyz"]
        xyzProj = projDict["xyzProj"]

        # also, we dont care about the normals, so the AD seeds for them should (?) be zero
        normProjb = np.zeros_like(normProjNotNorm)

        # also create the dIdtp for the triangulated surface nodes
        dIdptComp = np.zeros((dIdpt.shape[0], comp.nodes.shape[0], 3))

        # now propagate the ad seeds back for each function
        for i in range(dIdpt.shape[0]):
            # the derivative seeds for the projected points
            xyzProjb = dIdpt[i].copy()

            # Compute derivatives of the normalization process
            normProjNotNormb = tsurf_tools.normalize_b(normProjNotNorm, normProjb)

            # Call projection function
            # ATTENTION: The variable "xyz" here in Python corresponds to the variable "coor" in the Fortran code.
            # On the other hand, the variable "coor" here in Python corresponds to the variable "adtCoor" in Fortran.
            # I could not change this because the original ADT code already used "coor" to denote nodes that should be
            # projected.

            xyzb, coorb, nodal_normalsb = self.adtAPI.adtmindistancesearch_b(
                xyz.T,
                adtID,
                procID,
                elementType,
                elementID + 1,
                uvw.T,
                dist2,
                xyzProj.T,
                xyzProjb.T,
                comp.nodal_normals.T,
                normProjNotNorm.T,
                normProjNotNormb.T,
            )

            # Transpose results to make them consistent
            xyzb = xyzb.T
            coorb = coorb.T

            # Put the reverse ad seed back into dIdpt
            dIdpt[i] = xyzb
            # Also save the triangulated surface node seeds
            dIdptComp[i] = coorb

        # Now we are done with the ADT
        self.adtAPI.adtdeallocateadts(adtID)

        # Call the total sensitivity of the component's DVGeo
        compSens = comp.DVGeo.totalSensitivity(dIdptComp, "triMesh")

        # the entries in dIdpt is replaced with AD seeds of initial points that were projected
        # we also return the total sensitivity contributions from components' triMeshes
        return dIdpt, compSens

    def _getUpdatedCoords(self):
        # this code returns the updated coordinates

        # first comp a
        self.compA.updateTriMesh()

        # then comp b
        self.compB.updateTriMesh()

        return

    def _getIntersectionSeam(self, comm, firstCall=False):
        # we can parallelize here. each proc gets one intersection, but needs re-structuring of some of the code.

        # this function computes the intersection curve, cleans up the data and splits the curve based on features or curves specified by the user.

        # create the dictionary to save all intermediate variables for reverse differentiation
        self.seamDict = {}

        # Call pySurf with the quad information.
        dummyConn = np.zeros((0, 4))

        # compute the intersection curve, in the first step we just get the array sizes to hide allocatable arrays from python
        arraySizes = self.intersectionAPI.computeintersection(
            self.compA.nodes.T,
            self.compA.triConnStack.T,
            dummyConn.T,
            self.compB.nodes.T,
            self.compB.triConnStack.T,
            dummyConn.T,
            self.distTol,
            comm.py2f(),
        )

        # Retrieve results from Fortran if we have an intersection
        if np.max(arraySizes[1:]) > 0:
            # Second Fortran call to retrieve data from the CGNS file.
            intersectionArrays = self.intersectionAPI.retrievedata(*arraySizes)

            # We need to do actual copies, otherwise data will be overwritten if we compute another intersection.
            # We subtract one to make indices consistent with the Python 0-based indices.
            # We also need to transpose it since Python and Fortran use different orderings to store matrices in memory.

            intNodes = np.array(intersectionArrays[0]).T
            # The last entry of intNodes is always 0,0,0 for some reason, even for a proper intersection
            # This is probably a bug in pySurf
            # It has no effect here because the computation relies on connectivity, not individual points
            barsConn = np.array(intersectionArrays[1]).T - 1
            parentTria = np.array(intersectionArrays[2]).T - 1

            # Save these intermediate variables
            self.seamDict["barsConn"] = barsConn
            self.seamDict["parentTria"] = parentTria

        else:
            raise Error(f"The components {self.compA.name} and {self.compB.name} do not intersect.")

        # Release memory used by Fortran
        self.intersectionAPI.releasememory()

        # Sort the output
        newConn, newMap = tsurf_tools.FEsort(barsConn.tolist())

        # newConn might have multiple intersection curves
        if len(newConn) == 1:
            # we have a single intersection curve, just take this.
            seamConn = newConn[0].copy()
        # We have multiple intersection curves
        else:
            if self.intDir is None:
                # we have multiple intersection curves but the user did not specify which direction to pick
                for i in range(len(newConn)):
                    curvename = f"{self.compA.name}_{self.compB.name}_{i}"
                    tecplot_interface.writeTecplotFEdata(intNodes, newConn[i], curvename, curvename)
                raise Error(
                    f"More than one intersection curve between comps {self.compA.name} and {self.compB.name}. "
                    + "The curves are written as Tecplot files in the current directory. "
                    + "Try rerunning after specifying intDir for the intersection."
                )

            # the user did specify which direction to pick
            else:
                int_centers = np.zeros(len(newConn), dtype=self.dtype)
                # we will figure out the locations of these points and pick the one closer to the user picked direction
                for i in range(len(newConn)):
                    # get all the points
                    int_pts = intNodes[newConn[i]][:, 0]

                    # average the values
                    # the API uses a 1 based indexing, but here, we convert to a zero based indexing
                    int_centers[i] = np.average(int_pts[abs(self.intDir) - 1])

                # multiply the values with the sign of intDir
                int_centers *= np.sign(self.intDir)

                # get the argmax
                int_index = np.argmax(int_centers)

                # this is the intersection seam
                seamConn = newConn[int_index].copy()

        # Get the number of elements
        nElem = seamConn.shape[0]

        # now that we have a continuous, ordered seam connectivity in seamConn, we can try to detect features

        # we need to track the nodes that are closest to the supplied feature curves
        breakList = []
        curveBeg = {}
        curveBegCoor = {}

        # loop over the feature curves
        for curveName in self.featureCurveNames:
            # we need to initialize the dictionary here
            # to get the intermediate output from mindistancecurve call
            self.seamDict[curveName] = {}

            # if this curve is on compB, we use it to track intersection features
            if curveName in self.compB.barsConn and curveName not in self.remeshAll:
                # get the curve connectivity
                curveConn = self.compB.barsConn[curveName]

                # Use pySurf to project the point on curve
                # First, we need to get a list of nodes that define the intersection
                intNodesOrd = intNodes[seamConn[:, 0]]

                # Get number of points
                nPoints = len(intNodesOrd)

                # Initialize references
                dist2 = np.ones(nPoints, dtype=self.dtype) * 1e10
                xyzProj = np.zeros((nPoints, 3), dtype=self.dtype)
                tanProj = np.zeros((nPoints, 3), dtype=self.dtype)
                elemIDs = np.zeros((nPoints), dtype="int32")

                # then find the closest point to the curve

                # only call the Fortran code if we have at least one point
                # this is redundant but it is how its done in pySurf
                if nPoints > 0:
                    # This will modify xyzProj, tanProj, dist2, and elemIDs if we find better projections than dist2.
                    # Remember that we should adjust some indices before calling the Fortran code
                    # Remember to use [:] to don't lose the pointer (elemIDs is an input/output variable)
                    elemIDs[:] = (
                        elemIDs + 1
                    )  # (we need to do this separetely because Fortran will actively change elemIDs contents.
                    curveMask = self.curveSearchAPI.mindistancecurve(
                        intNodesOrd.T, self.compB.nodes.T, curveConn.T + 1, xyzProj.T, tanProj.T, dist2, elemIDs
                    )

                    # Adjust indices back to Python standards
                    elemIDs[:] = elemIDs - 1

                    self.seamDict[curveName]["curveMask"] = curveMask.copy()
                    self.seamDict[curveName]["elemIDs"] = elemIDs.copy()
                    self.seamDict[curveName]["intNodesOrd"] = intNodesOrd.copy()
                    self.seamDict[curveName]["xyzProj"] = xyzProj.copy()
                    self.seamDict[curveName]["tanProj"] = tanProj.copy()
                    self.seamDict[curveName]["dist2"] = dist2.copy()
                    self.seamDict[curveName]["projPtIndx"] = seamConn[:, 0][np.argmin(dist2)].copy()

                # now, find the index of the smallest distance
                breakList.append(np.argmin(dist2))

                # also get which element is the closest to the feature point
                curveBeg[curveName] = elemIDs[np.argmin(dist2)]

                # get which point on this element we projected to.
                curveBegCoor[curveName] = xyzProj[np.argmin(dist2)]

            else:
                # if it is not on compB, we still need to set up some variables so that we remesh the whole curve
                # set the beginning to the first element
                curveBeg[curveName] = 0

        # number of features we detected. This will be equal to the number of feature curves on compB
        nFeature = len(breakList)

        # if this is the first call,
        if firstCall:
            # we also save the initial curve with nodes and connectivities for distance calculations
            self.conn0 = seamConn.copy()
            self.nodes0 = intNodes.copy()
            self.nFeature = nFeature
        else:
            if nFeature != self.nFeature:
                raise Error("Number of features on the intersection curve has changed.")

        # flip
        # we want breakList to be in increasing order...
        ii = 0
        for i in range(nFeature):
            # we loop over the breaklist elements and check if the element index is going up or down
            if breakList[i] < breakList[np.mod(i + 1, nFeature)]:
                ii += 1

        # now check if we need to flip the curve
        if ii == 1:  # we need at least 2 features where the element number increases...
            # we need to reverse the order of our feature curves
            # and we will flip the elements too so keep track of this change
            breakList = np.mod(seamConn.shape[0] - np.array(breakList), seamConn.shape[0])

            # and we need to invert the curves themselves
            seamConn = np.flip(seamConn, axis=0)
            seamConn = np.flip(seamConn, axis=1)

        # roll so that the first breakList entry is the first node
        seamConn = np.roll(seamConn, -breakList[0], axis=0)

        # also adjust the feature indices
        breakList = np.mod(breakList - breakList[0], nElem)

        # get the number of elements between each feature
        curveSizes = []
        for i in range(nFeature - 1):
            curveSizes.append(np.mod(breakList[i + 1] - breakList[i], nElem))
        # check the last curve outside the loop
        curveSizes.append(np.mod(breakList[0] - breakList[-1], nElem))

        # copy the curveSizes for the first call
        if firstCall:
            self.nElems = curveSizes[:]

        # now loop over the curves between the feature nodes. We will remesh them separately to retain resolution between curve features, and just append the results since the features are already ordered
        curInd = 0
        seam = np.zeros((0, 3), dtype=self.dtype)
        finalConn = np.zeros((0, 2), dtype="int32")
        for i in range(nFeature):
            # just use the same number of points *2 for now
            nNewNodes = self.nElems[i] + 1
            coor = intNodes
            barsConn = seamConn[curInd : curInd + curveSizes[i]]
            curInd += curveSizes[i]
            method = "linear"
            spacing = "linear"
            initialSpacing = 0.1
            finalSpacing = 0.1

            # re-sample the curve (try linear for now), to get N number of nodes on it spaced linearly
            # Call Fortran code. Remember to adjust transposes and indices
            newCoor, newBarsConn = self.utilitiesAPI.remesh(
                nNewNodes, coor.T, barsConn.T + 1, method, spacing, initialSpacing, finalSpacing
            )
            newCoor = newCoor.T
            newBarsConn = newBarsConn.T - 1

            # add these n -resampled nodes back to back in seam and return a copy of the array
            # we don't need the connectivity info for now? we just need the coords
            # first increment the new connectivity by number of coordinates already in seam
            newBarsConn += len(seam)

            # now stack the nodes
            seam = np.vstack((seam, newCoor))

            # and the conn
            finalConn = np.vstack((finalConn, newBarsConn))

        if firstCall:
            # save the beginning and end indices of these elements
            self.seamBeg["intersection"] = 0
            self.seamEnd["intersection"] = len(finalConn)

        # save stuff to the dictionary for sensitivity computations...
        self.seamDict["intNodes"] = intNodes.copy()
        self.seamDict["seamConn"] = seamConn.copy()
        self.seamDict["curveSizes"] = curveSizes.copy()
        # size of the intersection seam w/o any feature curves
        self.seamDict["seamSize"] = len(seam)
        self.seamDict["curveBegCoor"] = curveBegCoor.copy()

        # Output the intersection curve
        if self.comm.rank == 0 and self.debug:
            curvename = f"{self.compA.name}_{self.compB.name}_{self.counter}"
            tecplot_interface.writeTecplotFEdata(intNodes, seamConn, curvename, curvename)

        # we need to re-mesh feature curves if the user wants...
        if self.incCurves:
            # we need to set up some variables
            if firstCall:
                self.nNodeFeature = {}
                self.distFeature = {}

            remeshedCurves = np.zeros((0, 3), dtype=self.dtype)
            remeshedCurveConn = np.zeros((0, 2), dtype="int32")

            # loop over each curve, figure out what nodes get re-meshed, re-mesh, and append to seam...
            for curveName in self.featureCurveNames:
                # figure out which comp owns this curve...
                if curveName in self.compB.barsConn:
                    curveComp = self.compB
                    dStarComp = self.dStarB
                elif curveName in self.compA.barsConn:
                    curveComp = self.compA
                    dStarComp = self.dStarA

                # connectivity for this curve.
                curveConn = curveComp.barsConn[curveName]

                # we already have the element that is closest to the intersection
                # or if this curve does not start from the intersection,
                # this is simply the first element
                elemBeg = curveBeg[curveName]

                # now lets split this element so that we get a better initial point...
                # this has to be on compB
                if curveName in curveBegCoor:
                    # save the original coordinate of the first point
                    ptBegSave = self.compB.nodes[curveConn[elemBeg, 0]].copy()
                    # and replace this with the starting point we want
                    self.compB.nodes[curveConn[elemBeg, 0]] = curveBegCoor[curveName].copy()

                # compute the element lengths starting from elemBeg
                firstNodes = curveComp.nodes[curveConn[elemBeg:, 0]]
                secondNodes = curveComp.nodes[curveConn[elemBeg:, 1]]
                diff = secondNodes - firstNodes
                dist2 = diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2
                elemDist = np.sqrt(dist2)

                # get the cumulative distance
                cumDist = np.cumsum(elemDist)

                # if no marchdir for this curve, we use all of it
                if curveName in self.remeshAll:
                    # we remesh all of this curve
                    elemBeg = 0
                    elemEnd = len(curveConn)
                    if firstCall:
                        self.nNodeFeature[curveName] = elemEnd + 1
                else:
                    # do the regular thing
                    if firstCall:
                        # compute the distances from curve nodes to intersection seam
                        curvePts = curveComp.nodes[curveConn[elemBeg:, 0]]

                        # Get number of points
                        nPoints = len(curvePts)

                        # Initialize references if user provided none
                        dist2 = np.ones(nPoints, dtype=self.dtype) * 1e10
                        xyzProj = np.zeros((nPoints, 3), dtype=self.dtype)
                        tanProj = np.zeros((nPoints, 3), dtype=self.dtype)
                        elemIDs = np.zeros((nPoints), dtype="int32")

                        # then find the closest point to the curve

                        # Only call the Fortran code if we have at least one point
                        if nPoints > 0:
                            # This will modify xyzProj, tanProj, dist2, and elemIDs if we find better projections than dist2.
                            # Remember that we should adjust some indices before calling the Fortran code
                            # Remember to use [:] to don't lose the pointer (elemIDs is an input/output variable)
                            elemIDs[:] = (
                                elemIDs + 1
                            )  # (we need to do this separetely because Fortran will actively change elemIDs contents.
                            curveMask = self.curveSearchAPI.mindistancecurve(
                                curvePts.T, self.nodes0.T, self.conn0.T + 1, xyzProj.T, tanProj.T, dist2, elemIDs
                            )

                        dNodes = np.sqrt(dist2)

                        # number of elements to use, subtract one to get the correct element count
                        nElem = (np.abs(dNodes - dStarComp * 1.3)).argmin() - 1

                        # we want to be one after the actual distance, so correct if needed
                        if dNodes[nElem] < dStarComp * 1.3:
                            nElem += 1

                        elemEnd = elemBeg + nElem

                        # get the total curve distance from elemBeg to this element.
                        distCurve = cumDist[nElem]

                        # save this distance as the remesh distance
                        self.distFeature[curveName] = distCurve

                        # also save how many nodes we have, we want 2 times this when re-meshing
                        self.nNodeFeature[curveName] = nElem + 1

                    else:
                        # figure out how many elements we need to go in this direction
                        elemEnd = (np.abs(cumDist - self.distFeature[curveName])).argmin() + elemBeg

                # get the new connectivity data between the initial and final elements
                curveConnTrim = curveConn[elemBeg : elemEnd + 1]

                # remesh the new connectivity curve, using nNode*2 times nodes
                nNewNodes = self.nNodeFeature[curveName]
                coor = curveComp.nodes
                barsConn = curveConnTrim
                method = "linear"
                spacing = "linear"
                initialSpacing = 0.1
                finalSpacing = 0.1

                # now re-sample the curve (try linear for now), to get N number of nodes on it spaced linearly
                # Call Fortran code. Remember to adjust transposes and indices
                newCoor, newBarsConn = self.utilitiesAPI.remesh(
                    nNewNodes, coor.T, barsConn.T + 1, method, spacing, initialSpacing, finalSpacing
                )
                newCoor = newCoor.T
                newBarsConn = newBarsConn.T - 1

                # increment the connectivitiy data
                newBarsConn += len(remeshedCurves)

                # append this new curve to the featureCurve data
                remeshedCurves = np.vstack((remeshedCurves, newCoor))
                remeshedCurveConn = np.vstack((remeshedCurveConn, newBarsConn))

                # number of new nodes added in the opposite direction
                nNewNodesReverse = 0
                if elemBeg > 0 and self.remeshBwd:
                    # also re-mesh the initial part of the curve, to prevent any negative volumes there
                    curveConnTrim = curveConn[:elemBeg]

                    nNewNodesReverse = self.nNodeFeature[curveName]
                    coor = self.compB.nodes
                    barsConn = curveConnTrim
                    method = "linear"
                    spacing = "linear"
                    initialSpacing = 0.1
                    finalSpacing = 0.1

                    # now re-sample the curve (try linear for now), to get N number of nodes on it spaced linearly
                    # Call Fortran code. Remember to adjust transposes and indices
                    newCoor, newBarsConn = self.utilitiesAPI.remesh(
                        nNewNodesReverse, coor.T, barsConn.T + 1, method, spacing, initialSpacing, finalSpacing
                    )
                    newCoor = newCoor.T
                    newBarsConn = newBarsConn.T - 1

                    newBarsConn = newBarsConn + len(remeshedCurves)

                    remeshedCurves = np.vstack((remeshedCurves, newCoor))
                    remeshedCurveConn = np.vstack((remeshedCurveConn, newBarsConn))

                if curveName in curveBegCoor:
                    # finally, put the modified initial and final points back in place.
                    self.compB.nodes[curveConn[elemBeg, 0]] = ptBegSave.copy()

                # save some info for gradient computations later on
                self.seamDict[curveName]["nNewNodes"] = nNewNodes
                self.seamDict[curveName]["nNewNodesReverse"] = nNewNodesReverse
                self.seamDict[curveName]["elemBeg"] = elemBeg
                self.seamDict[curveName]["elemEnd"] = elemEnd
                # this includes the initial coordinates of the points for each curve
                # that has a modified initial curve
                self.seamDict["curveBegCoor"] = curveBegCoor

                if firstCall:
                    # save the beginning and end indices of these elements
                    self.seamBeg[curveName] = (
                        len(finalConn) + len(remeshedCurveConn) - (nNewNodes + nNewNodesReverse) + 2
                    )
                    self.seamEnd[curveName] = len(finalConn) + len(remeshedCurveConn)

            # Output the feature curves
            if self.comm.rank == 0 and self.debug:
                curvename = f"featureCurves_{self.counter}"
                tecplot_interface.writeTecplotFEdata(remeshedCurves, remeshedCurveConn, curvename, curvename)

            # now we are done going over curves,
            # so we can append all the new curves to the "seam",
            # which now contains the intersection, and re-meshed feature curves

            # increment the conn from curves
            remeshedCurveConn += len(seam)
            # stack the nodes
            seam = np.vstack((seam, remeshedCurves))
            # stack the conn
            finalConn = np.vstack((finalConn, remeshedCurveConn))

        # save the connectivity
        self.seamConn = finalConn

        self.counter += 1

        return seam.copy()

    def _getIntersectionSeam_b(self, seamBar, comm):
        # seamBar contains all the bwd seeds for all coordinates in self.seam

        # seam bar has shape [N, nSeamPt, 3]
        # seeds for N functions
        N = seamBar.shape[0]
        # n points in total in the combined seam
        # 3 coordinates for each point

        # allocate the space for component coordinate seeds
        coorAb = np.zeros((N, self.compA.nodes.shape[0], self.compA.nodes.shape[1]))
        coorBb = np.zeros((N, self.compB.nodes.shape[0], self.compB.nodes.shape[1]))

        # first, extract the actual intersection coordinates from feature curves.
        # we might not have any feature curves but the intersection curve will be there
        seamSize = self.seamDict["seamSize"]

        # coordinates of the beginning points for feature curves on compB
        curveBegCoor = self.seamDict["curveBegCoor"]

        intBar = seamBar[:, :seamSize, :]
        curveBar = seamBar[:, seamSize:, :]

        # dictionary to save the accumulation of curve projection seeds
        curveProjb = {}

        # check if we included feature curves
        if self.incCurves:
            # offset for the derivative seeds for this curve
            iBeg = 0

            # loop over each curve
            for curveName in self.featureCurveNames:
                # get the fwd data
                curveDict = self.seamDict[curveName]
                nNewNodes = curveDict["nNewNodes"]
                elemBeg = curveDict["elemBeg"]
                elemEnd = curveDict["elemEnd"]

                # get the derivative seeds
                newCoorb = curveBar[:, iBeg : iBeg + nNewNodes, :].copy()
                iBeg += nNewNodes

                # figure out which comp owns this curve...
                if curveName in self.compB.barsConn:
                    curveComp = self.compB
                    coorb = coorBb

                elif curveName in self.compA.barsConn:
                    curveComp = self.compA
                    coorb = coorAb

                # connectivity for this curve.
                curveConn = curveComp.barsConn[curveName]

                # adjust the first coordinate of the curve
                if curveName in curveBegCoor:
                    # save the original coordinate of the first point
                    ptBegSave = self.compB.nodes[curveConn[elemBeg, 0]].copy()
                    # and replace this with the starting point we want
                    self.compB.nodes[curveConn[elemBeg, 0]] = curveBegCoor[curveName].copy()

                # get the coordinates of points
                coor = curveComp.nodes

                # connectivity for this curve.
                curveConn = curveComp.barsConn[curveName]

                # trim the connectivity data
                barsConn = curveComp.barsConn[curveName][elemBeg : elemEnd + 1]

                # constant inputs
                method = "linear"
                spacing = "linear"
                initialSpacing = 0.1
                finalSpacing = 0.1

                cb = np.zeros((N, coor.shape[0], coor.shape[1]))

                # loop over functions
                for ii in range(N):
                    # Call Fortran code. Remember to adjust transposes and indices
                    _, _, cbi = self.utilitiesAPI.remesh_b(
                        nNewNodes - 1,
                        coor.T,
                        newCoorb[ii].T,
                        barsConn.T + 1,
                        method,
                        spacing,
                        initialSpacing,
                        finalSpacing,
                    )
                    # derivative seeds for the coordinates.
                    cb[ii] = cbi.T.copy()

                # check if we adjusted the initial coordinate of the curve w/ a seam coordinate
                if elemBeg > 0:
                    if self.remeshBwd:
                        # first, we need to do the re-meshing of the other direction

                        # get the fwd data
                        nNewNodes = curveDict["nNewNodesReverse"]

                        # get the derivative seeds
                        newCoorb = curveBar[:, iBeg : iBeg + nNewNodes, :].copy()
                        iBeg += nNewNodes

                        # bars conn is everything up to elemBeg
                        barsConn = curveComp.barsConn[curveName][:elemBeg]

                        # loop over functions
                        for ii in range(N):
                            # Call Fortran code. Remember to adjust transposes and indices
                            _, _, cbi = self.utilitiesAPI.remesh_b(
                                nNewNodes - 1,
                                coor.T,
                                newCoorb[ii].T,
                                barsConn.T + 1,
                                method,
                                spacing,
                                initialSpacing,
                                finalSpacing,
                            )
                            # derivative seeds for the coordinates.
                            cb[ii] += cbi.T.copy()

                    # the first seed is for the projected point...
                    projb = cb[:, curveConn[elemBeg, 0], :].copy()

                    # zero out the seed of the replaced node
                    cb[:, curveConn[elemBeg, 0], :] = np.zeros((N, 3))

                    # put the modified initial and final points back in place.
                    self.compB.nodes[curveConn[elemBeg, 0]] = ptBegSave.copy()

                    # we need to call the curve projection routine to propagate the seed...
                    intNodesOrd = self.seamDict[curveName]["intNodesOrd"]
                    curveMask = self.seamDict[curveName]["curveMask"]
                    elemIDs = self.seamDict[curveName]["elemIDs"]
                    xyzProj = self.seamDict[curveName]["xyzProj"]
                    tanProj = self.seamDict[curveName]["tanProj"]
                    dist2 = self.seamDict[curveName]["dist2"]

                    # we need the full bars conn for this
                    barsConn = curveComp.barsConn[curveName]

                    # allocate zero seeds
                    xyzProjb = np.zeros_like(xyzProj)
                    tanProjb = np.zeros_like(tanProj)

                    curveProjb[curveName] = np.zeros((N, 3))

                    for ii in range(N):
                        # the only nonzero seed is indexed by argmin dist2
                        xyzProjb[np.argmin(dist2)] = projb[ii].copy()

                        xyzb_new, coorb_new = self.curveSearchAPI.mindistancecurve_b(
                            intNodesOrd.T,
                            self.compB.nodes.T,
                            barsConn.T + 1,
                            xyzProj.T,
                            xyzProjb.T,
                            tanProj.T,
                            tanProjb.T,
                            elemIDs + 1,
                            curveMask,
                        )

                        # add the coorb_new to coorBb[ii] since coorb_new has the seeds from mindistancecurve_b
                        coorBb[ii] += coorb_new.T

                        # xyzb_new is the seed for the intersection seam node
                        # instead of saving the array full of zeros, we just save the entry we know is nonzero
                        curveProjb[curveName][ii] = xyzb_new.T[np.argmin(dist2)]

                # all the remaining seeds in coorb live on the component tri-mesh...
                coorb += cb

        # now we only have the intersection seam...
        nFeature = len(self.nElems)
        intNodes = self.seamDict["intNodes"]
        seamConn = self.seamDict["seamConn"]
        curveSizes = self.seamDict["curveSizes"]

        # seeds for the original intersection
        intNodesb = np.zeros((N, intNodes.shape[0], intNodes.shape[1]))

        # loop over each feature and propagate the sensitivities
        curInd = 0
        curSeed = 0
        for i in range(nFeature):
            # just use the same number of points *2 for now
            nNewElems = self.nElems[i]
            nNewNodes = nNewElems + 1
            coor = intNodes
            barsConn = seamConn[curInd : curInd + curveSizes[i]]

            curInd += curveSizes[i]
            method = "linear"
            spacing = "linear"
            initialSpacing = 0.1
            finalSpacing = 0.1

            for ii in range(N):
                newCoorb = intBar[ii, curSeed : curSeed + nNewNodes, :]
                # re-sample the curve (try linear for now), to get N number of nodes on it spaced linearly
                # Call Fortran code. Remember to adjust transposes and indices
                newCoor, newBarsConn, cb = self.utilitiesAPI.remesh_b(
                    nNewElems, coor.T, newCoorb.T, barsConn.T + 1, method, spacing, initialSpacing, finalSpacing
                )
                intNodesb[ii] += cb.T

            curSeed += nNewNodes

        # add the contributions from the curve projection if we have any
        for curveName, v in curveProjb.items():
            # get the index
            idx = self.seamDict[curveName]["projPtIndx"]
            if self.debug:
                print(f"Adding contributions from {curveName}")
            for ii in range(N):
                # add the contribution
                intNodesb[ii, idx] += v[ii]

        dummyConn = np.zeros((0, 4))
        barsConn = self.seamDict["barsConn"]
        parentTria = self.seamDict["parentTria"]

        # do the reverse intersection computation to get the seeds of coordinates
        for ii in range(N):
            cAb, cBb = self.intersectionAPI.computeintersection_b(
                self.compA.nodes.T,
                self.compA.triConnStack.T,
                dummyConn.T,
                self.compB.nodes.T,
                self.compB.triConnStack.T,
                dummyConn.T,
                intNodes.T,
                intNodesb[ii].T,
                barsConn.T + 1,
                parentTria.T + 1,
                self.distTol,
            )

            coorAb[ii] += cAb.T
            coorBb[ii] += cBb.T

        # get the total sensitivities from both components
        compSens_local = {}
        compSensA = self.compA.DVGeo.totalSensitivity(coorAb, "triMesh")
        for k, v in compSensA.items():
            compSens_local[k] = v

        compSensB = self.compB.DVGeo.totalSensitivity(coorBb, "triMesh")
        for k, v in compSensB.items():
            compSens_local[k] = v

        # finally sum the results across procs if we are provided with a comm
        if comm:
            compSens = {}
            # because the results are in a dictionary, we need to loop over the items and sum
            for k, v in compSens_local.items():
                compSens[k] = comm.allreduce(v, op=MPI.SUM)
        else:
            # we can just pass the dictionary
            compSens = compSens_local

        return compSens

    def associatePointsToSurface(self, points, ptSetName, surface, surfaceEps):
        projDict = {}
        # Compute the distance from each point to this surface and store it in projDict
        if surface in self.compA.triConn:
            self._projectToComponent(points, self.compA, projDict, surface=surface)
            comp = "compA"
        elif surface in self.compB.triConn:
            self._projectToComponent(points, self.compB, projDict, surface=surface)
            comp = "compB"
        else:
            raise Error(f"Surface {surface} was not found in {self.compA.name} or {self.compB.name}.")

        # Identify the points that are within the given tolerance from this surface
        # surfaceInd contains indices of the provided points not the entire point set
        surfaceDist = np.sqrt(np.array(projDict["dist2"]))
        surfaceInd = [ind for ind, value in enumerate(surfaceDist) if (value < surfaceEps)]

        # Output the points associated with this surface
        if self.debug:
            data = [np.append(points[i], surfaceDist[i]) for i in surfaceInd]
            tecplot_interface.write_tecplot_scatter(
                f"{surface}_points_{self.comm.rank}.plt", f"{surface}", ["X", "Y", "Z", "dist"], data
            )

        # Save the indices only if there is at least one point
        if surfaceInd:
            self.projData[ptSetName][comp]["surfaceInd"][surface] = surfaceInd
            # Initialize a data dictionary for this surface
            self.projData[ptSetName][surface] = {}
