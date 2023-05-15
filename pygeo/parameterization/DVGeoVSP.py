# Standard Python modules
from collections import OrderedDict
import time

# External modules
from baseclasses.utils import Error
from mpi4py import MPI
import numpy as np
from packaging.version import Version
from pyspline.utils import searchQuads

# Local modules
from .DVGeoSketch import DVGeoSketch
from .designVars import vspDV

# openvsp python interface
try:
    # External modules
    import openvsp

    vspInstalled = True
except ImportError:
    try:
        # External modules
        import vsp as openvsp

        vspInstalled = True
    except ImportError:
        openvsp = None
        vspInstalled = False


vspOutOfDate = False
if vspInstalled:
    vspVersionStr = openvsp.GetVSPVersion()
    words = vspVersionStr.split()
    vspVersion = words[-1]
    # VSP is installed, but version too old
    if Version(vspVersion) < Version("3.28.0"):
        vspOutOfDate = True
    # Prior to OpenVSP 3.33.0, the "s" parameter varried between [0, 0.5]
    elif Version(vspVersion) < Version("3.33.0"):
        SMAX = 0.5
    # After this version, the range was changed to [0, 1.0].
    else:  # Version(vsp_version) >= Version("3.33.0")
        SMAX = 1.0


class DVGeometryVSP(DVGeoSketch):
    """
    A class for manipulating OpenVSP geometry.
    The purpose of the DVGeometryVSP class is to provide translation of the VSP geometry engine to externally supplied surfaces.
    This allows the use of VSP design variables to control the MACH framework.

    There are several important limitations:

    #. Since VSP is volume-based, it cannot be used to parameterize a geometry that does not lie within a VSP body.
    #. It cannot handle *moving* intersections. A geometry with static intersections is fine as long as the intersection doesn't move
    #. It does not support complex numbers for the complex-step method.
    #. It does not support separate configurations.
    #. Because OpenVSP does not provide sensitivities, this class uses parallel finite differencing to obtain the required Jacobian matrices.

    Parameters
    ----------
    fileName : str
       filename of .vsp3 file.

    comm : MPI Intra Comm
       Comm on which to build operate the object. This is used to
       perform embarrassingly parallel finite differencing. Defaults to
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
      >>> from pygeo import DVGeometryVSP
      >>> DVGeo = DVGeometryVSP("wing.vsp3", MPI_COMM_WORLD)
      >>> # Add a set of coordinates Xpt into the object
      >>> DVGeo.addPointSet(Xpt, 'myPoints')

    """

    def __init__(self, fileName, comm=MPI.COMM_WORLD, scale=1.0, comps=[], projTol=0.01):
        if not vspInstalled:
            raise ImportError(
                "The OpenVSP Python API is required in order to use DVGeometryVSP. "
                + "Ensure OpenVSP is installed properly and can be found on your path."
            )
        elif vspOutOfDate:
            raise AttributeError(
                "Out of date version of OpenVSP detected. "
                + "OpenVSP 3.28.0 or greater is required in order to use DVGeometryVSP"
            )

        if comm.rank == 0:
            print("Initializing DVGeometryVSP")
            t0 = time.time()

        super().__init__(fileName=fileName, comm=comm, scale=scale, projTol=projTol)

        if hasattr(openvsp, "VSPVehicle"):
            self.vspModel = openvsp.VSPVehicle()
        else:
            self.vspModel = openvsp

        self.exportComps = []

        # Clear the vsp model
        self.vspModel.ClearVSPModel()

        t1 = time.time()
        # read the model
        self.vspModel.ReadVSPFile(fileName)
        t2 = time.time()
        if self.comm.rank == 0:
            print("Loading the vsp model took:", (t2 - t1))

        # List of all components returned from VSP. Note that this
        # order is important. It is the order that we use to map the
        # actual geom_id by using the geom_names
        allComps = self.vspModel.FindGeoms()
        allNames = []
        for c in allComps:
            allNames.append(self.vspModel.GetContainerName(c))

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
            self.compNames.append(self.vspModel.GetContainerName(c))
            self.bbox[c] = self._getBBox(c)

        # Now, we need to form our own quad meshes for fast projections
        if comm.rank == 0:
            print("Building a quad mesh for fast projections.")
        self._getQuads()

        self.useComposite = False

        if comm.rank == 0:
            t3 = time.time()
            print("Initialized DVGeometry VSP in", (t3 - t0), "seconds.")

    def addPointSet(self, points, ptName, **kwargs):
        """
        Add a set of coordinates to DVGeometry

        The is the main way that geometry, in the form of a coordinate
        list is given to DVGeometry to be manipulated.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed. These coordinates *should* all
            project into the interior of the FFD volume.
        ptName : str
            A user supplied name to associate with the set of
            coordinates. Thisname will need to be provided when
            updating the coordinates or when getting the derivatives
            of the coordinates.

        Returns
        -------
        dMax_global : float
            The maximum distance betwen the points added and the
            projection on the VSP surfaces on any processor.
        """

        self.ptSetNames.append(ptName)

        # save this name so that we can zero out the jacobians properly
        # ADFlow checks self.points to see if something is added or not.
        self.points[ptName] = True

        points = np.array(points).real.astype("d")

        # we need to project each of these points onto the VSP geometry,
        # get geometry and surface IDs, u, v values, and coordinates of the projections.
        # then calculate the self.offset variable using the projected points.

        # first, to get a good initial guess on the geometry and u,v values,
        # we can use the adt projections in pyspline
        if len(points) > 0:
            # faceID has the index of the corresponding quad element.
            # uv has the parametric u and v weights of the projected point.

            faceID, uv = searchQuads(self.pts0.T, (self.conn + 1).T, points.T)
            uv = uv.T
            faceID -= 1  # Convert back to zero-based indexing.
            # after this point we should have the projected points.

        else:
            faceID = np.zeros((0), "intc")
            uv = np.zeros((0, 2), "intc")

        # now we need to figure out which surfaces the points got projected to
        # From the faceID we can back out what component each one is
        # connected to. This way if we have intersecting components we
        # only change the ones that are apart of the two surfaces.
        cumFaceSizes = np.zeros(len(self.sizes) + 1, "intc")
        for i in range(len(self.sizes)):
            nCellI = self.sizes[i][0] - 1
            nCellJ = self.sizes[i][1] - 1
            cumFaceSizes[i + 1] = cumFaceSizes[i] + nCellI * nCellJ
        compIDs = np.searchsorted(cumFaceSizes, faceID, side="right") - 1

        # coordinates to store the projected points
        pts = np.zeros(points.shape)

        # npoints * 3 list containing the geomID, u and v values
        # this can be improved if we can group points that get
        # projected to the same geometry.
        npoints = len(points)
        geom = np.zeros(npoints, dtype="intc")
        r = np.zeros(npoints)
        s = np.zeros(npoints)
        t = np.zeros(npoints)

        # initialize one 3dvec for projections
        pnt = openvsp.vec3d()

        # Keep track of the largest distance between cfd and vsp surfaces
        dMax = 1e-16

        t1 = time.time()
        for i in range(points.shape[0]):
            # this is the geometry our point got projected to in the adt code
            gind = compIDs[i]  # index
            gid = self.allComps[gind]  # ID

            # set the coordinates of the point object
            pnt.set_xyz(points[i, 0] * self.meshScale, points[i, 1] * self.meshScale, points[i, 2] * self.meshScale)

            # first, we call the fast projection code with the initial guess to find the nearest point on the surface

            # this is the global index of the first node of the projected element
            nodeInd = self.conn[faceID[i], 0]
            # get the local index of this node
            nn = nodeInd - self.cumSizes[gind]
            # figure out the i and j indices of the first point of the element
            # we projected this point to
            ii = np.mod(nn, self.sizes[gind, 0])
            jj = np.floor_divide(nn, self.sizes[gind, 0])

            # calculate the global u and v change in this element
            du = self.uv[gind][0][ii + 1] - self.uv[gind][0][ii]
            dv = self.uv[gind][1][jj + 1] - self.uv[gind][1][jj]

            # now get this points u,v coordinates on the vsp geometry and add
            # compute the initial guess using the  tesselation data of the surface
            ug = uv[i, 0] * du + self.uv[gind][0][ii]
            vg = uv[i, 1] * dv + self.uv[gind][1][jj]

            # We now convert the surface parameteric variables (u,v) to their equivalent volume parameterization (r,s,t)
            # this will serve as our initial guess for the volume projection procedure
            # In general, the conversion between the surface parametric coordinates (u, v)
            # and the first two volume parameteric coordinate (r, s)
            # are given by the equations below:
            #   u = r
            #   v_low = s
            #   v_upper = 1 - s
            # The 3rd parameteric coordinate interpolates between the the upper and lower surfaces
            #   X(r,s,t) = t * X_upper(r, s) + (1 - t) * X_lower(r, s)
            rg = ug
            if vg < 0.5:
                # This point is on the lower surface
                sg = SMAX * 2.0 * vg
            else:
                # This point is on the upper surface
                sg = SMAX * 2.0 * (1.0 - vg)
            # tg = 0.5 places the initial guess in the middle of the upper and lower surfaces for the volume
            # Note: If the point we're looking for actually lies on the surface openvsp's
            # projection algorithm (FindRSTGuess) will still quickly locate it, since our volume interpolation
            # is linear through the depth (i.e. in t)
            tg = 0.5
            # Now, find the closest volume projection
            d, r[i], s[i], t[i] = self.vspModel.FindRSTGuess(gid, 0, pnt, rg, sg, tg)
            geom[i] = gind

            # if we dont have a good projection, try projecting again to surfaces
            #  with the slow code.
            if d > self.projTol:
                # print('Guess code failed with projection distance',d)
                # for now, we need to check for all geometries separately.
                # Just pick the one that yields the smallest d
                gind = 0
                for gid in self.allComps:
                    # only project if the point is in the bounding box of the geometry
                    if (
                        (self.bbox[gid][0, 0] < points[i, 0] < self.bbox[gid][0, 1])
                        and (self.bbox[gid][1, 0] < points[i, 1] < self.bbox[gid][1, 1])
                        and (self.bbox[gid][2, 0] < points[i, 2] < self.bbox[gid][2, 1])
                    ):
                        # project the point onto the VSP geometry
                        dNew, rout, sout, tout = self.vspModel.FindRST(gid, 0, pnt)

                        # check if we are closer
                        if dNew < d:
                            # save this info if we found a closer projection
                            r[i] = rout
                            s[i] = sout
                            t[i] = tout
                            geom[i] = gind
                            d = dNew
                    gind += 1

            # check if the final d is larger than our previous largest value
            dMax = max(d, dMax)

            # We need to evaluate this pnt to get its coordinates in physical space
            pnt = self.vspModel.CompPntRST(self.allComps[geom[i]], 0, r[i], s[i], t[i])
            pts[i, 0] = pnt.x() * self.modelScale
            pts[i, 1] = pnt.y() * self.modelScale
            pts[i, 2] = pnt.z() * self.modelScale

        # some debug info
        dMax_global = self.comm.allreduce(dMax, op=MPI.MAX)
        t2 = time.time()

        if self.comm.rank == 0 or self.comm is None:
            print("DVGeometryVSP note:\nAdding pointset", ptName, "took", t2 - t1, "seconds.")
            print("Maximum distance between the added points and the VSP geometry is", dMax_global)

        # Create the little class with the data
        self.pointSets[ptName] = PointSet(points, pts, geom, r, s, t)

        # Set the updated flag to false because the jacobian is not up to date.
        self.updated[ptName] = False
        self.updatedJac[ptName] = False
        return dMax_global

    def setDesignVars(self, dvDict):
        """
        Standard routine for setting design variables from a design
        variable dictionary.

        Parameters
        ----------
        dvDict : dict
            Dictionary of design variables. The keys of the dictionary
            must correspond to the design variable names. Any
            additional keys in the dv-dictionary are simply ignored.
        """
        if self.useComposite:
            dvDict = self.mapXDictToDVGeo(dvDict)

        # Just dump in the values
        for key in dvDict:
            if key in self.DVs:
                self.DVs[key].value = dvDict[key]

        # we just need to set the design variables in the VSP model and we are done
        self._updateModel()

        # update the projected coordinates
        self._updateProjectedPts()

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

        # set the jacobian flag to false
        for ptName in self.pointSets:
            self.updatedJac[ptName] = False

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
        config : str or list
            Inactive parameter for DVGeometryVSP.
            See the same method in DVGeometry.py for its normal use.
        """

        # new cfd point coordinates are the updated projections minus the offset
        newPts = self.pointSets[ptSetName].pts - self.pointSets[ptSetName].offset

        # Finally flag this pointSet as being up to date:
        self.updated[ptSetName] = True

        return newPts

    def writeVSPFile(self, fileName, exportSet=0):
        """
        Take the current design and write a new VSP file

        Parameters
        ----------
        fileName : str
            Name of the output VSP file
        exportSet : int
            optional input parameter to select an export set in VSP
        """
        self.vspModel.WriteVSPFile(fileName, exportSet)

    def getNDV(self):
        """
        Return the number of DVs

        Returns
        -------
        len(self.DVs) : int
            number of design variables
        """
        return len(self.DVs)

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}}^T \frac{dI}{d_{pt}}`

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
        dIdxDict : dict
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

        # The following code computes the final sensitivity product:
        #
        #        T       T
        #   pXpt     pI
        #  ------  ------
        #   pXdv    pXpt
        #
        # Where I is the objective, Xpt are the externally coordinates
        # supplied in addPointSet

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])

        # reshape the dIdpt array from [N] * [nPt] * [3] to  [N] * [nPt*3]
        dIdpt = dIdpt.reshape((dIdpt.shape[0], dIdpt.shape[1] * 3))

        # jacobian of the projected points
        jac = self.pointSets[ptSetName].jac

        # local product
        dIdxT_local = jac.T.dot(dIdpt.T)

        # take the transpose to get the jacobian itself dI/dx
        dIdx_local = dIdxT_local.T

        # sum the results if we are provided a comm
        if comm:
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        if self.useComposite:
            dIdx = self.mapSensToComp(dIdx)
            dIdxDict = self.convertSensitivityToDict(dIdx, useCompositeNames=True)

        else:
            # Now convert to dict:
            dIdxDict = {}
            i = 0
            for dvName in self.DVs:
                arr = np.array(dIdx[:, i]).T
                dIdxDict[dvName] = arr.reshape(arr.shape[0], 1)
                i += 1

        return dIdxDict

    def totalSensitivityProd(self, vec, ptSetName, comm=None, config=None):
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}} \ vec`

        This is useful for forward AD mode.

        Parameters
        ----------
        vec : dictionary whose keys are the design variable names, and whose
              values are the derivative seeds of the corresponding design variable.

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            inactive parameter, this has no effect on the final result
            because with this method, the reduction is performed externally

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
        newvec = np.zeros(self.getNDV())

        # populate newvec
        for i, dv in enumerate(self.DVs):
            if dv in vec:
                newvec[i] = vec[dv]

        # we need to multiply with the surface jacobian
        dPt = self.pointSets[ptSetName].jac.dot(newvec)

        return dPt

    def totalSensitivityTransProd(self, dIdpt, ptSetName, comm=None, config=None):
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}}^T \frac{dI}{d_{pt}}`

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

        # The following code computes the final sensitivity product:
        #
        #        T       T
        #   pXpt     pI
        #  ------  ------
        #   pXdv    pXpt
        #
        # Where I is the objective, Xpt are the externally coordinates
        # supplied in addPointSet

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])

        # reshape the dIdpt array from [N] * [nPt] * [3] to  [N] * [nPt*3]
        dIdpt = dIdpt.reshape((dIdpt.shape[0], dIdpt.shape[1] * 3))

        jac = self.pointSets[ptSetName].jac.copy()
        dIdxT_local = jac.T.dot(dIdpt.T)
        dIdx_local = dIdxT_local.T

        if comm:
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # Now convert to dict:
        dIdxDict = {}
        i = 0
        for dvName in self.DVs:
            dIdxDict[dvName] = np.array(dIdx[:, i]).T
            i += 1

        return dIdxDict

    def addVariable(
        self, component, group, parm, value=None, lower=None, upper=None, scale=1.0, scaledStep=True, dh=1e-6
    ):
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

        container_id = self.vspModel.FindContainer(component, 0)
        if container_id == "":
            raise Error("Bad component for DV: %s" % component)

        parm_id = self.vspModel.FindParm(container_id, parm, group)
        if parm_id == "":
            raise Error(f"Bad group or parm: {component} {group} {parm}")

        # Now we know the parmID is ok. So we just get the value
        val = self.vspModel.GetParmVal(parm_id)

        dvName = f"{component}:{group}:{parm}"

        if value is None:
            value = val

        if scaledStep:
            dh = dh * value

            if value == 0:
                raise Error(
                    "Initial value is exactly 0. scaledStep option cannot be used"
                    "Specify an explicit dh with scaledStep=False"
                )

        self.DVs[dvName] = vspDV(parm_id, dvName, component, group, parm, value, lower, upper, scale, dh)

    def printDesignVariables(self):
        """
        Print a formatted list of design variables to the screen
        """
        print("-" * 85)
        print("{:>30}{:>20}{:>20}{:>15}".format("Component", "Group", "Parm", "Value"))
        print("-" * 85)
        for dvName in self.DVs:
            DV = self.DVs[dvName]
            print(f"{DV.component:>30}{DV.group:>20}{DV.parm:>20}{float(DV.value):15g}")

    def createDesignFile(self, fileName):
        """
        Take the current set of design variables and create a .des file

        Parameters
        ----------
        fileName : str
            name of the output .des file
        """
        f = open(fileName, "w")
        f.write("%d\n" % len(self.DVs))
        for dvName in self.DVs:
            DV = self.DVs[dvName]
            f.write(f"{DV.parmID}:{DV.component}:{DV.group}:{DV.parm}:{float(DV.value):20.15g}\n")
        f.close()

    def writePlot3D(self, fileName, exportSet=0):
        """
        Write the current design to Plot3D file

        Parameters
        ----------
        fileName : str
            name of the output Plot3D file
        exportSet : int
            optional argument to specify the export set in VSP
        """

        for dvName in self.DVs:
            DV = self.DVs[dvName]
            # We use float here since sometimes pyOptsparse will give
            # stupid numpy zero-dimensional arrays, which swig does
            # not like.
            self.vspModel.SetParmVal(DV.parmID, float(DV.value))
        self.vspModel.Update()

        # First set the export flag for exportSet to False for everyone
        for comp in self.allComps:
            self.vspModel.SetSetFlag(comp, exportSet, False)

        for comp in self.allComps:
            # Check if this one is in our list:
            compName = self.vspModel.GetContainerName(comp)
            if compName in self.compNames:
                self.vspModel.SetSetFlag(comp, exportSet, True)
                self.exportComps.append(compName)

        # Write the export file.
        self.vspModel.ExportFile(fileName, exportSet, openvsp.EXPORT_PLOT3D)

    # ----------------------------------------------------------------------- #
    #      THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER      #
    # ----------------------------------------------------------------------- #

    def _updateModel(self):
        """
        Set each of the DVs. We have the parmID stored so it's easy.
        """

        for dvName in self.DVs:
            DV = self.DVs[dvName]

            # for angle parameters, vsp only takes in degrees between -180 and +180,
            # which creates an unnecessary discontinuity at +-180.
            # to fix this, we take the mod of the value and set it to the correct range
            # that is allowed by VSP. Because all of the FD jacobian routine also goes
            # through here to update the model, we effectively maintain consistency
            if "angle" in DV.parm.lower():
                # set this new value separately to leave the DV.value itself untouched
                new_value = ((DV.value + 180.0) % 360.0) - 180.0
                self.vspModel.SetParmVal(DV.parmID, float(new_value))
            else:
                # We use float here since sometimes pyOptsparse will give
                # numpy zero-dimensional arrays, which swig does not like
                self.vspModel.SetParmValUpdate(DV.parmID, float(DV.value))

        # update the model
        self.vspModel.Update()

    def _updateProjectedPts(self):
        """
        Internally updates the coordinates of the projected points.
        """

        for ptSetName in self.pointSets:
            # get the current coordinates of projection points
            n = len(self.pointSets[ptSetName].points)
            newPts = np.zeros((n, 3))

            # newPts should get the new projected coords

            # get the info
            geom = self.pointSets[ptSetName].geom
            r = self.pointSets[ptSetName].r
            s = self.pointSets[ptSetName].s
            t = self.pointSets[ptSetName].t

            # This can all be done with arrays if we group points wrt geometry
            for i in range(n):
                # evaluate the new projected point coordinates
                pnt = self.vspModel.CompPntRST(self.allComps[geom[i]], 0, r[i], s[i], t[i])

                # update the coordinates
                newPts[i, :] = (pnt.x(), pnt.y(), pnt.z())

            # scale vsp coordinates to mesh coordinates, do it safely above for now
            newPts *= self.modelScale

            # set the updated coordinates
            self.pointSets[ptSetName].pts = newPts

    def _getBBox(self, comp):
        """
        This function computes the bounding box of the component. We add some buffer on each
        direction because we will use this bbox to determine which components to project points
        while adding point sets.
        """

        # initialize the array
        bbox = np.zeros((3, 2))

        # we need to get the number of main surfaces on this geometry
        nSurf = self.vspModel.GetNumMainSurfs(comp)
        nuv = self.bboxuv.shape[0]

        # allocate the arrays
        nodes = np.zeros((nSurf * nuv, 3))

        # loop over the surfaces
        for iSurf in range(nSurf):
            offset = iSurf * nuv
            # evaluate the points
            ptVec = self.vspModel.CompVecPnt01(comp, iSurf, self.bboxuv[:, 0], self.bboxuv[:, 1])
            # now extract the coordinates from the vec3dvec...sigh...
            for i in range(nuv):
                nodes[offset + i, :] = (ptVec[i].x(), ptVec[i].y(), ptVec[i].z())

        # get the min/max values of the coordinates
        for i in range(3):
            # this might be faster if we specify row/column major
            bbox[i, 0] = nodes[:, i].min()
            bbox[i, 1] = nodes[:, i].max()

        # finally scale the bounding box and return
        bbox *= self.modelScale

        # also give some offset on all directions
        bbox[:, 0] -= 0.1
        bbox[:, 1] += 0.1

        return bbox.copy()

    def _getuv(self):
        """
        Creates a uniform array of u-v combinations so that we can build a quad mesh ourselves.
        """

        # we need to sample the geometry, just do uniformly now
        nu = 20
        nv = 20

        # define the points on the parametric domain to sample
        ul = np.linspace(0, 1, nu + 1)
        vl = np.linspace(0, 1, nv + 1)
        uu, vv = np.meshgrid(ul, vl)
        uu = uu.flatten()
        vv = vv.flatten()

        # now create a concentrated uv array
        uv = np.dstack((uu, vv)).squeeze()

        return uv.copy()

    def _computeSurfJacobian(self):
        """
        This routine comptues the jacobian of the VSP surface with respect
        to the design variables. Since our point sets are rigidly linked to
        the VSP projection points, this is all we need to calculate. The input
        pointSets is a list or dictionary of pointSets to calculate the jacobian for.

        this routine runs in parallel, so it is important that any call leading to this
        subroutine is performed synchronously among self.comm

        VSP has a bug they refuse to fix. In a non-deterministic way, the spanwise u-v
        mapping can differ from run to run. Seems like there are two modes. The differences
        are small, but if we end up doing the FD with results from another processor
        the difference is large enough to completely mess up the sensitivity. Due to this
        we compute both baseline point and perturbed point on the processor that perturbs a given DV
        this is slightly slower but avoids this issue. the final gradient has some error still,
        but much more managable and unimportant compared to errors introduced by FD itself.
        See issue https://github.com/mdolab/pygeo/issues/58 for updates.
        """

        # timing stuff:
        t1 = time.time()
        tvsp = 0
        teval = 0
        tcomm = 0

        # counts
        nDV = self.getNDV()
        dvKeys = list(self.DVs.keys())
        nproc = self.comm.size
        rank = self.comm.rank

        # arrays to collect local pointset info
        rl = np.zeros(0)
        sl = np.zeros(0)
        tl = np.zeros(0)
        gl = np.zeros(0, dtype="intc")

        for ptSetName in self.pointSets:
            # initialize the Jacobians
            self.pointSets[ptSetName].jac = np.zeros((3 * self.pointSets[ptSetName].nPts, nDV))

            # first, we need to vstack all the point set info we have
            # counts of these are also important, saved in ptSet.nPts
            rl = np.concatenate((rl, self.pointSets[ptSetName].r))
            sl = np.concatenate((sl, self.pointSets[ptSetName].s))
            tl = np.concatenate((tl, self.pointSets[ptSetName].t))
            gl = np.concatenate((gl, self.pointSets[ptSetName].geom))

        # now figure out which proc has how many points.
        sizes = np.array(self.comm.allgather(len(rl)), dtype="intc")
        # displacements for allgather
        disp = np.array([np.sum(sizes[:i]) for i in range(nproc)], dtype="intc")
        # global number of points
        nptsg = np.sum(sizes)
        # create a local new point array. We will use this to get the new
        # coordinates as we perturb DVs. We just need one (instead of nDV times the size)
        # because we get the new points, calculate the jacobian and save it right after
        ptsNewL = np.zeros(len(rl) * 3)

        # create the arrays to receive the global info
        rg = np.zeros(nptsg)
        sg = np.zeros(nptsg)
        tg = np.zeros(nptsg)
        gg = np.zeros(nptsg, dtype="intc")

        # Now we do an allGatherv to get a long list of all pointset information
        self.comm.Allgatherv([rl, len(rl)], [rg, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([sl, len(sl)], [sg, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([tl, len(tl)], [tg, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([gl, len(gl)], [gg, sizes, disp, MPI.INT])

        # we now have all the point info on all procs.
        tcomm += time.time() - t1

        # We need to evaluate all the points on respective procs for FD computations

        # allocate memory
        pts0 = np.zeros((nptsg, 3))

        # evaluate the points
        for j in range(nptsg):
            pnt = self.vspModel.CompPntRST(self.allComps[gg[j]], 0, rg[j], sg[j], tg[j])
            pts0[j, :] = (pnt.x(), pnt.y(), pnt.z())

        # determine how many DVs this proc will perturb.
        n = 0
        for iDV in range(len(dvKeys)):
            # I have to do this one.
            if iDV % nproc == rank:
                n += 1

        # allocate the approriate sized numpy array for the perturbed points
        ptsNew = np.zeros((n, nptsg, 3))

        # perturb the DVs on different procs and compute the new point coordinates.
        i = 0  # Counter on local Jac

        for iDV in range(len(dvKeys)):
            # I have to do this one.
            if iDV % nproc == rank:
                # Step size for this particular DV
                dh = self.DVs[dvKeys[iDV]].dh

                # Perturb the DV
                dvSave = self.DVs[dvKeys[iDV]].value.copy()
                self.DVs[dvKeys[iDV]].value += dh

                # update the vsp model
                t11 = time.time()
                self._updateModel()
                t12 = time.time()
                tvsp += t12 - t11

                t11 = time.time()
                # evaluate the points
                for j in range(nptsg):
                    pnt = self.vspModel.CompPntRST(self.allComps[gg[j]], 0, rg[j], sg[j], tg[j])
                    ptsNew[i, j, :] = (pnt.x(), pnt.y(), pnt.z())
                t12 = time.time()
                teval += t12 - t11

                # now we can calculate the jac and put it back in ptsNew
                ptsNew[i, :, :] = (ptsNew[i, :, :] - pts0[:, :]) / dh

                # Reset the DV
                self.DVs[dvKeys[iDV]].value = dvSave.copy()

                # increment the counter
                i += 1

        # scale the points
        ptsNew *= self.modelScale

        # Now, we have perturbed points on each proc that perturbed a DV

        # reset the model.
        t11 = time.time()
        self._updateModel()
        t12 = time.time()
        tvsp += t12 - t11

        ii = 0
        # loop over the DVs and scatter the perturbed points to original procs
        for iDV in range(len(dvKeys)):
            # Step size for this particular DV
            dh = self.DVs[dvKeys[iDV]].dh

            t11 = time.time()
            # create the send/recv buffers for the scatter
            if iDV % nproc == rank:
                sendbuf = [ptsNew[ii, :, :].flatten(), sizes * 3, disp * 3, MPI.DOUBLE]
            else:
                sendbuf = [np.zeros((0, 3)), sizes * 3, disp * 3, MPI.DOUBLE]
            recvbuf = [ptsNewL, MPI.DOUBLE]

            # scatter the info from the proc that perturbed this DV to all procs
            self.comm.Scatterv(sendbuf, recvbuf, root=(iDV % nproc))

            t12 = time.time()
            tcomm += t12 - t11

            # calculate the jacobian here for the pointsets
            offset = 0
            for ptSet in self.pointSets:
                # number of points in this pointset
                nPts = self.pointSets[ptSet].nPts

                # indices to extract correct points from the long pointset array
                ibeg = offset * 3
                iend = ibeg + nPts * 3

                # ptsNewL has the jacobian itself...
                self.pointSets[ptSet].jac[0 : nPts * 3, iDV] = ptsNewL[ibeg:iend].copy()

                # TODO when OpenVSP fixes the bug in spanwise u-v distribution, we can disable the line above and
                # go back to the proper way below. we also need to clean up the evaluations themselves
                # self.pointSets[ptSet].jac[0:nPts*3, iDV] = (ptsNewL[ibeg:iend] - self.pointSets[ptSet].pts.flatten())/dh

                # increment the offset
                offset += nPts

            # pertrub the local counter on this proc.
            # This loops over the DVs that this proc perturbed
            if iDV % nproc == rank:
                ii += 1

        t2 = time.time()
        if rank == 0:
            print("FD jacobian calcs with dvgeovsp took", (t2 - t1), "seconds in total")
            print("updating the vsp model took", tvsp, "seconds")
            print("evaluating the new points took", teval, "seconds")
            print("communication took", tcomm, "seconds")

        # set the update flags
        for ptSet in self.pointSets:
            self.updatedJac[ptSet] = True

    def _getQuads(self):
        # build the quad mesh using the internal vsp geometry

        nSurf = len(self.allComps)
        pts = np.zeros((0, 3))
        conn = np.zeros((0, 4), dtype="intc")
        sizes = np.zeros((nSurf, 2), "intc")
        cumSizes = np.zeros(nSurf + 1, "intc")
        uv = []  # this will hold tessalation points
        offset = 0

        gind = 0
        for geom in self.allComps:
            # get uv tesselation
            utess, wtess = self.vspModel.GetUWTess01(geom, 0)
            # check if these values are good, otherwise, do it yourself!

            # save these values
            uv.append([np.array(utess), np.array(wtess)])
            nu = len(utess)
            nv = len(wtess)
            nElem = (nu - 1) * (nv - 1)

            # get u,v combinations of nodes
            uu, vv = np.meshgrid(utess, wtess)
            utess = uu.flatten()
            wtess = vv.flatten()

            # get the points
            ptvec = self.vspModel.CompVecPnt01(geom, 0, utess, wtess)

            # number of nodes for this geometry
            curSize = len(ptvec)

            # initialize coordinate and connectivity arrays
            compPts = np.zeros((curSize, 3))
            compConn = np.zeros((nElem, 4), dtype="intc")

            # get the coordinates of the points
            for i in range(curSize):
                compPts[i, :] = (ptvec[i].x(), ptvec[i].y(), ptvec[i].z())

            # build connectivity array
            k = 0
            for j in range(nv - 1):
                for i in range(nu - 1):
                    compConn[k, 0] = j * nu + i
                    compConn[k, 1] = j * nu + i + 1
                    compConn[k, 2] = (j + 1) * nu + i + 1
                    compConn[k, 3] = (j + 1) * nu + i
                    k += 1

            # apply the offset to the connectivities
            compConn += offset

            # stack the results
            pts = np.vstack((pts, compPts))
            conn = np.vstack((conn, compConn))

            # number of u and v point count
            sizes[gind, :] = (nu, nv)
            # cumilative number of points
            cumSizes[gind + 1] = cumSizes[gind] + curSize

            # increment the offset
            offset += curSize

            # increment geometry index
            gind += 1

        # finally, scale the points and save the data
        self.pts0 = pts * self.modelScale
        self.conn = conn
        self.sizes = sizes
        self.cumSizes = cumSizes
        self.uv = uv


class PointSet:
    """Internal class for storing the projection details of each pointset"""

    def __init__(self, points, pts, geom, r, s, t):
        self.points = points
        self.pts = pts
        self.geom = geom
        self.r = r
        self.s = s
        self.t = t
        self.offset = self.pts - self.points
        self.nPts = len(self.pts)
        self.jac = None
