# External modules
import numpy as np

# Local modules
from .. import geo_utils
from ..geo_utils.polygon import areaTri
from .baseConstraint import GeometricConstraint

try:
    # External modules
    from geograd import geograd_parallel  # noqa
except ImportError:
    geograd_parallel = None


class TriangulatedSurfaceConstraint(GeometricConstraint):
    """
    This class is used to enclose a triangulated object inside an
    aerodynamic surface.
    """

    def __init__(
        self,
        comm,
        name,
        surface_1,
        surface_1_name,
        DVGeo1,
        surface_2,
        surface_2_name,
        DVGeo2,
        scale,
        addToPyOpt,
        rho,
        perim_scale,
        max_perim,
        heuristic_dist,
    ):
        if geograd_parallel is None:
            raise ImportError("Geograd package must be installed to use triangulated surface constraint")

        super().__init__(name, 2, -1e10, 0.0, scale, None, addToPyOpt)

        self.comm = comm

        # get the point sets
        self.surface_1_name = surface_1_name
        self.surface_2_name = surface_2_name
        if DVGeo1 is None and DVGeo2 is None:
            raise ValueError(f"Must include at least one geometric parametrization in constraint {name}")
        self.DVGeo1 = DVGeo1
        self.DVGeo2 = DVGeo2

        self.surf1_size = surface_1[0].shape[0]
        self.surf1_p0 = surface_1[0].transpose()
        self.surf1_p1 = surface_1[1].transpose()
        self.surf1_p2 = surface_1[2].transpose()

        self.surf2_size = surface_2[0].shape[0]
        self.surf2_p0 = surface_2[0].transpose()
        self.surf2_p1 = surface_2[1].transpose()
        self.surf2_p2 = surface_2[2].transpose()

        xyzmax = np.maximum(np.maximum(self.surf2_p0.max(axis=1), self.surf2_p1.max(axis=1)), self.surf2_p2.max(axis=1))
        xyzmin = np.minimum(np.minimum(self.surf2_p0.min(axis=1), self.surf2_p1.min(axis=1)), self.surf2_p2.min(axis=1))

        computed_maxdim = np.sqrt(np.sum((xyzmax - xyzmin) ** 2))

        if heuristic_dist is not None:
            if heuristic_dist < computed_maxdim:
                raise ValueError(
                    "The heuristic distance must be less than the max diagonal"
                    "dimension of the bounding box, " + str(computed_maxdim)
                )
            self.maxdim = heuristic_dist
        else:
            self.maxdim = computed_maxdim * 1.05

        self.rho = rho
        self.perim_scale = perim_scale
        self.max_perim = max_perim
        self.smSize = None
        self.perim_length = None
        self.minimum_distance = None

    def getVarNames(self):
        """
        return the var names relevant to this constraint. By default, this is the DVGeo
        variables, but some constraints may extend this to include other variables.
        """
        if self.DVGeo1 is not None:
            varnamelist = self.DVGeo1.getVarNames(pyOptSparse=True)
            if self.DVGeo2 is not None:
                varnamelist.extend(self.DVGeo2.getVarNames(pyOptSparse=True))
        else:
            varnamelist = self.DVGeo2.getVarNames(pyOptSparse=True)

        return varnamelist

    def evalFunctions(self, funcs, config):
        """
        Evaluate the function this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        config : str
            The DVGeo configuration to apply this constraint to. Must be either None
            which will apply to *ALL* the local DV groups or a single string specifying
            a particular configuration.
        """
        # get the CFD triangulated mesh updates. need addToDVGeo = True when
        # running setSurface()

        # check if the first mesh has a DVGeo, and if it does, update the points

        if self.DVGeo1 is not None:
            self.surf1_p0 = self.DVGeo1.update(self.surface_1_name + "_p0", config=config).transpose()
            self.surf1_p1 = self.DVGeo1.update(self.surface_1_name + "_p1", config=config).transpose()
            self.surf1_p2 = self.DVGeo1.update(self.surface_1_name + "_p2", config=config).transpose()

        # check if the second mesh has a DVGeo, and if it does, update the points
        if self.DVGeo2 is not None:
            self.surf2_p0 = self.DVGeo2.update(self.surface_2_name + "_p0", config=config).transpose()
            self.surf2_p1 = self.DVGeo2.update(self.surface_2_name + "_p1", config=config).transpose()
            self.surf2_p2 = self.DVGeo2.update(self.surface_2_name + "_p2", config=config).transpose()

        KS, perim, failflag = self.evalTriangulatedSurfConstraint()
        funcs[self.name + "_KS"] = KS
        funcs[self.name + "_perim"] = perim
        if failflag:
            funcs["fail"] = failflag

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
            config : str
            The DVGeo configuration to apply this constraint to. Must be either None
            which will apply to *ALL* the local DV groups or a single string specifying
            a particular configuration.
        """
        tmpTotalKS = {}
        tmpTotalPerim = {}

        deriv_outputs = self.evalTriangulatedSurfConstraintSens()
        # deriv outputs contains:
        # KS, intersect_length, mindist, timings, unbalance (index 0 through 4)
        # dKSdA1, dKSdB1, dKSdC1, dKSdA2, dKSdB2, dKSdC2 (index 5 through 10)
        # dPdA1, dPdB1, dPdC1, dPdA2, dPdB2, dPdC2 (index 11 through 16)

        if self.DVGeo1 is not None:
            nDV1 = self.DVGeo1.getNDV()
        else:
            nDV1 = 0

        if nDV1 > 0:
            # compute sensitivity with respect to the first mesh
            # grad indices 0-2 are for mesh 1 p0, 1, 2 / 3-5 are for mesh 2 p0, 1, 2
            tmp_KS_p0 = self.DVGeo1.totalSensitivity(
                np.transpose(deriv_outputs[5]), self.surface_1_name + "_p0", config=config
            )
            tmp_KS_p1 = self.DVGeo1.totalSensitivity(
                np.transpose(deriv_outputs[6]), self.surface_1_name + "_p1", config=config
            )
            tmp_KS_p2 = self.DVGeo1.totalSensitivity(
                np.transpose(deriv_outputs[7]), self.surface_1_name + "_p2", config=config
            )
            tmp_perim_p0 = self.DVGeo1.totalSensitivity(
                np.transpose(deriv_outputs[11]), self.surface_1_name + "_p0", config=config
            )
            tmp_perim_p1 = self.DVGeo1.totalSensitivity(
                np.transpose(deriv_outputs[12]), self.surface_1_name + "_p1", config=config
            )
            tmp_perim_p2 = self.DVGeo1.totalSensitivity(
                np.transpose(deriv_outputs[13]), self.surface_1_name + "_p2", config=config
            )
            for key in tmp_KS_p0:
                tmpTotalKS[key] = tmp_KS_p0[key] + tmp_KS_p1[key] + tmp_KS_p2[key]
                tmpTotalPerim[key] = tmp_perim_p0[key] + tmp_perim_p1[key] + tmp_perim_p2[key]

        if self.DVGeo2 is not None:
            nDV2 = self.DVGeo2.getNDV()
        else:
            nDV2 = 0

        if nDV2 > 0:
            # compute sensitivity with respect to the first mesh
            # grad indices 0-2 are for mesh 1 p0, 1, 2 / 3-5 are for mesh 2 p0, 1, 2
            tmp_KS_p0 = self.DVGeo2.totalSensitivity(
                np.transpose(deriv_outputs[8]), self.surface_2_name + "_p0", config=config
            )
            tmp_KS_p1 = self.DVGeo2.totalSensitivity(
                np.transpose(deriv_outputs[9]), self.surface_2_name + "_p1", config=config
            )
            tmp_KS_p2 = self.DVGeo2.totalSensitivity(
                np.transpose(deriv_outputs[10]), self.surface_2_name + "_p2", config=config
            )
            tmp_perim_p0 = self.DVGeo2.totalSensitivity(
                np.transpose(deriv_outputs[14]), self.surface_2_name + "_p0", config=config
            )
            tmp_perim_p1 = self.DVGeo2.totalSensitivity(
                np.transpose(deriv_outputs[15]), self.surface_2_name + "_p1", config=config
            )
            tmp_perim_p2 = self.DVGeo2.totalSensitivity(
                np.transpose(deriv_outputs[16]), self.surface_2_name + "_p2", config=config
            )
            for key in tmp_KS_p0:
                tmpTotalKS[key] = tmp_KS_p0[key] + tmp_KS_p1[key] + tmp_KS_p2[key]
                tmpTotalPerim[key] = tmp_perim_p0[key] + tmp_perim_p1[key] + tmp_perim_p2[key]
        funcsSens[self.name + "_KS"] = tmpTotalKS
        funcsSens[self.name + "_perim"] = tmpTotalPerim

    def evalTriangulatedSurfConstraint(self):
        """
        Call geograd to compute the KS function and intersection length
        """
        # first compute the length of the intersection surface between the object and surf mesh
        mindist_tmp = 0.0

        # first run to get the minimum distance
        _, perim_length, mindist, _, _ = geograd_parallel.compute(
            self.surf1_p0,
            self.surf1_p1,
            self.surf1_p2,
            self.surf2_p0,
            self.surf2_p1,
            self.surf2_p2,
            mindist_tmp,
            self.rho,
            self.maxdim,
            self.comm.py2f(),
        )
        # second run gets the well-conditioned KS
        KS, perim_length, mindist, _, _ = geograd_parallel.compute(
            self.surf1_p0,
            self.surf1_p1,
            self.surf1_p2,
            self.surf2_p0,
            self.surf2_p1,
            self.surf2_p2,
            mindist,
            self.rho,
            self.maxdim,
            self.comm.py2f(),
        )

        self.perim_length = perim_length
        self.minimum_distance = mindist

        if self.perim_length > self.max_perim:
            failflag = True
            if self.comm.rank == 0:
                print(f"Intersection length {self.perim_length} exceeds tol {self.max_perim}, returning fail flag")
        else:
            failflag = False
        return KS, perim_length, failflag

    def evalTriangulatedSurfConstraintSens(self):
        """
        Call geograd to compute the derivatives of the KS function and intersection length
        """
        # first compute the length of the intersection surface between the object and surf mesh
        deriv_output = geograd_parallel.compute_derivs(
            self.surf1_p0,
            self.surf1_p1,
            self.surf1_p2,
            self.surf2_p0,
            self.surf2_p1,
            self.surf2_p2,
            self.minimum_distance,
            self.rho,
            self.maxdim,
            self.comm.py2f(),
        )
        return deriv_output

    def addConstraintsPyOpt(self, optProb, exclude_wrt=None):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addConGroup(
                self.name + "_KS", 1, lower=self.lower, upper=self.upper, scale=self.scale, wrt=self.getVarNames()
            )
            optProb.addConGroup(
                self.name + "_perim",
                1,
                lower=self.lower,
                upper=self.upper,
                scale=self.perim_scale,
                wrt=self.getVarNames(),
            )

    def writeTecplot(self, handle):
        raise NotImplementedError()


class SurfaceAreaConstraint(GeometricConstraint):
    """
    DVConstraints representation of a surface area
    constraint. One of these objects is created each time a
    addSurfaceAreaConstraints call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, p0, v1, v2, lower, upper, scale, scaled, DVGeo, addToPyOpt, compNames):
        super().__init__(name, 1, lower, upper, scale, DVGeo, addToPyOpt)
        self.scaled = scaled

        # create output array
        self.X = np.zeros(self.nCon)
        self.n = len(p0)

        # The first thing we do is convert v1 and v2 to coords
        self.p0 = p0
        self.p1 = v1 + p0
        self.p2 = v2 + p0

        # Now embed the coordinates into DVGeo
        # with the name provided:
        # TODO this is duplicating a DVGeo pointset (same as the surface which originally created the constraint)
        self.DVGeo.addPointSet(self.p0, self.name + "p0", compNames=compNames)
        self.DVGeo.addPointSet(self.p1, self.name + "p1", compNames=compNames)
        self.DVGeo.addPointSet(self.p2, self.name + "p2", compNames=compNames)

        # compute the reference area
        self.X0 = areaTri(self.p0, self.p1, self.p2)

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.p0 = self.DVGeo.update(self.name + "p0", config=config)
        self.p1 = self.DVGeo.update(self.name + "p1", config=config)
        self.p2 = self.DVGeo.update(self.name + "p2", config=config)

        self.X = areaTri(self.p0, self.p1, self.p2)
        if self.scaled:
            self.X /= self.X0
        funcs[self.name] = self.X

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dAdp0 = np.zeros((self.nCon, self.p0.shape[0], self.p0.shape[1]))
            dAdp1 = np.zeros((self.nCon, self.p1.shape[0], self.p1.shape[1]))

            dAdp2 = np.zeros((self.nCon, self.p2.shape[0], self.p2.shape[1]))

            p0 = self.p0
            p1 = self.p1
            p2 = self.p2
            for con in range(self.nCon):
                p0b = dAdp0[con, :, :]
                p1b = dAdp1[con, :, :]
                p2b = dAdp2[con, :, :]
                areab = 1
                areasb = np.empty(self.n)
                crossesb = np.empty((self.n, 3))
                v1b = np.empty((self.n, 3))
                v2b = np.empty((self.n, 3))
                if self.scaled:
                    areab = areab / self.X0
                areasb[:] = areab / 2.0

                v1 = p1 - p0
                v2 = p2 - p0

                crosses = np.cross(v1, v2)
                # for j in range(3):
                #     areas(i) = areas(i) + crosses(i, j)**2
                # areas[i] = np.sum(crosses[i, :]**2)
                areas = np.sum(crosses**2, axis=1)
                for i in range(self.n):  # DO i=1,n
                    if areas[i] == 0.0:
                        areasb[i] = 0.0
                    else:
                        areasb[i] = areasb[i] / (2.0 * np.sqrt(areas[i]))

                    # for j in reversed(range(3)):#DO j=3,1,-1
                    #     crossesb(i, j) = crossesb(i, j) + 2*crosses(i, j)*areasb(i)
                    crossesb[i, :] = 2 * crosses[i, :] * areasb[i]

                    v1b[i, :], v2b[i, :] = geo_utils.cross_b(v1[i, :], v2[i, :], crossesb[i, :])

                    # for j in reversed(range(3)):#DO j=3,1,-1
                    #      p2b(i, j) = p2b(i, j) + v2b(i, j)
                    #      p0b(i, j) = p0b(i, j) - v1b(i, j) - v2b(i, j)
                    #      v2b(i, j) = 0.0
                    #      p1b(i, j) = p1b(i, j) + v1b(i, j)
                    #      v1b(i, j) = 0.0
                    p2b[i, :] = v2b[i, :]
                    p0b[i, :] = -v1b[i, :] - v2b[i, :]
                    p1b[i, :] = p1b[i, :] + v1b[i, :]

            tmpp0 = self.DVGeo.totalSensitivity(dAdp0, self.name + "p0", config=config)
            tmpp1 = self.DVGeo.totalSensitivity(dAdp1, self.name + "p1", config=config)
            tmpp2 = self.DVGeo.totalSensitivity(dAdp2, self.name + "p2", config=config)
            tmpTotal = {}
            for key in tmpp0:
                tmpTotal[key] = tmpp0[key] + tmpp1[key] + tmpp2[key]

            funcsSens[self.name] = tmpTotal

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s_surface\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FETRIANGLE\n" % (3 * self.n, self.n))
        handle.write("DATAPACKING=POINT\n")
        for i in range(self.n):
            handle.write(f"{self.p0[i, 0]:f} {self.p0[i, 1]:f} {self.p0[i, 2]:f}\n")
        for i in range(self.n):
            handle.write(f"{self.p1[i, 0]:f} {self.p1[i, 1]:f} {self.p1[i, 2]:f}\n")

        for i in range(self.n):
            handle.write(f"{self.p2[i, 0]:f} {self.p2[i, 1]:f} {self.p2[i, 2]:f}\n")

        for i in range(self.n):
            handle.write("%d %d %d\n" % (i + 1, i + self.n + 1, i + self.n * 2 + 1))


class ProjectedAreaConstraint(GeometricConstraint):
    """
    DVConstraints representation of a surface area
    constraint. One of these objects is created each time a
    addSurfaceAreaConstraints call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, p0, v1, v2, axis, lower, upper, scale, scaled, DVGeo, addToPyOpt, compNames):
        super().__init__(name, 1, lower, upper, scale, DVGeo, addToPyOpt)
        self.scaled = scaled

        # create output array
        self.X = np.zeros(self.nCon)
        self.n = len(p0)
        self.axis = axis
        self.activeTris = np.zeros(self.n)

        # The first thing we do is convert v1 and v2 to coords
        self.p0 = p0
        self.p1 = v1 + p0
        self.p2 = v2 + p0

        # Now embed the coordinates into DVGeo
        # with the name provided:
        # TODO this is duplicating a DVGeo pointset (same as the surface which originally created the constraint)
        self.DVGeo.addPointSet(self.p0, self.name + "p0", compNames=compNames)
        self.DVGeo.addPointSet(self.p1, self.name + "p1", compNames=compNames)
        self.DVGeo.addPointSet(self.p2, self.name + "p2", compNames=compNames)

        # compute the reference area
        self.X0 = self._computeProjectedAreaTri(self.p0, self.p1, self.p2, self.axis)

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.p0 = self.DVGeo.update(self.name + "p0", config=config)
        self.p1 = self.DVGeo.update(self.name + "p1", config=config)
        self.p2 = self.DVGeo.update(self.name + "p2", config=config)

        self.X = self._computeProjectedAreaTri(self.p0, self.p1, self.p2, self.axis)
        if self.scaled:
            self.X /= self.X0
        funcs[self.name] = self.X

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dAdp0 = np.zeros((self.nCon, self.p0.shape[0], self.p0.shape[1]))
            dAdp1 = np.zeros((self.nCon, self.p1.shape[0], self.p1.shape[1]))

            dAdp2 = np.zeros((self.nCon, self.p2.shape[0], self.p2.shape[1]))
        p0 = self.p0
        p1 = self.p1
        p2 = self.p2
        for con in range(self.nCon):
            p0b = dAdp0[con, :, :]
            p1b = dAdp1[con, :, :]
            p2b = dAdp2[con, :, :]
            areab = 1
            areasb = np.empty(self.n)
            if self.scaled:
                areab = areab / self.X0
            areasb[:] = areab / 2.0

            for i in range(self.n):
                v1 = p1[i, :] - p0[i, :]
                v2 = p2[i, :] - p0[i, :]
                SAvec = np.cross(v1, v2)
                PA = np.dot(SAvec, self.axis)
                if PA > 0:
                    PAb = areasb[i]
                else:
                    PAb = 0.0
                SAvecb, _ = geo_utils.dot_b(SAvec, self.axis, PAb)
                v1b, v2b = geo_utils.cross_b(v1, v2, SAvecb)
                p2b[i, :] = p2b[i, :] + v2b
                p1b[i, :] = p1b[i, :] + v1b
                p0b[i, :] = p0b[i, :] - v1b - v2b

        tmpp0 = self.DVGeo.totalSensitivity(dAdp0, self.name + "p0", config=config)
        tmpp1 = self.DVGeo.totalSensitivity(dAdp1, self.name + "p1", config=config)
        tmpp2 = self.DVGeo.totalSensitivity(dAdp2, self.name + "p2", config=config)
        tmpTotal = {}
        for key in tmpp0:
            tmpTotal[key] = tmpp0[key] + tmpp1[key] + tmpp2[key]

        funcsSens[self.name] = tmpTotal

    def _computeProjectedAreaTri(self, p0, p1, p2, axis, plot=False):
        """
        Compute projected surface area
        """

        # Convert p1 and p2 to v1 and v2
        v1 = p1 - p0
        v2 = p2 - p0

        # Compute the surface area vectors for each triangle patch
        surfaceAreas = np.cross(v1, v2)

        # Compute the projected area of each triangle patch
        projectedAreas = np.dot(surfaceAreas, axis)

        # Cut out negative projected areas to get one side of surface
        if plot:
            for i in range(self.n):
                if projectedAreas[i] < 0:
                    self.activeTris[i] = 1
                else:
                    projectedAreas[i] = 0.0
        else:
            projectedAreas[projectedAreas < 0] = 0.0

        # Sum projected areas and divide by two for triangle area
        totalProjectedArea = np.sum(projectedAreas) / 2.0

        return totalProjectedArea

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """
        self._computeProjectedAreaTri(self.p0, self.p1, self.p2, self.axis, plot=True)
        nActiveTris = int(np.sum(self.activeTris))
        p0 = self.p0.copy()
        p1 = self.p1.copy()
        p2 = self.p2.copy()
        if self.axis[0] == 1.0:
            p0[:, 0] = np.zeros(self.n)
            p1[:, 0] = np.zeros(self.n)
            p2[:, 0] = np.zeros(self.n)
        if self.axis[1] == 1.0:
            p0[:, 1] = np.zeros(self.n)
            p1[:, 1] = np.zeros(self.n)
            p2[:, 1] = np.zeros(self.n)
        if self.axis[2] == 1.0:
            p0[:, 2] = np.zeros(self.n)
            p1[:, 2] = np.zeros(self.n)
            p2[:, 2] = np.zeros(self.n)

        handle.write("Zone T=%s_surface\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FETRIANGLE\n" % (3 * nActiveTris, nActiveTris))
        handle.write("DATAPACKING=POINT\n")
        for i in range(self.n):
            if self.activeTris[i]:
                handle.write(f"{p0[i, 0]:f} {p0[i, 1]:f} {p0[i, 2]:f}\n")
        for i in range(self.n):
            if self.activeTris[i]:
                handle.write(f"{p1[i, 0]:f} {p1[i, 1]:f} {p1[i, 2]:f}\n")
        for i in range(self.n):
            if self.activeTris[i]:
                handle.write(f"{p2[i, 0]:f} {p2[i, 1]:f} {p2[i, 2]:f}\n")
        iActive = 0
        for i in range(self.n):
            if self.activeTris[i]:
                handle.write("%d %d %d\n" % (iActive + 1, iActive + nActiveTris + 1, iActive + nActiveTris * 2 + 1))
                iActive += 1
