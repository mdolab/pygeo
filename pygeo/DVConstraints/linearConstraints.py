# ======================================================================
#         Imports
# ======================================================================
import numpy as np
from baseclasses import Error
from . import geo_utils
from GeometricConstraint import GeometricConstraint


class LinearConstraint(object):
    """
    This class is used to represet a set of generic set of linear
    constriants coupling local shape variables together.
    """

    def __init__(self, name, indSetA, indSetB, factorA, factorB, lower, upper, DVGeo, config):
        # No error checking here since the calling routine should have
        # already done it.
        self.name = name
        self.indSetA = indSetA
        self.indSetB = indSetB
        self.factorA = factorA
        self.factorB = factorB
        self.lower = lower
        self.upper = upper
        self.DVGeo = DVGeo
        self.ncon = 0
        self.wrt = []
        self.jac = {}
        self.config = config
        self._finalize()

    def evalFunctions(self, funcs):
        """
        Evaluate the function this object has and place in the funcs
        dictionary. Note that this function typically will not need to
        called since these constraints are supplied as a linear
        constraint jacobian they constraints themselves need to be
        revaluated.

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        cons = []
        for key in self.wrt:
            if key in self.DVGeo.DV_listLocal:
                cons.extend(self.jac[key].dot(self.DVGeo.DV_listLocal[key].value))
            elif key in self.DVGeo.DV_listSectionLocal:
                cons.extend(self.jac[key].dot(self.DVGeo.DV_listSectionLocal[key].value))
            elif key in self.DVGeo.DV_listSpanwiseLocal:
                cons.extend(self.jac[key].dot(self.DVGeo.DV_listSpanwiseLocal[key].value))
            else:
                raise Error(f"con {self.name} diffined wrt {key}, but {key} not found in DVGeo")
        funcs[self.name] = np.array(cons).real.astype("d")

    def evalFunctionsSens(self, funcsSens):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        funcsSens[self.name] = self.jac

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt. These constraints are added as
        linear constraints.
        """
        if self.ncon > 0:
            for key in self.jac:
                optProb.addConGroup(
                    self.name + "_" + key,
                    self.jac[key].shape[0],
                    lower=self.lower,
                    upper=self.upper,
                    scale=1.0,
                    linear=True,
                    wrt=key,
                    jac={key: self.jac[key]},
                )

    def _finalize(self):
        """
        We have postponed actually determining the constraint jacobian
        until this function is called. Here we determine the actual
        constraint jacobains as they relate to the actual sets of
        local shape variables that may (or may not) be present in the
        DVGeo object.
        """
        self.vizConIndices = {}
        # Local Shape Variables
        for key in self.DVGeo.DV_listLocal:
            if self.config is None or self.config in self.DVGeo.DV_listLocal[key].config:

                # end for (indSet loop)
                cons = self.DVGeo.DV_listLocal[key].mapIndexSets(self.indSetA, self.indSetB)
                ncon = len(cons)
                if ncon > 0:
                    # Now form the jacobian:
                    ndv = self.DVGeo.DV_listLocal[key].nVal
                    jacobian = np.zeros((ncon, ndv))
                    for i in range(ncon):
                        jacobian[i, cons[i][0]] = self.factorA[i]
                        jacobian[i, cons[i][1]] = self.factorB[i]
                    self.jac[key] = jacobian

                # Add to the number of constraints and store indices which
                # we need for tecplot visualization
                self.ncon += len(cons)
                self.vizConIndices[key] = cons

        # Section local shape variables
        for key in self.DVGeo.DV_listSectionLocal:
            if self.config is None or self.config in self.DVGeo.DV_listSectionLocal[key].config:

                # end for (indSet loop)
                cons = self.DVGeo.DV_listSectionLocal[key].mapIndexSets(self.indSetA, self.indSetB)
                ncon = len(cons)
                if ncon > 0:
                    # Now form the jacobian:
                    ndv = self.DVGeo.DV_listSectionLocal[key].nVal
                    jacobian = np.zeros((ncon, ndv))
                    for i in range(ncon):
                        jacobian[i, cons[i][0]] = self.factorA[i]
                        jacobian[i, cons[i][1]] = self.factorB[i]
                    self.jac[key] = jacobian

                # Add to the number of constraints and store indices which
                # we need for tecplot visualization
                self.ncon += len(cons)
                self.vizConIndices[key] = cons

        # Section local shape variables
        for key in self.DVGeo.DV_listSpanwiseLocal:
            if self.config is None or self.config in self.DVGeo.DV_listSpanwiseLocal[key].config:

                # end for (indSet loop)
                cons = self.DVGeo.DV_listSpanwiseLocal[key].mapIndexSets(self.indSetA, self.indSetB)
                ncon = len(cons)
                if ncon > 0:
                    # Now form the jacobian:
                    ndv = self.DVGeo.DV_listSpanwiseLocal[key].nVal
                    jacobian = np.zeros((ncon, ndv))
                    for i in range(ncon):
                        jacobian[i, cons[i][0]] = self.factorA[i]
                        jacobian[i, cons[i][1]] = self.factorB[i]
                    self.jac[key] = jacobian

                # Add to the number of constraints and store indices which
                # we need for tecplot visualization
                self.ncon += len(cons)
                self.vizConIndices[key] = cons

        # with-respect-to are just the keys of the jacobian
        self.wrt = list(self.jac.keys())

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of lete constraints
        to the open file handle
        """

        for key in self.vizConIndices:
            ncon = len(self.vizConIndices[key])
            nodes = np.zeros((ncon * 2, 3))
            for i in range(ncon):
                nodes[2 * i] = self.DVGeo.FFD.coef[self.indSetA[i]]
                nodes[2 * i + 1] = self.DVGeo.FFD.coef[self.indSetB[i]]

            handle.write("Zone T=%s\n" % (self.name + "_" + key))
            handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (ncon * 2, ncon))
            handle.write("DATAPACKING=POINT\n")
            for i in range(ncon * 2):
                handle.write("%f %f %f\n" % (nodes[i, 0], nodes[i, 1], nodes[i, 2]))

            for i in range(ncon):
                handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class GlobalLinearConstraint(object):
    """
    This class is used to represent a set of generic set of linear
    constriants coupling global design variables together.
    """

    def __init__(self, name, key, type, options, lower, upper, DVGeo, config):
        # No error checking here since the calling routine should have
        # already done it.
        self.name = name
        self.key = key
        self.type = type
        self.lower = lower
        self.upper = upper
        self.DVGeo = DVGeo
        self.ncon = 0
        self.jac = {}
        self.config = config
        if self.type == "monotonic":
            self.setMonotonic(options)

    def evalFunctions(self, funcs):
        """
        Evaluate the function this object has and place in the funcs
        dictionary. Note that this function typically will not need to
        called since these constraints are supplied as a linear
        constraint jacobian they constraints themselves need to be
        revaluated.

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        cons = []
        for key in self.jac:
            cons.extend(self.jac[key].dot(self.DVGeo.DV_listGlobal[key].value))

        funcs[self.name] = np.array(cons).real.astype("d")

    def evalFunctionsSens(self, funcsSens):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        funcsSens[self.name] = self.jac

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt. These constraints are added as
        linear constraints.
        """
        if self.ncon > 0:
            for key in self.jac:
                optProb.addConGroup(
                    self.name + "_" + key,
                    self.jac[key].shape[0],
                    lower=self.lower,
                    upper=self.upper,
                    scale=1.0,
                    linear=True,
                    wrt=key,
                    jac={key: self.jac[key]},
                )

    def setMonotonic(self, options):
        """
        Set up monotonicity jacobian for the given global design variable
        """
        self.vizConIndices = {}

        if self.config is None or self.config in self.DVGeo.DV_listGlobal[self.key].config:
            ndv = self.DVGeo.DV_listGlobal[self.key].nVal
            start = options["start"]
            stop = options["stop"]
            if stop == -1:
                stop = ndv

            # Since start and stop are inclusive, we need to add one to stop to
            # account for python indexing
            stop += 1
            ncon = len(np.zeros(ndv)[start:stop]) - 1

            jacobian = np.zeros((ncon, ndv))
            slope = options["slope"]
            for i in range(ncon):
                jacobian[i, start + i] = 1.0 * slope
                jacobian[i, start + i + 1] = -1.0 * slope
            self.jac[self.key] = jacobian
            self.ncon += ncon

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of lete constraints
        to the open file handle
        """

        pass


class ColinearityConstraint(GeometricConstraint):
    """
    DVConstraints representation of a colinearity constraint.
    Constrain that all of the points provided stay colinear with the
    specified axis.
    One of these objects is created each time an
    addColinearityConstraint call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, axis, origin, coords, lower, upper, scale, DVGeo, addToPyOpt):
        self.name = name
        self.nCon = len(coords)
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(
            self, self.name, self.nCon, self.lower, self.upper, self.scale, self.DVGeo, self.addToPyOpt
        )

        # create the output array
        self.X = np.zeros(self.nCon)

        # The first thing we do is convert v1 and v2 to coords
        self.axis = axis
        self.origin = origin
        self.coords = coords

        # Now embed the coordinates and origin into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.origin, self.name + "origin")
        self.DVGeo.addPointSet(self.coords, self.name + "coords")

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name + "coords", config=config)
        self.origin = self.DVGeo.update(self.name + "origin", config=config)

        # # Compute the direction from each point to the origin
        # dirVec = self.origin-self.coords

        # # compute the cross product with the desired axis. Cross product
        # # will be zero if the direction vector is the same as the axis
        # resultDir = np.cross(self.axis,dirVec)

        # for i in range(len(resultDir)):
        #     self.X[i] = geo_utils.euclideanNorm(resultDir[i,:])
        self.X = self._computeDist(self.origin, self.coords, self.axis)

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
            dCdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))
            dCdOrigin = np.zeros((self.nCon, self.origin.shape[0], self.origin.shape[1]))
            dCdAxis = np.zeros((self.nCon, self.axis.shape[0], self.axis.shape[1]))

            # Compute the direction from each point to the origin
            # for i in range(n):
            #     for j in range(3):
            #         dirvec[i, j] = origin[j] - coords[i, j]
            dirVec = self.origin - self.coords

            # axisb = 0.0
            # dirvecb = 0.0
            # for i in range(self.nCon):
            #     resultdir = np.cross(axis, dirvec[i, :])
            #     self.X[i] = 0
            #     for j in range(3):
            #         self.X[i] = self.X[i] + resultdir[j]**2
            resultDir = np.cross(self.axis, dirVec)
            tmpX = np.zeros(self.nCon)
            for i in range(len(resultDir)):
                # self.X[i] = geo_utils.euclideanNorm(resultDir[i,:])
                for j in range(3):
                    tmpX[i] += resultDir[i, j] ** 2

            resultdirb = np.zeros(3)
            dirvecb = np.zeros_like(dirVec)
            xb = np.zeros(self.nCon)
            for con in range(self.nCon):
                originb = dCdOrigin[con, 0, :]
                coordsb = dCdPt[con, :, :]
                axisb = dCdAxis[con, 0, :]
                xb[:] = 0.0
                xb[con] = 1.0

                for i in range(self.nCon):
                    if tmpX[i] == 0.0:
                        xb[i] = 0.0
                    else:
                        xb[i] = xb[i] / (2.0 * np.sqrt(tmpX[i]))

                    resultdirb[:] = 0.0
                    for j in reversed(range(3)):  # DO j=3,1,-1
                        resultdirb[j] = resultdirb[j] + 2 * resultDir[i, j] * xb[i]

                    xb[i] = 0.0
                    # CALL CROSS_B(axis, axisb, dirvec(i, :), dirvecb(i, :), resultdirb)
                    axisb, dirvecb[i, :] = geo_utils.cross_b(self.axis[0, :], dirVec[i, :], resultdirb)

                # coordsb = 0.0
                # originb = 0.0
                for i in reversed(range(len(coordsb))):  # DO i=n,1,-1
                    for j in reversed(range(3)):  # DO j=3,1,-1
                        originb[j] = originb[j] + dirvecb[i, j]
                        coordsb[i, j] = coordsb[i, j] - dirvecb[i, j]
                        dirvecb[i, j] = 0.0

            tmpPt = self.DVGeo.totalSensitivity(dCdPt, self.name + "coords", config=config)
            tmpOrigin = self.DVGeo.totalSensitivity(dCdOrigin, self.name + "origin", config=config)

            tmpTotal = {}
            for key in tmpPt:
                tmpTotal[key] = tmpPt[key] + tmpOrigin[key]

            tmpTotal[self.name + "axis"] = dCdAxis

            funcsSens[self.name] = tmpTotal

    def addVariablesPyOpt(self, optProb):
        """
        Add the axis variable for the colinearity constraint to pyOpt
        """

        if self.addVarToPyOpt:
            optProb.addVarGroup(
                self.name, self.nVal, "c", value=self.value, lower=self.lower, upper=self.upper, scale=self.scale
            )

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """
        handle.write("Zone T=%s_coords\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords) + 1, len(self.coords)))
        handle.write("DATAPACKING=POINT\n")
        handle.write("%f %f %f\n" % (self.origin[0, 0], self.origin[0, 1], self.origin[0, 2]))
        for i in range(len(self.coords)):
            handle.write("%f %f %f\n" % (self.coords[i, 0], self.coords[i, 1], self.coords[i, 2]))

        for i in range(len(self.coords)):
            handle.write("%d %d\n" % (i + 1, i + 2))

    def _computeDist(self, origin, coords, axis, dtype="d"):
        """
        compute the distance of coords from the defined axis.
        """
        # Compute the direction from each point to the origin
        dirVec = origin - coords

        # compute the cross product with the desired axis. Cross product
        # will be zero if the direction vector is the same as the axis
        resultDir = np.cross(axis, dirVec)

        X = np.zeros(len(coords), dtype)
        for i in range(len(resultDir)):
            X[i] = geo_utils.euclideanNorm(resultDir[i, :])

        return X
