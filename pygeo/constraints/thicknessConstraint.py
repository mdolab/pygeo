# External modules
import numpy as np

# Local modules
from .. import geo_utils
from .baseConstraint import GeometricConstraint


class ThicknessConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of thickness
    constraints. One of these objects is created each time a
    addThicknessConstraints2D or addThicknessConstraints1D call is
    made. The user should not have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords) // 2, lower, upper, scale, DVGeo, addToPyOpt)

        self.coords = coords
        self.scaled = scaled

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths
        self.D0 = np.zeros(self.nCon)
        for i in range(self.nCon):
            self.D0[i] = geo_utils.norm.euclideanNorm(self.coords[2 * i] - self.coords[2 * i + 1])

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        D = np.zeros(self.nCon)
        for i in range(self.nCon):
            D[i] = geo_utils.norm.euclideanNorm(self.coords[2 * i] - self.coords[2 * i + 1])
            if self.scaled:
                D[i] /= self.D0[i]
        funcs[self.name] = D

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
            dTdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))

            for i in range(self.nCon):
                p1b, p2b = geo_utils.eDist_b(self.coords[2 * i, :], self.coords[2 * i + 1, :])
                if self.scaled:
                    p1b /= self.D0[i]
                    p2b /= self.D0[i]
                dTdPt[i, 2 * i, :] = p1b
                dTdPt[i, 2 * i + 1, :] = p2b

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dTdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) // 2))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class ProjectedThicknessConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of projected thickness
    constraints. One of these objects is created each time a
    addThicknessConstraints2D or addThicknessConstraints1D call is
    made. The user should not have to deal with this class directly.

    This is different from ThicknessConstraints becuase it measures the projected
    thickness along the orginal direction of the constraint.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords) // 2, lower, upper, scale, DVGeo, addToPyOpt)

        self.coords = coords
        self.scaled = scaled

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths and directions
        self.D0 = np.zeros(self.nCon)
        self.dir_vec = np.zeros((self.nCon, 3))
        for i in range(self.nCon):
            vec = self.coords[2 * i] - self.coords[2 * i + 1]
            self.D0[i] = geo_utils.norm.euclideanNorm(vec)
            self.dir_vec[i] = vec / self.D0[i]

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        D = np.zeros(self.nCon)
        for i in range(self.nCon):
            vec = self.coords[2 * i] - self.coords[2 * i + 1]

            # take the dot product with the direction vector
            D[i] = vec[0] * self.dir_vec[i, 0] + vec[1] * self.dir_vec[i, 1] + vec[2] * self.dir_vec[i, 2]

            if self.scaled:
                D[i] /= self.D0[i]

        funcs[self.name] = D

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
            dTdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))
            for i in range(self.nCon):
                D_b = 1.0

                # the reverse mode seeds still need to be scaled
                if self.scaled:
                    D_b /= self.D0[i]

                # d(dot(vec,n))/d(vec) = n
                # where vec = thickness vector
                #   and  n = the reference direction
                #  This is easier to see if you write out the dot product
                # dot(vec, n) = vec_1*n_1 + vec_2*n_2 + vec_3*n_3
                # d(dot(vec,n))/d(vec_1) = n_1
                # d(dot(vec,n))/d(vec_2) = n_2
                # d(dot(vec,n))/d(vec_3) = n_3
                vec_b = self.dir_vec[i] * D_b

                # the reverse mode of calculating vec is just scattering the seed of vec_b to the coords
                # vec = self.coords[2 * i] - self.coords[2 * i + 1]
                # we just set the coordinate seeds directly into the jacobian
                dTdPt[i, 2 * i, :] = vec_b
                dTdPt[i, 2 * i + 1, :] = -vec_b

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dTdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) // 2))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))

        # create a seperate zone to plot the projected direction for each thickness constraint
        handle.write("Zone T=%s_ref_directions\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.dir_vec) * 2, len(self.dir_vec)))
        handle.write("DATAPACKING=POINT\n")

        for i in range(self.nCon):
            pt1 = self.coords[i * 2 + 1]
            pt2 = pt1 + self.dir_vec[i]
            handle.write(f"{pt1[0]:f} {pt1[1]:f} {pt1[2]:f}\n")
            handle.write(f"{pt2[0]:f} {pt2[1]:f} {pt2[2]:f}\n")

        for i in range(self.nCon):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class ThicknessToChordConstraint(GeometricConstraint):
    """These are almost identical to ThicknessConstraint but track and additional set of points along the leading and
    trailing edges used to compute a chord length for scaling the thickness.
    """

    def __init__(
        self,
        name,
        thicknessCoords,
        LeTeCoords,
        lower,
        upper,
        scaled,
        scale,
        DVGeo,
        addToPyOpt,
        compNames,
        sectionMax=False,
        ksRho=50.0,
    ):
        """Instantiate a ThicknessToChordConstraint object, this should not be called directly by the user but instead
        by addThicknessToChordConstraints1D or addThicknessToChordConstraints2D.

        Parameters
        ----------
        name : str
            See :meth:`GeometricConstraint.__init__ <.baseConstraint.GeometricConstraint.__init__>`.
        thicknessCoords : numpy array of shape (..., nSpan, 2, 3)
            Coordinates of the points used to measure thickness.
        LeTeCoords : numpy array of shape (nSpan, 2, 3)
            Coordinates of the leading and trailing edge points used to measure chord length.
        lower : float, array, or None
            See :meth:`GeometricConstraint.__init__ <.baseConstraint.GeometricConstraint.__init__>`.
        upper : float, array, or None
            See :meth:`GeometricConstraint.__init__ <.baseConstraint.GeometricConstraint.__init__>`.
        scaled : bool
            If true, the constraint values are normalized by their initial values.
        scale : _type_
            See :meth:`GeometricConstraint.__init__ <.baseConstraint.GeometricConstraint.__init__>`.
        DVGeo : _type_
            See :meth:`GeometricConstraint.__init__ <.baseConstraint.GeometricConstraint.__init__>`.
        addToPyOpt : _type_
            See :meth:`GeometricConstraint.__init__ <.baseConstraint.GeometricConstraint.__init__>`.
        compNames : list or None
            If using DVGeometryMulti, the components to which the point set associated with this constraint should be
            added. If None, the point set is added to all components.
        sectionMax : bool
            If True, computes the maximum thickness-to-chord ratio in each section using KS aggregation.
        ksRho : float
            The rho value to use for KS aggregation if ``sectionMax=True``
        """
        self.sectionMax = sectionMax
        self.ksRho = ksRho
        self.scaled = scaled

        # TODO: Alter this to deal with 1d or 2d sets of points
        self.numSpanPoints = thicknessCoords.shape[0]
        # The input coordinates may not have a chordwise dimension, so handle that here
        if len(thicknessCoords.shape) == 3:
            thicknessCoords = thicknessCoords.reshape((self.numSpanPoints, 1, 2, 3))
        self.origCoordsShape = thicknessCoords.shape
        self.numChordPoints = self.origCoordsShape[1]

        # No point in doing section max if there's only one chordwise point
        if self.numChordPoints == 1:
            self.sectionMax = False

        numCon = self.numSpanPoints if sectionMax else self.numSpanPoints * self.numChordPoints
        super().__init__(name, numCon, lower, upper, scale, DVGeo, addToPyOpt)

        # Verify that we have the right number of leading and trailing edge points
        self.origLeTeShape = LeTeCoords.shape
        if self.origLeTeShape != (self.numSpanPoints, 2, 3):
            raise ValueError(
                f"LeTeCoords has incorrect shape, should be ({self.numSpanPoints}, 2, 3), got {LeTeCoords.shape}"
            )

        self.thicknessConName = name + "_thickness"
        self.chordConName = name + "_chord"

        # Create a ThicknessConstraint object to handle the thickness part of this constraint
        self.thicknessConstraint = ThicknessConstraint(
            name=self.thicknessConName,
            coords=thicknessCoords.reshape((-1, 3)),
            lower=-np.inf,
            upper=np.inf,
            scaled=False,
            scale=scale,
            DVGeo=DVGeo,
            addToPyOpt=False,
            compNames=compNames,
        )

        # Create another thickness constraint to handle the chord length part of this constraint
        self.chordConstraint = ThicknessConstraint(
            name=self.chordConName,
            coords=LeTeCoords.reshape((-1, 3)),
            lower=-np.inf,
            upper=np.inf,
            scaled=False,
            scale=scale,
            DVGeo=DVGeo,
            addToPyOpt=False,
            compNames=compNames,
        )

        # Compute the initial thickness-to-chord ratios if we are scaling
        self.tOverCInit = None
        if scaled:
            funcs = {}
            self.evalFunctions(funcs, config=None)
            self.tOverCInit = funcs[self.name].reshape((self.numSpanPoints, self.numChordPoints))

    def evalFunctions(self, funcs, config):
        thickness, chord = self.computeThicknessAndChord(config)
        tOverC = thickness / chord
        if self.sectionMax:
            tOverC = self.ksMax(tOverC, self.ksRho, axis=1)

        if self.scaled and self.tOverCInit is not None:
            tOverC /= self.tOverCInit

        funcs[self.name] = tOverC.flatten()

    def computeThicknessAndChord(self, config):
        tempFuncs = {}
        self.thicknessConstraint.evalFunctions(tempFuncs, config)
        self.chordConstraint.evalFunctions(tempFuncs, config)

        thickness = tempFuncs[self.thicknessConName].reshape((self.numSpanPoints, self.numChordPoints))
        chord = tempFuncs[self.chordConName].reshape((self.numSpanPoints, 1))
        return thickness, chord

    def computeThicknessAndChordSens(self, config):
        tempFuncsSens = {}
        self.thicknessConstraint.evalFunctionsSens(tempFuncsSens, config)
        self.chordConstraint.evalFunctionsSens(tempFuncsSens, config)

        for dvName in self.getVarNames():
            numDVs = tempFuncsSens[self.thicknessConName][dvName].shape[-1]
            tempFuncsSens[self.thicknessConName][dvName] = tempFuncsSens[self.thicknessConName][dvName].reshape(
                (self.numSpanPoints, self.numChordPoints, numDVs)
            )
            tempFuncsSens[self.chordConName][dvName] = tempFuncsSens[self.chordConName][dvName].reshape(
                (self.numSpanPoints, 1, numDVs)
            )
        return tempFuncsSens[self.thicknessConName], tempFuncsSens[self.chordConName]

    def evalFunctionsSens(self, funcsSens, config):
        thickness, chord = self.computeThicknessAndChord(config)
        thicknessSens, chordSens = self.computeThicknessAndChordSens(config)

        funcsSens[self.name] = {}
        tOverCSens = funcsSens[self.name]
        dvNames = set(thicknessSens.keys()).union(set(chordSens.keys()))
        for dvName in dvNames:
            dtdx = thicknessSens[dvName]
            dcdx = chordSens[dvName]
            numDVs = dtdx.shape[-1]

            # Quotient rule says d(t/c)dx = (dtdx*c - t*dcdx) / c^2
            tOverCSens[dvName] = (dtdx * chord[:, :, np.newaxis] - thickness[:, :, np.newaxis] * dcdx) / (chord**2)[
                :, :, np.newaxis
            ]

            if self.sectionMax:
                dksdtc = self.ksMaxSens(thickness / chord, self.ksRho, axis=1)  # shape (nSpan, nChord)
                tOverCSens[dvName] = np.einsum("sc,scd->sd", dksdtc, tOverCSens[dvName])

            tOverCSens[dvName] = tOverCSens[dvName].reshape(-1, numDVs)
            if self.scaled:
                tOverCSens[dvName] /= self.tOverCInit.flatten()[:, np.newaxis]

    def writeTecplot(self, handle):
        self.thicknessConstraint.writeTecplot(handle)
        self.chordConstraint.writeTecplot(handle)

    @staticmethod
    def ksMax(f, rho, axis=None):
        """Approximate the maximum value of f along the given axis using KS aggregation.

        Parameters
        ----------
        f : array
            Input data
        rho : float
            KS aggregation parameter, larger values more closely approximate the max but are less smooth.
        axis : int, optional
            The dimension to compute the max along, by default computes max across entire array.
        """
        fMax = np.max(f, axis=axis, keepdims=True)
        exponents = np.exp(rho * (f - fMax))
        ks = fMax + np.log(np.sum(exponents, axis=axis, keepdims=True)) / rho
        return ks

    @staticmethod
    def ksMaxSens(f, rho, axis=None):
        """Compute the sensitivity of the KS max function.

        Parameters
        ----------
        f : array
            Input data
        rho : float
            KS aggregation parameter, larger values more closely approximate the max but are less smooth.
        axis : int, optional
            The dimension to compute the max along, by default computes max across entire array.
        """
        fMax = np.max(f, axis=axis, keepdims=True)
        exponents = np.exp(rho * (f - fMax))
        sumExp = np.sum(exponents, axis=axis, keepdims=True)
        dksdf = exponents / sumExp
        return dksdf


class ProximityConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of proximity
    constraints. The user should not have to deal with this
    class directly.
    """

    def __init__(
        self,
        name,
        coordsA,
        coordsB,
        pointSetKwargsA,
        pointSetKwargsB,
        lower,
        upper,
        scaled,
        scale,
        DVGeo,
        addToPyOpt,
        compNames,
    ):
        super().__init__(name, len(coordsA), lower, upper, scale, DVGeo, addToPyOpt)

        self.coordsA = coordsA
        self.coordsB = coordsB
        self.scaled = scaled

        # First thing we can do is embed the coordinates into the DVGeo.
        # ptsets A and B get different kwargs
        self.DVGeo.addPointSet(self.coordsA, f"{self.name}_A", compNames=compNames, **pointSetKwargsA)
        self.DVGeo.addPointSet(self.coordsB, f"{self.name}_B", compNames=compNames, **pointSetKwargsB)

        # Now get the reference lengths
        self.D0 = np.zeros(self.nCon)
        for i in range(self.nCon):
            self.D0[i] = geo_utils.norm.euclideanNorm(self.coordsA[i] - self.coordsB[i])

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coordsA = self.DVGeo.update(f"{self.name}_A", config=config)
        self.coordsB = self.DVGeo.update(f"{self.name}_B", config=config)
        D = np.zeros(self.nCon)
        for i in range(self.nCon):
            D[i] = geo_utils.norm.euclideanNorm(self.coordsA[i] - self.coordsB[i])
            if self.scaled:
                D[i] /= self.D0[i]
        funcs[self.name] = D

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
            dTdPtA = np.zeros((self.nCon, self.nCon, 3))
            dTdPtB = np.zeros((self.nCon, self.nCon, 3))

            for i in range(self.nCon):
                pAb, pBb = geo_utils.eDist_b(self.coordsA[i], self.coordsB[i])
                if self.scaled:
                    pAb /= self.D0[i]
                    pBb /= self.D0[i]
                dTdPtA[i, i, :] = pAb
                dTdPtB[i, i, :] = pBb

            funcSensA = self.DVGeo.totalSensitivity(dTdPtA, f"{self.name}_A", config=config)
            funcSensB = self.DVGeo.totalSensitivity(dTdPtB, f"{self.name}_B", config=config)

            funcsSens[self.name] = {}
            for key, value in funcSensA.items():
                funcsSens[self.name][key] = value
            for key, value in funcSensB.items():
                if key in funcsSens[self.name]:
                    funcsSens[self.name][key] += value
                else:
                    funcsSens[self.name][key] = value

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coordsA) * 2, len(self.coordsA)))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coordsA)):
            handle.write(f"{self.coordsA[i, 0]:f} {self.coordsA[i, 1]:f} {self.coordsA[i, 2]:f}\n")
            handle.write(f"{self.coordsB[i, 0]:f} {self.coordsB[i, 1]:f} {self.coordsB[i, 2]:f}\n")

        for i in range(len(self.coordsA)):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))
