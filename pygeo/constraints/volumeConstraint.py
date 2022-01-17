# ======================================================================
#         Imports
# ======================================================================
import numpy as np
from .baseConstraint import GeometricConstraint


class VolumeConstraint(GeometricConstraint):
    """
    This class is used to represet a single volume constraint. The
    parameter list is explained in the addVolumeConstaint() of
    the DVConstraints class
    """

    def __init__(self, name, nSpan, nChord, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt):
        super().__init__(name, 1, lower, upper, scale, DVGeo, addToPyOpt)

        self.nSpan = nSpan
        self.nChord = nChord
        self.coords = coords
        self.scaled = scaled
        self.flipVolume = False

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name)

        # Now get the reference volume
        self.V0 = self.evalVolume()

    def evalFunctions(self, funcs, config):
        """
        Evaluate the function this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        V = self.evalVolume()
        if self.scaled:
            V /= self.V0
        funcs[self.name] = V

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
            dVdPt = self.evalVolumeSens()
            if self.scaled:
                dVdPt /= self.V0

            # Now compute the DVGeo total sensitivity:
            funcsSens[self.name] = self.DVGeo.totalSensitivity(dVdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this volume constraint
        """
        # Reshape coordinates back to 3D format
        x = self.coords.reshape([self.nSpan, self.nChord, 2, 3])

        handle.write('ZONE T="%s" I=%d J=%d K=%d\n' % (self.name, self.nSpan, self.nChord, 2))
        handle.write("DATAPACKING=POINT\n")
        for k in range(2):
            for j in range(self.nChord):
                for i in range(self.nSpan):
                    handle.write(f"{x[i, j, k, 0]:f} {x[i, j, k, 1]:f} {x[i, j, k, 2]:f}\n")

    def evalVolume(self):
        """
        Evaluate the total volume of the current coordinates
        """
        Volume = 0.0
        x = self.coords.reshape((self.nSpan, self.nChord, 2, 3))
        for j in range(self.nChord - 1):
            for i in range(self.nSpan - 1):
                Volume += self.evalVolumeHex(
                    x[i, j, 0],
                    x[i + 1, j, 0],
                    x[i, j + 1, 0],
                    x[i + 1, j + 1, 0],
                    x[i, j, 1],
                    x[i + 1, j, 1],
                    x[i, j + 1, 1],
                    x[i + 1, j + 1, 1],
                )

        if Volume < 0:
            Volume = -Volume
            self.flipVolume = True

        return Volume

    def evalVolumeSens(self):
        """
        Evaluate the derivative of the volume with respect to the
        coordinates
        """
        x = self.coords.reshape((self.nSpan, self.nChord, 2, 3))
        xb = np.zeros_like(x)
        for j in range(self.nChord - 1):
            for i in range(self.nSpan - 1):
                self.evalVolumeHex_b(
                    x[i, j, 0],
                    x[i + 1, j, 0],
                    x[i, j + 1, 0],
                    x[i + 1, j + 1, 0],
                    x[i, j, 1],
                    x[i + 1, j, 1],
                    x[i, j + 1, 1],
                    x[i + 1, j + 1, 1],
                    xb[i, j, 0],
                    xb[i + 1, j, 0],
                    xb[i, j + 1, 0],
                    xb[i + 1, j + 1, 0],
                    xb[i, j, 1],
                    xb[i + 1, j, 1],
                    xb[i, j + 1, 1],
                    xb[i + 1, j + 1, 1],
                )
        # We haven't divided by 6.0 yet...lets do it here....
        xb /= 6.0

        if self.flipVolume:
            xb = -xb

        # Reshape back to flattened array for DVGeo
        xb = xb.reshape((self.nSpan * self.nChord * 2, 3))

        return xb

    def evalVolumeHex(self, x0, x1, x2, x3, x4, x5, x6, x7):
        """
        Evaluate the volume of the hexahedral volume defined by the
        the 8 corners.

        Parameters
        ----------
        x{0:7} : arrays or size (3)
            Array of defining the coordinates of the volume
        """

        p = np.average([x0, x1, x2, x3, x4, x5, x6, x7], axis=0)
        V = 0.0
        V += self.volpym(x0, x1, x3, x2, p)
        V += self.volpym(x0, x2, x6, x4, p)
        V += self.volpym(x0, x4, x5, x1, p)
        V += self.volpym(x1, x5, x7, x3, p)
        V += self.volpym(x2, x3, x7, x6, p)
        V += self.volpym(x4, x6, x7, x5, p)
        V /= 6.0

        return V

    def volpym(self, a, b, c, d, p):
        """
        Compute volume of a square-based pyramid
        """
        fourth = 1.0 / 4.0

        volpym = (
            (p[0] - fourth * (a[0] + b[0] + c[0] + d[0]))
            * ((a[1] - c[1]) * (b[2] - d[2]) - (a[2] - c[2]) * (b[1] - d[1]))
            + (p[1] - fourth * (a[1] + b[1] + c[1] + d[1]))
            * ((a[2] - c[2]) * (b[0] - d[0]) - (a[0] - c[0]) * (b[2] - d[2]))
            + (p[2] - fourth * (a[2] + b[2] + c[2] + d[2]))
            * ((a[0] - c[0]) * (b[1] - d[1]) - (a[1] - c[1]) * (b[0] - d[0]))
        )

        return volpym

    def evalVolumeHex_b(self, x0, x1, x2, x3, x4, x5, x6, x7, x0b, x1b, x2b, x3b, x4b, x5b, x6b, x7b):
        """
        Evaluate the derivative of the volume defined by the 8
        coordinates in the array x.

        Parameters
        ----------
        x{0:7} : arrays of len 3
            Arrays of defining the coordinates of the volume

        Returns
        -------
        xb{0:7} : arrays of len 3
            Derivatives of the volume wrt the points.
        """

        p = np.average([x0, x1, x2, x3, x4, x5, x6, x7], axis=0)
        pb = np.zeros(3)
        self.volpym_b(x0, x1, x3, x2, p, x0b, x1b, x3b, x2b, pb)
        self.volpym_b(x0, x2, x6, x4, p, x0b, x2b, x6b, x4b, pb)
        self.volpym_b(x0, x4, x5, x1, p, x0b, x4b, x5b, x1b, pb)
        self.volpym_b(x1, x5, x7, x3, p, x1b, x5b, x7b, x3b, pb)
        self.volpym_b(x2, x3, x7, x6, p, x2b, x3b, x7b, x6b, pb)
        self.volpym_b(x4, x6, x7, x5, p, x4b, x6b, x7b, x5b, pb)

        pb /= 8.0
        x0b += pb
        x1b += pb
        x2b += pb
        x3b += pb
        x4b += pb
        x5b += pb
        x6b += pb
        x7b += pb

    def volpym_b(self, a, b, c, d, p, ab, bb, cb, db, pb):
        """
        Compute the reverse-mode derivative of the square-based
        pyramid. This has been copied from reverse-mode AD'ed tapenade
        fortran code and converted to python to use vectors for the
        points.
        """
        fourth = 1.0 / 4.0
        volpymb = 1.0
        tempb = ((a[1] - c[1]) * (b[2] - d[2]) - (a[2] - c[2]) * (b[1] - d[1])) * volpymb
        tempb0 = -(fourth * tempb)
        tempb1 = (p[0] - fourth * (a[0] + b[0] + c[0] + d[0])) * volpymb
        tempb2 = (b[2] - d[2]) * tempb1
        tempb3 = (a[1] - c[1]) * tempb1
        tempb4 = -((b[1] - d[1]) * tempb1)
        tempb5 = -((a[2] - c[2]) * tempb1)
        tempb6 = ((a[2] - c[2]) * (b[0] - d[0]) - (a[0] - c[0]) * (b[2] - d[2])) * volpymb
        tempb7 = -(fourth * tempb6)
        tempb8 = (p[1] - fourth * (a[1] + b[1] + c[1] + d[1])) * volpymb
        tempb9 = (b[0] - d[0]) * tempb8
        tempb10 = (a[2] - c[2]) * tempb8
        tempb11 = -((b[2] - d[2]) * tempb8)
        tempb12 = -((a[0] - c[0]) * tempb8)
        tempb13 = ((a[0] - c[0]) * (b[1] - d[1]) - (a[1] - c[1]) * (b[0] - d[0])) * volpymb
        tempb14 = -(fourth * tempb13)
        tempb15 = (p[2] - fourth * (a[2] + b[2] + c[2] + d[2])) * volpymb
        tempb16 = (b[1] - d[1]) * tempb15
        tempb17 = (a[0] - c[0]) * tempb15
        tempb18 = -((b[0] - d[0]) * tempb15)
        tempb19 = -((a[1] - c[1]) * tempb15)
        pb[0] = pb[0] + tempb
        ab[0] = ab[0] + tempb16 + tempb11 + tempb0
        bb[0] = bb[0] + tempb19 + tempb10 + tempb0
        cb[0] = cb[0] + tempb0 - tempb11 - tempb16
        db[0] = db[0] + tempb0 - tempb10 - tempb19
        ab[1] = ab[1] + tempb18 + tempb7 + tempb2
        cb[1] = cb[1] + tempb7 - tempb18 - tempb2
        bb[2] = bb[2] + tempb14 + tempb12 + tempb3
        db[2] = db[2] + tempb14 - tempb12 - tempb3
        ab[2] = ab[2] + tempb14 + tempb9 + tempb4
        cb[2] = cb[2] + tempb14 - tempb9 - tempb4
        bb[1] = bb[1] + tempb17 + tempb7 + tempb5
        db[1] = db[1] + tempb7 - tempb17 - tempb5
        pb[1] = pb[1] + tempb6
        pb[2] = pb[2] + tempb13


class TriangulatedVolumeConstraint(GeometricConstraint):
    """
    This class is used to compute a volume constraint based on triangulated surface mesh geometry
    """

    def __init__(self, name, surface, surface_name, lower, upper, scaled, scale, DVGeo, addToPyOpt):
        super().__init__(name, 1, lower, upper, scale, DVGeo, addToPyOpt)

        self.surface = surface
        self.surface_name = surface_name
        self.surf_size = surface[0].shape[0]
        self.surf_p0 = surface[0].reshape(self.surf_size, 3)
        self.surf_p1 = surface[1].reshape(self.surf_size, 3)
        self.surf_p2 = surface[2].reshape(self.surf_size, 3)
        self.scaled = scaled
        self.vol_0 = None

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
        self.surf_p0 = self.DVGeo.update(self.surface_name + "_p0", config=config).reshape(self.surf_size, 3)
        self.surf_p1 = self.DVGeo.update(self.surface_name + "_p1", config=config).reshape(self.surf_size, 3)
        self.surf_p2 = self.DVGeo.update(self.surface_name + "_p2", config=config).reshape(self.surf_size, 3)

        volume = self.compute_volume(self.surf_p0, self.surf_p1, self.surf_p2)
        if self.vol_0 is None:
            self.vol_0 = volume

        if self.scaled:
            volume = 1.0 * volume / self.vol_0

        funcs[self.name] = volume

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Assumes that evalFunctions method was called just prior
        and results stashed on proc 0

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        config : str
            The DVGeo configuration to apply this constraint to. Must be either None
            which will apply to *ALL* the local DV groups or a single string specifying
            a particular configuration.
        """
        tmpTotal = {}

        # assume evalFunctions was called just prior and grad was stashed on rank=0
        grad_vol = self.compute_volume_sens(self.surf_p0, self.surf_p1, self.surf_p2)
        if self.scaled:
            tmp_p0 = self.DVGeo.totalSensitivity(grad_vol[0] / self.vol_0, self.surface_name + "_p0", config=config)
            tmp_p1 = self.DVGeo.totalSensitivity(grad_vol[1] / self.vol_0, self.surface_name + "_p1", config=config)
            tmp_p2 = self.DVGeo.totalSensitivity(grad_vol[2] / self.vol_0, self.surface_name + "_p2", config=config)
        else:
            tmp_p0 = self.DVGeo.totalSensitivity(grad_vol[0], self.surface_name + "_p0", config=config)
            tmp_p1 = self.DVGeo.totalSensitivity(grad_vol[1], self.surface_name + "_p1", config=config)
            tmp_p2 = self.DVGeo.totalSensitivity(grad_vol[2], self.surface_name + "_p2", config=config)

        for key in tmp_p0:
            tmpTotal[key] = tmp_p0[key] + tmp_p1[key] + tmp_p2[key]

        funcsSens[self.name] = tmpTotal

    def compute_volume(self, p0, p1, p2):
        """
        Compute the volume of a triangulated volume by computing
        the signed areas.

        The method is described in, among other places,
        EFFICIENT FEATURE EXTRACTION FOR 2D/3D OBJECTS IN MESH REPRESENTATION
        by Cha Zhang and Tsuhan Chen,
        http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf

        Parameters
        ----------
        p0, p1, p2 : arrays
            Coordinates of the vertices of the triangulated mesh

        Returns
        -------
        volume : float
            The volume of the triangulated surface
        """

        volume = (
            np.sum(
                p1[:, 0] * p2[:, 1] * p0[:, 2]
                + p2[:, 0] * p0[:, 1] * p1[:, 2]
                + p0[:, 0] * p1[:, 1] * p2[:, 2]
                - p1[:, 0] * p0[:, 1] * p2[:, 2]
                - p2[:, 0] * p1[:, 1] * p0[:, 2]
                - p0[:, 0] * p2[:, 1] * p1[:, 2]
            )
            / 6.0
        )

        return volume

    def compute_volume_sens(self, p0, p1, p2):
        """
        Compute the gradients of the volume with respect to
        the mesh vertices.

        Parameters
        ----------
        p0, p1, p2 : arrays
            Coordinates of the vertices of the triangulated mesh

        Returns
        -------
        grad_0, grad_1, grad_2 : arrays
            Gradients of volume with respect to vertex coordinates
        """

        num_pts = p0.shape[0]
        grad_0 = np.zeros((num_pts, 3))
        grad_1 = np.zeros((num_pts, 3))
        grad_2 = np.zeros((num_pts, 3))

        grad_0[:, 0] = p1[:, 1] * p2[:, 2] - p1[:, 2] * p2[:, 1]
        grad_0[:, 1] = p1[:, 2] * p2[:, 0] - p1[:, 0] * p2[:, 2]
        grad_0[:, 2] = p1[:, 0] * p2[:, 1] - p1[:, 1] * p2[:, 0]

        grad_1[:, 0] = p0[:, 2] * p2[:, 1] - p0[:, 1] * p2[:, 2]
        grad_1[:, 1] = p0[:, 0] * p2[:, 2] - p0[:, 2] * p2[:, 0]
        grad_1[:, 2] = p0[:, 1] * p2[:, 0] - p0[:, 0] * p2[:, 1]

        grad_2[:, 0] = p0[:, 1] * p1[:, 2] - p0[:, 2] * p1[:, 1]
        grad_2[:, 1] = p0[:, 2] * p1[:, 0] - p0[:, 0] * p1[:, 2]
        grad_2[:, 2] = p0[:, 0] * p1[:, 1] - p0[:, 1] * p1[:, 0]

        return grad_0 / 6.0, grad_1 / 6.0, grad_2 / 6.0

    def writeTecplot(self, handle):
        raise NotImplementedError()


class CompositeVolumeConstraint(GeometricConstraint):
    """This class is used to represet a single volume constraints that is a
    group of other VolumeConstraints.
    """

    def __init__(self, name, vols, lower, upper, scaled, scale, DVGeo, addToPyOpt):
        super().__init__(name, 1, lower, upper, scale, DVGeo, addToPyOpt)

        self.vols = vols
        self.scaled = scaled

        # Now get the reference volume
        self.V0 = 0.0
        for vol in self.vols:
            self.V0 += vol.evalVolume()

    def evalFunctions(self, funcs, config):
        """
        Evaluate the function this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        V = 0.0
        for vol in self.vols:
            V += vol.evalVolume()
        if self.scaled:
            V /= self.V0
        funcs[self.name] = V

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
            tmp = []  # List of dict derivatives
            for vol in self.vols:
                dVdPt = vol.evalVolumeSens()
                if self.scaled:
                    dVdPt /= self.V0
                tmp.append(vol.DVGeo.totalSensitivity(dVdPt, vol.name, config=config))

            # Now we need to sum up the derivatives:
            funcsSens[self.name] = tmp[0]
            for i in range(1, len(tmp)):
                for key in tmp[i]:
                    funcsSens[self.name][key] += tmp[i][key]

    def writeTecplot(self, handle):
        raise NotImplementedError()
