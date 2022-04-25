# ======================================================================
#         Imports
# ======================================================================
import numpy as np
from .baseConstraint import GeometricConstraint
from ..geo_utils.polygon import (
    volumeHex,
    volumeHex_b,
    volumeTriangulatedMesh,
    volumeTriangulatedMesh_b,
)


class VolumeConstraint(GeometricConstraint):
    """
    This class is used to represet a single volume constraint. The
    parameter list is explained in the addVolumeConstaint() of
    the DVConstraints class
    """

    def __init__(self, name, nSpan, nChord, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, 1, lower, upper, scale, DVGeo, addToPyOpt)

        self.nSpan = nSpan
        self.nChord = nChord
        self.coords = coords
        self.scaled = scaled
        self.flipVolume = False

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

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
                Volume += volumeHex(
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
                volumeHex_b(
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

        volume = volumeTriangulatedMesh(self.surf_p0, self.surf_p1, self.surf_p2)
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
        grad_vol = volumeTriangulatedMesh_b(self.surf_p0, self.surf_p1, self.surf_p2)
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
