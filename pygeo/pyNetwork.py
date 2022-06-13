# ======================================================================
#         Imports
# ======================================================================
import os
import numpy as np
from pyspline.utils import openTecplot, writeTecplot1D, closeTecplot, line
from .topology import CurveTopology


class pyNetwork:
    """
    A class for manipulating a collection of curve objects.

    pyNetwork is the 1 dimensional analog of pyGeo (surfaces 2D) and
    pyBlock (volumes 3D). The idea is that a 'network' is a collection
    of 1D splines that are connected in some manner. This module
    provides facility for dealing with such structures.

    Parameters
    ----------
    curves : list of pySpline.Curve objects
        Individual curves to form the network.
    """

    def __init__(self, curves):
        self.curves = curves
        self.nCurve = len(curves)
        self.topo = None
        self.coef = None
        self._doConnectivity()

    def _doConnectivity(self):
        """
        Compute the connectivity of the set of curve objects.
        """
        coords = np.zeros((self.nCurve, 2, 3))
        for icurve in range(self.nCurve):
            coords[icurve][0] = self.curves[icurve](0)
            coords[icurve][1] = self.curves[icurve](1)

        self.topo = CurveTopology(coords=coords)

        sizes = []
        for icurve in range(self.nCurve):
            sizes.append(self.curves[icurve].nCtl)
        self.topo.calcGlobalNumbering(sizes)

        self.coef = np.zeros((self.topo.nGlobal, 3))
        for i in range(len(self.coef)):
            icurve = self.topo.gIndex[i][0][0]
            ii = self.topo.gIndex[i][0][1]
            self.coef[i] = self.curves[icurve].coef[ii]

    # ----------------------------------------------------------------------
    #               Curve Writing Output Functions
    # ----------------------------------------------------------------------

    def writeTecplot(self, fileName, orig=False, curves=True, coef=True, curveLabels=False, nodeLabels=False):
        """Write the pyNetwork Object to Tecplot .dat file

        Parameters
        ----------
        fileName : str
            File name for tecplot file. Should have .dat extension
        curves : bool
            Flag to write discrete approximation of the actual curve
        coef : bool
            Flag to write b-spline coefficients
        curveLabels : bool
            Flag to write a separate label file with the curve indices
        nodeLabels : bool
            Flag to write a separate node label file with the node indices
        """

        f = openTecplot(fileName, 3)
        if curves:
            for icurve in range(self.nCurve):
                self.curves[icurve].computeData()
                writeTecplot1D(f, "interpolated", self.curves[icurve].data)
        if coef:
            for icurve in range(self.nCurve):
                writeTecplot1D(f, "coef", self.curves[icurve].coef)
        if orig:
            for icurve in range(self.nCurve):
                writeTecplot1D(f, "coef", self.curves[icurve].X)

        #    Write out The Curve and Node Labels
        dirName, fileName = os.path.split(fileName)
        fileBaseName, _ = os.path.splitext(fileName)

        if curveLabels:
            # Split the filename off
            labelFilename = dirName + "./" + fileBaseName + ".curve_labels.dat"
            f2 = open(labelFilename, "w")
            for icurve in range(self.nCurve):
                mid = np.floor(self.curves[icurve].nCtl / 2)
                textString = 'TEXT CS=GRID3D, X=%f,Y=%f,Z=%f,ZN=%d,T="S%d"\n' % (
                    self.curves[icurve].coef[mid, 0],
                    self.curves[icurve].coef[mid, 1],
                    self.curves[icurve].coef[mid, 2],
                    icurve + 1,
                    icurve,
                )
                f2.write("%s" % (textString))
            f2.close()

        if nodeLabels:
            # First we need to figure out where the corners actually *are*
            nNodes = len(np.unique(self.topo.nodeLink.flatten()))
            nodeCoord = np.zeros((nNodes, 3))
            for i in range(nNodes):
                # Try to find node i
                for icurve in range(self.nCurve):
                    if self.topo.nodeLink[icurve][0] == i:
                        coordinate = self.curves[icurve].getValueCorner(0)
                        break
                    elif self.topo.nodeLink[icurve][1] == i:
                        coordinate = self.curves[icurve].getValueCorner(1)
                        break
                    elif self.topo.nodeLink[icurve][2] == i:
                        coordinate = self.curves[icurve].getValueCorner(2)
                        break
                    elif self.topo.nodeLink[icurve][3] == i:
                        coordinate = self.curves[icurve].getValueCorner(3)
                        break
                nodeCoord[i] = coordinate

            # Split the filename off
            labelFilename = dirName + "./" + fileBaseName + ".node_labels.dat"
            f2 = open(labelFilename, "w")

            for i in range(nNodes):
                textString = 'TEXT CS=GRID3D, X=%f, Y=%f, Z=%f, T="n%d"\n' % (
                    nodeCoord[i][0],
                    nodeCoord[i][1],
                    nodeCoord[i][2],
                    i,
                )
                f2.write("%s" % (textString))
            f2.close()

        closeTecplot(f)

    def _updateCurveCoef(self):
        """update the coefficents on the pyNetwork update"""
        for ii in range(len(self.coef)):
            for jj in range(len(self.topo.gIndex[ii])):
                icurve = self.topo.gIndex[ii][jj][0]
                i = self.topo.gIndex[ii][jj][1]
                self.curves[icurve].coef[i] = self.coef[ii]

    def getBounds(self, curves=None):
        """Determine the extents of the set of curves.

        Parameters
        ----------
        curves : list
            Optional list of the indices of the curve objects to
            include.

        Returns
        -------
        xMin : array of length 3
            Lower corner of the bounding box
        xMax : array of length 3
            Upper corner of the bounding box
        """
        if curves is None:
            curves = np.arange(self.nCurve)

        Xmin0, Xmax0 = self.curves[curves[0]].getBounds()
        for i in range(1, len(curves)):
            icurve = curves[i]
            Xmin, Xmax = self.curves[icurve].getBounds()
            # Now check them
            if Xmin[0] < Xmin0[0]:
                Xmin0[0] = Xmin[0]
            if Xmin[1] < Xmin0[1]:
                Xmin0[1] = Xmin[1]
            if Xmin[2] < Xmin0[2]:
                Xmin0[2] = Xmin[2]
            if Xmax[0] > Xmax0[0]:
                Xmax0[0] = Xmax[0]
            if Xmax[1] > Xmax0[1]:
                Xmax0[1] = Xmax[1]
            if Xmax[2] > Xmax0[2]:
                Xmax0[2] = Xmax[2]

        return Xmin0, Xmax0

    def projectRays(self, points, axis, curves=None, raySize=1.5, **kwargs):
        """Given a set of points and a vector defining a direction,
        i.e. a ray, determine the minimum distance between these rays
        and any of the curves this object has.

        Parameters
        ----------
        points : array
            A single point (array length 3) or a set of points (N,3) array
        axis : array
            A single direction vector (length 3) or a (N,3) array of direction
            vectors
        curves : list
            An optional list of curve indices to you. If not given, all
            curve objects are used.
        raySize : float
            The ray direction is based on the axis vector. The magnitude of the
            ray is estimated based on the minimum distance between the point and
            the set of curves. That distance is then multiplied by "raySize" to
            get the final ray vector. Then we find the intersection between the
            ray and the curves. If the ray is not long enough to actually
            intersect with any of the curves, then the link will be drawn to the
            location on the curve that is closest to the end of the ray, which
            will not be a projection along "axis" unless the curve is
            perpendicular to the axis vector. The default of 1.5 works in most
            cases but can cause unexpected behavior sometimes which can be fixed
            by increasing the default.
        kwargs : dict
            Keyword arguments passed to Curve.projectCurve() function

        Returns
        -------
        curveID : int
            The index of the curve with the closest distance
        s : float or array
            The curve parameter on self.curves[curveID] that is cloested
            to the point(s).
        """

        # Do point project to determine the approximate distance such
        # that we know how large to make the line representing the ray.
        curveID0, s0 = self.projectPoints(points, curves=curves, **kwargs)

        D0 = np.zeros((len(s0), 3), "d")
        for i in range(len(s0)):
            D0[i, :] = self.curves[curveID0[i]](s0[i]) - points[i]

        if curves is None:
            curves = np.arange(self.nCurve)

        # Now do the same calc as before
        N = len(points)
        S = np.zeros((N, len(curves)))
        D = np.zeros((N, len(curves), 3))

        for i in range(len(curves)):
            icurve = curves[i]
            for j in range(N):
                ray = line(
                    points[j] - axis * raySize * np.linalg.norm(D0[j]),
                    points[j] + axis * raySize * np.linalg.norm(D0[j]),
                )

                S[j, i], t, D[j, i, :] = self.curves[icurve].projectCurve(ray, nIter=2000)
                if t == 0.0 or t == 1.0:
                    print(
                        "Warning: The link for attached point {:d} was drawn"
                        "from the curve to the end of the ray,"
                        "indicating that the ray might not have been long"
                        "enough to intersect the nearest curve.".format(j)
                    )

        s = np.zeros(N)
        curveID = np.zeros(N, "intc")

        # Now post-process to get the lowest one
        for i in range(N):
            d0 = np.linalg.norm(D[i, 0])
            s[i] = S[i, 0]
            curveID[i] = curves[0]
            for j in range(len(curves)):
                if np.linalg.norm(D[i, j]) < d0:
                    d0 = np.linalg.norm(D[i, j])
                    s[i] = S[i, j]
                    curveID[i] = curves[j]

        return curveID, s

    def projectPoints(self, points, *args, curves=None, **kwargs):
        """Project one or more points onto the nearest curve. This
        algorihm isn't exactly efficient: We simply project the nodes
        on each of the curves and take the lowest one.

        Parameters
        ----------
        points : array
            A single point (array length 3) or a set of points (N,3) array
        curves : list
            An optional list of curve indices to you. If not given, all
            curve objects are used.
        kwargs : dict
            Keyword arguments passed to curve.projectPoint() function

        Returns
        -------
        curveID : int
            The index of the curve with the closest distance
        s : float or array
            The curve parameter on self.curves[curveID] that is cloested
            to the point(s).
        """

        if curves is None:
            curves = np.arange(self.nCurve)

        N = len(points)
        S = np.zeros((N, len(curves)))
        D = np.zeros((N, len(curves), 3))
        for i in range(len(curves)):
            icurve = curves[i]
            S[:, i], D[:, i, :] = self.curves[icurve].projectPoint(points, *args, **kwargs)

        s = np.zeros(N)
        curveID = np.zeros(N, "intc")

        # Now post-process to get the lowest one
        for i in range(N):
            d0 = np.linalg.norm(D[i, 0])
            s[i] = S[i, 0]
            curveID[i] = curves[0]
            for j in range(len(curves)):
                if np.linalg.norm(D[i, j]) < d0:
                    d0 = np.linalg.norm(D[i, j])
                    s[i] = S[i, j]
                    curveID[i] = curves[j]

        return curveID, s
