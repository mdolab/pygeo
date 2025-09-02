# Standard Python modules
import os

# External modules
import numpy as np
from pyspline import Surface
from pyspline.utils import closeTecplot, line, openTecplot, writeTecplot1D

# Local modules
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

    def writeTecplot(
        self, fileName, orig=False, curves=True, coef=True, current=False, curveLabels=False, nodeLabels=False
    ):
        """Write the pyNetwork Object to Tecplot .dat file

        Parameters
        ----------
        fileName : str
            File name for tecplot file. Should have .dat extension
        orig : bool
            Flag to determine if we will write the original X data used to
            create this object
        curves : bool
            Flag to write discrete approximation of the actual curve
        coef : bool
            Flag to write b-spline coefficients
        current : bool
            Flag to determine if the current line is evaluated and added
            to the file. This is useful for higher order curves (k>2) where
            the coef array does not directly represent the curve.
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
                writeTecplot1D(f, "orig_data", self.curves[icurve].X)
        if current:
            # evaluate the curve with the current coefs and write
            for icurve in range(self.nCurve):
                current_line = self.curves[icurve](np.linspace(0, 1, 201))
                writeTecplot1D(f, "current_interp", current_line)

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
        """Update the coefficents on the pyNetwork update"""
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
            The curve parameter on self.curves[curveID] that is closest
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
            The curve parameter on self.curves[curveID] that is closest
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

    def intersectPlanes(self, points, axis, curves=None, raySize=1.5, **kwargs):
        """Find the intersection of the curves with the plane defined by the points and
        the normal vector. The ray size is used to define the extent of the plane
        about the points. The closest intersection to the original point is taken.
        The plane normal is determined by the "axis" parameter

        Parameters
        ----------
        points : array
            A single point (array length 3) or a set of points (N,3) array
            that lies on the plane. If multiple points are provided, one plane
            is defined with each point.
        axis : array of size 3
            Normal of the plane.
        curves : list
            An optional list of curve indices to use. If not given, all
            curve objects are used.
        raySize : float
            To define the plane, we use the point coordinates and the normal direction.
            The plane is extended by raySize in all directions.
        kwargs : dict
            Keyword arguments passed to Surface.projectCurve() function

        Returns
        -------
        curveID : int
            The index of the curve with the closest distance
        s : float or array
            The curve parameter on self.curves[curveID] that is closest
            to the point(s).
        """

        # given the normal vector in the axis parameter, we need to find two directions
        # that lie on the plane.

        # normalize axis
        axis /= np.linalg.norm(axis)

        # we now need to pick one direction that is not aligned with axis.
        # To do this, pick the smallest absolute component of the axis parameter.
        # we start with a unit vector in this direction, which is almost guaranteed
        # to be not perfectly aligned with the axis vector.
        dir1_ind = np.argmin(np.abs(axis))
        dir1 = np.zeros(3)
        dir1[dir1_ind] = 1.0

        # then we find the orthogonal component of dir1 to axis. this is the final dir1
        dir1 -= axis * axis.dot(dir1)
        dir1 /= np.linalg.norm(dir1)

        # get the third vector with a cross product
        dir2 = np.cross(axis, dir1)
        dir2 /= np.linalg.norm(dir2)

        # finally, we want to scale dir1 and dir2 by ray size. This controls
        # the size of the plane we create. Needs to be big enough to intersect
        # the curve.
        dir1 *= raySize
        dir2 *= raySize

        if curves is None:
            curves = np.arange(self.nCurve)

        N = len(points)
        S = np.zeros((N, len(curves)))
        D = np.zeros((N, len(curves), 3))

        for i in range(len(curves)):
            icurve = curves[i]
            for j in range(N):
                # we need to initialize a pySurface object for this point
                # the point is perturbed in dir 1 and dir2 to get 4 corners of the plane
                point = points[j]

                coef = np.zeros((2, 2, 3))
                # indexing:
                # 3 ------ 2
                # |        |
                # |   pt   |
                # |        |
                # 0 ------ 1
                # ^ dir2
                # |
                # |
                #  ---> dir 1
                coef[0, 0] = point - dir1 - dir2
                coef[1, 0] = point + dir1 - dir2
                coef[1, 1] = point + dir1 + dir2
                coef[0, 1] = point - dir1 + dir2

                ku = 2
                kv = 2
                tu = np.array([0.0, 0.0, 1.0, 1.0])
                tv = np.array([0.0, 0.0, 1.0, 1.0])

                surf = Surface(ku=ku, kv=kv, tu=tu, tv=tv, coef=coef)

                # now we project the current curve to this plane
                u, v, S[j, i], D[j, i, :] = surf.projectCurve(self.curves[icurve], **kwargs)

                if u == 0.0 or u == 1.0 or v == 0.0 or v == 1.0:
                    print(
                        "Warning: The link for attached point {:d} was drawn "
                        "from the curve to the end of the plane, "
                        "indicating that the plane might not have been large "
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
