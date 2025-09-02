# Standard Python modules
import copy
import os

# External modules
from baseclasses.utils import Error
import numpy as np
from pyspline import Curve, Surface
from pyspline.utils import closeTecplot, openTecplot, writeTecplot2D
from scipy import sparse
from scipy.sparse.linalg import factorized

# Local modules
from . import geo_utils
from .topology import SurfaceTopology


class pyGeo:
    """
    pyGeo is a (fairly) complete geometry surfacing engine. It
    performs multiple functions including producing surfaces from
    cross sections and globally fitting surfaces. The actual b-spline
    surfaces are of the pySpline Surface type.

    The initialization type, initType, specifies what type of
    initialization will be used. There are currently 3 initialization
    types: plot3d, iges, and liftingSurface

    Parameters
    ----------
    initType : str, {'plot3d', 'iges', 'liftingSurface'}
        A key word defining how this geo object will be defined.
    fileName : str
        Used for 'plot3d' and 'iges' only. Name of file to load.
    xsections : list of filenames
        List of the cross section coordinate files. Length = N
    scale : list or array
        List of the scaling factors (chord) for cross sections.
        Length = N
    offset : List or array
        List of x-y offset to apply *before* scaling. Length = N
    Xsec : List or array
        List of spatial coordinates as to the placement of
        cross sections. Size (N, 3)
    x, y, z : list or array
        If Xsec is not given, x,y,z arrays each of length N can
        be given individually.
    rot : List or array
        List of x-y-z rotations to apply to cross sections. Length = N
    rotX, rotY, rotZ : list or arrays
        Individual lists of x,y, and z rotations. Each of length N
    nCtl : int
        Number of control points to use for fitting. If it is None, local
        interpolation is performed which typically yields better results when
        a small number of sections. When trying to match wing geometries, a
        larger number of sections are often required.  In this case, the
        number of control points also needs to be increased to prevent the
        file from becoming overly large.
    kSpan : int
        The spline order in span-wise direction. 2 for linear, 3 for quadratic
        and 4 for cubic
    ku : int
        Spline order in u (for plot3d input files only)
    kv : int
        Spline order in v (for plot3d input files only)
    nCtlu : int
        Number of control points in u (for plot3d input files only)
    nCtlv : int
        Number of control points in v (for plot3d input files only)
    """

    def __init__(self, initType, *args, **kwargs):
        self.initType = initType
        print("pyGeo Initialization Type is: %s" % (initType))

        # ------------------- pyGeo Class Attributes -----------------
        self.topo = None  # The topology of the surfaces
        self.surfs = []  # The list of surface (pySpline surf)
        # objects
        self.nSurf = None  # The total number of surfaces
        self.coef = None  # The global (reduced) set of control
        # points

        if initType == "plot3d":
            self._readPlot3D(*args, **kwargs)
        elif initType == "iges":
            self._readIges(*args, **kwargs)
        elif initType == "liftingSurface":
            self._init_lifting_surface(*args, **kwargs)
        elif initType == "create":  # Don't do anything
            pass
        else:
            raise Error("Unknown init type. Valid Init types are 'plot3d', 'iges' and 'liftingSurface'")

    # ----------------------------------------------------------------------------
    #               Initialization Type Functions
    # ----------------------------------------------------------------------------

    def _readPlot3D(self, fileName, order="f", ku=4, kv=4, nCtlu=4, nCtlv=4):
        """Load a plot3D file and create the splines to go with each patch

        Parameters
        ----------
        fileName : str
            File name to load. Should end in .xyz
        order : str
            'f' for fortran ordering (usual), 'c' for c ordering
        ku : int
            Spline order in u
        kv : int
            Spline order in v
        nCtlu : int
            Number of control points in u
        nCtlv : int
            Number of control points in v
        """
        f = open(fileName)
        binary = False
        nSurf = geo_utils.readNValues(f, 1, "int", binary)[0]
        sizes = geo_utils.readNValues(f, nSurf * 3, "int", binary).reshape((nSurf, 3))

        # ONE of Patch Sizes index must be one
        nPts = 0
        for i in range(nSurf):
            if sizes[i, 0] == 1:  # Compress back to indices 0 and 1
                sizes[i, 0] = sizes[i, 1]
                sizes[i, 1] = sizes[i, 2]
            elif sizes[i, 1] == 1:
                sizes[i, 1] = sizes[i, 2]
            elif sizes[i, 2] == 1:
                pass
            else:
                raise Error("One of the plot3d indices must be 1")

            nPts += sizes[i, 0] * sizes[i, 1]

        surfs = []
        for i in range(nSurf):
            curSize = sizes[i, 0] * sizes[i, 1]
            surfs.append(np.zeros([sizes[i, 0], sizes[i, 1], 3]))
            for idim in range(3):
                surfs[-1][:, :, idim] = geo_utils.readNValues(f, curSize, "float", binary).reshape(
                    (sizes[i, 0], sizes[i, 1]), order=order
                )
        f.close()

        # Now create a list of spline surface objects:
        self.surfs = []
        self.surfs0 = surfs
        # Note This doesn't actually fit the surfaces...just produces
        # the parametrization and knot vectors
        self.nSurf = nSurf
        for isurf in range(self.nSurf):
            self.surfs.append(Surface(X=surfs[isurf], ku=ku, kv=kv, nCtlu=nCtlu, nCtlv=nCtlv))

    def _readIges(self, fileName):
        """Load a Iges file and create the splines to go with each patch

        Parameters
        ----------
        fileName : str
            Name of file to load.
        """
        f = open(fileName)
        Ifile = []
        for line in f:
            line = line.replace(";", ",")  # This is a bit of a hack...
            Ifile.append(line)
        f.close()

        start_lines = int(Ifile[-1][1:8])
        general_lines = int(Ifile[-1][9:16])
        directory_lines = int(Ifile[-1][17:24])
        # parameter_lines = int((Ifile[-1][25:32]))

        # Now we know how many lines we have to deal with
        dir_offset = start_lines + general_lines
        para_offset = dir_offset + directory_lines

        surf_list = []
        # Directory lines is ALWAYS a multiple of 2
        for i in range(directory_lines // 2):
            # 128 is bspline surface type
            if int(Ifile[2 * i + dir_offset][0:8]) == 128:
                start = int(Ifile[2 * i + dir_offset][8:16])
                num_lines = int(Ifile[2 * i + 1 + dir_offset][24:32])
                surf_list.append([start, num_lines])

        self.nSurf = len(surf_list)

        print("Found %d surfaces in Iges File." % (self.nSurf))

        self.surfs = []

        for isurf in range(self.nSurf):  # Loop over our patches
            data = []
            # Create a list of all data
            # -1 is for conversion from 1 based (iges) to python
            para_offset = surf_list[isurf][0] + dir_offset + directory_lines - 1

            for i in range(surf_list[isurf][1]):
                aux = Ifile[i + para_offset][0:69].split(",")
                for j in range(len(aux) - 1):
                    data.append(float(aux[j]))

            # Now we extract what we need
            Nctlu = int(data[1] + 1)
            Nctlv = int(data[2] + 1)
            ku = int(data[3] + 1)
            kv = int(data[4] + 1)

            counter = 10
            tu = data[counter : counter + Nctlu + ku]
            counter += Nctlu + ku

            tv = data[counter : counter + Nctlv + kv]
            counter += Nctlv + kv

            weights = data[counter : counter + Nctlu * Nctlv]
            weights = np.array(weights)
            if weights.all() != 1:
                print("WARNING: Not all weight in B-spline surface are 1. A NURBS surface CANNOT be replicated exactly")
            counter += Nctlu * Nctlv

            coef = np.zeros([Nctlu, Nctlv, 3])
            for j in range(Nctlv):
                for i in range(Nctlu):
                    coef[i, j, :] = data[counter : counter + 3]
                    counter += 3

            # Last we need the ranges
            prange = np.zeros(4)

            prange[0] = data[counter]
            prange[1] = data[counter + 1]
            prange[2] = data[counter + 2]
            prange[3] = data[counter + 3]

            # Re-scale the knot vectors in case the upper bound is not 1
            tu = np.array(tu)
            tv = np.array(tv)
            if not tu[-1] == 1.0:
                tu /= tu[-1]

            if not tv[-1] == 1.0:
                tv /= tv[-1]

            self.surfs.append(Surface(ku=ku, kv=kv, tu=tu, tv=tv, coef=coef))

            # Generate dummy data for connectivity to work
            u = np.linspace(0, 1, 3)
            v = np.linspace(0, 1, 3)
            [V, U] = np.meshgrid(v, u)
            self.surfs[-1].X = self.surfs[-1](U, V)
            self.surfs[-1].Nu = 3
            self.surfs[-1].Nv = 3
            self.surfs[-1].origData = True

    def _init_lifting_surface(
        self,
        xsections,
        X=None,
        x=None,
        y=None,
        z=None,
        rot=None,
        rotX=None,
        rotY=None,
        rotZ=None,
        scale=None,
        offset=None,
        nCtl=None,
        kSpan=3,
        teHeight=None,
        teHeightScaled=None,
        thickness=None,
        bluntTe=False,
        roundedTe=False,
        bluntTaperRange=0.1,
        squareTeTip=True,
        teScale=0.75,
        tip="rounded",
        tipScale=0.25,
        leOffset=0.001,
        teOffset=0.001,
        spanTang=0.5,
        upTang=0.5,
    ):
        """Create a lifting surface by distributing the cross
        sections. See pyGeo module documentation for information on
        the most commonly used options.
        """

        if X is not None:
            Xsec = np.array(X)
        else:
            # We have to use x, y, z
            Xsec = np.vstack([x, y, z]).T

        N = len(Xsec)

        if rot is not None:
            rot = np.array(rot)
        else:
            if rotX is None:
                rotX = np.zeros(N)
            if rotY is None:
                rotY = np.zeros(N)
            if rotZ is None:
                rotZ = np.zeros(N)
            rot = np.vstack([rotX, rotY, rotZ]).T

        if offset is None:
            offset = np.zeros((N, 2))

        if scale is None:
            scale = np.ones(N)

        # Limit kSpan to 2 if we only have two cross section
        if len(Xsec) == 2:
            kSpan = 2

        if bluntTe:
            if teHeight is None and teHeightScaled is None:
                raise Error("teHeight OR teHeightScaled must be supplied for bluntTe option")

            if teHeight:
                teHeight = np.atleast_1d(teHeight)
                if len(teHeight) == 1:
                    teHeight = np.ones(N) * teHeight
                teHeight /= scale

            if teHeightScaled:
                teHeight = np.atleast_1d(teHeightScaled)
                if len(teHeight) == 1:
                    teHeight = np.ones(N) * teHeight
        else:
            teHeight = [None for i in range(N)]

        # Load in and fit them all
        curves = []
        knots = []
        for i in range(len(xsections)):
            if xsections[i] is not None:
                x, y = geo_utils.readAirfoilFile(
                    xsections[i], bluntTe, bluntThickness=teHeight[i], bluntTaperRange=bluntTaperRange
                )
                weights = np.ones(len(x))
                weights[0] = -1
                weights[-1] = -1
                if nCtl is not None:
                    c = Curve(x=x, y=y, nCtl=nCtl, k=4, weights=weights)
                else:
                    c = Curve(x=x, y=y, localInterp=True)

                curves.append(c)
                knots.append(c.t)
            else:
                curves.append(None)

        # If we are fitting curves, blend knot vectors and recompute
        if nCtl is not None:
            newKnots = geo_utils.blendKnotVectors(knots, True)
            for i in range(len(xsections)):
                if curves[i] is not None:
                    curves[i].t = newKnots.copy()
                    curves[i].recompute(100, computeKnots=False)

            # If we want a pinched tip will will zero everything here.
            if tip == "pinched":
                # Just zero out the last section in y
                if curves[-1] is not None:
                    print("zeroing tip")
                    curves[-1].coef[:, 1] = 0

        else:
            # Otherwise do knot inserions
            origKnots = [None for i in range(N)]
            for i in range(N):
                if curves[i] is not None:
                    origKnots[i] = curves[i].t.copy()

            # First take all the knots of the first curve:
            baseKnots = []
            baseKnots.extend(origKnots[0])

            # For the rest of the curves
            for i in range(1, N):
                if curves[i] is not None:
                    knots = origKnots[i]
                    # Search for all indices
                    indices = np.searchsorted(baseKnots, knots, side="left")

                    toInsert = []
                    # Now go over the indices and see if we need to add
                    for j in range(len(indices)):
                        if abs(baseKnots[indices[j]] - knots[j]) > 1e-12:
                            toInsert.append(knots[j])

                    # Finally add the new indices and resort
                    baseKnots.extend(toInsert)
                    baseKnots.sort()

            # We have to know determine more information about the set
            # of baseKnots: We want just a list of just the knot
            # values and their multiplicity.
            newKnots = []
            mult = []
            i = 0
            Nmax = len(baseKnots)
            while i < len(baseKnots):
                curKnot = baseKnots[i]
                j = 1
                while i + j < Nmax and abs(baseKnots[i + j] - curKnot) < 1e-12:
                    j += 1
                i += j
                newKnots.append(curKnot)
                mult.append(j)

            # Now we have a knot vector that *ALL* curve *MUST* have
            # to form our surface. So we loop back over the curves and
            # insert the knots as necessary.
            for i in range(N):
                if curves[i] is not None:
                    for j in range(len(newKnots)):
                        if newKnots[j] not in curves[i].t:
                            curves[i].insertKnot(newKnots[j], mult[j])

            # If we want a pinched tip will will zero everything here.
            if tip == "pinched":
                # Just zero out the last section in y
                if curves[-1] is not None:
                    curves[-1].coef[:, 1] = 0

            # Finally force ALL curve to have PRECISELY identical knots
            for i in range(len(xsections)):
                if curves[i] is not None:
                    curves[i].t = curves[0].t.copy()

            newKnots = curves[0].t.copy()
        # end if (nCtl is not none)

        # Generate a curve from X just for the parametrization
        Xcurve = Curve(X=Xsec, k=kSpan)

        # Now blend the missing sections
        print("Interpolating missing sections ...")

        for i in range(len(xsections)):
            if xsections[i] is None:
                # Fist two curves bounding this unknown one:
                for j in range(i, -1, -1):
                    if xsections[j] is not None:
                        istart = j
                        break

                for j in range(i, len(xsections), 1):
                    if xsections[j] is not None:
                        iend = j
                        break

                # Now generate blending parameter alpha
                sStart = Xcurve.s[istart]
                sEnd = Xcurve.s[iend]
                s = Xcurve.s[i]

                alpha = (s - sStart) / (sEnd - sStart)

                coef = curves[istart].coef * (1 - alpha) + curves[iend].coef * (alpha)

                curves[i] = Curve(coef=coef, k=4, t=newKnots.copy())
        # end for (xsections)

        # Before we continue the user may want to artificially scale
        # the thickness of the sections. This is useful when a
        # different airfoil thickness is desired than the actual
        # airfoil coordinates.

        if thickness is not None:
            thickness = np.atleast_1d(thickness)
            if len(thickness) == 1:
                thickness = np.ones(len(thickness)) * thickness
            for i in range(N):
                # Only scale the interior control points; not the first and last
                curves[i].coef[1:-1, 1] *= thickness[i]

        # Now split each curve at uSplit which roughly corresponds to LE
        topCurves = []
        botCurves = []
        uSplit = curves[0].t[(curves[0].nCtl + 4 - 1) // 2]

        for i in range(len(xsections)):
            c1, c2 = curves[i].splitCurve(uSplit)
            topCurves.append(c1)
            c2.reverse()
            botCurves.append(c2)

        # Note that the number of control points on the upper and
        # lower surface MAY not be the same. We can fix this by doing
        # more knot insertions.
        knotsTop = topCurves[0].t.copy()
        knotsBot = botCurves[0].t.copy()

        print("Symmetrizing Knot Vectors ...")
        eps = 1e-12
        for i in range(len(knotsTop)):
            # Check if knotsTop[i] is not in knots_bot to within eps
            found = False
            for j in range(len(knotsBot)):
                if abs(knotsTop[i] - knotsBot[j]) < eps:
                    found = True

            if not found:
                # Add to all sections
                for ii in range(len(xsections)):
                    botCurves[ii].insertKnot(knotsTop[i], 1)

        for i in range(len(knotsBot)):
            # Check if knotsBot[i] is not in knotsTop to within eps
            found = False
            for j in range(len(knotsTop)):
                if abs(knotsBot[i] - knotsTop[j]) < eps:
                    found = True

            if not found:
                # Add to all sections
                for ii in range(len(xsections)):
                    topCurves[ii].insertKnot(knotsBot[i], 1)

        # We now have symmetrized knot vectors for the upper and lower
        # surfaces. We will copy the vectors to make sure they are
        # precisely the same:
        for i in range(len(xsections)):
            topCurves[i].t = topCurves[0].t.copy()
            botCurves[i].t = topCurves[0].t.copy()

        # Now we can set the surfaces
        ncoef = topCurves[0].nCtl
        coefTop = np.zeros((ncoef, len(xsections), 3))
        coefBot = np.zeros((ncoef, len(xsections), 3))

        for i in range(len(xsections)):
            # Scale, rotate and translate the coefficients
            coefTop[:, i, 0] = scale[i] * (topCurves[i].coef[:, 0] - offset[i, 0])
            coefTop[:, i, 1] = scale[i] * (topCurves[i].coef[:, 1] - offset[i, 1])
            coefTop[:, i, 2] = 0

            coefBot[:, i, 0] = scale[i] * (botCurves[i].coef[:, 0] - offset[i, 0])
            coefBot[:, i, 1] = scale[i] * (botCurves[i].coef[:, 1] - offset[i, 1])
            coefBot[:, i, 2] = 0

            for j in range(ncoef):
                coefTop[j, i, :] = geo_utils.rotzV(coefTop[j, i, :], rot[i, 2] * np.pi / 180)
                coefTop[j, i, :] = geo_utils.rotxV(coefTop[j, i, :], rot[i, 0] * np.pi / 180)
                coefTop[j, i, :] = geo_utils.rotyV(coefTop[j, i, :], rot[i, 1] * np.pi / 180)

                coefBot[j, i, :] = geo_utils.rotzV(coefBot[j, i, :], rot[i, 2] * np.pi / 180)
                coefBot[j, i, :] = geo_utils.rotxV(coefBot[j, i, :], rot[i, 0] * np.pi / 180)
                coefBot[j, i, :] = geo_utils.rotyV(coefBot[j, i, :], rot[i, 1] * np.pi / 180)

            # Finally translate according to  positions specified
            coefTop[:, i, :] += Xsec[i, :]
            coefBot[:, i, :] += Xsec[i, :]

        # Set the two main surfaces
        self.surfs.append(Surface(coef=coefTop, ku=4, kv=kSpan, tu=topCurves[0].t, tv=Xcurve.t))
        self.surfs.append(Surface(coef=coefBot, ku=4, kv=kSpan, tu=botCurves[0].t, tv=Xcurve.t))

        print("Computing TE surfaces ...")

        if bluntTe:
            if not roundedTe:
                coef = np.zeros((len(xsections), 2, 3), "d")
                coef[:, 0, :] = coefTop[0, :, :]
                coef[:, 1, :] = coefBot[0, :, :]
                self.surfs.append(Surface(coef=coef, ku=kSpan, kv=2, tu=Xcurve.t, tv=[0, 0, 1, 1]))
            else:
                coef = np.zeros((len(xsections), 4, 3), "d")
                coef[:, 0, :] = coefTop[0, :, :]
                coef[:, 3, :] = coefBot[0, :, :]

                # We need to get the tangent for the top and bottom
                # surface, multiply by a scaling factor and this gives
                # us the two inner rows of control points

                for j in range(len(xsections)):
                    projTop = coefTop[0, j] - coefTop[1, j]
                    projBot = coefBot[0, j] - coefBot[1, j]
                    projTop /= np.linalg.norm(projTop)
                    projBot /= np.linalg.norm(projBot)
                    curTeThick = np.linalg.norm(coefTop[0, j] - coefBot[0, j])
                    coef[j, 1] = coef[j, 0] + projTop * 0.5 * curTeThick * teScale
                    coef[j, 2] = coef[j, 3] + projBot * 0.5 * curTeThick * teScale

                self.surfs.append(Surface(coef=coef, ku=kSpan, kv=4, tu=Xcurve.t, tv=[0, 0, 0, 0, 1, 1, 1, 1]))

        self.nSurf = len(self.surfs)

        print("Computing Tip surfaces ...")
        # # Add on additional surfaces if required for a rounded pinch tip
        if tip == "rounded":
            # Generate the midpoint of the coefficients
            midPts = np.zeros([ncoef, 3])
            upVec = np.zeros([ncoef, 3])
            dsNorm = np.zeros([ncoef, 3])
            for j in range(ncoef):
                midPts[j] = 0.5 * (coefTop[j, -1] + coefBot[j, -1])
                upVec[j] = coefTop[j, -1] - coefBot[j, -1]
                ds = 0.5 * ((coefTop[j, -1] - coefTop[j, -2]) + (coefBot[j, -1] - coefBot[j, -2]))
                dsNorm[j] = ds / np.linalg.norm(ds)

            # Generate "average" projection Vector
            projVec = np.zeros((ncoef, 3), "d")
            for j in range(ncoef):
                offset = teOffset + (float(j) / (ncoef - 1)) * (leOffset - teOffset)
                projVec[j] = dsNorm[j] * (np.linalg.norm(upVec[j] * tipScale + offset))

            # Generate the tip "line"
            tipLine = np.zeros([ncoef, 3])
            for j in range(ncoef):
                tipLine[j] = midPts[j] + projVec[j]

            # Generate a k=4 (cubic) surface
            coefTopTip = np.zeros([ncoef, 4, 3])
            coefBotTip = np.zeros([ncoef, 4, 3])

            for j in range(ncoef):
                coefTopTip[j, 0] = coefTop[j, -1]
                coefTopTip[j, 1] = coefTop[j, -1] + projVec[j] * spanTang
                coefTopTip[j, 2] = tipLine[j] + upTang * upVec[j]
                coefTopTip[j, 3] = tipLine[j]

                coefBotTip[j, 0] = coefBot[j, -1]
                coefBotTip[j, 1] = coefBot[j, -1] + projVec[j] * spanTang
                coefBotTip[j, 2] = tipLine[j] - upTang * upVec[j]
                coefBotTip[j, 3] = tipLine[j]

            # Modify for square_te_tip... taper over last 20%
            if squareTeTip and not roundedTe:
                tipDist = geo_utils.eDist(tipLine[0], tipLine[-1])

                for j in range(ncoef):  # Going from back to front:
                    fraction = geo_utils.eDist(tipLine[j], tipLine[0]) / tipDist
                    if fraction < 0.10:
                        fact = (1 - fraction / 0.10) ** 2
                        omfact = 1.0 - fact
                        coefTopTip[j, 1] = (
                            fact * ((5.0 / 6.0) * coefTopTip[j, 0] + (1.0 / 6.0) * coefBotTip[j, 0])
                            + omfact * coefTopTip[j, 1]
                        )
                        coefTopTip[j, 2] = (
                            fact * ((4.0 / 6.0) * coefTopTip[j, 0] + (2.0 / 6.0) * coefBotTip[j, 0])
                            + omfact * coefTopTip[j, 2]
                        )
                        coefTopTip[j, 3] = (
                            fact * ((1.0 / 2.0) * coefTopTip[j, 0] + (1.0 / 2.0) * coefBotTip[j, 0])
                            + omfact * coefTopTip[j, 3]
                        )

                        coefBotTip[j, 1] = (
                            fact * ((1.0 / 6.0) * coefTopTip[j, 0] + (5.0 / 6.0) * coefBotTip[j, 0])
                            + omfact * coefBotTip[j, 1]
                        )
                        coefBotTip[j, 2] = (
                            fact * ((2.0 / 6.0) * coefTopTip[j, 0] + (4.0 / 6.0) * coefBotTip[j, 0])
                            + omfact * coefBotTip[j, 2]
                        )
                        coefBotTip[j, 3] = (
                            fact * ((1.0 / 2.0) * coefTopTip[j, 0] + (1.0 / 2.0) * coefBotTip[j, 0])
                            + omfact * coefBotTip[j, 3]
                        )

            surfTopTip = Surface(coef=coefTopTip, ku=4, kv=4, tu=topCurves[0].t, tv=[0, 0, 0, 0, 1, 1, 1, 1])
            surfBotTip = Surface(coef=coefBotTip, ku=4, kv=4, tu=botCurves[0].t, tv=[0, 0, 0, 0, 1, 1, 1, 1])
            self.surfs.append(surfTopTip)
            self.surfs.append(surfBotTip)
            self.nSurf += 2

            if bluntTe:
                # This is the small surface at the trailing edge
                # tip. There are a couple of different things that can
                # happen: If we rounded TE we MUST have a
                # rounded-spherical-like surface (second piece of code
                # below). Otherwise we only have a surface if
                # square_te_tip is false in which case a flat curved
                # surface results.

                if not roundedTe and not squareTeTip:
                    coef = np.zeros((4, 2, 3), "d")
                    coef[:, 0] = coefTopTip[0, :]
                    coef[:, 1] = coefBotTip[0, :]

                    self.surfs.append(Surface(coef=coef, ku=4, kv=2, tu=[0, 0, 0, 0, 1, 1, 1, 1], tv=[0, 0, 1, 1]))
                    self.nSurf += 1
                elif roundedTe:
                    coef = np.zeros((4, 4, 3), "d")
                    coef[:, 0] = coefTopTip[0, :]
                    coef[:, 3] = coefBotTip[0, :]

                    # We will actually recompute the coefficients
                    # on the last sections since we need to do a
                    # couple of more for this surface
                    for i in range(4):
                        projTop = coefTopTip[0, i] - coefTopTip[1, i]
                        projBot = coefBotTip[0, i] - coefBotTip[1, i]
                        projTop /= np.linalg.norm(projTop)
                        projBot /= np.linalg.norm(projBot)
                        curTeThick = np.linalg.norm(coefTopTip[0, i] - coefBotTip[0, i])
                        coef[i, 1] = coef[i, 0] + projTop * 0.5 * curTeThick * teScale
                        coef[i, 2] = coef[i, 3] + projBot * 0.5 * curTeThick * teScale

                    self.surfs.append(
                        Surface(coef=coef, ku=4, kv=4, tu=[0, 0, 0, 0, 1, 1, 1, 1], tv=[0, 0, 0, 0, 1, 1, 1, 1])
                    )
                    self.nSurf += 1

            # end if bluntTe
        elif tip == "pinched":
            pass
        else:
            print("No tip specified")

        # Cheat and make "original data" so that the edge connectivity works
        u = np.linspace(0, 1, 3)
        v = np.linspace(0, 1, 3)
        [V, U] = np.meshgrid(u, v)
        for i in range(self.nSurf):
            self.surfs[i].origData = True
            self.surfs[i].X = self.surfs[i](U, V)
            self.surfs[i].Nu = 3
            self.surfs[i].Nv = 3

        self._calcConnectivity(1e-6, 1e-6)
        sizes = []
        for isurf in range(self.nSurf):
            sizes.append([self.surfs[isurf].nCtlu, self.surfs[isurf].nCtlv])
        self.topo.calcGlobalNumbering(sizes)

        self.setSurfaceCoef()

    def fitGlobal(self):
        """
        Perform a global B-spline surface fit to determine the
        coefficients of each patch. This is only used with an plot3D
        init type
        """

        print("Global Fitting")
        nCtl = self.topo.nGlobal
        print(" -> Copying Topology")
        origTopo = copy.deepcopy(self.topo)

        print(" -> Creating global numbering")
        sizes = []
        for isurf in range(self.nSurf):
            sizes.append([self.surfs[isurf].Nu, self.surfs[isurf].Nv])

        # Get the Global number of the original data
        origTopo.calcGlobalNumbering(sizes)
        N = origTopo.nGlobal
        print(" -> Creating global point list")
        pts = np.zeros((N, 3))
        for ii in range(N):
            pts[ii] = self.surfs[origTopo.gIndex[ii][0][0]].X[origTopo.gIndex[ii][0][1], origTopo.gIndex[ii][0][2]]

        # Get the maximum k (ku, kv for each surf)
        kmax = 2
        for isurf in range(self.nSurf):
            if self.surfs[isurf].ku > kmax:
                kmax = self.surfs[isurf].ku
            if self.surfs[isurf].kv > kmax:
                kmax = self.surfs[isurf].kv

        nnz = N * kmax * kmax
        vals = np.zeros(nnz)
        rowPtr = [0]
        colInd = np.zeros(nnz, "intc")

        for ii in range(N):
            isurf = origTopo.gIndex[ii][0][0]
            i = origTopo.gIndex[ii][0][1]
            j = origTopo.gIndex[ii][0][2]

            u = self.surfs[isurf].U[i, j]
            v = self.surfs[isurf].V[i, j]

            vals, colInd = self.surfs[isurf].getBasisPt(u, v, vals, rowPtr[ii], colInd, self.topo.lIndex[isurf])

            kinc = self.surfs[isurf].ku * self.surfs[isurf].kv
            rowPtr.append(rowPtr[-1] + kinc)

        # Now we can crop out any additional values in col_ptr and vals
        vals = vals[: rowPtr[-1]]
        colInd = colInd[: rowPtr[-1]]
        # Now make a sparse matrix

        NN = sparse.csr_matrix((vals, colInd, rowPtr))
        print(" -> Multiplying N^T * N")
        NNT = NN.T
        NTN = NNT * NN
        print(" -> Factorizing...")
        solve = factorized(NTN)
        print(" -> Back Solving...")
        self.coef = np.zeros((nCtl, 3))
        for idim in range(3):
            self.coef[:, idim] = solve(NNT * pts[:, idim])

        print(" -> Setting Surface Coefficients...")
        self._updateSurfaceCoef()

    # ----------------------------------------------------------------------
    #                     Topology Information Functions
    # ----------------------------------------------------------------------

    def doConnectivity(self, fileName=None, nodeTol=1e-4, edgeTol=1e-4):
        """
        This is the only public edge connectivity function.
        If fileName exists it loads the file OR it calculates the connectivity
        and saves to that file.

        Parameters
        ----------
        fileName : str
            Filename for con file
        nodeTol : float
            The tolerance for identical nodes
        edgeTol : float
            The tolerance for midpoint of edges being identical
        """
        if fileName is not None and os.path.isfile(fileName):
            print("Reading Connectivity File: %s" % (fileName))
            self.topo = SurfaceTopology(fileName=fileName)
            if self.initType != "iges":
                self._propagateKnotVectors()

            sizes = []
            for isurf in range(self.nSurf):
                sizes.append([self.surfs[isurf].nCtlu, self.surfs[isurf].nCtlv])
                self.surfs[isurf].recompute()
            self.topo.calcGlobalNumbering(sizes)
        else:
            self._calcConnectivity(nodeTol, edgeTol)
            sizes = []
            for isurf in range(self.nSurf):
                sizes.append([self.surfs[isurf].nCtlu, self.surfs[isurf].nCtlv])
            self.topo.calcGlobalNumbering(sizes)
            if self.initType != "iges":
                self._propagateKnotVectors()
            if fileName is not None:
                print("Writing Connectivity File: %s" % (fileName))
                self.topo.writeConnectivity(fileName)

        if self.initType == "iges":
            self.setSurfaceCoef()

    def _calcConnectivity(self, nodeTol, edgeTol):
        """This function attempts to automatically determine the connectivity
        between the patches
        """

        # Calculate the 4 corners and 4 midpoints for each surface

        coords = np.zeros((self.nSurf, 8, 3))

        for isurf in range(self.nSurf):
            beg, mid, end = self.surfs[isurf].getOrigValuesEdge(0)
            coords[isurf][0] = beg
            coords[isurf][1] = end
            coords[isurf][4] = mid
            beg, mid, end = self.surfs[isurf].getOrigValuesEdge(1)
            coords[isurf][2] = beg
            coords[isurf][3] = end
            coords[isurf][5] = mid
            beg, mid, end = self.surfs[isurf].getOrigValuesEdge(2)
            coords[isurf][6] = mid
            beg, mid, end = self.surfs[isurf].getOrigValuesEdge(3)
            coords[isurf][7] = mid

        self.topo = SurfaceTopology(coords=coords, nodeTol=nodeTol, edgeTol=edgeTol)

    def printConnectivity(self):
        """
        Print the Edge connectivity to the screen
        """
        self.topo.printConnectivity()

    def _propagateKnotVectors(self):
        """Propagate the knot vectors to make consistent"""
        # First get the number of design groups
        nDG = -1
        ncoef = []
        for i in range(self.topo.nEdge):
            if self.topo.edges[i].dg > nDG:
                nDG = self.topo.edges[i].dg
                ncoef.append(self.topo.edges[i].N)

        nDG += 1
        for isurf in range(self.nSurf):
            dgU = self.topo.edges[self.topo.edgeLink[isurf][0]].dg
            dgV = self.topo.edges[self.topo.edgeLink[isurf][2]].dg
            self.surfs[isurf].nCtlu = ncoef[dgU]
            self.surfs[isurf].nCtlv = ncoef[dgV]
            if self.surfs[isurf].ku < self.surfs[isurf].nCtlu:
                if self.surfs[isurf].nCtlu > 4:
                    self.surfs[isurf].ku = 4
                else:
                    self.surfs[isurf].ku = self.surfs[isurf].nCtlu

            if self.surfs[isurf].kv < self.surfs[isurf].nCtlv:
                if self.surfs[isurf].nCtlv > 4:
                    self.surfs[isurf].kv = 4
                else:
                    self.surfs[isurf].kv = self.surfs[isurf].nCtlv

            self.surfs[isurf].calcKnots()

        # Now loop over the number of design groups, accumulate all
        # the knot vectors that corresponds to this dg, then merge them all

        for idg in range(nDG):
            knotVectors = []
            flip = []
            for isurf in range(self.nSurf):
                for iedge in range(4):
                    if self.topo.edges[self.topo.edgeLink[isurf][iedge]].dg == idg:
                        if self.topo.edgeDir[isurf][iedge] == -1:
                            flip.append(True)
                        else:
                            flip.append(False)

                        if iedge in [0, 1]:
                            knotVec = self.surfs[isurf].tu
                        elif iedge in [2, 3]:
                            knotVec = self.surfs[isurf].tv

                        if flip[-1]:
                            knotVectors.append((1 - knotVec)[::-1].copy())
                        else:
                            knotVectors.append(knotVec)

            # end for (isurf)

            # Now blend all the knot vectors
            newKnotVec = geo_utils.blendKnotVectors(knotVectors, False)
            newKnotVecFlip = (1 - newKnotVec)[::-1]

            counter = 0
            for isurf in range(self.nSurf):
                for iedge in range(4):
                    if self.topo.edges[self.topo.edgeLink[isurf][iedge]].dg == idg:
                        if iedge in [0, 1]:
                            if flip[counter]:
                                self.surfs[isurf].tu = newKnotVecFlip.copy()
                            else:
                                self.surfs[isurf].tu = newKnotVec.copy()

                        elif iedge in [2, 3]:
                            if flip[counter]:
                                self.surfs[isurf].tv = newKnotVecFlip.copy()
                            else:
                                self.surfs[isurf].tv = newKnotVec.copy()
                        counter += 1

    # ----------------------------------------------------------------------
    #                   Surface Writing Output Functions
    # ----------------------------------------------------------------------

    def writeTecplot(
        self,
        fileName,
        orig=False,
        surfs=True,
        coef=True,
        directions=False,
        surfLabels=False,
        edgeLabels=False,
        nodeLabels=False,
    ):
        """Write the pyGeo Object to Tecplot dat file

        Parameters
        ----------
        fileName : str
            File name for tecplot file. Should have .dat extension
        surfs : bool
            Flag to write discrete approximation of the actual surface
        coef: bool
            Flag to write b-spline coefficients
        directions : bool
            Flag to write surface direction visualization
        surfLabels : bool
            Flag to write file with surface labels
        edgeLabels : bool
            Flag to write file with edge labels
        nodeLabels : bool
            Falg to write file with node labels
        """

        f = openTecplot(fileName, 3)

        # Write out the Interpolated Surfaces
        if surfs:
            for isurf in range(self.nSurf):
                self.surfs[isurf].computeData()
                writeTecplot2D(f, "interpolated", self.surfs[isurf].data)

        # Write out the Control Points
        if coef:
            for isurf in range(self.nSurf):
                writeTecplot2D(f, "control_pts", self.surfs[isurf].coef)

        # Write out the Original Data
        if orig:
            for isurf in range(self.nSurf):
                writeTecplot2D(f, "control_pts", self.surfs[isurf].X)

        # Write out The Surface Directions
        if directions:
            for isurf in range(self.nSurf):
                self.surfs[isurf].writeDirections(f, isurf)

        # Write out The Surface, Edge and Node Labels
        dirName, fileName = os.path.split(fileName)
        fileBaseName, _ = os.path.splitext(fileName)

        if surfLabels:
            # Split the filename off
            labelFilename = dirName + "./" + fileBaseName + ".surf_labels.dat"
            f2 = open(labelFilename, "w")
            for isurf in range(self.nSurf):
                midu = np.floor(self.surfs[isurf].nCtlu / 2)
                midv = np.floor(self.surfs[isurf].nCtlv / 2)
                textString = 'TEXT CS=GRID3D, X=%f, Y=%f, Z=%f, ZN=%d, T="S%d"\n' % (
                    self.surfs[isurf].coef[midu, midv, 0],
                    self.surfs[isurf].coef[midu, midv, 1],
                    self.surfs[isurf].coef[midu, midv, 2],
                    isurf + 1,
                    isurf,
                )
                f2.write("%s" % (textString))
            f2.close()

        if edgeLabels:
            # Split the filename off
            labelFilename = dirName + "./" + fileBaseName + "edge_labels.dat"
            f2 = open(labelFilename, "w")
            for iedge in range(self.topo.nEdge):
                surfaces = self.topo.getSurfaceFromEdge(iedge)
                pt = self.surfs[surfaces[0][0]].edgeCurves[surfaces[0][1]](0.5)
                textString = 'TEXT CS=GRID3D X=%f, Y=%f, Z=%f, T="E%d"\n' % (pt[0], pt[1], pt[2], iedge)
                f2.write("%s" % (textString))
            f2.close()

        if nodeLabels:
            # First we need to figure out where the corners actually *are*
            nNodes = len(geo_utils.unique(self.topo.nodeLink.flatten()))
            nodeCoord = np.zeros((nNodes, 3))
            for i in range(nNodes):
                # Try to find node i
                for isurf in range(self.nSurf):
                    if self.topo.nodeLink[isurf][0] == i:
                        coordinate = self.surfs[isurf].getValueCorner(0)
                        break
                    elif self.topo.nodeLink[isurf][1] == i:
                        coordinate = self.surfs[isurf].getValueCorner(1)
                        break
                    elif self.topo.nodeLink[isurf][2] == i:
                        coordinate = self.surfs[isurf].getValueCorner(2)
                        break
                    elif self.topo.nodeLink[isurf][3] == i:
                        coordinate = self.surfs[isurf].getValueCorner(3)
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

        # Close out the file
        closeTecplot(f)

    def writeIGES(self, fileName):
        """
        Write the surface to IGES format

        Parameters
        ----------
        fileName : str
            File name of iges file. Should have .igs extension.
        """
        f = open(fileName, "w")

        # Note: Eventually we may want to put the CORRECT Data here
        f.write("                                                                        S      1\n")
        f.write("1H,,1H;,7H128-000,11H128-000.IGS,9H{unknown},9H{unknown},16,6,15,13,15, G      1\n")
        f.write("7H128-000,1.,6,1HM,8,0.016,15H19970830.165254, 0.0001,0.,               G      2\n")
        f.write("21Hdennette@wiz-worx.com,23HLegacy PDD AP Committee,11,3,               G      3\n")
        f.write("13H920717.080000,23HMIL-PRF-28000B0,CLASS 1;                            G      4\n")

        Dcount = 1
        Pcount = 1

        for isurf in range(self.nSurf):
            Pcount, Dcount = self.surfs[isurf].writeIGES_directory(f, Dcount, Pcount)

        Pcount = 1
        counter = 1

        for isurf in range(self.nSurf):
            Pcount, counter = self.surfs[isurf].writeIGES_parameters(f, Pcount, counter)

        # Write the terminate statement
        f.write("S%7dG%7dD%7dP%7d%40sT%6s1\n" % (1, 4, Dcount - 1, counter - 1, " ", " "))
        f.close()

    def writeTin(self, fileName):
        """
        Write the surfaces to ICEMCFD .tin format

        Parameters
        ----------
        fileName : str
            File name of tin file. Should have .tin extension.
        """
        f = open(fileName, "w")
        # Standard Python modules
        import datetime

        # Write the required header info here:
        f.write("// tetin file version 1.1\n")
        f.write("// written by pyLayoutGeo - on %s\n" % (datetime.datetime.now()))
        f.write("set_triangulation_tolerance 0.001\n")

        # Now loop over the componets and each will write the info it
        # has to the .tin file:
        for i in range(self.nSurf):
            if self.surfs[i].name is None:
                name = "surface_%d" % i
            else:
                name = self.surfs[i].name
            s = "define_surface name surf.%d family %s tetra_size %f\n" % (i, name, 1.0)
            f.write(s)
            self.surfs[i].writeTin(f)

        # Write the closing info:
        f.write("affix 0\n")
        f.write("define_model 1e+10 reference_size 1\n")
        f.write("return\n")
        f.close()

    # ----------------------------------------------------------------------
    #                Update and Derivative Functions
    # ----------------------------------------------------------------------

    def _updateSurfaceCoef(self):
        """Copy the pyGeo list of control points back to the surfaces"""
        for ii in range(len(self.coef)):
            for jj in range(len(self.topo.gIndex[ii])):
                isurf = self.topo.gIndex[ii][jj][0]
                i = self.topo.gIndex[ii][jj][1]
                j = self.topo.gIndex[ii][jj][2]
                self.surfs[isurf].coef[i, j] = self.coef[ii].astype("d")

        for isurf in range(self.nSurf):
            self.surfs[isurf].setEdgeCurves()

    def setSurfaceCoef(self):
        """Set the surface coef list from the pyspline surfaces"""
        self.coef = np.zeros((self.topo.nGlobal, 3))
        for isurf in range(self.nSurf):
            surf = self.surfs[isurf]
            for i in range(surf.nCtlu):
                for j in range(surf.nCtlv):
                    self.coef[self.topo.lIndex[isurf][i, j]] = surf.coef[i, j]

    def getBounds(self, surfs=None):
        """Determine the extents of the collection of surfaces

        Parameters
        ----------
        surfs : list or array
            Indices of surfaces defining subset for which to get the bounding
            box

        Returns
        -------
        xMin : array of length 3
            Lower corner of the bounding box
        xMax : array of length 3
            Upper corner of the bounding box
        """
        if surfs is None:
            surfs = np.arange(self.nSurf)

        Xmin0, Xmax0 = self.surfs[surfs[0]].getBounds()
        for i in range(1, len(surfs)):
            isurf = surfs[i]
            Xmin, Xmax = self.surfs[isurf].getBounds()
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

    def projectCurve(self, curve, *args, surfs=None, **kwargs):
        """
        Project a pySpline curve onto the pyGeo object

        Parameters
        ----------
        curve : pySpline.Curve object
            Curve to use for the intersection
        surfs : list or array
            Indices of surface defining subset for which to get the bounding
            box

        Returns
        -------
        Try it and see!

        Notes
        -----
        This algorithm is not efficient at all.  We basically do the
        curve-surface projection algorithm for each surface the loop
        back over them to see which is the best in terms of closest
        distance. This intent is that the curve *actually* intersects
        one of the surfaces.
        """

        if surfs is None:
            surfs = np.arange(self.nSurf)

        temp = np.zeros((len(surfs), 4))
        result = np.zeros((len(surfs), 4))
        patchID = np.zeros(len(surfs), "intc")

        for i in range(len(surfs)):
            isurf = surfs[i]
            u, v, s, d = self.surfs[isurf].projectCurve(curve, *args, **kwargs)
            temp[i, :] = [u, v, s, np.linalg.norm(d)]

        # Sort the results by distance
        index = np.argsort(temp[:, 3])

        for i in range(len(surfs)):
            result[i] = temp[index[i]]
            patchID[i] = surfs[index[i]]

        return result, patchID

    def projectPoints(self, points, *args, surfs=None, **kwargs):
        """Project on or more points onto the nearest surface.

        Parameters
        ----------
        points : list or array
            Singe point (size 3) or list of points size (N,3) points
            to project onto the surfaces

        surfs : list or array
            Indices of surface defining subset for which to use for
            projection

        Returns
        -------
        u : float or array
            u parameter values of closest point
        v : float or array
            v parameter values of closest point
        PID : int or int array
            Patch index corresponding to the u,v parameter values
        """

        if surfs is None:
            surfs = np.arange(self.nSurf)

        N = len(points)
        U = np.zeros((N, len(surfs)))
        V = np.zeros((N, len(surfs)))
        D = np.zeros((N, len(surfs), 3))
        for i in range(len(surfs)):
            isurf = surfs[i]
            U[:, i], V[:, i], D[:, i, :] = self.surfs[isurf].projectPoint(points, *args, **kwargs)

        u = np.zeros(N)
        v = np.zeros(N)
        patchID = np.zeros(N, "intc")

        # Now post-process to get the lowest one
        for i in range(N):
            d0 = np.linalg.norm(D[i, 0])
            u[i] = U[i, 0]
            v[i] = V[i, 0]
            patchID[i] = surfs[0]
            for j in range(len(surfs)):
                if np.linalg.norm(D[i, j]) < d0:
                    d0 = np.linalg.norm(D[i, j])
                    u[i] = U[i, j]
                    v[i] = V[i, j]
                    patchID[i] = surfs[j]

        return u, v, patchID
