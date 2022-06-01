import numpy as np


def write_wing_FFD_file(fileName, slices, N0, N1, N2, axes=None, dist=None):
    """
    This function can be used to generate a simple FFD. The FFD can be made up
    of more than one volume, but the volumes will be connected. It is meant for
    doing simple wing FFDs.

    Parameters
    ----------
    fileName : str
        Name of output file. File is written in plot3d format.

    slices : numpy array of (Nvol+1, 2, 2, 3)
        Array of slices. Each slice should contain four points in 3D that will
        be the corners of the FFD on that slice. If the zeroth dimension size
        is greater than 2, then multiple volumes will be created, connected by
        the intermediate slice.

    N0 : integer or list
        Number of points to distribute along the zeroth dimension (along the
        slice direction).

    N1 : integer or list
        Number of points to distribute along the first dimension.

    N2 : integer or list
        Number of points to distribute along the second dimension.

    axes : list of ['i', 'j', 'k'] in arbitrary order
        The user can interchange which index of the FFD corresponds with each
        dimension of slices. By default 'k' -> 0, 'j' -> 1, 'i' -> 2.

    dist : list
        For each volume, the user can specify the distribution of points along
        each dimension. Options include:

            - linear
            - cosine
            - left (tighter spacing on the left side)
            - right (tighter spacing on the other left side)

    Examples
    --------
    This is an example of two volumes:

    .. code-block:: python

        axes = ['k', 'j', 'i']
        slices = np.array([
            # Slice 1
            [[[0, 0, 0], [1, 0, 0]],
            [[0, 0.2, 0], [1, 0.2, 0]]],
            # Slice 2
            [[[0, 0, 2], [1, 0, 2]],
            [[0, 0.2, 2], [1, 0.2, 2]]],
            # Slice 3
            [[[0.5, 0, 6], [1, 0, 6]],
            [[0.5, 0.2, 6], [1, 0.2, 6]]],
        ])

        N0 = 5
        N1 = 2
        N2 = 8

        dist = [
            ['left', 'linear', 'linear'],
            ['cosine', 'linear', 'right']
        ]

    """

    Nvol = slices.shape[0] - 1

    if axes is None:
        axes = ["k", "j", "i"]
    if dist is None:
        dist = [["linear", "linear", "linear"]] * Nvol

    assert len(dist) == Nvol

    # Make sure the sizes are the right type in each dimension. If an integer is
    # given, use that same size for every volume.
    size = [N0, N1, N2]
    for iVol, item in enumerate(size):
        if isinstance(item, int):
            size[iVol] = [item] * Nvol
        elif not isinstance(item, list):
            print("Incorrect type for N0, N1, or N2.")

        assert len(size[iVol]) == Nvol
    N0, N1, N2 = size

    f = open(fileName, "w")
    f.write(f"{Nvol}\n")

    def getDistribution(distIn, N):
        if not isinstance(distIn, str):
            assert len(distIn) == N
            dist = distIn.copy()
        elif distIn == "linear":
            dist = np.linspace(0, 1, N)
        elif distIn == "cosine":
            dist = (1 - np.cos(np.linspace(0, np.pi, N))) / 2.0
        elif distIn == "left":
            dist = np.linspace(0, 1, N) ** (3.0 / 2.0)
        elif distIn == "right":
            dist = np.linspace(0, 1, N) ** (2.0 / 3.0)
        return dist

    for i in range(Nvol):
        size = [N0[i], N1[i], N2[i]]
        Ni = size[axes.index("i")]
        Nj = size[axes.index("j")]
        Nk = size[axes.index("k")]
        f.write("%d\t%d\t%d\n" % (Ni, Nj, Nk))

    for iVol in range(Nvol):
        size = [N0[iVol], N1[iVol], N2[iVol]]
        Ni = size[axes.index("i")]
        Nj = size[axes.index("j")]
        Nk = size[axes.index("k")]
        # Get distributions for each axis
        d0 = getDistribution(dist[iVol][0], size[0])
        d1 = getDistribution(dist[iVol][1], size[1])
        d2 = getDistribution(dist[iVol][2], size[2])

        # Initialize coordinate arrays
        X = np.zeros(size + [3])

        for j in range(size[0]):
            P = slices[iVol, 0, 0] + np.outer(d0, (slices[iVol + 1, 0, 0] - slices[iVol, 0, 0]))[j]
            Q = slices[iVol, 0, 1] + np.outer(d0, (slices[iVol + 1, 0, 1] - slices[iVol, 0, 1]))[j]
            R = slices[iVol, 1, 0] + np.outer(d0, (slices[iVol + 1, 1, 0] - slices[iVol, 1, 0]))[j]
            S = slices[iVol, 1, 1] + np.outer(d0, (slices[iVol + 1, 1, 1] - slices[iVol, 1, 1]))[j]
            for k in range(size[1]):
                U = P + np.outer(d1, (R - P))[k]
                V = Q + np.outer(d1, (S - Q))[k]
                X[j, k] = U + np.outer(d2, (V - U))

        for dim in range(3):
            line = ""
            for k in range(Nk):
                for j in range(Nj):
                    for i in range(Ni):
                        idc = [-1, -1, -1]
                        idc[axes.index("i")] = i
                        idc[axes.index("j")] = j
                        idc[axes.index("k")] = k
                        line += f"{X[idc[0], idc[1], idc[2], dim]: .4e}\t"
                        if len(line) + 11 > 80:
                            f.write(line + "\n")
                            line = ""
            if len(line) > 0:
                f.write(line + "\n")

    f.close()


def createFittedWingFFD(surf, surfFormat, outFile, leList, teList, nSpan, nChord, absMargins, relMargins, liftIndex):

    """
    Generates a wing FFD with chordwise points that follow the airfoil geometry.

    Parameters
    ----------
    surf : pyGeo object or list or str
        The surface around which the FFD will be created.
        See the documentation for :meth:`pygeo.DVConstraints.setSurface` for details.

    surfFormat : str
        The surface format.
        See the documentation for :meth:`pygeo.DVConstraints.setSurface` for details.

    outFile : str
        Name of output file written in PLOT3D format.

    leList : list or array
        List or array of points (of size Nx3 where N is at least 2) defining the 'leading edge'.

    teList : list or array
        Same as leList but for the trailing edge.

    nSpan : int or list of int
        Number of spanwise sections in the FFD.
        Use a list of length N-1 to specify the number for each segment defined by leList and teList
        and to precisely match intermediate locations.

    nChord : int
        Number of chordwise points in the FFD.

    absMargins : list of float
        List with 3 items specifying the absolute margins in the [chordwise, spanwise, thickness] directions.
        This is useful for areas where the relative margin is too small, such as the trailing edge or wing tip.
        The total margin is the sum of the absolute and relative margins.

    relMargins : list of float
        List with 3 items specifying the relative margins in the [chordwise, spanwise, thickness] directions.
        Relative margins are applied as a fraction of local chord, wing span, and local thickness.
        The total margin is the sum of the absolute and relative margins.

    liftIndex : int
        Index specifying which direction lift is in (same as the ADflow option).
        Either 2 for the y-axis or 3 for the z-axis.
        This is used to determine the wing's spanwise direction.

    Examples
    --------
    >>> CFDSolver = ADFLOW(options=aeroOptions)
    >>> surf = CFDSolver.getTriangulatedMeshSurface()
    >>> surfFormat = "point-vector"
    >>> outFile = "wing_ffd.xyz"
    >>> nSpan = [4, 4]
    >>> nChord = 8
    >>> relMargins = [0.01, 0.001, 0.01]
    >>> absMargins = [0.05, 0.001, 0.05]
    >>> liftIndex = 3
    >>> createFittedWingFFD(surf, surfFormat, outFile, leList, teList, nSpan, nChord, absMargins, relMargins, liftIndex)

    """

    # Import inside this function to avoid circular imports
    from pygeo import DVConstraints

    # Set the triangulated surface in DVCon
    DVCon = DVConstraints()
    DVCon.setSurface(surf, surfFormat=surfFormat)

    # Get the surface intersections; surfCoords has dimensions [nSpanTotal, nChord, 2, 3]
    surfCoords = DVCon._generateIntersections(leList, teList, nSpan, nChord, surfaceName="default")

    nSpanTotal = np.sum(nSpan)

    # Initialize FFD coordinates to the surface coordinates
    FFDCoords = surfCoords.copy()

    # Swap axes to get the FFD coordinates into PLOT3D ordering [x, y, z, 3]
    FFDCoords = np.swapaxes(FFDCoords, 0, 1)  # [nChord, nSpanTotal, 2, 3]
    if liftIndex == 2:
        # Swap axes again so that z is the spanwise direction instead of y
        FFDCoords = np.swapaxes(FFDCoords, 1, 2)  # [nChord, 2, nSpanTotal, 3]

    # Assign coordinates and dimensions in each direction

    # x is always the chordwise direction
    leadingEdge = FFDCoords[0, :, :, 0]
    trailingEdge = FFDCoords[-1, :, :, 0]
    Nx = nChord

    # y and z depend on the liftIndex
    if liftIndex == 2:
        root = FFDCoords[:, :, 0, 2]
        tip = FFDCoords[:, :, -1, 2]

        upperSurface = FFDCoords[:, 0, :, 1]
        lowerSurface = FFDCoords[:, 1, :, 1]

        Ny = 2
        Nz = nSpanTotal
    elif liftIndex == 3:
        root = FFDCoords[:, 0, :, 1]
        tip = FFDCoords[:, -1, :, 1]

        upperSurface = FFDCoords[:, :, 1, 2]
        lowerSurface = FFDCoords[:, :, 0, 2]

        Ny = nSpanTotal
        Nz = 2
    else:
        raise ValueError("liftIndex must be 2 (for y-axis) or 3 (for z-axis)")

    # Add margins to FFD coordinates
    chordLength = trailingEdge - leadingEdge
    leadingEdge -= chordLength * relMargins[0] + absMargins[0]
    trailingEdge += chordLength * relMargins[0] + absMargins[0]

    span = np.max(tip - root)
    root -= span * relMargins[1] + absMargins[1]
    tip += span * relMargins[1] + absMargins[1]

    thickness = upperSurface - lowerSurface
    upperSurface += thickness * relMargins[2] + absMargins[2]
    lowerSurface -= thickness * relMargins[2] + absMargins[2]

    # Write FFD file
    f = open(outFile, "w")
    f.write("1\n")
    f.write(f"{Nx} {Ny} {Nz}\n")
    for ell in range(3):
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    f.write("%.15f " % (FFDCoords[i, j, k, ell]))
                f.write("\n")
    f.close()
