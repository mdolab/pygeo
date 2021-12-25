import numpy as np

# --------------------------------------------------------------
#                I/O Functions
# --------------------------------------------------------------


def readNValues(handle, N, dtype, binary=False, sep=" "):
    """Read N values of dtype 'float' or 'int' from file handle"""
    if binary:
        sep = ""

    if dtype == "int":
        values = np.fromfile(handle, dtype="int", count=N, sep=sep)
    else:
        values = np.fromfile(handle, dtype="float", count=N, sep=sep)
    return values


def writeValues(handle, values, dtype, binary=False):
    """Read N values of type 'float' or 'int' from file handle"""
    if binary:
        values.tofile(handle)
    else:
        if dtype == "float":
            values.tofile(handle, sep=" ", format="%f")
        elif dtype == "int":
            values.tofile(handle, sep=" ", format="%d")


def readAirfoilFile(fileName, bluntTe=False, bluntTaperRange=0.1, bluntThickness=0.002):
    """Load the airfoil file"""
    f = open(fileName)
    line = f.readline()  # Read (and ignore) the first line
    r = []
    try:
        r.append([float(s) for s in line.split()])
    except Exception:
        pass

    while 1:
        line = f.readline()
        if not line:
            break  # end of file
        if line.isspace():
            break  # blank line
        r.append([float(s) for s in line.split()])

    rr = np.array(r)
    x = rr[:, 0]
    y = rr[:, 1]
    npt = len(x)

    xMin = min(x)

    # There are 4 possibilites we have to deal with:
    # a. Given a sharp TE -- User wants a sharp TE
    # b. Given a sharp TE -- User wants a blunt TE
    # c. Given a blunt TE -- User wants a sharp TE
    # d. Given a blunt TE -- User wants a blunt TE
    #    (possibly with different TE thickness)

    # Check for blunt TE:
    if bluntTe is False:
        if y[0] != y[-1]:
            print("Blunt Trailing Edge on airfoil: %s" % (fileName))
            print("Merging to a point over final %f ..." % (bluntTaperRange))
            yAvg = 0.5 * (y[0] + y[-1])
            xAvg = 0.5 * (x[0] + x[-1])
            yTop = y[0]
            yBot = y[-1]
            xTop = x[0]
            xBot = x[-1]

            # Indices on the TOP surface of the wing
            indices = np.where(x[0 : npt // 2] >= (1 - bluntTaperRange))[0]
            for i in range(len(indices)):
                fact = (x[indices[i]] - (x[0] - bluntTaperRange)) / bluntTaperRange
                y[indices[i]] = y[indices[i]] - fact * (yTop - yAvg)
                x[indices[i]] = x[indices[i]] - fact * (xTop - xAvg)

            # Indices on the BOTTOM surface of the wing
            indices = np.where(x[npt // 2 :] >= (1 - bluntTaperRange))[0]
            indices = indices + npt // 2

            for i in range(len(indices)):
                fact = (x[indices[i]] - (x[-1] - bluntTaperRange)) / bluntTaperRange
                y[indices[i]] = y[indices[i]] - fact * (yBot - yAvg)
                x[indices[i]] = x[indices[i]] - fact * (xBot - xAvg)

    elif bluntTe is True:
        # Since we will be rescaling the TE regardless, the sharp TE
        # case and the case where the TE is already blunt can be
        # handled in the same manner

        # Get the current thickness
        curThick = y[0] - y[-1]

        # Set the new TE values:
        xBreak = 1.0 - bluntTaperRange

        # Rescale upper surface:
        for i in range(0, npt // 2):
            if x[i] > xBreak:
                s = (x[i] - xMin - xBreak) / bluntTaperRange
                y[i] += s * 0.5 * (bluntThickness - curThick)

        # Rescale lower surface:
        for i in range(npt // 2, npt):
            if x[i] > xBreak:
                s = (x[i] - xMin - xBreak) / bluntTaperRange
                y[i] -= s * 0.5 * (bluntThickness - curThick)

    return x, y


def writeAirfoilFile(fileName, name, x, y):
    """write an airfoil file"""
    f = open(fileName, "w")
    f.write("%s\n" % name)

    for i in range(len(x)):
        f.write(f"{x[i]:12.10f} {y[i]:12.10f}\n")

    f.close()


def getCoordinatesFromFile(fileName):
    """Get a list of coordinates from a file - useful for testing

    Parameters
    ----------
    fileName : str'
        filename for file

    Returns
    -------
    coordinates : list
        list of coordinates
    """

    f = open(fileName)
    coordinates = []
    for line in f:
        aux = line.split()
        coordinates.append([float(aux[0]), float(aux[1]), float(aux[2])])

    f.close()
    coordinates = np.transpose(np.array(coordinates))

    return coordinates


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
