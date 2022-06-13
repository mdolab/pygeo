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


def readPlot3DSurfFile(fileName):
    """Read a plot3d file and return the points and connectivity in
    an unstructured mesh format"""

    pts = None

    f = open(fileName)
    nSurf = np.fromfile(f, "int", count=1, sep=" ")[0]
    sizes = np.fromfile(f, "int", count=3 * nSurf, sep=" ").reshape((nSurf, 3))
    nElem = 0
    for i in range(nSurf):
        nElem += (sizes[i, 0] - 1) * (sizes[i, 1] - 1)

    # Generate the uncompacted point and connectivity list:
    p0 = np.zeros((nElem * 2, 3))
    v1 = np.zeros((nElem * 2, 3))
    v2 = np.zeros((nElem * 2, 3))

    elemCount = 0

    for iSurf in range(nSurf):
        curSize = sizes[iSurf, 0] * sizes[iSurf, 1]
        pts = np.zeros((curSize, 3))
        for idim in range(3):
            pts[:, idim] = np.fromfile(f, "float", curSize, sep=" ")

        pts = pts.reshape((sizes[iSurf, 0], sizes[iSurf, 1], 3), order="f")
        for j in range(sizes[iSurf, 1] - 1):
            for i in range(sizes[iSurf, 0] - 1):
                # Each quad is split into two triangles
                p0[elemCount] = pts[i, j]
                v1[elemCount] = pts[i + 1, j] - pts[i, j]
                v2[elemCount] = pts[i, j + 1] - pts[i, j]

                elemCount += 1

                p0[elemCount] = pts[i + 1, j]
                v1[elemCount] = pts[i + 1, j + 1] - pts[i + 1, j]
                v2[elemCount] = pts[i, j + 1] - pts[i + 1, j]

                elemCount += 1

    return p0, v1, v2
