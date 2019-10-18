from __future__ import print_function
from __future__ import division
# =============================================================================
# Utility Functions for Use in pyNetwork, pyGeo, pyBlock, DVGeometry,
# and pyLayout
# =============================================================================
import numpy as np
import sys, os
from pyspline import pySpline
# Set a (MUCH) larger recursion limit. For meshes with extremely large
# numbers of blocs (> 5000) the recursive edge propagation may hit a
# recursion limit.
sys.setrecursionlimit(10000)

# --------------------------------------------------------------
#                Rotation Functions
# --------------------------------------------------------------
def rotxM(theta):
    """Return x rotation matrix"""
    theta = theta*np.pi/180
    M = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], \
             [0, np.sin(theta), np.cos(theta)]]
    return M

def rotyM(theta):
    """ Return y rotation matrix"""
    theta = theta*np.pi/180
    M = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], \
             [-np.sin(theta), 0, np.cos(theta)]]
    return M

def rotzM(theta):
    """ Return z rotation matrix"""
    theta = theta*np.pi/180
    M = [[np.cos(theta), -np.sin(theta), 0], \
             [np.sin(theta), np.cos(theta), 0],[0, 0, 1]]
    return M

def rotxV(x, theta):
    """ Rotate a coordinate in the local x frame"""
    M = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], \
             [0, np.sin(theta), np.cos(theta)]]
    return np.dot(M, x)

def rotyV(x, theta):
    """Rotate a coordinate in the local y frame"""
    M = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], \
             [-np.sin(theta), 0, np.cos(theta)]]
    return np.dot(M, x)

def rotzV(x, theta):
    """Roate a coordinate in the local z frame"""
    M = [[np.cos(theta), -np.sin(theta), 0], \
             [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    return np.dot(M, x)

def rotVbyW(V, W, theta):
    """ Rotate a vector V, about an axis W by angle theta"""
    ux = W[0]
    uy = W[1]
    uz = W[2]

    c = np.cos(theta)
    s = np.sin(theta)
    if np.array(theta).dtype==np.dtype('D') or \
            np.array(W).dtype==np.dtype('D') or \
            np.array(V).dtype==np.dtype('D'):
        dtype = 'D'
    else:
        dtype = 'd'

    R = np.zeros((3, 3), dtype)

    R[0, 0] = ux**2 + (1-ux**2)*c
    R[0, 1] = ux*uy*(1-c) - uz*s
    R[0, 2] = ux*uz*(1-c) + uy*s

    R[1, 0] = ux*uy*(1-c) + uz*s
    R[1, 1] = uy**2 + (1-uy**2)*c
    R[1, 2] = uy*uz*(1-c) - ux*s

    R[2, 0] = ux*uz*(1-c) - uy*s
    R[2, 1] = uy*uz*(1-c) + ux*s
    R[2, 2] = uz**2+(1-uz**2)*c

    return np.dot(R, V)

 # -------------------------------------------------------------
 #               Norm Functions
 # -------------------------------------------------------------

def euclideanNorm(inVec):
    """
    perform the euclidean 2 norm of the vector inVec
    required because linalg.norm() provides incorrect results for
    CS derivatives.
    """
    inVec = np.array(inVec)
    temp = 0.0
    for i in range(inVec.shape[0]):
        temp += inVec[i]**2

    return np.sqrt(temp)

def cross_b(a, b,  crossb):
    """
    Do the reverse accumulation through a cross product.
    """
    ab = np.zeros_like(a)
    bb = np.zeros_like(b)

    ab[0] = ab[0] + b[1]*crossb[2]
    bb[1] = bb[1] + a[0]*crossb[2]
    ab[1] = ab[1] - b[0]*crossb[2]
    bb[0] = bb[0] - a[1]*crossb[2]
    crossb[2] = 0.0
    ab[2] = ab[2] + b[0]*crossb[1]
    bb[0] = bb[0] + a[2]*crossb[1]
    ab[0] = ab[0] - b[2]*crossb[1]
    bb[2] = bb[2] - a[0]*crossb[1]
    crossb[1] = 0.0
    ab[1] = ab[1] + b[2]*crossb[0]
    bb[2] = bb[2] + a[1]*crossb[0]
    ab[2] = ab[2] - b[1]*crossb[0]
    bb[1] = bb[1] - a[2]*crossb[0]
    crossb[0] = 0.0
    return ab, bb

def dot_b(a, b, dotb):
    """
    Do the reverse accumulation through a dot product.
    """
    ab = np.zeros_like(a)
    bb = np.zeros_like(b)

    ab = ab + b*dotb
    bb = bb + a*dotb

    return ab, bb

def calculateCentroid(p0, v1, v2):
    '''
    take in a triangulated surface and calculate the centroid
    '''
    p0 = np.array(p0)
    p1 = np.array(v1)+p0
    p2 = np.array(v2)+p0

    #compute the areas
    areaVec = np.cross(v1, v2)/2.0
    area = np.linalg.norm(areaVec, axis=1)

    # compute the cell centroids
    cellCenter = (p0+p1+p2)/3.

    centerSum = area.dot(cellCenter)
    areaSum = np.sum(area)

    centroid = centerSum/areaSum

    return centroid

def calculateAverageNormal(p0, v1, v2):
    '''
    take in a triangulated surface and calculate the centroid
    '''
    p0 = np.array(p0)
    p1 = np.array(v1)+p0
    p2 = np.array(v2)+p0

    #compute the normal of each triangle
    normal = np.cross(v1, v2)
    sumNormal = np.sum(normal,axis=0)
    lengthNorm = np.linalg.norm(sumNormal)

    unitNormal = sumNormal/lengthNorm

    return unitNormal

def calculateRadii(centroid, p0, v1, v2):
    '''
    take the centroid and compute inner and outer radii of surface
    '''
    p0 = np.array(p0)
    p1 = np.array(v1)+p0
    p2 = np.array(v2)+p0

    # take the difference between the points and the centroid
    d0 = p0-centroid
    d1 = p1-centroid
    d2 = p2-centroid

    radO = np.zeros(3)
    radI = np.zeros(3)
    d0 = np.linalg.norm(d0, axis=1)
    radO[0] = np.max(d0)
    radI[0] = np.min(d0)
    d1 = np.linalg.norm(d1, axis=1)
    radO[1] = np.max(d1)
    radI[1] = np.min(d1)
    d2 = np.linalg.norm(d2, axis=1)
    radO[2] = np.max(d2)
    radI[2] = np.min(d2)

    outerRadius = np.max(radO)
    innerRadius = np.min(radI)

    return innerRadius,outerRadius

 # --------------------------------------------------------------
 #                I/O Functions
 # --------------------------------------------------------------

def readNValues(handle, N, dtype, binary=False, sep=' '):
    """Read N values of dtype 'float' or 'int' from file handle"""
    if binary:
        sep = ""

    if dtype == 'int':
        values = np.fromfile(handle, dtype='int', count=N, sep=sep)
    else:
        values = np.fromfile(handle, dtype='float', count=N, sep=sep)
    return values

def writeValues(handle, values, dtype, binary=False):
    """Read N values of type 'float' or 'int' from file handle"""
    if binary:
        values.tofile(handle)
    else:
        if dtype == 'float':
            values.tofile(handle, sep=" ", format="%f")
        elif dtype == 'int':
            values.tofile(handle, sep=" ", format="%d")

def readAirfoilFile(fileName, bluntTe=False, bluntTaperRange=0.1,
             bluntThickness=.002):
    """ Load the airfoil file"""
    f = open(fileName, 'r')
    line  = f.readline() # Read (and ignore) the first line
    r = []
    try:
        r.append([float(s) for s in line.split()])
    except:
        r = []

    while 1:
        line = f.readline()
        if not line:
            break # end of file
        if line.isspace():
            break # blank line
        r.append([float(s) for s in line.split()])

    rr = np.array(r)
    x = rr[:, 0]
    y = rr[:, 1]
    npt = len(x)

    xMin = min(x)
    xMax = max(x)

    # There are 4 possibilites we have to deal with:
    # a. Given a sharp TE -- User wants a sharp TE
    # b. Given a sharp TE -- User wants a blunt TE
    # c. Given a blunt TE -- User wants a sharp TE
    # d. Given a blunt TE -- User wants a blunt TE
    #    (possibly with different TE thickness)

    # Check for blunt TE:
    if bluntTe is False:
        if y[0] != y[-1]:
            print('Blunt Trailing Edge on airfoil: %s'%(fileName))
            print('Merging to a point over final %f ...'%(bluntTaperRange))
            yAvg = 0.5*(y[0] + y[-1])
            xAvg = 0.5*(x[0] + x[-1])
            yTop = y[0]
            yBot = y[-1]
            xTop = x[0]
            xBot = x[-1]

            # Indices on the TOP surface of the wing
            indices = np.where(x[0:npt//2]>=(1-bluntTaperRange))[0]
            for i in range(len(indices)):
                fact = (x[indices[i]]- (x[0]-bluntTaperRange))/bluntTaperRange
                y[indices[i]] = y[indices[i]]- fact*(yTop-yAvg)
                x[indices[i]] = x[indices[i]]- fact*(xTop-xAvg)

            # Indices on the BOTTOM surface of the wing
            indices = np.where(x[npt//2:]>=(1-bluntTaperRange))[0]
            indices = indices + npt//2

            for i in range(len(indices)):
                fact = (x[indices[i]]- (x[-1]-bluntTaperRange))/bluntTaperRange
                y[indices[i]] = y[indices[i]]- fact*(yBot-yAvg)
                x[indices[i]] = x[indices[i]]- fact*(xBot-xAvg)

    elif bluntTe is True:
        # Since we will be rescaling the TE regardless, the sharp TE
        # case and the case where the TE is already blunt can be
        # handled in the same manner

        # Get the current thickness
        curThick = y[0] - y[-1]

        # Set the new TE values:
        xBreak = 1.0-bluntTaperRange

        # Rescale upper surface:
        for i in range(0,npt//2):
            if x[i] > xBreak:
                s = (x[i]-xMin-xBreak)/bluntTaperRange
                y[i] += s*0.5*(bluntThickness-curThick)

        # Rescale lower surface:
        for i in range(npt//2,npt):
            if x[i] > xBreak:
                s = (x[i]-xMin-xBreak)/bluntTaperRange
                y[i] -= s*0.5*(bluntThickness-curThick)

    return x, y


def writeAirfoilFile(fileName,name, x,y):
    """ write an airfoil file """
    f = open(fileName, 'w')
    f.write("%s\n"%name)

    for i in range(len(x)):
        f.write('%12.10f %12.10f\n'%(x[i],y[i]))

    f.close()


    return

def getCoordinatesFromFile(fileName):
    """Get a list of coordinates from a file - useful for testing
    Required:
        fileName: filename for file
    Returns:
        coordinates: list of coordinates
    """

    f = open(fileName, 'r')
    coordinates = []
    for line in f:
        aux = line.split()
        coordinates.append([float(aux[0]), float(aux[1]), float(aux[2])])

    f.close()
    coordinates = np.transpose(np.array(coordinates))

    return coordinates

def write_wing_FFD_file(fileName, slices, N0, N1, N2, axes=None, dist=None):
    '''
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

    Example of two volumes
    -------
    axes = ['k', 'j', 'i']
    slices = numpy.array([
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

    '''

    Nvol = slices.shape[0] - 1

    if axes is None:
        axes=['k', 'j', 'i']
    if dist is None:
        dist = [['linear', 'linear', 'linear']]*Nvol

    assert(len(dist) == Nvol)

    # Make sure the sizes are the right type in each dimension. If an integer is
    # given, use that same size for every volume.
    size = [N0, N1, N2]
    for i, item in enumerate(size):
        if type(item) is int:
            size[i] = [item]*Nvol
        elif type(item) is not list:
            print('Incorrect type for N0, N1, or N2.')

        assert(len(size[i]) == Nvol)
    N0, N1, N2 = size

    f = open(fileName, 'w')
    f.write('{}\n'.format(Nvol))

    def getDistribution(distIn, N):
        if type(distIn) is not str:
            assert(len(distIn) == N)
            dist = distIn.copy()
        elif distIn == 'linear':
            dist = np.linspace(0, 1, N)
        elif distIn == 'cosine':
            dist = (1 - np.cos(np.linspace(0, np.pi, N))) / 2.0
        elif distIn == 'left':
            dist = np.linspace(0, 1, N)**(3.0/2.0)
        elif distIn == 'right':
            dist = np.linspace(0, 1, N)**(2.0/3.0)
        return dist

    for i in range(Nvol):
        size = [N0[i], N1[i], N2[i]]
        Ni = size[axes.index('i')]
        Nj = size[axes.index('j')]
        Nk = size[axes.index('k')]
        f.write('%d\t%d\t%d\n' % (Ni, Nj, Nk))

    for i in range(Nvol):
        size = [N0[i], N1[i], N2[i]]
        Ni = size[axes.index('i')]
        Nj = size[axes.index('j')]
        Nk = size[axes.index('k')]
        # Get distributions for each axis
        d0 = getDistribution(dist[i][0], size[0])
        d1 = getDistribution(dist[i][1], size[1])
        d2 = getDistribution(dist[i][2], size[2])

        # Initialize coordinate arrays
        X = np.zeros(size + [3])

        for j in range(size[0]):
            P = slices[i,0,0] + np.outer(d0, (slices[i+1,0,0] - slices[i,0,0]))[j]
            Q = slices[i,0,1] + np.outer(d0, (slices[i+1,0,1] - slices[i,0,1]))[j]
            R = slices[i,1,0] + np.outer(d0, (slices[i+1,1,0] - slices[i,1,0]))[j]
            S = slices[i,1,1] + np.outer(d0, (slices[i+1,1,1] - slices[i,1,1]))[j]
            for k in range(size[1]):
                U = P + np.outer(d1, (R - P))[k]
                V = Q + np.outer(d1, (S - Q))[k]
                X[j,k] = U + np.outer(d2, (V - U))


        for dim in range(3):
            line = ''
            for j in range(Nk):
                for k in range(Nj):
                    for l in range(Ni):
                        idc = [-1, -1, -1]
                        idc[axes.index('i')] = l
                        idc[axes.index('j')] = k
                        idc[axes.index('k')] = j
                        line += '{: .4e}\t'.format(X[idc[0],idc[1],idc[2],dim])
                        if len(line) + 11 > 80:
                            f.write(line+'\n')
                            line = ''
            if len(line) > 0:
                f.write(line+'\n')

    f.close()
# --------------------------------------------------------------
#            Working with Edges Function
# --------------------------------------------------------------

def eDist(x1, x2):
    """Get the eculidean distance between two points"""
    return euclideanNorm(x1-x2)#np.linalg.norm(x1-x2)

def eDist2D(x1, x2):
    """Get the eculidean distance between two points"""
    return np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)

def eDist_b(x1, x2):
    x1b = 0.0
    x2b = 0.0
    db  = 1.0
    x1b = np.zeros(3)
    x2b = np.zeros(3)
    if ((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 + (x1[2]-x2[2])**2 == 0.0):
        tempb = 0.0
    else:
        tempb = db/(2.0*np.sqrt(
                (x1[0]-x2[0])**2+(x1[1]-x2[1])**2+(x1[2]-x2[2])**2))

    tempb0 = 2*(x1[0]-x2[0])*tempb
    tempb1 = 2*(x1[1]-x2[1])*tempb
    tempb2 = 2*(x1[2]-x2[2])*tempb
    x1b[0] = tempb0
    x2b[0] = -tempb0
    x1b[1] = tempb1
    x2b[1] = -tempb1
    x1b[2] = tempb2
    x2b[2] = -tempb2

    return x1b, x2b


# --------------------------------------------------------------
#             Truly Miscellaneous Functions
# --------------------------------------------------------------

def unique(s):
    """Return a list of the elements in s, but without duplicates.

    For example, unique([1,2,3,1,2,3]) is some permutation of [1,2,3],
    unique("abcabc") some permutation of ["a", "b", "c"], and
    unique(([1, 2], [2, 3], [1, 2])) some permutation of
    [[2, 3], [1, 2]].

    For best speed, all sequence elements should be hashable.  Then
    unique() will usually work in linear time.

    If not possible, the sequence elements should enjoy a total
    ordering, and if list(s).sort() doesn't raise TypeError it's
    assumed that they do enjoy a total ordering.  Then unique() will
    usually work in O(N*log2(N)) time.

    If that's not possible either, the sequence elements must support
    equality-testing.  Then unique() will usually work in quadratic
    time.
    """

    n = len(s)
    if n == 0:
        return []

    # Try using a dict first, as that's the fastest and will usually
    # work.  If it doesn't work, it will usually fail quickly, so it
    # usually doesn't np.cost much to *try* it.  It requires that all the
    # sequence elements be hashable, and support equality comparison.
    u = {}
    try:
        for x in s:
            u[x] = 1
    except TypeError:
        pass
    else:
        return sorted(list(u.keys()))

    # We can't hash all the elements.  Second fastest is to sort,
    # which brings the equal elements together; then duplicates are
    # easy to weed out in a single pass.
    # NOTE:  Python's list.sort() was designed to be efficient in the
    # presence of many duplicate elements.  This isn't true of all
    # sort functions in all languages or libraries, so this approach
    # is more effective in Python than it may be elsewhere.

    try:
        t = list(s)
        t.sort()
    except TypeError:
        pass
    else:
        assert n > 0
        last = t[0]
        lasti = i = 1
        while i < n:
            if t[i] != last:
                t[lasti] = last = t[i]
                lasti += 1
            i += 1
        return t[:lasti]

    # Brute force is all that's left.

    u = []
    for x in s:
        if x not in u:
            u.append(x)
    return u

def uniqueIndex(s, sHash=None):
    """
    This function is based on unique

    The idea is to take a list s, and reduce it as per unique.

    However, it additionally calculates a linking index arrary that is
    the same size as the original s, and points to where it ends up in
    the the reduced list

    if sHash is not specified for sorting, s is used

    """
    if sHash is not None:
        ind = np.argsort(np.argsort(sHash))
    else:
        ind = np.argsort(np.argsort(s))

    n = len(s)
    t = list(s)
    t.sort()

    diff = np.zeros(n, 'bool')

    last = t[0]
    lasti = i = 1
    while i < n:
        if t[i] != last:
            t[lasti] = last = t[i]
            lasti += 1
        else:
            diff[i] = True
        i += 1

    b = np.where(diff)[0]
    for i in range(n):
        ind[i] -= b.searchsorted(ind[i], side='right')

    return t[:lasti], ind

def pointReduce(points, nodeTol=1e-4):
    """Given a list of N points in ndim space, with possible
    duplicates, return a list of the unique points AND a pointer list
    for the original points to the reduced set"""

    # First
    points = np.array(points)
    N = len(points)
    if N == 0:
        return points, None
    dists = []
    for ipt in range(N):
        dists.append(np.sqrt(np.dot(points[ipt], points[ipt])))

    temp = np.array(dists)
    temp.sort()
    ind = np.argsort(dists)
    i = 0
    cont = True
    newPoints = []
    link = np.zeros(N, 'intc')
    linkCounter = 0

    while cont:
        cont2 = True
        tempInd = []
        j = i
        while cont2:
            if abs(dists[ind[i]]-dists[ind[j]])<nodeTol:
                tempInd.append(ind[j])
                j = j + 1
                if j == N: # Overrun check
                    cont2 = False
            else:
                cont2 = False

        subPoints = [] # Copy of the list of sub points with the dists
        for ii in range(len(tempInd)):
            subPoints.append(points[tempInd[ii]])

        # Brute Force Search them
        subUniquePts, subLink = pointReduceBruteForce(subPoints, nodeTol)
        newPoints.extend(subUniquePts)

        for ii in range(len(tempInd)):
            link[tempInd[ii]] = subLink[ii] + linkCounter

        linkCounter += max(subLink) + 1

        i = j - 1 + 1
        if i == N:
            cont = False

    return np.array(newPoints), np.array(link)

def pointReduceBruteForce(points,  nodeTol=1e-4):
    """Given a list of N points in ndim space, with possible
    duplicates, return a list of the unique points AND a pointer list
    for the original points to the reduced set

    BRUTE FORCE VERSION

    """
    N = len(points)
    if N == 0:
        return points, None
    uniquePoints = [points[0]]
    link = [0]
    for i in range(1, N):
        foundIt = False
        for j in range(len(uniquePoints)):
            if eDist(points[i], uniquePoints[j]) < nodeTol:
                link.append(j)
                foundIt = True
                break

        if not foundIt:
            uniquePoints.append(points[i])
            link.append(j+1)

    return np.array(uniquePoints), np.array(link)

def edgeOrientation(e1, e2):
    """Compare two edge orientations. Basically if the two nodes are
    in the same order return 1 if they are in opposite order, return
    1"""

    if [e1[0], e1[1]] == [e2[0], e2[1]]:
        return 1
    elif [e1[1], e1[0]] == [e2[0], e2[1]]:
        return -1
    else:
        print('Error with edgeOrientation: Not possible.')
        print('Orientation 1 [%d %d]'%(e1[0], e1[1]))
        print('Orientation 2 [%d %d]'%(e2[0], e2[1]))
        sys.exit(0)

def faceOrientation(f1, f2):
    """Compare two face orientations f1 and f2 and return the
    transform to get f1 back to f2"""

    if [f1[0], f1[1], f1[2], f1[3]] == [f2[0], f2[1], f2[2], f2[3]]:
        return 0
    elif [f1[0], f1[1], f1[2], f1[3]] == [f2[1], f2[0], f2[3], f2[2]]:
        return 1
    elif [f1[0], f1[1], f1[2], f1[3]] == [f2[2], f2[3], f2[0], f2[1]]:
        return 2
    elif [f1[0], f1[1], f1[2], f1[3]] == [f2[3], f2[2], f2[1], f2[0]]:
        return 3
    elif [f1[0], f1[1], f1[2], f1[3]] == [f2[0], f2[2], f2[1], f2[3]]:
        return 4
    elif [f1[0], f1[1], f1[2], f1[3]] == [f2[2], f2[0], f2[3], f2[1]]:
        return 5
    elif [f1[0], f1[1], f1[2], f1[3]] == [f2[1], f2[3], f2[0], f2[2]]:
        return 6
    elif [f1[0], f1[1], f1[2], f1[3]] == [f2[3], f2[1], f2[2], f2[0]]:
        return 7
    else:
        print('Error with faceOrientation: Not possible.')
        print('Orientation 1 [%d %d %d %d]'%(f1[0], f1[1], f1[2], f1[3]))
        print('Orientation 2 [%d %d %d %d]'%(f2[0], f2[1], f2[2], f2[3]))
        sys.exit(0)

def quadOrientation(pt1, pt2):
    """Given two sets of 4 points in ndim space, pt1 and pt2,
    determine the orientation of pt2 wrt pt1
    This works for both exact quads and "loosely" oriented quads
    ."""
    dist = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            dist[i, j] = eDist(pt1[i], pt2[j])

    # Now compute the 8 distances for the 8 possible orientation
    sumDist = np.zeros(8)
    # corners = [0, 1, 2, 3]
    sumDist[0] = dist[0, 0] + dist[1, 1] + dist[2, 2] + dist[3, 3]
    # corners = [1, 0, 3, 2]
    sumDist[1] = dist[0, 1] + dist[1, 0] + dist[2, 3] + dist[3, 2]
    # corners = [2, 3, 0, 1]
    sumDist[2] = dist[0, 2] + dist[1, 3] + dist[2, 0] + dist[3, 1]
    # corners = [3, 2, 1, 0]
    sumDist[3] = dist[0, 3] + dist[1, 2] + dist[2, 1] + dist[3, 0]
    # corners = [0, 2, 1, 3]
    sumDist[4] = dist[0, 0] + dist[1, 2] + dist[2, 1] + dist[3, 3]
    # corners = [2, 0, 3, 1]
    sumDist[5] = dist[0, 2] + dist[1, 0] + dist[2, 3] + dist[3, 1]
    # corners = [1, 3, 0, 2]
    sumDist[6] = dist[0, 1] + dist[1, 3] + dist[2, 0] + dist[3, 2]
    # corners = [3, 1, 2, 0]
    sumDist[7] = dist[0, 3] + dist[1, 1] + dist[2, 2] + dist[3, 0]

    index = sumDist.argmin()

    return index

def orientArray(index, inArray):
    """Take an input array inArray, and rotate/flip according to the index
    output from quadOrientation"""

    if index == 0:
        outArray = inArray.copy()
    elif index == 1:
        outArray = rotateCCW(inArray)
        outArray = rotateCCW(outArray)
        outArray = reverseRows(outArray)
    elif index == 2:
        outArray = reverseRows(inArray)
    elif index == 3:
        outArray = rotateCCW(inArray) # Verified working
        outArray = rotateCCW(outArray)
    elif index == 4:
        outArray = rotateCW(inArray)
        outArray = reverseRows(outArray)
    elif index == 5:
        outArray = rotateCCW(inArray)
    elif index == 6:
        outArray = rotateCW(inArray)
    elif index == 7:
        outArray = rotateCCW(inArray)
        outArray = reverseRows(outArray)

    return outArray

def directionAlongSurface(surface, line):
    """Determine the dominate (u or v) direction of line along surface"""
    # Now Do two tests: Take N points in u and test N groups
    # against dn and take N points in v and test the N groups
    # again

    N = 3
    sn = np.linspace(0, 1, N)
    dn = np.zeros((N, 3))
    s = np.linspace(0, 1, N)
    for i in range(N):
        dn[i, :] = line.getDerivative(sn[i])

    uDotTot = 0
    for i in range(N):
        for n in range(N):
            du, dv = surface.getDerivative(s[i], s[n])
            uDotTot += np.dot(du, dn[n, :])

    vDotTot = 0
    for j in range(N):
        for n in range(N):
            du, dv = surface.getDerivative(s[n], s[j])
            vDotTot += np.dot(dv, dn[n, :])

    if abs(uDotTot) > abs(vDotTot):
        # Its along u now get
        if uDotTot >= 0:
            return 0 # U same direction
        else:
            return 1 # U opposite direction
    else:
        if vDotTot >= 0:
            return 2 # V same direction
        else:
            return 3 # V opposite direction

def curveDirection(curve1, curve2):
    """Determine if the direction of curve 1 is basically in the same
    direction as curve2. Return 1 for same direction, -1 for opposite
    direction"""

    N = 4
    s = np.linspace(0, 1, N)
    tot = 0
    dForward = 0
    dBackward = 0
    for i in range(N):
        tot += np.dot(curve1.getDerivative(s[i]), curve2.getDerivative(s[i]))
        dForward += eDist(curve1.getValue(s[i]), curve2.getValue(s[i]))
        dBackward += eDist(curve1.getValue(s[i]), curve2.getValue(s[N-i-1]))

    if tot > 0:
        return tot, dForward
    else:
        return tot, dBackward

def indexPosition1D(i, N):
    """This function is a generic function which determines if index
    over a list of length N is an interior point or node 0 or node 1.
    """
    if i > 0 and i < N-1: # Interior
        return 0, None
    elif i == 0: # Node 0
        return 1, 0
    elif i == N-1: # Node 1
        return 1, 1

def indexPosition2D(i, j, N, M):
    """This function is a generic function which determines if for a grid
    of data NxM with index i going 0->N-1 and j going 0->M-1, it
    determines if i,j is on the interior, on an edge or on a corner

    The funtion return four values:
    type: this is 0 for interior, 1 for on an edge and 2 for on a corner
    edge: this is the edge number if type==1
    node: this is the node number if type==2
    index: this is the value index along the edge of interest --
    only defined for edges"""

    if i > 0 and i < N - 1 and j > 0 and j < M-1: # Interior
        return 0, None, None, None
    elif i > 0 and i < N - 1 and j == 0:     # Edge 0
        return 1, 0, None, i
    elif i > 0 and i < N - 1 and j == M - 1: # Edge 1
        return 1, 1, None, i
    elif i == 0 and j > 0 and j < M - 1:     # Edge 2
        return 1, 2, None, j
    elif i == N - 1 and j > 0 and j < M - 1: # Edge 3
        return 1, 3, None, j
    elif i == 0 and j == 0:                  # Node 0
        return 2, None, 0, None
    elif i == N - 1 and j == 0:              # Node 1
        return 2, None, 1, None
    elif i == 0 and j == M -1 :              # Node 2
        return 2, None, 2, None
    elif i == N - 1 and j == M - 1:          # Node 3
        return 2, None, 3, None

def indexPosition3D(i, j, k, N, M, L):
    """This function is a generic function which determines if for a
    3dgrid of data NxMXL with index i going 0->N-1 and j going 0->M-1
    k going 0->L-1, it determines if i,j,k is on the interior, on a
    face, on an edge or on a corner

    The funtion return theses values:
    type: this is 0 for interior, 1 for on an face,
           3 for an edge and 4 for on a corner
    number: this is the face number if type==1,
            this is the edge number if type==2
            this is the node number if type==3

    index1: this is the value index along 0th dir the face
        of interest OR edge of interest
    index2: this is the value index along 1st dir the face
        of interest
        """

    # Note to interior->Faces->Edges->Nodes to minimize number of if checks

    # Interior:
    if i > 0 and i < N-1 and j > 0 and j < M-1 and k > 0 and k < L-1:
        return 0, None, None, None

    elif i > 0 and i < N-1 and j > 0 and j < M-1 and k == 0:   # Face 0
        return 1, 0, i, j
    elif i > 0 and i < N-1 and j > 0 and j < M-1 and k == L-1: # Face 1
        return 1, 1, i, j
    elif i == 0 and j > 0 and j < M-1 and k > 0 and k < L-1:   # Face 2
        return 1, 2, j, k
    elif i == N-1 and j > 0 and j < M-1 and k > 0 and k < L-1: # Face 3
        return 1, 3, j, k
    elif i > 0 and i < N-1 and j == 0 and k > 0 and k < L-1:   # Face 4
        return 1, 4, i, k
    elif i > 0 and i < N-1 and j == M-1 and k > 0 and k < L-1: # Face 5
        return 1, 5, i, k

    elif i > 0 and i < N-1 and j == 0 and k == 0:       # Edge 0
        return 2, 0, i, None
    elif i > 0 and i < N-1 and j == M-1 and k == 0:     # Edge 1
        return 2, 1, i, None
    elif i == 0 and j > 0 and j < M-1 and k == 0:       # Edge 2
        return 2, 2, j, None
    elif i == N-1 and j > 0 and j < M-1 and k == 0:     # Edge 3
        return 2, 3, j, None
    elif i > 0 and i < N-1 and j == 0 and k == L-1:     # Edge 4
        return 2, 4, i, None
    elif i > 0 and i < N-1 and j == M-1 and k == L-1:   # Edge 5
        return 2, 5, i, None
    elif i == 0 and j > 0 and j < M-1 and k == L-1:     # Edge 6
        return 2, 6, j, None
    elif i == N-1 and j > 0 and j < M-1 and k == L-1:   # Edge 7
        return 2, 7, j, None
    elif i == 0 and j == 0 and k > 0 and k < L-1:       # Edge 8
        return 2, 8, k, None
    elif i == N-1 and j == 0 and k > 0 and k < L-1:     # Edge 9
        return 2, 9, k, None
    elif i == 0 and j == M-1 and k > 0 and k < L-1:     # Edge 10
        return 2, 10, k, None
    elif i == N-1 and j == M-1 and k > 0 and k < L-1:   # Edge 11
        return 2, 11, k, None

    elif i == 0 and j == 0 and k == 0:                  # Node 0
        return 3, 0, None, None
    elif i == N-1 and j == 0 and k == 0:                # Node 1
        return 3, 1, None, None
    elif i == 0 and j == M-1 and k == 0:                # Node 2
        return 3, 2, None, None
    elif i == N-1 and j == M-1 and k == 0:              # Node 3
        return 3, 3, None, None
    elif i == 0 and j == 0 and k == L-1:                # Node 4
        return 3, 4, None, None
    elif i == N-1 and j == 0 and k == L-1:              # Node 5
        return 3, 5, None, None
    elif i == 0 and j == M-1 and k == L-1:              # Node 6
        return 3, 6, None, None
    elif i == N-1 and j == M-1 and k == L-1:            # Node 7
        return 3, 7, None, None

# --------------------------------------------------------------
#                     Node/Edge Functions
# --------------------------------------------------------------

def edgeFromNodes(n1, n2):
    """Return the edge coorsponding to nodes n1, n2"""
    if (n1 == 0 and n2 == 1) or (n1 == 1 and n2 == 0):
        return 0
    elif (n1 == 0 and n2 == 2) or (n1 == 2 and n2 == 0):
        return 2
    elif (n1 == 3 and n2 == 1) or (n1 == 1 and n2 == 3):
        return 3
    elif (n1 == 3 and n2 == 2) or (n1 == 2 and n2 == 3):
        return 1

def edgesFromNode(n):
    """ Return the two edges coorsponding to node n"""
    if n == 0:
        return 0, 2
    if n == 1:
        return 0, 3
    if n == 2:
        return 1, 2
    if n == 3:
        return 1, 3

def edgesFromNodeIndex(n, N, M):
    """ Return the two edges coorsponding to node n AND return the index
of the node on the edge according to the size (N, M)"""
    if n == 0:
        return 0, 2, 0, 0
    if n == 1:
        return 0, 3, N-1, 0
    if n == 2:
        return 1, 2, 0, M-1
    if n == 3:
        return 1, 3, N-1, M-1

def nodesFromEdge(edge):
    """Return the nodes on either edge of a standard edge"""
    if edge == 0:
        return 0, 1
    elif edge == 1:
        return 2, 3
    elif edge == 2:
        return 0, 2
    elif edge == 3:
        return 1, 3
    elif edge == 4:
        return 4, 5
    elif edge == 5:
        return 6, 7
    elif edge == 6:
        return 4, 6
    elif edge == 7:
        return 5, 7
    elif edge == 8:
        return 0, 4
    elif edge == 9:
        return 1, 5
    elif edge == 10:
        return 2, 6
    elif edge == 11:
        return 3, 7

# Volume Face/edge functions
def nodesFromFace(face):
    if face == 0:
        return [0, 1, 2, 3]
    elif face == 1:
        return [4, 5, 6, 7]
    elif face == 2:
        return [0, 2, 4, 6]
    elif face == 3:
        return [1, 3, 5, 7]
    elif face == 4:
        return [0, 1, 4, 5]
    elif face == 5:
        return [2, 3, 6, 7]

def edgesFromFace(face):
    if face == 0:
        return [0, 1, 2, 3]
    elif face == 1:
        return [4, 5, 6, 7]
    elif face == 2:
        return [2, 6, 8, 10]
    elif face == 3:
        return [3, 7, 9, 11]
    elif face == 4:
        return [0, 4, 8, 9]
    elif face == 5:
        return [1, 5, 10, 11]

def setNodeValue(arr, value, nodeIndex):
    # Set "value" in 3D array "arr" at node "nodeIndex":
    if nodeIndex == 0:
        arr[0,0,0] = value
    elif nodeIndex == 1:
        arr[-1,0,0] = value
    elif nodeIndex == 2:
        arr[0,-1,0] = value
    elif nodeIndex == 3:
        arr[-1,-1,0] = value

    if nodeIndex == 4:
        arr[0,0,-1] = value
    elif nodeIndex == 5:
        arr[-1,0,-1] = value
    elif nodeIndex == 6:
        arr[0,-1,-1] = value
    elif nodeIndex == 7:
        arr[-1,-1,-1] = value

    return arr

def setEdgeValue(arr, values, dir, edgeIndex):

    if dir == -1: # Reverse values
        values = values[::-1]

    if edgeIndex == 0:
        arr[1:-1,0,0] = values
    elif edgeIndex == 1:
        arr[1:-1,-1,0] = values
    elif edgeIndex == 2:
        arr[0, 1:-1, 0] = values
    elif edgeIndex == 3:
        arr[-1, 1:-1, 0] = values

    elif edgeIndex == 4:
        arr[1:-1,0,-1] = values
    elif edgeIndex == 5:
        arr[1:-1,-1,-1] = values
    elif edgeIndex == 6:
        arr[0, 1:-1, -1] = values
    elif edgeIndex == 7:
        arr[-1, 1:-1, -1] = values

    elif edgeIndex == 8:
        arr[0,0,1:-1] = values
    elif edgeIndex == 9:
        arr[-1,0,1:-1] = values
    elif edgeIndex == 10:
        arr[0,-1,1:-1] = values
    elif edgeIndex == 11:
        arr[-1,-1,1:-1] = values

    return arr

def setFaceValue(arr, values, faceDir, faceIndex):

    # Orient the array first according to the dir:

    values = orientArray(faceDir, values)

    if faceIndex == 0:
        arr[1:-1,1:-1,0] = values
    elif faceIndex == 1:
        arr[1:-1,1:-1,-1] = values
    elif faceIndex == 2:
        arr[0,1:-1,1:-1] = values
    elif faceIndex == 3:
        arr[-1,1:-1,1:-1] = values
    elif faceIndex == 4:
        arr[1:-1,0,1:-1] = values
    elif faceIndex == 5:
        arr[1:-1,-1,1:-1] = values

    return arr

def setFaceValue2(arr, values, faceDir, faceIndex):

    # Orient the array first according to the dir:

    values = orientArray(faceDir, values)

    if faceIndex == 0:
        arr[1:-1,1:-1,0] = values
    elif faceIndex == 1:
        arr[1:-1,1:-1,-1] = values
    elif faceIndex == 2:
        arr[0,1:-1,1:-1] = values
    elif faceIndex == 3:
        arr[-1,1:-1,1:-1] = values
    elif faceIndex == 4:
        arr[1:-1,0,1:-1] = values
    elif faceIndex == 5:
        arr[1:-1,-1,1:-1] = values

    return arr

def getFaceValue(arr, faceIndex, offset):
    # Return the values from 'arr' on faceIndex with offset of offset:

    if   faceIndex == 0:
        values = arr[:,:,offset]
    elif faceIndex == 1:
        values = arr[:,:,-1-offset]
    elif faceIndex == 2:
        values = arr[offset,:,:]
    elif faceIndex == 3:
        values = arr[-1-offset,:,:]
    elif faceIndex == 4:
        values = arr[:,offset,:]
    elif faceIndex == 5:
        values = arr[:,-1-offset,:]

    return values.copy()

# --------------------------------------------------------------
#                  Knot Vector Manipulation Functions
# --------------------------------------------------------------

def blendKnotVectors(knotVectors, sym):
    """Take in a list of knot vectors and average them"""

    nVec = len(knotVectors)

    if sym: # Symmetrize each knot vector first
        for i in range(nVec):
            curKnotVec = knotVectors[i].copy()
            if np.mod(len(curKnotVec), 2) == 1: #its odd
                mid = (len(curKnotVec) -1)//2
                beg1 = curKnotVec[0:mid]
                beg2 = (1-curKnotVec[mid+1:])[::-1]
                # Average
                beg = 0.5*(beg1+beg2)
                curKnotVec[0:mid] = beg
                curKnotVec[mid+1:] = (1-beg)[::-1]
                curKnotVec[mid] = 0.5
            else: # its even
                mid = len(curKnotVec)//2
                beg1 = curKnotVec[0:mid]
                beg2 = (1-curKnotVec[mid:])[::-1]
                beg = 0.5*(beg1+beg2)
                curKnotVec[0:mid] = beg
                curKnotVec[mid:] = (1-beg)[::-1]

            knotVectors[i] = curKnotVec

    # Now average them all
    newKnotVec = np.zeros(len(knotVectors[0]))
    for i in range(nVec):
        newKnotVec += knotVectors[i]

    newKnotVec = newKnotVec / nVec
    return newKnotVec

class PointSelect(object):

    def __init__(self, psType, *args, **kwargs):

        """Initialize a control point selection class. There are several ways
        to initialize this class depending on the 'type' qualifier:

        Inputs:

        psType: string which inidicates the initialization type:

        'x': Define two corners (pt1=,pt2=) on a plane parallel to the
        x=0 plane

        'y': Define two corners (pt1=,pt2=) on a plane parallel to the
        y=0 plane

        'z': Define two corners (pt1=,pt2=) on a plane parallel to the
        z=0 plane

        'quad': Define FOUR corners (pt1=,pt2=,pt3=,pt4=) in a
        COUNTER-CLOCKWISE orientation

        'ijkBounds': Dictionary of int[3x2] defining upper and lower block indices to which we will apply the DVs.
        It should follow this format:
        ijkBounds = {volID:[[ilow, ihigh],
                            [jlow, jhigh],
                            [klow, khigh]]}
        volID is the same block identifier used in volList.
        If the user provides none, then we will apply the normal DVs to all FFD nodes.
        This is how you call PointSelect for ijkBounds:
        ps = PointSelect('ijkBounds',ijkBounds =  {volID:[[ilow, ihigh],
                                                          [jlow, jhigh],
                                                          [klow, khigh]]})
        Then to get the point indices you need to use:
        ps.getPointsijk(FFD)


        'list': Define the indices of a list that will be used to
        extract the points
        """

        if psType == 'x' or psType == 'y' or psType == 'z':
            assert 'pt1' in kwargs and 'pt2' in kwargs, 'Error:, two points \
must be specified with initialization type x,y, or z. Points are specified \
with kwargs pt1=[x1,y1,z1],pt2=[x2,y2,z2]'

        elif psType == 'quad':
            assert 'pt1' in kwargs and 'pt2' in kwargs and 'pt3' in kwargs \
                and 'pt4' in kwargs, 'Error:, four points \
must be specified with initialization type quad. Points are specified \
with kwargs pt1=[x1,y1,z1],pt2=[x2,y2,z2],pt3=[x3,y3,z3],pt4=[x4,y4,z4]'

        elif psType == 'ijkBounds':
            assert 'ijkBounds' in kwargs, 'Error:, ijkBounds selection method requires a dictonary with \
            the specific ijkBounds for each volume.'


        corners = np.zeros([4, 3])
        if psType in ['x', 'y', 'z', 'corners']:
            if psType == 'x':
                corners[0] = kwargs['pt1']

                corners[1][1] = kwargs['pt2'][1]
                corners[1][2] = kwargs['pt1'][2]

                corners[2][1] = kwargs['pt1'][1]
                corners[2][2] = kwargs['pt2'][2]

                corners[3] = kwargs['pt2']

                corners[:, 0] = 0.5*(kwargs['pt1'][0] + kwargs['pt2'][0])

            elif psType == 'y':
                corners[0] = kwargs['pt1']

                corners[1][0] = kwargs['pt2'][0]
                corners[1][2] = kwargs['pt1'][2]

                corners[2][0] = kwargs['pt1'][0]
                corners[2][2] = kwargs['pt2'][2]

                corners[3] = kwargs['pt2']

                corners[:, 1] = 0.5*(kwargs['pt1'][1] + kwargs['pt2'][1])

            elif psType == 'z':
                corners[0] = kwargs['pt1']

                corners[1][0] = kwargs['pt2'][0]
                corners[1][1] = kwargs['pt1'][1]

                corners[2][0] = kwargs['pt1'][0]
                corners[2][1] = kwargs['pt2'][1]

                corners[3] = kwargs['pt2']

                corners[:, 2] = 0.5*(kwargs['pt1'][2] + kwargs['pt2'][2])

            elif psType == 'quad':
                corners[0] = kwargs['pt1']
                corners[1] = kwargs['pt2']
                corners[2] = kwargs['pt4'] # Note the switch here from
                                           # CC orientation
                corners[3] = kwargs['pt3']

            X = corners

            self.box = pySpline.bilinearSurface(X)
            self.type = 'box'

        elif psType == 'list':
            self.box = None
            self.type = 'list'
            self.indices = np.array(args[0])

            # Check if the list is unique:
            if len(self.indices) != len(np.unique(self.indices)):
                raise ValueError('The indices provided to pointSelect are not unique.')

        elif psType == 'ijkBounds':

            self.ijkBounds = kwargs['ijkBounds'] # Store the ijk bounds dictionary
            self.type = 'ijkBounds'

    def getPoints(self, points):

        """Take in a list of points and return the ones that statify
        the point select class."""
        ptList = []
        indList = []
        if self.type == 'box':
            for i in range(len(points)):
                u0, v0, D = self.box.projectPoint(points[i])
                if u0 > 0 and u0 < 1 and v0 > 0 and v0 < 1: #Its Inside
                    ptList.append(points[i])
                    indList.append(i)

        elif self.type == 'list':
            for i in range(len(self.indices)):
                ptList.append(points[self.indices[i]])

            indList = self.indices.copy()

        elif self.type == 'ijkBounds':
            raise NameError('Use PointSelect.getPoints_ijk() to return indices of an object initialized with ijkBounds.')

        return ptList, indList

    def getPoints_ijk(self, DVGeo):

        """ Receives a DVGeo object (with an embedded FFD) and uses the ijk bounds specified in the initialization to extract
        the corresponding indices.

        You can only use this method if you initialized PointSelect with 'ijkBounds' option.

        DVGeo : DVGeo object"""

        # Initialize list to hold indices in the DVGeo ordering
        indList = []

        # Loop over every dictionary entry to get cooresponding indices
        for iVol in self.ijkBounds:

            # Get current bounds
            ilow = self.ijkBounds[iVol][0][0]
            ihigh = self.ijkBounds[iVol][0][1]
            jlow = self.ijkBounds[iVol][1][0]
            jhigh = self.ijkBounds[iVol][1][1]
            klow = self.ijkBounds[iVol][2][0]
            khigh = self.ijkBounds[iVol][2][1]

            # Retrieve current points
            indList.extend(DVGeo.FFD.topo.lIndex[iVol][ilow:ihigh,jlow:jhigh,klow:khigh].flatten())

        # Now get the corresponding coordinates
        ptList = [DVGeo.FFD.coef[ii] for ii in indList]

        return ptList, indList

class Topology(object):
    """
    The base topology class from which the BlockTopology,
    SurfaceTology and CuveTopology classes inherit from

    The topology object contains all the info required for the block
    topology (most complex) however, simpiler topologies are handled
    accordingly.

    Class Attributes:
        nVol : The number of volumes in the topology (may be 0)
        nFace: The number of unique faces on the topology (may be 0)
        nEdge: The number of uniuqe edges on the topology
        nNode: The number of unique nodes on the topology

        nEnt: The number of "entities" in the topology class. This may
        be curves, faces or volumes

        mNodeEnt: The number of NODES per entity. For curves it's 2, for
        surfaces 4 and for volumes 8.

        mEdgeEnt: The number of EDGES per entity. For curves it's 1,
        for surfaces, 4 and for volumes, 12

        mFaceEnt: The number of faces per entity. For curves its's 0,
        for surfaces, 1 and for volumes,6

        mVolEnt: The number of volumes per entity. For curves it's 0,
        for surfaces, 0 and for volumnes, 1

        nodeLink: The array of size nEnt x mNodesEnt which points
                   to the node for each entity
        edgeLink: The array of size nEnt x mEdgeEnt which points
                   to the edge for each edge of entity
        faceLink: The array of size nEnt x mFaceEnt which points to
                   the face of each face on an entity

        edgeDir:  The array of size nEnt x mEdgeEnt which detrmines
                   if the intrinsic direction of this edge is
                   opposite of the direction as recorded in the
                   edge list. edgeDir[entity#][#] = 1 means same direction;
                   -1 is opposite direction.

        faceDir:  The array of size nFace x 6 which determines the
                   intrinsic direction of this face. It is one of 0->7

        lIndex:   The local->global list of arrays for each volue
        gIndex:   The global->local list points for the entire topology
        edges:     The list of edge objects defining the topology
        simple    : A flag to determine of this is a "simple" topology
                   which means there are NO degernate Edges,
                   NO multiple edges sharing the same nodes and NO
                   edges which loop back and have the same nodes
                   MUST BE SIMPLE
    """

    def __init__(self):
        # Not sure what should go here...

        self.nVol = None; self.nFace = None; self.nEdge = None
        self.nNode = None; self.nExt = None
        self.mNodeEnt = None; self.mEdgeEnt = None; self.mFaceEnt = None
        self.nodeLink = None; self.edgeLink = None; self.faceLink = None
        self.edgeDir = None; self.faceDir = None
        self.lIndex = None; self.gIndex = None
        self.edges = None; self.simple = None
        self.topoType = None

    def _calcDGs(self, edges, edgeLink, edgeLinkSorted, edgeLinkInd):

        dgCounter = -1
        for i in range(self.nEdge):
            if edges[i][2] == -1: # Not set yet
                dgCounter += 1
                edges[i][2] = dgCounter
                self._addDGEdge(i, edges, edgeLink,
                                edgeLinkSorted, edgeLinkInd)

        self.nDG = dgCounter + 1

    def _addDGEdge(self, i, edges, edgeLink, edgeLinkSorted, edgeLinkInd):
        left  = edgeLinkSorted.searchsorted(i, side='left')
        right = edgeLinkSorted.searchsorted(i, side='right')
        res   = edgeLinkInd[slice(left, right)]

        for j in range(len(res)):
            ient = res[j]//self.mEdgeEnt
            iedge = np.mod(res[j], self.mEdgeEnt)

            pEdges = self._getParallelEdges(iedge)
            oppositeEdges = []
            for iii in range(len(pEdges)):
                oppositeEdges.append(
                    edgeLink[self.mEdgeEnt*ient + pEdges[iii]])

            for ii in range(len(pEdges)):
                if edges[oppositeEdges[ii]][2] == -1:
                    edges[oppositeEdges[ii]][2] = edges[i][2]
                    if not edges[oppositeEdges[ii]][0] == \
                            edges[oppositeEdges[ii]][1]:
                        self._addDGEdge(oppositeEdges[ii], edges,
                                        edgeLink, edgeLinkSorted,
                                        edgeLinkInd)

    def _getParallelEdges(self, iedge):
        """Return parallel edges for surfaces and volumes"""

        if self.topoType == 'surface':
            if iedge == 0: return [1]
            if iedge == 1: return [0]
            if iedge == 2: return [3]
            if iedge == 3: return [2]

        if self.topoType == 'volume':
            if iedge == 0:
                return [1, 4, 5]
            if iedge == 1:
                return [0, 4, 5]
            if iedge == 2:
                return [3, 6, 7]
            if iedge == 3:
                return [2, 6, 7]
            if iedge == 4:
                return [0, 1, 5]
            if iedge == 5:
                return [0, 1, 4]
            if iedge == 6:
                return [2, 3, 7]
            if iedge == 7:
                return [2, 3, 6]
            if iedge == 8:
                return [9, 10, 11]
            if iedge == 9:
                return [8, 10, 11]
            if iedge == 10:
                return [8, 9, 11]
            if iedge == 11:
                return [8, 9, 10]
        if self.topoType == 'curve':
            return None

    def printConnectivity(self):
        """Print the Edge Connectivity to the screen"""

        print('-----------------------------------------------\
-------------------------')
        print('%4d  %4d  %4d  %4d  %4d '%(
                self.nNode, self.nEdge, self.nFace, self.nVol, self.nDG))
        nList = self._getDGList()
        print('Design Group | Number')
        for i in range(self.nDG):
            print('%5d        | %5d       '%(i, nList[i]))

        # Always have edges!
        print('Edge Number    |   n0  |   n1  |  Cont | Degen | Intsct|\
   DG   |  N     |')
        for i in range(len(self.edges)):
            self.edges[i].writeInfo(i, sys.stdout)

        print('%9s Num |'% (self.topoType),)
        for i in range(self.mNodeEnt):
            print(' n%2d|'% (i),)
        for i in range(self.mEdgeEnt):
            print(' e%2d|'% (i), )
        print(' ')# Get New line

        for i in range(self.nEnt):
            print(' %5d        |'% (i),)
            for j in range(self.mNodeEnt):
                print('%4d|'%self.nodeLink[i][j],)

            for j in range(self.mEdgeEnt):
                print('%4d|'% (self.edgeLink[i][j]*self.edgeDir[i][j]),)
            print(' ')

        print('----------------------------------------------------\
--------------------')

        if self.topoType == 'volume':
            print('Vol Number | f0 | f1 | f2 | f3 | f4 | f5 |f0dir|\
f1dir|f2dir|f3dir|f4dir|f5dir|')
            for i in range(self.nVol):
                print(' %5d     |%4d|%4d|%4d|%4d|%4d|%4d|%5d|%5d|\
%5d|%5d|%5d|%5d|'\
                             %(i, self.faceLink[i][0], self.faceLink[i][1],
                               self.faceLink[i][2], self.faceLink[i][3],
                               self.faceLink[i][3], self.faceLink[i][5],
                               self.faceDir[i][0], self.faceDir[i][1],
                               self.faceDir[i][2], self.faceDir[i][3],
                               self.faceDir[i][4], self.faceDir[i][5]))

    def writeConnectivity(self, fileName):
        """Write the full edge connectivity to a file fileName"""

        f = open(fileName, 'w')
        f.write('%4d  %4d  %4d   %4d  %4d\n'%(
                self.nNode, self.nEdge, self.nFace, self.nVol, self.nDG))
        f.write('Design Group |  Number\n')
            # Write out the design groups and their number parameter
        nList = self._getDGList()
        for i in range(self.nDG):
            f.write('%5d        | %5d       \n'%(i, nList[i]))

        f.write('Edge Number    |   n0  |   n1  |  Cont | Degen |\
 Intsct|   DG   |  N     |\n')
        for i in range(len(self.edges)):
            self.edges[i].writeInfo(i, f)

        f.write('%9s Num |'%(self.topoType))
        for i in range(self.mNodeEnt):
            f.write(' n%2d|'%(i))
        for i in range(self.mEdgeEnt):
            f.write(' e%2d|'%(i))
        f.write('\n')

        for i in range(self.nEnt):
            f.write(' %5d        |'%(i))
            for j in range(self.mNodeEnt):
                f.write('%4d|'%self.nodeLink[i][j])

            for j in range(self.mEdgeEnt):
                f.write('%4d|'%(self.edgeLink[i][j]*self.edgeDir[i][j]))

            f.write('\n')

        if self.topoType == 'volume':

            f.write('Vol Number | f0 | f1 | f2 | f3 | f4 | f5 |\
f0dir|f1dir|f2dir|f3dir|f4dir|f5dir|\n')
            for i in range(self.nVol):
                f.write(' %5d     |%4d|%4d|%4d|%4d|%4d|%4d|%5d|\
%5d|%5d|%5d|%5d|%5d|\n'% (i, self.faceLink[i][0], self.faceLink[i][1],
                          self.faceLink[i][2], self.faceLink[i][3],
                          self.faceLink[i][4], self.faceLink[i][5],
                          self.faceDir[i][0], self.faceDir[i][1],
                          self.faceDir[i][2], self.faceDir[i][3],
                          self.faceDir[i][4], self.faceDir[i][5]))
            f.close()

    def readConnectivity(self, fileName):
        """Read the full edge connectivity from a file fileName"""
        # We must be able to populate the following:
        #nNode, nEdge, nFace,nVol,nodeLink,edgeLink,
        # faceLink,edgeDir,faceDir

        f = open(fileName, 'r')
        aux = f.readline().split()
        self.nNode = int(aux[0])
        self.nEdge = int(aux[1])
        self.nFace = int(aux[2])
        self.nVol  = int(aux[3])
        self.nDG   = int(aux[4])
        self.edges = []

        if self.topoType == 'volume':
            self.nEnt = self.nVol
        elif self.topoType == 'surface':
            self.nEnt = self.nFace
        elif self.topoType == 'curve':
            self.nEnt = self.nEdge

        f.readline() # This is the header line so ignore

        nList = np.zeros(self.nDG, 'intc')
        for i in range(self.nDG):
            aux = f.readline().split('|')
            nList[i] = int(aux[1])

        f.readline() # Second Header line

        for i in range(self.nEdge):
            aux = f.readline().split('|')
            self.edges.append(Edge(int(aux[1]), int(aux[2]), int(aux[3]),
                                   int(aux[4]), int(aux[5]), int(aux[6]),
                                   int(aux[7])))

        f.readline() # This is the third header line so ignore

        self.edgeLink = np.zeros((self.nEnt, self.mEdgeEnt), 'intc')
        self.nodeLink = np.zeros((self.nEnt, self.mNodeEnt), 'intc')
        self.edgeDir  = np.zeros((self.nEnt, self.mEdgeEnt), 'intc')

        for i in range(self.nEnt):
            aux = f.readline().split('|')
            for j in range(self.mNodeEnt):
                self.nodeLink[i][j] = int(aux[j+1])
            for j in range(self.mEdgeEnt):
                self.edgeDir[i][j]  = np.sign(int(aux[j+1+self.mNodeEnt]))
                self.edgeLink[i][j] = int(
                    aux[j+1+self.mNodeEnt])*self.edgeDir[i][j]

        if self.topoType == 'volume':
            f.readline() # This the fourth header line so ignore

            self.faceLink = np.zeros((self.nVol, 6), 'intc')
            self.faceDir  = np.zeros((self.nVol, 6), 'intc')
            for ivol in range(self.nVol):
                aux = f.readline().split('|')
                self.faceLink[ivol] = [int(aux[i]) for i in range(1, 7)]
                self.faceDir[ivol]  = [int(aux[i]) for i in range(7, 13)]

        # Set the nList to the edges
        for iedge in range(self.nEdge):
            self.edges[iedge].N = nList[self.edges[iedge].dg]

        return

    def _getDGList(self):
        """After calcGlobalNumbering is called with the size
        parameters, we can now produce a list of length ndg with the
        each entry coorsponing to the number N associated with that DG"""

        # This can be run in linear time...just loop over each edge
        # and add to dg list
        nList = np.zeros(self.nDG, 'intc')
        for iedge in range(self.nEdge):
            nList[self.edges[iedge].dg] = self.edges[iedge].N

        return nList

class CurveTopology(Topology):
    """
    See topology class for more information
    """
    def __init__(self, coords=None, file=None):
        """Initialize the class with data required to compute the topology"""
        Topology.__init__(self)
        self.mNodeEnt = 2
        self.mEdgeEnt = 1
        self.mfaceEnt = 0
        self.mVolEnt  = 0
        self.nVol = 0
        self.nFace = 0
        self.topoType = 'curve'
        self.gIndex = None
        self.lIndex = None
        self.nGlobal = None
        if file is not None:
            self.readConnectivity(file)
            return

        self.edges = None
        self.simple = True

        # Must have curves
        # Get the end points of each curve
        self.nEdge = len(coords)
        coords = coords.reshape((self.nEdge*2, 3))
        nodeList, self.nodeLink = pointReduce(coords)
        self.nodeLink = self.nodeLink.reshape((self.nEdge, 2))
        self.nNode = len(nodeList)
        self.edges = []
        self.edgeLink = np.zeros((self.nEdge, 1), 'intc')
        for iedge in range(self.nEdge):
            self.edgeLink[iedge][0] = iedge

        self.edgeDir  = np.zeros((self.nEdge, 1), 'intc')

        for iedge in range(self.nEdge):
            n1 = self.nodeLink[iedge][0]
            n2 = self.nodeLink[iedge][1]
            if n1 < n2:
                self.edges.append(Edge(n1, n2, 0, 0, 0, iedge, 2))
                self.edgeDir[iedge][0] = 1
            else:
                self.edges.append(Edge(n2, n1, 0, 0, 0, iedge, 2))
                self.edgeDir[iedge][0] = -1

        self.nDG = self.nEdge
        self.nEnt = self.nEdge
        return

    def calcGlobalNumbering(self, sizes, curveList=None):
        """Internal function to calculate the global/local numbering
        for each curve"""
        for i in range(len(sizes)):
            self.edges[self.edgeLink[i][0]].N = sizes[i]

        if curveList is None:
            curveList = np.arange(self.nEdge)

        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        lIndex = []

        assert len(sizes) == len(curveList), 'Error: The list of sizes and \
the list of surfaces must be the same length'

        # Assign unique numbers to the corners -> Corners are indexed
        # sequentially
        nodeIndex = np.arange(self.nNode)
        counter = len(nodeIndex)
        edgeIndex = [ [] for i in range(len(self.edges))]

        # Assign unique numbers to the edges
        for ii in range(len(curveList)):
            curSize = [sizes[ii]]
            icurve = curveList[ii]
            for iedge in range(1):
                edge = self.edgeLink[ii][iedge]

                if edgeIndex[edge] == []:# Not added yet
                    for jj in range(curSize[iedge]-2):
                        edgeIndex[edge].append(counter)
                        counter += 1

        gIndex = [ [] for i in range(counter)] # We must add [] for
                                                 # each global node

        for ii in range(len(curveList)):
            icurve = curveList[ii]
            N = sizes[ii]
            lIndex.append(-1*np.ones(N, 'intc'))

            for i in range(N):
                _type, node = indexPosition1D(i, N)

                if _type == 1: # Node
                    curNode = self.nodeLink[ii][node]
                    lIndex[ii][i] = nodeIndex[curNode]
                    gIndex[nodeIndex[curNode]].append([icurve, i])
                else:
                    if self.edgeDir[ii][0] == -1:
                        curIndex = edgeIndex[self.edgeLink[ii][0]][N-i-2]
                    else:
                        curIndex = edgeIndex[self.edgeLink[ii][0]][i-1]

                    lIndex[ii][i] = curIndex
                    gIndex[curIndex].append([icurve, i])

        self.nGlobal = len(gIndex)
        self.gIndex = gIndex
        self.lIndex = lIndex

        return

class SurfaceTopology(Topology):
    """
    See topology class for more information
    """
    def __init__(self, coords=None, faceCon=None, fileName=None, nodeTol=1e-4,
                 edgeTol=1e-4):
        """Initialize the class with data required to compute the topology"""
        Topology.__init__(self)
        self.mNodeEnt = 4
        self.mEdgeEnt = 4
        self.mfaceEnt = 1
        self.mVolEnt  = 0
        self.nVol = 0
        self.topoType = 'surface'
        self.gIndex = None
        self.lIndex = None
        self.nGlobal = None
        if fileName is not None:
            self.readConnectivity(fileName)
            return

        self.edges = None
        self.faceIndex = None
        self.simple = False

        if faceCon is not None:
            faceCon = np.array(faceCon)
            midpoints = None
            self.nFace = len(faceCon)
            self.nEnt = self.nFace
            self.simple = True
            # Check to make sure nodes are sequential
            self.nNode = len(unique(faceCon.flatten()))
            if self.nNode != max(faceCon.flatten())+1:
                # We don't have sequential nodes
                print("Error: Nodes are not sequential")
                sys.exit(1)

            edges = []
            edgeHash = []
            for iface in range(self.nFace):
                #             n1                ,n2               ,dg,n,degen
                edges.append([faceCon[iface][0], faceCon[iface][1], -1, 0, 0])
                edges.append([faceCon[iface][2], faceCon[iface][3], -1, 0, 0])
                edges.append([faceCon[iface][0], faceCon[iface][2], -1, 0, 0])
                edges.append([faceCon[iface][1], faceCon[iface][3], -1, 0, 0])

            edgeDir = np.ones(len(edges), 'intc')
            for iedge in range(self.nFace*4):
                if edges[iedge][0] > edges[iedge][1]:
                    temp = edges[iedge][0]
                    edges[iedge][0] = edges[iedge][1]
                    edges[iedge][1] = temp
                    edgeDir[iedge] = -1

                edgeHash.append(
                    edges[iedge][0]*4*self.nFace + edges[iedge][1])

            edges, edgeLink = uniqueIndex(edges, edgeHash)

            self.nEdge = len(edges)
            self.edgeLink = np.array(edgeLink).reshape((self.nFace, 4))
            self.nodeLink = np.array(faceCon)
            self.edgeDir  = np.array(edgeDir).reshape((self.nFace, 4))

            edgeLinkSorted = np.sort(edgeLink)
            edgeLinkInd    = np.argsort(edgeLink)

        elif coords is not None:
            self.nFace = len(coords)
            self.nEnt  = self.nFace
            # We can use the pointReduce algorithim on the nodes
            nodeList, nodeLink = pointReduce(
                coords[:, 0:4, :].reshape((self.nFace*4, 3)), nodeTol)
            nodeLink = nodeLink.reshape((self.nFace, 4))

            # Next Calculate the EDGE connectivity. -- This is Still
            # Brute Force

            edges = []
            midpoints = []
            edgeLink = -1*np.ones(self.nFace*4, 'intc')
            edgeDir  = np.zeros((self.nFace, 4), 'intc')

            for iface in range(self.nFace):
                for iedge in range(4):
                    n1, n2 = nodesFromEdge(iedge)
                    n1 = nodeLink[iface][n1]
                    n2 = nodeLink[iface][n2]
                    midpoint = coords[iface][iedge + 4]
                    if len(edges) == 0:
                        edges.append([n1, n2, -1, 0, 0])
                        midpoints.append(midpoint)
                        edgeLink[4*iface + iedge] = 0
                        edgeDir [iface][iedge] = 1
                    else:
                        foundIt = False
                        for i in range(len(edges)):
                            if [n1, n2] == edges[i][0:2] and n1 != n2:
                                if eDist(midpoint, midpoints[i]) < edgeTol:
                                    edgeLink[4*iface + iedge] = i
                                    edgeDir [iface][iedge] = 1
                                    foundIt = True

                            elif [n2, n1] == edges[i][0:2] and n1 != n2:
                                if eDist(midpoint, midpoints[i]) < edgeTol:
                                    edgeLink[4*iface + iedge] = i
                                    edgeDir[iface][iedge] = -1
                                    foundIt = True
                        # end for

                        # We went all the way though the list so add
                        # it at end and return index
                        if not foundIt:
                            edges.append([n1, n2, -1, 0, 0])
                            midpoints.append(midpoint)
                            edgeLink[4*iface + iedge] = i+1
                            edgeDir [iface][iedge] = 1
            # end for (iFace)

            self.nEdge = len(edges)
            self.edgeLink = np.array(edgeLink).reshape((self.nFace, 4))
            self.nodeLink = np.array(nodeLink)
            self.nNode = len(unique(self.nodeLink.flatten()))
            self.edgeDir = edgeDir

            edgeLinkSorted = np.sort(edgeLink.flatten())
            edgeLinkInd    = np.argsort(edgeLink.flatten())
        # end if

        # Next Calculate the Design Group Information
        self._calcDGs(edges, edgeLink, edgeLinkSorted, edgeLinkInd)

        # Set the edge ojects
        self.edges = []
        for i in range(self.nEdge): # Create the edge objects
            if midpoints: # If they exist
                if edges[i][0] == edges[i][1] and \
                        eDist(midpoints[i], nodeList[edges[i][0]]) < nodeTol:
                    self.edges.append(Edge(edges[i][0], edges[i][1],
                                           0, 1, 0, edges[i][2], edges[i][3]))
                else:
                    self.edges.append(Edge(edges[i][0], edges[i][1],
                                           0, 0, 0, edges[i][2], edges[i][3]))
            else:
                self.edges.append(Edge(edges[i][0], edges[i][1],
                                       0, 0, 0, edges[i][2], edges[i][3]))

    def calcGlobalNumberingDummy(self, sizes, surfaceList=None):
        """Internal function to calculate the global/local numbering
        for each surface"""
        for i in range(len(sizes)):
            self.edges[self.edgeLink[i][0]].N = sizes[i][0]
            self.edges[self.edgeLink[i][1]].N = sizes[i][0]
            self.edges[self.edgeLink[i][2]].N = sizes[i][1]
            self.edges[self.edgeLink[i][3]].N = sizes[i][1]

        if surfaceList is None:
            surfaceList = np.arange(0, self.nFace)

        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        assert len(sizes) == len(surfaceList), 'Error: The list of sizes and \
the list of surfaces must be the same length'

        # Assign unique numbers to the corners -> Corners are indexed
        # sequentially
        nodeIndex = np.arange(self.nNode)
        counter = len(nodeIndex)
        edgeIndex = [ [] for i in range(len(self.edges))]

        # Assign unique numbers to the edges
        for ii in range(len(surfaceList)):
            iSurf = surfaceList[ii]
            curSize = [sizes[iSurf][0], sizes[iSurf][0],
                       sizes[iSurf][1], sizes[iSurf][1]]

            for iedge in range(4):
                edge = self.edgeLink[ii][iedge]

                if edgeIndex[edge] == []:# Not added yet
                    if self.edges[edge].degen == 1:
                        # Get the counter value for this "node"
                        index = nodeIndex[self.edges[edge].n1]
                        for jj in range(curSize[iedge]-2):
                            edgeIndex[edge].append(index)
                    else:
                        for jj in range(curSize[iedge]-2):
                            edgeIndex[edge].append(counter)
                            counter += 1

        lIndex = []
        # Now actually fill everything up
        for ii in range(len(surfaceList)):
            isurf = surfaceList[ii]
            N = sizes[iSurf][0]
            M = sizes[iSurf][1]
            lIndex.append(-1*np.ones((N, M), 'intc'))

        self.lIndex = lIndex

    def calcGlobalNumbering(self, sizes, surfaceList=None):
        """Internal function to calculate the global/local numbering
        for each surface"""
        for i in range(len(sizes)):
            self.edges[self.edgeLink[i][0]].N = sizes[i][0]
            self.edges[self.edgeLink[i][1]].N = sizes[i][0]
            self.edges[self.edgeLink[i][2]].N = sizes[i][1]
            self.edges[self.edgeLink[i][3]].N = sizes[i][1]

        if surfaceList is None:
            surfaceList = np.arange(0, self.nFace)

        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        gIndex = []
        lIndex = []

        assert len(sizes) == len(surfaceList), 'Error: The list of sizes and \
the list of surfaces must be the same length'

        # Assign unique numbers to the corners -> Corners are indexed
        # sequentially
        nodeIndex = np.arange(self.nNode)
        counter = len(nodeIndex)
        edgeIndex = [ [] for i in range(len(self.edges))]

        # Assign unique numbers to the edges
        for ii in range(len(surfaceList)):
            curSize = [sizes[ii][0], sizes[ii][0], sizes[ii][1], sizes[ii][1]]
            isurf = surfaceList[ii]
            for iedge in range(4):
                edge = self.edgeLink[ii][iedge]

                if edgeIndex[edge] == []:# Not added yet
                    if self.edges[edge].degen == 1:
                        # Get the counter value for this "node"
                        index = nodeIndex[self.edges[edge].n1]
                        for jj in range(curSize[iedge]-2):
                            edgeIndex[edge].append(index)
                    else:
                        for jj in range(curSize[iedge]-2):
                            edgeIndex[edge].append(counter)
                            counter += 1

        gIndex = [ [] for i in range(counter)] # We must add [] for
                                                 # each global node
        lIndex = []
        # Now actually fill everything up
        for ii in range(len(surfaceList)):
            isurf = surfaceList[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            lIndex.append(-1*np.ones((N, M), 'intc'))

            for i in range(N):
                for j in range(M):

                    _type, edge, node, index = indexPosition2D(i, j, N, M)

                    if _type == 0:           # Interior
                        lIndex[ii][i, j] = counter
                        gIndex.append([[isurf, i, j]])
                        counter += 1
                    elif _type == 1:         # Edge
                        if edge in [0, 1]:
                            # Its a reverse dir
                            if self.edgeDir[ii][edge] == -1:
                                curIndex = edgeIndex[
                                    self.edgeLink[ii][edge]][N-i-2]
                            else:
                                curIndex = edgeIndex[
                                    self.edgeLink[ii][edge]][i-1]
                        else: # edge in [2, 3]
                            # Its a reverse dir
                            if self.edgeDir[ii][edge] == -1:
                                curIndex = edgeIndex[
                                    self.edgeLink[ii][edge]][M-j-2]
                            else:
                                curIndex = edgeIndex[
                                    self.edgeLink[ii][edge]][j-1]
                        lIndex[ii][i, j] = curIndex
                        gIndex[curIndex].append([isurf, i, j])

                    else:                  # Node
                        curNode = self.nodeLink[ii][node]
                        lIndex[ii][i, j] = nodeIndex[curNode]
                        gIndex[nodeIndex[curNode]].append([isurf, i, j])
        # end for (surface loop)

        # Reorder the indices with a greedy scheme
        newIndices = np.zeros(len(gIndex), 'intc')
        newIndices[:] = -1
        newGIndex = [[] for i in range(len(gIndex))]
        counter = 0

        # Re-order the lIndex
        for ii in range(len(surfaceList)):
            isurf = surfaceList[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            for i in range(N):
                for j in range(M):
                    if newIndices[lIndex[ii][i, j]] == -1:
                        newIndices[lIndex[ii][i, j]] = counter
                        lIndex[ii][i, j] = counter
                        counter += 1
                    else:
                        lIndex[ii][i, j] = newIndices[lIndex[ii][i, j]]

        # Re-order the gIndex
        for ii in range(len(gIndex)):
            isurf = gIndex[ii][0][0]
            i     = gIndex[ii][0][1]
            j     = gIndex[ii][0][2]
            pt = lIndex[isurf][i, j]
            newGIndex[pt] = gIndex[ii]

        self.nGlobal = len(gIndex)
        self.gIndex = newGIndex
        self.lIndex = lIndex

        return

    def getSurfaceFromEdge(self,  edge):
        """Determine the surfaces and their edgeLink index that
        points to edge iedge"""
        # Its not efficient but it works - scales with Nface not constant
        surfaces = []
        for isurf in range(self.nFace):
            for iedge in range(4):
                if self.edgeLink[isurf][iedge] == edge:
                    surfaces.append([isurf, iedge])

        return surfaces

    def makeSizesConsistent(self, sizes, order):
        """
        Take a given list of [Nu x Nv] for each surface and return
        the sizes list such that all sizes are consistent

        prescedence is given according to the order list: 0 is highest
        prescedence,  1 is next highest ect.
        """

        # First determine how many "order" loops we have
        nloops = max(order)+1
        edgeNumber = -1*np.ones(self.nDG, 'intc')
        for iedge in range(self.nEdge):
            self.edges[iedge].N = -1

        for iloop in range(nloops):
            for iface in range(self.nFace):
                if order[iface] == iloop: # Set this edge
                    for iedge in range(4):
                        dg = self.edges[self.edgeLink[iface][iedge]].dg
                        if edgeNumber[dg] == -1:
                            if iedge in [0, 1]:
                                edgeNumber[dg] = sizes[iface][0]
                            else:
                                edgeNumber[dg] = sizes[iface][1]

        # Now re-populate the sizes:
        for iface in range(self.nFace):
            for i in [0, 1]:
                dg = self.edges[self.edgeLink[iface][i*2]].dg
                sizes[iface][i] = edgeNumber[dg]

        # And return the number of elements on each actual edge
        nEdge = []
        for iedge in range(self.nEdge):
            self.edges[iedge].N = edgeNumber[self.edges[iedge].dg]
            nEdge.append(edgeNumber[self.edges[iedge].dg])

        return sizes, nEdge

class BlockTopology(Topology):
    """
    See Topology base class for more information
    """

    def __init__(self, coords=None, nodeTol=1e-4, edgeTol=1e-4, fileName=None):
        """Initialize the class with data required to compute the topology"""

        Topology.__init__(self)
        self.mNodeEnt = 8
        self.mEdgeEnt = 12
        self.mFaceEnt = 6
        self.mVolEnt  = 1
        self.topoType = 'volume'
        self.gIndex = None
        self.lIndex = None
        self.nGlobal = None
        if fileName is not None:
            self.readConnectivity(fileName)
            return

        coords = np.atleast_2d(coords)
        nVol = len(coords)

        if coords.shape[1] == 8: # Just the corners are given --- Just
                                 # put in np.zeros for the edge and face
                                 # mid points
            temp = np.zeros((nVol, (8 + 12 + 6), 3))
            temp[:, 0:8, :] = coords
            coords = temp.copy()

        # ----------------------------------------------------------
        #                     Unique Nodes
        # ----------------------------------------------------------

        # Do the pointReduce Agorithm on the corners
        un, nodeLink = pointReduce(coords[:, 0:8, :].reshape((nVol*8, 3)),
                                   nodeTol=nodeTol)
        nodeLink = nodeLink.reshape((nVol, 8))

        # ----------------------------------------------------------
        #                     Unique Edges
        # ----------------------------------------------------------
         # Now determine the unique edges:
        edgeObjs = []
        origEdges = []
        for ivol in range(nVol):
            for iedge in range(12):
                # Node number on volume
                n1, n2 = nodesFromEdge(iedge)

                # Actual Global Node Number
                n1 = nodeLink[ivol][n1]
                n2 = nodeLink[ivol][n2]

                # Midpoint
                midpoint = coords[ivol][iedge + 8]

                # Sorted Nodes:
                ns = sorted([n1, n2])

                # Append the new edgeCmp Object
                edgeObjs.append(EdgeCmpObject(
                        ns[0], ns[1], n1, n2, midpoint, edgeTol))

                # Keep track of original edge orientation---needed for
                # face direction
                origEdges.append([n1, n2])

        # Generate unique set of edges
        uniqueEdgeObjs,  edgeLink = uniqueIndex(edgeObjs)

        edgeDir = []
        for i in range(len(edgeObjs)): # This is nVol * 12
            edgeDir.append(edgeOrientation(
                    origEdges[i], uniqueEdgeObjs[edgeLink[i]].nodes))
        # ----------------------------------------------------------
        #                     Unique Faces
        # ----------------------------------------------------------

        faceObjs = []
        origFaces = []
        for ivol in range(nVol):
            for iface in range(6):
                # Node number on volume
                n1, n2, n3, n4 = nodesFromFace(iface)

                # Actual Global Node Number
                n1 = nodeLink[ivol][n1]
                n2 = nodeLink[ivol][n2]
                n3 = nodeLink[ivol][n3]
                n4 = nodeLink[ivol][n4]

                # Midpoint --> May be [0, 0, 0] -> This is OK
                midpoint = coords[ivol][iface + 8 + 12]

                # Sort the nodes before they go into the faceObject
                ns = sorted([n1, n2, n3, n4])
                faceObjs.append(FaceCmpObject(ns[0], ns[1], ns[2], ns[3],
                                                 n1, n2, n3, n4,
                                                 midpoint, 1e-4))
                # Keep track of original face orientation---needed for
                # face direction
                origFaces.append([n1, n2, n3, n4])

        # Generate unique set of faces
        uniqueFaceObjs, faceLink = uniqueIndex(faceObjs)

        faceDir = []
        faceDirRev = []
        for i in range(len(faceObjs)): # This is nVol * 12
            faceDir.append(faceOrientation(
                    uniqueFaceObjs[faceLink[i]].nodes, origFaces[i]))
            faceDirRev.append(faceOrientation(
                    origFaces[i], uniqueFaceObjs[faceLink[i]].nodes))

        # --------- Set the Requried Data for this class ------------
        self.nNode = len(un)
        self.nEdge = len(uniqueEdgeObjs)
        self.nFace = len(uniqueFaceObjs)
        self.nVol  = len(coords)
        self.nEnt  = self.nVol

        self.nodeLink = nodeLink
        self.edgeLink = np.array(edgeLink).reshape((nVol, 12))
        self.faceLink = np.array(faceLink).reshape((nVol, 6))

        self.edgeDir  = np.array(edgeDir).reshape((nVol, 12))
        self.faceDir  = np.array(faceDir).reshape((nVol, 6))
        self.faceDirRev  = np.array(faceDirRev).reshape((nVol, 6))

        # Next Calculate the Design Group Information
        edgeLinkSorted = np.sort(edgeLink.flatten())
        edgeLinkInd    = np.argsort(edgeLink.flatten())

        ue = []
        for i in range(len(uniqueEdgeObjs)):
            ue.append([uniqueEdgeObjs[i].nodes[0],
                       uniqueEdgeObjs[i].nodes[1], -1, 0, 0])

        self._calcDGs(ue, edgeLink, edgeLinkSorted, edgeLinkInd)

        # Set the edge ojects
        self.edges = []
        for i in range(self.nEdge): # Create the edge objects
            self.edges.append(Edge(
                    ue[i][0], ue[i][1], 0, 0, 0, ue[i][2], ue[i][3]))

        return

    def calcGlobalNumbering(self, sizes=None, volumeList=None,
                            greedyReorder=False,gIndex=True):
        """Internal function to calculate the global/local numbering
        for each volume"""

        if sizes is not None:
            for i in range(len(sizes)):
                self.edges[self.edgeLink[i][0]].N = sizes[i][0]
                self.edges[self.edgeLink[i][1]].N = sizes[i][0]
                self.edges[self.edgeLink[i][4]].N = sizes[i][0]
                self.edges[self.edgeLink[i][5]].N = sizes[i][0]

                self.edges[self.edgeLink[i][2]].N = sizes[i][1]
                self.edges[self.edgeLink[i][3]].N = sizes[i][1]
                self.edges[self.edgeLink[i][6]].N = sizes[i][1]
                self.edges[self.edgeLink[i][7]].N = sizes[i][1]

                self.edges[self.edgeLink[i][8]].N  = sizes[i][2]
                self.edges[self.edgeLink[i][9]].N  = sizes[i][2]
                self.edges[self.edgeLink[i][10]].N = sizes[i][2]
                self.edges[self.edgeLink[i][11]].N = sizes[i][2]
        else: # N is already set in the edge objects, use them
            sizes = np.zeros((self.nVol, 3), 'intc')
            for ivol in range(self.nVol):
                sizes[ivol][0] = self.edges[self.edgeLink[ivol][0]].N
                sizes[ivol][1] = self.edges[self.edgeLink[ivol][2]].N
                sizes[ivol][2] = self.edges[self.edgeLink[ivol][8]].N

        if volumeList is None:
            volumeList = np.arange(0, self.nVol)

        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        gIndex = []
        lIndex = []

        assert len(sizes) == len(volumeList), 'Error: The list of sizes and \
the list of volumes must be the same length'

        # Assign unique numbers to the corners -> Corners are indexed
        # sequentially
        nodeIndex = np.arange(self.nNode)
        counter = len(nodeIndex)

        edgeIndex = [ np.empty((0), 'intc') for i in range(self.nEdge)]
        faceIndex = [ np.empty((0, 0), 'intc') for i in range(self.nFace)]
        # Assign unique numbers to the edges

        for ii in range(len(volumeList)):
            curSizeE = [sizes[ii][0], sizes[ii][0], sizes[ii][1],
                          sizes[ii][1], sizes[ii][0], sizes[ii][0],
                          sizes[ii][1], sizes[ii][1], sizes[ii][2],
                          sizes[ii][2], sizes[ii][2], sizes[ii][2]]

            curSizeF = [[sizes[ii][0], sizes[ii][1]],
                          [sizes[ii][0], sizes[ii][1]],
                          [sizes[ii][1], sizes[ii][2]],
                          [sizes[ii][1], sizes[ii][2]],
                          [sizes[ii][0], sizes[ii][2]],
                          [sizes[ii][0], sizes[ii][2]]]

            ivol = volumeList[ii]
            for iedge in range(12):
                edge = self.edgeLink[ii][iedge]
                if edgeIndex[edge].shape == (0, ):# Not added yet
                    edgeIndex[edge] = np.resize(
                        edgeIndex[edge], curSizeE[iedge]-2)
                    if self.edges[edge].degen == 1:
                        # Get the counter value for this "node"
                        index = nodeIndex[self.edges[edge].n1]
                        for jj in range(curSizeE[iedge]-2):
                            edgeIndex[edge][jj] = index
                    else:
                        for jj in range(curSizeE[iedge]-2):
                            edgeIndex[edge][jj] = counter
                            counter += 1

            for iface in range(6):
                face = self.faceLink[ii][iface]
                if faceIndex[face].shape == (0, 0):
                    faceIndex[face] = np.resize(faceIndex[face],
                                               [curSizeF[iface][0]-2,
                                                curSizeF[iface][1]-2])
                    for iii in range(curSizeF[iface][0]-2):
                        for jjj in range(curSizeF[iface][1]-2):
                            faceIndex[face][iii, jjj] = counter
                            counter += 1

        # end for (volume list)

        gIndex = [ [] for i in range(counter)] # We must add [] for
                                                 # each global node
        lIndex = []

        def addNode(i, j, k, N, M, L):
            _type, number, index1, index2 = indexPosition3D(i, j, k, N, M, L)

            if _type == 1:         # Face

                if number in [0, 1]:
                    icount = i;imax = N
                    jcount = j;jmax = M
                elif number in [2, 3]:
                    icount = j;imax = M
                    jcount = k;jmax = L
                elif number in [4, 5]:
                    icount = i;imax = N
                    jcount = k;jmax = L

                if self.faceDir[ii][number] == 0:
                    curIndex = faceIndex[
                        self.faceLink[ii][number]][
                        icount-1, jcount-1]
                elif self.faceDir[ii][number] == 1:
                    curIndex = faceIndex[
                        self.faceLink[ii][number]][
                        imax-icount-2, jcount-1]
                elif self.faceDir[ii][number] == 2:
                    curIndex = faceIndex[
                        self.faceLink[ii][number]][
                        icount-1, jmax-jcount-2]
                elif self.faceDir[ii][number] == 3:
                    curIndex = faceIndex[
                        self.faceLink[ii][number]][
                        imax-icount-2, jmax-jcount-2]
                elif self.faceDir[ii][number] == 4:
                    curIndex = faceIndex[
                        self.faceLink[ii][number]][
                        jcount-1, icount-1]
                elif self.faceDir[ii][number] == 5:
                    curIndex = faceIndex[
                        self.faceLink[ii][number]][
                        jmax-jcount-2, icount-1]
                elif self.faceDir[ii][number] == 6:
                    curIndex = faceIndex[
                        self.faceLink[ii][number]][
                        jcount-1, imax-icount-2]
                elif self.faceDir[ii][number] == 7:
                    curIndex = faceIndex[
                        self.faceLink[ii][number]][
                        jmax-jcount-2, imax-icount-2]

                lIndex[ii][i, j, k] = curIndex
                gIndex[curIndex].append([ivol, i, j, k])

            elif _type == 2:         # Edge

                if number in [0, 1, 4, 5]:
                    if self.edgeDir[ii][number] == -1: # Its a reverse dir
                        curIndex = \
                            edgeIndex[self.edgeLink[ii][number]][N-i-2]
                    else:
                        curIndex = \
                            edgeIndex[self.edgeLink[ii][number]][i-1]

                elif number in [2, 3, 6, 7]:
                    if self.edgeDir[ii][number] == -1: # Its a reverse dir
                        curIndex = \
                            edgeIndex[self.edgeLink[ii][number]][M-j-2]
                    else:
                        curIndex = \
                            edgeIndex[self.edgeLink[ii][number]][j-1]

                elif number in [8, 9, 10, 11]:
                    if self.edgeDir[ii][number] == -1: # Its a reverse dir
                        curIndex = \
                            edgeIndex[self.edgeLink[ii][number]][L-k-2]
                    else:
                        curIndex = \
                            edgeIndex[self.edgeLink[ii][number]][k-1]

                lIndex[ii][i, j, k] = curIndex
                gIndex[curIndex].append([ivol, i, j, k])

            elif _type == 3:                  # Node
                curNode = self.nodeLink[ii][number]
                lIndex[ii][i, j, k] = nodeIndex[curNode]
                gIndex[nodeIndex[curNode]].append([ivol, i, j, k])

        # end for (volume loop)

        # Now actually fill everything up
        for ii in range(len(volumeList)):
            ivol = volumeList[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            L = sizes[ii][2]
            lIndex.append(-1*np.ones((N, M, L), 'intc'))

            # DO the 6 planes
            for k in [0, L-1]:
                for i in range(N):
                    for j in range(M):
                        addNode(i, j, k, N, M, L)
            for j in [0, M-1]:
                for i in range(N):
                    for k in range(1, L-1):
                        addNode(i, j, k, N, M, L)

            for i in [0, N-1]:
                for j in range(1, M-1):
                    for k in range(1, L-1):
                        addNode(i, j, k, N, M, L)

        # Add the remainder
        for ii in range(len(volumeList)):
            ivol = volumeList[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            L = sizes[ii][2]

            NN = sizes[ii][0]-2
            MM = sizes[ii][1]-2
            LL = sizes[ii][2]-2

            toAdd = NN*MM*LL

            lIndex[ii][1:N-1,1:M-1,1:L-1] = \
                np.arange(counter,counter+toAdd).reshape((NN,MM,LL))

            counter = counter + toAdd
            A = np.zeros((toAdd,1,4),'intc')
            A[:,0,0] = ivol
            A[:,0,1:] = np.mgrid[1:N-1,1:M-1,1:L-1].transpose(
                (1,2,3,0)).reshape((toAdd,3))
            gIndex.extend(A)

        # Set the following as atributes
        self.nGlobal = len(gIndex)
        self.gIndex = gIndex
        self.lIndex = lIndex

        if greedyReorder:

            # Reorder the indices with a greedy scheme
            newIndices = np.zeros(len(gIndex), 'intc')
            newIndices[:] = -1
            newGIndex = [[] for i in range(len(gIndex))]
            counter = 0

            # Re-order the lIndex
            for ii in range(len(volumeList)):
                ivol = volumeList[ii]
                N = sizes[ii][0]
                M = sizes[ii][1]
                L = sizes[ii][2]
                for i in range(N):
                    for j in range(M):
                        for k in range(L):
                            if newIndices[lIndex[ii][i, j, k]] == -1:
                                newIndices[lIndex[ii][i, j, k]] = counter
                                lIndex[ii][i, j, k] = counter
                                counter += 1
                            else:
                                lIndex[ii][i, j, k] = \
                                    newIndices[lIndex[ii][i, j, k]]

            # Re-order the gIndex
            for ii in range(len(gIndex)):
                ivol  = gIndex[ii][0][0]
                i     = gIndex[ii][0][1]
                j     = gIndex[ii][0][2]
                k     = gIndex[ii][0][3]
                pt = lIndex[ivol][i, j, k]
                newGIndex[pt] = gIndex[ii]

            self.gIndex = newGIndex
            self.lIndex = lIndex
        # end if (greedy reorder)

        return

    def calcGlobalNumbering2(self, sizes=None, gIndex=True, volumeList=None,
                           greedyReorder=False):
        """Internal function to calculate the global/local numbering
        for each volume"""
        if sizes is not None:
            for i in range(len(sizes)):
                self.edges[self.edgeLink[i][0]].N = sizes[i][0]
                self.edges[self.edgeLink[i][1]].N = sizes[i][0]
                self.edges[self.edgeLink[i][4]].N = sizes[i][0]
                self.edges[self.edgeLink[i][5]].N = sizes[i][0]

                self.edges[self.edgeLink[i][2]].N = sizes[i][1]
                self.edges[self.edgeLink[i][3]].N = sizes[i][1]
                self.edges[self.edgeLink[i][6]].N = sizes[i][1]
                self.edges[self.edgeLink[i][7]].N = sizes[i][1]

                self.edges[self.edgeLink[i][8]].N  = sizes[i][2]
                self.edges[self.edgeLink[i][9]].N  = sizes[i][2]
                self.edges[self.edgeLink[i][10]].N = sizes[i][2]
                self.edges[self.edgeLink[i][11]].N = sizes[i][2]
        else: # N is already set in the edge objects, use them
            sizes = np.zeros((self.nVol, 3), 'intc')
            for ivol in range(self.nVol):
                sizes[ivol][0] = self.edges[self.edgeLink[ivol][0]].N
                sizes[ivol][1] = self.edges[self.edgeLink[ivol][2]].N
                sizes[ivol][2] = self.edges[self.edgeLink[ivol][8]].N

        if volumeList is None:
            volumeList = np.arange(0, self.nVol)

        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        lIndex = []

        assert len(sizes) == len(volumeList), 'Error: The list of sizes and \
the list of volumes must be the same length'

        # Assign unique numbers to the corners -> Corners are indexed
        # sequentially
        nodeIndex = np.arange(self.nNode)
        counter = len(nodeIndex)

        edgeIndex = [ np.empty((0), 'intc') for i in range(self.nEdge)]
        faceIndex = [ np.empty((0, 0), 'intc') for i in range(self.nFace)]
        # Assign unique numbers to the edges

        for ii in range(len(volumeList)):
            curSizeE = [sizes[ii][0], sizes[ii][0], sizes[ii][1],
                          sizes[ii][1], sizes[ii][0], sizes[ii][0],
                          sizes[ii][1], sizes[ii][1], sizes[ii][2],
                          sizes[ii][2], sizes[ii][2], sizes[ii][2]]

            curSizeF = [[sizes[ii][0], sizes[ii][1]],
                          [sizes[ii][0], sizes[ii][1]],
                          [sizes[ii][1], sizes[ii][2]],
                          [sizes[ii][1], sizes[ii][2]],
                          [sizes[ii][0], sizes[ii][2]],
                          [sizes[ii][0], sizes[ii][2]]]

            ivol = volumeList[ii]
            for iedge in range(12):
                edge = self.edgeLink[ii][iedge]
                if edgeIndex[edge].shape == (0, ):# Not added yet
                    edgeIndex[edge] = np.resize(
                        edgeIndex[edge], curSizeE[iedge]-2)
                    if self.edges[edge].degen == 1:
                        # Get the counter value for this "node"
                        index = nodeIndex[self.edges[edge].n1]
                        for jj in range(curSizeE[iedge]-2):
                            edgeIndex[edge][jj] = index
                    else:
                        edgeIndex[edge][:] = np.arange(counter,counter+curSizeE[iedge]-2)
                        counter += curSizeE[iedge]-2

            for iface in range(6):
                face = self.faceLink[ii][iface]
                if faceIndex[face].shape == (0, 0):
                    faceIndex[face] = np.resize(faceIndex[face],
                                               [curSizeF[iface][0]-2,
                                                curSizeF[iface][1]-2])
                    N = curSizeF[iface][0]-2
                    M = curSizeF[iface][1]-2
                    faceIndex[face] = np.arange(counter,counter+N*M).reshape((N,M))
                    counter += N*M
        # end for (volume list)

        lIndex = []

        # Now actually fill everything up
        for ii in range(len(volumeList)):
            iVol = volumeList[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            L = sizes[ii][2]
            lIndex.append(-1*np.ones((N, M, L), 'intc'))

            # 8 Corners
            for iNode in range(8):
                curNode = self.nodeLink[iVol][iNode]
                lIndex[ii] = setNodeValue(lIndex[ii], nodeIndex[curNode], iNode)

            # 12 Edges
            for iEdge in range(12):
                curEdge = self.edgeLink[iVol][iEdge]
                edgeDir = self.edgeDir[iVol][iEdge]
                lIndex[ii] = setEdgeValue(lIndex[ii], edgeIndex[curEdge],
                                       edgeDir, iEdge)
            # 6 Faces
            for iFace in range(6):
                curFace = self.faceLink[iVol][iFace]
                faceDir = self.faceDirRev[iVol][iFace]
                lIndex[ii] = setFaceValue(lIndex[ii], faceIndex[curFace],
                                       faceDir, iFace)
            # Interior
            toAdd = (N-2)*(M-2)*(L-2)

            lIndex[ii][1:N-1,1:M-1,1:L-1] = \
                np.arange(counter,counter+toAdd).reshape((N-2,M-2,L-2))
            counter = counter + toAdd
        # end for

        if gIndex:
            # We must add [] for each global node
            gIndex = [ [] for i in range(counter)]

            for ii in range(len(volumeList)):
                iVol = volumeList[ii]
                N = sizes[ii][0]
                M = sizes[ii][1]
                L = sizes[ii][2]

                for i in range(N):
                    for j in range(M):
                        for k in range(L):
                            gIndex[lIndex[ii][i,j,k]].append([iVol,i,j,k])
        else:
            gIndex = None

        self.nGlobal = counter
        self.gIndex = gIndex
        self.lIndex = lIndex

        if greedyReorder:

            # Reorder the indices with a greedy scheme
            newIndices = np.zeros(len(gIndex), 'intc')
            newIndices[:] = -1
            newGIndex = [[] for i in range(len(gIndex))]
            counter = 0

            # Re-order the lIndex
            for ii in range(len(volumeList)):
                ivol = volumeList[ii]
                N = sizes[ii][0]
                M = sizes[ii][1]
                L = sizes[ii][2]
                for i in range(N):
                    for j in range(M):
                        for k in range(L):
                            if newIndices[lIndex[ii][i, j, k]] == -1:
                                newIndices[lIndex[ii][i, j, k]] = counter
                                lIndex[ii][i, j, k] = counter
                                counter += 1
                            else:
                                lIndex[ii][i, j, k] = \
                                    newIndices[lIndex[ii][i, j, k]]

            # Re-order the gIndex
            for ii in range(len(gIndex)):
                ivol  = gIndex[ii][0][0]
                i     = gIndex[ii][0][1]
                j     = gIndex[ii][0][2]
                k     = gIndex[ii][0][3]
                pt = lIndex[ivol][i, j, k]
                newGIndex[pt] = gIndex[ii]

            self.gIndex = newGIndex
            self.lIndex = lIndex
        # end if (greedy reorder)

        return

    def reOrder(self, reOrderList):
        """This function takes as input a permutation list which is
        used to reorder the entities in the topology object"""

        # Class atributates that possible need to be modified
        for i in range(8):
            self.nodeLink[:, i] = self.nodeLink[:, i].take(reOrderList)

        for i in range(12):
            self.edgeLink[:, i] = self.edgeLink[:, i].take(reOrderList)
            self.edgeDir[:, i] = self.edgeDir[:, i].take(reOrderList)

        for i in range(6):
            self.faceLink[:, i] = self.faceLink[:, i].take(reOrderList)
            self.faceDir[:, i] = self.faceDir[:, i].take(reOrderList)

        return

class Edge(object):
    """A class for edge objects"""

    def __init__(self, n1, n2, cont, degen, intersect, dg, N):
        self.n1        = n1        # Integer for node 1
        self.n2        = n2        # Integer for node 2
        self.cont      = cont      # Integer: 0 for c0 continuity, 1
                                   # for c1 continuity
        self.degen     = degen     # Integer: 1 for degenerate, 0 otherwise
        self.intersect = intersect # Integer: 1 for an intersected
                                   # edge, 0 otherwise
        self.dg        = dg        # Design Group index
        self.N         = N         # Number of control points for this edge

    def writeInfo(self, i, handle):
        handle.write('  %5d        | %5d | %5d | %5d | %5d | %5d |\
  %5d |  %5d |\n'\
                     %(i, self.n1, self.n2, self.cont, self.degen,
                       self.intersect, self.dg, self.N))


class EdgeCmpObject(object):
    """A temporary class for sorting edge objects"""

    def __init__(self, n1, n2, n1o, n2o, midPt, tol):
        self.n1 = n1
        self.n2 = n2
        self.nodes = [n1o, n2o]
        self.midPt = midPt
        self.tol = tol

    def __repr__(self):
        return 'Node1: %d Node2: %d MidPt: %f %f %f'% (
            self.n1, self.n2, self.midPt[0], self.midPt[1], self.midPt[2])

    def __lt__(self, other):

        if self.n1 != other.n1:
            return self.n1 < other.n1

        if self.n2 != other.n2:
            return self.n2 < other.n2

        if eDist(self.midPt, other.midPt) < self.tol:
            return False
        else:
            if self.midPt[0] != other.midPt[0]:
                return self.midPt[0] < other.midPt[0]
            if self.midPt[1] != other.midPt[1]:
                return self.midPt[1] < other.midPt[1]
            if self.midPt[2] != other.midPt[2]:
                return self.midPt[2] < other.midPt[2]
            return False

    def __eq__(self, other):
        if (self.n1 == other.n1 and self.n2 == other.n2 and
            eDist(self.midPt, other.midPt) < self.tol):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

class FaceCmpObject(object):
    """A temporary class for sorting edge objects"""

    def __init__(self, n1, n2, n3, n4, n1o, n2o, n3o, n4o, midPt, tol):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.nodes = [n1o, n2o, n3o, n4o]
        self.midPt = midPt
        self.tol = tol

    def __repr__(self):
        return 'n1: %d n2: %d n3: %d n4: %d MidPt: %f %f %f'%(
            self.n1, self.n2, self.n3, self.n4, self.midPt[0], self.midPt[1], self.midPt[2])

    def __lt__(self, other):

        if self.n1 != other.n1:
            return self.n1 < other.n1

        if self.n2 != other.n2:
            return self.n2 < other.n2

        if self.n3 != other.n3:
            return self.n3 < other.n3

        if self.n4 != other.n4:
            return self.n4 < other.n4

        if eDist(self.midPt, other.midPt) < self.tol:
            return False
        else:
            if self.midPt[0] != other.midPt[0]:
                return self.midPt[0] < other.midPt[0]
            if self.midPt[1] != other.midPt[1]:
                return self.midPt[1] < other.midPt[1]
            if self.midPt[2] != other.midPt[2]:
                return self.midPt[2] < other.midPt[2]
            return False

    def __eq__(self, other):
        if (self.n1 == other.n1 and self.n2 == other.n2 and
            self.n3 == other.n3 and self.n4 == other.n4 and
            eDist(self.midPt, other.midPt) < self.tol):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

# --------------------------------------------------------------
#                Array Rotation and Flipping Functions
# --------------------------------------------------------------

def rotateCCW(inArray):
    """Rotate the inArray array 90 degrees CCW"""
    rows = inArray.shape[0]
    cols = inArray.shape[1]
    output = np.empty([cols, rows], inArray.dtype)

    for row in range(rows):
        for col in range(cols):
            output[cols-col-1][row] = inArray[row][col]

    return output

def rotateCW(inArray):
    """Rotate the inArray array 90 degrees CW"""
    rows = inArray.shape[0]
    cols = inArray.shape[1]
    output = np.empty([cols, rows], inArray.dtype)

    for row in range(rows):
        for col in range(cols):
            output[col][rows-row-1] = inArray[row][col]

    return output

def reverseRows(inArray):
    """Flip Rows (horizontally)"""
    rows = inArray.shape[0]
    cols = inArray.shape[1]
    output = np.empty([rows, cols], inArray.dtype)
    for row in range(rows):
        output[row] = inArray[row][::-1].copy()

    return output

def reverseCols(inArray):
    """Flip Cols (vertically)"""
    rows = inArray.shape[0]
    cols = inArray.shape[1]
    output = np.empty([rows, cols], inArray.dtype)
    for col in range(cols):
        output[:, col] = inArray[:, col][::-1].copy()

    return output

def getBiLinearMap(edge0, edge1, edge2, edge3):
    """Get the UV coordinates on a square defined from spacing on the edges"""

    assert len(edge0)==len(edge1), 'Error, getBiLinearMap:\
 The len of edge0 and edge1 are not the same'
    assert len(edge2)==len(edge3), 'Error, getBiLinearMap:\
 The len of edge2 and edge3 are no the same'

    N = len(edge0)
    M = len(edge2)

    UV = np.zeros((N, M, 2))

    UV[:, 0, 0] = edge0
    UV[:, 0, 1] = 0.0

    UV[:, -1, 0] = edge1
    UV[:, -1, 1] = 1.0

    UV[0, :, 0] = 0.0
    UV[0, :, 1] = edge2

    UV[-1, :, 0] = 1.0
    UV[-1, :, 1] = edge3

    for i in range(1, N-1):
        x1 = edge0[i]
        y1 = 0.0

        x2 = edge1[i]
        y2 = 1.0

        for j in range(1, M-1):
            x3 = 0
            y3 = edge2[j]
            x4 = 1.0
            y4 = edge3[j]
            UV[i, j] = calcIntersection(x1, y1, x2, y2, x3, y3, x4, y4)


    return UV

def calcIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calc the intersection between two line segments defined by
    # (x1,y1) to (x2,y2) and (x3,y3) to (x4,y4)

    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    ua = ((x4-x3)*(y1-y3)-(y4-y3)*(x1-x3))/denom
    xi = x1 + ua*(x2-x1)
    yi = y1 + ua*(y2-y1)

    return xi, yi

def fillKnots(t, k, level):
    t = t[k-1:-k+1] # Strip out the np.zeros
    newT = np.zeros(len(t) + (len(t)-1)*level)
    start = 0
    for i in range(len(t)-1):
        tmp = np.linspace(t[i], t[i+1], level+2)
        for j in range(level+2):
            newT[start + j] = tmp[j]

        start += level + 1

    return newT

def projectNodePID(pt, upVec, p0, v1, v2, uv0, uv1, uv2, PID):
    """
    Project a point pt onto a triagnulated surface and return the patchID
    and the u,v coordinates in that patch.

    pt: The initial point
    upVec: The vector pointing in the search direction
    p0: A numpy array of triangle origins
    v1: A numpy array of the first triangle vectors
    v2: A numpy array of the second triangle vectors
    uu: An array containing u coordinates
    vv: An array containing v coordinates
    pid: An array containing pid values

    """

    # Get the bounds of the geo object so we know what to scale by

    fail = 0
    if p0.shape[0] == 0:
        fail = 2
        return None, None, fail

    tmpSol,tmpPid, nSol = pySpline.libspline.line_plane(pt, upVec, p0.T, v1.T, v2.T)
    tmpSol = tmpSol.T
    tmpPid -= 1

    # Check to see if any of the solutions happen be identical.
    points = []
    for i in range(nSol):
        points.append(tmpSol[i, 3:6])

    if nSol > 1:
        tmp, link = pointReduce(points, nodeTol=1e-12)
        nUnique = np.max(link) + 1
        points = np.zeros((nUnique,3))
        uu = np.zeros(nUnique)
        vv = np.zeros(nUnique)
        ss = np.zeros(nUnique)
        pid = np.zeros(nUnique,'intc')

        for i in range(nSol):
            points[link[i]] = tmpSol[i, 3:6]
            uu[link[i]] = tmpSol[i, 1]
            vv[link[i]] = tmpSol[i, 2]
            ss[link[i]] = tmpSol[i, 0]
            pid[link[i]] = tmpPid[i]

        nSol = len(points)
    else:
        nUnique = 1
        points = np.zeros((nUnique,3))
        uu = np.zeros(nUnique)
        vv = np.zeros(nUnique)
        ss = np.zeros(nUnique)
        pid = np.zeros(nUnique,'intc')

        points[0] = tmpSol[0, 3:6]
        uu[0] = tmpSol[0, 1]
        vv[0] = tmpSol[0, 2]
        ss[0] = tmpSol[0, 0]
        pid[0] = tmpPid[0]

    if nSol == 0:
        fail = 2
        return None, None, fail
    elif nSol == 1:
        fail = 1

        first  = points[0]
        firstPatchID = PID[pid[0]]
        firstU = uv0[pid[0]][0] + uu[0]*(uv1[pid[0]][0] - uv0[pid[0]][0])
        firstV = uv0[pid[0]][1] + vv[0]*(uv2[pid[0]][1] - uv0[pid[0]][1])
        firstS = ss[0]
        return [first, firstPatchID, firstU, firstV, firstS],  None,  fail
    elif nSol == 2:
        fail = 0

        # Determine the 'top' and 'bottom' solution
        first  = points[0]
        second = points[1]

        firstPatchID = PID[pid[0]-1]
        secondPatchID = PID[pid[1]-1]

        firstU = uv0[pid[0]][0] + uu[0]*(uv1[pid[0]][0] - uv0[pid[0]][0])
        firstV = uv0[pid[0]][1] + vv[0]*(uv2[pid[0]][1] - uv0[pid[0]][1])
        firstS = ss[0]

        secondU = uv0[pid[1]][0] + uu[1]*(uv1[pid[1]][0] - uv0[pid[1]][0])
        secondV = uv0[pid[1]][1] + vv[1]*(uv2[pid[1]][1] - uv0[pid[1]][1])
        secondS = ss[1]

        if np.dot(first - pt, upVec) >= np.dot(second - pt, upVec):

            return [first, firstPatchID, firstU, firstV, firstS],\
                [second, secondPatchID, secondU, secondV, secondS], fail
        else:
            return [second, secondPatchID, secondU, secondV, secondS],\
                [first, firstPatchID, firstU, firstV, firstS], fail

    else:
        print('This functionality is not implemtned in geoUtils yet')
        sys.exit(1)


def projectNodePIDPosOnly(pt, upVec, p0, v1, v2, uv0, uv1, uv2, PID):

    # Get the bounds of the geo object so we know what to scale by

    fail = 0
    if p0.shape[0] == 0:
        fail = 1
        return None, fail

    sol, pid, nSol = pySpline.libspline.line_plane(pt, upVec, p0.T, v1.T, v2.T)
    sol = sol.T
    pid -= 1

    if nSol == 0:
        fail = 1
        return None, fail
    elif nSol >= 1:
        # Find the least positve solution
        minIndex = -1
        d = 0.0
        for k in range(nSol):
            dn = np.dot(sol[k, 3:6] - pt, upVec)
            if dn >= 0.0 and (minIndex == -1 or dn < d):
                minIndex = k
                d = dn

        if minIndex >= 0:
            patchID = PID[pid[minIndex]]
            u = uv0[pid[minIndex]][0] + sol[minIndex, 1]*\
                (uv1[pid[minIndex]][0] - uv0[pid[minIndex]][0])

            v = uv0[pid[minIndex]][1] + sol[minIndex, 2]*\
                (uv2[pid[minIndex]][1] - uv0[pid[minIndex]][1])
            s = sol[minIndex,0]
            tmp = [sol[minIndex,3], sol[minIndex,4], sol[minIndex,5]]
            return [tmp, patchID, u, v, s], fail

    fail = 1
    return None, fail

def projectNode(pt, upVec, p0, v1, v2):
    """
    Project a point pt onto a triagnulated surface and return two
    intersections.

    pt: The initial point
    upVec: The vector pointing in the search direction
    p0: A numpy array of triangle origins
    v1: A numpy array of the first triangle vectors
    v2: A numpy array of the second triangle vectors
    """

    # Get the bounds of the geo object so we know what to scale by

    fail = 0
    if p0.shape[0] == 0:
        fail = 2
        return None, None, fail

    sol,pid, nSol = pySpline.libspline.line_plane(pt, upVec, p0.T, v1.T, v2.T)
    sol = sol.T

    # Check to see if any of the solutions happen be identical.
    if nSol > 1:
        points = []
        for i in range(nSol):
            points.append(sol[i, 3:6])

        newPoints, link = pointReduce(points, nodeTol=1e-12)
        nSol = len(newPoints)
    else:
        newPoints = []
        for i in range(nSol):
            newPoints.append(sol[i, 3:6])

    if nSol == 0:
        fail = 2
        return None, None, fail
    elif nSol == 1:
        fail = 1
        return newPoints[0],  None,  fail
    elif nSol == 2:
        fail = 0

        # Determine the 'top' and 'bottom' solution
        first  = newPoints[0]
        second = newPoints[1]

        if np.dot(first - pt, upVec) >= np.dot(second - pt, upVec):
            return first, second, fail
        else:
            return second, first, fail
    else:
        # This just returns the two points with the minimum absolute
        # distance to the points that have been found.
        fail = -1

        pmin = abs(np.dot(newPoints[:nSol] - pt, upVec))
        minIndex = np.argsort(pmin)

        return newPoints[minIndex[0]], newPoints[minIndex[1]], fail

def projectNodePosOnly(pt, upVec, p0, v1, v2):
    """
    Project a point pt onto a triagnulated surface and the solution
    that is the closest in the positive direction (as defined by
    upVec).

    pt: The initial point
    upVec: The vector pointing in the search direction
    p0: A numpy array of triangle origins
    v1: A numpy array of the first triangle vectors
    v2: A numpy array of the second triangle vectors
    """

    # Get the bounds of the geo object so we know what to scale by

    fail = 0
    if p0.shape[0] == 0:
        fail = 1
        return None, fail

    sol, pid, nSol = pySpline.libspline.line_plane(pt, upVec, p0.T, v1.T, v2.T)
    sol = sol.T

    if nSol == 0:
        fail = 1
        return None, fail
    elif nSol >= 1:
        # Find the least positve solution
        minIndex = -1
        d = 0.0
        for k in range(nSol):
            dn = np.dot(sol[k, 3:6] - pt, upVec)
            if dn >= 0.0 and (minIndex == -1 or dn < d):
                minIndex = k
                d = dn

        if minIndex >= 0:
            return sol[minIndex][3:6], fail

    fail = 1
    return None, fail

def tfi_2d(e0, e1, e2, e3):
    # Input
    # e0: Nodes along edge 0. Size Nu x 3
    # e1: Nodes along edge 1. Size Nu x 3
    # e0: Nodes along edge 2. Size Nv x 3
    # e1: Nodes along edge 3. Size Nv x 3

    try:
        X = pySpline.libspline.tfi2d(e0.T, e1.T, e2.T, e3.T).T
    except:

        Nu = len(e0)
        Nv = len(e2)
        assert Nu == len(e1), 'Number of nodes on edge0 and edge1\
 are not the same, %d %d'%(len(e0), len(e1))
        assert Nv == len(e3), 'Number of nodes on edge2 and edge3\
 are not the same, %d %d'%(len(e2), len(e3))

        U = np.linspace(0, 1, Nu)
        V = np.linspace(0, 1, Nv)

        X = np.zeros((Nu, Nv, 3))

        for i in range(Nu):
            for j in range(Nv):
                X[i, j] = (1-V[j])*e0[i] + V[j]*e1[i] +\
                    (1-U[i])*e2[j] + U[i]*e3[j] - \
                    (U[i]*V[j]*e1[-1] + U[i]*(1-V[j])*e0[-1] +\
                         V[j]*(1-U[i])*e1[0] + (1-U[i])*(1-V[j])*e0[0])
    return X

def linearEdge(pt1, pt2, N):
    # Return N points along line from pt1 to pt2
    pts = np.zeros((N, len(pt1)))
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    for i in range(N):
        pts[i] = float(i)/(N-1)*(pt2-pt1) + pt1
    return pts

def splitQuad(e0, e1, e2, e3, alpha, beta, NO):
    # This function takes the coordinates of a quad patch, and
    # creates an O-grid inside the quad making 4 quads and leaving
    # a hole in the center whose size is determined by alpha and
    # beta

    # Input                        Output
    #       2             3        2      e1     3
    #       +-------------+        +-------------+
    #       |             |        |\           /|
    #       |             |        | \c2  P1 c3/ |
    #       |             |        |  \       /  |
    #       |             |        |   \6   7/   |
    #       |             |        |    \***/    |
    #       |             |     e2 | P2 *   * P3 | e3
    #       |             |        |    /***\    |
    #       |             |        |   / 4  5\   |
    #       |             |        |  /       \  |
    #       |             |        | /c0 P0  c1\ |
    #       |             |        |/           \|
    #       +-------------+        +-------------+
    #       0             1        0     e0      1

    # Input:
    # e0: points along edge0
    # e1: points along edge1
    # e2: points along edge2
    # e3: points along edge3
    # alpha: Fraction of hole covered by u-direction
    # beta : Fraction of hole covered by v-direction

    # Makeing the assumption each edge is fairly straight
    Nu = len(e0)
    Nv = len(e2)

    # Corners of patch
    pts = np.zeros((4, 3))
    pts[0] = e0[0]
    pts[1] = e0[-1]
    pts[2] = e1[0]
    pts[3] = e1[-1]

    # First generate edge lengths
    l = np.zeros(4)
    l[0] = eDist(pts[0], pts[1])
    l[1] = eDist(pts[2], pts[3])
    l[2] = eDist(pts[0], pts[2])
    l[3] = eDist(pts[1], pts[3])

    # Vector along edges 0->3
    vec = np.zeros((4, 3))
    vec[0] = pts[1]-pts[0]
    vec[1] = pts[3]-pts[2]
    vec[2] = pts[2]-pts[0]
    vec[3] = pts[3]-pts[1]

    U = 0.5*(vec[0]+vec[1])
    V = 0.5*(vec[2]+vec[3])
    u = U/euclideanNorm(U)
    v = V/euclideanNorm(V)

    mid  = np.average(pts, axis=0)

    uBar = 0.5*(l[0]+l[1])*alpha
    vBar = 0.5*(l[2]+l[3])*beta

    aspect = uBar/vBar

    if aspect < 1: # its higher than wide, logically roate the element
        v, u = u, -v
        vBar, uBar = uBar, vBar
        alpha, beta = beta, alpha
        Nv, Nu = Nu, Nv

        E0 = e2[::-1, :].copy()
        E1 = e3[::-1, :].copy()
        E2 = e1.copy()
        E3 = e0.copy()

        #Also need to permute points
        PTS = np.zeros((4, 3))
        PTS[0] = pts[2].copy()
        PTS[1] = pts[0].copy()
        PTS[2] = pts[3].copy()
        PTS[3] = pts[1].copy()
    else:
        E0 = e0.copy()
        E1 = e1.copy()
        E2 = e2.copy()
        E3 = e3.copy()
        PTS = pts.copy()

    rectCorners = np.zeros((4, 3))

    # These are the output pactch object
    P0 = np.zeros((Nu, 4, 3), 'd')
    P1 = np.zeros((Nu, 4, 3), 'd')
    P2 = np.zeros((Nv, 4, 3), 'd')
    P3 = np.zeros((Nv, 4, 3), 'd')

    rad = vBar*beta
    rectLen = uBar-2*rad
    if rectLen < 0:
        rectLen = 0.0

    # Determine 4 corners of rectangular part
    rectCorners[0] = mid-u*(rectLen/2)-np.sin(np.pi/4)*\
        rad*v-np.cos(np.pi/4)*rad*u
    rectCorners[1] = mid+u*(rectLen/2)-np.sin(np.pi/4)*\
        rad*v+np.cos(np.pi/4)*rad*u
    rectCorners[2] = mid-u*(rectLen/2)+np.sin(np.pi/4)*\
        rad*v-np.cos(np.pi/4)*rad*u
    rectCorners[3] = mid+u*(rectLen/2)+np.sin(np.pi/4)*\
        rad*v+np.cos(np.pi/4)*rad*u

    arcLen = np.pi*rad/2 + rectLen # Two quarter circles straight line
    eighthArc = 0.25*np.pi*rad
    # We have to distribute Nu-2 nodes over this arc-length
    spacing = arcLen/(Nu-1)

    botEdge = np.zeros((Nu, 3), 'd')
    topEdge = np.zeros((Nu, 3), 'd')
    botEdge[0] = rectCorners[0]
    botEdge[-1] = rectCorners[1]
    topEdge[0] = rectCorners[2]
    topEdge[-1] = rectCorners[3]
    for i in range(Nu-2):
        distAlongArc = (i+1)*spacing
        if distAlongArc < eighthArc:
            theta = distAlongArc/rad # Angle in radians
            botEdge[i+1] = mid-u*(rectLen/2) - \
                np.sin(theta+np.pi/4)*rad*v - np.cos(theta+np.pi/4)*rad*u
            topEdge[i+1] = mid-u*(rectLen/2) + \
                np.sin(theta+np.pi/4)*rad*v - np.cos(theta+np.pi/4)*rad*u
        elif distAlongArc > rectLen+eighthArc:
            theta = (distAlongArc-rectLen-eighthArc)/rad
            botEdge[i+1] = mid+u*(rectLen/2) + \
                np.sin(theta)*rad*u - np.cos(theta)*rad*v
            topEdge[i+1] = mid+u*(rectLen/2) + \
                np.sin(theta)*rad*u + np.cos(theta)*rad*v
        else:
            topEdge[i+1] = mid -u*rectLen/2 + rad*v + \
                (distAlongArc-eighthArc)*u
            botEdge[i+1] = mid -u*rectLen/2 - rad*v + \
                (distAlongArc-eighthArc)*u

    leftEdge = np.zeros((Nv, 3), 'd')
    rightEdge = np.zeros((Nv, 3), 'd')
    theta = np.linspace(-np.pi/4, np.pi/4, Nv)

    for i in range(Nv):
        leftEdge[i]  = mid-u*(rectLen/2) + \
            np.sin(theta[i])*rad*v - np.cos(theta[i])*rad*u
        rightEdge[i] = mid+u*(rectLen/2) + \
            np.sin(theta[i])*rad*v + np.cos(theta[i])*rad*u

    # Do the corner edges
    c0 = linearEdge(PTS[0], rectCorners[0], NO)
    c1 = linearEdge(PTS[1], rectCorners[1], NO)
    c2 = linearEdge(PTS[2], rectCorners[2], NO)
    c3 = linearEdge(PTS[3], rectCorners[3], NO)

    # Now we can finally do the pactches
    P0 = tfi_2d(E0, botEdge, c0, c1)
    P1 = tfi_2d(E1, topEdge, c2, c3)
    P2 = tfi_2d(E2, leftEdge, c0, c2)
    P3 = tfi_2d(E3, rightEdge, c1, c3)

    if aspect < 1:
        return P3, P2, P0[::-1, :, :], P1[::-1, :, :]
    else:
        return P0, P1, P2, P3

def createTriPanMesh(geo, tripanFile, wakeFile,
                     specsFile=None, defaultSize=0.1):
    """
    Create a TriPan mesh from a pyGeo object.

    geo:          The pyGeo object
    tripanFile:  The name of the TriPan File
    wakeFile:    The name of the wake file
    specsFile:   The specification of panels/edge and edge types
    defaultSize: If no specsFile is given, attempt to make edges with
    defaultSize-length panels

    This cannot be run in parallel!
    """

    # Use the topology of the entire geo object
    topo = geo.topo

    nEdge = topo.nEdge
    nFace = topo.nFace

    # Face orientation
    faceOrientation = [1]*nFace
    # edgeNumber == number of panels along a given edge
    edgeNumber = -1*np.ones(nEdge, 'intc')
    # edgeType == what type of parametrization to use along an edge
    edgeType   = ['linear']*nEdge
    wakeEdges = []
    wakeDir   = []

    if specsFile:
        f = open(specsFile, 'r')
        line = f.readline().split()
        if int(line[0]) != nFace:
            print('Number of faces do not match in specs file')
        if int(line[1]) != nEdge:
            print('Number of edges do not match in specs file')
        # Discard a line
        f.readline()
        # Read in the face info
        for iface in range(nFace):
            aux = f.readline().split()
            faceOrientation[iface] = int(aux[1])
        f.readline()
        # Read in the edge info
        for iedge in range(nEdge):
            aux = f.readline().split()
            edgeNumber[iedge] = int(aux[1])
            edgeType[iedge] = aux[2]
            if int(aux[5]) > 0:
                wakeEdges.append(iedge)
                wakeDir.append(1)
            elif int(aux[5]) < 0:
                wakeEdges.append(iedge)
                wakeDir.append(-1)
        f.close()
    else:
        defaultSize = float(defaultSize)
        # First Get the default number on each edge

        for iface in range(nFace):
            for iedge in range(4):
                # First check if we even have to do it
                if edgeNumber[topo.edgeLink[iface][iedge]] == -1:
                    # Get the physical length of the edge
                    edgeLength = \
                        geo.surfs[iface].edgeCurves[iedge].getLength()

                    # Using defaultSize calculate the number of panels
                    # along this edge
                    edgeNumber[topo.edgeLink[iface][iedge]] = \
                        int(np.floor(edgeLength/defaultSize))+2
    # end if

    # Create the sizes Geo for the make consistent function
    sizes = []
    order = [0]*nFace
    for iface in range(nFace):
        sizes.append([edgeNumber[topo.edgeLink[iface][0]],
                      edgeNumber[topo.edgeLink[iface][2]]])

    sizes, edgeNumber = topo.makeSizesConsistent(sizes, order)

    # Now we need to get the edge parameter spacing for each edge
    topo.calcGlobalNumbering(sizes) # This gets gIndex,lIndex and counter

    # Now calculate the intrinsic spacing for each edge:
    edgePara = []
    for iedge in range(nEdge):
        if edgeType[iedge] == 'linear':
            spacing = np.linspace(0.0, 1.0, edgeNumber[iedge])
            edgePara.append(spacing)
        elif edgeType[iedge] == 'cos':
            spacing = 0.5*(1.0 - np.cos(np.linspace(
                        0, np.pi, edgeNumber[iedge])))
            edgePara.append(spacing)
        elif edgeType[iedge] == 'hyperbolic':
            x = np.linspace(0.0, 1.0, edgeNumber[iedge])
            beta = 1.8
            spacing = x - beta*x*(x - 1.0)*(x - 0.5)
            edgePara.append(spacing)
        else:
            print('Warning: Edge type %s not understood. \
Using a linear type'%(edgeType[iedge]))
            edgePara.append(np.linspace(0, 1, edgeNumber[iedge]))

    # Get the number of panels
    nPanels = 0
    nNodes = len(topo.gIndex)
    for iface in range(nFace):
        nPanels += (sizes[iface][0]-1)*(sizes[iface][1]-1)

    # Open the outputfile
    fp = open(tripanFile, 'w')

    # Write he number of points and panels
    fp.write('%5d %5d\n'%(nNodes, nPanels))

    # Output the Points First
    UV = []
    for iface in range(nFace):
        UV.append(getBiLinearMap(
            edgePara[topo.edgeLink[iface][0]],
            edgePara[topo.edgeLink[iface][1]],
            edgePara[topo.edgeLink[iface][2]],
            edgePara[topo.edgeLink[iface][3]]))

    for ipt in range(len(topo.gIndex)):
        iface = topo.gIndex[ipt][0][0]
        i     = topo.gIndex[ipt][0][1]
        j     = topo.gIndex[ipt][0][2]
        pt = geo.surfs[iface].getValue(UV[iface][i, j][0],  UV[iface][i, j][1])
        fp.write( '%12.10e %12.10e %12.10e \n'%(pt[0], pt[1], pt[2]))

    # Output the connectivity Next
    for iface in range(nFace):
        if faceOrientation[iface] >= 0:
            for i in range(sizes[iface][0]-1):
                for j in range(sizes[iface][1]-1):
                    fp.write('%d %d %d %d \n'%(topo.lIndex[iface][i, j],
                                               topo.lIndex[iface][i+1, j],
                                               topo.lIndex[iface][i+1, j+1],
                                               topo.lIndex[iface][i, j+1]))
        else:
            for i in range(sizes[iface][0]-1):
                for j in range(sizes[iface][1]-1):
                    fp.write('%d %d %d %d \n'%(topo.lIndex[iface][i, j],
                                               topo.lIndex[iface][i, j+1],
                                               topo.lIndex[iface][i+1, j+1],
                                               topo.lIndex[iface][i+1, j]))

    fp.write('\n')
    fp.close()

    # Output the wake file
    fp = open(wakeFile,  'w')
    fp.write('%d\n'%(len(wakeEdges)))
    print('wakeEdges:', wakeEdges)

    for k in range(len(wakeEdges)):
        # Get a surface/edge for this edge
        surfaces = topo.getSurfaceFromEdge(wakeEdges[k])
        iface = surfaces[0][0]
        iedge = surfaces[0][1]
        if iedge == 0:
            indices = topo.lIndex[iface][:, 0]
        elif iedge == 1:
            indices = topo.lIndex[iface][:, -1]
        elif iedge == 2:
            indices = topo.lIndex[iface][0, :]
        elif iedge == 3:
            indices = topo.lIndex[iface][-1, :]

        fp.write('%d\n'%(len(indices)))

        if wakeDir[k] > 0:
            for i in range(len(indices)):
                # A constant in TriPan to indicate projected wake
                teNodeType = 3
                fp.write('%d %d\n'%(indices[i],  teNodeType))
        else:
            for i in range(len(indices)):
                teNodeType = 3
                fp.write('%d %d\n'%(indices[len(indices)-1-i], teNodeType))
    # end for
    fp.close()

    # Write out the default specFile
    if specsFile is None:
        (dirName, fileName) = os.path.split(tripanFile)
        (fileBaseName, fileExtension) = os.path.splitext(fileName)
        if dirName != '':
            newSpecsFile = dirName+'/'+fileBaseName+'.specs'
        else:
            newSpecsFile = fileBaseName+'.specs'

        specsFile = newSpecsFile

    if not os.path.isfile(specsFile):
        f = open(specsFile, 'w')
        f.write('%d %d Number of faces and number of edges\n'%(nFace, nEdge))
        f.write('Face number   Normal (1 for regular, -1 for\
 reversed orientation\n')
        for iface in range(nFace):
            f.write('%d %d\n'%(iface, faceOrientation[iface]))
        f.write('Edge Number #Node Type     Start Space   End Space\
   WakeEdge\n')
        for iedge in range(nEdge):
            if iedge in wakeEdges:
                f.write( '  %4d    %5d %10s %10.4f %10.4f  %1d \n'%(\
                        iedge, edgeNumber[iedge], edgeType[iedge],
                        .1, .1, 1))
            else:
                f.write( '  %4d    %5d %10s %10.4f %10.4f  %1d \n'%(\
                        iedge, edgeNumber[iedge], edgeType[iedge],
                        .1, .1, 0))
        f.close()

# 2D Doubly connected edge list implementation.
#Copyright 2008, Angel Yanguas-Gil

class DCELEdge(object):
    def __init__(self, v1, v2, X, PID, uv, tag):
        # Create a representation of a surface edge that contains the
        # required information to be able to construct a trimming
        # curve on the orignal skin surfaces

        self.X = X
        self.PID = PID
        self.uv = uv
        tmp = tag.split('-')
        self.tag = tmp[0]
        if len(tmp) > 1:
            self.seg = tmp[1]
        else:
            self.seg = None

        self.v1 = v1
        self.v2 = v2
        if X is not None:
            self.x1 = 0.5*(X[0,0] + X[0,1])
            self.x2 = 0.5*(X[-1,0] + X[-1,1])

        self.con = [v1,v2]

    def __repr__(self):

        str1 = 'v1: %f %f\nv2: %f %f'% (self.v1[0],self.v1[1],
                                        self.v2[0],self.v2[1])
        return str1

    def midPt(self):
        #return [0.5*(self.v1.x + self.v2.x), 0.5*(self.v1.y + self.v2.y)]
        return 0.5*self.x1 + 0.5*self.x2

class DCELVertex:
    """Minimal implementation of a vertex of a 2D dcel"""

    def __init__(self, uv, X):
        self.x = uv[0]
        self.y = uv[1]
        self.X = X
        self.hedgelist = []

    def sortincident(self):

        self.hedgelist.sort(self.hsort, reverse=True)

    def hsort(self, h1, h2):
        """Sorts two half edges counterclockwise"""

        if h1.angle < h2.angle:
            return -1
        elif h1.angle > h2.angle:
            return 1
        else:
            return 0


class DCELHedge:
    """Minimal implementation of a half-edge of a 2D dcel"""

    def __init__(self,v1,v2,X,PID,uv,tag=None):
        #The origin is defined as the vertex it points to
        self.origin = v2
        self.twin = None
        self.face = None
        self.sface = None
        self.uv = uv
        self.PID = PID
        self.nexthedge = None
        self.angle = hangle(v2.x-v1.x, v2.y-v1.y)
        self.prevhedge = None
        self.length = np.sqrt((v2.x-v1.x)**2 + (v2.y-v1.y)**2)
        self.tag = tag


class DCELFace:
    """Implements a face of a 2D dcel"""

    def __init__(self):
        self.wedge = None
        self.data = None
        self.external = None
        self.tag = 'EXTERNAL'
        self.id = None

    def area(self):
        h = self.wedge
        a = 0
        while(not h.nexthedge is self.wedge):
            p1 = h.origin
            p2 = h.nexthedge.origin
            a += p1.x*p2.y - p2.x*p1.y
            h = h.nexthedge

        p1 = h.origin
        p2 = self.wedge.origin
        a = (a + p1.x*p2.y - p2.x*p1.y)/2.0
        return a

    def calcCentroid(self):
        h = self.wedge
        center = np.zeros(2)
        center += [h.origin.x,h.origin.y]
        counter = 1
        while(not h.nexthedge is self.wedge):
            counter += 1
            h = h.nexthedge
            center += [h.origin.x,h.origin.y]

        self.centroid = center/counter

    def calcSpatialCentroid(self):

        h = self.wedge
        center = np.zeros(3)
        center += h.origin.X
        counter = 1
        while(not h.nexthedge is self.wedge):
            counter += 1
            h = h.nexthedge
            center += h.origin.X

        self.spatialCentroid = center/counter

    def perimeter(self):
        h = self.wedge
        p = 0
        while (not h.nexthedge is self.wedge):
            p += h.length
            h = h.nexthedge
        return p

    def vertexlist(self):
        h = self.wedge
        pl = [h.origin]
        while(not h.nexthedge is self.wedge):
            h = h.nexthedge
            pl.append(h.origin)
        return pl



    def isinside(self, P):
        """Determines whether a point is inside a face using a
        winding formula"""
        pl = self.vertexlist()
        V = []
        for i in range(len(pl)):
            V.append([pl[i].x,pl[i].y])
        V.append([pl[0].x,pl[0].y])

        wn = 0
        # loop through all edges of the polygon
        for i in range(len(V)-1):     # edge from V[i] to V[i+1]
            if V[i][1] <= P[1]:        # start y <= P[1]
                if V[i+1][1] > P[1]:     # an upward crossing
                    if isLeft(V[i], V[i+1], P) > 0: # P left of edge
                        wn += 1           # have a valid up intersect
            else:                      # start y > P[1] (no test needed)
                if V[i+1][1] <= P[1]:    # a downward crossing
                    if isLeft(V[i], V[i+1], P) < 0: # P right of edge
                        wn -= 1           # have a valid down intersect
        if wn == 0:
            return False
        else:
            return True

class DCEL(object):
    """
    Implements a doubly-connected edge list
    """

    def __init__(self, vl=None, el=None, fileName=None):
        self.vertices = []
        self.hedges = []
        self.faces = []
        self.faceInfo = None
        if vl is not None and el is not None:
            self.vl = vl
            self.el = el
            self.buildDcel()
        elif fileName is not None:
            self.loadDCEL(fileName)
            self.buildDcel()
        else:
            # The user is going to manually create the hedges and
            # faces
            pass

    def buildDcel(self):
        """
        Creates the dcel from the list of vertices and edges
        """

        # Do some pruning first:
        self.vertices = self.vl
        ii = 0
        while ii < 1000: # Trim at most 1000 edges
            ii += 1

            mult = np.zeros(self.nvertices(),'intc')
            for e in self.el:
                mult[e.con[0]] += 1
                mult[e.con[1]] += 1

            multCheck = mult < 2

            if np.any(multCheck):

                # We need to do a couple of things:
                # 1. The bad vertices need to be removed from the vertex list
                # 2. Remaning vertices must be renamed
                # 3. Edges that reference deleted vertices must be popped
                # 4. Remaining edges must have connectivity info updated

                # First generate new mapping:
                count = 0
                mapping = -1*np.ones(self.nvertices(),'intc')
                deletedVertices = []
                for i in range(self.nvertices()):
                    if  multCheck[i]:
                        # Vertex must be removed
                        self.vertices.pop(i-len(deletedVertices))
                        deletedVertices.append(i)
                    else:
                        mapping[i] = count # Other wise get the mapping count:
                        count += 1

                # Now prune the edges:
                nEdgeDeleted = 0
                for i in range(len(self.el)):
                    if self.el[i-nEdgeDeleted].con[0] in deletedVertices or \
                            self.el[i-nEdgeDeleted].con[1] in deletedVertices:
                        # Edge must be deleted
                        self.el.pop(i-nEdgeDeleted)
                        nEdgeDeleted += 1
                    else:
                        # Mapping needs to be updated:
                        curCon = self.el[i-nEdgeDeleted].con
                        self.el[i-nEdgeDeleted].con[0] = mapping[curCon[0]]
                        self.el[i-nEdgeDeleted].con[1] = mapping[curCon[1]]
            else:
                break

        # end while

#Step 2: hedge list creation. Assignment of twins and
#vertices

        self.hedges = []
        appendCount = 0

        for e in self.el:

            h1 = DCELHedge(self.vertices[e.con[0]],
                       self.vertices[e.con[1]],
                       e.X, e.PID, e.uv, e.tag)
            h2 = DCELHedge(self.vertices[e.con[1]],
                       self.vertices[e.con[0]],
                       e.X, e.PID, e.uv, e.tag)

            h1.twin = h2
            h2.twin = h1
            self.vertices[e.con[1]].hedgelist.append(h1)
            self.vertices[e.con[0]].hedgelist.append(h2)
            appendCount += 2
            self.hedges.append(h2)
            self.hedges.append(h1)


        #Step 3: Identification of next and prev hedges
        for v in self.vertices:
            v.sortincident()
            l = len(v.hedgelist)

            for i in range(l-1):
                v.hedgelist[i].nexthedge = v.hedgelist[i+1].twin
                v.hedgelist[i+1].prevhedge = v.hedgelist[i]

            v.hedgelist[l-1].nexthedge = v.hedgelist[0].twin
            v.hedgelist[0].prevhedge = v.hedgelist[l-1]

        #Step 4: Face assignment
        provlist = self.hedges[:]
        nf = 0
        nh = len(self.hedges)

        while nh > 0:
            h = provlist.pop()
            nh -= 1

            #We check if the hedge already points to a face
            if h.face is None:
                f = DCELFace()
                nf += 1
                #We link the hedge to the new face
                f.wedge = h
                f.wedge.face = f
                #And we traverse the boundary of the new face
                while (not h.nexthedge is f.wedge):
                    h = h.nexthedge
                    h.face = f
                self.faces.append(f)
        #And finally we have to determine the external face
        for f in self.faces:
            f.external = f.area() < 0
            f.calcCentroid()
            f.calcSpatialCentroid()

        if self.faceInfo is not None:
            for i in range(len(self.faceInfo)):
                self.faces[i].tag = self.faceInfo[i]

    def writeTecplot(self, fileName):

        f = open(fileName, 'w')
        f.write ('VARIABLES = "X","Y"\n')
        for i in range(len(self.el)):
            f.write('Zone T=\"edge%d\" I=%d\n'%(i, 2))
            f.write('DATAPACKING=POINT\n')
            v1 = self.el[i].con[0]
            v2 = self.el[i].con[1]

            f.write('%g %g\n'%(self.vl[v1].x,
                               self.vl[v1].y))

            f.write('%g %g\n'%(self.vl[v2].x,
                               self.vl[v2].y))

        f.close()
    def findpoints(self, pl, onetoone=False):
        """Given a list of points pl, returns a list of
        with the corresponding face each point belongs to and
        None if it is outside the map.

        """

        ans = []
        if onetoone:
            fl = self.faces[:]
            for p in pl:
                found = False
                for f in fl:
                    if f.external:
                        continue
                    if f.isinside(p):
                        fl.remove(f)
                        found = True
                        ans.append(f)
                        break
                if not found:
                    ans.append(None)

        else:
            for p in pl:
                found = False
                for f in self.faces:
                    if f.external:
                        continue
                    if f.isinside(p):
                        found = True
                        ans.append(f)
                        break
                if not found:
                    ans.append(None)

        return ans

    def areas(self):
        return [f.area() for f in self.faces if not f.external]

    def perimeters(self):
        return [f.perimeter() for f in self.faces if not f.external]

    def nfaces(self):
        return len(self.faces)

    def nvertices(self):
        return len(self.vertices)

    def nedges(self):
        return len(self.hedges)//2

    def saveDCEL(self, fileName):

        f = open(fileName,'w')
        f.write('%d %d %d\n'%(
                self.nvertices(), self.nedges(), self.nfaces()))
        for i in range(self.nvertices()):
            f.write('%g %g %g %g %g \n'%(
                    self.vertices[i].x,self.vertices[i].y,
                    self.vertices[i].X[0],
                    self.vertices[i].X[1],
                    self.vertices[i].X[2]))

        for i in range(self.nedges()):
            if self.el[i].seg is not None:
                f.write('%d %d %g %g %g %g %g %g %s-%s\n'%(
                        self.el[i].con[0],self.el[i].con[1],
                        self.el[i].x1[0],self.el[i].x1[1],self.el[i].x1[2],
                        self.el[i].x2[0],self.el[i].x2[1],self.el[i].x2[2],
                        self.el[i].tag, self.el[i].seg))
            else:
                f.write('%d %d %g %g %g %g %g %g %s\n'%(
                        self.el[i].con[0],self.el[i].con[1],
                        self.el[i].x1[0],self.el[i].x1[1],self.el[i].x1[2],
                        self.el[i].x2[0],self.el[i].x2[1],self.el[i].x2[2],
                        self.el[i].tag))

        for i in range(self.nfaces()):
            f.write('%s\n'%(self.faces[i].tag))
        f.close()

    def loadDCEL(self, fileName):

        f = open(fileName,'r')
        # Read sizes
        tmp = f.readline().split()
        nvertices = int(tmp[0])
        nedges    = int(tmp[1])
        nfaces    = int(tmp[2])

        self.vl = []
        self.el = []
        self.faceInfo = []
        for i in range(nvertices):
            a = f.readline().split()
            self.vl.append(DCELVertex([float(a[0]),float(a[1])],
                                      np.array([float(a[2]),
                                                float(a[3]),
                                                float(a[4])])))

        for i in range(nedges):
            a = f.readline().split()
            self.el.append(DCELEdge(int(a[0]),int(a[1]), None, None, None, a[8]))
            self.el[-1].x1 = np.array([float(a[2]),float(a[3]),float(a[4])])
            self.el[-1].x2 = np.array([float(a[5]),float(a[6]),float(a[7])])

        for i in range(nfaces):
            a = f.readline().split()
            self.faceInfo.append(a[0])

        f.close()

#Misc. functions
def area2(hedge, point):
    """Determines the area of the triangle formed by a hedge and
    an external point"""

    pa = hedge.twin.origin
    pb = hedge.origin
    pc = point
    return (pb.x - pa.x)*(pc[1] - pa.y) - (pc[0] - pa.x)*(pb.y - pa.y)


def isLeft(P0, P1, P2):
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

def lefton(hedge, point):
    """Determines if a point is to the left of a hedge"""

    return area2(hedge, point) >= 0


def hangle(dx,dy):
    """Determines the angle with respect to the x axis of a segment
    of coordinates dx and dy
    """

    l = np.sqrt(dx*dx + dy*dy)

    if dy > 0:
        return np.arccos(dx/l)
    else:
        return 2*np.pi - np.arccos(dx/l)

# --------------------- Polygon geometric functions -----------------
def areaPoly(nodes):
    # Return the area of the polygon. Note that the input need not be
    # strictly a polygon, (closed curve in 2 dimensions.) The approach
    # we take here is to find the centroid, then sum the area of the
    # 3d triangles. THIS APPROACH ONLY WORKS FOR CONVEX POLYGONS!

    c = np.average(nodes, axis=0)
    area = 0.0
    for ii in range(len(nodes)):
        xi = nodes[ii]
        xip1 = nodes[np.mod(ii+1, len(nodes))]
        area = area + 0.5*np.linalg.norm(np.cross(xi-c,xip1-c))

    return np.abs(area)

def volumePoly(lowerNodes, upperNodes):
    # Compute the volume of a 'polyhedreon' defined by a loop of nodes
    # on the 'bottom' and a loop on the 'top'. Like areaPoly, we use
    # the centroid to split the polygon into triangles. THIS ONLY
    # WORKS FOR CONVEX POLYGONS

    lc = np.average(lowerNodes, axis=0)
    uc = np.average(upperNodes, axis=0)
    volume = 0.0
    ln = len(lowerNodes)
    n =  np.zeros((6,3))
    for ii in range(len(lowerNodes)):
        # Each "wedge" can be sub-divided to 3 tetrahedra

        # The following indices define the decomposition into three
        # tetra hedrea

        # 3 5 4 1
        # 5 2 1 0
        # 0 3 1 5

        #      3 +-----+ 5
        #        |\   /|
        #        | \ / |
        #        |  + 4|
        #        |  |  |
        #      0 +--|--+ 2
        #        \  |  /
        #         \ | /
        #          \|/
        #           + 1
        n[0] = lowerNodes[ii]
        n[1] = lc
        n[2] = lowerNodes[np.mod(ii+1, ln)]

        n[3] = upperNodes[ii]
        n[4] = uc
        n[5] = upperNodes[np.mod(ii+1, ln)]
        volume += volTetra([n[3],n[5],n[4],n[1]])
        volume += volTetra([n[5],n[2],n[1],n[0]])
        volume += volTetra([n[0],n[3],n[1],n[5]])

    return volume

def volTetra(nodes):
    # Compute volume of tetrahedra given by 4 nodes
    a = nodes[1] - nodes[0]
    b = nodes[2] - nodes[0]
    c = nodes[3] - nodes[0]
    # Scalar triple product
    V = (1.0/6.0)*np.linalg.norm(np.dot(a, np.cross(b,c)))

    return V
