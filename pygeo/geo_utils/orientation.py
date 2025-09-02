# Standard Python modules
import sys

# External modules
import numpy as np

# Local modules
from .norm import eDist

# --------------------------------------------------------------
#                Orientation Functions
# --------------------------------------------------------------


def edgeOrientation(e1, e2):
    """Compare two edge orientations. Basically if the two nodes are
    in the same order return 1 if they are in opposite order, return
    1
    """

    if [e1[0], e1[1]] == [e2[0], e2[1]]:
        return 1
    elif [e1[1], e1[0]] == [e2[0], e2[1]]:
        return -1
    else:
        print("Error with edgeOrientation: Not possible.")
        print("Orientation 1 [%d %d]" % (e1[0], e1[1]))
        print("Orientation 2 [%d %d]" % (e2[0], e2[1]))
        sys.exit(0)


def faceOrientation(f1, f2):
    """Compare two face orientations f1 and f2 and return the
    transform to get f1 back to f2
    """

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
        print("Error with faceOrientation: Not possible.")
        print("Orientation 1 [%d %d %d %d]" % (f1[0], f1[1], f1[2], f1[3]))
        print("Orientation 2 [%d %d %d %d]" % (f2[0], f2[1], f2[2], f2[3]))
        sys.exit(0)


def quadOrientation(pt1, pt2):
    """Given two sets of 4 points in ndim space, pt1 and pt2,
    determine the orientation of pt2 wrt pt1
    This works for both exact quads and "loosely" oriented quads
    .
    """
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
            return 0  # U same direction
        else:
            return 1  # U opposite direction
    else:
        if vDotTot >= 0:
            return 2  # V same direction
        else:
            return 3  # V opposite direction


def curveDirection(curve1, curve2):
    """Determine if the direction of curve 1 is basically in the same
    direction as curve2. Return 1 for same direction, -1 for opposite
    direction
    """

    N = 4
    s = np.linspace(0, 1, N)
    tot = 0
    dForward = 0
    dBackward = 0
    for i in range(N):
        tot += np.dot(curve1.getDerivative(s[i]), curve2.getDerivative(s[i]))
        dForward += eDist(curve1.getValue(s[i]), curve2.getValue(s[i]))
        dBackward += eDist(curve1.getValue(s[i]), curve2.getValue(s[N - i - 1]))

    if tot > 0:
        return tot, dForward
    else:
        return tot, dBackward
