# Standard Python modules
import sys

# External modules
import numpy as np
from pyspline.utils import line_plane

# Local modules
from .remove_duplicates import pointReduce

# --------------------------------------------------------------
#                Projection functions
# --------------------------------------------------------------


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

    tmpSol, tmpPid, nSol = line_plane(pt, upVec, p0.T, v1.T, v2.T)
    tmpSol = tmpSol.T
    tmpPid -= 1

    # Check to see if any of the solutions happen be identical.
    points = []
    for i in range(nSol):
        points.append(tmpSol[i, 3:6])

    if nSol > 1:
        _, link = pointReduce(points, nodeTol=1e-12)
        nUnique = np.max(link) + 1
        points = np.zeros((nUnique, 3))
        uu = np.zeros(nUnique)
        vv = np.zeros(nUnique)
        ss = np.zeros(nUnique)
        pid = np.zeros(nUnique, "intc")

        for i in range(nSol):
            points[link[i]] = tmpSol[i, 3:6]
            uu[link[i]] = tmpSol[i, 1]
            vv[link[i]] = tmpSol[i, 2]
            ss[link[i]] = tmpSol[i, 0]
            pid[link[i]] = tmpPid[i]

        nSol = len(points)
    else:
        nUnique = 1
        points = np.zeros((nUnique, 3))
        uu = np.zeros(nUnique)
        vv = np.zeros(nUnique)
        ss = np.zeros(nUnique)
        pid = np.zeros(nUnique, "intc")

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

        first = points[0]
        firstPatchID = PID[pid[0]]
        firstU = uv0[pid[0]][0] + uu[0] * (uv1[pid[0]][0] - uv0[pid[0]][0])
        firstV = uv0[pid[0]][1] + vv[0] * (uv2[pid[0]][1] - uv0[pid[0]][1])
        firstS = ss[0]
        return [first, firstPatchID, firstU, firstV, firstS], None, fail
    elif nSol == 2:
        fail = 0

        # Determine the 'top' and 'bottom' solution
        first = points[0]
        second = points[1]

        firstPatchID = PID[pid[0] - 1]
        secondPatchID = PID[pid[1] - 1]

        firstU = uv0[pid[0]][0] + uu[0] * (uv1[pid[0]][0] - uv0[pid[0]][0])
        firstV = uv0[pid[0]][1] + vv[0] * (uv2[pid[0]][1] - uv0[pid[0]][1])
        firstS = ss[0]

        secondU = uv0[pid[1]][0] + uu[1] * (uv1[pid[1]][0] - uv0[pid[1]][0])
        secondV = uv0[pid[1]][1] + vv[1] * (uv2[pid[1]][1] - uv0[pid[1]][1])
        secondS = ss[1]

        if np.dot(first - pt, upVec) >= np.dot(second - pt, upVec):

            return (
                [first, firstPatchID, firstU, firstV, firstS],
                [second, secondPatchID, secondU, secondV, secondS],
                fail,
            )
        else:
            return (
                [second, secondPatchID, secondU, secondV, secondS],
                [first, firstPatchID, firstU, firstV, firstS],
                fail,
            )

    else:
        print("This functionality is not implemtned in geoUtils yet")
        sys.exit(1)


def projectNodePIDPosOnly(pt, upVec, p0, v1, v2, uv0, uv1, uv2, PID):

    # Get the bounds of the geo object so we know what to scale by

    fail = 0
    if p0.shape[0] == 0:
        fail = 1
        return None, fail

    sol, pid, nSol = line_plane(pt, upVec, p0.T, v1.T, v2.T)
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
            u = uv0[pid[minIndex]][0] + sol[minIndex, 1] * (uv1[pid[minIndex]][0] - uv0[pid[minIndex]][0])

            v = uv0[pid[minIndex]][1] + sol[minIndex, 2] * (uv2[pid[minIndex]][1] - uv0[pid[minIndex]][1])
            s = sol[minIndex, 0]
            tmp = [sol[minIndex, 3], sol[minIndex, 4], sol[minIndex, 5]]
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

    sol, _, nSol = line_plane(pt, upVec, p0.T, v1.T, v2.T)
    sol = sol.T

    # Check to see if any of the solutions happen be identical.
    if nSol > 1:
        points = []
        for i in range(nSol):
            points.append(sol[i, 3:6])

        newPoints, _ = pointReduce(points, nodeTol=1e-12)
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
        return newPoints[0], None, fail
    elif nSol == 2:
        fail = 0

        # Determine the 'top' and 'bottom' solution
        first = newPoints[0]
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

    sol, _, nSol = line_plane(pt, upVec, p0.T, v1.T, v2.T)
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
