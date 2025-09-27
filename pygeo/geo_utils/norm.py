# External modules
import numpy as np

# -------------------------------------------------------------
#               Norm Functions
# -------------------------------------------------------------


def euclideanNorm(inVec):
    """
    Perform the euclidean 2 norm of the vector inVec
    required because linalg.norm() provides incorrect results for
    CS derivatives.
    """
    inVec = np.array(inVec)
    return np.sqrt(inVec.dot(inVec))


def cross_b(a, b, crossb):
    """
    Do the reverse accumulation through a cross product.
    """
    ab = np.zeros_like(a)
    bb = np.zeros_like(b)

    ab[0] = ab[0] + b[1] * crossb[2]
    bb[1] = bb[1] + a[0] * crossb[2]
    ab[1] = ab[1] - b[0] * crossb[2]
    bb[0] = bb[0] - a[1] * crossb[2]

    ab[2] = ab[2] + b[0] * crossb[1]
    bb[0] = bb[0] + a[2] * crossb[1]
    ab[0] = ab[0] - b[2] * crossb[1]
    bb[2] = bb[2] - a[0] * crossb[1]

    ab[1] = ab[1] + b[2] * crossb[0]
    bb[2] = bb[2] + a[1] * crossb[0]
    ab[2] = ab[2] - b[1] * crossb[0]
    bb[1] = bb[1] - a[2] * crossb[0]

    return ab, bb


def dot_b(a, b, dotb):
    """
    Do the reverse accumulation through a dot product.
    """
    ab = np.zeros_like(a)
    bb = np.zeros_like(b)

    ab = b * dotb
    bb = a * dotb

    return ab, bb


def calculateCentroid(p0, v1, v2):
    """
    Take in a triangulated surface and calculate the centroid
    """
    p0 = np.array(p0)
    p1 = np.array(v1) + p0
    p2 = np.array(v2) + p0

    # compute the areas
    areaVec = np.cross(v1, v2) / 2.0
    area = np.linalg.norm(areaVec, axis=1)

    # compute the cell centroids
    cellCenter = (p0 + p1 + p2) / 3.0

    centerSum = area.dot(cellCenter)
    areaSum = np.sum(area)

    centroid = centerSum / areaSum

    return centroid


def calculateAverageNormal(p0, v1, v2):
    """
    Take in a triangulated surface and calculate the centroid
    """
    p0 = np.array(p0)

    # compute the normal of each triangle
    normal = np.cross(v1, v2)
    sumNormal = np.sum(normal, axis=0)
    lengthNorm = np.linalg.norm(sumNormal)

    unitNormal = sumNormal / lengthNorm

    return unitNormal


def calculateRadii(centroid, p0, v1, v2):
    """
    Take the centroid and compute inner and outer radii of surface
    """
    p0 = np.array(p0)
    p1 = np.array(v1) + p0
    p2 = np.array(v2) + p0

    # take the difference between the points and the centroid
    d0 = p0 - centroid
    d1 = p1 - centroid
    d2 = p2 - centroid

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

    return innerRadius, outerRadius


def computeDistToAxis(origin, coords, axis, dtype="d"):
    """
    Compute the distance of coords from the defined axis.
    """
    # Compute the direction from each point to the origin
    dirVec = origin - coords

    # compute the cross product with the desired axis. Cross product
    # will be zero if the direction vector is the same as the axis
    resultDir = np.cross(axis, dirVec)

    X = np.zeros(len(coords), dtype)
    for i in range(len(resultDir)):
        X[i] = euclideanNorm(resultDir[i, :])

    return X


# --------------------------------------------------------------
#            Edge distance Function
# --------------------------------------------------------------


def eDist(x1, x2):
    """Get the eculidean distance between two points"""
    return euclideanNorm(x1 - x2)  # np.linalg.norm(x1-x2)


def eDist2D(x1, x2):
    """Get the eculidean distance between two points"""
    return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)


def eDist_b(x1, x2):
    x1b = 0.0
    x2b = 0.0
    db = 1.0
    x1b = np.zeros(3)
    x2b = np.zeros(3)
    if (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2 + (x1[2] - x2[2]) ** 2 == 0.0:
        tempb = 0.0
    else:
        tempb = db / (2.0 * np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2 + (x1[2] - x2[2]) ** 2))

    tempb0 = 2 * (x1[0] - x2[0]) * tempb
    tempb1 = 2 * (x1[1] - x2[1]) * tempb
    tempb2 = 2 * (x1[2] - x2[2]) * tempb
    x1b[0] = tempb0
    x2b[0] = -tempb0
    x1b[1] = tempb1
    x2b[1] = -tempb1
    x1b[2] = tempb2
    x2b[2] = -tempb2

    return x1b, x2b
