# External modules
import numpy as np

# Local modules
from . import euclideanNorm

# --------------------------------------------------------------
#                Polygon geometric functions
# --------------------------------------------------------------


def areaTri(p0, p1, p2):
    """
    compute area based on three point arrays
    """
    # convert p1 and p2 to v1 and v2
    v1 = p1 - p0
    v2 = p2 - p0

    # compute the areas
    areaVec = np.cross(v1, v2)

    # area = np.linalg.norm(areaVec,axis=1)
    area = 0
    for i in range(len(areaVec)):
        area += euclideanNorm(areaVec[i, :])

    # return np.sum(area)/2.0
    return area / 2.0


def areaPoly(nodes):
    """Return the area of the polygon.
    Note that the input need not be strictly a polygon (closed curve in 2 dimensions).
    The approach we take here is to find the centroid, then sum the area of the
    3d triangles.

    .. warning:: This approach only works for convex polygons
    """

    c = np.average(nodes, axis=0)
    area = 0.0
    for ii in range(len(nodes)):
        xi = nodes[ii]
        xip1 = nodes[np.mod(ii + 1, len(nodes))]
        area = area + 0.5 * np.linalg.norm(np.cross(xi - c, xip1 - c))

    return np.abs(area)


def volumePoly(lowerNodes, upperNodes):
    """
    Compute the volume of a 'polyhedron' defined by a loop of nodes
    on the 'bottom' and a loop on the 'top'.
    Like areaPoly, we use the centroid to split the polygon into triangles.

    .. warning:: This only works for convex polygons
    """

    lc = np.average(lowerNodes, axis=0)
    uc = np.average(upperNodes, axis=0)
    volume = 0.0
    ln = len(lowerNodes)
    n = np.zeros((6, 3))
    for ii in range(len(lowerNodes)):
        # Each "wedge" can be sub-divided to 3 tetrahedra

        # The following indices define the decomposition into three tetrahedra

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
        n[2] = lowerNodes[np.mod(ii + 1, ln)]

        n[3] = upperNodes[ii]
        n[4] = uc
        n[5] = upperNodes[np.mod(ii + 1, ln)]
        volume += volumeTetra([n[3], n[5], n[4], n[1]])
        volume += volumeTetra([n[5], n[2], n[1], n[0]])
        volume += volumeTetra([n[0], n[3], n[1], n[5]])

    return volume


def volumeTetra(nodes):
    """
    Compute volume of tetrahedra given by 4 nodes
    """
    a = nodes[1] - nodes[0]
    b = nodes[2] - nodes[0]
    c = nodes[3] - nodes[0]
    # Scalar triple product
    V = (1.0 / 6.0) * np.linalg.norm(np.dot(a, np.cross(b, c)))

    return V


def volumePyramid(a, b, c, d, p):
    """
    Compute volume of a square-based pyramid
    """
    fourth = 1.0 / 4.0

    volume = (
        (p[0] - fourth * (a[0] + b[0] + c[0] + d[0])) * ((a[1] - c[1]) * (b[2] - d[2]) - (a[2] - c[2]) * (b[1] - d[1]))
        + (p[1] - fourth * (a[1] + b[1] + c[1] + d[1]))
        * ((a[2] - c[2]) * (b[0] - d[0]) - (a[0] - c[0]) * (b[2] - d[2]))
        + (p[2] - fourth * (a[2] + b[2] + c[2] + d[2]))
        * ((a[0] - c[0]) * (b[1] - d[1]) - (a[1] - c[1]) * (b[0] - d[0]))
    )

    return volume


def volumePyramid_b(a, b, c, d, p, ab, bb, cb, db, pb):
    """
    Compute the reverse-mode derivative of the square-based
    pyramid. This has been copied from reverse-mode AD'ed tapenade
    fortran code and converted to python to use vectors for the
    points.
    """
    fourth = 1.0 / 4.0
    volpymb = 1.0
    tempb = ((a[1] - c[1]) * (b[2] - d[2]) - (a[2] - c[2]) * (b[1] - d[1])) * volpymb
    tempb0 = -(fourth * tempb)
    tempb1 = (p[0] - fourth * (a[0] + b[0] + c[0] + d[0])) * volpymb
    tempb2 = (b[2] - d[2]) * tempb1
    tempb3 = (a[1] - c[1]) * tempb1
    tempb4 = -((b[1] - d[1]) * tempb1)
    tempb5 = -((a[2] - c[2]) * tempb1)
    tempb6 = ((a[2] - c[2]) * (b[0] - d[0]) - (a[0] - c[0]) * (b[2] - d[2])) * volpymb
    tempb7 = -(fourth * tempb6)
    tempb8 = (p[1] - fourth * (a[1] + b[1] + c[1] + d[1])) * volpymb
    tempb9 = (b[0] - d[0]) * tempb8
    tempb10 = (a[2] - c[2]) * tempb8
    tempb11 = -((b[2] - d[2]) * tempb8)
    tempb12 = -((a[0] - c[0]) * tempb8)
    tempb13 = ((a[0] - c[0]) * (b[1] - d[1]) - (a[1] - c[1]) * (b[0] - d[0])) * volpymb
    tempb14 = -(fourth * tempb13)
    tempb15 = (p[2] - fourth * (a[2] + b[2] + c[2] + d[2])) * volpymb
    tempb16 = (b[1] - d[1]) * tempb15
    tempb17 = (a[0] - c[0]) * tempb15
    tempb18 = -((b[0] - d[0]) * tempb15)
    tempb19 = -((a[1] - c[1]) * tempb15)
    pb[0] = pb[0] + tempb
    ab[0] = ab[0] + tempb16 + tempb11 + tempb0
    bb[0] = bb[0] + tempb19 + tempb10 + tempb0
    cb[0] = cb[0] + tempb0 - tempb11 - tempb16
    db[0] = db[0] + tempb0 - tempb10 - tempb19
    ab[1] = ab[1] + tempb18 + tempb7 + tempb2
    cb[1] = cb[1] + tempb7 - tempb18 - tempb2
    bb[2] = bb[2] + tempb14 + tempb12 + tempb3
    db[2] = db[2] + tempb14 - tempb12 - tempb3
    ab[2] = ab[2] + tempb14 + tempb9 + tempb4
    cb[2] = cb[2] + tempb14 - tempb9 - tempb4
    bb[1] = bb[1] + tempb17 + tempb7 + tempb5
    db[1] = db[1] + tempb7 - tempb17 - tempb5
    pb[1] = pb[1] + tempb6
    pb[2] = pb[2] + tempb13


def volumeHex(x0, x1, x2, x3, x4, x5, x6, x7):
    """
    Evaluate the volume of the hexahedral volume defined by the
    the 8 corners.

    Parameters
    ----------
    x{0:7} : arrays or size (3)
        Array of defining the coordinates of the volume
    """

    p = np.average([x0, x1, x2, x3, x4, x5, x6, x7], axis=0)
    V = 0.0
    V += volumePyramid(x0, x1, x3, x2, p)
    V += volumePyramid(x0, x2, x6, x4, p)
    V += volumePyramid(x0, x4, x5, x1, p)
    V += volumePyramid(x1, x5, x7, x3, p)
    V += volumePyramid(x2, x3, x7, x6, p)
    V += volumePyramid(x4, x6, x7, x5, p)
    V /= 6.0

    return V


def volumeHex_b(x0, x1, x2, x3, x4, x5, x6, x7, x0b, x1b, x2b, x3b, x4b, x5b, x6b, x7b):
    """
    Evaluate the derivative of the volume defined by the 8
    coordinates in the array x.

    Parameters
    ----------
    x{0:7} : arrays of len 3
        Arrays of defining the coordinates of the volume

    Returns
    -------
    xb{0:7} : arrays of len 3
        Derivatives of the volume wrt the points.
    """

    p = np.average([x0, x1, x2, x3, x4, x5, x6, x7], axis=0)
    pb = np.zeros(3)
    volumePyramid_b(x0, x1, x3, x2, p, x0b, x1b, x3b, x2b, pb)
    volumePyramid_b(x0, x2, x6, x4, p, x0b, x2b, x6b, x4b, pb)
    volumePyramid_b(x0, x4, x5, x1, p, x0b, x4b, x5b, x1b, pb)
    volumePyramid_b(x1, x5, x7, x3, p, x1b, x5b, x7b, x3b, pb)
    volumePyramid_b(x2, x3, x7, x6, p, x2b, x3b, x7b, x6b, pb)
    volumePyramid_b(x4, x6, x7, x5, p, x4b, x6b, x7b, x5b, pb)

    pb /= 8.0
    x0b += pb
    x1b += pb
    x2b += pb
    x3b += pb
    x4b += pb
    x5b += pb
    x6b += pb
    x7b += pb


def volumeTriangulatedMesh(p0, p1, p2):
    """
    Compute the volume of a triangulated volume by computing
    the signed areas.

    Parameters
    ----------
    p0, p1, p2 : arrays
        Coordinates of the vertices of the triangulated mesh

    Returns
    -------
    volume : float
        The volume of the triangulated surface

    References
    ----------
    The method is described in, among other places,
    EFFICIENT FEATURE EXTRACTION FOR 2D/3D OBJECTS IN MESH REPRESENTATION
    by Cha Zhang and Tsuhan Chen,
    http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf
    """

    volume = (
        np.sum(
            p1[:, 0] * p2[:, 1] * p0[:, 2]
            + p2[:, 0] * p0[:, 1] * p1[:, 2]
            + p0[:, 0] * p1[:, 1] * p2[:, 2]
            - p1[:, 0] * p0[:, 1] * p2[:, 2]
            - p2[:, 0] * p1[:, 1] * p0[:, 2]
            - p0[:, 0] * p2[:, 1] * p1[:, 2]
        )
        / 6.0
    )

    return volume


def volumeTriangulatedMesh_b(p0, p1, p2):
    """
    Compute the gradients of the volume with respect to
    the mesh vertices.

    Parameters
    ----------
    p0, p1, p2 : arrays
        Coordinates of the vertices of the triangulated mesh

    Returns
    -------
    grad_0, grad_1, grad_2 : arrays
        Gradients of volume with respect to vertex coordinates
    """

    num_pts = p0.shape[0]
    grad_0 = np.zeros((num_pts, 3))
    grad_1 = np.zeros((num_pts, 3))
    grad_2 = np.zeros((num_pts, 3))

    grad_0[:, 0] = p1[:, 1] * p2[:, 2] - p1[:, 2] * p2[:, 1]
    grad_0[:, 1] = p1[:, 2] * p2[:, 0] - p1[:, 0] * p2[:, 2]
    grad_0[:, 2] = p1[:, 0] * p2[:, 1] - p1[:, 1] * p2[:, 0]

    grad_1[:, 0] = p0[:, 2] * p2[:, 1] - p0[:, 1] * p2[:, 2]
    grad_1[:, 1] = p0[:, 0] * p2[:, 2] - p0[:, 2] * p2[:, 0]
    grad_1[:, 2] = p0[:, 1] * p2[:, 0] - p0[:, 0] * p2[:, 1]

    grad_2[:, 0] = p0[:, 1] * p1[:, 2] - p0[:, 2] * p1[:, 1]
    grad_2[:, 1] = p0[:, 2] * p1[:, 0] - p0[:, 0] * p1[:, 2]
    grad_2[:, 2] = p0[:, 0] * p1[:, 1] - p0[:, 1] * p1[:, 0]

    return grad_0 / 6.0, grad_1 / 6.0, grad_2 / 6.0
