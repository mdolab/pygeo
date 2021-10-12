import numpy as np

# --------------------------------------------------------------
#                Polygon geometric functions
# --------------------------------------------------------------


def areaPoly(nodes):
    # Return the area of the polygon. Note that the input need not be
    # strictly a polygon, (closed curve in 2 dimensions.) The approach
    # we take here is to find the centroid, then sum the area of the
    # 3d triangles. THIS APPROACH ONLY WORKS FOR CONVEX POLYGONS!

    c = np.average(nodes, axis=0)
    area = 0.0
    for ii in range(len(nodes)):
        xi = nodes[ii]
        xip1 = nodes[np.mod(ii + 1, len(nodes))]
        area = area + 0.5 * np.linalg.norm(np.cross(xi - c, xip1 - c))

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
    n = np.zeros((6, 3))
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
        n[2] = lowerNodes[np.mod(ii + 1, ln)]

        n[3] = upperNodes[ii]
        n[4] = uc
        n[5] = upperNodes[np.mod(ii + 1, ln)]
        volume += volTetra([n[3], n[5], n[4], n[1]])
        volume += volTetra([n[5], n[2], n[1], n[0]])
        volume += volTetra([n[0], n[3], n[1], n[5]])

    return volume


def volTetra(nodes):
    # Compute volume of tetrahedra given by 4 nodes
    a = nodes[1] - nodes[0]
    b = nodes[2] - nodes[0]
    c = nodes[3] - nodes[0]
    # Scalar triple product
    V = (1.0 / 6.0) * np.linalg.norm(np.dot(a, np.cross(b, c)))

    return V
