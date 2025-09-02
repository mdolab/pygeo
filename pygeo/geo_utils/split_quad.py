# External modules
import numpy as np
from pyspline.utils import tfi2d

# Local modules
from .norm import eDist, euclideanNorm


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
    length = np.zeros(4)
    length[0] = eDist(pts[0], pts[1])
    length[1] = eDist(pts[2], pts[3])
    length[2] = eDist(pts[0], pts[2])
    length[3] = eDist(pts[1], pts[3])

    # Vector along edges 0->3
    vec = np.zeros((4, 3))
    vec[0] = pts[1] - pts[0]
    vec[1] = pts[3] - pts[2]
    vec[2] = pts[2] - pts[0]
    vec[3] = pts[3] - pts[1]

    U = 0.5 * (vec[0] + vec[1])
    V = 0.5 * (vec[2] + vec[3])
    u = U / euclideanNorm(U)
    v = V / euclideanNorm(V)

    mid = np.average(pts, axis=0)

    uBar = 0.5 * (length[0] + length[1]) * alpha
    vBar = 0.5 * (length[2] + length[3]) * beta

    aspect = uBar / vBar

    if aspect < 1:  # its higher than wide, logically rotate the element
        v, u = u, -v
        vBar, uBar = uBar, vBar
        alpha, beta = beta, alpha
        Nv, Nu = Nu, Nv

        E0 = e2[::-1, :].copy()
        E1 = e3[::-1, :].copy()
        E2 = e1.copy()
        E3 = e0.copy()

        # Also need to permute points
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
    P0 = np.zeros((Nu, 4, 3), "d")
    P1 = np.zeros((Nu, 4, 3), "d")
    P2 = np.zeros((Nv, 4, 3), "d")
    P3 = np.zeros((Nv, 4, 3), "d")

    rad = vBar * beta
    rectLen = uBar - 2 * rad
    if rectLen < 0:
        rectLen = 0.0

    # Determine 4 corners of rectangular part
    rectCorners[0] = mid - u * (rectLen / 2) - np.sin(np.pi / 4) * rad * v - np.cos(np.pi / 4) * rad * u
    rectCorners[1] = mid + u * (rectLen / 2) - np.sin(np.pi / 4) * rad * v + np.cos(np.pi / 4) * rad * u
    rectCorners[2] = mid - u * (rectLen / 2) + np.sin(np.pi / 4) * rad * v - np.cos(np.pi / 4) * rad * u
    rectCorners[3] = mid + u * (rectLen / 2) + np.sin(np.pi / 4) * rad * v + np.cos(np.pi / 4) * rad * u

    arcLen = np.pi * rad / 2 + rectLen  # Two quarter circles straight line
    eighthArc = 0.25 * np.pi * rad
    # We have to distribute Nu-2 nodes over this arc-length
    spacing = arcLen / (Nu - 1)

    botEdge = np.zeros((Nu, 3), "d")
    topEdge = np.zeros((Nu, 3), "d")
    botEdge[0] = rectCorners[0]
    botEdge[-1] = rectCorners[1]
    topEdge[0] = rectCorners[2]
    topEdge[-1] = rectCorners[3]
    for i in range(Nu - 2):
        distAlongArc = (i + 1) * spacing
        if distAlongArc < eighthArc:
            theta = distAlongArc / rad  # Angle in radians
            botEdge[i + 1] = (
                mid - u * (rectLen / 2) - np.sin(theta + np.pi / 4) * rad * v - np.cos(theta + np.pi / 4) * rad * u
            )
            topEdge[i + 1] = (
                mid - u * (rectLen / 2) + np.sin(theta + np.pi / 4) * rad * v - np.cos(theta + np.pi / 4) * rad * u
            )
        elif distAlongArc > rectLen + eighthArc:
            theta = (distAlongArc - rectLen - eighthArc) / rad
            botEdge[i + 1] = mid + u * (rectLen / 2) + np.sin(theta) * rad * u - np.cos(theta) * rad * v
            topEdge[i + 1] = mid + u * (rectLen / 2) + np.sin(theta) * rad * u + np.cos(theta) * rad * v
        else:
            topEdge[i + 1] = mid - u * rectLen / 2 + rad * v + (distAlongArc - eighthArc) * u
            botEdge[i + 1] = mid - u * rectLen / 2 - rad * v + (distAlongArc - eighthArc) * u

    leftEdge = np.zeros((Nv, 3), "d")
    rightEdge = np.zeros((Nv, 3), "d")
    theta = np.linspace(-np.pi / 4, np.pi / 4, Nv)

    for i in range(Nv):
        leftEdge[i] = mid - u * (rectLen / 2) + np.sin(theta[i]) * rad * v - np.cos(theta[i]) * rad * u
        rightEdge[i] = mid + u * (rectLen / 2) + np.sin(theta[i]) * rad * v + np.cos(theta[i]) * rad * u

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


def tfi_2d(e0, e1, e2, e3):
    # Input
    # e0: Nodes along edge 0. Size Nu x 3
    # e1: Nodes along edge 1. Size Nu x 3
    # e0: Nodes along edge 2. Size Nv x 3
    # e1: Nodes along edge 3. Size Nv x 3

    try:
        X = tfi2d(e0.T, e1.T, e2.T, e3.T).T
    except Exception:
        Nu = len(e0)
        Nv = len(e2)
        if Nu != len(e1):
            raise ValueError(f"Number of nodes on edge0 and edge1 are not the same: {len(e0)} {len(e1)}") from None
        if Nv != len(e3):
            raise ValueError(f"Number of nodes on edge2 and edge3 are not the same: {len(e2)} {len(e3)}") from None

        U = np.linspace(0, 1, Nu)
        V = np.linspace(0, 1, Nv)

        X = np.zeros((Nu, Nv, 3))

        for i in range(Nu):
            for j in range(Nv):
                X[i, j] = (
                    (1 - V[j]) * e0[i]
                    + V[j] * e1[i]
                    + (1 - U[i]) * e2[j]
                    + U[i] * e3[j]
                    - (
                        U[i] * V[j] * e1[-1]
                        + U[i] * (1 - V[j]) * e0[-1]
                        + V[j] * (1 - U[i]) * e1[0]
                        + (1 - U[i]) * (1 - V[j]) * e0[0]
                    )
                )
    return X


def linearEdge(pt1, pt2, N):
    # Return N points along line from pt1 to pt2
    pts = np.zeros((N, len(pt1)))
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    for i in range(N):
        pts[i] = float(i) / (N - 1) * (pt2 - pt1) + pt1
    return pts
