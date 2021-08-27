import numpy as np


def getBiLinearMap(edge0, edge1, edge2, edge3):
    """Get the UV coordinates on a square defined from spacing on the edges"""

    if len(edge0) != len(edge1):
        raise ValueError("getBiLinearMap: The len of edge0 and edge1 are not the same")
    if len(edge2) != len(edge3):
        raise ValueError("getBiLinearMap: The len of edge2 and edge3 are no the same")

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

    for i in range(1, N - 1):
        x1 = edge0[i]
        y1 = 0.0

        x2 = edge1[i]
        y2 = 1.0

        for j in range(1, M - 1):
            x3 = 0
            y3 = edge2[j]
            x4 = 1.0
            y4 = edge3[j]
            UV[i, j] = calcIntersection(x1, y1, x2, y2, x3, y3, x4, y4)

    return UV


def calcIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calc the intersection between two line segments defined by
    # (x1,y1) to (x2,y2) and (x3,y3) to (x4,y4)

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    xi = x1 + ua * (x2 - x1)
    yi = y1 + ua * (y2 - y1)

    return xi, yi
