import numpy as np

# --------------------------------------------------------------
#                  Knot Vector Manipulation Functions
# --------------------------------------------------------------


def blendKnotVectors(knotVectors, sym):
    """Take in a list of knot vectors and average them"""

    nVec = len(knotVectors)

    if sym:  # Symmetrize each knot vector first
        for i in range(nVec):
            curKnotVec = knotVectors[i].copy()
            if np.mod(len(curKnotVec), 2) == 1:  # its odd
                mid = (len(curKnotVec) - 1) // 2
                beg1 = curKnotVec[0:mid]
                beg2 = (1 - curKnotVec[mid + 1 :])[::-1]
                # Average
                beg = 0.5 * (beg1 + beg2)
                curKnotVec[0:mid] = beg
                curKnotVec[mid + 1 :] = (1 - beg)[::-1]
                curKnotVec[mid] = 0.5
            else:  # its even
                mid = len(curKnotVec) // 2
                beg1 = curKnotVec[0:mid]
                beg2 = (1 - curKnotVec[mid:])[::-1]
                beg = 0.5 * (beg1 + beg2)
                curKnotVec[0:mid] = beg
                curKnotVec[mid:] = (1 - beg)[::-1]

            knotVectors[i] = curKnotVec

    # Now average them all
    newKnotVec = np.zeros(len(knotVectors[0]))
    for i in range(nVec):
        newKnotVec += knotVectors[i]

    newKnotVec = newKnotVec / nVec
    return newKnotVec
