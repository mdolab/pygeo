from __future__ import print_function
from __future__ import division
# =============================================================================
# Utility Functions for Use in pyNetwork, pyGeo, pyBlock, DVGeometry,
# and pyLayout
# =============================================================================
import numpy as np
from scipy.optimize import fsolve

from pyspline import pySpline


# this functions are already defined in another place, so I'm going to leave them there as not to
# break anything
from geo_utils import readAirfoilFile, writeAirfoilFile


# --------------------------------------------------------------
#             IO Functions
# --------------------------------------------------------------
# some io functions for airfoils are defined in geo_utils, but those don't work in the way we need
# and


# --------------------------------------------------------------
#             Transformation Functions
# --------------------------------------------------------------


def translateCoords(x, y, dx=0., dy=0.):
    """shifts the input coordinates by dx and dy"""
    x = np.array(x) + dx
    y = np.array(y) + dy

    return x, y


def rotateCoords(x, y, angle, rotationPoint=None):
    """retruns the coordinates rotated about the rotationPoint by angle (in deg)"""

    # Define 2D postion vector
    if not rotationPoint == 'None':
        x_avg = rotationPoint[0]
        y_avg = rotationPoint[1]
    else:
        x_avg = np.mean(x)
        y_avg = np.mean(y)
        print (x_avg, y_avg)

    R = np.sqrt((x - x_avg)**2 + (y - y_avg)**2)
    ang = np.arctan2((y - y_avg), (x - x_avg)) + float(angle) * np.pi / 180

    x_new = np.cos(ang) * R + x_avg
    y_new = np.sin(ang) * R + y_avg

    return x_new, y_new


def scaleCoords(x, y, scale):
    """scales the coordinates in both dimension by the scaling factor"""
    x = np.array(x) * scale
    y = np.array(y) * scale
    return x, y


# def bluntTE():
#     npt = len(x)

#     xMin = min(x)
#     xMax = max(x)
#     # Since we will be rescaling the TE regardless, the sharp TE
#     # case and the case where the TE is already blunt can be
#     # handled in the same manner

#     # Get the current thickness
#     curThick = y[0] - y[-1]

#     # Set the new TE values:
#     xBreak = 1.0 - bluntTaperRange

#     # Rescale upper surface:
#     for i in range(0, npt // 2):
#         if x[i] > xBreak:
#             s = (x[i] - xMin - xBreak) / bluntTaperRange
#             y[i] += s * 0.5 * (bluntThickness - curThick)

#     # Rescale lower surface:
#     for i in range(npt // 2, npt):
#         if x[i] > xBreak:
#             s = (x[i] - xMin - xBreak) / bluntTaperRange
#             y[i] -= s * 0.5 * (bluntThickness - curThick)


# def sharpenTE():

#     # There are 4 possibilites we have to deal with:
#     # a. Given a sharp TE -- User wants a sharp TE
#     # b. Given a sharp TE -- User wants a blunt TE
#     # c. Given a blunt TE -- User wants a sharp TE
#     # d. Given a blunt TE -- User wants a blunt TE
#     #    (possibly with different TE thickness)

#         # Check for blunt TE:
#         if y[0] != y[-1]:
#             print('Blunt Trailing Edge on airfoil: %s' % (fileName))
#             print('Merging to a point over final %f ...' % (bluntTaperRange))
#             yAvg = 0.5 * (y[0] + y[-1])
#             xAvg = 0.5 * (x[0] + x[-1])
#             yTop = y[0]
#             yBot = y[-1]
#             xTop = x[0]
#             xBot = x[-1]

#             # Indices on the TOP surface of the wing
#             indices = np.where(x[0:npt // 2] >= (1 - bluntTaperRange))[0]
#             for i in range(len(indices)):
#                 fact = (x[indices[i]] - (x[0] - bluntTaperRange)) / bluntTaperRange
#                 y[indices[i]] = y[indices[i]] - fact * (yTop - yAvg)
#                 x[indices[i]] = x[indices[i]] - fact * (xTop - xAvg)

#             # Indices on the BOTTOM surface of the wing
#             indices = np.where(x[npt // 2:] >= (1 - bluntTaperRange))[0]
#             indices = indices + npt // 2

#             for i in range(len(indices)):
#                 fact = (x[indices[i]] - (x[-1] - bluntTaperRange)) / bluntTaperRange
#                 y[indices[i]] = y[indices[i]] - fact * (yBot - yAvg)
#                 x[indices[i]] = x[indices[i]] - fact * (xBot - xAvg)

#     elif bluntTe is True:
#         # Since we will be rescaling the TE regardless, the sharp TE
#         # case and the case where the TE is already blunt can be
#         # handled in the same manner

#         # Get the current thickness
#         curThick = y[0] - y[-1]

#         # Set the new TE values:
#         xBreak = 1.0-bluntTaperRange

#         # Rescale upper surface:
#         for i in range(0,npt//2):
#             if x[i] > xBreak:
#                 s = (x[i]-xMin-xBreak)/bluntTaperRange
#                 y[i] += s*0.5*(bluntThickness-curThick)

#         # Rescale lower surface:
#         for i in range(npt//2,npt):
#             if x[i] > xBreak:
#                 s = (x[i]-xMin-xBreak)/bluntTaperRange
#                 y[i] -= s*0.5*(bluntThickness-curThick)


# -------------------------------------------------------------
#               generating new Coordinates
# -------------------------------------------------------------


#  ------------ spacing functions for coordinate --------------


def cosSpacing(n, m=np.pi):
    x = np.linspace(0, m, n)
    s = np.cos(x)
    return s / 2 + 0.5


def ellipticalSpacing(n,  b=1,  m=np.pi):
    x = np.linspace(0, m, n)
    s = 1 + b / np.sqrt(np.cos(x)**2 * b**2 + np.sin(x)**2) * np.cos(x)
    return s * b


def parabolicSpacing(n, m=np.pi):
    # angles = np.linspace(0, m, (n + 1) // 2)
    angles = np.linspace(0, m, n)
    # x = np.linspace(1, -1, n)
    s = np.array([])
    for ang in angles:
        if ang <= np.pi / 2:
            s = np.append(s, (-np.tan(ang) + np.sqrt(np.tan(ang)**2 + 4)) / 2)
        else:
            s = np.append(s, (-np.tan(ang) - np.sqrt(np.tan(ang)**2 + 4)) / 2)

    # print 's', s, -1 * s[-2::-1]
    # s = np.append(s, -1 * s[-2::-1])[::-1]
    return s / 2 + 0.5


def polynomialSpacing(n, m=np.pi, order=5):

    def func(x):
        return np.abs(x)**order + np.tan(ang) * x - 1
    angles = np.linspace(0, m, n)

    s = np.array([])
    for ang in angles:
        s = np.append(s, fsolve(func, np.cos(ang))[0])

    return s / 2 + 0.5


def joinedSpacing(n, spacingFunc, s_LE=0.5, equalPts=False, repeat=False, **kwargs):
    """
    function that returns two point distributions joined at s_LE

                        s1                            s2
    || |  |   |    |     |     |    |   |  | |||| |  |   |    |     |     |    |   |  | ||
                                                /\
                                                s_LE

    """
    offset1 = np.pi / (n * s_LE)
    if repeat:
        offset2 = 0
    else:
        offset2 = np.pi / (n * s_LE)

    if equalPts:
        ns1 = n * .5
        ns2 = ns1
    else:
        ns1 = n * s_LE
        ns2 = n * (1 - s_LE)

    s1 = spacingFunc(ns1, m=np.pi - offset1, **kwargs) * s_LE
    s2 = spacingFunc(ns2, m=np.pi - offset2, **kwargs) * (1 - s_LE) + s_LE

    return np.append(s2, s1)[::-1]

# ------------------------- create new coords based upon existing ---------------


def getSLeadingEdge(curve, tol=1e-13):
    """
        find the leading edge of airfoil spline using a binary search
    """
    lower = 0.
    upper = 1.
    sLE = lower + (upper - lower) / 2
    val = curve.getDerivative(sLE)[0]
    while np.abs(val) >= tol:   # use < instead of <=
        # print lower, sLE, upper, val

        if val > 0:
            upper = sLE
        elif val < 0:
            lower = sLE
        elif val == 0:
            break

        sLE = lower + (upper - lower) / 2
        val = curve.getDerivative(sLE)[0]

    return sLE


def genAirfoilCoords(coordinates, numPoints, spacingFunc=polynomialSpacing, findLE=False, **kwargs):
    """
    given a set of airfoil coordinates and a spacing function return a number of points that are
    along a spline formed from a spline of the coordinates

    Parameters
    ----------
    coordinates : array
        coordinates of the airfoil
    numPoints: int
        number of points to evaluate the spline at
    spacingFunc : function
        outputs the parametric positions along the curve to evaluate at. the function signature must
        be spacingFunc(numPoints, s_LE=s_LE ), if findLE is True and spacingFunc(numPoints) if
        findLE is False.
    findLE : bool
        used to toggle finding the leading edge (LE) and clustering pts there

    Returns
    -------
    pts : array  (2 x numPoints)
        points along the airfoil at the locations specified by the number of
        points and the spacing function
    """

    # create spline to of airfoil
    airfoilCurve = pySpline.Curve(X=coordinates, k=3)

    if findLE:
        # fine the leading edge location along the curve
        s_LE = getSLeadingEdge(airfoilCurve)
        s = joinedSpacing(numPoints, spacingFunc, s_LE=s_LE, **kwargs)
    else:
        s = joinedSpacing(numPoints, spacingFunc, **kwargs)

    # print s
    pts = airfoilCurve.getValue(s)

    return pts


# -------------------------------------------------------------
#               Post process slice files
#  -------------------------------------------------------------


def getSliceData(fileName):
    headerLines = 5

    sliceData = {}
    sortedData = {}

    with open(fileName, 'r') as f:
        lines = f.readlines()
        numElem = int(lines[3].split()[4])
        numNodes = int(lines[3].split()[2])
        data = np.genfromtxt(fileName, skip_header=headerLines, skip_footer=(numElem))
        for varIndex, variable in enumerate(lines[1].split()[2:]):
            print variable
            sliceData[variable.replace('"', '')] = data[:, varIndex]
            sortedData[variable.replace('"', '')] = []

    # we have all of the data in a dictionary but now we need to sort it based upon the element connectives

    # get element to node connectivity data
    elem_data = np.genfromtxt(fileName, dtype=int, skip_header=(headerLines + numNodes))

    start = True
    for elem in elem_data:

        # if there is no data for that variable
        if start:
            for variable in ['XoC', 'YoC']:
                sortedData[variable].append(np.array([sliceData[variable][elem[0] - 1]]))

            start = False

        # if the next node is adjacent in the node list, add the data from the node to sortedData
        if elem[1] == (elem[0] + 1):
            for variable in ['XoC', 'YoC']:
                sortedData[variable][-1] = np.append(
                    sortedData[variable][-1], sliceData[variable][elem[1] - 1])

        # if the next node isn't adjacent in the node list, create a new array
        else:

            start = True

    return sliceData
