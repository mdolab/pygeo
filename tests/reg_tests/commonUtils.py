# Standard Python modules
import os

# External modules
import numpy as np
from pyspline import Curve

# First party modules
from pygeo import DVGeometry, DVGeometryAxi


def assert_check_totals(totals, atol=1e-6, rtol=1e-6):
    """
    Check the totals dictionary for the forward and reverse mode derivatives.

    This is better than OpenMDAO's `assert_check_totals` because it uses numpy's `assert_allclose` which eliminates the
    issue of huge relative errors when comparing very small values.
    """
    for key in totals:
        derivs = totals[key]
        ref = derivs["J_fd"]
        if "J_fwd" in derivs:
            np.testing.assert_allclose(
                derivs["J_fwd"],
                ref,
                atol=atol,
                rtol=rtol,
                err_msg=f"Forward derivatives of {key[0]} w.r.t {key[1]} do not match finite difference",
            )
        if "J_rev" in derivs:
            np.testing.assert_allclose(
                derivs["J_rev"],
                ref,
                atol=atol,
                rtol=rtol,
                err_msg=f"Reverse derivatives of {key[0]} w.r.t {key[1]} do not match finite difference",
            )


##################
# DVGeometry Tests
##################


def setupDVGeo(base_path, rotType=None, parentName=None, childName=None):
    # create the Parent FFD
    FFDFile = os.path.join(base_path, "../../input_files/outerBoxFFD.xyz")
    DVGeo = DVGeometry(FFDFile, name=parentName)

    # create a reference axis for the parent
    axisPoints = [[-1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    c1 = Curve(X=axisPoints, k=2)
    if rotType is not None:
        DVGeo.addRefAxis("mainAxis", curve=c1, axis="y", rotType=rotType)

    else:
        DVGeo.addRefAxis("mainAxis", curve=c1, axis="y")

    # create the child FFD
    FFDFile = os.path.join(base_path, "../../input_files/simpleInnerFFD.xyz")
    DVGeoChild = DVGeometry(FFDFile, child=True, name=childName)

    # create a reference axis for the child
    axisPoints = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
    c1 = Curve(X=axisPoints, k=2)
    DVGeoChild.addRefAxis("nestedAxis", curve=c1, axis="y")

    return DVGeo, DVGeoChild


def setupDVGeoD8(base_path, isComplex):
    # create the Parent FFD
    FFDFile = os.path.join(base_path, "../../input_files/bodyFFD.xyz")
    DVGeo = DVGeometry(FFDFile, isComplex=isComplex)

    # create a reference axis for the parent
    axisPoints = [[0.0, 0.0, 0.0], [26.0, 0.0, 0.0], [30.5, 0.0, 0.9], [32.5, 0.0, 1.01], [34.0, 0.0, 0.95]]
    c1 = Curve(X=axisPoints, k=2)
    DVGeo.addRefAxis("mainAxis", curve=c1, axis="y")

    # create the child FFD
    FFDFile = os.path.join(base_path, "../../input_files/nozzleFFD.xyz")
    DVGeoChild = DVGeometry(FFDFile, child=True, isComplex=isComplex)

    # create a reference axis for the child
    axisPoints = [[32.4, 1.0, 1.0], [34, 1.0, 0.9]]
    c1 = Curve(X=axisPoints, k=2)
    DVGeoChild.addRefAxis("nestedAxis", curve=c1, axis="y")

    return DVGeo, DVGeoChild


def setupDVGeoAxi(base_path):
    FFDFile = os.path.join(base_path, "../../input_files/axiTestFFD.xyz")
    DVGeo = DVGeometryAxi(FFDFile, center=(0.0, 0.0, 0.0), collapse_into=("x", "z"))
    axisPoints = [[0, 0.0, 0.0], [0, 0.0, 1.0]]
    c1 = Curve(X=axisPoints, k=2)
    DVGeo.addRefAxis("stretch", curve=c1, axis="z")

    return DVGeo


# define a nested global design variable
def childAxisPoints(val, geo):
    C = geo.extractCoef("nestedAxis")

    # Set the coefficients
    C[0, 0] = val[0]

    geo.restoreCoef(C, "nestedAxis")


# define a nested global design variable
def mainAxisPoints(val, geo):
    C = geo.extractCoef("mainAxis")

    # Set the coefficients
    C[0, 0] = val[0]

    geo.restoreCoef(C, "mainAxis")


# define a nested global design variable
def childAxisPointsD8(val, geo):
    C = geo.extractCoef("nestedAxis")

    # Set the coefficients
    for i in range(len(val)):
        C[i, 0] = val[i]

    geo.restoreCoef(C, "nestedAxis")


# define a nested global design variable
def mainAxisPointsD8(val, geo):
    C = geo.extractCoef("mainAxis")

    # Set the coefficients
    for i in range(len(val)):
        C[i, 0] = val[i]

    geo.restoreCoef(C, "mainAxis")


def mainAxisPointAxi(val, DVgeo):
    C = DVgeo.extractCoef("stretch")
    C[0, 2] = val[0]

    DVgeo.restoreCoef(C, "stretch")


def totalSensitivityFD(DVGeo, nPt, ptName, step=1e-1):
    xDV = DVGeo.getValues()
    refPoints = DVGeo.update(ptName)
    # now get FD Sensitivity
    dIdxFD = {}
    # step = 1e-1#8
    for key in xDV:
        baseVar = xDV[key].copy()
        nDV = len(baseVar)
        dIdxFD[key] = np.zeros([nPt, nDV])
        for i in range(nDV):
            xDV[key][i] = baseVar[i] + step
            DVGeo.setDesignVars(xDV)
            newPoints = DVGeo.update(ptName)

            deriv = (newPoints - refPoints) / step
            dIdxFD[key][:, i] = deriv.flatten()
            # print('Deriv',key, i,deriv)
            xDV[key][i] = baseVar[i]

    return dIdxFD


def totalSensitivityCS(DVGeo, nPt, ptName):
    xDV = DVGeo.getValues()

    # now get CS Sensitivity
    dIdxCS = {}
    step = 1e-40j
    for key in xDV:
        baseVar = xDV[key].copy()
        dIdxCS[key] = np.zeros([nPt, len(baseVar)])
        for i in range(len(baseVar)):
            xDV[key][i] = baseVar[i] + step

            DVGeo.setDesignVars(xDV)
            newPoints = DVGeo.update(ptName)

            deriv = np.imag(newPoints) / np.imag(step)
            dIdxCS[key][:, i] = deriv.flatten()
            # print 'Deriv',key, i,deriv
            xDV[key][i] = baseVar[i]

    # Before we exit make sure we have reset the DVs
    DVGeo.setDesignVars(xDV)

    return dIdxCS


def testSensitivities(DVGeo, refDeriv, handler, pointset=1):
    # create test points
    points = np.zeros([2, 3])
    if pointset == 1:
        points[0, :] = [0.25, 0, 0]
        points[1, :] = [-0.25, 0, 0]
    elif pointset == 2:
        points[0, :] = [0.25, 0.4, 4]
        points[1, :] = [-0.8, 0.2, 7]
    elif pointset == 3:
        points[0, :] = [3.0, 0.0, 3.0]
        points[1, :] = [6.25, 0.0, 9.30]
    else:
        raise Warning("Enter a valid pointset")

    # add points to the geometry object
    ptName = "testPoints"
    DVGeo.addPointSet(points, ptName)

    # generate dIdPt
    nPt = 6
    dIdPt = np.zeros([nPt, 2, 3])
    dIdPt[0, 0, 0] = 1.0
    dIdPt[1, 0, 1] = 1.0
    dIdPt[2, 0, 2] = 1.0
    dIdPt[3, 1, 0] = 1.0
    dIdPt[4, 1, 1] = 1.0
    dIdPt[5, 1, 2] = 1.0
    # get analytic sensitivity
    if refDeriv:
        dIdx = totalSensitivityFD(DVGeo, nPt, ptName)
    else:
        dIdx = DVGeo.totalSensitivity(dIdPt, ptName)

    handler.root_add_dict("dIdx", dIdx, rtol=1e-7, atol=1e-7)


def testSensitivitiesD8(DVGeo, refDeriv, handler):
    # create test points
    nPoints = 50
    points = np.zeros([nPoints, 3])
    for i in range(nPoints):
        nose = 0.01
        tail = 34.0
        delta = (tail - nose) / nPoints
        points[i, :] = [nose + i * delta, 1.0, 0.5]

    # print('points',points)

    # add points to the geometry object
    ptName = "testPoints"
    DVGeo.addPointSet(points, ptName)

    # generate dIdPt
    nPt = nPoints * 3
    dIdPt = np.zeros([nPt, nPoints, 3])
    counter = 0
    for i in range(nPoints):
        for j in range(3):
            dIdPt[counter, i, j] = 1.0
            counter += 1
    # get analytic sensitivity
    if refDeriv:
        # dIdx = totalSensitivityFD(DVGeo,nPt,ptName)
        dIdx = totalSensitivityCS(DVGeo, nPt, ptName)
    else:
        dIdx = DVGeo.totalSensitivity(dIdPt, ptName)

    handler.root_add_dict("dIdx", dIdx, rtol=1e-7, atol=1e-7)


# --- Adding standard twist and single axis scaling functions ---
# These functions are added for Test 24 but could be extended to other tests

fix_root_sect = 1
nRefAxPts = 4


def twist(val, geo):
    axis_key = list(geo.axis.keys())[0]
    for i in range(fix_root_sect, nRefAxPts):
        geo.rot_theta[axis_key].coef[i] = val[i - fix_root_sect]


def thickness(val, geo):
    axis_key = list(geo.axis.keys())[0]

    for i in range(1, nRefAxPts):
        geo.scale_z[axis_key].coef[i] = val[i - fix_root_sect]


def chord(val, geo):
    axis_key = list(geo.axis.keys())[0]

    for i in range(1, nRefAxPts):
        geo.scale_x[axis_key].coef[i] = val[i - fix_root_sect]


def span(val, geo):
    axis_key = list(geo.axis.keys())[0]
    C = geo.extractCoef(axis_key)
    for i in range(1, C.shape[0]):
        C[i, 2] *= val

    geo.restoreCoef(C, axis_key)


def spanX(val, geo):
    axis_key = list(geo.axis.keys())[0]
    C = geo.extractCoef(axis_key)
    for i in range(1, C.shape[0]):
        C[i, 0] *= val

    geo.restoreCoef(C, axis_key)


def getShapeFunc(lidx, direction=None):
    """
    Get shape dictionaries for use with shape function DVs. Common to DVGeometry and MPhys DVGeo tests.
    Requires local index from DVGeo object and optionally a three-element numpy array for the
    positive direction vector and returns shapes.
    """
    shape_1 = {}
    shape_2 = {}

    k_center = 2
    i_center = 1
    n_chord = lidx.shape[0]

    d_up = np.array([0.0, 1.0, 0.0])
    if direction is not None:
        d_up = direction
        d_up /= np.linalg.norm(d_up)

    for kk in [-1, 0, 1]:
        if kk == 0:
            k_weight = 1.0
        else:
            k_weight = 0.5

        for ii in range(n_chord):
            # compute the chord weight. we want the shape to peak at i_center
            if ii == i_center:
                i_weight = 1.0
            elif ii < i_center:
                # we are ahead of the center point
                i_weight = ii / i_center
            else:
                # we are behind the center point
                i_weight = (n_chord - ii - 1) / (n_chord - i_center - 1)

            # scale direction by the i and k weights
            d_up_scaled = d_up * k_weight * i_weight

            # get this point's global index and add to the dictionary with the direction vector.
            gidx_up = lidx[ii, 1, kk + k_center]
            gidx_down = lidx[ii, 0, kk + k_center]

            shape_1[gidx_up] = d_up_scaled
            # the lower face is perturbed with a separate dictionary
            shape_2[gidx_down] = -d_up_scaled

    shapes = [shape_1, shape_2]
    return shapes
