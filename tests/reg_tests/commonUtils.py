from __future__ import print_function
import os
import numpy
from pygeo import DVGeometry, DVGeometryAxi
from pyspline import Curve



def printHeader(testName):
    if MPI.COMM_WORLD.rank == 0:
        print('+' + '-'*78 + '+')
        print('| Test Name: ' + '%-66s'%testName + '|')
        print('+' + '-'*78 + '+')

##################
# DVGeometry Tests
##################

def setupDVGeo(base_path):
    #create the Parent FFD
    FFDFile =  os.path.join(base_path,'../inputFiles/outerBoxFFD.xyz')
    DVGeo = DVGeometry(FFDFile)

    # create a reference axis for the parent
    axisPoints = [[ -1.0,   0.  ,   0.],[ 1.5,   0.,   0.]]
    c1 = Curve(X=axisPoints,k=2)
    DVGeo.addRefAxis('mainAxis',curve=c1, axis='y')

    # create the child FFD
    FFDFile = os.path.join(base_path,'../inputFiles/simpleInnerFFD.xyz')
    DVGeoChild = DVGeometry(FFDFile,child=True)

    # create a reference axis for the child
    axisPoints = [[ -0.5,   0.  ,   0.],[ 0.5,   0.,   0.]]
    c1 = Curve(X=axisPoints,k=2)
    DVGeoChild.addRefAxis('nestedAxis',curve=c1, axis='y')

    return DVGeo,DVGeoChild

def setupDVGeoD8(base_path, isComplex):
    #create the Parent FFD
    FFDFile = os.path.join(base_path,'../inputFiles/bodyFFD.xyz')
    DVGeo = DVGeometry(FFDFile,complex=isComplex)

    # create a reference axis for the parent
    axisPoints = [[0.,0.,0.],[26.,0.,0.],[30.5,0.,0.9],
                  [ 32.5, 0., 1.01],[ 34.0,   0., 0.95]]
    c1 = Curve(X=axisPoints,k=2)
    DVGeo.addRefAxis('mainAxis',curve=c1, axis='y')

    # create the child FFD
    FFDFile =  os.path.join(base_path,'../inputFiles/nozzleFFD.xyz')
    DVGeoChild = DVGeometry(FFDFile,child=True,complex=isComplex)

    # create a reference axis for the child
    axisPoints = [[32.4,   1.  ,   1.],[ 34,   1.,   0.9]]
    c1 = Curve(X=axisPoints,k=2)
    DVGeoChild.addRefAxis('nestedAxis',curve=c1, axis='y')

    return DVGeo, DVGeoChild

def setupDVGeoAxi(base_path):
    FFDFile = os.path.join(base_path,"../inputFiles/axiTestFFD.xyz")
    DVGeo = DVGeometryAxi(FFDFile, center=(0., 0., 0.), collapse_into=("x", "z"))
    axisPoints = [[ 0,   0.  ,   0.],[ 0,  0.,  1.]]
    c1 = Curve(X=axisPoints,k=2)
    DVGeo.addRefAxis('stretch',curve=c1, axis='z')

    return DVGeo


# define a nested global design variable
def childAxisPoints(val,geo):
    C = geo.extractCoef('nestedAxis')

    # Set the coefficients
    C[0,0] = val[0]

    geo.restoreCoef(C, 'nestedAxis')

    return

#define a nested global design variable
def mainAxisPoints(val,geo):
    C = geo.extractCoef('mainAxis')

    # Set the coefficients
    C[0,0] = val[0]

    geo.restoreCoef(C, 'mainAxis')

    return

#define a nested global design variable
def childAxisPointsD8(val,geo):
    C = geo.extractCoef('nestedAxis')

    # Set the coefficients
    for i in range(len(val)):
        C[i,0] = val[i]

    geo.restoreCoef(C, 'nestedAxis')

    return

#define a nested global design variable
def mainAxisPointsD8(val,geo):
    C = geo.extractCoef('mainAxis')

    # Set the coefficients
    for i in range(len(val)):
        C[i,0] = val[i]

    geo.restoreCoef(C, 'mainAxis')

    return

def mainAxisPointAxi(val, DVgeo):
    C = DVgeo.extractCoef('stretch')
    C[0,2] = val[0]

    DVgeo.restoreCoef(C, 'stretch')
    return

def totalSensitivityFD(DVGeo,nPt,ptName,step=1e-1):
    xDV = DVGeo.getValues()
    refPoints = DVGeo.update(ptName)
    #now get FD Sensitivity
    dIdxFD = {}
    #step = 1e-1#8
    for key in xDV:
        baseVar = xDV[key].copy()
        nDV = len(baseVar)
        dIdxFD[key] = numpy.zeros([nPt,nDV])
        for i in range(nDV):
            #print('perturbing',key)
            xDV[key][i] = baseVar[i]+step
            #print('setting design vars')
            DVGeo.setDesignVars(xDV)
            #print('calling top level update')
            newPoints = DVGeo.update(ptName)

            deriv = (newPoints-refPoints)/step
            dIdxFD[key][:,i] = deriv.flatten()
            #print('Deriv',key, i,deriv)
            xDV[key][i] = baseVar[i]

    return dIdxFD

def totalSensitivityCS(DVGeo,nPt,ptName):
    xDV = DVGeo.getValues()
    
    #now get CS Sensitivity
    dIdxCS = {}
    step = 1e-40j
    for key in xDV:
        baseVar = xDV[key].copy()
        dIdxCS[key] =numpy.zeros([nPt,len(baseVar)])
        for i in range(len(baseVar)):
            xDV[key][i] = baseVar[i]+step
            
            DVGeo.setDesignVars(xDV)
            newPoints = DVGeo.update(ptName)

            deriv = numpy.imag(newPoints)/numpy.imag(step)
            dIdxCS[key][:,i] = deriv.flatten()
            #print 'Deriv',key, i,deriv
            xDV[key][i] = baseVar[i]

    # Before we exit make sure we have reset the DVs
    DVGeo.setDesignVars(xDV)

    return dIdxCS

def testSensitivities(DVGeo,refDeriv,handler):
    #create test points
    points = numpy.zeros([2,3])
    points[0,:] = [0.25,0,0]
    points[1,:] = [-0.25,0,0]

    # add points to the geometry object
    ptName = 'testPoints'
    DVGeo.addPointSet(points,ptName)

    # generate dIdPt
    nPt = 6
    dIdPt = numpy.zeros([nPt,2,3])
    dIdPt[0,0,0] = 1.0
    dIdPt[1,0,1] = 1.0
    dIdPt[2,0,2] = 1.0
    dIdPt[3,1,0] = 1.0
    dIdPt[4,1,1] = 1.0
    dIdPt[5,1,2] = 1.0
    #get analytic sensitivity
    if refDeriv:
        dIdx = totalSensitivityFD(DVGeo,nPt,ptName)
    else:
        dIdx = DVGeo.totalSensitivity(dIdPt,ptName)

    handler.root_add_dict(dIdx,1e-7,1e-7)

def testSensitivitiesD8(DVGeo,refDeriv,handler):
    #create test points
    nPoints = 50
    points = numpy.zeros([nPoints,3])
    for i in range(nPoints):
        nose = 0.01
        tail = 34.0
        delta = (tail-nose)/nPoints
        points[i,:] = [nose+i*delta,1.0,0.5]

    #print('points',points)

    # add points to the geometry object
    ptName = 'testPoints'
    DVGeo.addPointSet(points,ptName,faceFreeze={})

    # generate dIdPt
    nPt = nPoints*3
    dIdPt = numpy.zeros([nPt,nPoints,3])
    counter = 0
    for i in range(nPoints):
        for j in range(3):
            dIdPt[counter,i,j] = 1.0
            counter+=1
    #get analytic sensitivity
    if refDeriv:
        # dIdx = totalSensitivityFD(DVGeo,nPt,ptName)
        dIdx = totalSensitivityCS(DVGeo,nPt,ptName)
    else:
        dIdx = DVGeo.totalSensitivity(dIdPt,ptName)

    handler.root_add_dict(dIdx,1e-7,1e-7)
