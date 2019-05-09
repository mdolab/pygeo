
############################################################
# DO NOT USE THIS SCRIPT AS A REFERENCE FOR HOW TO USE pygeo
# THIS SCRIPT USES PRIVATE INTERNAL FUNCTIONALITY THAT IS
# SUBJECT TO CHANGE!!
############################################################

# ======================================================================
#         Imports
# ======================================================================
from __future__ import print_function
import sys,os,copy
import numpy
from pyspline import *
from mdo_regression_helper import *
from pygeo import *
from mpi4py import MPI

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", help='run mode.', choices=['train', 'test'],
                    type=str, default='test')
parser.add_argument("--task", help='what to do',
                    choices=['all', 'test1', 'test2', 'test3', 'test4',
                             'test5', 'test6', 'test7', 'test8',
                             'test9', 'test10', 'test11', 'test12',
                             'test13', 'test14', 'test15', 'test16',
                             'test17', 'test18', 'test19', 'test20',
                         ], default='all')

args = parser.parse_args()

def printHeader(testName):
    if MPI.COMM_WORLD.rank == 0:
        print('+' + '-'*78 + '+')
        print('| Test Name: ' + '%-66s'%testName + '|')
        print('+' + '-'*78 + '+')

##################
# DVGeometry Tests
##################

# setup a basic case

def setupDVGeo():
    #create the Parent FFD
    FFDFile =  './inputFiles/outerBoxFFD.xyz'
    DVGeo = DVGeometry(FFDFile)

    # create a reference axis for the parent
    axisPoints = [[ -1.0,   0.  ,   0.],[ 1.5,   0.,   0.]]
    c1 = Curve(X=axisPoints,k=2)
    DVGeo.addRefAxis('mainAxis',curve=c1, axis='y')

    # create the child FFD
    FFDFile = './inputFiles/simpleInnerFFD.xyz'
    DVGeoChild = DVGeometry(FFDFile,child=True)

    # create a reference axis for the child
    axisPoints = [[ -0.5,   0.  ,   0.],[ 0.5,   0.,   0.]]
    c1 = Curve(X=axisPoints,k=2)
    DVGeoChild.addRefAxis('nestedAxis',curve=c1, axis='y')

    return DVGeo,DVGeoChild

def setupDVGeoD8(isComplex):
    #create the Parent FFD
    FFDFile = './inputFiles/bodyFFD.xyz'
    DVGeo = DVGeometry(FFDFile,complex=isComplex)

    # create a reference axis for the parent
    axisPoints = [[0.,0.,0.],[26.,0.,0.],[30.5,0.,0.9],
                  [ 32.5, 0., 1.01],[ 34.0,   0., 0.95]]
    c1 = Curve(X=axisPoints,k=2)
    DVGeo.addRefAxis('mainAxis',curve=c1, axis='y')

    # create the child FFD
    FFDFile =  './inputFiles/nozzleFFD.xyz'
    DVGeoChild = DVGeometry(FFDFile,child=True,complex=isComplex)

    # create a reference axis for the child
    axisPoints = [[32.4,   1.  ,   1.],[ 34,   1.,   0.9]]
    c1 = Curve(X=axisPoints,k=2)
    DVGeoChild.addRefAxis('nestedAxis',curve=c1, axis='y')

    return DVGeo, DVGeoChild

def setupDVGeoAxi():
    DVGeo = DVGeometryAxi("./inputFiles/axiTestFFD.xyz", center=(0., 0., 0.), collapse_into=("x", "z"))
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

def totalSensitivityFD(DVGeo,nPt,ptName):
    xDV = DVGeo.getValues()
    refPoints = DVGeo.update(ptName)
    #now get FD Sensitivity
    dIdxFD = {}
    step = 1e-1#8
    for key in xDV:
        baseVar = xDV[key].copy()
        dIdxFD[key] =numpy.zeros([nPt,len(baseVar)])
        for i in range(len(baseVar)):
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
    #now get FD Sensitivity
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

    return dIdxCS

def testSensitivities(DVGeo,refDeriv):
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
    #get analitic sensitivity
    if refDeriv:
        dIdx = totalSensitivityFD(DVGeo,nPt,ptName)
    else:
        dIdx = DVGeo.totalSensitivity(dIdPt,ptName)

    for key in sorted(list(dIdx)):
        print(key)
        for i in range(dIdx[key].shape[0]):
            for j in range(dIdx[key].shape[1]):
                reg_write(dIdx[key][i,j],1e-7,1e-7)

def testSensitivitiesD8(DVGeo,refDeriv):
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
    #get analitic sensitivity
    if refDeriv:
        # dIdx = totalSensitivityFD(DVGeo,nPt,ptName)
        dIdx = totalSensitivityCS(DVGeo,nPt,ptName)
    else:
        dIdx = DVGeo.totalSensitivity(dIdPt,ptName)

    for key in list(sorted(dIdx)):
        print(key)
        for i in range(dIdx[key].shape[0]):
            for j in range(dIdx[key].shape[1]):
                reg_write(dIdx[key][i,j],1e-7,1e-7)



def test1(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 1: Basic FFD, global DVs")

    DVGeo,DVGeoChild = setupDVGeo()

    #create global DVs on the parent
    DVGeo.addGeoDVGlobal('mainX', -1.0, mainAxisPoints,
                          lower=-1., upper=0., scale=1.0)

    testSensitivities(DVGeo,refDeriv)


    del DVGeo
    del DVGeoChild

def test2(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 2: Basic FFD, global DVs and local DVs")

    DVGeo,DVGeoChild = setupDVGeo()

    #create global DVs on the parent
    DVGeo.addGeoDVGlobal('mainX', -1.0, mainAxisPoints,
                          lower=-1., upper=0., scale=1.0)
    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    testSensitivities(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild

def test3(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 3: Basic + Nested FFD, global DVs only")

    DVGeo,DVGeoChild = setupDVGeo()

    #create global DVs on the parent
    DVGeo.addGeoDVGlobal('mainX', -1.0, mainAxisPoints,
                          lower=-1., upper=0., scale=1.0)
    #create global DVs on the child
    DVGeoChild.addGeoDVGlobal('nestedX', -0.5, childAxisPoints,
                              lower=-1., upper=0., scale=1.0)
    DVGeo.addChild(DVGeoChild)

    testSensitivities(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild

def test4(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 4: Basic + Nested FFD, global DVs and local DVs on parent global on child")

    DVGeo,DVGeoChild = setupDVGeo()

    #create global DVs on the parent
    DVGeo.addGeoDVGlobal('mainX', -1.0, mainAxisPoints,
                          lower=-1., upper=0., scale=1.0)
    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    #create global DVs on the child
    DVGeoChild.addGeoDVGlobal('nestedX', -0.5, childAxisPoints,
                              lower=-1., upper=0., scale=1.0)
    DVGeo.addChild(DVGeoChild)

    testSensitivities(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild

def test5(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 5: Basic + Nested FFD,  global DVs and local DVs on both parent and child")

    DVGeo,DVGeoChild = setupDVGeo()

    #create global DVs on the parent
    DVGeo.addGeoDVGlobal('mainX', -1.0, mainAxisPoints,
                          lower=-1., upper=0., scale=1.0)
    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)
    #create global DVs on the child
    DVGeoChild.addGeoDVGlobal('nestedX', -0.5, childAxisPoints,
                              lower=-1., upper=0., scale=1.0)
    DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
    DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
    DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivities(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild


def test6(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 6: Basic + Nested FFD, local DVs only")

    DVGeo,DVGeoChild = setupDVGeo()

    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
    DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
    DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivities(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild

def test6b(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 6: Basic + Nested FFD, local DVs only on parent")

    DVGeo,DVGeoChild = setupDVGeo()

    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivities(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild

def test6c(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 6: Basic + Nested FFD, local DVs only on child")

    DVGeo,DVGeoChild = setupDVGeo()

    DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
    DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
    DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivities(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild

def test7(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 7: Basic + Nested FFD, local DVs only on parent, global on child")

    DVGeo,DVGeoChild = setupDVGeo()

    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    DVGeoChild.addGeoDVGlobal('nestedX', -0.5, childAxisPoints,
                              lower=-1., upper=0., scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivities(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild

def test8(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 8: Basic + Nested FFD, local DVs only on parent, global and local on child")

    DVGeo,DVGeoChild = setupDVGeo()

    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    DVGeoChild.addGeoDVGlobal('nestedX', -0.5, childAxisPoints,
                              lower=-1., upper=0., scale=1.0)
    DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
    DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
    DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivities(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild

def test9(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 9: Basic + Nested FFD, global DVs and local DVs on parent local on child")

    DVGeo,DVGeoChild = setupDVGeo()

    #create global DVs on the parent
    DVGeo.addGeoDVGlobal('mainX', -1.0, mainAxisPoints,
                          lower=-1., upper=0., scale=1.0)
    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    #create global DVs on the child
    DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
    DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
    DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivities(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild

# -------------------
# D8 Tests
# -------------------

def test10(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 10: D8 FFD, global DVs")
    if refDeriv:
        DVGeo,DVGeoChild = setupDVGeoD8(True)
    else:
        DVGeo,DVGeoChild = setupDVGeoD8(False)

    #create global DVs on the parent
    axisX = [0.,26.,30.5,32.5, 34.0]
    DVGeo.addGeoDVGlobal('mainX', axisX , mainAxisPoints,
                          lower=0., upper=35., scale=1.0)

    testSensitivitiesD8(DVGeo,refDeriv)
    sys.stdout.flush()
    printHeader("Test 10b: D8 FFD,  random DV perturbation on test 10")
    xDV = DVGeo.getValues()
    for key in xDV:
        numpy.random.seed(42)
        xDV[key]+=numpy.random.rand(len(xDV[key]))

    DVGeo.setDesignVars(xDV)
    testSensitivitiesD8(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild

def test11(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 11: D8 FFD, global DVs and local DVs")

    if refDeriv:
        DVGeo,DVGeoChild = setupDVGeoD8(True)
    else:
        DVGeo,DVGeoChild = setupDVGeoD8(False)


    #create global DVs on the parent
    axisX = [0.,26.,30.5,32.5, 34.0]
    DVGeo.addGeoDVGlobal('mainX', axisX , mainAxisPoints,
                         lower=0., upper=35., scale=1.0)
    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    testSensitivitiesD8(DVGeo,refDeriv)
    sys.stdout.flush()
    printHeader("Test 11b: D8 FFD,  random DV perturbation on test 11")
    xDV = DVGeo.getValues()
    for key in xDV:
        numpy.random.seed(42)
        xDV[key]+=numpy.random.rand(len(xDV[key]))

    DVGeo.setDesignVars(xDV)
    testSensitivitiesD8(DVGeo,refDeriv)
    del DVGeo
    del DVGeoChild

def test12(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 12: D8 + Nozzle FFD, global DVs only")

    if refDeriv:
        DVGeo,DVGeoChild = setupDVGeoD8(True)
    else:
        DVGeo,DVGeoChild = setupDVGeoD8(False)

    #create global DVs on the parent
    axisX = [0.,26.,30.5,32.5, 34.0]
    DVGeo.addGeoDVGlobal('mainX', axisX , mainAxisPointsD8,
                         lower=0., upper=35., scale=1.0)
    #create global DVs on the child
    childAxisX = [32.4, 34]
    DVGeoChild.addGeoDVGlobal('nestedX',childAxisX, childAxisPointsD8,
                              lower=0., upper=35., scale=1.0)
    DVGeo.addChild(DVGeoChild)

    testSensitivitiesD8(DVGeo,refDeriv)
    sys.stdout.flush()
    printHeader("Test 12b: D8 + Nozzle FFD,  random DV perturbation on test 12")
    xDV = DVGeo.getValues()
    for key in xDV:
        numpy.random.seed(42)
        xDV[key]+=numpy.random.rand(len(xDV[key]))

    DVGeo.setDesignVars(xDV)
    testSensitivitiesD8(DVGeo,refDeriv)
    del DVGeo
    del DVGeoChild

def test13(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 13: D8 + Nozzle FFD, global DVs and local DVs on parent global on child")

    if refDeriv:
        DVGeo,DVGeoChild = setupDVGeoD8(True)
    else:
        DVGeo,DVGeoChild = setupDVGeoD8(False)

    #create global DVs on the parent
    axisX = [0.,26.,30.5,32.5, 34.0]
    DVGeo.addGeoDVGlobal('mainX', axisX , mainAxisPoints,
                         lower=0., upper=35., scale=1.0)
    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    #create global DVs on the child
    childAxisX = [32.4, 34]
    DVGeoChild.addGeoDVGlobal('nestedX',childAxisX, childAxisPoints,
                              lower=0., upper=35., scale=1.0)
    DVGeo.addChild(DVGeoChild)

    testSensitivitiesD8(DVGeo,refDeriv)
    sys.stdout.flush()
    printHeader("Test 13b: D8 + Nozzle FFD,  random DV perturbation on test 13")
    xDV = DVGeo.getValues()
    for key in xDV:
        numpy.random.seed(42)
        xDV[key]+=numpy.random.rand(len(xDV[key]))

    DVGeo.setDesignVars(xDV)
    testSensitivitiesD8(DVGeo,refDeriv)
    del DVGeo
    del DVGeoChild

def test14(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 14: D8 + Nozzle FFD,  global DVs and local DVs on both parent and child")

    if refDeriv:
        DVGeo,DVGeoChild = setupDVGeoD8(True)
    else:
        DVGeo,DVGeoChild = setupDVGeoD8(False)

    #create global DVs on the parent
    axisX = [0.,26.,30.5,32.5, 34.0]
    DVGeo.addGeoDVGlobal('mainX', axisX , mainAxisPoints,
                         lower=0., upper=35., scale=1.0)
    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)
    #create global DVs on the child
    childAxisX = [32.4, 34]
    DVGeoChild.addGeoDVGlobal('nestedX',childAxisX, childAxisPoints,
                              lower=0., upper=35., scale=1.0)
    DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
    DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
    DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivitiesD8(DVGeo,refDeriv)
    sys.stdout.flush()
    printHeader("Test 14b: D8 + Nozzle FFD,  random DV perturbation on test 14")
    xDV = DVGeo.getValues()
    for key in xDV:
        numpy.random.seed(42)
        xDV[key]+=numpy.random.rand(len(xDV[key]))

    DVGeo.setDesignVars(xDV)
    testSensitivitiesD8(DVGeo,refDeriv)
    del DVGeo
    del DVGeoChild


def test15(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 15: D8 + Nozzle FFD, local DVs only")

    if refDeriv:
        DVGeo,DVGeoChild = setupDVGeoD8(True)
    else:
        DVGeo,DVGeoChild = setupDVGeoD8(False)

    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
    DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
    DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivitiesD8(DVGeo,refDeriv)
    sys.stdout.flush()
    printHeader("Test 15b: D8 + Nozzle FFD,  random DV perturbationon test 15")
    xDV = DVGeo.getValues()
    for key in xDV:
        numpy.random.seed(42)
        xDV[key]+=numpy.random.rand(len(xDV[key]))

    DVGeo.setDesignVars(xDV)
    testSensitivitiesD8(DVGeo,refDeriv)
    del DVGeo
    del DVGeoChild

def test16(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 16: D8 + Nozzle FFD, local DVs only on parent, global on child")

    if refDeriv:
        DVGeo,DVGeoChild = setupDVGeoD8(True)
    else:
        DVGeo,DVGeoChild = setupDVGeoD8(False)

    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    childAxisX = [32.4, 34]
    DVGeoChild.addGeoDVGlobal('nestedX',childAxisX, childAxisPoints,
                              lower=0., upper=35., scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivitiesD8(DVGeo,refDeriv)
    sys.stdout.flush()
    printHeader("Test 16b: D8 + Nozzle FFD,  random DV perturbationon test 16")
    xDV = DVGeo.getValues()
    for key in xDV:
        numpy.random.seed(42)
        xDV[key]+=numpy.random.rand(len(xDV[key]))

    DVGeo.setDesignVars(xDV)
    testSensitivitiesD8(DVGeo,refDeriv)
    del DVGeo
    del DVGeoChild

def test17(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 17: D8 + Nozzle FFD, local DVs only on parent, global and local on child")

    if refDeriv:
        DVGeo,DVGeoChild = setupDVGeoD8(True)
    else:
        DVGeo,DVGeoChild = setupDVGeoD8(False)

    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    childAxisX = [32.4, 34]
    DVGeoChild.addGeoDVGlobal('nestedX',childAxisX, childAxisPoints,
                              lower=0., upper=35., scale=1.0)
    DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
    DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
    DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivitiesD8(DVGeo,refDeriv)
    sys.stdout.flush()
    printHeader("Test 17b: D8 + Nozzle FFD,  random DV perturbationon test 17")
    xDV = DVGeo.getValues()
    for key in xDV:
        numpy.random.seed(42)
        xDV[key]+=numpy.random.rand(len(xDV[key]))

    DVGeo.setDesignVars(xDV)
    testSensitivitiesD8(DVGeo,refDeriv)
    del DVGeo
    del DVGeoChild

def test18(refDeriv=False):
    # Test a basic case. Single FFD
    sys.stdout.flush()
    printHeader("Test 18: D8 + Nozzle FFD, global DVs and local DVs on parent local on child")

    if refDeriv:
        DVGeo,DVGeoChild = setupDVGeoD8(True)
    else:
        DVGeo,DVGeoChild = setupDVGeoD8(False)

    #create global DVs on the parent
    axisX = [0.,26.,30.5,32.5, 34.0]
    DVGeo.addGeoDVGlobal('mainX', axisX , mainAxisPoints,
                         lower=0., upper=35., scale=1.0)
    #create local DVs on the parent
    DVGeo.addGeoDVLocal('xdir', lower=-1.0, upper=1.0, axis='x', scale=1.0)
    DVGeo.addGeoDVLocal('ydir', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    DVGeo.addGeoDVLocal('zdir', lower=-1.0, upper=1.0, axis='z', scale=1.0)

    #create global DVs on the child
    DVGeoChild.addGeoDVLocal('childxdir', lower=-1.1, upper=1.1, axis='x', scale=1.0)
    DVGeoChild.addGeoDVLocal('childydir', lower=-1.1, upper=1.1, axis='y', scale=1.0)
    DVGeoChild.addGeoDVLocal('childzdir', lower=-1.1, upper=1.1, axis='z', scale=1.0)

    DVGeo.addChild(DVGeoChild)

    testSensitivitiesD8(DVGeo,refDeriv)
    sys.stdout.flush()
    printHeader("Test 18b: D8 + Nozzle FFD,  random DV perturbationon test 18")
    xDV = DVGeo.getValues()
    for key in xDV:
        numpy.random.seed(42)
        xDV[key]+=numpy.random.rand(len(xDV[key]))

    DVGeo.setDesignVars(xDV)
    testSensitivitiesD8(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild

def test19(refDeriv=False):
    printHeader("Test 19: Axisymmetric FFD, global and local DVs")
    # Test with a single point along the 45 ` degree theta direction
    sys.stdout.flush()

    DVGeo = setupDVGeoAxi()

    s_pts = numpy.array([[0, .5, .5],], dtype="float")


    DVGeo.addGeoDVGlobal('mainAxis', numpy.zeros(1), mainAxisPointAxi)

    DVGeo.addGeoDVLocal('x_axis', lower=-2, upper=2, axis="x")
    DVGeo.addGeoDVLocal('z_axis', lower=-2, upper=2, axis="z")
    DVGeo.addGeoDVLocal('y_axis', lower=-2, upper=2, axis="y")

    DVGeo.addPointSet(points=s_pts, ptName="point")

    DVGeo.computeTotalJacobian("point")

    if not refDeriv:
        J_analytic = DVGeo.JT['point'].T.toarray()

        reg_write(J_analytic,1e-7,1e-7)

    else:
        # generate an FD jacobian
        xDV = DVGeo.getValues()
        global_var_names = DVGeo.DV_listGlobal.keys()
        local_var_names = DVGeo.DV_listLocal.keys()
        refPoints = DVGeo.update("point")
        n_pt = refPoints.shape[0]

        step = 1e-5
        J_fd = numpy.empty((3, 0))

        for dv_name in global_var_names + local_var_names:
            n_dv = xDV[dv_name].shape[0]

            dPtdDV_var = numpy.empty((3*n_pt, n_dv))

            baseVar = xDV[dv_name].copy()

            for i in range(n_dv):
                xDV[dv_name][i] = baseVar[i]+ step
                DVGeo.setDesignVars(xDV)
                newPoints = DVGeo.update("point")
                deriv = (newPoints-refPoints)/step
                dPtdDV_var[:, i] = deriv.flatten()
                xDV[dv_name][i] = baseVar[i]

            J_fd = numpy.hstack((J_fd, dPtdDV_var))

        reg_write(J_fd,1e-7,1e-7)

def test20(refDeriv=False):
    printHeader("Test FFD writing function")
    sys.stdout.flush()

    # Write duplicate of outerbox FFD
    axes = ['i', 'k', 'j']
    slices = numpy.array([
        # Slice 1
        [[[-1, -1, -1], [-1, 1, -1]],
        [[-1, -1, 1], [-1, 1, 1]]],
        # Slice 2
        [[[1, -1, -1], [1, 1, -1]],
        [[1, -1, 1], [1, 1, 1]]],
        # Slice 3
        [[[2, -1, -1], [2, 1, -1]],
        [[2, -1, 1], [2, 1, 1]]],
    ])

    N0 = [2,2]
    N1 = [2,2]
    N2 = [2,2]

    copyName = 'inputFiles/test1.xyz'
    geo_utils.write_wing_FFD_file(copyName, slices, N0, N1, N2, axes=axes)

    # Load original and duplicate
    origFFD = DVGeometry('inputFiles/outerBoxFFD.xyz')
    copyFFD = DVGeometry(copyName)
    norm_diff = numpy.linalg.norm(origFFD.FFD.coef - copyFFD.FFD.coef)
    reg_write(norm_diff, 1e-7, 1e-7)
    os.remove(copyName)

######################
# DV constraints Tests
######################



refDeriv = False
if args.mode=='train':
    refDeriv = True
if args.task=='all':
    test1(refDeriv)
    test2(refDeriv)
    test3(refDeriv)
    test4(refDeriv)
    test5(refDeriv)
    test6(refDeriv)
    test6b(refDeriv)
    test6c(refDeriv)
    test7(refDeriv)
    test8(refDeriv)
    test9(refDeriv)
    test10(refDeriv)
    test11(refDeriv)
    test12(refDeriv)
    test13(refDeriv)
    test14(refDeriv)
    test15(refDeriv)
    test16(refDeriv)
    test17(refDeriv)
    test18(refDeriv)
    test19(refDeriv)
    test20(refDeriv)
elif args.task=='test1':
    test1(refDeriv)
elif args.task=='test2':
    test2(refDeriv)
elif args.task=='test3':
    test3(refDeriv)
elif args.task=='test4':
    test4(refDeriv)
elif args.task=='test5':
    test5(refDeriv)
elif args.task=='test6':
    test6(refDeriv)
    test6b(refDeriv)
    test6c(refDeriv)
elif args.task=='test7':
    test7(refDeriv)
elif args.task=='test8':
    test8(refDeriv)
elif args.task=='test9':
    test9(refDeriv)
elif args.task=='test10':
    test10(refDeriv)
elif args.task=='test11':
    test11(refDeriv)
elif args.task=='test12':
    test12(refDeriv)
elif args.task=='test13':
    test13(refDeriv)
elif args.task=='test14':
    test14(refDeriv)
elif args.task=='test15':
    test15(refDeriv)
elif args.task=='test16':
    test16(refDeriv)
elif args.task=='test17':
    test17(refDeriv)
elif args.task=='test18':
    test18(refDeriv)
elif args.task=='test19':
    test19(refDeriv)
elif args.task=='test20':
    test20(refDeriv)
