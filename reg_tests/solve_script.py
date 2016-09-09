
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
                    choices=['all', 'test1', 'test2', 'test3', 'test4'], default='all')

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
    
#define a nested global design variable
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

def totalSensitivityFD(DVGeo,nPt,ptName):
    xDV = DVGeo.getValues()
    refPoints = DVGeo.update(ptName)
    #now get FD Sensitivity
    dIdxFD = {}
    step = 1e-8
    for key in xDV:
        baseVar = xDV[key].copy()
        dIdxFD[key] =numpy.zeros([nPt,len(baseVar)])
        for i in range(len(baseVar)):
            xDV[key][i] = baseVar[i]+step
            DVGeo.setDesignVars(xDV)
            newPoints = DVGeo.update(ptName)

            deriv = (newPoints-refPoints)/step
            dIdxFD[key][:,i] = deriv.flatten()
            #print 'Deriv',key, i,deriv
            xDV[key][i] = baseVar[i]

    return dIdxFD

# def totalSensitivityCS(DVGeo,nPt,ptName):
#     xDV = DVGeo.getValues()
#     #now get FD Sensitivity
#     dIdxCS = {}
#     step = 1e-40j
#     for key in xDV:
#         baseVar = xDV[key].copy()
#         dIdxCS[key] =numpy.zeros([nPt,len(baseVar)])
#         for i in range(len(baseVar)):
#             xDV[key][i] = baseVar[i]+step
#             DVGeo.setDesignVars(xDV)
#             newPoints = DVGeo.update(ptName)

#             deriv = numpy.imag(newPoints)/numpy.imag(step)
#             dIdxCS[key][:,i] = deriv.flatten()
#             #print 'Deriv',key, i,deriv
#             xDV[key][i] = baseVar[i]

#     return dIdxCS

def testSensitvities(DVGeo,refDeriv):
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

    for key in dIdx:
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

    testSensitvities(DVGeo,refDeriv)
        

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

    testSensitvities(DVGeo,refDeriv)

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

    testSensitvities(DVGeo,refDeriv)

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

    testSensitvities(DVGeo,refDeriv)

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

    testSensitvities(DVGeo,refDeriv)

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

    testSensitvities(DVGeo,refDeriv)

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

    testSensitvities(DVGeo,refDeriv)

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

    testSensitvities(DVGeo,refDeriv)

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

    testSensitvities(DVGeo,refDeriv)

    del DVGeo
    del DVGeoChild


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
    test7(refDeriv)
    test8(refDeriv)
    test9(refDeriv)
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
elif args.task=='test7':
    test7(refDeriv)
elif args.task=='test8':
    test8(refDeriv)
elif args.task=='test9':
    test8(refDeriv)
