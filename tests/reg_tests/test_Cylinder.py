from __future__ import print_function
import os
import unittest
import numpy
from baseclasses import BaseRegTest
import commonUtils
from pygeo import DVGeometry, DVConstraints, geo_utils


class RegTestPyGeo(unittest.TestCase):

    N_PROCS = 1

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def make_cylinder_mesh(self, radius=1.0, height=2.0):
        Nazimuth = 1000
        Nextrude = 100
        Npts = Nazimuth * Nextrude

        theta = numpy.linspace(0, 2*numpy.pi, Nazimuth)
        z = numpy.linspace(0, height, Nextrude)

        pts = numpy.zeros((Npts, 3))

        # First populate the points
        for i in range(Nextrude):
            for j in range(Nazimuth):
                x = radius * numpy.cos(theta[j])
                y = radius * numpy.sin(theta[j])

                k = i*Nazimuth + j
                pts[k] = [x, y, z[i]]

        p0 = []
        v1 = []
        v2 = []

        # Now create the triangulation
        for i in range(Nextrude-1):
            for j in range(Nazimuth-1):
                cur_level = i * Nazimuth
                next_level = (i + 1) * Nazimuth

                pA = pts[cur_level + j]
                pB = pts[cur_level + j + 1]
                pC = pts[next_level + j]
                pD = pts[next_level + j + 1]

                # Triangle 1
                p0.append(pA)
                v1.append(pB - pA)
                v2.append(pD - pA)

                # Triangle 2
                p0.append(pA)
                v1.append(pC - pA)
                v2.append(pD - pA)

        p0 = numpy.vstack(p0)
        v1 = numpy.vstack(v1)
        v2 = numpy.vstack(v2)

        return [p0, v1, v2]

    def make_ffd(self, file_name, radius=1.0, height=2.0):
        # Write duplicate of outerbox FFD
        axes = ['i', 'k', 'j']
        r = radius
        h = height
        dh = 0.01
        slices = numpy.array([
            # Slice 1
            [[[-r, -r, -dh], [r, -r, -dh]],
            [[-r, r, -dh], [r, r, -dh]]],
            # Slice 2
            [[[-r, -r, h+dh], [r, -r, h+dh]],
            [[-r, r, h+dh], [r, r, h+dh]]],
        ])

        N0 = [5]
        N1 = [2]
        N2 = [2]

        geo_utils.write_wing_FFD_file(file_name, slices, N0, N1, N2, axes=axes)

    def train_1(self, train=True, refDeriv=True):
        self.test_1(train=train, refDeriv=refDeriv)

    def test_1(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path,'ref/test_Cylinder_01.ref')

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 1: Basic FFD, global DVs")
            radius = 1.0
            height = 10.0

            DVCon = DVConstraints()
            surf = self.make_cylinder_mesh(radius, height)
            DVCon.setSurface(surf)
            # DVCon.writeSurfaceTecplot('cylinder_surface.dat')

            ffd_name = os.path.join(self.base_path,'../inputFiles/cylinder_ffd.xyz')
            self.make_ffd(ffd_name, radius, height)
            DVGeo = DVGeometry(ffd_name)
            nAxPts = DVGeo.addRefAxis('thru', xFraction=0.5, alignIndex='i', raySize=1.0)

            def scale_circle(val, geo):
                for i in range(nAxPts):
                    geo.scale['thru'].coef[i] = val[0]

            DVGeo.addGeoDVGlobal('scale_circle', func=scale_circle, value=[1])
            DVCon.setDVGeo(DVGeo)

            leList = [[0, 0, 0 ], [-radius/2, 0, height]]
            xAxis = [-1, 0, 0]
            yAxis = [0, 1, 0]
            DVCon.addLERadiusConstraints(leList, nSpan=5, axis=yAxis,
                                         chordDir=xAxis, scaled=False)
            # DVCon.writeTecplot('cylinder_constraints.dat')

            funcs = {}
            DVCon.evalFunctions(funcs)
            print(funcs)
            handler.root_add_dict(funcs, 1e-6, 1e-6)

            DVGeo.setDesignVars({'scale_circle':0.5})

            funcs = {}
            DVCon.evalFunctions(funcs)
            handler.root_add_dict(funcs, 1e-6, 1e-6)
            print(funcs)

            funcsSens = {}
            DVCon.evalFunctionsSens(funcsSens)
            handler.root_add_dict(funcsSens, 1e-6, 1e-6)
            print(funcsSens)


