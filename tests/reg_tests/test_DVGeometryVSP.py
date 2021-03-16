from __future__ import print_function
import os
import unittest
import numpy
from baseclasses import BaseRegTest
import commonUtils
from pygeo import geo_utils
from parameterized import parameterized_class
from mpi4py import MPI

try:
    import openvsp
    missing_openvsp = False
except ImportError:
    missing_openvsp = True

if not missing_openvsp:
    from pygeo import DVGeometryVSP

test_params = [
    # # Tutorial scalar JST
    { "N_PROCS": 1, "name":'serial'},
    { "N_PROCS": 4, "name":'parallel_4procs'},
]

@unittest.skipIf(missing_openvsp, 'requires openvsp Python API')
@parameterized_class(test_params)
class RegTestPyGeoVSP(unittest.TestCase):

    # this will be tested in serial and parallel automatically
    N_PROCS = 1



    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def train_1(self, train=True, refDeriv=True):
        self.test_1(train=train, refDeriv=refDeriv)

    def test_1(self, train=False, refDeriv=False):
        """
        Test 1: Basic OpenVSP sphere test

        A sphere centered at 1, 0, 0 but strech scales from the origin 0, 0, 0
        """
        def sample_uv(nu, nv):
            # function to create sample uv from the surface and save these points.
            u = numpy.linspace(0, 1, nu)
            v = numpy.linspace(0, 1, nv)
            uu, vv = numpy.meshgrid(u, v)
            uv = numpy.array((uu.flatten(), vv.flatten()))
            return uv


        refFile = os.path.join(self.base_path,'ref/test_DVGeometryVSP_01.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 1: Basic OpenVSP sphere")
            vspFile = os.path.join(self.base_path, '../inputFiles/simpleEll_med.vsp3')
            DVGeo = DVGeometryVSP(vspFile)
            dh = 0.1
            # we have a sphere centered at x,y,z = 1, 0, 0 with radius 1

            # add design variables for the radius values in x-y-z
            DVGeo.addVariable("Ellipsoid", "Design", "A_Radius", lower=0.5, upper=3.0, scale=1.0, dh=dh)
            DVGeo.addVariable("Ellipsoid", "Design", "B_Radius", lower=0.5, upper=3.0, scale=1.0, dh=dh)
            DVGeo.addVariable("Ellipsoid", "Design", "C_Radius", lower=0.5, upper=3.0, scale=1.0, dh=dh)

            # dictionary of design variables
            x = DVGeo.getValues()
            nDV = len(x)
            dvList = list(x.keys())

            # add some known points to the sphere
            points = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]
            pointSet1 = numpy.array(points)
            nPts = len(points)
            dMax_global = DVGeo.addPointSet(pointSet1, 'known_points')
            handler.assert_allclose(dMax_global, 0.0,
                                    name='pointset1_projection_tol', rtol=1e0, atol=1e-10)


           # add some random points
           # randomly generate points
            nPts = 100
            # just get nPts ^2 points
            uv = sample_uv(10, 10)

            # now lets get the coordinates of these uv combinations
            ptvec = openvsp.CompVecPnt01(DVGeo.allComps[0], 0, uv[0, :], uv[1, :])

            # convert the ptvec into list of coordinates
            points = []
            radError = 1e-20
            radii = []
            for pt in ptvec:
                # print (pt)
                points.append([pt.x(), pt.y(), pt.z()])
                radius = ((pt.x() - 1.0) ** 2 + pt.y() ** 2 + pt.z() ** 2) ** 0.5
                radii.append(radius)
            pointSet2 = numpy.array(points)
            handler.assert_allclose(numpy.array(radii), 1.0,
                                    name='pointset2_diff_from_sphere', rtol=1e-3, atol=1e-3)

            dim = 3
            # add this point set since our points EXACTLY lie on the sphere, we should get 0 distance in the
            # projections to machine precision
            dMax_global = DVGeo.addPointSet(pointSet2, 'generated_points')
            handler.assert_allclose(dMax_global, 0.0,
                                    name='pointset1_projection_tol', rtol=1e0, atol=1e-15)

            # lets get the gradients wrt design variables. For this we can define our dummy jacobian for dIdpt
            # that is an (N, nPts, 3) array. We will just monitor how each component in each point changes so
            # we will need nDV*dim*nPts functions of interest.
            dIdpt = numpy.zeros((nPts * dim, nPts, dim))

            # first 3*nDV entries will correspond to the first point's x,y,z direction
            for i in range(nPts):
                for j in range(dim):
                    # initialize this seed to 1
                    dIdpt[dim * i + j, i, j] = 1.0

            # get the total sensitivities
            funcSens = DVGeo.totalSensitivity(dIdpt, 'generated_points')
            # lets read variables from the total sensitivities and check
            maxError = 1e-20

            # loop over our pointset
            for i in range(nPts):
                point = pointSet2[i, :]

                # we have 3 DVs, radius a,b, and c. These should only change coordinates in the x,y,z directions
                for j in range(nDV):
                    dv = dvList[j]

                    # loop over coordinates
                    for k in range(dim):
                        if k == j:
                            # this sensitivity should be equal to the "i"th coordinate of the original point.
                            error = abs(funcSens[dv][dim * i + k] - point[j])
                        else:
                            # this sensitivity should be zero.
                            error = abs(funcSens[dv][dim * i + k])

                        # print('Error for dv %s on the %d th coordinate of point at (%1.1f, %1.1f, %1.1f) is = %1.16f'%(dv, k+1, point[0],point[1],point[2], error ))
                        maxError = max(error, maxError)
            handler.assert_allclose(maxError, 0.0, 'sphere_derivs', rtol=1e0, atol=1e-14)



    def train_2(self, train=True, refDeriv=True):
        self.test_2(train=train, refDeriv=refDeriv)

    def test_2(self, train=False, refDeriv=False):
        """
        Test 2: OpenVSP wing test
        """
        def sample_uv(nu, nv):
            # function to create sample uv from the surface and save these points.
            u = numpy.linspace(0, 1, nu + 1)
            v = numpy.linspace(0, 1, nv + 1)
            uu, vv = numpy.meshgrid(u, v)
            # print (uu.flatten(), vv.flatten())
            uv = numpy.array((uu.flatten(), vv.flatten()))
            return uv

        refFile = os.path.join(self.base_path,'ref/test_DVGeometryVSP_02.ref')
        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 2: OpenVSP NACA 0012 wing")
            vspFile = os.path.join(self.base_path, '../inputFiles/naca0012.vsp3')
            DVGeo = DVGeometryVSP(vspFile)
            dh = 1e-6


            openvsp.ClearVSPModel()
            openvsp.ReadVSPFile(vspFile)
            geoms = openvsp.FindGeoms()

            comps = []

            DVGeo = DVGeometryVSP(vspFile)
            comp = "WingGeom"
            # loop over sections
            # normally, there are 9 sections so we should loop over range(9) for the full test
            # to have it run faster, we just pick 2 sections
            for i in [0, 5]:
                # Twist
                DVGeo.addVariable(comp, "XSec_%d" % i, "Twist", lower=-10.0, upper=10.0, scale=1e-2, scaledStep=False, dh=dh)

                # loop over coefs
                # normally, there are 7 coeffs so we should loop over range(7) for the full test
                # to have it run faster, we just pick 2 sections
                for j in [0, 4]:
                    # CST Airfoil shape variables
                    group = "UpperCoeff_%d" % i
                    var = "Au_%d" % j
                    DVGeo.addVariable(comp, group, var, lower=-0.1, upper=0.5, scale=1e-3, scaledStep=False, dh=dh)
                    group = "LowerCoeff_%d" % i
                    var = "Al_%d" % j
                    DVGeo.addVariable(comp, group, var, lower=-0.5, upper=0.1, scale=1e-3, scaledStep=False, dh=dh)

            # now lets generate ourselves a quad mesh of these cubes.
            uv_g = sample_uv(8, 8)

            # total number of points
            ntot = uv_g.shape[1]

            # rank on this proc
            rank = MPI.COMM_WORLD.rank

            # first, equally divide
            nuv = ntot // MPI.COMM_WORLD.size
            # then, add the remainder
            if rank < ntot % MPI.COMM_WORLD.size:
                nuv += 1

            # allocate the uv array on this proc
            uv = numpy.zeros((2, nuv))

            # print how mant points we have
            MPI.COMM_WORLD.Barrier()

            # loop over the points and save all that this proc owns
            ii = 0
            for i in range(ntot):
                if i % MPI.COMM_WORLD.size == rank:
                    uv[:, ii] = uv_g[:, i]
                    ii += 1

            # get the coordinates
            nNodes = len(uv[0, :])
            ptVecA = openvsp.CompVecPnt01(geoms[0], 0, uv[0, :], uv[1, :])

            # extract node coordinates and save them in a numpy array
            coor = numpy.zeros((nNodes, 3))
            for i in range(nNodes):
                pnt = openvsp.CompPnt01(geoms[0], 0, uv[0, i], uv[1, i])
                coor[i, :] = (pnt.x(), pnt.y(), pnt.z())

            # Add this pointSet to DVGeo
            DVGeo.addPointSet(coor, "test_points")

            # We will have nNodes*3 many functions of interest...
            dIdpt = numpy.zeros((nNodes * 3, nNodes, 3))

            # set the seeds to one in the following fashion:
            # first function of interest gets the first coordinate of the first point
            # second func gets the second coord of first point etc....
            for i in range(nNodes):
                for j in range(3):
                    dIdpt[i * 3 + j, i, j] = 1

            # first get the dvgeo result
            funcSens = DVGeo.totalSensitivity(dIdpt.copy(), "test_points")

            # now perturb the design with finite differences and compute FD gradients
            DVs = DVGeo.getValues()

            funcSensFD = {}

            for x in DVs:
                # perturb the design
                xRef = DVs[x].copy()
                DVs[x] += dh
                DVGeo.setDesignVars(DVs)

                # get the new points
                coorNew = DVGeo.update("test_points")

                # calculate finite differences
                funcSensFD[x] = (coorNew.flatten() - coor.flatten()) / dh

                # set back the DV
                DVs[x] = xRef.copy()

            # now loop over the values and compare
            # when this is run with multiple procs, VSP sometimes has a bug
            # that leads to different procs having different spanwise
            # u-v distributions. as a result, the final values can differ up to 1e-5 levels
            # this issue does not come up if this tests is ran with a single proc
            biggest_deriv = 1e-16
            for x in DVs:
                err = numpy.array(funcSens[x].squeeze()) - numpy.array(funcSensFD[x])
                maxderiv = numpy.max(numpy.abs(funcSens[x].squeeze()))
                normalizer = numpy.median(numpy.abs(funcSensFD[x].squeeze()))
                if numpy.abs(normalizer) < 1:
                    normalizer = numpy.ones(1)
                normalized_error = err / normalizer
                if maxderiv > biggest_deriv:
                    biggest_deriv = maxderiv
                handler.assert_allclose(normalized_error, 0.0,
                                    name='{}_grad_normalized_error'.format(x), rtol=1e0, atol=5e-5)
            # make sure that at least one derivative is nonzero
            self.assertGreater(biggest_deriv, 0.005)







if __name__ == '__main__':
    unittest.main()


