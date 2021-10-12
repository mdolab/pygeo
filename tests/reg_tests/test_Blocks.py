import os
import unittest
import numpy as np
from baseclasses import BaseRegTest
import commonUtils
from pygeo import DVGeometry, geo_utils


class RegTestPyGeo(unittest.TestCase):

    N_PROCS = 1

    def setUp(self):
        # Store the path where this current script lives
        # This all paths in the script are relative to this path
        # This is needed to support testflo running directories and files as inputs
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def make_cube_ffd(self, file_name, x0, y0, z0, dx, dy, dz):
        # Write cube ffd with i along x-axis, j along y-axis, and k along z-axis
        axes = ["k", "j", "i"]
        slices = np.array(
            # Slice 1
            [
                [[[x0, y0, z0], [x0 + dx, y0, z0]], [[x0, y0 + dy, z0], [x0 + dx, y0 + dy, z0]]],
                # Slice 2
                [[[x0, y0, z0 + dz], [x0 + dx, y0, z0 + dz]], [[x0, y0 + dy, z0 + dz], [x0 + dx, y0 + dy, z0 + dz]]],
            ],
            dtype="d",
        )

        N0 = [2]
        N1 = [2]
        N2 = [2]

        geo_utils.write_wing_FFD_file(file_name, slices, N0, N1, N2, axes=axes)

    def setup_blocks(self, testID, isComplex=False):
        # Make tiny FFD
        ffd_name = f"../../input_files/tiny_cube_{testID:02d}.xyz"
        file_name = os.path.join(self.base_path, ffd_name)
        self.make_cube_ffd(file_name, 1, 1, 1, 1, 1, 1)
        tiny = DVGeometry(file_name, child=True, isComplex=isComplex)
        os.remove(file_name)
        tiny.addRefAxis("ref", xFraction=0.5, alignIndex="j", rotType=7)

        # Make tiny FFD
        ffd_name = f"../../input_files/small_cube_{testID:02d}.xyz"
        file_name = os.path.join(self.base_path, ffd_name)
        self.make_cube_ffd(file_name, 0, 0, 0, 2, 2, 2)
        small = DVGeometry(file_name, child=True, isComplex=isComplex)
        os.remove(file_name)
        small.addRefAxis("ref", xFraction=0.5, alignIndex="j")

        # Make big FFD
        ffd_name = f"../../input_files/big_cube_{testID:02d}.xyz"
        file_name = os.path.join(self.base_path, ffd_name)
        self.make_cube_ffd(file_name, 0, 0, 0, 3, 3, 3)
        big = DVGeometry(file_name, isComplex=isComplex)
        os.remove(file_name)
        big.addRefAxis("ref", xFraction=0.5, alignIndex="i")
        big.addChild(small)
        small.addChild(tiny)

        # Add point set
        points = np.array(
            [
                [0.5, 0.5, 0.5],
                [1.25, 1.25, 1.25],
                [1.5, 1.5, 1.5],
                [2.0, 2.5, 0.5],
            ]
        )
        big.addPointSet(points, "X")

        return big, small, tiny

    def compute_values(self, DVGeo, handler, refDeriv):
        # Calculate updated point coordinates
        Xnew = DVGeo.update("X")
        handler.root_add_val("pointvals", Xnew, rtol=1e-12, atol=1e-12, msg="Updated points")
        # Need to get design variables so that we can reset the Jacobians
        # for each call
        x = DVGeo.getValues()

        # Calculate Jacobians
        DVGeo.setDesignVars(x)
        DVGeo.update("X")
        DVGeo.computeTotalJacobian("X")
        Jac = DVGeo.JT["X"].toarray()

        DVGeo.setDesignVars(x)
        DVGeo.update("X")
        DVGeo.computeTotalJacobianCS("X")
        JacCS = DVGeo.JT["X"]

        DVGeo.setDesignVars(x)
        DVGeo.update("X")
        DVGeo.computeTotalJacobianFD("X")
        JacFD = DVGeo.JT["X"]

        if refDeriv:
            handler.root_add_val("jacobian", JacCS, rtol=1e-12, atol=1e-12, msg="Check jacobian")
        else:
            handler.root_add_val("jacobian", Jac, rtol=1e-12, atol=1e-12, msg="Check jacobian")

        # Test that they are equal to eachother
        np.testing.assert_allclose(Jac, JacCS, rtol=1e-12, atol=1e-12, err_msg="Analytic vs complex-step")
        np.testing.assert_allclose(Jac, JacFD, rtol=1e-6, atol=1e-6, err_msg="Analytic vs finite difference")

        # Make sure we reset everything
        DVGeo.setDesignVars(x)
        DVGeo.update("X")

        # Create dIdPt with one function for each point coordinate
        Npt = 4
        dIdPt = np.zeros([Npt * 3, Npt, 3])
        for i in range(Npt):
            dIdPt[i * 3 : (i + 1) * 3, i] = np.eye(3)

        # Test sensitivity dictionaries
        if refDeriv:
            # Generate reference from finite differences
            sens = commonUtils.totalSensitivityFD(DVGeo, Npt * 3, "X", step=1e-6)
            handler.root_add_dict("dIdx", sens, rtol=1e-6, atol=1e-6, msg="Check sens dict")
        else:
            # Compute the analytic derivatives
            sens = DVGeo.totalSensitivity(dIdPt, "X")
            handler.root_add_dict("dIdx", sens, rtol=1e-6, atol=1e-6, msg="Check sens dict")

    def train_1(self, train=True, refDeriv=True):
        self.test_1(train=train, refDeriv=refDeriv)

    def test_1(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_Blocks_01.ref")

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 1")

            big, small, tiny = self.setup_blocks(testID=1)
            add_vars(small, "small", rotate="y")
            add_vars(tiny, "tiny", rotate="y")

            # Rotate small cube by 45 and then rotate tiny cube by -45
            x = {}
            x["rotate_y_small"] = 45
            x["rotate_y_tiny"] = -45
            big.setDesignVars(x)

            # Compute tests
            self.compute_values(big, handler, refDeriv)

    def train_2(self, train=True, refDeriv=True):
        self.test_2(train=train, refDeriv=refDeriv)

    def test_2(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_Blocks_02.ref")

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 2")

            big, small, tiny = self.setup_blocks(testID=2)
            add_vars(big, "big", rotate="x")
            add_vars(small, "small", translate=True)
            add_vars(tiny, "tiny", rotate="y")

            # Modify design variables
            x = {}
            x["rotate_x_big"] = 10
            x["translate_small"] = [0.5, -1, 0.7]
            x["rotate_y_tiny"] = -20
            big.setDesignVars(x)

            # Compute tests
            self.compute_values(big, handler, refDeriv)

    def train_3(self, train=True, refDeriv=True):
        self.test_3(train=train, refDeriv=refDeriv)

    def test_3(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_Blocks_03.ref")

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 3")

            big, small, tiny = self.setup_blocks(testID=3)
            add_vars(small, "small", rotate="y")
            add_vars(tiny, "tiny", rotate="y", slocal=True)

            # Modify design variables
            x = big.getValues()
            x["rotate_y_small"] = 10
            x["rotate_y_tiny"] = -20
            np.random.seed(11)
            x["sectionlocal_tiny"] = np.random.random(*x["sectionlocal_tiny"].shape)
            big.setDesignVars(x)

            # Compute tests
            self.compute_values(big, handler, refDeriv)

    def train_4(self, train=True, refDeriv=True):
        self.test_4(train=train, refDeriv=refDeriv)

    def test_4(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_Blocks_04.ref")

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 4")

            big, small, tiny = self.setup_blocks(testID=4)

            # Add only translation variables
            add_vars(big, "big", translate=True)
            add_vars(small, "small", translate=True)
            add_vars(tiny, "tiny", translate=True)

            # Modify design variables
            x = big.getValues()
            np.random.seed(11)
            x["translate_big"] = np.random.random(3)
            x["translate_small"] = np.random.random(3)
            x["translate_tiny"] = np.random.random(3)
            big.setDesignVars(x)

            # Compute tests
            self.compute_values(big, handler, refDeriv)

    def train_5(self, train=True, refDeriv=True):
        self.test_5(train=train, refDeriv=refDeriv)

    def test_5(self, train=False, refDeriv=False):
        refFile = os.path.join(self.base_path, "ref/test_Blocks_05.ref")

        with BaseRegTest(refFile, train=train) as handler:
            handler.root_print("Test 5")

            if refDeriv:
                isComplex = True
            else:
                isComplex = False
            big, small, tiny = self.setup_blocks(testID=5, isComplex=isComplex)

            # Add only translation variables
            add_vars(big, "big", translate=True)
            add_vars(small, "small", rotate="y")
            add_vars(tiny, "tiny", rotate="y")

            # Modify design variables
            x = big.getValues()
            np.random.seed(11)
            x["translate_big"] = np.random.random(3)
            x["rotate_y_small"] = 45
            x["rotate_y_tiny"] = -45
            big.setDesignVars(x)

            # Compute tests
            big.setDesignVars(x)
            big.update("X")

            # Create dIdPt with one function for each point coordinate
            Npt = 4
            dIdPt = np.zeros([Npt * 3, Npt, 3])
            for i in range(Npt):
                dIdPt[i * 3 : (i + 1) * 3, i] = np.eye(3)

            # Test sensitivity dictionaries
            if refDeriv:
                # Generate reference from finite differences
                sens = commonUtils.totalSensitivityCS(big, Npt * 3, "X")
                handler.root_add_dict("dIdx", sens, rtol=1e-12, atol=1e-12, msg="Check sens dict")
            else:
                # Compute the analytic derivatives
                sens = big.totalSensitivity(dIdPt, "X")
                handler.root_add_dict("dIdx", sens, rtol=1e-12, atol=1e-12, msg="Check sens dict")


"""
The following are some helper functions for setting up the design variables for
the different test cases.
"""


def add_vars(geo, name, translate=False, rotate=None, scale=None, local=None, slocal=False):

    if translate:
        dvName = f"translate_{name}"
        geo.addGlobalDV(dvName=dvName, value=[0] * 3, func=f_translate)

    if rotate is not None:
        rot_funcs = {"x": f_rotate_x, "y": f_rotate_y, "z": f_rotate_z, "theta": f_rotate_theta}
        assert rotate in rot_funcs.keys()

        dvName = f"rotate_{rotate}_{name}"
        dvFunc = rot_funcs[rotate]
        geo.addGlobalDV(dvName=dvName, value=0, func=dvFunc)

    if local is not None:
        assert local in ["x", "y", "z"]
        dvName = f"local_{local}_{name}"
        geo.addLocalDV(dvName, axis=local)

    if slocal:
        dvName = f"sectionlocal_{name}"
        geo.addLocalSectionDV(dvName, secIndex="j", axis=1, orient0="i", orient2="ffd")


def f_translate(val, geo):
    C = geo.extractCoef("ref")
    for i in range(len(val)):
        C[:, i] += val[i]
    geo.restoreCoef(C, "ref")


def f_rotate_x(val, geo):
    geo.rot_x["ref"].coef[:] = val[0]


def f_rotate_y(val, geo):
    geo.rot_y["ref"].coef[:] = val[0]


def f_rotate_z(val, geo):
    geo.rot_z["ref"].coef[:] = val[0]


def f_rotate_theta(val, geo):
    geo.rot_theta["ref"].coef[:] = val[0]


if __name__ == "__main__":
    unittest.main()
