import unittest
import numpy as np
from pygeo import DVGeometry, geo_utils
from test_Blocks import add_vars


class RegTestPyGeo(unittest.TestCase):

    N_PROCS = 1

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

    def test_parent_shape_child_rot(self, train=False, refDeriv=False):
        ffd_name = "../../input_files/small_cube.xyz"
        self.make_cube_ffd(ffd_name, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8)
        small = DVGeometry(ffd_name, child=True)
        small.addRefAxis("ref", xFraction=0.5, alignIndex="j")

        x0 = 0.0
        y0 = 0.0
        z0 = 0.0
        dx = 1.0
        dy = 1.0
        dz = 1.0

        axes = ["k", "j", "i"]
        slices = np.array(
            # Slice 1
            [
                [
                    [[x0, y0, z0], [x0 + dx, y0, z0], [x0 + 2 * dx, y0, z0]],
                    [[x0, y0 + dy, z0], [x0 + dx, y0 + dy, z0], [x0 + 2 * dx, y0 + dy, z0]],
                ],
                # Slice 2
                [
                    [[x0, y0, z0 + dz], [x0 + dx, y0, z0 + dz], [x0 + 2 * dx, y0, z0 + dz]],
                    [[x0, y0 + dy, z0 + dz], [x0 + dx, y0 + dy, z0 + dz], [x0 + 2 * dx, y0 + dy, z0 + dz]],
                ],
            ],
            dtype="d",
        )

        N0 = [2]
        N1 = [2]
        N2 = [3]
        ffd_name = "../../input_files/big_cube.xyz"

        geo_utils.write_wing_FFD_file(ffd_name, slices, N0, N1, N2, axes=axes)
        big = DVGeometry(ffd_name)
        big.addRefAxis("ref", xFraction=0.5, alignIndex="j")
        big.addChild(small)

        # Add point set
        points = np.array([[0.5, 0.5, 0.5]])
        big.addPointSet(points, "X")

        # Add only translation variables
        add_vars(big, "big", local="z", rotate="y")
        add_vars(small, "small", rotate="y")

        ang = 45
        ang_r = np.deg2rad(ang)

        # Modify design variables
        x = big.getValues()

        # add a local shape change
        x["local_z_big"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        big.setDesignVars(x)
        Xs = big.update("X")

        # add a rotation of the child FFD
        x["rotate_y_small"] = ang
        big.setDesignVars(x)
        Xrot_ffd = big.update("X")

        # the modification caused by the child FFD should be the same as rotating the deformed point of the parent
        # (you would think)
        rot_mat = np.array([[np.cos(ang_r), 0, np.sin(ang_r)], [0, 1, 0], [-np.sin(ang_r), 0, np.cos(ang_r)]])
        Xrot = np.dot(rot_mat, (Xs - points).T) + points.T

        np.testing.assert_array_almost_equal(Xrot_ffd.T, Xrot)


if __name__ == "__main__":
    unittest.main()
