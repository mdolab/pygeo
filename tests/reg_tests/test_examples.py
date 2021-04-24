import unittest
import os
import subprocess
from parameterized import parameterized

baseDir = os.path.dirname(os.path.abspath(__file__))


class TestExamples(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.output_file_list = []

    def common_test(self, test_dir, run_file, args=None):
        """
        This function runs a given Python script, makes sure it does not exit with an error code,
        and then checks that the output files are created.

        Parameters
        ----------
        test_dir : str
            The file directory, relative to where this file is
        run_file : str
            The name of the python script to run
        args : list, optional
            The list of command line arguments, by default None
        """
        if args is None:
            args = []
        full_test_dir = os.path.abspath(os.path.join(baseDir, "../../examples", test_dir))
        os.chdir(full_test_dir)
        cmd = ["python", run_file] + args
        subprocess.check_call(cmd)
        for f in self.output_file_list:
            self.assertTrue(os.path.isfile(f))

    def test_bwb(self):
        self.output_file_list = ["bwb.igs"]
        self.common_test("bwb", "bwb.py")

    def test_c172(self):
        self.output_file_list = [
            "c172_sharp_te_rounded_tip.dat",
            "c172_sharp_te_rounded_tip.igs",
            "c172_sharp_te_pinched_tip.dat",
            "c172_sharp_te_pinched_tip.igs",
            "c172_sharp_te_rounded_tip_fitted.dat",
            "c172_sharp_te_rounded_tip_fitted.igs",
            "c172_blunt_te_rounded_tip.dat",
            "c172_blunt_te_rounded_tip.igs",
            "c172_rounded_te_rounded_tip.dat",
            "c172_rounded_te_rounded_tip.igs",
        ]
        self.common_test("c172_wing", "c172.py")

    @parameterized.expand(["iges", "plot3d", "liftingsurface"])
    def test_deform(self, input_type):
        self.output_file_list = ["wingNew.plt"]
        self.common_test("deform_geometry", "runScript.py", args=["--input_type", input_type])

    def tearDown(self):
        for f in self.output_file_list:
            os.remove(f)
        os.chdir(self.cwd)
