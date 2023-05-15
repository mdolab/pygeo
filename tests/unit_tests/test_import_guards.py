# Standard Python modules
import sys
import unittest
from unittest.mock import patch
import warnings


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)

    return do_test


class TestImportGuards(unittest.TestCase):
    N_PROCS = 1

    @ignore_warnings
    def test_DVGeometryVSP_import_openvsp(self):
        with patch.dict(sys.modules, {"openvsp": None}):
            # First party modules
            from pygeo import DVGeometryVSP

            with self.assertRaises(ImportError) as context:
                DVGeometryVSP("wing.vsp3")

            self.assertEqual(
                str(context.exception),
                "The OpenVSP Python API is required in order to use DVGeometryVSP. "
                + "Ensure OpenVSP is installed properly and can be found on your path.",
            )

    @ignore_warnings
    def test_DVGeometryVSP_openvsp_out_of_date(self):
        class DummyOpenVSPModule:
            def __init__(self):
                pass

            def GetVSPVersion(self):
                return "OpenVSP 0.0.0"

        dummy_module = DummyOpenVSPModule()
        with patch.dict(sys.modules, {"openvsp": dummy_module}):
            with self.assertRaises(AttributeError) as context:
                # First party modules
                from pygeo import DVGeometryVSP

                DVGeometryVSP("wing.vsp3")

            self.assertEqual(
                str(context.exception),
                "Out of date version of OpenVSP detected. "
                + "OpenVSP 3.28.0 or greater is required in order to use DVGeometryVSP",
            )

    @ignore_warnings
    def test_DVGeometryMulti_import_pysurf(self):
        with patch.dict(sys.modules, {"pysurf": None}):
            # First party modules
            from pygeo import DVGeometryMulti

            with self.assertRaises(ImportError) as context:
                DVGeometryMulti()

            self.assertEqual(str(context.exception), "pySurf is not installed and is required to use DVGeometryMulti.")

    @ignore_warnings
    def test_DVGeometryCST_import_prefoil(self):
        with patch.dict(sys.modules, {"prefoil": None}):
            # First party modules
            from pygeo import DVGeometryCST

            with self.assertRaises(ImportError) as context:
                DVGeometryCST("test.dat")

            self.assertEqual(str(context.exception), "preFoil is not installed and is required to use DVGeometryCST.")

    @ignore_warnings
    def test_DVGeometryESP_import_ocsm(self):
        with patch.dict(sys.modules, {"pyOCSM": None}):
            # First party modules
            from pygeo import DVGeometryESP

            with self.assertRaises(ImportError) as context:
                DVGeometryESP("test.csm")

            self.assertEqual(str(context.exception), "OCSM and pyOCSM must be installed to use DVGeometryESP.")


if __name__ == "__main__":
    unittest.main()
