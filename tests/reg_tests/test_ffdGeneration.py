import os
import unittest
import numpy as np
from stl.mesh import Mesh
from pygeo import DVGeometry
from pygeo.geo_utils import createFittedWingFFD

baseDir = os.path.dirname(os.path.abspath(__file__))


class TestFittedFFD(unittest.TestCase):

    N_PROCS = 1

    def test_c172_fitted(self):

        # Scale all dimensions from millimeters to meters so that the tolerances match a regular use case
        leList = np.array([[0.0, 0.0, 0.1], [10.0, 0.0, 2500.0], [160.0, 0.0, 5280.0]]) * 1e-3
        teList = np.array([[1600.0, 0.0, 0.1], [1650.0, 0.0, 2500.0], [1320.0, 0.0, 5280.0]]) * 1e-3

        meshFile = os.path.join(baseDir, "../../input_files/c172.stl")
        stlMesh = Mesh.from_file(meshFile)
        p0 = stlMesh.vectors[:, 0, :] * 1e-3
        p1 = stlMesh.vectors[:, 1, :] * 1e-3
        p2 = stlMesh.vectors[:, 2, :] * 1e-3
        surf = [p0, p1, p2]
        surfFormat = "point-point"
        outFile = "wing_ffd.xyz"
        nSpan = [4, 4]
        nChord = 8
        relMargins = [0.02, 0.01, 0.01]
        absMargins = [0.04, 0.01, 0.02]
        liftIndex = 2
        createFittedWingFFD(surf, surfFormat, outFile, leList, teList, nSpan, nChord, absMargins, relMargins, liftIndex)

        # Check that the generated FFD file matches the reference
        referenceFFD = DVGeometry(os.path.join(baseDir, "../../input_files/c172_fitted.xyz"))
        outputFFD = DVGeometry(outFile)
        np.testing.assert_allclose(referenceFFD.FFD.coef, outputFFD.FFD.coef, rtol=1e-15)

        # Check that the embedding works
        # This is not an actual test because no errors are raised if the projection does not work
        DVGeo = DVGeometry(outFile)
        DVGeo.addPointSet(p0, "wing_p0")
        DVGeo.addPointSet(p1, "wing_p1")
        DVGeo.addPointSet(p2, "wing_p2")

        # Delete the generated FFD file
        os.remove(outFile)


if __name__ == "__main__":
    unittest.main()
