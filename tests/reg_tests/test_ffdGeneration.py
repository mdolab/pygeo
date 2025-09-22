# Standard Python modules
import os
import unittest

# External modules
import numpy as np
from stl.mesh import Mesh

# First party modules
from pygeo import DVGeometry
from pygeo.geo_utils import createFittedWingFFD, createMidsurfaceMesh, write_wing_FFD_file

baseDir = os.path.dirname(os.path.abspath(__file__))


class TestFFDGeneration(unittest.TestCase):
    N_PROCS = 1

    def setUp(self):
        # Get the surface definition from the STL file
        meshFile = os.path.join(baseDir, "../../input_files/c172.stl")
        stlMesh = Mesh.from_file(meshFile)
        self.p0 = stlMesh.vectors[:, 0, :] * 1e-3
        self.p1 = stlMesh.vectors[:, 1, :] * 1e-3
        self.p2 = stlMesh.vectors[:, 2, :] * 1e-3
        self.surf = [self.p0, self.p1, self.p2]
        self.surfFormat = "point-point"
        self.liftIndex = 2

    def test_box_ffd(self, train=False, refDeriv=False):
        # Write duplicate of outerBoxFFD
        axes = ["i", "k", "j"]
        slices = np.array(
            [
                # Slice 1
                [[[-1, -1, -1], [-1, 1, -1]], [[-1, -1, 1], [-1, 1, 1]]],
                # Slice 2
                [[[1, -1, -1], [1, 1, -1]], [[1, -1, 1], [1, 1, 1]]],
                # Slice 3
                [[[2, -1, -1], [2, 1, -1]], [[2, -1, 1], [2, 1, 1]]],
            ]
        )

        N0 = [2, 2]
        N1 = [2, 2]
        N2 = [2, 2]

        copyName = os.path.join(baseDir, "../../input_files/test1.xyz")
        write_wing_FFD_file(copyName, slices, N0, N1, N2, axes=axes)

        # Load original and duplicate
        origFFD = DVGeometry(os.path.join(baseDir, "../../input_files/outerBoxFFD.xyz"))
        copyFFD = DVGeometry(copyName)
        np.testing.assert_allclose(origFFD.FFD.coef, copyFFD.FFD.coef, rtol=1e-7)

        # Delete the duplicate FFD file
        os.remove(copyName)

    def test_c172_fitted(self):
        # Scale all dimensions from millimeters to meters so that the tolerances match a regular use case
        leList = np.array([[0.0, 0.0, 0.1], [10.0, 0.0, 2500.0], [160.0, 0.0, 5280.0]]) * 1e-3
        teList = np.array([[1600.0, 0.0, 0.1], [1650.0, 0.0, 2500.0], [1320.0, 0.0, 5280.0]]) * 1e-3

        # Set the other FFD generation inputs
        outFile = "wing_ffd.xyz"
        nSpan = [4, 4]
        nChord = 8
        relMargins = [0.02, 0.01, 0.01]
        absMargins = [0.04, 0.01, 0.02]

        createFittedWingFFD(
            self.surf,
            self.surfFormat,
            outFile,
            leList,
            teList,
            nSpan,
            nChord,
            absMargins,
            relMargins,
            self.liftIndex,
        )

        # Check that the generated FFD file matches the reference
        referenceFFD = DVGeometry(os.path.join(baseDir, "../../input_files/c172_fitted.xyz"))
        outputFFD = DVGeometry(outFile)
        np.testing.assert_allclose(referenceFFD.FFD.coef, outputFFD.FFD.coef, rtol=1e-13)

        # Check that the embedding works
        # This is not an actual test because no errors are raised if the projection does not work
        DVGeo = DVGeometry(outFile)
        DVGeo.addPointSet(self.p0, "wing_p0")
        DVGeo.addPointSet(self.p1, "wing_p1")
        DVGeo.addPointSet(self.p2, "wing_p2")

        # Delete the generated FFD file
        os.remove(outFile)

    def test_c172_midsurface_mesh(self):
        # These LE and TE coordinates account for the twist in the wing unlike those used in test_c172_fitted
        x = [0, 0, 0.125 * 1.18]
        y = [0, 0, 0]
        z = [0.0, 2.5, 10.58 / 2]
        chord = [1.67, 1.67, 1.18]
        rot_z = [0, 0, 2]
        leList = np.array([x, y, z]).T
        teList = np.array([x, y, z]).T
        for ii in range(len(chord)):
            teList[ii] = leList[ii] + np.array(
                [chord[ii] * np.cos(np.deg2rad(rot_z[ii])), chord[ii] * np.sin(np.deg2rad(rot_z[ii])), 0]
            )
        nSpan = 21
        nChord = 21
        mesh = createMidsurfaceMesh(
            self.surf, self.surfFormat, leList, teList, nSpan, nChord, self.liftIndex, chordCosSpacing=0.75
        )

        # Check that the generated mesh matches the reference
        flattenedMesh = mesh.reshape((-1, 3))
        refMesh = np.loadtxt(os.path.join(baseDir, "../../input_files/c172-midsurfaceMesh.txt"))
        np.testing.assert_allclose(flattenedMesh, refMesh, rtol=1e-13)

        # Check that the generated mesh fits inside the FFD
        # This is not an actual test because no errors are raised if the projection does not work
        DVGeo = DVGeometry(os.path.join(baseDir, "../../input_files/c172_fitted.xyz"))
        DVGeo.addPointSet(flattenedMesh, "midsurfaceMesh")


if __name__ == "__main__":
    unittest.main()
