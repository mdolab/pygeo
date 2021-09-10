# Imports
import numpy as np
from pygeo import pyGeo
import filecmp as fc
import unittest
import os


class TestPyGeo(unittest.TestCase):
    def setUp(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def testFiles(self):
        dirName = os.path.join(self.base_path, "../../input_files")
        # dirName = "../../input_files"
        fileName = "wing"

        # names of output files
        datOut = fileName + ".dat"
        igsOut = fileName + ".igs"
        tinOut = fileName + ".tin"

        # names + locations of reference files
        datRef = dirName + "/" + fileName + ".dat"
        igsRef = dirName + "/" + fileName + ".igs"
        tinRef = dirName + "/" + fileName + ".tin"

        # Airfoil file
        airfoil_list = [dirName + "/rae2822.dat"] * 2
        naf = len(airfoil_list)  # number of airfoils

        # Wing definition
        # Airfoil leading edge positions
        x = [0.0, 7.5]
        y = [0.0, 0.0]
        z = [0.0, 14.0]
        offset = np.zeros((naf, 2))  # x-y offset applied to airfoil position before scaling

        # Airfoil rotations
        rot_x = [0.0, 0.0]
        rot_y = [0.0, 0.0]
        rot_z = [0.0, 0.0]

        # Airfoil scaling
        chord = [5.0, 1.5]  # chord lengths

        # Run pyGeo
        wing = pyGeo(
            "liftingSurface",
            xsections=airfoil_list,
            scale=chord,
            offset=offset,
            x=x,
            y=y,
            z=z,
            rotX=rot_x,
            rotY=rot_y,
            rotZ=rot_z,
            tip="rounded",
            bluntTe=True,
            squareTeTip=True,
            teHeight=0.25 * 0.0254,
        )

        # Write output files
        wing.writeTecplot(datOut)
        wing.writeIGES(igsOut)
        wing.writeTin(tinOut)

        f = open(tinOut, "r")
        lines = f.readlines()
        f.close()

        del lines[1]
        new = open(tinOut, "w")
        for line in lines:
            new.write(line)
        new.close()

        self.assertTrue((fc.cmp(datOut, datRef, shallow=False)))
        self.assertTrue((fc.cmp(igsOut, igsRef, shallow=False)))
        self.assertTrue((fc.cmp(tinOut, tinRef, shallow=False)))

        os.remove(datOut)
        os.remove(igsOut)
        os.remove(tinOut)
