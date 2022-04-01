"""
==============================================================================
DVGeoCST: Test suite for the DVGeoCST module.
==============================================================================
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Extension modules
# ==============================================================================
from pygeo import DVGeometryCST

class DVGeometryCSTUnitTest(unittest.TestCase):

    N_PROCS = 1

    def setUp(self):
        self.sensTol = 1e-10
        self.coordTol = 1e-10

    def test_CSTCoords(self):
        """Test that for w_i = 1, the CST curve has the expected shape
        """
        x = np.linspace(0,1,100)
        N1 = 0.5
        N2 = 1.0
        yte = 1e-2
        yExact = np.sqrt(x)*(1-x) + yte*x
        plt.plot(x, yExact, 'k-', label='Exact')
        maxNumCoeff = 10
        for n in range(1, maxNumCoeff+1):
            w = np.ones(n)
            y = DVGeometryCST.computeCSTCoordinates(x, N1, N2, w, yte)
            plt.plot(x, y, '-', label='n = %d' % n)
        plt.show()
        np.testing.assert_allclose(y, yExact, atol=self.coordTol, rtol=self.coordTol)

if __name__ == '__main__':
    unittest.main()