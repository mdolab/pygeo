# ======================================================================
#         Imports
# ======================================================================
import numpy as np
from geo_utils import misc


class geoDVGlobal(object):
    def __init__(self, dv_name, value, lower, upper, scale, function, config):
        """Create a geometric design variable (or design variable group)
        See addGlobalDV in DVGeometry class for more information
        """
        self.name = dv_name
        self.value = np.atleast_1d(np.array(value)).astype("D")
        self.nVal = len(self.value)
        self.lower = None
        self.upper = None
        self.config = config
        self.function = function
        if lower is not None:
            self.lower = misc.convertTo1D(lower, self.nVal)
        if upper is not None:
            self.upper = misc.convertTo1D(upper, self.nVal)
        if scale is not None:
            self.scale = misc.convertTo1D(scale, self.nVal)

    def __call__(self, geo, config):
        """When the object is called, actually apply the function"""
        # Run the user-supplied function
        d = np.dtype(complex)

        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            # If the geo object is complex, which is indicated by .coef
            # being complex, run with complex numbers. Otherwise, convert
            # to real before calling. This eliminates casting warnings.
            if geo.coef.dtype == d or geo.complex:
                return self.function(self.value, geo)
            else:
                return self.function(np.real(self.value), geo)
