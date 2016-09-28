# from __future__ import print_function

import numpy as np 

try:
    from collections import OrderedDict
except ImportError:
    try:
        from ordereddict import OrderedDict
    except ImportError:
        print("Could not find any OrderedDict class. For 2.6 and earlier, "
              "use:\n pip install ordereddict")

from . import DVGeometry 

AXES_2_IDX = {"x": 0, "y": 1, "z": 2}
AXES = set(['x', 'y', 'z'])

class _AxiTransform(object): 
    """Collapses a set of cartesian coordiantes into a single plane to allow 
    for axi-symmetric FFD. Also expands them back to their original annular location

    Parameters
    -----------

    pts: (n,3) 
        array of points to be transformed

    center: (3,)
        center of the axi-symmetric body;
        This can be any point along the rotation axis of the body

    collapse_into: 2-tuple of strings
        Two coordinate axes that you wish to collapse your points into. This should 
        align with the specific direction you are moving FFD control points. The first item in
        the tuple is taken as the rotation axis for the axi-symmetric coordinate system. So ("x","z")
        means to collapse into the x,z plane and use x as the rotational axis. ("z", "x") means to collapse 
        into the x,z plane and use z as the rotational axis
    """ 

    def __init__(self, pts, center, collapse_into): 
        self.c_plane = collapse_into
        self.n_points = pts.shape[0]
        self.center = center

        self.alpha_idx = AXES_2_IDX[self.c_plane[0]] 
        self.beta_idx = AXES_2_IDX[self.c_plane[1]]
        self.gamma_idx = AXES_2_IDX[AXES.difference(set(self.c_plane)).pop()]  # which ever one isn't in the c_plane tuple!

        self.pts = pts.copy()

        # re-order the columns to a consistent alpha, beta, gamma frame
        self.alpha = self.pts[:, self.alpha_idx]  
        self.beta = self.pts[:, self.beta_idx] - center[self.beta_idx]
        self.gamma = self.pts[:, self.gamma_idx] - center[self.gamma_idx]

        self.radii = np.sqrt(self.beta**2 + self.gamma**2)
        self.thetas = np.arctan2(self.gamma, self.beta)

        # points collapsed into the prescribed plane
        # self.c_pts_axi = np.vstack((self.alpha, self.radii, np.zeros(self.n_points))).T
        self.c_pts = np.empty(self.pts.shape)
        self.c_pts[:, 0] = self.alpha 
        self.c_pts[:, self.beta_idx] = self.radii
        self.c_pts[:, self.gamma_idx] = 0.

    def expand(self, new_c_pts): 
        """given new collapsed points, re-expands them into physical space"""

        self.c_pts = new_c_pts 
        self.pts[:, self.alpha_idx] = new_c_pts[:, 0] 
        self.pts[:, self.beta_idx] = new_c_pts[:, self.beta_idx]*np.cos(self.thetas) + self.center[self.beta_idx]
        self.pts[:, self.gamma_idx] = new_c_pts[:, self.beta_idx]*np.sin(self.thetas) + self.center[self.gamma_idx]

        return self.pts

class DVGeometryAxi(DVGeometry):


    def __init__(self, fileName, complex=False, child=False, *args, **kwargs): 

        self.axiTransforms = OrderedDict()  # TODO: Why is this ordered? 

        super(DVGeometryAxi, self).__init__(fileName, complex, child, *args, **kwargs)

    def addPointSet(self, points, center, collapse_into, ptName, origConfig=True, **kwargs): 
        """
        Add a set of coordinates to DVGeometry

        The is the main way that geometry, in the form of a coordinate
        list is given to DVGeoemtry to be manipulated.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed. These cordinates *should* all
            project into the interior of the FFD volume. 
        center : array, size (3,1)
            The center about which the axisymmetric FFD should be applied.
            This can be any point along the rotation axis of the body
        collapse_into: 2-tuple of strings
            Two coordinate axes that you wish to collapse your points into. 
            This should align with the directions you are moving FFD 
            control points. The first item in the tuple is taken as the rotation 
            axis for the axi-symmetric coordinate system. So ("x","z") means to 
            collapse into the x,z plane and use x as the rotational axis. 
            ("z", "x") means to collapse into the x,z plane and use z as the 
            rotational axis
        ptName : str
            A user supplied name to associate with the set of
            coordinates. This name will need to be provided when
            updating the coordinates or when getting the derivatives
            of the coordinates.
        origConfig : bool
            Flag determine if the coordinates are projected into the
            undeformed or deformed configuration. This should almost
            always be True except in circumstances when the user knows
            exactly what they are doing."""

        xform = self.axiTransforms[ptName] = _AxiTransform(points, center, collapse_into)

        super(DVGeometryAxi, self).addPointSet(xform.c_pts, ptName, origConfig, **kwargs)

    def update(self, ptSetName, childDelta=True, config=None): 

        new_c_pts = super(DVGeometryAxi, self).update(ptSetName, childDelta, config)

        xform = self.axiTransforms[ptSetName]
        coords = xform.expand(new_c_pts)
        # coords = new_c_pts

        return coords 


