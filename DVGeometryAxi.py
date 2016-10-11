from __future__ import print_function

import numpy as np 
from scipy import sparse

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

        # self.pts = pts.copy()

        # re-order the columns to a consistent alpha, beta, gamma frame
        alpha = pts[:, self.alpha_idx]  
        beta = pts[:, self.beta_idx] - center[self.beta_idx]
        gamma = pts[:, self.gamma_idx] - center[self.gamma_idx]

        self.radii = np.sqrt(beta**2 + gamma**2)
        self.thetas = np.arctan2(gamma, beta)

        self.cos_thetas = np.cos(self.thetas)
        self.sin_thetas = np.sin(self.thetas)

        # transformation jacobian to account for the axisymmetric transformation
        n_pts = len(pts)

        row = np.empty(3*n_pts, dtype="int")
        col = np.empty(3*n_pts, dtype="int")
        data = np.empty(3*n_pts)
        # for j in range(n_pts): 
        #     # Pt_j_x
        #     row[j] = 0 + 3*j
        #     col[j] = self.alpha_idx + 3*j
        #     data[j] = 1.
        #     # Pt_j_y
        #     row[j+1] = 1 + 3*j
        #     col[j+1] = self.beta_idx + 3*j
        #     data[j+1] = self.sin_thetas[j]
        #     # Pt_j_z
        #     row[j+2] = 2 + 3*j
        #     col[j+2] = self.beta_idx + 3*j
        #     data[j+2] = self.cos_thetas[j]

        # vectorized
        idx = 3*np.arange(n_pts, dtype="int")
        row[0::3] = idx
        row[1::3] = 1+idx
        row[2::3] = 2+idx

        col[0::3] = self.alpha_idx + idx
        col[1::3] = self.beta_idx + idx
        col[2::3] = self.beta_idx + idx

        data[0::3] = 1.
        data[1::3] = self.sin_thetas
        data[2::3] = self.cos_thetas

        # derivative of the Cartesian points w.r.t the collapsed axi-symmetric points
        self.dPtCdPtA = sparse.coo_matrix((data, (row, col)), 
                                          shape=(3*n_pts, 3*n_pts)).tocsr()

        # points collapsed into the prescribed plane
        # self.c_pts_axi = np.vstack((self.alpha, self.radii, np.zeros(self.n_points))).T
        self.c_pts = np.empty((self.n_points, 3))
        self.c_pts[:, 0] = alpha 
        self.c_pts[:, self.beta_idx] = self.radii
        self.c_pts[:, self.gamma_idx] = 0.  # no need to store zeros

    def expand(self, new_c_pts): 
        """given new collapsed points, re-expands them into physical space"""

        self.c_pts = new_c_pts 
        pts = np.empty((self.n_points, 3))
        pts[:, self.alpha_idx] = new_c_pts[:, 0] 

        new_rads = new_c_pts[:, self.beta_idx]
        pts[:, self.beta_idx] = new_rads*self.cos_thetas + self.center[self.beta_idx]
        pts[:, self.gamma_idx] = new_rads*self.sin_thetas + self.center[self.gamma_idx]

        return pts


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

    def computeTotalJacobian(self, ptSetName, config=None): 
        """ compute the total point jacobian in CSR format since we
        need this for TACS"""

        super(DVGeometryAxi, self).computeTotalJacobian(ptSetName, config)

        xform = self.axiTransforms[ptSetName]

        self.JT[ptSetName] = xform.dPtCdPtA.dot(self.JT[ptSetName].T).T


    # TODO JSG: the computeTotalJacobianFD method is broken in DVGeometry Base class
    # def computeTotalJacobianFD(self, ptSetName, config=None): 

    #     super(DVGeometryAxi, self).computeTotalJacobianFD(ptSetName, config)

    #     xform = self.axiTransforms[ptSetName]

    #     self.JT[ptSetName] = xform.dPtCdPtA.dot(self.JT[ptSetName].T).T






