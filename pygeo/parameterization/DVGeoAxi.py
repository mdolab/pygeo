# Standard Python modules
from collections import OrderedDict
import warnings

# External modules
import numpy as np
from scipy import sparse

# Local modules
from .DVGeo import DVGeometry

AXES_2_IDX = {"x": 0, "y": 1, "z": 2}
AXES = {"x", "y", "z"}


class _AxiTransform:
    """Collapses a set of cartesian coordinates into a single plane to allow
    for axi-symmetric FFD. Also expands them back to their original annular location

    Parameters
    ----------
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

    def __init__(self, pts, center, collapse_into, isComplex=False, **kwargs):
        # FIXME: for backwards compatibility we still allow the argument complex=True/False
        # which we now check in kwargs and overwrite
        if "complex" in kwargs:
            isComplex = kwargs.pop("complex")
            warnings.warn("The keyword argument 'complex' is deprecated, use 'isComplex' instead.", stacklevel=2)

        self.complex = isComplex

        self.c_plane = collapse_into
        self.n_points = pts.shape[0]
        self.center = center

        self.alpha_idx = AXES_2_IDX[self.c_plane[0]]
        self.beta_idx = AXES_2_IDX[self.c_plane[1]]
        self.gamma_idx = AXES_2_IDX[
            AXES.difference(set(self.c_plane)).pop()
        ]  # which ever one isn't in the c_plane tuple!

        # self.pts = pts.copy()

        # re-order the columns to a consistent alpha, beta, gamma frame
        alpha = pts[:, self.alpha_idx]
        beta = pts[:, self.beta_idx] - center[self.beta_idx]
        gamma = pts[:, self.gamma_idx] - center[self.gamma_idx]

        self.radii = np.sqrt(beta**2 + gamma**2)
        # need to get the real part because arctan2 is not complex save
        # but its ok, becuase these are constants
        self.thetas = np.arctan2(gamma.real, beta.real)

        self.cos_thetas = np.cos(self.thetas)
        self.sin_thetas = np.sin(self.thetas)

        # transformation jacobian to account for the axisymmetric transformation
        n_pts = len(pts)

        row = np.empty(3 * n_pts, dtype="int")
        col = np.empty(3 * n_pts, dtype="int")
        data = np.empty(3 * n_pts)
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
        idx = 3 * np.arange(n_pts, dtype="int")
        row[0::3] = idx
        row[1::3] = 1 + idx
        row[2::3] = 2 + idx

        col[0::3] = self.alpha_idx + idx
        col[1::3] = self.beta_idx + idx
        col[2::3] = self.beta_idx + idx

        data[0::3] = 1.0
        data[1::3] = self.sin_thetas
        data[2::3] = self.cos_thetas

        # derivative of the Cartesian points w.r.t the collapsed axi-symmetric points
        self.dPtCdPtA = sparse.coo_matrix((data, (row, col)), shape=(3 * n_pts, 3 * n_pts)).tocsr()

        # points collapsed into the prescribed plane
        # self.c_pts_axi = np.vstack((self.alpha, self.radii, np.zeros(self.n_points))).T
        if self.complex:
            self.c_pts = np.empty((self.n_points, 3), dtype="complex")
        else:
            self.c_pts = np.empty((self.n_points, 3))

        self.c_pts[:, 0] = alpha
        self.c_pts[:, self.beta_idx] = self.radii
        self.c_pts[:, self.gamma_idx] = 0.0  # no need to store zeros

    def expand(self, new_c_pts):
        """Given new collapsed points, re-expands them into physical space"""

        self.c_pts = new_c_pts
        if self.complex:
            pts = np.empty((self.n_points, 3), dtype="complex")
        else:
            pts = np.empty((self.n_points, 3))
        pts[:, self.alpha_idx] = new_c_pts[:, 0]

        new_rads = new_c_pts[:, self.beta_idx]
        pts[:, self.beta_idx] = new_rads * self.cos_thetas + self.center[self.beta_idx]
        pts[:, self.gamma_idx] = new_rads * self.sin_thetas + self.center[self.gamma_idx]

        return pts


class DVGeometryAxi(DVGeometry):
    """
    A class for manipulating geometry.

    The purpose of the DVGeometry class is to provide a mapping from
    user-supplied design variables to an arbitrary set of discrete,
    three-dimensional coordinates. These three-dimensional coordinates
    can in general represent anything, but will typically be the
    surface of an aerodynamic mesh, the nodes of a FE mesh or the
    nodes of another geometric construct.

    In a very general sense, DVGeometry performs two primary
    functions:

    1. Given a new set of design variables, update the
       three-dimensional coordinates: :math:`X_{DV}\\rightarrow
       X_{pt}` where :math:`X_{pt}` are the coordinates and :math:`X_{DV}`
       are the user variables.

    2. Determine the derivative of the coordinates with respect to the
       design variables. That is the derivative :math:`\\frac{dX_{pt}}{dX_{DV}}`

    DVGeometry uses the *Free-Form Deformation* approach for geometry
    manipulation. The basic idea is the coordinates are *embedded* in
    a clear-flexible jelly-like block. Then by stretching moving and
    'poking' the volume, the coordinates that are embedded inside move
    along with overall deformation of the volume.

    Parameters
    ----------
    fileName : str
       filename of FFD file. This must be a ascii formatted plot3D file
       in fortran ordering.
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
    complex : bool
        Make the entire object complex. This should **only** be used when
        debugging the entire tool-chain with the complex step method.

    child : bool
        Flag to indicate that this object is a child of parent DVGeo object


    Examples
    --------
    The general sequence of operations for using DVGeometry is as follows::
      >>> from pygeo import DVGeometryAxi
      >>> DVGeo = DVGeometryAxi('FFD_file.fmt', center=(0., 0., 0.), collapse_into=("x", "z"))
      >>> # Embed a set of coordinates Xpt into the object
      >>> DVGeo.addPointSet(Xpt, 'myPoints')
      >>> # Associate a 'reference axis' for large-scale manipulation
      >>> DVGeo.addRefAxis('wing_axis', axis_curve)
      >>> # Define a global design variable function:
      >>> def twist(val, geo):
      >>>     geo.rot_z['wing_axis'].coef[:] = val[:]
      >>> # Now add this as a global variable:
      >>> DVGeo.addGlobalDV('wing_twist', 0.0, twist, lower=-10, upper=10)
      >>> # Now add local (shape) variables
      >>> DVGeo.addLocalDV('shape', lower=-0.5, upper=0.5, axis='y')
    """

    def __init__(self, fileName, center, collapse_into, *args, isComplex=False, child=False, **kwargs):
        self.axiTransforms = OrderedDict()  # TODO: Why is this ordered?

        # FIXME: for backwards compatibility we still allow the argument complex=True/False
        # which we now check in kwargs and overwrite
        if "complex" in kwargs:
            isComplex = kwargs.pop("complex")
            warnings.warn("The keyword argument 'complex' is deprecated, use 'isComplex' instead.", stacklevel=2)

        super().__init__(fileName, *args, isComplex=isComplex, child=child, **kwargs)

        self.center = center
        self.collapse_into = collapse_into

    def addPointSet(self, points, ptName, origConfig=True, **kwargs):
        """
        Add a set of coordinates to DVGeometry

        The is the main way that geometry, in the form of a coordinate
        list is given to DVGeometry to be manipulated.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed. These coordinates *should* all
            project into the interior of the FFD volume.
        ptName : str
            A user supplied name to associate with the set of
            coordinates. This name will need to be provided when
            updating the coordinates or when getting the derivatives
            of the coordinates.
        origConfig : bool
            Flag determine if the coordinates are projected into the
            undeformed or deformed configuration. This should almost
            always be True except in circumstances when the user knows
            exactly what they are doing.
        """

        xform = self.axiTransforms[ptName] = _AxiTransform(points, self.center, self.collapse_into, self.complex)

        super().addPointSet(xform.c_pts, ptName, origConfig, **kwargs)

    def update(self, ptSetName, childDelta=True, config=None):
        new_c_pts = super().update(ptSetName, childDelta, config)

        xform = self.axiTransforms[ptSetName]
        coords = xform.expand(new_c_pts)
        # coords = new_c_pts

        return coords

    def computeTotalJacobian(self, ptSetName, config=None):
        """Compute the total point jacobian in CSR format since we
        need this for TACS
        """

        super().computeTotalJacobian(ptSetName, config)
        if self.JT[ptSetName] is not None:
            xform = self.axiTransforms[ptSetName]

            self.JT[ptSetName] = xform.dPtCdPtA.dot(self.JT[ptSetName].T).T

    # TODO JSG: the computeTotalJacobianFD method is broken in DVGeometry Base class
    # def computeTotalJacobianFD(self, ptSetName, config=None):

    #     super(DVGeometryAxi, self).computeTotalJacobianFD(ptSetName, config)

    #     xform = self.axiTransforms[ptSetName]

    #     self.JT[ptSetName] = xform.dPtCdPtA.dot(self.JT[ptSetName].T).T
