# Standard Python modules
from collections import OrderedDict

# External modules
from baseclasses.utils import Error
import numpy as np
from pyspline import Curve

# Local modules
from .. import geo_utils, pyGeo
from ..geo_utils.file_io import readPlot3DSurfFile
from ..geo_utils.misc import convertTo2D
from .areaConstraint import ProjectedAreaConstraint, SurfaceAreaConstraint, TriangulatedSurfaceConstraint
from .baseConstraint import GlobalLinearConstraint, LinearConstraint
from .circularityConstraint import CircularityConstraint
from .colinearityConstraint import ColinearityConstraint
from .curvatureConstraint import CurvatureConstraint, CurvatureConstraint1D
from .gearPostConstraint import GearPostConstraint
from .locationConstraint import LocationConstraint
from .planarityConstraint import PlanarityConstraint
from .radiusConstraint import RadiusConstraint
from .thicknessConstraint import ThicknessConstraint, ThicknessToChordConstraint
from .volumeConstraint import CompositeVolumeConstraint, TriangulatedVolumeConstraint, VolumeConstraint


class DVConstraints:
    """DVConstraints provides a convenient way of defining geometric
    constraints for wings. This can be very useful for a constrained
    aerodynamic or aerostructural optimization. Three types of
    constraints are supported:

    1. Thickness 'tooth-pick' constraints: Thickness constraints are
       enforced at specific locations. A relative or absolute
       minimum/maximum thickness can be specified. Two variants are
       supplied: a '2d' variant for thickness constraints over an area
       such as a spar box :func:`addThicknessConstraints2D` and a '1d'
       variant for thickness constraints along a (poly) line
       :func:`addThicknessConstraints1D`.

    2. Volume constraint: This computes and enforces a volume constraint
       over the specified domain. The input is identical to the '2d'
       thickness constraints. See :func:`addVolumeConstraint`.

    3. LE/TE Constraints: These geometric constraints are required when
       using FFD volumes with shape variables. The leading and trailing
       edges must be fixed wrt the shape variables so these enforce that the
       coefficients on the leading edge can only move in equal and opposite
       directions

    4. Fixed Location Constraints: These constraints allow you to specify
       certain location in the FFD that can not move.

    5. Gear Post Constraint: This is a very highly specialized
       constraint used to ensure there is sufficient vertical space to
       install wing-mounted landing gear and that it is correctly positioned
       relative to the wing. See the class definition for more information.

    Analytic sensitivity information is computed for all functions and a
    facility for adding the constraints automatically to a pyOptSparse
    optimization problem is also provided.

    Parameters
    ----------
    name : str
        A name for this object. Used to distinguish between DVCon objects
        if multiple DVConstraint objects are used in an optimization.

    """

    def __init__(self, name="DVCon1"):
        """
        Create a (empty) DVconstrains object. Specific types of
        constraints will added individually
        """

        self.name = name

        self.constraints = OrderedDict()
        self.linearCon = OrderedDict()

        # Data for the discrete surface

        self.surfaces = {}
        self.DVGeometries = {}

    def setSurface(self, surf, name="default", addToDVGeo=False, DVGeoName="default", surfFormat="point-vector"):
        """
        Set the surface DVConstraints will use to perform projections.

        Parameters
        ----------
        surf : pyGeo object or list or str
            The triangulated surface representation to use for projections.
            There are a few possible ways of defining a surface.
            1) A pyGeo surface object. `surfFormat` must be "point-vector".
            2) List of [p0, v1, v2] with `surfFormat` "point-vector".
            3) List of [p0, p1, p2] with `surfFormat` "point-point".
            4) Path to a PLOT3D surface file. `surfFormat` must be "point-vector".

            Option 2 is the most common, where the list is computed by an AeroSolver like ADflow.

        addToDVGeo : bool
            Flag to embed the surface point set in a DVGeo object.
            If True, `DVGeoName` must be set appropriately.

        name : str
            Name associated with the surface. Must be unique. For backward compatibility,
            the name is 'default' by default

        DVGeoName : str
            Name of the DVGeo object to set the surface to. You only
            need to set this if you're using multiple DVGeo objects
            for a problem. For backward compatibility, the name is 'default' by default

        surfFormat : str
            The surface format. Either "point-vector" or "point-point".
            See `surf` for details.

        Examples
        --------
        >>> CFDsolver = ADFLOW(comm=comm, options=aeroOptions)
        >>> surf = CFDsolver.getTriangulatedMeshSurface()
        >>> DVCon.setSurface(surf)
        >>> # Or using a pyGeo surface object:
        >>> surf = pyGeo('iges',fileName='wing.igs')
        >>> DVCon.setSurface(surf)

        """
        if name in self.surfaces.keys():
            raise KeyError("Surface names must be unique. Repeated surface name: " + str(name))

        self.surfaces[name] = []
        if surfFormat == "point-vector":
            if isinstance(surf, list):
                # Data from ADflow
                p0 = np.array(surf[0])
                v1 = np.array(surf[1])
                v2 = np.array(surf[2])
            elif isinstance(surf, str):
                # Load the surf as a plot3d file
                p0, v1, v2 = readPlot3DSurfFile(surf)

            elif isinstance(surf, pyGeo):  # Assume it's a pyGeo surface
                p0, v1, v2 = self._generateDiscreteSurface(surf)
            else:
                raise TypeError("surf given is not a supported type [List, plot3D file name, or pyGeo surface]")

            p1 = p0 + v1
            p2 = p0 + v2
        elif surfFormat == "point-point":
            if isinstance(surf, str):
                # load from file
                raise NotImplementedError
            elif isinstance(surf, list):
                # for now, do NOT add the object geometry to dvgeo
                p0 = np.array(surf[0])
                p1 = np.array(surf[1])
                p2 = np.array(surf[2])
            elif isinstance(surf, np.ndarray):
                surf_length = surf[:, 0, :].shape[0]
                p0 = surf[:, 0, :].reshape(surf_length, 3)
                p1 = surf[:, 1, :].reshape(surf_length, 3)
                p2 = surf[:, 2, :].reshape(surf_length, 3)
            else:
                raise TypeError("surf given is not supported [list, np.ndarray]")

        self.surfaces[name].append(p0)
        self.surfaces[name].append(p1)
        self.surfaces[name].append(p2)

        if addToDVGeo:
            self._checkDVGeo(name=DVGeoName)
            self.DVGeometries[DVGeoName].addPointSet(self.surfaces[name][0], name + "_p0")
            self.DVGeometries[DVGeoName].addPointSet(self.surfaces[name][1], name + "_p1")
            self.DVGeometries[DVGeoName].addPointSet(self.surfaces[name][2], name + "_p2")

    def setDVGeo(self, DVGeo, name="default"):
        """
        Set the DVGeometry object that will manipulate this object.
        Note that DVConstraints doesn't **strictly** need a DVGeometry
        object set, but if optimization is desired it is required.

        Parameters
        ----------
        dvGeo : A DVGeometry object.
            Object responsible for manipulating the constraints that
            this object is responsible for.

        Examples
        --------
        >>> dvCon.setDVGeo(DVGeo)
        """

        # double check that there are no design variables with shared names
        # must be unique
        for existing_DVGeo_name in self.DVGeometries:
            existing_DVGeo = self.DVGeometries[existing_DVGeo_name]
            for dvName in DVGeo.getVarNames():
                for existing_dvName in existing_DVGeo.getVarNames():
                    if dvName == existing_dvName:
                        msg = (
                            f"Design variable {dvName} in the newly-added DVGeo already exists in DVGeo"
                            f"object named {existing_DVGeo_name} on this DVCon"
                        )
                        raise ValueError(msg)
        self.DVGeometries[name] = DVGeo

    def addConstraintsPyOpt(self, optProb, exclude_wrt=None):
        """
        Add all constraints to the optProb object. Only constraints
        the that have the addToPyOpt flags are actually added.

        Parameters
        ----------
        optProb : pyOpt_optimization object
            Optimization description to which the constraints are added

        Examples
        --------
        >>> DVCon.addConstraintsPyOpt(optProb)
        """

        # loop over the generated constraint objects and add the necessary
        # constraints to pyOpt
        for conTypeKey in self.constraints:
            constraint = self.constraints[conTypeKey]
            for key in constraint:
                constraint[key].addConstraintsPyOpt(optProb, exclude_wrt=exclude_wrt)

        # add the linear constraints separately, since they are treated a bit differently
        for key in self.linearCon:
            self.linearCon[key].addConstraintsPyOpt(optProb)

    def addVariablesPyOpt(self, optProb):
        """
        Add all constraint variables to the optProb object.

        Parameters
        ----------
        optProb : pyOpt_optimization object
            Optimization description to which the constraints are added

        Examples
        --------
        >>> DVCon.addVariablesPyOpt(optProb)
        """

        # loop over the generated constraint objects and add the necessary
        # variables to pyOpt
        for conTypeKey in self.constraints:
            constraint = self.constraints[conTypeKey]
            for key in constraint:
                constraint[key].addVariablesPyOpt(optProb)

        # linear constraints are ignored because at the moment there are no linear
        # constraints that have independent variables

    def setDesignVars(self, dvDict):
        """
        Standard routine for setting design variables from a design
        variable dictionary.

        Parameters
        ----------
        dvDict : dict
            Dictionary of design variables. The keys of the dictionary
            must correspond to the design variable names. Any
            additional keys in the dfvdictionary are simply ignored.
        """

        # loop over the generated constraint objects and add the necessary
        # variables to pyOpt
        for conTypeKey in self.constraints:
            constraint = self.constraints[conTypeKey]
            for key in constraint:
                constraint[key].setDesignVars(dvDict)

        # linear constraints are ignored because at the moment there are no linear
        # constraints that have independent variables

    def evalFunctions(self, funcs, includeLinear=False, config=None):
        """
        Evaluate all the 'functions' that this object has. Of course,
        these functions are usually just the desired constraint
        values. These values will be set directly into the funcs
        dictionary.

        Parameters
        ----------
        funcs : dict
            Dictionary into which the function values are placed.
        includeLeTe : bool
            Flag to include Leading/Trailing edge
            constraints. Normally this can be false since pyOptSparse
            does not need linear constraints to be returned.
        """

        # loop over the generated constraints and evaluate their function values
        for conTypeKey in self.constraints:
            constraint = self.constraints[conTypeKey]
            for key in constraint:
                constraint[key].evalFunctions(funcs, config)

        if includeLinear:
            for key in self.linearCon:
                self.linearCon[key].evalFunctions(funcs)

    def evalFunctionsSens(self, funcsSens, includeLinear=False, config=None):
        """
        Evaluate the derivative of all the 'functions' that this
        object has. These functions are just the constraint values.
        These values will be set directly in the funcSens dictionary.

        Parameters
        ----------
        funcSens : dict
            Dictionary into which the sensitivities are added.
        includeLeTe : bool
            Flag to include Leading/Trailing edge
            constraints. Normally this can be false since pyOptSparse
            does not need linear constraints to be returned.
        """

        # loop over the generated constraints and evaluate their function values
        for conTypeKey in self.constraints:
            constraint = self.constraints[conTypeKey]
            for key in constraint:
                constraint[key].evalFunctionsSens(funcsSens, config)

        if includeLinear:
            for key in self.linearCon:
                self.linearCon[key].evalFunctionsSens(funcsSens)

    def writeTecplot(self, fileName):
        """
        This function writes a visualization file for constraints. All
        currently added constraints are written to a tecplot. This is
        useful for publication purposes as well as determine if the
        constraints are *actually* what the user expects them to be.

        Parameters
        ----------
        fileName : str
            File name for tecplot file. Should have a .dat extension or a
            .dat extension will be added automatically.
        """

        f = open(fileName, "w")
        f.write('TITLE = "DVConstraints Data"\n')
        f.write('VARIABLES = "CoordinateX" "CoordinateY" "CoordinateZ"\n')

        # loop over the constraints and add their data to the tecplot file
        for conTypeKey in self.constraints:
            constraint = self.constraints[conTypeKey]
            for key in constraint:
                constraint[key].writeTecplot(f)

        for key in self.linearCon:
            self.linearCon[key].writeTecplot(f)
        f.close()

    def writeSurfaceTecplot(self, fileName, surfaceName="default"):
        """
        Write the triangulated surface mesh used in the constraint object
        to a tecplot file for visualization.

        Parameters
        ----------
        fileName : str
            File name for tecplot file. Should have a .dat extension.
        surfaceName : str
            Which DVConstraints surface to write to file (default is 'default')
        """
        p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)

        f = open(fileName, "w")
        f.write('TITLE = "DVConstraints Surface Mesh"\n')
        f.write('VARIABLES = "CoordinateX" "CoordinateY" "CoordinateZ"\n')
        f.write("Zone T=%s\n" % ("surf"))
        f.write("Nodes = %d, Elements = %d ZONETYPE=FETRIANGLE\n" % (len(p0) * 3, len(p0)))
        f.write("DATAPACKING=POINT\n")
        for i in range(len(p0)):
            points = [p0[i], p1[i], p2[i]]
            for j in range(len(points)):
                f.write(f"{points[j][0]:f} {points[j][1]:f} {points[j][2]:f}\n")

        for i in range(len(p0)):
            f.write("%d %d %d\n" % (3 * i + 1, 3 * i + 2, 3 * i + 3))

        f.close()

    def writeSurfaceSTL(self, fileName, surfaceName="default", fromDVGeo=None):
        """
        Write the triangulated surface mesh to a .STL file for manipulation and visualization.

        Parameters
        ----------
        fileName : str
            File name for stl file. Should have a .stl extension.
        surfaceName : str
            Which DVConstraints surface to write to file (default is 'default')
        fromDVGeo : str or None
            Name of the DVGeo object to obtain the surface from (default is 'None')
        """
        try:
            # External modules
            from stl import mesh
        except ImportError as e:
            raise ImportError("numpy-stl package must be installed") from e
        if fromDVGeo is None:
            p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)
        else:
            p0 = self.DVGeometries[fromDVGeo].update(surfaceName + "_p0")
            p1 = self.DVGeometries[fromDVGeo].update(surfaceName + "_p1")
            p2 = self.DVGeometries[fromDVGeo].update(surfaceName + "_p2")

        stlmesh = mesh.Mesh(np.zeros(p0.shape[0], dtype=mesh.Mesh.dtype))
        stlmesh.vectors[:, 0, :] = p0
        stlmesh.vectors[:, 1, :] = p1
        stlmesh.vectors[:, 2, :] = p2

        # Write the mesh to file "cube.stl"
        stlmesh.save(fileName)

    def addThicknessConstraints2D(
        self,
        leList,
        teList,
        nSpan,
        nChord,
        lower=1.0,
        upper=3.0,
        scaled=True,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        r"""
        Add a set of thickness constraints that span a logically a
        two-dimensional region. A little ASCII art can help here

        .. code-block:: text

          Planform view of the wing: The '+' are the (three dimensional)
          points that are supplied in leList and teList. The 'o' is described below.

          Physical extent of wing
                                   \
          __________________________\_________
          |                                  |
          +-----------+--o-----------+       |
          |   /                       \      |
          | leList      teList         \     |
          |                   \         \    |
          +------------------------------+   |
          |                                  |
          |__________________________________/


        Things to consider:

        * The region given by leList and teList must lie completely
          inside the wing

        * The number of points in leList and teList do not need to be
          the same length.

        * The leading and trailing edges are approximated using
          2-order splines (line segments) and nSpan points are
          interpolated in a linear fashion. For integer nSpan, the thickness
          constraint may not correspond *exactly* to intermediate
          locations in leList and teList. In the example above,
          with len(leList)=3 and nSpan=3, the three thickness
          constraints on the leading edge of the 2D domain would be at
          the left and right boundaries, and at the point denoted by
          'o' which is equidistant between the root and tip.
          To match intermediate locations exactly, pass a list for nSpan.

        * If a curved leading or trailing edge domain is desired,
          simply pass in lists for leList and teList with a sufficient
          number of points that can be approximated with straight line
          segments.

        * The two-dimensional data is projected onto the surface along
          the normals of the ruled surface formed by leList and teList

        * Normally, leList and teList are in given such that the the two curves
          entirely lie in a plane. This ensure that the projection vectors
          are always exactly normal to this plane.

        * If the surface formed by leList and teList is NOT precisely
          normal, issues can arise near the end of an open surface (ie
          root of a single wing) which can result in failing
          intersections.

        Parameters
        ----------
        leList : list or array
            A list or array of points (size should be (Nx3) where N is
            at least 2) defining the 'leading edge' or the start of the
            domain.

        teList : list or array
           Same as leList but for the trailing edge.

        nSpan : int or list of int
            The number of thickness constraints to be (linear)
            interpolated *along* the leading and trailing edges.
            A list of length N-1 can be used to specify the number
            for each segment defined by leList and teList and
            precisely match intermediate locations.

        nChord : int
            The number of thickness constraints to be (linearly)
            interpolated between the leading and trailing edges.

        lower : float or array of size (nSpan x nChord)
            The lower bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint.

        upper : float or array of size (nSpan x nChord)
            The upper bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint.

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial length of each thickness
              constraint is defined to be 1.0. In this case, the lower
              and upper bounds are given in multiple of the initial
              length. lower=0.85, upper=1.15, would allow for 15%
              change in each direction from the original length. For
              aerodynamic shape optimizations, this option is used
              most often.

            * scaled=False: No scaling is applied and the physical lengths
              must be specified for the lower and upper bounds.

        scale : float or array of size (nSpan x nChord)

            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If the thickness constraints are scaled, this
            already results in well-scaled constraint values, and
            scale can be left at 1.0. If scaled=False, it may changed
            to a more suitable value of the resulting physical
            thickness have magnitudes vastly different than O(1).

        name : str
            Normally this does not need to be set. Only use this if
            you have multiple DVCon objects and the constraint names
            need to be distinguished **or** the values are to be used
            in a subsequent computation.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) *before* they are
            given to the optimizer, set this flag to False.

        surfaceName : str
            Name of the surface to project to. This should be the same
            as the surfaceName provided when setSurface() was called.
            For backward compatibility, the name is 'default' by default.

        DVGeoName : str
            Name of the DVGeo object to compute the constraint with. You only
            need to set this if you're using multiple DVGeo objects
            for a problem. For backward compatibility, the name is 'default' by default

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        Examples
        --------
        >>> # Take unique square in x-z plane and and 10 along z-direction (spanWise)
        >>> # and the along x-direction (chordWise)
        >>> leList = [[0, 0, 0], [0, 0, 1]]
        >>> teList = [[1, 0, 0], [0, 0, 1]]
        >>> DVCon.addThicknessConstraints2D(leList, teList, 10, 3,
                                lower=1.0, scaled=True)
        """

        self._checkDVGeo(DVGeoName)

        coords = self._generateIntersections(leList, teList, nSpan, nChord, surfaceName)

        # Get the total number of spanwise sections
        nSpanTotal = np.sum(nSpan)

        # Create the thickness constraint object:
        coords = coords.reshape((nSpanTotal * nChord * 2, 3))

        typeName = "thickCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        upper = convertTo2D(upper, nSpanTotal, nChord).flatten()
        lower = convertTo2D(lower, nSpanTotal, nChord).flatten()
        scale = convertTo2D(scale, nSpanTotal, nChord).flatten()

        # Create a name
        if name is None:
            conName = "%s_thickness_constraints_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = ThicknessConstraint(
            conName, coords, lower, upper, scaled, scale, self.DVGeometries[DVGeoName], addToPyOpt, compNames
        )

    def addThicknessConstraints1D(
        self,
        ptList,
        nCon,
        axis,
        lower=1.0,
        upper=3.0,
        scaled=True,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        r"""
        Add a set of thickness constraints oriented along a poly-line.

        See below for a schematic

        .. code-block:: text

          Planform view of the wing: The '+' are the (three dimensional)
          points that are supplied in ptList:

          Physical extent of wing
                                   \
          __________________________\_________
          |                  +               |
          |                -/                |
          |                /                 |
          | +-------+-----+                  |
          |              4-points defining   |
          |              poly-line           |
          |                                  |
          |__________________________________/


        Parameters
        ----------
        ptList : list or array of size (N x 3) where N >=2
            The list of points forming a poly-line along which the
            thickness constraints will be added.

        nCon : int
            The number of thickness constraints to add

        axis : list or array of length 3
            The direction along which the projections will occur.
            Typically this will be y or z axis ([0,1,0] or [0,0,1])

        lower : float or array of size nCon
            The lower bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint.

        upper : float or array of size nCon
            The upper bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint.

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial length of each thickness
              constraint is defined to be 1.0. In this case, the lower
              and upper bounds are given in multiple of the initial
              length. lower=0.85, upper=1.15, would allow for 15%
              change in each direction from the original length. For
              aerodynamic shape optimizations, this option is used
              most often.

            * scaled=False: No scaling is applied and the physical lengths
              must be specified for the lower and upper bounds.

        scale : float or array of size nCon
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If the thickness constraints are scaled, this
            already results in well-scaled constraint values, and
            scale can be left at 1.0. If scaled=False, it may changed
            to a more suitable value of the resulting physical
            thickness have magnitudes vastly different than O(1).

        name : str
            Normally this does not need to be set. Only use this if
            you have multiple DVCon objects and the constraint names
            need to be distinguished **or** you are using this set of
            thickness constraints for something other than a direct
            constraint in pyOptSparse.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) *before* they are
            given to the optimizer, set this flag to False.

        surfaceName : str
            Name of the surface to project to. This should be the same
            as the surfaceName provided when setSurface() was called.
            For backward compatibility, the name is 'default' by default.

        DVGeoName : str
            Name of the DVGeo object to compute the constraint with. You only
            need to set this if you're using multiple DVGeo objects
            for a problem. For backward compatibility, the name is 'default' by default

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """
        self._checkDVGeo(DVGeoName)

        p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)

        # Create mesh of intersections
        constr_line = Curve(X=ptList, k=2)
        s = np.linspace(0, 1, nCon)
        X = constr_line(s)
        coords = np.zeros((nCon, 2, 3))
        # Project all the points
        for i in range(nCon):
            # Project actual node:
            up, down, fail = geo_utils.projectNode(X[i], axis, p0, p1 - p0, p2 - p0)
            if fail > 0:
                raise Error(
                    "There was an error projecting a node "
                    "at (%f, %f, %f) with normal (%f, %f, %f)." % (X[i, 0], X[i, 1], X[i, 2], axis[0], axis[1], axis[2])
                )
            coords[i, 0] = up
            coords[i, 1] = down

        # Create the thickness constraint object:
        coords = coords.reshape((nCon * 2, 3))

        typeName = "thickCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_thickness_constraints_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = ThicknessConstraint(
            conName, coords, lower, upper, scaled, scale, self.DVGeometries[DVGeoName], addToPyOpt, compNames
        )

    def addLERadiusConstraints(
        self,
        leList,
        nSpan,
        axis,
        chordDir,
        lower=1.0,
        upper=3.0,
        scaled=True,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        r"""
        Add a set of leading edge radius constraints. The constraint is set up
        similar to the 1D thickness or thickness-to-chord constraints. The user
        provides a polyline near the leading edge and specifies how many
        locations should be sampled along the polyline. The sampled points are
        then projected to the upper and lower surface along the provided axis.
        A third projection is made to the leading edge along chordDir. We then
        compute the radius of the circle circumscribed by these three points.

        In order for this radius calculation to be a reasonable approximation
        for the actual leading edge radius, it is critical that the polyline be
        drawn very close to the leading edge (less than 0.5% chord). We include
        a check to make sure that the points fall on the same hemisphere of the
        circumscribed circle, however we recommend that the user export the
        Tecplot view of the constraint using the writeTecplot function to verify
        that the circles do coincide with the leading edge radius.

        See below for a schematic

        .. code-block:: text

          Planform view of the wing: The '+' are the (three dimensional)
          points that are supplied in leList:

          Physical extent of wing
                                   \
          __________________________\_________
          | +-------+-----+-------+-----+    |
          |              5 points defining   |
          |              poly-line           |
          |                                  |
          |                                  |
          |                                  |
          |__________________________________/


        Parameters
        ----------
        leList : list or array of size (N x 3) where N >=2
            The list of points forming a poly-line along which the
            thickness constraints will be added.

        nSpan : int
            The number of thickness constraints to add

        axis : list or array of length 3
            The direction along which the up-down projections will occur.
            Typically this will be y or z axis ([0,1,0] or [0,0,1])

        chordDir : list or array or length 3
            The vector pointing from the leList to the leading edge. This will
            typically be the negative xaxis ([-1,0,0]). The magnitude of the
            vector doesn't matter, but the direction does.

        lower : float or array of size nSpan
            The lower bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint.

        upper : float or array of size nSpan
            The upper bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint.

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial radius of each constraint is defined to
              be 1.0. In this case, the lower and upper bounds are given in
              multiples of the initial radius. lower=0.85, upper=1.15, would
              allow for 15% change in each direction from the original radius.

            * scaled=False: No scaling is applied and the physical radii
              must be specified for the lower and upper bounds.

        scale : float or array of size nSpan
            This is the optimization scaling of the constraint. Typically this
            parameter will not need to be changed. If the radius constraints are
            scaled, this already results in well-scaled constraint values, and
            scale can be left at 1.0. If scaled=False, it may  be changed to a
            more suitable value if the resulting physical thickness have
            magnitudes vastly different than O(1).

        name : str
            Normally this does not need to be set. Only use this if you have
            multiple DVCon objects and the constraint names need to be
            distinguished **or** you are using this set of thickness constraints
            for something other than a direct constraint in pyOptSparse.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) *before* they are
            given to the optimizer, set this flag to False.

        surfaceName : str
            Name of the surface to project to. This should be the same
            as the surfaceName provided when setSurface() was called.
            For backward compatibility, the name is 'default' by default.

        DVGeoName : str
            Name of the DVGeo object to compute the constraint with. You only
            need to set this if you're using multiple DVGeo objects
            for a problem. For backward compatibility, the name is 'default' by default

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """
        self._checkDVGeo(DVGeoName)

        # Create mesh of intersections
        constr_line = Curve(X=leList, k=2)
        s = np.linspace(0, 1, nSpan)
        X = constr_line(s)
        coords = np.zeros((nSpan, 3, 3))
        p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)
        # Project all the points
        for i in range(nSpan):
            # Project actual node:
            up, down, fail = geo_utils.projectNode(X[i], axis, p0, p1 - p0, p2 - p0)
            if fail > 0:
                raise Error(
                    "There was an error projecting a node "
                    "at (%f, %f, %f) with normal (%f, %f, %f)." % (X[i, 0], X[i, 1], X[i, 2], axis[0], axis[1], axis[2])
                )
            coords[i, 0] = up
            coords[i, 1] = down

        # Calculate mid-points
        midPts = (coords[:, 0, :] + coords[:, 1, :]) / 2.0

        # Project to get leading edge point
        lePts = np.zeros((nSpan, 3))
        chordDir = np.array(chordDir, dtype="d").flatten()
        chordDir /= np.linalg.norm(chordDir)
        for i in range(nSpan):
            # Project actual node:
            up, down, fail = geo_utils.projectNode(X[i], chordDir, p0, p1 - p0, p2 - p0)
            if fail > 0:
                raise Error(
                    "There was an error projecting a node "
                    "at (%f, %f, %f) with normal (%f, %f, %f)."
                    % (X[i, 0], X[i, 1], X[i, 2], chordDir[0], chordDir[1], chordDir[2])
                )
            lePts[i] = up

        # Check that points can form radius
        d = np.linalg.norm(coords[:, 0, :] - coords[:, 1, :], axis=1)
        r = np.linalg.norm(midPts - lePts, axis=1)
        for i in range(nSpan):
            if d[i] < 2 * r[i]:
                raise Error(
                    "Leading edge radius points are too far from the "
                    "leading edge point to form a circle between the "
                    "three points."
                )

        # Add leading edge points and stack points into shape accepted by DVGeo
        coords[:, 2, :] = lePts
        coords = np.vstack((coords[:, 0, :], coords[:, 1, :], coords[:, 2, :]))

        # Create the thickness constraint object
        typeName = "radiusCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_leradius_constraints_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = RadiusConstraint(
            conName, coords, lower, upper, scaled, scale, self.DVGeometries[DVGeoName], addToPyOpt, compNames
        )

    def addLocationConstraints1D(
        self,
        ptList,
        nCon,
        lower=None,
        upper=None,
        scaled=False,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        DVGeoName="default",
        compNames=None,
    ):
        """
        Add a polyline in space that cannot move.

        Parameters
        ----------
        ptList : list or array of size (N x 3) where N >=2
            The list of points forming a poly-line along which the
            points will be fixed.

        nCon : int
            The number of points constraints to add

        lower : float or array of size nCon
            The lower bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint. If
            no value is provided, the bounds will default to the points,
            giving equality constraints. Using the default is recommended.

        upper : float or array of size nCon
            The upper bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint.  If
            no value is provided, the bounds will default to the points,
            giving equality constraints. Using the default is recommended.

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial location of each location
              constraint is defined to be 1.0. In this case, the lower
              and upper bounds are given in multiple of the initial
              location. lower=0.85, upper=1.15, would allow for 15%
              change in each direction from the original location. However,
              for initial points close to zero this blows up, so this should
              be used with caution, therefore unscaled is the default.

            * scaled=False: No scaling is applied and the physical locations
              must be specified for the lower and upper bounds.

        scale : float or array of size nCon
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If the location constraints are scaled, this
            already results in well-scaled constraint values, and
            scale can be left at 1.0. If scaled=False, it may changed
            to a more suitable value if the resulting physical
            location have magnitudes vastly different than O(1).

        name : str
            Normally this does not need to be set. Only use this if
            you have multiple DVCon objects and the constraint names
            need to be distinguished **or** you are using this set of
            location constraints for something other than a direct
            constraint in pyOptSparse.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) *before* they are
            given to the optimizer, set this flag to False.

        DVGeoName : str
            Name of the DVGeo object to compute the constraint with. You only
            need to set this if you're using multiple DVGeo objects
            for a problem. For backward compatibility, the name is 'default' by default

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """
        self._checkDVGeo(DVGeoName)
        # Create the points to constrain
        constr_line = Curve(X=ptList, k=2)
        s = np.linspace(0, 1, nCon)
        X = constr_line(s)
        # X should now be in the shape we need

        if lower is None:
            lower = X.flatten()
        if upper is None:
            upper = X.flatten()

        # Create the location constraint object
        typeName = "locCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_location_constraints_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = LocationConstraint(
            conName, X, lower, upper, scaled, scale, self.DVGeometries[DVGeoName], addToPyOpt, compNames
        )

    def addProjectedLocationConstraints1D(
        self,
        ptList,
        nCon,
        axis,
        bias=0.5,
        lower=None,
        upper=None,
        scaled=False,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        """This is similar to addLocationConstraints1D except that the actual
        poly line is determined by first projecting points on to the
        surface in a similar manner as addConstraints1D, and then
        taking the mid-point (or user specificed fraction) blend of
        the upper and lower surface locations.

        Parameters
        ----------
        ptList : list or array of size (N x 3) where N >=2
            The list of points from which to perform the projection

        nCon : int
            The number of points constraints to add

        axis : list or array of length 3
            The direction along which the projections will occur.
            Typically this will be y or z axis ([0,1,0] or [0,0,1])

        bias : float
            The blending of the upper/lower surface points to use. Default
            is 0.5 which is the average. 0.0 cooresponds to taking the
            lower point, 1.0 the upper point.

        lower : float or array of size nCon
            The lower bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint. If
            no value is provided, the bounds will default to the points,
            giving equality constraints. Using the default is recommended.

        upper : float or array of size nCon
            The upper bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint.  If
            no value is provided, the bounds will default to the points,
            giving equality constraints. Using the default is recommended.

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial location of each location
              constraint is defined to be 1.0. In this case, the lower
              and upper bounds are given in multiple of the initial
              location. lower=0.85, upper=1.15, would allow for 15%
              change in each direction from the original location. However,
              for initial points close to zero this blows up, so this should
              be used with caution, therefore unscaled is the default.

            * scaled=False: No scaling is applied and the physical locations
              must be specified for the lower and upper bounds.

        scale : float or array of size nCon
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If the location constraints are scaled, this
            already results in well-scaled constraint values, and
            scale can be left at 1.0. If scaled=False, it may changed
            to a more suitable value if the resulting physical
            location have magnitudes vastly different than O(1).

        name : str
            Normally this does not need to be set. Only use this if
            you have multiple DVCon objects and the constraint names
            need to be distinguished **or** you are using this set of
            location constraints for something other than a direct
            constraint in pyOptSparse.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) *before* they are
            given to the optimizer, set this flag to False.

        DVGeoName : str
            Name of the DVGeo object to compute the constraint with. You only
            need to set this if you're using multiple DVGeo objects
            for a problem. For backward compatibility, the name is 'default' by default

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """
        self._checkDVGeo(DVGeoName)
        # Create the points to constrain

        p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)

        constr_line = Curve(X=ptList, k=2)
        s = np.linspace(0, 1, nCon)
        X = constr_line(s)

        coords = np.zeros((nCon, 2, 3))
        # Project all the points
        for i in range(nCon):
            # Project actual node:
            up, down, fail = geo_utils.projectNode(X[i], axis, p0, p1 - p0, p2 - p0)
            if fail > 0:
                raise Error(
                    "There was an error projecting a node "
                    "at (%f, %f, %f) with normal (%f, %f, %f)." % (X[i, 0], X[i, 1], X[i, 2], axis[0], axis[1], axis[2])
                )
            coords[i, 0] = up
            coords[i, 1] = down

        X = (1 - bias) * coords[:, 1] + bias * coords[:, 0]

        # X is now what we want to constrain
        if lower is None:
            lower = X.flatten()
        if upper is None:
            upper = X.flatten()

        # Create the location constraint object
        typeName = "locCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_location_constraints_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = LocationConstraint(
            conName, X, lower, upper, scaled, scale, self.DVGeometries[DVGeoName], addToPyOpt, compNames
        )

    def addThicknessToChordConstraints1D(
        self,
        ptList,
        nCon,
        axis,
        chordDir,
        lower=1.0,
        upper=3.0,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        r"""
        Add a set of thickness-to-chord ratio constraints oriented along a poly-line.

        See below for a schematic

        .. code-block:: text

          Planform view of the wing: The '+' are the (three dimensional)
          points that are supplied in ptList:

          Physical extent of wing
                                   \
          __________________________\_________
          |                  +               |
          |                -/                |
          |                /                 |
          | +-------+-----+                  |
          |              4-points defining   |
          |              poly-line           |
          |                                  |
          |__________________________________/


        Parameters
        ----------
        ptList : list or array of size (N x 3) where N >=2
            The list of points forming a poly-line along which the
            thickness constraints will be added.

        nCon : int
            The number of thickness to chord ratio constraints to add

        axis : list or array of length 3
            The direction along which the projections will occur.
            Typically this will be y or z axis ([0,1,0] or [0,0,1])

        chordDir : list or array or length 3
            The direction defining "chord". This will typically be the
            xasis ([1,0,0]). The magnitude of the vector doesn't
            matter.

        lower : float or array of size nCon

            The lower bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint. This
            constraint can only be used in "scaled" mode. That means,
            the actual t/c is *never* computed. This constraint can
            only be used to constrain the relative change in t/c. A
            lower bound of 1.0, therefore mean the t/c cannot
            decrease. This is the typical use of this constraint.

        upper : float or array of size nCon
            The upper bound for the constraint. A single float will
            apply the same bounds to all constraints, while the array
            option will use different bounds for each constraint.

        scale : float or array of size nCon
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If the thickness constraints are scaled, this
            already results in well-scaled constraint values, and
            scale can be left at 1.0. If scaled=False, it may changed
            to a more suitable value of the resulting physical
            thickness have magnitudes vastly different than O(1).

        name : str
            Normally this does not need to be set. Only use this if
            you have multiple DVCon objects and the constraint names
            need to be distinguished **or** you are using this set of
            thickness constraints for something other than a direct
            constraint in pyOptSparse.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) *before* they are
            given to the optimizer, set this flag to False.

        DVGeoName : str
            Name of the DVGeo object to compute the constraint with. You only
            need to set this if you're using multiple DVGeo objects
            for a problem. For backward compatibility, the name is 'default' by default

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """
        self._checkDVGeo(DVGeoName)

        p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)

        constr_line = Curve(X=ptList, k=2)
        s = np.linspace(0, 1, nCon)
        X = constr_line(s)
        coords = np.zeros((nCon, 4, 3))
        chordDir /= np.linalg.norm(np.array(chordDir, "d"))
        # Project all the points
        for i in range(nCon):
            # Project actual node:
            up, down, fail = geo_utils.projectNode(X[i], axis, p0, p1 - p0, p2 - p0)
            if fail:
                raise Error(
                    "There was an error projecting a node "
                    "at (%f, %f, %f) with normal (%f, %f, %f)." % (X[i, 0], X[i, 1], X[i, 2], axis[0], axis[1], axis[2])
                )

            coords[i, 0] = up
            coords[i, 1] = down
            height = np.linalg.norm(coords[i, 0] - coords[i, 1])
            # Third point is the mid-point of those
            coords[i, 2] = 0.5 * (up + down)

            # Fourth point is along the chordDir
            coords[i, 3] = coords[i, 2] + 0.1 * height * chordDir

        # Create the thickness constraint object:
        coords = coords.reshape((nCon * 4, 3))

        typeName = "thickCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()
        if name is None:
            conName = "%s_thickness_to_chord_constraints_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = ThicknessToChordConstraint(
            conName, coords, lower, upper, scale, self.DVGeometries[DVGeoName], addToPyOpt, compNames
        )

    def addTriangulatedSurfaceConstraint(
        self,
        surface_1_name=None,
        DVGeo_1_name="default",
        surface_2_name="default",
        DVGeo_2_name="default",
        rho=50.0,
        heuristic_dist=None,
        perim_scale=0.1,
        max_perim=3.0,
        name=None,
        scale=1.0,
        addToPyOpt=True,
    ):
        """
        Add a single triangulated surface constraint to an aerosurface.
        This constraint is designed to keep a general 'blob' of watertight
        geometry contained within an aerodynamic hull (e.g., a wing)

        Parameters
        ----------
        surface_1_name : str
            The name of the first triangulated surface to constrain.
            This should be the surface with the larger number of triangles.
            By default, it's the ADflow triangulated surface mesh.

        DVGeo_1_name : str
            The name of the DVGeo object to associate surface_1 to.
            If None, surface_1 will remain static during optimization.
            By default, it's the 'default' DVGeo object

        surface_2_name : str
            The name of the second triangulated surface to constrain.
            This should be the surface with the smaller number of triangles.

        DVGeo_2_name : str
            The name of the DVGeo object to associate surface_2 to.
            If None, surface_2 will remain static during optimization.
            By default, it's the 'default' DVGeo object

        rho : float
            The rho factor of the KS function of min distance.

        heuristic_dist : float
            The triangulated surface constraint uses a procedure to skip
            pairs of facets that are farther apart than a heuristic distance
            in order to save computation time. By default, this is set
            to the maximum linear dimension of the second object's bounding box.
            You can set this to a large number to compute an "exact" KS.

        perim_scale : float
            Apply a scaling factor to the intersection perimeter length.

        max_perim : float
            Maximum allowable intersection length before fail flag is returned

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constraint names need to
             be distinguished **OR** you are using this
             computation for something other than a direct constraint
             in pyOpt, i.e. it is required for a subsequent
             computation.

        scale : float
            This is the optimization scaling of the
            constraint. It may changed to a more suitable value of
            the resulting physical volume magnitude is vastly different
            from O(1).

        addToPyOpt : bool
            Normally this should be left at the default of True if this
            is to be used as a constraint. If this to
            used in a subsequent calculation and not a constraint
            directly, addToPyOpt should be False, and name
            specified to a logical name for this computation. with
            addToPyOpt=False, the lower, upper and scale variables are
            meaningless
        """
        if DVGeo_1_name is not None:
            self._checkDVGeo(DVGeo_1_name)
            DVGeo1 = self.DVGeometries[DVGeo_1_name]
        else:
            DVGeo1 = None
        if DVGeo_2_name is not None:
            self._checkDVGeo(DVGeo_2_name)
            DVGeo2 = self.DVGeometries[DVGeo_2_name]
        else:
            DVGeo2 = None
        if DVGeo1 is None and DVGeo2 is None:
            raise ValueError("At least one DVGeo object must be specified")

        typeName = "triSurfCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_trisurf_constraint_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name

        surface_1 = self._getSurfaceVertices(surface_1_name)
        surface_2 = self._getSurfaceVertices(surface_2_name)

        # Finally add constraint object
        self.constraints[typeName][conName] = TriangulatedSurfaceConstraint(
            conName,
            surface_1,
            surface_1_name,
            DVGeo1,
            surface_2,
            surface_2_name,
            DVGeo2,
            scale,
            addToPyOpt,
            rho,
            perim_scale,
            max_perim,
            heuristic_dist,
        )

    def addTriangulatedVolumeConstraint(
        self,
        lower=1.0,
        upper=99999.0,
        scaled=True,
        scale=1.0,
        name=None,
        surfaceName="default",
        DVGeoName="default",
        addToPyOpt=True,
    ):
        """
        Add a single triangulated volume constraint to a surface.
        Computes and constrains the volume of a closed, triangulated surface.
        The surface normals must ALL be outward-oriented!
        This is typical for most stl files but
        should be verified in a program like Meshmixer.

        Parameters
        ----------
        lower : float
            The lower bound for the volume constraint.

        upper : float
            The upper bound for the volume constraint.

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial volume is defined to be 1.0.
              In this case, the lower and upper bounds are given in
              multiple of the initial volume. lower=0.85, upper=1.15,
              would allow for 15% change in volume both upper and
              lower. For aerodynamic optimization, this is the most
              widely used option .

            * scaled=False: No scaling is applied and the physical
              volume. lower and upper refer to the physical volumes.

        scale : float
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If scaled=True, this automatically results in a
            well-scaled constraint and scale can be left at 1.0. If
            scaled=False, it may changed to a more suitable value of
            the resulting physical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constraint names need to
             be distinguished **OR** you are using this volume
             computation for something other than a direct constraint
             in pyOpt, i.e. it is required for a subsequent
             computation.

        surfaceName : str
            Name the triangulated surface attached to DVConstraints which should
            be used for the constraint. 'default' uses the main aerodynamic
            surface mesh

        DVGeoName : str
            Name the DVGeo object affecting the geometry of the
            surface. 'default' uses the main DVGeo object of the DVConstraints instance

        addToPyOpt : bool
            Normally this should be left at the default of True if the
            volume is to be used as a constraint. If the volume is to
            used in a subsequent calculation and not a constraint
            directly, addToPyOpt should be False, and name
            specified to a logical name for this computation. with
            addToPyOpt=False, the lower, upper and scale variables are
            meaningless
        """

        self._checkDVGeo(DVGeoName)
        DVGeo = self.DVGeometries[DVGeoName]

        typeName = "triVolCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_trivolume_constraint_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name

        surface = self._getSurfaceVertices(surfaceName)

        # Finally add constraint object
        self.constraints[typeName][conName] = TriangulatedVolumeConstraint(
            conName, surface, surfaceName, lower, upper, scaled, scale, DVGeo, addToPyOpt
        )

    def addVolumeConstraint(
        self,
        leList,
        teList,
        nSpan,
        nChord,
        lower=1.0,
        upper=3.0,
        scaled=True,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        r"""
        Add a single volume constraint to the wing. The volume
        constraint is defined over a logically two-dimensional region
        as shown below

        .. code-block:: text

          Planform view of the wing: The '+' are the (three dimensional)
          points that are supplied in leList and teList.

          Physical extent of wing
                                   \
          __________________________\_________
          |                                  |
          +--------------------------+       |
          |   /      (Volume in here) \      |
          | leList      teList         \     |
          |                   \         \    |
          +------------------------------+   |
          |                                  |
          |__________________________________/

        The region defined by the '----' boundary in the figure above
        will be meshed with nSpan x nChord points to form a 2D domain
        and then projected up and down onto the surface to form 3D
        hexahedral volumes. The accuracy of the volume computation
        depends on how well these linear hexahedral volumes
        approximate the (assumed) continuous underlying surface.

        See `addThicknessConstraints2D` for additional information.

        Parameters
        ----------
        leList : list or array
           A list or array of points (size should be (Nx3) where N is
           at least 2) defining the 'leading edge' or the start of the
           domain.

        teList : list or array
           Same as leList but for the trailing edge.

        nSpan : int or list of int
            The number of projected points to be (linear)
            interpolated *along* the leading and trailing edges.
            A list of length N-1 can be used to specify the number
            for each segment defined by leList and teList and
            precisely match intermediate locations.

        nChord : int
            The number of projected points to be (linearly)
            interpolated between the leading and trailing edges.

        lower : float
            The lower bound for the volume constraint.

        upper : float
            The upper bound for the volume constraint.

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial volume is defined to be 1.0.
              In this case, the lower and upper bounds are given in
              multiple of the initial volume. lower=0.85, upper=1.15,
              would allow for 15% change in volume both upper and
              lower. For aerodynamic optimization, this is the most
              widely used option .

            * scaled=False: No scaling is applied and the physical
              volume. lower and upper refer to the physical volumes.

        scale : float
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If scaled=True, this automatically results in a
            well-scaled constraint and scale can be left at 1.0. If
            scaled=False, it may changed to a more suitable value of
            the resulting physical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constraint names need to
             be distinguished **OR** you are using this volume
             computation for something other than a direct constraint
             in pyOpt, i.e. it is required for a subsequent
             computation.

        addToPyOpt : bool
            Normally this should be left at the default of True if the
            volume is to be used as a constraint. If the volume is to
            used in a subsequent calculation and not a constraint
            directly, addToPyOpt should be False, and name
            specified to a logical name for this computation. with
            addToPyOpt=False, the lower, upper and scale variables are
            meaningless

        surfaceName : str
            Name of the surface to project to. This should be the same
            as the surfaceName provided when setSurface() was called.
            For backward compatibility, the name is 'default' by default.

        DVGeoName : str
            Name of the DVGeo object to compute the constraint with. You only
            need to set this if you're using multiple DVGeo objects
            for a problem. For backward compatibility, the name is 'default' by default

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """
        self._checkDVGeo(DVGeoName)

        typeName = "volCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_volume_constraint_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name

        coords = self._generateIntersections(leList, teList, nSpan, nChord, surfaceName)

        # Get the total number of spanwise sections
        nSpanTotal = np.sum(nSpan)

        coords = coords.reshape((nSpanTotal * nChord * 2, 3))

        # Finally add the volume constraint object
        self.constraints[typeName][conName] = VolumeConstraint(
            conName,
            nSpanTotal,
            nChord,
            coords,
            lower,
            upper,
            scaled,
            scale,
            self.DVGeometries[DVGeoName],
            addToPyOpt,
            compNames,
        )

    def addCompositeVolumeConstraint(
        self, vols, lower=1.0, upper=3.0, scaled=True, scale=1.0, name=None, addToPyOpt=True, DVGeoName="default"
    ):
        """
        Add a composite volume constraint. This used previously added
        constraints and combines them to form a single volume constraint.

        The general ussage is as follows::

          DVCon.addVolumeConstraint(leList1, teList1, nSpan, nChord,
                                    name='part1', addToPyOpt=False)
          DVCon.addVolumeConstraint(leList2, teList2, nSpan, nChord,
                                    name='part2', addToPyOpt=False)
          DVCon.addCompositeVolumeConstraint(['part1', 'part2'], lower=1)


        Parameters
        ----------
        vols : list of strings
           A list containing the names of the previously added
           volumes to be used.

        lower : float
            The lower bound for the volume constraint.

        upper : float
            The upper bound for the volume constraint.

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial volume is defined to be 1.0.
              In this case, the lower and upper bounds are given in
              multiple of the initial volume. lower=0.85, upper=1.15,
              would allow for 15% change in volume both upper and
              lower. For aerodynamic optimization, this is the most
              widely used option .

            * scaled=False: No scaling is applied and the physical
              volume. lower and upper refer to the physical volumes.

        scale : float
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If scaled=True, this automatically results in a
            well-scaled constraint and scale can be left at 1.0. If
            scaled=False, it may changed to a more suitable value of
            the resulting physical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constraint names need to
             be distinguished **OR** you are using this volume
             computation for something other than a direct constraint
             in pyOpt, i.e. it is required for a subsequent
             computation.

        addToPyOpt : bool
            Normally this should be left at the default of True if the
            volume is to be used as a constraint. If the volume is to
            used in a subsequent calculation and not a constraint
            directly, addToPyOpt should be False, and name
            specified to a logical name for this computation. with
            addToPyOpt=False, the lower, upper and scale variables are
            meaningless
        """
        self._checkDVGeo(DVGeoName)

        typeName = "volCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_composite_volume_constraint_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name

        # Determine the list of volume constraint objects
        volCons = []
        for vol in vols:
            try:
                volCons.append(self.constraints[typeName][vol])
            except KeyError as e:
                raise Error(
                    f"The supplied volume '{vol}' has not already been added with a call to addVolumeConstraint()"
                ) from e
        self.constraints[typeName][conName] = CompositeVolumeConstraint(
            conName, volCons, lower, upper, scaled, scale, self.DVGeometries[DVGeoName], addToPyOpt
        )

    def addLeTeConstraints(
        self,
        volID=None,
        faceID=None,
        topID=None,
        indSetA=None,
        indSetB=None,
        name=None,
        config=None,
        childIdx=None,
        comp=None,
        DVGeoName="default",
    ):
        """
        Add a set of 'leading edge' or 'trailing edge' constraints to
        DVConstraints. These are just a particular form of linear
        constraints for shape variables. The need for these
        constraints arise when the shape variables can effectively
        emulate a 'twist' variable (actually a shearing twist). The
        purpose of these constraints is to make control points at the
        leading and trailing edge move in equal and opposite direction.

        .. math:: x_1 - x_2 = 0.0

        where :math:`x_1` is the movement (in 1, 2, or 3 directions) of a
        control point on the top of the FFD and :math:`x_2` is the control
        point on the bottom of the FFD.

        There are two ways of specifying these constraints:

        ``volID`` and ``faceID``
            Provide the index of the FFD block and the
            ``faceID`` (one of ``ilow``, ``ihigh``, ``jlow``, ``jhigh``, ``klow``, or
            ``khigh``). This it the preferred approach. Both ``volID`` and ``faceID``
            can be determined by examining the FFD file in TecPlot or ICEM.
            Use 'prob data' tool in TecPlot to click on the surface of which
            you want to put constraints on (e.g. the front or LE of FFD and
            the back surface or TE of the FFD). You will see which plane
            it coresponding to. For example, 'I-Plane' with I-index = 1 is
            ``iLow``.
            ``topID`` provides a second input for blocks that have 2x2 faces.
        ``indSetA`` and ``indSetB``
            Alternatively, two sets of indices can be provided.
            Both must be the same length. These indices may
            be obtained from the ``lindex`` array of the FFD object.

                >>> lIndex = DVGeo.getLocalIndex(iVol)

            ``lIndex`` is a three dimensional set of indices that provide the
            index into the global set of control points. See below for
            examples.

        .. note::
            These constraints *will* be added to pyOptSparse automatically with a call to
            :func:`addConstraintsPyOpt`

        Parameters
        ----------
        volID : int
            Single integer specifying the volume index
        faceID : str {'iLow', 'iHigh', 'jLow', 'jHigh', 'kLow', 'kHigh'}
            One of above specifying the node on which face to constrain.
        topID : str {'i','j', 'k'}
            One of the above specifing the symmetry direction, should
            only be used on 2x2 faces
        indSetA : array of int
            Indices of control points on one side of the FFD
        indSetB : array of int
            Indices of control points on the *other* side of the FFD
        name : str
            Normally this does not need to be set; a default name will
            be generated automatically. Only use this if you have
            multiple DVCon objects and the constraint names need to
            be distinguished
        config : str
            The DVGeo configuration to apply this constraint to. Must be either None
            which will apply to *ALL* the local DV groups or a single string specifying
            a particular configuration.
        childIdx : int
            The zero-based index of the child FFD, if this constraint is being applied to a child FFD.
            The index is defined by the order in which you add the child FFD to the parent.
            For example, the first child FFD has an index of 0, the second an index of 1, and so on.
        comp: str
            The component name if using DVGeometryMulti.


        Examples
        --------
        >>> # Preferred way: Constraints at the front and back (ifaces) of volume 0
        >>> DVCon.addLeTeConstraints(0, 'iLow')
        >>> DVCon.addLeTeConstraints(0, 'iHigh')
        >>> # Alternative way -- this can be specialzed as required:
        >>> lIndex = DVGeo.getLocalIndex(1) # Volume 2
        >>> indSetA = []; indSetB = [];
        >>> for k in range(4,7): # 4 on the leading edge
        >>>     indSetA.append(lIndex[0, 0, k])
        >>>     indSetB.append(lIndex[0, 1, k])
        >>> lIndex = DVGeo.getLocalIndex(4) # Volume 5 (different from LE)
        >>> for k in range(4,8): # 5 on the trailing edge
        >>>     indSetA.append(lIndex[-1, 0, k])
        >>>     indSetB.append(lIndex[-1, 1, k])
        >>> # Now add to DVCon
        >>> DVCon.addLeTeConstraints(0, indSetA, indSetB)
        """
        self._checkDVGeo(DVGeoName)

        if comp is None:
            DVGeo = self.DVGeometries[DVGeoName]
        else:
            DVGeo = self.DVGeometries[DVGeoName].DVGeoDict[comp]

        if childIdx is not None:
            DVGeo = DVGeo.children[childIdx]

        # Now determine what type of specification we have:
        if volID is not None and faceID is not None:
            lIndex = DVGeo.getLocalIndex(volID)
            iFace = False
            jFace = False
            kFace = False
            if faceID.lower() == "ilow":
                indices = lIndex[0, :, :]
                iFace = True
            elif faceID.lower() == "ihigh":
                indices = lIndex[-1, :, :]
                iFace = True
            elif faceID.lower() == "jlow":
                indices = lIndex[:, 0, :]
                jFace = True
            elif faceID.lower() == "jhigh":
                indices = lIndex[:, -1, :]
                jFace = True
            elif faceID.lower() == "klow":
                indices = lIndex[:, :, 0]
                kFace = True
            elif faceID.lower() == "khigh":
                indices = lIndex[:, :, -1]
                kFace = True
            else:
                raise Error("faceID must be one of iLow, iHigh, jLow, jHigh, " "kLow or kHigh.")

            # Now we see if one and exactly one length in 2:
            shp = indices.shape
            if shp[0] == 2 and shp[1] > 2:
                indSetA = indices[0, :]
                indSetB = indices[1, :]
            elif shp[1] == 2 and shp[0] > 2:
                indSetA = indices[:, 0]
                indSetB = indices[:, 1]
            else:
                if topID is not None:
                    if topID.lower() == "i" and not iFace:
                        indSetA = indices[0, :]
                        indSetB = indices[1, :]
                    elif topID.lower() == "j" and not jFace:
                        if iFace:
                            indSetA = indices[0, :]
                            indSetB = indices[1, :]
                        else:
                            indSetA = indices[:, 0]
                            indSetB = indices[:, 1]
                    elif topID.lower() == "k" and not kFace:
                        indSetA = indices[:, 0]
                        indSetB = indices[:, 1]
                    else:
                        raise Error("Invalid value for topID. value must be" " i, j or k")

                else:
                    raise Error(
                        "Cannot add leading edge constraints. One (and "
                        "exactly one) of FFD block dimensions on the"
                        " specified face must be 2. The dimensions of "
                        "the selected face are: "
                        "(%d, %d). For this case you must specify "
                        "topID" % (shp[0], shp[1])
                    )

        elif indSetA is not None and indSetB is not None:
            if len(indSetA) != len(indSetB):
                raise Error("The length of the supplied indices are not " "the same length")
        else:
            raise Error(
                "Incorrect data supplied to addLeTeConstraint. The "
                "keyword arguments 'volID' and 'faceID' must be "
                "specified **or** 'indSetA' and 'indSetB'"
            )

        if name is None:
            conName = "%s_lete_constraint_%d" % (self.name, len(self.linearCon))
        else:
            conName = name

        # Finally add the volume constraint object
        n = len(indSetA)
        self.linearCon[conName] = LinearConstraint(
            conName, indSetA, indSetB, np.ones(n), np.ones(n), lower=0, upper=0, DVGeo=DVGeo, config=config
        )

    def addLinearConstraintsShape(
        self,
        indSetA,
        indSetB,
        factorA,
        factorB,
        lower=0,
        upper=0,
        name=None,
        config=None,
        childIdx=None,
        comp=None,
        DVGeoName="default",
    ):
        """
        Add a complete generic set of linear constraints for the shape
        variables that have been added to DVGeo. The constraints are
        specified in the following general form::

            lower <= factorA*dvA + factorB*dvB <= upper

        The lists ``indSetA`` and ``indSetB`` are used to specify the pairs of
        control points that are to be linked with linear variables. If
        more than one pair is specified (i.e. :code:`len(indSetA)=len(indSetB)
        > 1`) then ``factorA``, ``factorB``, ``lower`` and ``upper`` may all be arrays
        of the same length or a constant which will applied to all.

        Two sets of indices can be provided, ``indSetA``
        and ``indSetB``. Both must be the same length. These indices may
        be obtained from the ``lindex`` array of the FFD object.

            >>> lIndex = DVGeo.getLocalIndex(iVol)

        ``lIndex`` is a three dimensional set of indices that provide the
        index into the global set of control points. See below for
        examples.

        .. note::
            These constraints will be added to pyOptSparse automatically with a call to
            :func:`addConstraintsPyOpt`.

        Parameters
        ----------
        indSetA : array of int
            Indices of 'A' control points on one side of the FFD
        indSetB : array of int
            Indices of 'B' control points on one side of the FFD
        factorA : float or array
            Coefficient for DV on control point(s) A
        factorB : float or array
            Coefficient for DV on control point(s) B
        lower : float or array
            The lower bound of the constraint(s)
        upper : float or array
            The upper bound of the constraint(s)
        name : str
            Normally this does not need to be set; a default name will
            be generated automatically. Only use this if you have
            multiple DVCon objects and the constraint names need to
            be distinguished
        config : str
            The DVGeo configuration to apply this constraint to. Must be either None
            which will apply to *ALL* the local DV groups or a single string specifying
            a particular configuration.
        childIdx : int
            The zero-based index of the child FFD, if this constraint is being applied to a child FFD.
            The index is defined by the order in which you add the child FFD to the parent.
            For example, the first child FFD has an index of 0, the second an index of 1, and so on.
        comp: str
            The component name if using DVGeometryMulti.

        Examples
        --------
        >>> # Make two sets of controls points move the same amount:
        >>> lIndex = DVGeo.getLocalIndex(1) # Volume 2
        >>> indSetA = []; indSetB = [];
        >>> for i in range(lIndex.shape[0]):
        >>>     indSetA.append(lIndex[i, 0, 0])
        >>>     indSetB.append(lIndex[i, 0, 1])
        >>> DVCon.addLinearConstraintShape(indSetA, indSetB,
        >>>                                factorA=1.0, factorB=-1.0,
        >>>                                lower=0, upper=0)
        """

        self._checkDVGeo(DVGeoName)

        if comp is None:
            DVGeo = self.DVGeometries[DVGeoName]
        else:
            DVGeo = self.DVGeometries[DVGeoName].DVGeoDict[comp]

        if childIdx is not None:
            DVGeo = DVGeo.children[childIdx]

        if len(indSetA) != len(indSetB):
            raise Error("The length of the supplied indices are not " "the same length")

        if name is None:
            conName = "%s_linear_constraint_%d" % (self.name, len(self.linearCon))
        else:
            conName = name

        # Process the inputs to be arrays of length n if necessary.
        factorA = np.atleast_1d(factorA)
        factorB = np.atleast_1d(factorB)
        lower = np.atleast_1d(lower)
        upper = np.atleast_1d(upper)
        n = len(indSetA)

        if len(factorA) == 1:
            factorA = factorA[0] * np.ones(n)
        elif len(factorA) != n:
            raise Error("Length of factorA invalid!")

        if len(factorB) == 1:
            factorB = factorB[0] * np.ones(n)
        elif len(factorB) != n:
            raise Error("Length of factorB invalid!")

        if len(lower) == 1:
            lower = lower[0] * np.ones(n)
        elif len(lower) != n:
            raise Error("Length of lower invalid!")

        if len(upper) == 1:
            upper = upper[0] * np.ones(n)
        elif len(upper) != n:
            raise Error("Length of upper invalid!")

        # Finally add the linear constraint object
        self.linearCon[conName] = LinearConstraint(
            conName, indSetA, indSetB, factorA, factorB, lower, upper, DVGeo, config=config
        )

    def addGearPostConstraint(
        self,
        wimpressCalc,
        position,
        axis,
        thickLower=1.0,
        thickUpper=3.0,
        thickScaled=True,
        MACFracLower=0.50,
        MACFracUpper=0.60,
        name=None,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        """Code for doing landing gear post constraints on the fly in an
        optimization. As it turns out, this is a critical constraint
        for wing-mounted landing gear and high-aspect ratio swept
        wings. This constraint actually encompasses *two*
        optimization constraints:

        1. The first is a physical depth constraint that uses DVCon's
        built in thickness constraint class.

        2. The second constraint is that the x-position of the the
        gear post as a fraction of the wing MAC must be greater than
        MACFracLower which will typically be 50%.

        The calculation uses a wimpressCalc object to determine the
        nominal trapezodial planform to determine the MAC and the
        LE-MAC.

        Parameters
        ----------
        wimpressCalc : wimpressCalc class
            An instance of the wimpress calc class. This is required for
            computing the MAC and the xLEMac

        position : array of size 3
            Three dimensional position of the gear post constraint.

        axis : array of size 3
            Direction to perform projection. Same as 'axis'
            in addThicknessConstraints1D

        thickLower : float
            Lower bound for thickness constraint. If thickScaled=True,
            this is the physical distance scaled by the initial length.
            This value is used as the optimization constraint lower bound.

        thickUpper : float
            Upper bound for optimization constraint. See thickLower.

        thickScaled : bool
            Flag specifying if the constraint should be scaled.
            It is true by default. The default values of thickScaled=True,
            thickLower=1.0, ensures that the initial thickness does not
            decrease.

        MACFracLower : float
            The desired lower bound for the gear post location as a
            fraction of MAC. Default is 0.50

        MACFracUpper : float
            The desired upper bound for the gear post location as a
            fraction of MAC. Default is 0.60

        name : str
            Normally this does not need to be set. Only use this if
            you have multiple DVCon objects and the constraint names
            need to be distinguished **or** the values are to be used
            in a subsequent computation.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) *before* they are
            given to the optimizer, set this flag to False.

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """

        self._checkDVGeo(DVGeoName)
        p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)

        typeName = "gearCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_gear_constraint_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name

        # Project the actual location we were give:
        up, down, fail = geo_utils.projectNode(position, axis, p0, p1 - p0, p2 - p0)
        if fail > 0:
            raise Error(
                "There was an error projecting a node " "at (%f, %f, %f) with normal (%f, %f, %f)." % (position)
            )

        self.constraints[typeName][conName] = GearPostConstraint(
            conName,
            wimpressCalc,
            up,
            down,
            thickLower,
            thickUpper,
            thickScaled,
            MACFracLower,
            MACFracUpper,
            self.DVGeometries[DVGeoName],
            addToPyOpt,
            compNames,
        )

    def addCircularityConstraint(
        self,
        origin,
        rotAxis,
        radius,
        zeroAxis,
        angleCW,
        angleCCW,
        nPts=4,
        upper=1.0,
        lower=1.0,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        DVGeoName="default",
        compNames=None,
    ):
        """
        Add a contraint to keep a certain portion of your geometry circular.
        Define the origin, central axis and radius to define the circle.
        Define the zero axis, and two angles to define the portion of the circle to
        use for the constraint
        The constraint will enforce that the radial lengths from the origin to the
        nPts around the circle stay equal.

        Parameters
        ----------
        origin: vector
              The coordinate of the origin

        rotation: vector
              The central axis of the circle

        radius: float
              The radius of the circle

        zeroAxis: vector
              The axis defining the zero rotation angle around the circle

        angleCW : float
              Angle in the clockwise direction to extend the circularity constraint.
              Angle should be positive. Angles are specified in degrees.

        angleCCW : float
              Angle in the counter-clockwise direction to extend the
              circularity constraint. Angle should be positive.
              Angles are specified in degrees.

        nPts : int
             Number of points in the discretization of the circle

        lower : float
            Lower bound for circularity. This is the ratio of the target length
            relative to the first length calculated

        upper : float
            Upper bound for optimization constraint. See lower.

        scale : float
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If scaled=True, this automatically results in a
            well-scaled constraint and scale can be left at 1.0. If
            scaled=False, it may changed to a more suitable value of
            the resulting physical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constraint names need to
             be distinguished **OR** you are using this volume
             computation for something other than a direct constraint
             in pyOpt, i.e. it is required for a subsequent
             computation.

        addToPyOpt : bool
            Normally this should be left at the default of True if the
            volume is to be used as a constraint. If the volume is to
            used in a subsequent calculation and not a constraint
            directly, addToPyOpt should be False, and name
            specified to a logical name for this computation. with
            addToPyOpt=False, the lower, upper and scale variables are
            meaningless

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """

        self._checkDVGeo(DVGeoName)
        coords = self._generateCircle(origin, rotAxis, radius, zeroAxis, angleCW, angleCCW, nPts)

        # Create the circularity constraint object:
        coords = coords.reshape((nPts, 3))
        origin = np.array(origin).reshape((1, 3))

        typeName = "circCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        # Create a name
        if name is None:
            conName = "%s_circularity_constraints_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = CircularityConstraint(
            conName, origin, coords, lower, upper, scale, self.DVGeometries[DVGeoName], addToPyOpt, compNames
        )

    def addSurfaceAreaConstraint(
        self,
        lower=1.0,
        upper=3.0,
        scaled=True,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        """
        Sum up the total surface area of the triangles included in the DVCon surface

        Parameters
        ----------
        lower : float
            The lower bound for the area constraint.

        upper : float
            The upper bound for the area constraint.

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial area is defined to be 1.0.
              In this case, the lower and upper bounds are given in
              multiple of the initial area. lower=0.85, upper=1.15,
              would allow for 15% change in area both upper and
              lower. For aerodynamic optimization, this is the most
              widely used option .

            * scaled=False: No scaling is applied and the physical
              area. lower and upper refer to the physical areas.

        scale : float
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If scaled=True, this automatically results in a
            well-scaled constraint and scale can be left at 1.0. If
            scaled=False, it may changed to a more suitable value of
            the resulting physical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constraint names need to
             be distinguished **OR** you are using this volume
             computation for something other than a direct constraint
             in pyOpt, i.e. it is required for a subsequent
             computation.

        addToPyOpt : bool
            Normally this should be left at the default of True if the
            volume is to be used as a constraint. If the volume is to
            used in a subsequent calculation and not a constraint
            directly, addToPyOpt should be False, and name
            specified to a logical name for this computation. with
            addToPyOpt=False, the lower, upper and scale variables are
            meaningless

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """

        self._checkDVGeo(DVGeoName)
        p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)

        typeName = "surfAreaCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        # Create a name
        if name is None:
            conName = "%s_surfaceArea_constraints_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = SurfaceAreaConstraint(
            conName,
            p0,
            p1 - p0,
            p2 - p0,
            lower,
            upper,
            scale,
            scaled,
            self.DVGeometries[DVGeoName],
            addToPyOpt,
            compNames,
        )

    def addProjectedAreaConstraint(
        self,
        axis="y",
        lower=1.0,
        upper=3.0,
        scaled=True,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        """
        Sum up the total surface area of the triangles included in the
        DVCon surface projected to a plane defined by "axis".

        Parameters
        ----------
        axis : str
            The axis normal to the projection plane. ('x', 'y', or 'z')
        lower : float
            The lower bound for the area constraint.

        upper : float
            The upper bound for the area constraint.

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial area is defined to be 1.0.
              In this case, the lower and upper bounds are given in
              multiple of the initial area. lower=0.85, upper=1.15,
              would allow for 15% change in area both upper and
              lower. For aerodynamic optimization, this is the most
              widely used option .

            * scaled=False: No scaling is applied and the physical
              area. lower and upper refer to the physical areas.

        scale : float
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If scaled=True, this automatically results in a
            well-scaled constraint and scale can be left at 1.0. If
            scaled=False, it may changed to a more suitable value of
            the resulting physical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constraint names need to
             be distinguished **OR** you are using this volume
             computation for something other than a direct constraint
             in pyOpt, i.e. it is required for a subsequent
             computation.

        addToPyOpt : bool
            Normally this should be left at the default of True if the
            volume is to be used as a constraint. If the volume is to
            used in a subsequent calculation and not a constraint
            directly, addToPyOpt should be False, and name
            specified to a logical name for this computation. with
            addToPyOpt=False, the lower, upper and scale variables are
            meaningless

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """

        self._checkDVGeo(DVGeoName)
        p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)

        if axis == "x":
            axis = np.array([1, 0, 0])
        elif axis == "y":
            axis = np.array([0, 1, 0])
        elif axis == "z":
            axis = np.array([0, 0, 1])

        typeName = "projAreaCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        # Create a name
        if name is None:
            conName = "%s_projectedArea_constraints_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = ProjectedAreaConstraint(
            conName,
            p0,
            p1 - p0,
            p2 - p0,
            axis,
            lower,
            upper,
            scale,
            scaled,
            self.DVGeometries[DVGeoName],
            addToPyOpt,
            compNames,
        )

    def addPlanarityConstraint(
        self,
        origin,
        planeAxis,
        upper=0.0,
        lower=0.0,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        """
        Add a contraint to keep the surface in set in DVCon planar
        Define the origin, and the plane axis.
        The constraint will enforce that all of the surface points lie on
        the plane.

        Parameters
        ----------
        origin: vector
              The coordinate of the origin

        planeAxis: vector
              Vector defining the plane of interest

        lower : float
            Lower bound for circularity. This is the ratio of the target length
            relative to the first length calculated

        upper : float
            Upper bound for optimization constraint. See lower.

        scale : float
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If scaled=True, this automatically results in a
            well-scaled constraint and scale can be left at 1.0. If
            scaled=False, it may changed to a more suitable value of
            the resulting physical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constraint names need to
             be distinguished **OR** you are using this volume
             computation for something other than a direct constraint
             in pyOpt, i.e. it is required for a subsequent
             computation.

        addToPyOpt : bool
            Normally this should be left at the default of True if the
            volume is to be used as a constraint. If the volume is to
            used in a subsequent calculation and not a constraint
            directly, addToPyOpt should be False, and name
            specified to a logical name for this computation. with
            addToPyOpt=False, the lower, upper and scale variables are
            meaningless

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """

        self._checkDVGeo(DVGeoName)
        p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)

        # Create the circularity constraint object:
        origin = np.array(origin).reshape((1, 3))
        planeAxis = np.array(planeAxis).reshape((1, 3))

        # Create a name
        typeName = "planeCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_planarity_constraints_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = PlanarityConstraint(
            conName,
            planeAxis,
            origin,
            p0,
            p1 - p0,
            p2 - p0,
            lower,
            upper,
            scale,
            self.DVGeometries[DVGeoName],
            addToPyOpt,
            compNames,
        )

    def addColinearityConstraint(
        self,
        origin,
        lineAxis,
        distances,
        upper=0.0,
        lower=0.0,
        scale=1.0,
        name=None,
        addToPyOpt=True,
        DVGeoName="default",
        compNames=None,
    ):
        """
        Add a contraint to keep a set of points aligned.
        Define the origin, and axis of the line and then a set of distances
        along the axis to constrain.
        The constraint will compute the points to constrain.

        Parameters
        ----------
        origin: vector
              The coordinate of the origin (3x numpy array)

        lineAxis: vector
              The line of colinearity (3x numpy array)

        distances: list
              List of distances from origin to constrain

        lower : float
            Lower bound for circularity. This is the ratio of the target length
            relative to the first length calculated

        upper : float
            Upper bound for optimization constraint. See lower.

        scale : float
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If scaled=True, this automatically results in a
            well-scaled constraint and scale can be left at 1.0. If
            scaled=False, it may changed to a more suitable value of
            the resulting physical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constraint names need to
             be distinguished **OR** you are using this volume
             computation for something other than a direct constraint
             in pyOpt, i.e. it is required for a subsequent
             computation.

        addToPyOpt : bool
            Normally this should be left at the default of True if the
            volume is to be used as a constraint. If the volume is to
            used in a subsequent calculation and not a constraint
            directly, addToPyOpt should be False, and name
            specified to a logical name for this computation. with
            addToPyOpt=False, the lower, upper and scale variables are
            meaningless

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """

        self._checkDVGeo(DVGeoName)
        nPts = len(distances)
        coords = []
        for dist in distances:
            coords.append(dist * lineAxis + origin)

        # Create the circularity constraint object:
        coords = np.array(coords).reshape((nPts, 3))
        origin = np.array(origin).reshape((1, 3))
        lineAxis = np.array(lineAxis).reshape((1, 3))

        # Create a name
        typeName = "coLinCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_colinearity_constraints_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = ColinearityConstraint(
            conName, lineAxis, origin, coords, lower, upper, scale, self.DVGeometries[DVGeoName], addToPyOpt, compNames
        )

    def addCurvatureConstraint(
        self,
        surfFile,
        curvatureType="Gaussian",
        lower=-1e20,
        upper=1e20,
        scaled=True,
        scale=1.0,
        KSCoeff=None,
        name=None,
        addToPyOpt=False,
        DVGeoName="default",
        compNames=None,
    ):
        """
        Add a curvature contraint for the prescribed surface. The only required input for this
        constraint is a structured plot 3D file of the surface (there can be multiple
        surfaces in the file). This value is meant to be corelated with manufacturing costs.

        Parameters
        ----------
        surfFile: vector
            Plot3D file with desired surface, should be sufficiently refined to
            accurately capture surface curvature

        curvatureType: str
            The type of curvature to calculate. Options are: 'Gaussian', 'mean', 'combined', or 'KSmean'.
            Here the Gaussian curvature is K=kappa_1*kappa_2, the mean curvature is H = 0.5*(kappa_1+kappa_2),
            the combined curvature C = kappa_1^2+kappa_2^2=(2*H)^2-2*K, and the KSmean curvature applies the
            KS function to the mean curvature, which is essentially the max local mean curvature on the prescribed
            surface. In practice, we compute the squared integrated value over the surface, e.g., sum(H*H*dS), for the
            Gaussian, mean and combined curvatures. While for the KSmean, we applied the KS function, i.e., KS(H*H*dS)

        lower : float
            Lower bound for curvature integral.

        upper : float
            Upper bound for optimization constraint. See lower.

        scale : float
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If scaled=True, this automatically results in a
            well-scaled constraint and scale can be left at 1.0. If
            scaled=False, it may changed to a more suitable value of
            the resulting physical volume magnitude is vastly different
            from O(1).

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial curvature is defined to be 1.0.
              In this case, the lower and upper bounds are given in
              multiple of the initial curvature. lower=0.85, upper=1.15,
              would allow for 15% change in curvature both upper and
              lower. For aerodynamic optimization, this is the most
              widely used option .

            * scaled=False: No scaling is applied and the physical
              curvature. lower and upper refer to the physical curvatures.

        KSCoeff : float
            The coefficient for KS function when curvatureType=KSmean.
            This controls how close the KS function approximates the original
            functions. One should select a KSCoeff such that the printed "Reference curvature"
            is only slightly larger than the printed "Max curvature" for the baseline surface.
            The default value of KSCoeff is the number of points in the plot3D files.

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constraint names need to
             be distinguished **OR** you are using this volume
             computation for something other than a direct constraint
             in pyOpt, i.e. it is required for a subsequent
             computation.

        addToPyOpt : bool
            Normally this should be left at the default of False if the
            cost is part of the objective. If the integral is to be
            used directly as a constraint, addToPyOpt should be True, and name
            specified to a logical name for this computation. with
            addToPyOpt=False, the lower, upper and scale variables are
            meaningless

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        """

        self._checkDVGeo(DVGeoName)

        # Use pyGeo to load the plot3d file
        geo = pyGeo("plot3d", surfFile)
        # node and edge tolerance for pyGeo (these are never used so
        # we just fix them)
        node_tol = 1e-8
        edge_tol = 1e-8
        # Explicitly do the connectivity here since we don't want to
        # write a con file:
        geo._calcConnectivity(node_tol, edge_tol)
        surfs = geo.surfs
        typeName = "curveCon"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()
        # Create a name
        if name is None:
            if curvatureType == "Gaussian":
                conName = "%s_gaussian_curvature_constraint_%d" % (self.name, len(self.constraints[typeName]))
            elif curvatureType == "mean":
                conName = "%s_mean_curvature_constraint_%d" % (self.name, len(self.constraints[typeName]))
            elif curvatureType == "combined":
                conName = "%s_combined_curvature_constraint_%d" % (self.name, len(self.constraints[typeName]))
            elif curvatureType == "KSmean":
                conName = "%s_ksmean_curvature_constraint_%d" % (self.name, len(self.constraints[typeName]))
            else:
                raise Error(
                    "The curvatureType parameter should be Gaussian, mean, combined, or KSmean "
                    "%s is not supported!" % curvatureType
                )
        else:
            conName = name
        self.constraints[typeName][conName] = CurvatureConstraint(
            conName,
            surfs,
            curvatureType,
            lower,
            upper,
            scaled,
            scale,
            KSCoeff,
            self.DVGeometries[DVGeoName],
            addToPyOpt,
            compNames,
        )

    def addCurvatureConstraint1D(
        self,
        start,
        end,
        nPts,
        axis,
        curvatureType="mean",
        lower=-1e20,
        upper=1e20,
        scaled=True,
        scale=1.0,
        KSCoeff=1.0,
        name=None,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        """
        Add a curvature contraint along the prescribed straightline on the design surface.
        This can be used to impose a spanwise curvature constraint for wing aerodynamic optimization.

        .. note:: the output is the square of the curvature to make sure the values are always positive

        See below for a schematic.

        .. code-block:: text

          Planform view of the wing:

          Physical extent of wing
          ____________________________________
          |                                  |
          |                                  |
          |                                  |
          | +---x--x---x---x---x---x---x---+ |
          |                                  |
          |                                  |
          |                                  |
          |__________________________________/

          The '+' are the (three dimensional) points defined by 'start' and 'end'. Once the straightline
          is defined, we generate nPts-2 intermediate points along it and project these points to the design
          surface mesh in the prescirbed axis direction. Here 'x' are the intermediate points added by setting
          nPts = 9. The curvature will be calculated based on the projected intermediate points (x) on the design
          surface.

        .. note::
           We do not calculate the curvatures at the two end points (+).
           So make sure to extend the start and end points a bit to fully cover the area where you want to compute the curvature

        Parameters
        ----------
        start/end : list of size 3
            The 3D points forming a straight line along which the
            curvature constraints will be added.

        nPts : int
            The number of intermediate points to add. We should prescribe nPts such that the point interval
            should be larger than the mesh size and smaller than the interval of FFD points in that direction

        axis : list of size 3
            The direction along which the projections will occur.
            Typically this will be y or z axis ([0,1,0] or [0,0,1])

            .. note:: we also compute the curvature based on this axis dir

        curvatureType : str
            What type of curvature constraint to compute. Either mean or aggregated

        lower : float
            Lower bound of curvature integral used for optimization constraint

        upper : float
            Upper bound of curvature integral used for optimization constraint

        scale : float
            This is the optimization scaling of the
            constraint. Typically this parameter will not need to be
            changed. If scaled=True, this automatically results in a
            well-scaled constraint and scale can be left at 1.0. If
            scaled=False, it may changed to a more suitable value of
            the resulting physical volume magnitude is vastly different
            from O(1).

        scaled : bool
            Flag specifying whether or not the constraint is to be
            implemented in a scaled fashion or not.

            * scaled=True: The initial curvature is defined to be 1.0.
              In this case, the lower and upper bounds are given in
              multiple of the initial curvature. lower=0.85, upper=1.15,
              would allow for 15% change in curvature both upper and
              lower. For aerodynamic optimization, this is the most
              widely used option .

            * scaled=False: No scaling is applied and the physical
              curvature. lower and upper refer to the physical curvatures.

        KSCoeff : float
            The coefficient for KS function. This controls how close the KS function approximates
            the original functions.

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constraint names need to
             be distinguished **OR** you are using this volume
             computation for something other than a direct constraint
             in pyOpt, i.e. it is required for a subsequent
             computation.

        addToPyOpt : bool
            Normally this should be left at the default of False if the
            cost is part of the objective. If the integral is to be
            used directly as a constraint, addToPyOpt should be True, and name
            specified to a logical name for this computation. with
            addToPyOpt=False, the lower, upper and scale variables are
            meaningless

        surfaceName : str
            Name of the surface to project to. This should be the same
            as the surfaceName provided when setSurface() was called.
            For backward compatibility, the name is 'default' by default.

        DVGeoName : str
            Name of the DVGeo object to compute the constraint with. You only
            need to set this if you're using multiple DVGeo objects
            for a problem. For backward compatibility, the name is 'default' by default

        compNames : list
            If using DVGeometryMulti, the components to which the point set associated
            with this constraint should be added.
            If None, the point set is added to all components.

        Examples
        --------
        >>> # define a 2 point poly-line along the wing spanwise direction (z)
        >>> # and project to the design surface along y
        >>> start = [0, 0, 0]
        >>> end = [0, 0, 1]
        >>> nPts = 10
        >>> axis = [0, 1, 0]
        >>> DVCon.addCurvatureConstraint1D(start, end, nPts, axis, "mean", lower=1.0, upper=3, scaled=True)
        """

        self._checkDVGeo(DVGeoName)

        p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)

        if nPts < 5:
            raise Error("nPts should be at least 5 \n " "while nPts = %d is given." % nPts)

        # Create mesh of intersections
        ptList = [start, end]
        constr_line = Curve(X=ptList, k=2)
        s = np.linspace(0, 1, nPts)
        X = constr_line(s)
        coords = np.zeros((nPts, 3))

        # calculate the distance between coords, it should be uniform for all points
        eps = np.linalg.norm(X[1] - X[0])

        # Project all the points
        for i in range(nPts):
            # Project actual node:
            up, _, fail = geo_utils.projectNode(X[i], axis, p0, p1 - p0, p2 - p0)
            if fail > 0:
                raise Error(
                    "There was an error projecting a node "
                    "at (%f, %f, %f) with normal (%f, %f, %f)." % (X[i, 0], X[i, 1], X[i, 2], axis[0], axis[1], axis[2])
                )
            coords[i] = up
            # NOTE: we do not use the down projection

        typeName = "curvCon1D"
        if typeName not in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = "%s_curvature_constraints_1d_%d" % (self.name, len(self.constraints[typeName]))
        else:
            conName = name

        self.constraints[typeName][conName] = CurvatureConstraint1D(
            conName,
            curvatureType,
            coords,
            axis,
            eps,
            KSCoeff,
            lower,
            upper,
            scaled,
            scale,
            self.DVGeometries[DVGeoName],
            addToPyOpt,
            compNames,
        )

    def addMonotonicConstraints(
        self, key, slope=1.0, name=None, start=0, stop=-1, config=None, childIdx=None, comp=None, DVGeoName="default"
    ):
        """
        Add monotonic constraints to a given design variable.

        Parameters
        ----------
        key : str
            Name of the global design variable we are dealing with.
        slope : float
            Direction of monotonic decrease
                1.0 - from left to right along design variable vector
                -1.0 - from right to left along design variable vector
        name : str
            Normally this does not need to be set; a default name will
            be generated automatically. Only use this if you have
            multiple DVCon objects and the constraint names need to
            be distinguished
        start/stop: int
            This allows the user to specify a slice of the design variable to
            constrain if it is not desired to set a monotonic constraint on the
            entire vector. The start/stop indices are inclusive indices, so for
            a design variable vector [4, 3, 6.5, 2, -5.4, -1], start=1 and
            stop=4 would constrain [3, 6.5, 2, -5.4] to be a monotonic sequence.
        config : str
            The DVGeo configuration to apply this constraint to. Must be either None
            which will apply to *ALL* the local DV groups or a single string specifying
            a particular configuration.
        childIdx : int
            The zero-based index of the child FFD, if this constraint is being applied to a child FFD.
            The index is defined by the order in which you add the child FFD to the parent.
            For example, the first child FFD has an index of 0, the second an index of 1, and so on.
        comp: str
            The component name if using DVGeometryMulti.

        Examples
        --------
        >>> DVCon.addMonotonicConstraints('chords', 1.0)
        """
        self._checkDVGeo(DVGeoName)

        if comp is None:
            DVGeo = self.DVGeometries[DVGeoName]
        else:
            DVGeo = self.DVGeometries[DVGeoName].DVGeoDict[comp]

        if childIdx is not None:
            DVGeo = DVGeo.children[childIdx]

        if name is None:
            conName = "%s_monotonic_constraint_%d" % (self.name, len(self.linearCon))
        else:
            conName = name

        options = {"slope": slope, "start": start, "stop": stop}
        # Finally add the global linear constraint object
        self.linearCon[conName] = GlobalLinearConstraint(
            conName,
            key,
            conType="monotonic",
            options=options,
            lower=0,
            upper=None,
            DVGeo=DVGeo,
            config=config,
        )

    def _checkDVGeo(self, name="default"):
        """check if DVGeo exists"""
        if name not in self.DVGeometries.keys():
            raise Error(
                "A DVGeometry object must be added to DVCon before "
                "using a call to DVCon.setDVGeo(DVGeo) before "
                "constraints can be added."
            )

    def _getSurfaceVertices(self, surfaceName):
        if surfaceName not in self.surfaces.keys():
            raise KeyError('Need to add surface "' + surfaceName + '" to the DVConstraints object')
        p0 = self.surfaces[surfaceName][0]
        p1 = self.surfaces[surfaceName][1]
        p2 = self.surfaces[surfaceName][2]
        return p0, p1, p2

    def _generateIntersections(self, leList, teList, nSpan, nChord, surfaceName):
        """
        Internal function to generate the grid points (nSpan x nChord)
        and to actual perform the intersections. This is in a separate
        functions since addThicknessConstraints2D, and volume based
        constraints use the same code. The list of projected
        coordinates are returned.
        """
        p0, p1, p2 = self._getSurfaceVertices(surfaceName=surfaceName)

        # Create mesh of intersections
        le_s = Curve(X=leList, k=2)
        te_s = Curve(X=teList, k=2)
        root_s = Curve(X=[leList[0], teList[0]], k=2)
        tip_s = Curve(X=[leList[-1], teList[-1]], k=2)

        # Generate spanwise parametric distances
        if isinstance(nSpan, int):
            # Use equal spacing along the curve
            le_span_s = te_span_s = np.linspace(0.0, 1.0, nSpan)
        elif isinstance(nSpan, list):
            # Use equal spacing within each segment defined by leList and teList

            # We use the same nSpan for the leading and trailing edges, so check that the lists are the same size
            if len(leList) != len(teList):
                raise ValueError("leList and teList must be the same length if nSpan is provided as a list.")

            # Also check that nSpan is the correct length
            numSegments = len(leList) - 1
            if len(nSpan) != numSegments:
                raise ValueError(f"nSpan must be of length {numSegments}.")

            # Find the parametric distances of the break points that define each segment
            le_breakPoints = le_s.projectPoint(leList)[0]
            te_breakPoints = te_s.projectPoint(teList)[0]

            # Initialize empty arrays for the full spanwise parameteric distances
            le_span_s = np.array([])
            te_span_s = np.array([])

            for i in range(numSegments):
                # Only include the endpoint if this is the last segment to avoid double counting points
                if i == numSegments - 1:
                    endpoint = True
                else:
                    endpoint = False

                # Interpolate over this segment and append to the parametric distance array
                le_span_s = np.append(
                    le_span_s, np.linspace(le_breakPoints[i], le_breakPoints[i + 1], nSpan[i], endpoint=endpoint)
                )
                te_span_s = np.append(
                    te_span_s, np.linspace(te_breakPoints[i], te_breakPoints[i + 1], nSpan[i], endpoint=endpoint)
                )
        else:
            raise TypeError("nSpan must be either an int or a list.")

        # Generate chordwise parametric distances
        chord_s = np.linspace(0.0, 1.0, nChord)

        # Get the total number of spanwise sections
        nSpanTotal = np.sum(nSpan)

        # Generate a 2D region of intersections
        X = geo_utils.tfi_2d(le_s(le_span_s), te_s(te_span_s), root_s(chord_s), tip_s(chord_s))
        coords = np.zeros((nSpanTotal, nChord, 2, 3))
        for i in range(nSpanTotal):
            for j in range(nChord):
                # Generate the 'up_vec' from taking the cross product
                # across a quad
                if i == 0:
                    uVec = X[i + 1, j] - X[i, j]
                elif i == nSpanTotal - 1:
                    uVec = X[i, j] - X[i - 1, j]
                else:
                    uVec = X[i + 1, j] - X[i - 1, j]

                if j == 0:
                    vVec = X[i, j + 1] - X[i, j]
                elif j == nChord - 1:
                    vVec = X[i, j] - X[i, j - 1]
                else:
                    vVec = X[i, j + 1] - X[i, j - 1]

                upVec = np.cross(uVec, vVec)
                # Project actual node:
                up, down, fail = geo_utils.projectNode(X[i, j], upVec, p0, p1 - p0, p2 - p0)

                if fail == 0:
                    coords[i, j, 0] = up
                    coords[i, j, 1] = down
                elif fail == -1:
                    # More than 2 solutions. Returned in sorted distance.
                    coords[i, j, 0] = down
                    coords[i, j, 1] = up
                else:
                    raise Error(
                        "There was an error projecting a node at (%f, %f, %f) with normal (%f, %f, %f)."
                        % (X[i, j, 0], X[i, j, 1], X[i, j, 2], upVec[0], upVec[1], upVec[2])
                    )

        return coords

    def _generateDiscreteSurface(self, wing):
        """
        Take a pygeo surface and create a discrete triangulated
        surface from it. This is quite dumb code and does not pay any
        attention to things like how well the triangles approximate
        the surface or the underlying parametrization of the surface
        """

        p0 = []
        v1 = []
        v2 = []
        level = 1
        for isurf in range(wing.nSurf):
            surf = wing.surfs[isurf]
            ku = surf.ku
            kv = surf.kv
            tu = surf.tu
            tv = surf.tv

            u = geo_utils.fillKnots(tu, ku, level)
            v = geo_utils.fillKnots(tv, kv, level)

            for i in range(len(u) - 1):
                for j in range(len(v) - 1):
                    P0 = surf(u[i], v[j])
                    P1 = surf(u[i + 1], v[j])
                    P2 = surf(u[i], v[j + 1])
                    P3 = surf(u[i + 1], v[j + 1])

                    p0.append(P0)
                    v1.append(P1 - P0)
                    v2.append(P2 - P0)

                    p0.append(P3)
                    v1.append(P2 - P3)
                    v2.append(P1 - P3)

        return np.array(p0), np.array(v1), np.array(v2)

    def _generateCircle(self, origin, rotAxis, radius, zeroAxis, angleCW, angleCCW, nPts):
        """
        generate the coordinates for a circle. The user should not have to call this
        directly.

        Parameters
        ----------
        origin: vector
              The coordinate of the origin

        rotation: vector
              The central axis of the circle

        radius: float
              The radius of the circle

        zeroAxis: vector
              The axis defining the zero rotation angle around the circle

        angleCW : float
              Angle in the clockwise direction to extend the circularity constraint.
              Angles are specified in degrees.

        angleCCW : float
              Angle in the counter-clockwise direction to extend the
              circularity constraint. Angles are specified in degrees.

        nPts : int
             Number of points in the discretization of the circle

        """
        # enforce the shape of the origin
        origin = np.array(origin).reshape((3,))

        # Create the coordinate array
        coords = np.zeros((nPts, 3))

        # get the angles about the zero axis for the points
        if angleCW < 0:
            raise Error("Negative angle specified. angleCW should be positive.")
        if angleCCW < 0:
            raise Error("Negative angle specified. angleCCW should be positive.")

        angles = np.linspace(np.deg2rad(-angleCW), np.deg2rad(angleCCW), nPts)

        # ---------
        # Generate a unit vector in the zero axis direction
        # ----
        # get the third axis by taking the cross product of rotAxis and zeroAxis
        axis = np.cross(zeroAxis, rotAxis)

        # now use these axis to regenerate the orthogonal zero axis
        zeroAxisOrtho = np.cross(rotAxis, axis)

        # now normalize the length of the zeroAxisOrtho
        length = np.linalg.norm(zeroAxisOrtho)
        zeroAxisOrtho /= length

        # -------
        # Normalize the rotation axis
        # -------
        length = np.linalg.norm(rotAxis)
        rotAxis /= length

        # ---------
        # now rotate, multiply by radius ,and add to the origin to get the coords
        # ----------

        for i in range(nPts):
            newUnitVec = geo_utils.rotVbyW(zeroAxisOrtho, rotAxis, angles[i])
            newUnitVec *= radius
            coords[i, :] = newUnitVec + origin

        return coords
