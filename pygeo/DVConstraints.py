# ======================================================================
#         Imports
# ======================================================================
import numpy,copy
from . import geo_utils, pyGeo
from pyspline import pySpline
from mpi4py import MPI
from scipy.sparse import csr_matrix
try:
    from collections import OrderedDict
except ImportError:
    try:
        from ordereddict import OrderedDict
    except ImportError:
        print("Could not find any OrderedDict class. For 2.6 and earlier, "
              "use:\n pip install ordereddict")

class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a expliclty raised exception.
    """
    def __init__(self, message):
        msg = '\n+'+'-'*78+'+'+'\n' + '| DVCon Error: '
        i = 14
        for word in message.split():
            if len(word) + i + 1 > 78: # Finish line and start new one
                msg += ' '*(78-i)+'|\n| ' + word + ' '
                i = 1 + len(word)+1
            else:
                msg += word + ' '
                i += len(word)+1
        msg += ' '*(78-i) + '|\n' + '+'+'-'*78+'+'+'\n'
        print(msg)
        Exception.__init__(self)

class Warning(object):
    """
    Format a warning message
    """
    def __init__(self, message):
        msg = '\n+'+'-'*78+'+'+'\n' + '| DVConstraints Warning: '
        i = 24
        for word in message.split():
            if len(word) + i + 1 > 78: # Finish line and start new one
                msg += ' '*(78-i)+'|\n| ' + word + ' '
                i = 1 + len(word)+1
            else:
                msg += word + ' '
                i += len(word)+1
        msg += ' '*(78-i) + '|\n' + '+'+'-'*78+'+'+'\n'
        print(msg)

class GeometricConstraint(object):
    """
    This is a generic base class for all of the geometric constraints.

    """
    def __init__(self,name, nCon, lower, upper, scale, DVGeo, addToPyOpt):
        """
        General init function. Every constraint has these functions
        """
        self.name = name
        self.nCon = nCon
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        return

    def setDesignVars(self,x):
        """
        take in the design var vector from pyopt and set the variables for this constraint
        This function is constraint specific, so the baseclass doesn't implement anything.
        """
        pass

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary.
        This function is constraint specific, so the baseclass doesn't implement anything.

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        pass

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary
        This function is constraint specific, so the baseclass doesn't implement anything.

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        pass

    def getVarNames(self):
        """
        return the var names relevant to this constraint. By default, this is the DVGeo
        variables, but some constraints may extend this to include other variables.
        """
        return self.DVGeo.getVarNames()

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addConGroup(self.name, self.nCon, lower=self.lower,
                                upper=self.upper, scale=self.scale,
                                wrt=self.getVarNames())

    def addVariablesPyOpt(self, optProb):
        """
        Add the variables to pyOpt, if the flag is set
        """
        # if self.addToPyOpt:
        #     optProb.addVarGroup(self.name, self.nCon, lower=self.lower,
        #                         upper=self.upper, scale=self.scale,
        #                         wrt=self.getVarNames())

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        This function is constraint specific, so the baseclass doesn't implement anything.
        """
        pass


class DVConstraints(object):
    """DVConstraints provides a convenient way of defining geometric
    constraints for WINGS. This can be very useful for a constrained
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
    name: str
        A name for this object. Used to distiguish between DVCon objects
        if multiple DVConstraint objects are used in an optimization.

    """

    def __init__(self,name='DVCon1'):
        """
        Create a (empty) DVconstrains object. Specific types of
        constraints will added individually
        """

        self.name = name

        self.constraints = OrderedDict()
        self.linearCon = OrderedDict()

        self.DVGeo = None
        # Data for the discrete surface
        self.p0 = None
        self.v1 = None
        self.v2 = None

    def setSurface(self, surf):
        """
        Set the surface DVConstraints will use to perform projections.

        Parameters
        ----------
        surf : pyGeo object or list

            This is the surface representation to use for
            projections. If available, a pyGeo surface object can be
            used OR a triagnulaed surface in the form [p0, v1, v2] can
            be used. This triangulated surface form can be supplied
            form pyADflow or from pyTrian.

        Examples
        --------
        >>> CFDsolver = ADFLOW(comm=comm, options=aeroOptions)
        >>> surf = CFDsolver.getTriangulatedMeshSurface()
        >>> DVCon.setSurface(surf)
        >>> # Or using a pyGeo surface object:
        >>> surf = pyGeo('iges',fileName='wing.igs')
        >>> DVCon.setSurface(surf)

        """

        if type(surf) == list:
            # Data from ADflow
            self.p0 = numpy.array(surf[0])
            self.v1 = numpy.array(surf[1])
            self.v2 = numpy.array(surf[2])

        elif isinstance(surf, str):
            # Load the surf as a plot3d file
            self.p0, self.v1, self.v2 = self._readPlot3DSurfFile(surf)

        else: # Assume it's a pyGeo surface
            self._generateDiscreteSurface(surf)

    def setDVGeo(self, DVGeo):
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

        self.DVGeo = DVGeo

    def addConstraintsPyOpt(self, optProb):
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
        # constraints to pyopt
        for conTypeKey in self.constraints:
            constraint = self.constraints[conTypeKey]
            for key in constraint:
                constraint[key].addConstraintsPyOpt(optProb)

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
        # variables to pyopt
        for conTypeKey in self.constraints:
            constraint = self.constraints[conTypeKey]
            for key in constraint:
                constraint[key].addVariablesPyOpt(optProb)

        # linear contraints are ignored because at the moment there are no linear
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
        # variables to pyopt
        for conTypeKey in self.constraints:
            constraint = self.constraints[conTypeKey]
            for key in constraint:
                constraint[key].setDesignVars(dvDict)

        # linear contraints are ignored because at the moment there are no linear
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
        Evaluate the derivative of all the 'funcitons' that this
        object has. These functions are just the constraint values.
        Thse values will be set directly in the funcSens dictionary.

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

        f = open(fileName, 'w')
        f.write("TITLE = \"DVConstraints Data\"\n")
        f.write("VARIABLES = \"CoordinateX\" \"CoordinateY\" \"CoordinateZ\"\n")

        # loop over the constraints and add their data to the tecplot file
        for conTypeKey in self.constraints:
            constraint = self.constraints[conTypeKey]
            for key in constraint:
                constraint[key].writeTecplot(f)

        for key in self.linearCon:
            self.linearCon[key].writeTecplot(f)
        f.close()

    def writeSurfaceTecplot(self,fileName):
        """
        Write the triangulated surface mesh used in the constraint object
        to a tecplot file for visualization.

        Parameters
        ----------
        fileName : str
            File name for tecplot file. Should have a .dat extension.

        """
        f = open(fileName, 'w')
        f.write("TITLE = \"DVConstraints Surface Mesh\"\n")
        f.write("VARIABLES = \"CoordinateX\" \"CoordinateY\" \"CoordinateZ\"\n")
        f.write('Zone T=%s\n'%('surf'))
        f.write('Nodes = %d, Elements = %d ZONETYPE=FETRIANGLE\n'% (
            len(self.p0)*3, len(self.p0)))
        f.write('DATAPACKING=POINT\n')
        for i in range(len(self.p0)):
            points = []
            points.append(self.p0[i])
            points.append(self.p0[i]+self.v1[i])
            points.append(self.p0[i]+self.v2[i])
            for i in range(len(points)):
                f.write('%f %f %f\n'% (points[i][0], points[i][1],points[i][2]))

        for i in range(len(self.p0)):
            f.write('%d %d %d\n'% (3*i+1, 3*i+2,3*i+3))

        f.close()

    def addThicknessConstraints2D(self, leList, teList, nSpan, nChord,
                                  lower=1.0, upper=3.0, scaled=True, scale=1.0,
                                  name=None, addToPyOpt=True):
        """
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
          interpolated in a linear fashion. Note that the thickness
          constraint may not correspond **EXACT** to intermediate
          locations in leList and teList. For example, in the example
          above, with leList=3 and nSpan=3, the three thickness
          constraints on the leading edge of the 2D domain would be at
          the left and right boundaries, and at the point denoted by
          'o' which is equidistance between the root and tip.

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
          normal, issues can arise near the end of an opensurface (ie
          root of a single wing) which can result in failing
          intersections.

        Parameters
        ----------
        leList : list or array
            A list or array of points (size should be (Nx3) where N is
            at least 2) defining the 'leading edge' or the start of the
            domain

        teList : list or array
           Same as leList but for the trailing edge.

        nSpan : int
            The number of thickness constraints to be (linear)
            interpolated *along* the leading and trailing edges

        nChord : int
            The number of thickness constraints to be (linearly)
            interpolated between the leading and trailing edges

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

            * scaled=False: No scaling is applied and the phyical lengths
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
            you have multiple DVCon objects and the constriant names
            need to be distinguished **or** the values are to be used
            in a subsequent computation.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) BEFORE they are
            given to the optimizer, set this flag to False.

        Examples
        --------
        >>> # Take unique square in x-z plane and and 10 along z-direction (spanWise)
        >>> # and the along x-direction (chordWise)
        >>> leList = [[0, 0, 0], [0, 0, 1]]
        >>> teList = [[1, 0, 0], [0, 0, 1]]
        >>> DVCon.addThicknessConstraints2D(leList, teList, 10, 3,
                                lower=1.0, scaled=True)
        """

        self._checkDVGeo()
        upper = self._convertTo2D(upper, nSpan, nChord).flatten()
        lower = self._convertTo2D(lower, nSpan, nChord).flatten()
        scale = self._convertTo2D(scale, nSpan, nChord).flatten()

        coords = self._generateIntersections(leList, teList, nSpan, nChord)

        # Create the thickness constraint object:
        coords = coords.reshape((nSpan*nChord*2, 3))

        typeName = 'thickCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        # Create a name
        if name is None:
            conName = '%s_thickness_constraints_%d'%(self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = ThicknessConstraint(
            conName, coords, lower, upper, scaled, scale, self.DVGeo,
            addToPyOpt)


    def addThicknessConstraints1D(self, ptList, nCon, axis,
                                  lower=1.0, upper=3.0, scaled=True,
                                  scale=1.0, name=None,
                                  addToPyOpt=True):
        """
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

            * scaled=False: No scaling is applied and the phyical lengths
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
            you have multiple DVCon objects and the constriant names
            need to be distinguished **or** you are using this set of
            thickness constraints for something other than a direct
            constraint in pyOptSparse.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) BEFORE they are
            given to the optimizer, set this flag to False.
        """
        self._checkDVGeo()
        # Create mesh of itersections
        constr_line = pySpline.Curve(X=ptList, k=2)
        s = numpy.linspace(0, 1, nCon)
        X = constr_line(s)
        coords = numpy.zeros((nCon, 2, 3))
        # Project all the points
        for i in range(nCon):
            # Project actual node:
            up, down, fail = geo_utils.projectNode(
                X[i], axis, self.p0, self.v1, self.v2)
            if fail > 0:
                raise Error("There was an error projecting a node "
                            "at (%f, %f, %f) with normal (%f, %f, %f)."% (
                                X[i, 0], X[i, 1], X[i, 2], axis[0], axis[1], axis[2]))
            coords[i, 0] = up
            coords[i, 1] = down

        # Create the thickness constraint object:
        coords = coords.reshape((nCon*2, 3))

        typeName = 'thickCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = '%s_thickness_constraints_%d'%(self.name,len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = ThicknessConstraint(
            conName, coords, lower, upper, scaled, scale, self.DVGeo,
            addToPyOpt)

    def addLERadiusConstraints(self, leList, nSpan, axis, chordDir,
                               lower=1.0, upper=3.0, scaled=True,
                               scale=1.0, name=None, addToPyOpt=True):
        """
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

            * scaled=False: No scaling is applied and the phyical radii
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
            multiple DVCon objects and the constriant names need to be
            distinguished **or** you are using this set of thickness constraints
            for something other than a direct constraint in pyOptSparse.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) BEFORE they are
            given to the optimizer, set this flag to False.
        """
        self._checkDVGeo()

        # Create mesh of itersections
        constr_line = pySpline.Curve(X=leList, k=2)
        s = numpy.linspace(0, 1, nSpan)
        X = constr_line(s)
        coords = numpy.zeros((nSpan, 3, 3))

        # Project all the points
        for i in range(nSpan):
            # Project actual node:
            up, down, fail = geo_utils.projectNode(
                X[i], axis, self.p0, self.v1, self.v2)
            if fail > 0:
                raise Error("There was an error projecting a node "
                            "at (%f, %f, %f) with normal (%f, %f, %f)."% (
                                X[i, 0], X[i, 1], X[i, 2],
                                axis[0], axis[1], axis[2]))
            coords[i, 0] = up
            coords[i, 1] = down

        # Calculate mid-points
        midPts = (coords[:,0,:] + coords[:,1,:]) / 2.0

        # Project to get leading edge point
        lePts = numpy.zeros((nSpan, 3))
        chordDir = numpy.array(chordDir, dtype='d').flatten()
        chordDir /= numpy.linalg.norm(chordDir)
        for i in range(nSpan):
            # Project actual node:
            up, down, fail = geo_utils.projectNode(
                X[i], chordDir, self.p0, self.v1, self.v2)
            if fail > 0:
                raise Error("There was an error projecting a node "
                            "at (%f, %f, %f) with normal (%f, %f, %f)."% (
                                X[i, 0], X[i, 1], X[i, 2],
                                chordDir[0], chordDir[1], chordDir[2]))
            lePts[i] = up

        # Check that points can form radius
        d = numpy.linalg.norm(coords[:,0,:] - coords[:,1,:], axis=1)
        r = numpy.linalg.norm(midPts - lePts, axis=1)
        for i in range(nSpan):
            if d[i] < 2*r[i]:
                raise Error("Leading edge radius points are too far from the "
                            "leading edge point to form a circle between the "
                            "three points.")

        # Add leading edge points and stack points into shape accepted by DVGeo
        coords[:,2,:] = lePts
        coords = numpy.vstack((coords[:,0,:], coords[:,1,:], coords[:,2,:]))

        # Create the thickness constraint object
        typeName = 'radiusCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = '%s_leradius_constraints_%d'%(self.name,len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = RadiusConstraint(
            conName, coords, lower, upper, scaled, scale, self.DVGeo,
            addToPyOpt)


    def addLocationConstraints1D(self, ptList, nCon,lower=None, upper=None,
                                 scaled=False, scale=1.0, name=None,
                                 addToPyOpt=True):
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

            * scaled=False: No scaling is applied and the phyical locations
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
            you have multiple DVCon objects and the constriant names
            need to be distinguished **or** you are using this set of
            location constraints for something other than a direct
            constraint in pyOptSparse.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) BEFORE they are
            given to the optimizer, set this flag to False.
        """
        self._checkDVGeo()
        # Create the points to constrain
        constr_line = pySpline.Curve(X=ptList, k=2)
        s = numpy.linspace(0, 1, nCon)
        X = constr_line(s)
        # X shouls now be in the shape we need

        if lower is None:
            lower = X.flatten()
        if upper is None:
            upper = X.flatten()

        # Create the location constraint object
        typeName = 'locCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = '%s_location_constraints_%d'%(self.name,len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = LocationConstraint(
            conName, X, lower, upper, scaled, scale, self.DVGeo,
            addToPyOpt)


    def addProjectedLocationConstraints1D(self, ptList, nCon, axis, bias=0.5,
                                          lower=None, upper=None,
                                          scaled=False, scale=1.0, name=None,
                                          addToPyOpt=True):
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

            * scaled=False: No scaling is applied and the phyical locations
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
            you have multiple DVCon objects and the constriant names
            need to be distinguished **or** you are using this set of
            location constraints for something other than a direct
            constraint in pyOptSparse.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) BEFORE they are
            given to the optimizer, set this flag to False.

        """
        self._checkDVGeo()
        # Create the points to constrain
        constr_line = pySpline.Curve(X=ptList, k=2)
        s = numpy.linspace(0, 1, nCon)
        X = constr_line(s)

        coords = numpy.zeros((nCon, 2, 3))
        # Project all the points
        for i in range(nCon):
            # Project actual node:
            up, down, fail = geo_utils.projectNode(
                X[i], axis, self.p0, self.v1, self.v2)
            if fail > 0:
                raise Error("There was an error projecting a node "
                            "at (%f, %f, %f) with normal (%f, %f, %f)."% (
                        X[i, 0], X[i, 1], X[i, 2], axis[0], axis[1], axis[2]))
            coords[i, 0] = up
            coords[i, 1] = down

        X = (1-bias)*coords[:, 1] + bias*coords[:, 0]

        # X is now what we want to constrain
        if lower is None:
            lower = X.flatten()
        if upper is None:
            upper = X.flatten()

        # Create the location constraint object
        typeName = 'locCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = '%s_location_constraints_%d'%(self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = LocationConstraint(
            conName, X, lower, upper, scaled, scale, self.DVGeo,
            addToPyOpt)

    def addThicknessToChordConstraints1D(self, ptList, nCon, axis, chordDir,
                                         lower=1.0, upper=3.0, scale=1.0,
                                         name=None, addToPyOpt=True):
        """
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
            the *actual* t/c is *NEVER* computed. This constraint can
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
            you have multiple DVCon objects and the constriant names
            need to be distinguished **or** you are using this set of
            thickness constraints for something other than a direct
            constraint in pyOptSparse.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) BEFORE they are
            given to the optimizer, set this flag to False.
        """
        self._checkDVGeo()

        constr_line = pySpline.Curve(X=ptList, k=2)
        s = numpy.linspace(0, 1, nCon)
        X = constr_line(s)
        coords = numpy.zeros((nCon, 4, 3))
        chordDir /= numpy.linalg.norm(numpy.array(chordDir, 'd'))
        # Project all the points
        for i in range(nCon):
            # Project actual node:
            up, down, fail = geo_utils.projectNode(
                X[i], axis, self.p0, self.v1, self.v2)
            if fail:
                raise Error("There was an error projecting a node "
                            "at (%f, %f, %f) with normal (%f, %f, %f)." % (
                        X[i, 0], X[i, 1], X[i, 2], axis[0], axis[1], axis[2]))

            coords[i, 0] = up
            coords[i, 1] = down
            height = numpy.linalg.norm(coords[i, 0] - coords[i, 1])
            # Third point is the mid-point of thsoe
            coords[i, 2] = 0.5*(up + down)

            # Fourth point is along the chordDir
            coords[i, 3] = coords[i, 2] + 0.1*height*chordDir

        # Create the thickness constraint object:
        coords = coords.reshape((nCon*4, 3))

        typeName = 'thickCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()
        if name is None:
            conName = '%s_thickness_to_chord_constraints_%d'%(self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = ThicknessToChordConstraint(
            conName, coords, lower, upper, scale, self.DVGeo, addToPyOpt)

    def addVolumeConstraint(self, leList, teList, nSpan, nChord,
                            lower=1.0, upper=3.0, scaled=True, scale=1.0,
                            name=None, addToPyOpt=True):
        """
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
        hexahedreal volumes. The accuracy of the volume computation
        depends on how well these linear hexahedral volumes
        approximate the (assumed) continuous underlying surface.

        See `addThicknessConstraints2D` for additional information.

        Parameters
        ----------
        leList : list or array
           A list or array of points (size should be (Nx3) where N is
           at least 2) defining the 'leading edge' or the start of the
           domain

        teList : list or array
           Same as leList but for the trailing edge.

        nSpan : int
            The number of thickness constraints to be (linear)
            interpolated *along* the leading and trailing edges

        nChord : int
            The number of thickness constraints to be (linearly)
            interpolated between the leading and trailing edges

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
            the resulting phyical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constriant names need to
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
        self._checkDVGeo()

        typeName = 'volCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = '%s_volume_constraint_%d'%(self.name, len(self.constraints[typeName]))
        else:
            conName = name

        coords = self._generateIntersections(leList, teList, nSpan, nChord)
        coords = coords.reshape((nSpan*nChord*2, 3))

        # Finally add the volume constraint object
        self.constraints[typeName][conName] = VolumeConstraint(
            conName, nSpan, nChord, coords, lower, upper, scaled, scale,
            self.DVGeo, addToPyOpt)


    def addCompositeVolumeConstraint(self, vols, lower=1.0, upper=3.0,
                                     scaled=True, scale=1.0, name=None,
                                     addToPyOpt=True):
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
            the resulting phyical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constriant names need to
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
        self._checkDVGeo()

        typeName = 'volCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = '%s_composite_volume_constraint_%d'%(self.name, len(self.constraints[typeName]))
        else:
            conName = name

        # Determine the list of volume constraint objects
        volCons = []
        for vol in vols:
            try:
                volCons.append(self.constraints[typeName][vol])
            except KeyError:
                raise Error("The supplied volume name '%s' has not"
                            " already been added with a call to "
                            "addVolumeConstraint()"% vol)
        self.constraints[typeName][conName] = CompositeVolumeConstraint(
            conName, volCons, lower, upper, scaled, scale, self.DVGeo,
            addToPyOpt)

    def addLeTeConstraints(self, volID=None, faceID=None,topID=None,
                           indSetA=None, indSetB=None, name=None,
                           config=None, childIdx=None):
        """
        Add a set of 'leading edge' or 'trailing edge' constraints to
        DVConstraints. These are just a particular form of linear
        constraints for shape variables. The need for these
        constraints arise when the shape variables can effectively
        emulate a 'twist' variable (actually a shearing twist). The
        purpose of these constraints is to make control points at the
        leading and trailing edge move in equal and opposite diretion.

        ie. x1 - x2 = 0.0

        where x1 is the movment (in 1, 2, or 3 directions) of a
        control point on the top of the FFD and x2 is the control
        poin on teh bottom of the FFD.

        There are two ways of specifying these constraints:

        volID and faceID: Provide the index of the FFD block and the
        faceID (one of 'ilow', 'ihigh', 'jlow', 'jhigh', 'klow', or
        'khigh'). This it the preferred approach. Both volID and faceID
        can be determined by examining the FFD file in TecPlot or ICEM.
        Use 'prob data' tool in TecPlot to click on the surface of which
        you want to put constraints on (e.g. the front or LE of FFD and
        the back surface or TE of the FFD). You will see which plane
        it coresponding to. For example, 'I-Plane' with I-index = 1 is
        'iLow'.
        topID provides a second input for blocks that have 2x2 faces.

        Alternatively, two sets of indices can be provided, 'indSetA'
        and 'indSetB'. Both must be the same length. These indices may
        be obtained from the 'lindex' array of the FFD object.

        lIndex = DVGeo.getLocalIndex(iVol)

        lIndex is a three dimensional set of indices that provide the
        index into the global set of control points. See below for
        examples.

        Note that these constraints *will* be added to pyOptSparse
        automatically with a call to addConstraintsPyOpt()

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
             multiple DVCon objects and the constriant names need to
             be distinguished
        config : str
             The DVGeo configuration to apply this LETE con to. Must be either None
             which will allpy to *ALL* the local DV groups or a single string specifying
             a particular configuration.

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
        self._checkDVGeo()

        if childIdx is not None:
            DVGeo = self.DVGeo.children[childIdx]
        else:
            DVGeo = self.DVGeo

        # Now determine what type of specification we have:
        if volID is not None and faceID is not None:
            lIndex = DVGeo.getLocalIndex(volID)
            iFace = False
            jFace = False
            kFace = False
            if faceID.lower() == 'ilow':
                indices = lIndex[0, :, :]
                iFace = True
            elif faceID.lower() == 'ihigh':
                indices = lIndex[-1, :, :]
                iFace = True
            elif faceID.lower() == 'jlow':
                indices = lIndex[:, 0, :]
                jFace = True
            elif faceID.lower() == 'jhigh':
                indices = lIndex[:, -1, :]
                jFace = True
            elif faceID.lower() == 'klow':
                indices = lIndex[:, :, 0]
                kFace = True
            elif faceID.lower() == 'khigh':
                indices = lIndex[:, :, -1]
                kFace = True
            else:
                raise Error("faceID must be one of iLow, iHigh, jLow, jHigh, "
                            "kLow or kHigh.")

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
                    if topID.lower()=='i' and not iFace:
                        indSetA = indices[0, :]
                        indSetB = indices[1, :]
                    elif topID.lower()=='j' and not jFace:
                        if iFace:
                            indSetA = indices[0, :]
                            indSetB = indices[1, :]
                        else:
                            indSetA = indices[:, 0]
                            indSetB = indices[:, 1]
                    elif topID.lower()=='k' and not kFace:
                        indSetA = indices[:, 0]
                        indSetB = indices[:, 1]
                    else:
                        raise Error("Invalid value for topID. value must be"
                                    " i, j or k")

                else:
                    raise Error("Cannot add leading edge constraints. One (and "
                                "exactly one) of FFD block dimensions on the"
                                " specified face must be 2. The dimensions of "
                                "the selected face are: "
                                "(%d, %d). For this case you must specify "
                                "topID" % (shp[0], shp[1]))

        elif indSetA is not None and indSetB is not None:
            if len(indSetA) != len(indSetB):
                raise Error("The length of the supplied indices are not "
                            "the same length")
        else:
            raise Error("Incorrect data supplied to addLeTeConstraint. The "
                        "keyword arguments 'volID' and 'faceID' must be "
                        "specified **or** 'indSetA' and 'indSetB'")

        if name is None:
            conName = '%s_lete_constraint_%d'%(self.name, len(self.linearCon))
        else:
            conName = name

        # Finally add the volume constraint object
        n = len(indSetA)
        self.linearCon[conName] = LinearConstraint(
            conName, indSetA, indSetB, numpy.ones(n), numpy.ones(n),
            lower=0, upper=0, DVGeo=DVGeo, config=config)

    def addLinearConstraintsShape(self, indSetA, indSetB, factorA, factorB,
                                  lower=0, upper=0, name=None, config=None):
        """
        Add a complete generic set of linear constraints for the shape
        variables that have been added to DVGeo. The constraints are
        specified in the following general form:

        lower <= factorA*dvA + factorB*dvB <= upper

        The lists indSetA and indSetB are used to specify the pairs of
        control points that are to be linked with linear variables. If
        more than one pair is specified (ie len(indSetA)=len(indSetB)
        > 1) then factorA, factorB, lower and upper may all be arrays
        of the same length or a constant which will applied to all.

        Two sets of indices can be provided, 'indSetA'
        and 'indSetB'. Both must be the same length. These indices may
        be obtained from the 'lindex' array of the FFD object.

        lIndex = DVGeo.getLocalIndex(iVol)

        lIndex is a three dimensional set of indices that provide the
        index into the global set of control points. See below for
        examples.

        Note that these constraints *will* be added to pyOptSparse
        automatically with a call to addConstraintsPyOpt()

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
             multiple DVCon objects and the constriant names need to
             be distinguished

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

        if self.DVGeo is None:
            raise Error("A DVGeometry object must be added to DVCon before "
                        "using a call to DVCon.setDVGeo(DVGeo) before "
                        "constraints can be added.")

        if len(indSetA) != len(indSetB):
            raise Error("The length of the supplied indices are not "
                        "the same length")

        if name is None:
            conName = '%s_linear_constraint_%d'%(self.name, len(self.linearCon))
        else:
            conName = name

        # Process the inputs to be arrays of length n if necessary.
        factorA = numpy.atleast_1d(factorA)
        factorB = numpy.atleast_1d(factorB)
        lower = numpy.atleast_1d(lower)
        upper = numpy.atleast_1d(upper)
        n = len(indSetA)

        if len(factorA) == 1:
            factorA = factorA[0]*numpy.ones(n)
        elif len(factorA) != n:
            raise Error('Length of factorA invalid!')

        if len(factorB) == 1:
            factorB = factorB[0]*numpy.ones(n)
        elif len(factorB) != n:
            raise Error('Length of factorB invalid!')

        if len(lower) == 1:
            lower = lower[0]*numpy.ones(n)
        elif len(lower) != n:
            raise Error('Length of lower invalid!')

        if len(upper) == 1:
            upper = upper[0]*numpy.ones(n)
        elif len(upper) != n:
            raise Error('Length of upper invalid!')

        # Finally add the linear constraint object
        self.linearCon[conName] = LinearConstraint(
            conName, indSetA, indSetB, factorA, factorB, lower, upper,
            self.DVGeo,config=config)

    def addGearPostConstraint(self, wimpressCalc, position, axis,
                              thickLower=1.0, thickUpper=3.0,
                              thickScaled=True,
                              MACFracLower=0.50, MACFracUpper=0.60,
                              name=None, addToPyOpt=True):

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
            Direction to perofrm projection. Same as 'axis'
            in addThicknessConstraints1D

        thickLower : float
            Lower bound for thickness constraint. If thickScaled=True,
            this is the pysical distance scaled by the initial length.
            This value is used as the optimization constraint lower bound.

        thickUpper : float
            Upper bound for optimization constraint. See thickLower.

        thickScaled : bool
            Flag specifiying if the constraint should be scaled.
            It is true by default. The defalut values of thickScaled=True,
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
            you have multiple DVCon objects and the constriant names
            need to be distinguished **or** the values are to be used
            in a subsequent computation.

        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) BEFORE they are
            given to the optimizer, set this flag to False.

        """

        self._checkDVGeo()

        typeName = 'gearCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = '%s_gear_constraint_%d'%(self.name, len(self.constraints[typeName]))
        else:
            conName = name

        # Project the actual location we were give:
        up, down, fail = geo_utils.projectNode(
            position, axis, self.p0, self.v1, self.v2)
        if fail > 0:
            raise Error("There was an error projecting a node "
                        "at (%f, %f, %f) with normal (%f, %f, %f)."% (
                            position))

        self.constraints[typeName][conName] = GearPostConstraint(
            conName, wimpressCalc, up, down, thickLower, thickUpper,
            thickScaled, MACFracLower, MACFracUpper, self.DVGeo,
            addToPyOpt)


    def addCircularityConstraint(self,origin,rotAxis,radius,
                                 zeroAxis,angleCW,angleCCW,
                                 nPts=4,
                                 upper=1.0,lower=1.0, scale=1.0,
                                 name=None, addToPyOpt=True):
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
            the resulting phyical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constriant names need to
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

        self._checkDVGeo()
        coords = self._generateCircle(origin,rotAxis,radius,zeroAxis,angleCW,angleCCW,
                                      nPts)

        # Create the circularity constraint object:
        coords = coords.reshape((nPts, 3))
        origin = numpy.array(origin).reshape((1, 3))

        typeName = 'circCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()


        # Create a name
        if name is None:
            conName = '%s_circularity_constraints_%d'%(self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = CircularityConstraint(
            conName, origin, coords, lower, upper, scale, self.DVGeo,
            addToPyOpt)

    def addSurfaceAreaConstraint(self,lower=1.0, upper=3.0, scaled=True, scale=1.0,
                                 name=None, addToPyOpt=True):
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
            the resulting phyical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constriant names need to
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

        if self.p0 is None or self.v1 is None or self.v2 is None:
            raise Error("DVCon surface is not properly defined. Check that setSurface"
                        "is called.")

        self._checkDVGeo()

        typeName = 'surfAreaCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        # Create a name
        if name is None:
            conName = '%s_surfaceArea_constraints_%d'%(self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = SurfaceAreaConstraint(
            conName, self.p0, self.v1, self.v2, lower, upper, scale, scaled, self.DVGeo,
            addToPyOpt)

    def addProjectedAreaConstraint(self, axis='y', lower=1.0, upper=3.0, scaled=True,
                                 scale=1.0, name=None, addToPyOpt=True):
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
            the resulting phyical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constriant names need to
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

        if self.p0 is None or self.v1 is None or self.v2 is None:
            raise Error("DVCon surface is not properly defined. Check that setSurface"
                        "is called.")

        self._checkDVGeo()

        if axis=='x':
            axis = numpy.array([1,0,0])
        elif axis=='y':
            axis = numpy.array([0,1,0])
        elif axis=='z':
            axis = numpy.array([0,0,1])

        typeName = 'projAreaCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        # Create a name
        if name is None:
            conName = '%s_projectedArea_constraints_%d'%(self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = ProjectedAreaConstraint(
            conName, self.p0, self.v1, self.v2, axis, lower, upper, scale, scaled,
            self.DVGeo, addToPyOpt)

    def addPlanarityConstraint(self,origin,planeAxis,
                               upper=0.0,lower=0.0, scale=1.0,
                               name=None, addToPyOpt=True):
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
            the resulting phyical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constriant names need to
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

        self._checkDVGeo()

        # Create the circularity constraint object:
        origin = numpy.array(origin).reshape((1, 3))
        planeAxis = numpy.array(planeAxis).reshape((1, 3))

        # Create a name
        typeName = 'planeCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = '%s_planarity_constraints_%d'%(self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = PlanarityConstraint(
            conName, planeAxis, origin, self.p0, self.v1, self.v2, lower, upper, scale, self.DVGeo,
            addToPyOpt)

    def addColinearityConstraint(self, origin, lineAxis, distances,
                                 upper=0.0,lower=0.0, scale=1.0,
                                 name=None, addToPyOpt=True):
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
            the resulting phyical volume magnitude is vastly different
            from O(1).

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constriant names need to
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

        self._checkDVGeo()
        nPts = len(distances)
        coords = []
        for dist in distances:
            coords.append(dist*lineAxis+origin)

        # Create the circularity constraint object:
        coords = numpy.array(coords).reshape((nPts, 3))
        origin = numpy.array(origin).reshape((1, 3))
        lineAxis = numpy.array(lineAxis).reshape((1, 3))

        # Create a name
        typeName = 'coLinCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()

        if name is None:
            conName = '%s_colinearity_constraints_%d'%(self.name, len(self.constraints[typeName]))
        else:
            conName = name
        self.constraints[typeName][conName] = ColinearityConstraint(
            conName, lineAxis, origin, coords, lower, upper, scale,
            self.DVGeo, addToPyOpt)

    def addCurvatureConstraint(self, surfFile, curvatureType='Gaussian', lower=-1e20, upper=1e20,
                               scaled=True, scale=1.0, KSCoeff=None,name=None,addToPyOpt=False):
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
            the resulting phyical volume magnitude is vastly different
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
            The coefficient for KS function when curvatyreType=KSmean.
            This controls how close the KS function approximates the original
            functions. One should select a KSCoeff such that the printed "Reference curvature"
            is only slightly larger than the printed "Max curvature" for the baseline surface.
            The default value of KSCoeff is the number of points in the plot3D files.

        name : str
             Normally this does not need to be set; a default name will
             be generated automatically. Only use this if you have
             multiple DVCon objects and the constriant names need to
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

        """


        self._checkDVGeo()

        # Use pyGeo to load the plot3d file
        geo = pyGeo('plot3d', surfFile)
        # node and edge tolerance for pyGeo (these are never used so
        # we just fix them)
        node_tol =  1e-8
        edge_tol =  1e-8
        # Explicity do the connectivity here since we don't want to
        # write a con file:
        geo._calcConnectivity(node_tol, edge_tol)
        surfs = geo.surfs
        typeName = 'curveCon'
        if not typeName in self.constraints:
            self.constraints[typeName] = OrderedDict()
        # Create a name
        if name is None:
            if curvatureType == 'Gaussian':
                conName = '%s_gaussian_curvature_constraint_%d'%(self.name, len(self.constraints[typeName]))
            elif curvatureType == 'mean':
                conName = '%s_mean_curvature_constraint_%d'%(self.name, len(self.constraints[typeName]))
            elif curvatureType == 'combined':
                conName = '%s_combined_curvature_constraint_%d'%(self.name, len(self.constraints[typeName]))
            elif curvatureType == 'KSmean':
                conName = '%s_ksmean_curvature_constraint_%d'%(self.name, len(self.constraints[typeName]))
            else:
                raise Error("The curvatureType parameter should be Gaussian, mean, combined, or KSmean "
                            "%s is not supported!"%curvatureType)
        else:
            conName = name
        self.constraints[typeName][conName] = CurvatureConstraint(
            conName, surfs, curvatureType, lower, upper, scaled, scale, KSCoeff, self.DVGeo,addToPyOpt)

    def addMonotonicConstraints(self, key, slope=1.0, name=None, start=0,
                                stop=-1, config=None):
        """
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
            multiple DVCon objects and the constriant names need to
            be distinguished
        start/stop: int
            This allows the user to specify a slice of the design variable to
            constrain if it is not desired to set a monotonic constraint on the
            entire vector. The start/stop indices are inclusive indices, so for
            a design variable vector [4, 3, 6.5, 2, -5.4, -1], start=1 and
            stop=4 would constrain [3, 6.5, 2, -5.4] to be a monotonic sequence.
        config : str
            The DVGeo configuration to apply this LETE con to. Must be either None
            which will allpy to *ALL* the local DV groups or a single string specifying
            a particular configuration.

        Examples
        --------
        >>> DVCon.addMonotonicConstraints('chords', 1.0)
        """
        self._checkDVGeo()

        if name is None:
            conName = '%s_monotonic_constraint_%d'%(self.name, len(self.linearCon))
        else:
            conName = name

        options = {
            'slope':slope,
            'start':start,
            'stop':stop,
        }
        # Finally add the global linear constraint object
        self.linearCon[conName] = GlobalLinearConstraint(
            conName, key, type='monotonic', options=options,
            lower=0, upper=None, DVGeo=self.DVGeo, config=config)

    def _readPlot3DSurfFile(self, fileName):
        """Read a plot3d file and return the points and connectivity in
        an unstructured mesh format"""

        pts = None
        conn = None

        f = open(fileName, 'r')
        nSurf = numpy.fromfile(f, 'int', count=1, sep=' ')[0]
        sizes = numpy.fromfile(f, 'int', count=3*nSurf, sep=' ').reshape((nSurf, 3))
        nElem = 0
        for i in range(nSurf):
            nElem += (sizes[i, 0]-1)*(sizes[i, 1]-1)

        # Generate the uncompacted point and connectivity list:
        p0 = numpy.zeros((nElem*2, 3))
        v1 = numpy.zeros((nElem*2, 3))
        v2 = numpy.zeros((nElem*2, 3))

        elemCount = 0

        for iSurf in range(nSurf):
            curSize = sizes[iSurf, 0]*sizes[iSurf, 1]
            pts = numpy.zeros((curSize, 3))
            for idim in range(3):
                pts[:, idim] = numpy.fromfile(f, 'float', curSize, sep=' ')

            pts = pts.reshape((sizes[iSurf,0], sizes[iSurf,1], 3), order='f')
            for j in range(sizes[iSurf, 1]-1):
                for i in range(sizes[iSurf, 0]-1):
                    # Each quad is split into two triangles
                    p0[elemCount] = pts[i, j]
                    v1[elemCount] = pts[i+1, j] - pts[i, j]
                    v2[elemCount] = pts[i, j+1] - pts[i, j]

                    elemCount += 1

                    p0[elemCount] = pts[i+1, j]
                    v1[elemCount] = pts[i+1, j+1] - pts[i+1, j]
                    v2[elemCount] = pts[i, j+1] - pts[i+1, j]

                    elemCount += 1

        return p0, v1, v2


    def _checkDVGeo(self):

        """check if DVGeo exists"""
        if self.DVGeo is None:
            raise Error("A DVGeometry object must be added to DVCon before "
                        "using a call to DVCon.setDVGeo(DVGeo) before "
                        "constraints can be added.")

    def _convertTo2D(self, value, dim1, dim2):
        """
        Generic function to process 'value'. In the end, it must be dim1
        by dim2. value is already that shape, excellent, otherwise, a
        scalar will be 'upcast' to that size
        """

        if numpy.isscalar:
            return value*numpy.ones((dim1, dim2))
        else:
            temp = numpy.atleast_2d(value)
            if temp.shape[0] == dim1 and temp.shape[1] == dim2:
                return value
            else:
                raise Error('The size of the 2D array was the incorret shape')

    def _convertTo1D(self, value, dim1):
        """
        Generic function to process 'value'. In the end, it must be
        array of size dim1. value is already that shape, excellent,
        otherwise, a scalar will be 'upcast' to that size
        """

        if numpy.isscalar:
            return value*numpy.ones(dim1)
        else:
            temp = numpy.atleast_1d(value)
            if temp.shape[0] == dim1:
                return value
            else:
                raise Error('The size of the 1D array was the incorret shape')

    def _generateIntersections(self, leList, teList, nSpan, nChord):
        """
        Internal function to generate the grid points (nSpan x nChord)
        and to actual perform the intersections. This is in a separate
        functions since addThicknessConstraints2D, and volume based
        constraints use the same conde. The list of projected
        coordinates are returned.
        """

        # Create mesh of itersections
        le_s = pySpline.Curve(X=leList, k=2)
        te_s = pySpline.Curve(X=teList, k=2)
        root_s = pySpline.Curve(X=[leList[0], teList[0]], k=2)
        tip_s  = pySpline.Curve(X=[leList[-1], teList[-1]], k=2)

        # Generate parametric distances
        span_s = numpy.linspace(0.0, 1.0, nSpan)
        chord_s = numpy.linspace(0.0, 1.0, nChord)

        # Generate a 2D region of intersections
        X = geo_utils.tfi_2d(le_s(span_s), te_s(span_s),
                             root_s(chord_s), tip_s(chord_s))
        coords = numpy.zeros((nSpan, nChord, 2, 3))
        for i in range(nSpan):
            for j in range(nChord):
                # Generate the 'up_vec' from taking the cross product
                # across a quad
                if i == 0:
                    uVec = X[i+1, j]-X[i, j]
                elif i == nSpan - 1:
                    uVec = X[i, j] - X[i-1, j]
                else:
                    uVec = X[i+1, j] - X[i-1, j]

                if j == 0:
                    vVec = X[i, j+1]-X[i, j]
                elif j == nChord - 1:
                    vVec = X[i, j] - X[i, j-1]
                else:
                    vVec = X[i, j+1] - X[i, j-1]

                upVec = numpy.cross(uVec, vVec)
                # Project actual node:
                up, down, fail = geo_utils.projectNode(
                    X[i ,j], upVec, self.p0, self.v1, self.v2)

                if fail == 0:
                    coords[i, j, 0] = up
                    coords[i, j, 1] = down
                elif fail == -1:
                    # More than 2 solutoins. Returned in sorted distance.
                    coords[i, j, 0] = down
                    coords[i, j, 1] = up
                else:
                    raise Error('There was an error projecting a node \
                     at (%f, %f, %f) with normal (%f, %f, %f).'% (
                            X[i, j, 0], X[i, j, 1], X[i, j, 2],
                            upVec[0], upVec[1], upVec[2]))

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

            for i in range(len(u)-1):
                for j in range(len(v)-1):
                    P0 = surf(u[i  ], v[j  ])
                    P1 = surf(u[i+1], v[j  ])
                    P2 = surf(u[i  ], v[j+1])
                    P3 = surf(u[i+1], v[j+1])

                    p0.append(P0)
                    v1.append(P1-P0)
                    v2.append(P2-P0)

                    p0.append(P3)
                    v1.append(P2-P3)
                    v2.append(P1-P3)

        self.p0 = numpy.array(p0)
        self.v1 = numpy.array(v1)
        self.v2 = numpy.array(v2)

    def _generateCircle(self,origin,rotAxis,radius,zeroAxis,angleCW,angleCCW,nPts):
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
        origin = numpy.array(origin).reshape((3,))

        # Create the coordinate array
        coords = numpy.zeros((nPts,3))

        # get the angles about the zero axis for the points
        if angleCW<0:
            raise Error("Negative angle specified. angleCW should be positive.")
        if angleCCW<0:
            raise Error("Negative angle specified. angleCCW should be positive.")

        angles = numpy.linspace(numpy.deg2rad(-angleCW),numpy.deg2rad(angleCCW),nPts)

        # ---------
        # Generate a unit vector in the zero axis direction
        # ----
        # get the third axis by taking the cross product of rotAxis and zeroAxis
        axis = numpy.cross(zeroAxis,rotAxis)

        #now use these axis to regenerate the orthogonal zero axis
        zeroAxisOrtho = numpy.cross(rotAxis,axis)

        # now normalize the length of the zeroAxisOrtho
        length = numpy.linalg.norm(zeroAxisOrtho)
        zeroAxisOrtho /=length

        # -------
        # Normalize the rotation axis
        # -------
        length = numpy.linalg.norm(rotAxis)
        rotAxis /=length

        # ---------
        # now rotate, multiply by radius ,and add to the origin to get the coords
        # ----------

        for i in range(nPts):
            newUnitVec = geo_utils.rotVbyW(zeroAxisOrtho, rotAxis, angles[i])
            newUnitVec*=radius
            coords[i,:] = newUnitVec+origin

        return coords

class ThicknessConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of thickness
    constraints. One of these objects is created each time a
    addThicknessConstraints2D or addThicknessConstraints1D call is
    made. The user should not have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo,
                 addToPyOpt):
        self.name = name
        self.coords = coords
        self.nCon = len(self.coords)//2
        self.lower = lower
        self.upper = upper
        self.scaled = scaled
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name)

        # Now get the reference lengths
        self.D0 = numpy.zeros(self.nCon)
        for i in range(self.nCon):
            self.D0[i] = numpy.linalg.norm(
                self.coords[2*i] - self.coords[2*i+1])

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        D = numpy.zeros(self.nCon)
        for i in range(self.nCon):
            D[i] = numpy.linalg.norm(self.coords[2*i] - self.coords[2*i+1])
            if self.scaled:
                D[i] /= self.D0[i]
        funcs[self.name] = D

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dTdPt = numpy.zeros((self.nCon,
                                 self.coords.shape[0],
                                 self.coords.shape[1]))

            for i in range(self.nCon):
                p1b, p2b = geo_utils.eDist_b(
                    self.coords[2*i, :], self.coords[2*i+1, :])
                if self.scaled:
                    p1b /= self.D0[i]
                    p2b /= self.D0[i]
                dTdPt[i, 2*i  , :] = p1b
                dTdPt[i, 2*i+1, :] = p2b

            funcsSens[self.name] = self.DVGeo.totalSensitivity(
                dTdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write('Zone T=%s\n'% self.name)
        handle.write('Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n'% (
            len(self.coords), len(self.coords)//2))
        handle.write('DATAPACKING=POINT\n')
        for i in range(len(self.coords)):
            handle.write('%f %f %f\n'% (self.coords[i, 0], self.coords[i, 1],
                                        self.coords[i, 2]))

        for i in range(len(self.coords)//2):
            handle.write('%d %d\n'% (2*i+1, 2*i+2))

class RadiusConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of radius of curvature
    constraints. One of these objects is created each time a
    addLERadiusConstraints call is made. The user should not have
    to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo,
                 addToPyOpt):
        self.name = name
        self.coords = coords
        self.nCon = len(self.coords)//3
        self.lower = lower
        self.upper = upper
        self.scaled = scaled
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name)

        # Now get the reference lengths
        self.r0, self.c0 = self.computeCircle(self.coords)

    def splitPointSets(self, coords):
        p1 = coords[:self.nCon]
        p2 = coords[self.nCon:self.nCon*2]
        p3 = coords[self.nCon*2:]
        return p1, p2, p3

    def computeReferenceFrames(self, coords):
        p1, p2, p3 = self.splitPointSets(coords)

        # Compute origin and unit vectors (xi, eta) of 2d space
        origin = (p1 + p2) / 2.0
        nxi = p1 - origin
        neta = p3 - origin
        for i in range(self.nCon):
            nxi[i] /= geo_utils.euclideanNorm(nxi[i])
            neta[i] /= geo_utils.euclideanNorm(neta[i])

        # Compute component of eta in the xi direction
        eta_on_xi = numpy.einsum('ij,ij->i', nxi, neta)
        xi_of_eta = numpy.einsum('ij,i->ij', nxi, eta_on_xi)

        # Remove component of eta in the xi direction
        neta = neta - xi_of_eta
        for i in range(self.nCon):
            neta[i] /= geo_utils.euclideanNorm(neta[i])

        return origin, nxi, neta

    def computeCircle(self, coords):
        '''
        A circle in a 2D coordinate system is defined by the equation:

            A*xi**2 + A*eta**2 + B*xi + C*eta + D = 0

        First, we get the coordinates of our three points in 2D reference space.
        Then, we can get the coefficients A, B, C, and D by solving for some
        determinants. Then the radius and center of the circle can be calculated
        from:

            x = -B / 2 / A
            y = -C / 2 / A
            r = sqrt((B**2 + C**2 - 4*A*D) / 4 / A**2)

        Finally, we convert the reference coordinates of the center back into
        3D space.
        '''
        p1, p2, p3 = self.splitPointSets(coords)

        # Compute origin and unit vectors (xi, eta) of 2d space
        origin, nxi, neta = self.computeReferenceFrames(coords)

        # Compute xi component of p1, p2, and p3
        xi1 = numpy.einsum('ij,ij->i', p1 - origin, nxi)
        xi2 = numpy.einsum('ij,ij->i', p2 - origin, nxi)
        xi3 = numpy.einsum('ij,ij->i', p3 - origin, nxi)

        # Compute eta component of p1, p2, and p3
        eta1 = numpy.einsum('ij,ij->i', p1 - origin, neta)
        eta2 = numpy.einsum('ij,ij->i', p2 - origin, neta)
        eta3 = numpy.einsum('ij,ij->i', p3 - origin, neta)

        # Compute the radius of curvature
        A = xi1*(eta2 - eta3) - eta1*(xi2 - xi3) + xi2*eta3 - xi3*eta2
        B = (xi1**2 + eta1**2)*(eta3 - eta2) + (xi2**2 + eta2**2)*(eta1 - eta3) \
            + (xi3**2 + eta3**2)*(eta2 - eta1)
        C = (xi1**2 + eta1**2)*(xi2 - xi3) + (xi2**2 + eta2**2)*(xi3 - xi1) \
            +  (xi3**2 + eta3**2)*(xi1 - xi2)
        D = (xi1**2 + eta1**2)*(xi3*eta2 - xi2*eta3) \
            + (xi2**2 + eta2**2)*(xi1*eta3 - xi3*eta1) \
            + (xi3**2 + eta3**2)*(xi2*eta1 - xi1*eta2)

        xiC = -B / 2 / A
        etaC = -C / 2 / A
        r = numpy.sqrt((B**2 + C**2 - 4*A*D) / 4 / A**2)

        # Convert center coordinates back
        center = origin + nxi*xiC[:,None] + neta*etaC[:,None]

        return r, center

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        r, c = self.computeCircle(self.coords)
        if self.scaled:
            r /= self.r0
        funcs[self.name] = r

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            # This is the sensitivity of the radius of curvature w.r.t. the
            # coordinates of each of the three points that make it up
            # row 0: dr0dp0x dr0dp0y dr0dp0z dr0dp1x dr0dp1y dr0dp1z dr0dp2x ...
            # row 1: dr1dp0x dr1dp0y dr1dp0z dr1dp1x dr1dp1y dr1dp1z dr1dp2x ...
            # :
            drdPt = numpy.zeros((self.nCon, 9))

            coords = self.coords.astype('D')
            for i in range(3): # loop over pts at given slice
                for j in range(3): # loop over coordinates in pt
                    coords[i*self.nCon:(i+1)*self.nCon,j] += 1e-40j
                    r, c = self.computeCircle(coords)

                    drdPt[:,i*3+j] = r.imag / 1e-40
                    coords[i*self.nCon:(i+1)*self.nCon,j] -= 1e-40j

            # We now need to convert to the 3d sparse matrix form of the jacobian.
            # We need the derivative of each radius w.r.t. all of the points
            # in coords w.r.t. all of the coordinates for a given point.
            # So the final matrix dimensions are (ncon, ncon*3, 3)

            # We also have to scale the sensitivities if scale is True.
            if self.scaled:
                eye = numpy.diag(1/self.r0)
            else:
                eye = numpy.eye(self.nCon)
            drdPt_sparse = numpy.einsum('ij,jk->ijk', eye, drdPt)
            drdPt_sparse = drdPt_sparse.reshape(self.nCon, self.nCon*3, 3)
            drdPt_sparse = numpy.hstack([
                drdPt_sparse[:,::3,:],
                drdPt_sparse[:,1::3,:],
                drdPt_sparse[:,2::3,:],
            ])

            funcsSens[self.name] = self.DVGeo.totalSensitivity(
                drdPt_sparse, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """
        r, c = self.computeCircle(self.coords)

        # Compute origin and unit vectors (xi, eta) of 2d space
        origin, nxi, neta = self.computeReferenceFrames(self.coords)

        nres = 50
        theta = numpy.linspace(0, 2*numpy.pi, nres+1)[:-1]
        handle.write('Zone T=%s\n'% self.name)
        handle.write('Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n'% (
            self.nCon*nres, self.nCon*nres))
        handle.write('DATAPACKING=POINT\n')
        for i in range(self.nCon):
            cos_part = numpy.outer(numpy.cos(theta), nxi*r[i])
            sin_part = numpy.outer(numpy.sin(theta), neta*r[i])
            x = c[i,0] + cos_part[:,0] + sin_part[:,0]
            y = c[i,1] + cos_part[:,1] + sin_part[:,1]
            z = c[i,2] + cos_part[:,2] + sin_part[:,2]

            for j in range(nres):
                handle.write('%f %f %f\n'% (x[j], y[j], z[j]))

        for i in range(self.nCon):
            for j in range(nres):
                handle.write('%d %d\n'% (i*nres + j + 1, i*nres + (j+1)%nres + 1))

class LocationConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of location
    constraints. One of these objects is created each time a
    addLocationConstraints1D call is
    made. The user should not have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo,
                 addToPyOpt):
        self.name = name
        self.coords = coords
        self.nCon = len(self.coords.flatten())
        self.lower = lower
        self.upper = upper
        self.scaled = scaled
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name)

        # Now get the reference lengths
        self.X0 = numpy.zeros(self.nCon)
        X = self.coords.flatten()
        for i in range(self.nCon):
            self.X0[i] = X[i]

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        X = self.coords.flatten()
        if self.scaled:
            for i in range(self.nCon):
                X[i] /= self.X0[i]

        funcs[self.name] = X

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dTdPt = numpy.zeros((self.nCon,
                                 self.coords.shape[0],
                                 self.coords.shape[1]))
            counter = 0
            for i in range( self.coords.shape[0]):
                for j in range( self.coords.shape[1]):
                    dTdPt[counter][i][j] = 1.0
                    if self.scaled:
                        dTdPt[counter][i][j] /= self.X0[i]
                    counter+=1

            funcsSens[self.name] = self.DVGeo.totalSensitivity(
                dTdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write('Zone T=%s\n'% self.name)
        handle.write('Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n'% (
            len(self.coords), len(self.coords)-1))
        handle.write('DATAPACKING=POINT\n')
        for i in range(len(self.coords)):
            handle.write('%f %f %f\n'% (self.coords[i, 0], self.coords[i, 1],
                                        self.coords[i, 2]))

        for i in range(len(self.coords)-1):
            handle.write('%d %d\n'% (i+1, i+2))


class ThicknessToChordConstraint(GeometricConstraint):
    """
    ThicknessToChordConstraint represents of a set of
    thickess-to-chord ratio constraints. One of these objects is
    created each time a addThicknessToChordConstraints2D or
    addThicknessToChordConstraints1D call is made. The user should not
    have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scale, DVGeo, addToPyOpt):
        self.name = name
        self.coords = coords
        self.nCon = len(self.coords)//4
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name)

        # Now get the reference lengths
        self.ToC0 = numpy.zeros(self.nCon)
        for i in range(self.nCon):
            t = numpy.linalg.norm(self.coords[4*i] - self.coords[4*i+1])
            c = numpy.linalg.norm(self.coords[4*i+2] - self.coords[4*i+3])
            self.ToC0[i] = t/c

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        ToC = numpy.zeros(self.nCon)
        for i in range(self.nCon):
            t = geo_utils.eDist(self.coords[4*i], self.coords[4*i+1])
            c = geo_utils.eDist(self.coords[4*i+2], self.coords[4*i+3])
            ToC[i] = (t/c)/self.ToC0[i]

        funcs[self.name] = ToC

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dToCdPt = numpy.zeros((self.nCon,
                                   self.coords.shape[0],
                                   self.coords.shape[1]))

            for i in range(self.nCon):
                t = geo_utils.eDist(self.coords[4*i], self.coords[4*i+1])
                c = geo_utils.eDist(self.coords[4*i+2], self.coords[4*i+3])

                p1b, p2b = geo_utils.eDist_b(
                    self.coords[4*i, :], self.coords[4*i+1, :])
                p3b, p4b = geo_utils.eDist_b(
                    self.coords[4*i+2, :], self.coords[4*i+3, :])

                dToCdPt[i, 4*i  , :] = p1b/c / self.ToC0[i]
                dToCdPt[i, 4*i+1, :] = p2b/c / self.ToC0[i]
                dToCdPt[i, 4*i+2, :] = (-p3b*t/c**2) / self.ToC0[i]
                dToCdPt[i, 4*i+3, :] = (-p4b*t/c**2) / self.ToC0[i]

            funcsSens[self.name] = self.DVGeo.totalSensitivity(
                dToCdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write('Zone T=%s\n'% self.name)
        handle.write('Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n'% (
            len(self.coords), len(self.coords)//2))
        handle.write('DATAPACKING=POINT\n')
        for i in range(len(self.coords)):
            handle.write('%f %f %f\n'% (self.coords[i, 0], self.coords[i, 1],
                                        self.coords[i, 2]))

        for i in range(len(self.coords)//2):
            handle.write('%d %d\n'% (2*i+1, 2*i+2))

class VolumeConstraint(GeometricConstraint):
    """
    This class is used to represet a single volume constraint. The
    parameter list is explained in the addVolumeConstaint() of
    the DVConstraints class
    """

    def __init__(self, name, nSpan, nChord, coords, lower, upper, scaled,
                 scale, DVGeo, addToPyOpt):

        self.name = name
        self.nCon = 1
        self.nSpan = nSpan
        self.nChord = nChord
        self.coords = coords
        self.lower = lower
        self.upper = upper
        self.scaled = scaled
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt
        self.flipVolume = False

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name)

        # Now get the reference volume
        self.V0 = self.evalVolume()

    def evalFunctions(self, funcs, config):
        """
        Evaluate the function this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        V = self.evalVolume()
        if self.scaled:
            V /= self.V0
        funcs[self.name] = V

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dVdPt = self.evalVolumeSens()
            if self.scaled:
                dVdPt /= self.V0

            # Now compute the DVGeo total sensitivity:
            funcsSens[self.name] = self.DVGeo.totalSensitivity(
                dVdPt, self.name, config=config)


    def writeTecplot(self, handle):
        """
        Write the visualization of this volume constriant
        """
        # Reshape coordinates back to 3D format
        x = self.coords.reshape([self.nSpan, self.nChord, 2, 3])

        handle.write("ZONE T=\"%s\" I=%d J=%d K=%d\n"%(
            self.name, self.nSpan, self.nChord, 2))
        handle.write("DATAPACKING=POINT\n")
        for k in range(2):
            for j in range(self.nChord):
                for i in range(self.nSpan):
                    handle.write('%f %f %f\n'%(x[i, j, k, 0],
                                               x[i, j, k, 1],
                                               x[i, j, k, 2]))

    def evalVolume(self):
        """
        Evaluate the total volume of the current coordinates
        """
        Volume = 0.0
        x = self.coords.reshape((self.nSpan, self.nChord, 2, 3))
        for j in range(self.nChord-1):
            for i in range(self.nSpan-1):
                Volume += self.evalVolumeHex(
                    x[i, j, 0], x[i+1, j, 0], x[i, j+1, 0], x[i+1, j+1, 0],
                    x[i, j, 1], x[i+1, j, 1], x[i, j+1, 1], x[i+1, j+1, 1])

        if Volume < 0:
            Volume = -Volume
            self.flipVolume = True

        return Volume

    def evalVolumeSens(self):
        """
        Evaluate the derivative of the volume with respect to the
        coordinates
        """
        x = self.coords.reshape((self.nSpan, self.nChord, 2, 3))
        xb = numpy.zeros_like(x)
        for j in range(self.nChord-1):
            for i in range(self.nSpan-1):
                self.evalVolumeHex_b(
                    x[i, j, 0], x[i+1, j, 0], x[i, j+1, 0], x[i+1, j+1, 0],
                    x[i, j, 1], x[i+1, j, 1], x[i, j+1, 1], x[i+1, j+1, 1],
                    xb[i, j, 0], xb[i+1, j, 0], xb[i, j+1, 0], xb[i+1, j+1, 0],
                    xb[i, j, 1], xb[i+1, j, 1], xb[i, j+1, 1], xb[i+1, j+1, 1])
        # We haven't divided by 6.0 yet...lets do it here....
        xb /= 6.0

        if self.flipVolume:
            xb = -xb

        # Reshape back to flattened array for DVGeo
        xb = xb.reshape((self.nSpan*self.nChord*2, 3))

        return xb

    def evalVolumeHex(self, x0, x1, x2, x3, x4, x5, x6, x7):
        """
        Evaluate the volume of the hexahedreal volume defined by the
        the 8 corners.

        Parameters
        ----------
        x{0:7} : arrays or size (3)
            Array of defining the coordinates of the volume
        """

        p = numpy.average([x0, x1, x2, x3, x4, x5, x6, x7], axis=0)
        V = 0.0
        V += self.volpym(x0, x1, x3, x2, p)
        V += self.volpym(x0, x2, x6, x4, p)
        V += self.volpym(x0, x4, x5, x1, p)
        V += self.volpym(x1, x5, x7, x3, p)
        V += self.volpym(x2, x3, x7, x6, p)
        V += self.volpym(x4, x6, x7, x5, p)
        V /= 6.0

        return V

    def volpym(self, a, b, c, d, p):
        """
        Compute volume of a square-based pyramid
        """
        fourth = 1.0/4.0

        volpym = (p[0] - fourth*(a[0] + b[0]  + c[0] + d[0]))  \
            * ((a[1] - c[1])*(b[2] - d[2]) - (a[2] - c[2])*(b[1] - d[1]))   + \
            (p[1] - fourth*(a[1] + b[1]  + c[1] + d[1]))                \
            * ((a[2] - c[2])*(b[0] - d[0]) - (a[0] - c[0])*(b[2] - d[2]))   + \
            (p[2] - fourth*(a[2] + b[2]  + c[2] + d[2]))                \
            * ((a[0] - c[0])*(b[1] - d[1]) - (a[1] - c[1])*(b[0] - d[0]))

        return volpym

    def evalVolumeHex_b(self, x0, x1, x2, x3, x4, x5, x6, x7,
                        x0b, x1b, x2b, x3b, x4b, x5b, x6b, x7b):
        """
        Evaluate the derivative of the volume defined by the 8
        coordinates in the array x.

        Parameters
        ----------
        x{0:7} : arrays of len 3
            Arrays of defining the coordinates of the volume

        Returns
        -------
        xb{0:7} : arrays of len 3
            Derivatives of the volume wrt the points.
        """

        p = numpy.average([x0, x1, x2, x3, x4, x5, x6, x7], axis=0)
        pb = numpy.zeros(3)
        self.volpym_b(x0, x1, x3, x2, p, x0b, x1b, x3b, x2b, pb)
        self.volpym_b(x0, x2, x6, x4, p, x0b, x2b, x6b, x4b, pb)
        self.volpym_b(x0, x4, x5, x1, p, x0b, x4b, x5b, x1b, pb)
        self.volpym_b(x1, x5, x7, x3, p, x1b, x5b, x7b, x3b, pb)
        self.volpym_b(x2, x3, x7, x6, p, x2b, x3b, x7b, x6b, pb)
        self.volpym_b(x4, x6, x7, x5, p, x4b, x6b, x7b, x5b, pb)

        pb /= 8.0
        x0b += pb
        x1b += pb
        x2b += pb
        x3b += pb
        x4b += pb
        x5b += pb
        x6b += pb
        x7b += pb

    def volpym_b(self, a, b, c, d, p, ab, bb, cb, db, pb):
        """
        Compute the reverse-mode derivative of the square-based
        pyramid. This has been copied from reverse-mode AD'ed tapenade
        fortran code and converted to python to use vectors for the
        points.
        """
        fourth = 1.0/4.0
        volpymb = 1.0
        tempb = ((a[1]-c[1])*(b[2]-d[2])-(a[2]-c[2])*(b[1]-d[1]))*volpymb
        tempb0 = -(fourth*tempb)
        tempb1 = (p[0]-fourth*(a[0]+b[0]+c[0]+d[0]))*volpymb
        tempb2 = (b[2]-d[2])*tempb1
        tempb3 = (a[1]-c[1])*tempb1
        tempb4 = -((b[1]-d[1])*tempb1)
        tempb5 = -((a[2]-c[2])*tempb1)
        tempb6 = ((a[2]-c[2])*(b[0]-d[0])-(a[0]-c[0])*(b[2]-d[2]))*volpymb
        tempb7 = -(fourth*tempb6)
        tempb8 = (p[1]-fourth*(a[1]+b[1]+c[1]+d[1]))*volpymb
        tempb9 = (b[0]-d[0])*tempb8
        tempb10 = (a[2]-c[2])*tempb8
        tempb11 = -((b[2]-d[2])*tempb8)
        tempb12 = -((a[0]-c[0])*tempb8)
        tempb13 = ((a[0]-c[0])*(b[1]-d[1])-(a[1]-c[1])*(b[0]-d[0]))*volpymb
        tempb14 = -(fourth*tempb13)
        tempb15 = (p[2]-fourth*(a[2]+b[2]+c[2]+d[2]))*volpymb
        tempb16 = (b[1]-d[1])*tempb15
        tempb17 = (a[0]-c[0])*tempb15
        tempb18 = -((b[0]-d[0])*tempb15)
        tempb19 = -((a[1]-c[1])*tempb15)
        pb[0] = pb[0] + tempb
        ab[0] = ab[0] + tempb16 + tempb11 + tempb0
        bb[0] = bb[0] + tempb19 + tempb10 + tempb0
        cb[0] = cb[0] + tempb0 - tempb11 - tempb16
        db[0] = db[0] + tempb0 - tempb10 - tempb19
        ab[1] = ab[1] + tempb18 + tempb7 + tempb2
        cb[1] = cb[1] + tempb7 - tempb18 - tempb2
        bb[2] = bb[2] + tempb14 + tempb12 + tempb3
        db[2] = db[2] + tempb14 - tempb12 - tempb3
        ab[2] = ab[2] + tempb14 + tempb9 + tempb4
        cb[2] = cb[2] + tempb14 - tempb9 - tempb4
        bb[1] = bb[1] + tempb17 + tempb7 + tempb5
        db[1] = db[1] + tempb7 - tempb17 - tempb5
        pb[1] = pb[1] + tempb6
        pb[2] = pb[2] + tempb13


class CompositeVolumeConstraint(GeometricConstraint):
    """This class is used to represet a single volume constraints that is a
    group of other VolumeConstraints.
    """

    def __init__(self, name, vols, lower, upper, scaled, scale,
                 DVGeo, addToPyOpt):
        self.name = name
        self.nCon = 1
        self.vols = vols
        self.scaled = scaled
        self.lower = lower
        self.upper = upper
        self.scaled = scaled
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)

        # Now get the reference volume
        self.V0 = 0.0
        for vol in self.vols:
            self.V0 += vol.evalVolume()

    def evalFunctions(self, funcs, config):
        """
        Evaluate the function this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        V = 0.0
        for vol in self.vols:
            V += vol.evalVolume()
        if self.scaled:
            V /= self.V0
        funcs[self.name] = V

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            tmp = [] # List of dict derivatives
            for vol in self.vols:
                dVdPt = vol.evalVolumeSens()
                if self.scaled:
                    dVdPt /= self.V0
                tmp.append(vol.DVGeo.totalSensitivity(dVdPt, vol.name, config=config))

            # Now we need to sum up the derivatives:
            funcsSens[self.name] = tmp[0]
            for i in range(1, len(tmp)):
                for key in tmp[i]:
                    funcsSens[self.name][key] += tmp[i][key]



    def writeTecplot(self, handle):
        """No need to write the composite volume since each of the
        individual ones are already written"""
        pass

class LinearConstraint(object):
    """
    This class is used to represet a set of generic set of linear
    constriants coupling local shape variables together.
    """
    def __init__(self, name, indSetA, indSetB, factorA, factorB,
                 lower, upper, DVGeo, config):
        # No error checking here since the calling routine should have
        # already done it.
        self.name = name
        self.indSetA = indSetA
        self.indSetB = indSetB
        self.factorA = factorA
        self.factorB = factorB
        self.lower = lower
        self.upper = upper
        self.DVGeo = DVGeo
        self.ncon = 0
        self.wrt = []
        self.jac = {}
        self.config = config
        self._finalize()

    def evalFunctions(self, funcs):
        """
        Evaluate the function this object has and place in the funcs
        dictionary. Note that this function typically will not need to
        called since these constraints are supplied as a linear
        constraint jacobian they constraints themselves need to be
        revaluated.

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        cons = []
        for key in self.wrt:
            if key in self.DVGeo.DV_listLocal:
                cons.extend(self.jac[key].dot(self.DVGeo.DV_listLocal[key].value))
            elif key in self.DVGeo.DV_listSectionLocal:
                cons.extend(self.jac[key].dot(self.DVGeo.DV_listSectionLocal[key].value))

        funcs[self.name] = numpy.array(cons).real.astype('d')

    def evalFunctionsSens(self, funcsSens):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        funcsSens[self.name] = self.jac

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt. These constraints are added as
        linear constraints.
        """
        if self.ncon > 0:
            for key in self.jac:
                optProb.addConGroup(self.name+'_'+key, self.jac[key].shape[0],
                                    lower=self.lower, upper=self.upper, scale=1.0,
                                    linear=True, wrt=key, jac={key:self.jac[key]})
    def _finalize(self):
        """
        We have postponed actually determining the constraint jacobian
        until this function is called. Here we determine the actual
        constraint jacobains as they relate to the actual sets of
        local shape variables that may (or may not) be present in the
        DVGeo object.
        """
        self.vizConIndices = {}

        # Local Shape Variables
        for key in self.DVGeo.DV_listLocal:
             if self.config is None or self.config in self.DVGeo.DV_listLocal[key].config:

                # end for (indSet loop)
                cons = self.DVGeo.DV_listLocal[key].mapIndexSets(self.indSetA,self.indSetB)
                ncon = len(cons)
                if ncon > 0:
                    # Now form the jacobian:
                    ndv = self.DVGeo.DV_listLocal[key].nVal
                    jacobian = numpy.zeros((ncon, ndv))
                    for i in range(ncon):
                        jacobian[i, cons[i][0]] = self.factorA[i]
                        jacobian[i, cons[i][1]] = self.factorB[i]
                    self.jac[key] = jacobian

                # Add to the number of constraints and store indices which
                # we need for tecplot visualization
                self.ncon += len(cons)
                self.vizConIndices[key] = cons

        # Section local shape variables
        for key in self.DVGeo.DV_listSectionLocal:
             if self.config is None or self.config in self.DVGeo.DV_listSectionLocal[key].config:

                # end for (indSet loop)
                cons = self.DVGeo.DV_listSectionLocal[key].mapIndexSets(self.indSetA,self.indSetB)
                ncon = len(cons)
                if ncon > 0:
                    # Now form the jacobian:
                    ndv = self.DVGeo.DV_listSectionLocal[key].nVal
                    jacobian = numpy.zeros((ncon, ndv))
                    for i in range(ncon):
                        jacobian[i, cons[i][0]] = self.factorA[i]
                        jacobian[i, cons[i][1]] = self.factorB[i]
                    self.jac[key] = jacobian

                # Add to the number of constraints and store indices which
                # we need for tecplot visualization
                self.ncon += len(cons)
                self.vizConIndices[key] = cons

        # with-respect-to are just the keys of the jacobian
        self.wrt = list(self.jac.keys())

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of lete constraints
        to the open file handle
        """

        for key in self.vizConIndices:
            ncon = len(self.vizConIndices[key])
            nodes = numpy.zeros((ncon*2, 3))
            for i in range(ncon):
                nodes[2*i] = self.DVGeo.FFD.coef[self.indSetA[i]]
                nodes[2*i+1] = self.DVGeo.FFD.coef[self.indSetB[i]]

            handle.write('Zone T=%s\n'% (self.name+'_'+key))
            handle.write('Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n'% (
                ncon*2, ncon))
            handle.write('DATAPACKING=POINT\n')
            for i in range(ncon*2):
                handle.write('%f %f %f\n'% (nodes[i, 0], nodes[i, 1], nodes[i, 2]))

            for i in range(ncon):
                handle.write('%d %d\n'% (2*i+1, 2*i+2))

class GearPostConstraint(GeometricConstraint):
    """
    This class is used to represet a single volume constraint. The
    parameter list is explained in the addVolumeConstaint() of
    the DVConstraints class
    """
    def __init__(self, name, wimpressCalc, up, down, thickLower,
                 thickUpper, thickScaled, MACFracLower, MACFracUpper,
                 DVGeo, addToPyOpt):

        self.name = name
        self.wimpress = wimpressCalc
        self.thickLower = thickLower
        self.thickUpper = thickUpper
        self.thickScaled = thickScaled
        self.MACFracLower = MACFracLower
        self.MACFracUpper = MACFracUpper
        self.coords = numpy.array([up, down])
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, None, None,
                                     None,None, self.DVGeo,
                                     self.addToPyOpt)
        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name)

        # Compute the reference length
        self.D0 = numpy.linalg.norm(self.coords[0] - self.coords[1])

    def evalFunctions(self, funcs, config):

        # Update the gear post locations
        self.coords = self.DVGeo.update(self.name, config=config)

        # Compute the thickness constraint
        D = numpy.linalg.norm(self.coords[0] - self.coords[1])
        if self.thickScaled:
            D = D/self.D0

        # Compute the values we need from the wimpress calc
        wfuncs = {}
        self.wimpress.evalFunctions(wfuncs)

        # Now the constraint value is
        postLoc = 0.5*(self.coords[0, 0] + self.coords[1, 0])
        locCon = (postLoc - wfuncs['xLEMAC'])/wfuncs['MAC']

        # Final set of two constrains
        funcs[self.name + '_thick'] = D
        funcs[self.name + '_MAC'] = locCon

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        nDV = self.DVGeo.getNDV()
        if nDV > 0:

            wfuncs = {}
            self.wimpress.evalFunctions(wfuncs)

            wSens = {}
            self.wimpress.evalFunctionsSens(wSens)

            # Accumulate the derivative into p1b and p2b
            p1b, p2b = geo_utils.eDist_b(
                self.coords[0, :], self.coords[1, :])
            if self.thickScaled:
                p1b /= self.D0
                p2b /= self.D0

            funcsSens[self.name + '_thick'] = self.DVGeo.totalSensitivity(
                numpy.array([[p1b, p2b]]), self.name, config=config)

            # And now we need the sensitivty of the conLoc calc
            p1b[:] = 0
            p2b[:] = 0
            p1b[0] += 0.5/wfuncs['MAC']
            p2b[0] += 0.5/wfuncs['MAC']

            tmpSens = self.DVGeo.totalSensitivity(
                numpy.array([[p1b, p2b]]), self.name, config=config)

            # And we need the sensitity of conLoc wrt 'xLEMAC' and 'MAC'
            postLoc = 0.5*(self.coords[0, 0] + self.coords[1, 0])
            for key in wSens['xLEMAC']:
                tmpSens[key] -= wSens['xLEMAC'][key]/wfuncs['MAC']
                tmpSens[key] += wfuncs['xLEMAC']/wfuncs['MAC']**2 * wSens['MAC'][key]
                tmpSens[key] -= postLoc/wfuncs['MAC']**2 * wSens['MAC'][key]
            funcsSens[self.name + '_MAC'] = tmpSens

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addCon(self.name + '_thick', lower=self.thickLower,
                           upper=self.thickUpper, wrt=self.DVGeo.getVarNames())

            optProb.addCon(self.name + '_MAC', lower=self.MACFracLower,
                           upper=self.MACFracUpper, wrt=self.DVGeo.getVarNames())


class CircularityConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of circularity
    constraint. One of these objects is created each time a
    addCircularityConstraints call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, center, coords, lower, upper, scale, DVGeo,
                 addToPyOpt):
        self.name = name
        self.center = numpy.array(center).reshape((1,3))
        self.coords = coords
        self.nCon = self.coords.shape[0]-1
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)


        self.X = numpy.zeros(self.nCon)

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name+'coords')
        self.DVGeo.addPointSet(self.center, self.name+'center')

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name+'coords', config=config)
        self.center = self.DVGeo.update(self.name+'center', config=config)

        self._computeLengths(self.center,self.coords,self.X)

        funcs[self.name] = self.X

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dLndPt = numpy.zeros((self.nCon,
                                 self.coords.shape[0],
                                 self.coords.shape[1]))
            dLndCn = numpy.zeros((self.nCon,
                                  self.center.shape[0],
                                  self.center.shape[1]))

            xb = numpy.zeros(self.nCon)
            for con in range(self.nCon):
                centerb = dLndCn[con,0,:]
                coordsb = dLndPt[con,:,:]
                xb[:] = 0.
                xb[con] = 1.
                # reflength2 = 0
                # for i in range(3):
                #     reflength2 = reflength2 + (center[i]-coords[0,i])**2
                reflength2 = numpy.sum((self.center-self.coords[0,:])**2)
                reflength2b = 0.0
                for i in range(self.nCon):
                    # length2 = 0
                    # for j in range(3):
                    #     length2 = length2 + (center[j]-coords[i+1, j])**2
                    length2 = numpy.sum((self.center-self.coords[i+1,:])**2)

                    if (length2/reflength2 == 0.0):
                        tempb1 = 0.0
                    else:
                        tempb1 = xb[i]/(2.0*numpy.sqrt(length2/reflength2)*reflength2)
                    length2b = tempb1
                    reflength2b = reflength2b - length2*tempb1/reflength2
                    xb[i] = 0.0
                    for j in reversed(range(3)):
                        tempb0 = 2*(self.center[0,j]-self.coords[i+1, j])*length2b
                        centerb[j] = centerb[j] + tempb0
                        coordsb[i+1, j] = coordsb[i+1, j] - tempb0
                for j in reversed(range(3)):#DO i=3,1,-1
                    tempb = 2*(self.center[0,j]-self.coords[0, j])*reflength2b
                    centerb[j] = centerb[j] + tempb
                    coordsb[0, j] = coordsb[0, j] - tempb

            tmpPt = self.DVGeo.totalSensitivity(dLndPt, self.name+'coords', config=config)
            tmpCn = self.DVGeo.totalSensitivity(dLndCn, self.name+'center', config=config)
            tmpTotal = {}
            for key in tmpPt:
                tmpTotal[key] = tmpPt[key]+tmpCn[key]

            funcsSens[self.name] = tmpTotal

    def _computeLengths(self,center,coords,X):
        '''
        compute the lengths from the center and coordinates
        '''
        reflength2 = numpy.sum((center-coords[0,:])**2)
        for i in range(self.nCon):
            length2 = numpy.sum((self.center-self.coords[i+1,:])**2)
            X[i] = numpy.sqrt(length2/reflength2)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write('Zone T=%s_coords\n'% self.name)
        handle.write('Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n'% (
            len(self.coords), len(self.coords)-1))
        handle.write('DATAPACKING=POINT\n')
        for i in range(len(self.coords)):
            handle.write('%f %f %f\n'% (self.coords[i, 0], self.coords[i, 1],
                                        self.coords[i, 2]))

        for i in range(len(self.coords)-1):
            handle.write('%d %d\n'% (i+1, i+2))

        handle.write('Zone T=%s_center\n'% self.name)
        handle.write('Nodes = 2, Elements = 1 ZONETYPE=FELINESEG\n')
        handle.write('DATAPACKING=POINT\n')
        handle.write('%f %f %f\n'% (self.center[0,0], self.center[0,1],
                                    self.center[0,2]))
        handle.write('%f %f %f\n'% (self.center[0,0], self.center[0,1],
                                    self.center[0,2]))
        handle.write('%d %d\n'% (1, 2))

class PlanarityConstraint(GeometricConstraint):
    """
    DVConstraints representation of a surface planarity constraint.
    Constrain that all of the points on this surface are co-planar.
    One of these objects is created each time an
    addPlanarityConstraint call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, axis, origin, p0, v1, v2, lower, upper, scale,
                 DVGeo, addToPyOpt):
        self.name = name
        self.nCon = 1#len(p0)*3
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)

        # create the output array
        self.X = numpy.zeros(self.nCon)
        self.n = len(p0)

        # The first thing we do is convert v1 and v2 to coords
        self.axis = axis
        self.p0 = p0
        self.p1 = v1+p0
        self.p2 = v2+p0
        self.origin = origin

        # Now embed the coordinates and origin into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.p0, self.name+'p0')
        self.DVGeo.addPointSet(self.p1, self.name+'p1')
        self.DVGeo.addPointSet(self.p2, self.name+'p2')
        self.DVGeo.addPointSet(self.origin, self.name+'origin')


    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.p0 = self.DVGeo.update(self.name+'p0', config=config)
        self.p1 = self.DVGeo.update(self.name+'p1', config=config)
        self.p2 = self.DVGeo.update(self.name+'p2', config=config)
        self.origin = self.DVGeo.update(self.name+'origin', config=config)

        allPoints = numpy.vstack([self.p0,self.p1,self.p2])

        # Compute the distance from the origin to each point
        dist = allPoints-self.origin

        #project it onto the axis
        self.X[0] = 0
        for i in range(self.n*3):
            self.X[0] += numpy.dot(self.axis,dist[i,:])**2
        self.X[0]= numpy.sqrt(self.X[0])
        funcs[self.name] = self.X

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dPdp0 = numpy.zeros((self.nCon,
                                 self.p0.shape[0],
                                 self.p0.shape[1]))
            dPdp1 = numpy.zeros((self.nCon,
                                 self.p1.shape[0],
                                 self.p1.shape[1]))

            dPdp2 = numpy.zeros((self.nCon,
                                 self.p2.shape[0],
                                 self.p2.shape[1]))

            dPdO = numpy.zeros((self.nCon,
                                self.origin.shape[0],
                                self.origin.shape[1]))

        # copy data into all points array
        # allpoints(1:n) = p0
        # allpoints(n:2*n) = p1
        # allpoints(2*n:3*n) = p2
        allPoints = numpy.vstack([self.p0,self.p1,self.p2])

        # Compute the distance from the origin to each point
        # for i in range(n*3):#DO i=1,n*3
        #     for j in range(3):#DO j=1,3
        #         dist(i, j) = allpoints(i, j) - origin(j)
        dist = allPoints-self.origin

        scalardist = numpy.zeros(self.n*3)
        tmpX = 0
        for i in range(self.n*3):
            scalardist[i] = numpy.dot(self.axis,dist[i,:])
            tmpX+=scalardist[i]**2

        xb = numpy.zeros(self.nCon)
        axisb = numpy.zeros(3)

        scalardistb = numpy.zeros((self.n*3))
        allpointsb = numpy.zeros((self.n*3,3))
        distb = numpy.zeros((self.n*3,3))
        for con in range(self.nCon):
            p0b = dPdp0[con,:,:]
            p1b = dPdp1[con,:,:]
            p2b = dPdp2[con,:,:]
            originb = dPdO[con,0,:]
            axisb[:] = 0.0
            originb[:] = 0.0
            scalardistb[:] = 0.0
            allpointsb[:,:] = 0.0
            distb[:,:] = 0.0
            xb[:] = 0
            xb[con] = 1.0
            if(self.X[0] == 0.0):
                xb[con] = 0.0
            else:
                xb[con] = xb[con]/(2.0*numpy.sqrt(tmpX))

            for i in reversed(range(self.n*3)):#DO i=3*n,1,-1
                scalardistb[i] = scalardistb[i] + 2.0*scalardist[i]*xb[con]#/(self.n*3)
                # CALL DOT_B(axis, axisb, dist(i, :), distb(i, :), scalardist(i), &
                #            &        scalardistb(i))
                axisb, distb[i,:] = geo_utils.dot_b(self.axis, dist[i, :], scalardistb[i])
                scalardistb[i] = 0.0
                for j in reversed(range(3)):#DO j=3,1,-1
                    allpointsb[i, j] = allpointsb[i, j] + distb[i, j]
                    originb[j] = originb[j] - distb[i, j]
                    distb[i, j] = 0.0

            p2b[:,:] = 0.0
            p2b[:,:] = allpointsb[2*self.n:3*self.n]
            allpointsb[2*self.n:3*self.n] = 0.0
            p1b[:,:] = 0.0
            p1b[:,:] = allpointsb[self.n:2*self.n]
            allpointsb[self.n:2*self.n] = 0.0
            p0b[:,:] = 0.0
            p0b[:,:] = allpointsb[0:self.n]


            # map back to DVGeo
            tmpp0 = self.DVGeo.totalSensitivity(dPdp0, self.name+'p0',
                                                config=config)
            tmpp1 = self.DVGeo.totalSensitivity(dPdp1, self.name+'p1',
                                                config=config)
            tmpp2 = self.DVGeo.totalSensitivity(dPdp2, self.name+'p2',
                                                config=config)
            tmpO = self.DVGeo.totalSensitivity(dPdO, self.name+'origin',
                                               config=config)

            tmpTotal = {}
            for key in tmpp0:
                tmpTotal[key] = tmpp0[key]+tmpp1[key]+tmpp2[key]+tmpO[key]


            funcsSens[self.name] = tmpTotal


    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write('Zone T=%s_surface\n'% self.name)
        handle.write('Nodes = %d, Elements = %d ZONETYPE=FETRIANGLE\n'% (
            3*self.n, self.n))
        handle.write('DATAPACKING=POINT\n')
        for i in range(self.n):
            handle.write('%f %f %f\n'% (self.p0[i, 0], self.p0[i, 1],
                                        self.p0[i, 2]))
        for i in range(self.n):
            handle.write('%f %f %f\n'% (self.p1[i, 0], self.p1[i, 1],
                                        self.p1[i, 2]))

        for i in range(self.n):
            handle.write('%f %f %f\n'% (self.p2[i, 0], self.p2[i, 1],
                                        self.p2[i, 2]))

        for i in range(self.n):
            handle.write('%d %d %d\n'% (i+1, i+self.n+1, i+self.n*2+1))

        handle.write('Zone T=%s_center\n'% self.name)
        handle.write('Nodes = 2, Elements = 1 ZONETYPE=FELINESEG\n')
        handle.write('DATAPACKING=POINT\n')
        handle.write('%f %f %f\n'% (self.origin[0,0], self.origin[0,1],
                                    self.origin[0,2]))
        handle.write('%f %f %f\n'% (self.origin[0,0], self.origin[0,1],
                                    self.origin[0,2]))
        handle.write('%d %d\n'% (1, 2))

class ColinearityConstraint(GeometricConstraint):
    """
    DVConstraints representation of a colinearity constraint.
    Constrain that all of the points provided stay colinear with the
    specified axis.
    One of these objects is created each time an
    addColinearityConstraint call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, axis, origin, coords, lower, upper, scale,
                 DVGeo, addToPyOpt):
        self.name = name
        self.nCon = len(coords)
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)

        # create the output array
        self.X = numpy.zeros(self.nCon)

        # The first thing we do is convert v1 and v2 to coords
        self.axis = axis
        self.origin = origin
        self.coords = coords

        # Now embed the coordinates and origin into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.origin, self.name+'origin')
        self.DVGeo.addPointSet(self.coords, self.name+'coords')


    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name+'coords', config=config)
        self.origin = self.DVGeo.update(self.name+'origin', config=config)

        # # Compute the direction from each point to the origin
        # dirVec = self.origin-self.coords

        # # compute the cross product with the desired axis. Cross product
        # # will be zero if the direction vector is the same as the axis
        # resultDir = numpy.cross(self.axis,dirVec)

        # for i in range(len(resultDir)):
        #     self.X[i] = geo_utils.euclideanNorm(resultDir[i,:])
        self.X = self._computeDist(self.origin,self.coords,self.axis)

        funcs[self.name] = self.X

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dCdPt = numpy.zeros((self.nCon,
                                     self.coords.shape[0],
                                     self.coords.shape[1]))
            dCdOrigin = numpy.zeros((self.nCon,
                                     self.origin.shape[0],
                                     self.origin.shape[1]))
            dCdAxis = numpy.zeros((self.nCon,
                                     self.axis.shape[0],
                                     self.axis.shape[1]))

            #Compute the direction from each point to the origin
            # for i in range(n):
            #     for j in range(3):
            #         dirvec[i, j] = origin[j] - coords[i, j]
            dirVec = self.origin-self.coords

            # axisb = 0.0
            # dirvecb = 0.0
            # for i in range(self.nCon):
            #     resultdir = numpy.cross(axis, dirvec[i, :])
            #     self.X[i] = 0
            #     for j in range(3):
            #         self.X[i] = self.X[i] + resultdir[j]**2
            resultDir = numpy.cross(self.axis,dirVec)
            tmpX = numpy.zeros(self.nCon)
            for i in range(len(resultDir)):
                #self.X[i] = geo_utils.euclideanNorm(resultDir[i,:])
                for j in range(3):
                    tmpX[i] += resultDir[i,j]**2

            resultdirb = numpy.zeros(3)
            dirvecb = numpy.zeros_like(dirVec)
            xb = numpy.zeros(self.nCon)
            for con in range(self.nCon):
                originb = dCdOrigin[con,0,:]
                coordsb = dCdPt[con,:,:]
                axisb = dCdAxis[con,0,:]
                xb[:] = 0.
                xb[con] = 1.

                for i in range(self.nCon):
                    if (tmpX[i] == 0.0):
                        xb[i] = 0.0
                    else:
                        xb[i] = xb[i]/(2.0*numpy.sqrt(tmpX[i]))

                    resultdirb[:] = 0.0
                    for j in reversed(range(3)):#DO j=3,1,-1
                        resultdirb[j] = resultdirb[j] + 2*resultDir[i,j]*xb[i]

                    xb[i] = 0.0
                    #CALL CROSS_B(axis, axisb, dirvec(i, :), dirvecb(i, :), resultdirb)
                    axisb, dirvecb[i,:] = geo_utils.cross_b(self.axis[0,:],dirVec[i, :], resultdirb)

                # coordsb = 0.0
                # originb = 0.0
                for i in reversed(range(len(coordsb))):#DO i=n,1,-1
                    for j in reversed(range(3)):#DO j=3,1,-1
                        originb[j] = originb[j] + dirvecb[i, j]
                        coordsb[i, j] = coordsb[i, j] - dirvecb[i, j]
                        dirvecb[i, j] = 0.0

            tmpPt = self.DVGeo.totalSensitivity(dCdPt, self.name+'coords',
                                                config=config)
            tmpOrigin = self.DVGeo.totalSensitivity(dCdOrigin, self.name+'origin',
                                                    config=config)

            tmpTotal = {}
            for key in tmpPt:
                tmpTotal[key] = tmpPt[key]+tmpOrigin[key]

            tmpTotal[self.name+'axis'] =  dCdAxis

            funcsSens[self.name] = tmpTotal

    def addVariablesPyOpt(self,optProb):
        """
        Add the axis variable for the colinearity constraint to pyOpt
        """

        if self.addVarToPyOpt:
            optProb.addVarGroup(dv.name, dv.nVal, 'c', value=dv.value,
                                lower=dv.lower, upper=dv.upper,
                                scale=dv.scale)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """
        handle.write('Zone T=%s_coords\n'% self.name)
        handle.write('Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n'% (
            len(self.coords)+1, len(self.coords)))
        handle.write('DATAPACKING=POINT\n')
        handle.write('%f %f %f\n'% (self.origin[0,0], self.origin[0,1],
                                    self.origin[0,2]))
        for i in range(len(self.coords)):
            handle.write('%f %f %f\n'% (self.coords[i, 0], self.coords[i, 1],
                                        self.coords[i, 2]))

        for i in range(len(self.coords)):
            handle.write('%d %d\n'% (i+1, i+2))

    def _computeDist(self,origin,coords,axis, dtype='d'):
        """
        compute the distance of coords from the defined axis.
        """
        # Compute the direction from each point to the origin
        dirVec = origin-coords

        # compute the cross product with the desired axis. Cross product
        # will be zero if the direction vector is the same as the axis
        resultDir = numpy.cross(axis,dirVec)

        X = numpy.zeros(len(coords),dtype)
        for i in range(len(resultDir)):
            X[i] = geo_utils.euclideanNorm(resultDir[i,:])

        return X

class SurfaceAreaConstraint(GeometricConstraint):
    """
    DVConstraints representation of a surface area
    constraint. One of these objects is created each time a
    addSurfaceAreaConstraints call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, p0, v1, v2, lower, upper, scale, scaled, DVGeo,
                 addToPyOpt):
        self.name = name
        self.nCon = 1
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.scaled = scaled
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)

        # create output array
        self.X = numpy.zeros(self.nCon)
        self.n = len(p0)

        # The first thing we do is convert v1 and v2 to coords
        self.p0 = p0
        self.p1 = v1+p0
        self.p2 = v2+p0

        # Now embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.p0, self.name+'p0')
        self.DVGeo.addPointSet(self.p1, self.name+'p1')
        self.DVGeo.addPointSet(self.p2, self.name+'p2')

        # compute the reference area
        self.X0 = self._computeArea(self.p0,self.p1,self.p2)

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.p0 = self.DVGeo.update(self.name+'p0', config=config)
        self.p1 = self.DVGeo.update(self.name+'p1', config=config)
        self.p2 = self.DVGeo.update(self.name+'p2', config=config)

        self.X = self._computeArea(self.p0,self.p1,self.p2)
        if self.scaled:
            self.X/= self.X0
        funcs[self.name] = self.X

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dAdp0 = numpy.zeros((self.nCon,
                                 self.p0.shape[0],
                                 self.p0.shape[1]))
            dAdp1 = numpy.zeros((self.nCon,
                                 self.p1.shape[0],
                                 self.p1.shape[1]))

            dAdp2 = numpy.zeros((self.nCon,
                                 self.p2.shape[0],
                                 self.p2.shape[1]))

            p0 = self.p0
            p1 = self.p1
            p2 = self.p2
            for con in range(self.nCon):
                p0b = dAdp0[con,:,:]
                p1b = dAdp1[con,:,:]
                p2b = dAdp2[con,:,:]
                areab = 1
                areasb = numpy.empty(self.n)
                crossesb = numpy.empty((self.n,3))
                v1b = numpy.empty((self.n,3))
                v2b = numpy.empty((self.n,3))
                if self.scaled:
                    areab = areab/self.X0
                areasb[:] = areab/2.

                v1 = p1 - p0
                v2 = p2 - p0

                crosses = numpy.cross(v1, v2)
                    # for j in range(3):
                    #     areas(i) = areas(i) + crosses(i, j)**2
                    #areas[i] = numpy.sum(crosses[i, :]**2)
                areas = numpy.sum(crosses**2,axis=1)
                for i in range(self.n):#DO i=1,n
                    if (areas[i] == 0.0):
                        areasb[i] = 0.0
                    else:
                        areasb[i] = areasb[i]/(2.0*numpy.sqrt(areas[i]))

                    # for j in reversed(range(3)):#DO j=3,1,-1
                    #     crossesb(i, j) = crossesb(i, j) + 2*crosses(i, j)*areasb(i)
                    crossesb[i, :] = 2*crosses[i, :]*areasb[i]

                    v1b[i,:],v2b[i,:] = geo_utils.cross_b(v1[i, :], v2[i, :], crossesb[i, :])

                    # for j in reversed(range(3)):#DO j=3,1,-1
                    #      p2b(i, j) = p2b(i, j) + v2b(i, j)
                    #      p0b(i, j) = p0b(i, j) - v1b(i, j) - v2b(i, j)
                    #      v2b(i, j) = 0.0
                    #      p1b(i, j) = p1b(i, j) + v1b(i, j)
                    #      v1b(i, j) = 0.0
                    p2b[i, :] = v2b[i, :]
                    p0b[i, :] = - v1b[i, :] - v2b[i, :]
                    p1b[i, :] = p1b[i, :] + v1b[i, :]



            tmpp0 = self.DVGeo.totalSensitivity(dAdp0, self.name+'p0',
                                                config=config)
            tmpp1 = self.DVGeo.totalSensitivity(dAdp1, self.name+'p1',
                                                config=config)
            tmpp2 = self.DVGeo.totalSensitivity(dAdp2, self.name+'p2',
                                                config=config)
            tmpTotal = {}
            for key in tmpp0:
                tmpTotal[key] = tmpp0[key]+tmpp1[key]+tmpp2[key]


            funcsSens[self.name] = tmpTotal

    def _computeArea(self, p0, p1, p2):
        """
        compute area based on three point arrays
        """
        # convert p1 and p2 to v1 and v2
        v1 = p1- p0
        v2 = p2- p0

        #compute the areas
        areaVec = numpy.cross(v1, v2)

        #area = numpy.linalg.norm(areaVec,axis=1)
        area = 0
        for i in range(len(areaVec)):
            area += geo_utils.euclideanNorm(areaVec[i,:])

        #return numpy.sum(area)/2.0
        return area/2.0

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write('Zone T=%s_surface\n'% self.name)
        handle.write('Nodes = %d, Elements = %d ZONETYPE=FETRIANGLE\n'% (
            3*self.n, self.n))
        handle.write('DATAPACKING=POINT\n')
        for i in range(self.n):
            handle.write('%f %f %f\n'% (self.p0[i, 0], self.p0[i, 1],
                                        self.p0[i, 2]))
        for i in range(self.n):
            handle.write('%f %f %f\n'% (self.p1[i, 0], self.p1[i, 1],
                                        self.p1[i, 2]))

        for i in range(self.n):
            handle.write('%f %f %f\n'% (self.p2[i, 0], self.p2[i, 1],
                                        self.p2[i, 2]))

        for i in range(self.n):
            handle.write('%d %d %d\n'% (i+1, i+self.n+1, i+self.n*2+1))


class ProjectedAreaConstraint(GeometricConstraint):
    """
    DVConstraints representation of a surface area
    constraint. One of these objects is created each time a
    addSurfaceAreaConstraints call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, p0, v1, v2, axis, lower, upper, scale, scaled,
                DVGeo, addToPyOpt):
        self.name = name
        self.nCon = 1
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.scaled = scaled
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)

        # create output array
        self.X = numpy.zeros(self.nCon)
        self.n = len(p0)
        self.axis = axis
        self.activeTris = numpy.zeros(self.n)

        # The first thing we do is convert v1 and v2 to coords
        self.p0 = p0
        self.p1 = v1+p0
        self.p2 = v2+p0

        # Now embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.p0, self.name+'p0')
        self.DVGeo.addPointSet(self.p1, self.name+'p1')
        self.DVGeo.addPointSet(self.p2, self.name+'p2')

        # compute the reference area
        self.X0 = self._computeArea(self.p0, self.p1, self.p2, self.axis)

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.p0 = self.DVGeo.update(self.name+'p0', config=config)
        self.p1 = self.DVGeo.update(self.name+'p1', config=config)
        self.p2 = self.DVGeo.update(self.name+'p2', config=config)

        self.X = self._computeArea(self.p0, self.p1, self.p2, self.axis)
        if self.scaled:
            self.X /= self.X0
        funcs[self.name] = self.X

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dAdp0 = numpy.zeros((self.nCon,
                                 self.p0.shape[0],
                                 self.p0.shape[1]))
            dAdp1 = numpy.zeros((self.nCon,
                                 self.p1.shape[0],
                                 self.p1.shape[1]))

            dAdp2 = numpy.zeros((self.nCon,
                                 self.p2.shape[0],
                                 self.p2.shape[1]))
        p0 = self.p0
        p1 = self.p1
        p2 = self.p2
        for con in range(self.nCon):
            p0b = dAdp0[con,:,:]
            p1b = dAdp1[con,:,:]
            p2b = dAdp2[con,:,:]
            areab = 1
            areasb = numpy.empty(self.n)
            if self.scaled:
                areab = areab/self.X0
            areasb[:] = areab/2.

            for i in range(self.n):
                v1 = p1[i,:] - p0[i,:]
                v2 = p2[i,:] - p0[i,:]
                SAvec = numpy.cross(v1, v2)
                PA = numpy.dot(SAvec, self.axis)
                if PA > 0:
                    PAb = areasb[i]
                else:
                    PAb = 0.0
                SAvecb, axisb = geo_utils.dot_b(SAvec, self.axis, PAb)
                v1b, v2b = geo_utils.cross_b(v1, v2, SAvecb)
                p2b[i,:] = p2b[i,:] + v2b
                p1b[i,:] = p1b[i,:] + v1b
                p0b[i,:] = p0b[i,:] - v1b - v2b

        tmpp0 = self.DVGeo.totalSensitivity(dAdp0, self.name+'p0',
                                            config=config)
        tmpp1 = self.DVGeo.totalSensitivity(dAdp1, self.name+'p1',
                                            config=config)
        tmpp2 = self.DVGeo.totalSensitivity(dAdp2, self.name+'p2',
                                            config=config)
        tmpTotal = {}
        for key in tmpp0:
            tmpTotal[key] = tmpp0[key]+tmpp1[key]+tmpp2[key]

        funcsSens[self.name] = tmpTotal

    def _computeArea(self, p0, p1, p2, axis, plot=False):
        """
        Compute projected surface area
        """

        # Convert p1 and p2 to v1 and v2
        v1 = p1- p0
        v2 = p2- p0

        # Compute the surface area vectors for each triangle patch
        surfaceAreas = numpy.cross(v1, v2)

        # Compute the projected area of each triangle patch
        projectedAreas = numpy.dot(surfaceAreas, axis)

        # Cut out negative projected areas to get one side of surface
        if plot:
            for i in range(self.n):
                if projectedAreas[i] < 0:
                    self.activeTris[i] = 1
                else:
                    projectedAreas[i] = 0.0
        else:
            projectedAreas[projectedAreas<0] = 0.0

        # Sum projected areas and divide by two for triangle area
        totalProjectedArea = numpy.sum(projectedAreas)/2.0

        return totalProjectedArea


    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """
        self._computeArea(self.p0, self.p1, self.p2, self.axis, plot=True)
        nActiveTris = int(numpy.sum(self.activeTris))
        p0 = self.p0.copy()
        p1 = self.p1.copy()
        p2 = self.p2.copy()
        if self.axis[0] == 1.0:
            p0[:,0] = numpy.zeros(self.n)
            p1[:,0] = numpy.zeros(self.n)
            p2[:,0] = numpy.zeros(self.n)
        if self.axis[1] == 1.0:
            p0[:,1] = numpy.zeros(self.n)
            p1[:,1] = numpy.zeros(self.n)
            p2[:,1] = numpy.zeros(self.n)
        if self.axis[2] == 1.0:
            p0[:,2] = numpy.zeros(self.n)
            p1[:,2] = numpy.zeros(self.n)
            p2[:,2] = numpy.zeros(self.n)

        handle.write('Zone T=%s_surface\n'% self.name)
        handle.write('Nodes = %d, Elements = %d ZONETYPE=FETRIANGLE\n'% (
            3*nActiveTris, nActiveTris))
        handle.write('DATAPACKING=POINT\n')
        for i in range(self.n):
            if self.activeTris[i]:
                handle.write('%f %f %f\n'% (p0[i, 0], p0[i, 1], p0[i, 2]))
        for i in range(self.n):
            if self.activeTris[i]:
                handle.write('%f %f %f\n'% (p1[i, 0], p1[i, 1], p1[i, 2]))
        for i in range(self.n):
            if self.activeTris[i]:
                handle.write('%f %f %f\n'% (p2[i, 0], p2[i, 1], p2[i, 2]))
        iActive = 0
        for i in range(self.n):
            if self.activeTris[i]:
                handle.write('%d %d %d\n'% (iActive+1, iActive+nActiveTris+1,
                        iActive+nActiveTris*2+1))
                iActive += 1

class CurvatureConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of the curvature constraint.
    One of these objects is created each time a addCurvatureConstraint is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, surfs, curvatureType, lower, upper, scaled, scale, KSCoeff, DVGeo,
                 addToPyOpt):
        self.name = name
        self.nSurfs = len(surfs) # we support multiple surfaces (plot3D files)
        self.X = []
        self.X_map = []
        self.node_map = []
        self.coords = []
        for iSurf in range(self.nSurfs):
            # A list of the coordinates arrays for each surface, flattened in order
            # to vectorize operations
            self.X += [numpy.reshape(surfs[iSurf].X,-1)]
            # A list of maping arrays used to translate from the structured index
            # to the flatten index number of X
            # For example: X[iSurf][X_map[iSurf][i,j,2]] gives the z coordinate
            # of the node in the i-th row and j-th column on surface iSurf
            self.X_map += [numpy.reshape(numpy.array(range(surfs[iSurf].X.size)),surfs[iSurf].X.shape)]
            # A list of maping arrays used to provide a unique node number for
            # every node on each surface
            # For example: node_map[iSurf][i,j] gives the node number
            # of the node in the i-th row and j-th column on surface iSurf
            self.node_map += [numpy.reshape(numpy.array(range(surfs[iSurf].X.size//3)),(surfs[iSurf].X.shape[0],surfs[iSurf].X.shape[1]))]
            # A list of the coordinates arrays for each surface, in the shape that DVGeo expects (N_nodes,3)
            self.coords += [numpy.reshape(self.X[iSurf],(surfs[iSurf].X.shape[0]*surfs[iSurf].X.shape[1],3))]
        self.nCon = 1
        self.curvatureType = curvatureType
        self.lower = lower
        self.upper = upper
        self.scaled = scaled
        self.scale = scale
        self.KSCoeff=KSCoeff
        if self.KSCoeff==None:
            # set KSCoeff to be the number of points in the plot 3D files
            self.KSCoeff=0.0
            for i in range(len(self.coords)):
                self.KSCoeff+=len(self.coords[i])
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        GeometricConstraint.__init__(self, self.name, self.nCon, self.lower,
                                     self.upper, self.scale, self.DVGeo,
                                     self.addToPyOpt)

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided. We need to add a point set for each surface:
        for iSurf in range(self.nSurfs):
            self.DVGeo.addPointSet(self.coords[iSurf], self.name+'%d'%(iSurf))

        # compute the reference curvature for normalization
        self.curvatureRef=0.0
        for iSurf in range(self.nSurfs):
            self.curvatureRef += self.evalCurvArea(iSurf)[0]

        if(MPI.COMM_WORLD.rank==0):
            print("Reference curvature: ",self.curvatureRef)


    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates for each surface:
        funcs[self.name] = 0
        for iSurf in range(self.nSurfs):
            self.coords[iSurf] = self.DVGeo.update(self.name+'%d'%(iSurf), config=config)
            self.X[iSurf] = numpy.reshape(self.coords[iSurf],-1)
            if self.scaled:
                funcs[self.name] += self.evalCurvArea(iSurf)[0]/self.curvatureRef
            else:
                funcs[self.name] += self.evalCurvArea(iSurf)[0]

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            # Add the sensitivity of the curvature integral over all surfaces
            for iSurf in range(self.nSurfs):
                DkSDX = self.evalCurvAreaSens(iSurf)
                if self.scaled:
                    DkSDX /= self.curvatureRef
                # Reshape the Xpt sensitivity to the shape DVGeo is expecting
                DkSDpt = numpy.reshape(DkSDX, self.coords[iSurf].shape)
                if iSurf == 0:
                    funcsSens[self.name] = self.DVGeo.totalSensitivity(
                        DkSDpt, self.name+'%d'%(iSurf), config=config)
                else:
                    tmp = self.DVGeo.totalSensitivity(
                        DkSDpt, self.name+'%d'%(iSurf), config=config)
                    for key in funcsSens[self.name]:
                        funcsSens[self.name][key] += tmp[key]

    def evalCurvArea(self, iSurf):
        '''
        Evaluate the integral K**2 over the surface area of the wing.
        Where K is the Gaussian curvature.
        '''
        # Evaluate the derivitive of the position vector of every point on the
        # surface wrt to the parameteric corrdinate u and v
        t_u = self.evalDiff(iSurf, self.X[iSurf], 'u')
        t_v = self.evalDiff(iSurf, self.X[iSurf], 'v')
        # Compute the normal vector by taking the cross product of t_u and t_v
        n = self.evalCross(iSurf, t_u,t_v)
        # Compute the norm of tu_ x tv
        n_norm = self.evalNorm(iSurf, n)
        # Normalize the normal vector
        n_hat = numpy.zeros_like(n)
        n_hat[self.X_map[iSurf][:,:,0]]=n[self.X_map[iSurf][:,:,0]]/n_norm[self.node_map[iSurf][:,:]]
        n_hat[self.X_map[iSurf][:,:,1]]=n[self.X_map[iSurf][:,:,1]]/n_norm[self.node_map[iSurf][:,:]]
        n_hat[self.X_map[iSurf][:,:,2]]=n[self.X_map[iSurf][:,:,2]]/n_norm[self.node_map[iSurf][:,:]]
        # Evaluate the second derivitives of the position vector wrt u and v
        t_uu = self.evalDiff(iSurf, t_u, 'u')
        t_vv = self.evalDiff(iSurf, t_v, 'v')
        t_uv = self.evalDiff(iSurf, t_v, 'u')
        # Compute the components of the first fundamental form of a parameteric
        # surface
        E = self.evalInProd(iSurf, t_u,t_u)
        F = self.evalInProd(iSurf, t_v,t_u)
        G = self.evalInProd(iSurf, t_v,t_v)
        # Compute the components of the second fundamental form of a parameteric
        # surface
        L = self.evalInProd(iSurf, t_uu,n_hat)
        M = self.evalInProd(iSurf, t_uv,n_hat)
        N = self.evalInProd(iSurf, t_vv,n_hat)
        # Compute Gaussian and mean curvature (K and H)
        K = (L*N-M*M)/(E*G-F*F)
        H = (E*N - 2*F*M + G*L)/(2*(E*G-F*F))
        # Compute the combined curvature (C)
        C = 4.0*H*H-2.0*K
        # Assign integration weights for each point
        # 1   for center nodes
        # 1/2 for edge nodes
        # 1/4 for corner nodes
        wt = numpy.zeros_like(n_norm)+1
        wt[self.node_map[iSurf][0,:]] *= 0.5
        wt[self.node_map[iSurf][-1,:]] *= 0.5
        wt[self.node_map[iSurf][:,0]] *= 0.5
        wt[self.node_map[iSurf][:,-1]] *= 0.5
        # Compute discrete area associated with each node
        dS = wt*n_norm
        one = numpy.ones(self.node_map[iSurf].size)

        if self.curvatureType == 'Gaussian':
            # Now compute integral (K**2) over S, equivelent to sum(K**2*dS)
            kS = numpy.dot(one,K*K*dS)
            return [kS, K, H, C]
        elif self.curvatureType == 'mean':
            # Now compute integral (H**2) over S, equivelent to sum(H**2*dS)
            hS = numpy.dot(one,H*H*dS)
            return [hS, K, H, C]
        elif self.curvatureType == 'combined':
            # Now compute integral C over S, equivelent to sum(C*dS)
            cS = numpy.dot(one,C*dS)
            return [cS, K, H, C]
        elif self.curvatureType == 'KSmean':
            # Now compute the KS function for mean curvature, equivelent to KS(H*H*dS)
            sigmaH=numpy.dot(one,numpy.exp(self.KSCoeff*H*H*dS))
            KSmean=numpy.log(sigmaH)/self.KSCoeff
            if(MPI.COMM_WORLD.rank==0):
                print("Max curvature: ",max(H*H*dS))
            return [KSmean, K, H, C]
        else:
            raise Error("The curvatureType parameter should be Gaussian, mean, or combined, "
                        "%s is not supported!"%curvatureType)


    def evalCurvAreaSens(self, iSurf):
        '''
        Compute sensitivity of the integral K**2 wrt the coordinate
        locations X
        '''
        # Evaluate the derivitive of the position vector of every point on the
        # surface wrt to the parameteric corrdinate u and v
        t_u = self.evalDiff(iSurf, self.X[iSurf], 'u')
        Dt_uDX = self.evalDiffSens(iSurf, 'u')
        t_v = self.evalDiff(iSurf, self.X[iSurf], 'v')
        Dt_vDX = self.evalDiffSens(iSurf,'v')
        # Compute the normal vector by taking the cross product of t_u and t_v
        n = self.evalCross(iSurf,t_u,t_v)
        [DnDt_u, DnDt_v] = self.evalCrossSens(iSurf,t_u,t_v)
        DnDX = DnDt_u.dot(Dt_uDX) + DnDt_v.dot(Dt_vDX)
        # Compute the norm of tu_ x tv
        n_norm = self.evalNorm(iSurf,n)
        Dn_normDn = self.evalNormSens(iSurf,n)
        Dn_normDX = Dn_normDn.dot(DnDX)
        # Normalize the normal vector
        n_hat = numpy.zeros_like(n)
        n_hat[self.X_map[iSurf][:,:,0]]=n[self.X_map[iSurf][:,:,0]]/n_norm[self.node_map[iSurf][:,:]]
        n_hat[self.X_map[iSurf][:,:,1]]=n[self.X_map[iSurf][:,:,1]]/n_norm[self.node_map[iSurf][:,:]]
        n_hat[self.X_map[iSurf][:,:,2]]=n[self.X_map[iSurf][:,:,2]]/n_norm[self.node_map[iSurf][:,:]]

        ii = []
        data = []
        for i in range(3):
            # Dn_hat[self.X_map[iSurf][:,:,i]]/Dn[self.X_map[iSurf][:,:,i]]
            ii += list(numpy.reshape(self.X_map[iSurf][:,:,i],-1))
            data +=list(numpy.reshape(n_norm[self.node_map[iSurf][:,:]]**-1,-1))
        Dn_hatDn = csr_matrix((data,[ii,ii]),shape=(self.X[iSurf].size,self.X[iSurf].size))

        ii = []
        jj = []
        data = []
        for i in range(3):
            # Dn_hat[self.X_map[iSurf][:,:,i]]/Dn_norm[self.node_map[iSurf][:,:]]
            ii += list(numpy.reshape(self.X_map[iSurf][:,:,i],-1))
            jj += list(numpy.reshape(self.node_map[iSurf][:,:],-1))
            data +=list(numpy.reshape(-n[self.X_map[iSurf][:,:,i]]/(n_norm[self.node_map[iSurf][:,:]]**2),-1))
        Dn_hatDn_norm = csr_matrix((data,[ii,jj]),shape=(n_hat.size,n_norm.size))

        Dn_hatDX=Dn_hatDn.dot(DnDX)+Dn_hatDn_norm.dot(Dn_normDX)
        # Evaluate the second derivitives of the position vector wrt u and v
        t_uu = self.evalDiff(iSurf,t_u, 'u')
        Dt_uuDt_u = self.evalDiffSens(iSurf,'u')
        Dt_uuDX = Dt_uuDt_u.dot(Dt_uDX)

        t_vv = self.evalDiff(iSurf,t_v, 'v')
        Dt_vvDt_v = self.evalDiffSens(iSurf,'v')
        Dt_vvDX = Dt_vvDt_v.dot(Dt_vDX)

        t_uv = self.evalDiff(iSurf,t_v, 'u')
        Dt_uvDt_v = self.evalDiffSens(iSurf,'u')
        Dt_uvDX = Dt_uvDt_v.dot(Dt_vDX)
        # Compute the components of the first fundamental form of a parameteric
        # surface
        E = self.evalInProd(iSurf,t_u,t_u)
        [DEDt_u, _] = self.evalInProdSens(iSurf,t_u,t_u)
        DEDt_u*=2
        DEDX = DEDt_u.dot(Dt_uDX)

        F = self.evalInProd(iSurf,t_v,t_u)
        [DFDt_v, DFDt_u] = self.evalInProdSens(iSurf,t_v,t_u)
        DFDX = DFDt_v.dot(Dt_vDX) + DFDt_u.dot(Dt_uDX)

        G = self.evalInProd(iSurf,t_v,t_v)
        [DGDt_v, _] = self.evalInProdSens(iSurf,t_v,t_v)
        DGDt_v*=2
        DGDX = DGDt_v.dot(Dt_vDX)

        # Compute the components of the second fundamental form of a parameteric
        # surface
        L = self.evalInProd(iSurf,t_uu,n_hat)
        [DLDt_uu, DLDn_hat] = self.evalInProdSens(iSurf,t_uu,n_hat)
        DLDX = DLDt_uu.dot(Dt_uuDX)+DLDn_hat.dot(Dn_hatDX)

        M = self.evalInProd(iSurf,t_uv,n_hat)
        [DMDt_uv, DMDn_hat] = self.evalInProdSens(iSurf,t_uv,n_hat)
        DMDX = DMDt_uv.dot(Dt_uvDX)+DMDn_hat.dot(Dn_hatDX)

        N = self.evalInProd(iSurf,t_vv,n_hat)
        [DNDt_vv, DNDn_hat] = self.evalInProdSens(iSurf,t_vv,n_hat)
        DNDX = DNDt_vv.dot(Dt_vvDX)+DNDn_hat.dot(Dn_hatDX)

        # Compute Gaussian and mean curvature (K and H)
        K = (L*N-M*M)/(E*G-F*F)
        DKDE = self.diags(-(L*N-M*M)/(E*G-F*F)**2*G)
        DKDF = self.diags((L*N-M*M)/(E*G-F*F)**2*2*F)
        DKDG = self.diags(-(L*N-M*M)/(E*G-F*F)**2*E)
        DKDL = self.diags(N/(E*G-F*F))
        DKDM = self.diags(2*M/(E*G-F*F))
        DKDN = self.diags(L/(E*G-F*F))
        DKDX = DKDE.dot(DEDX) + DKDF.dot(DFDX) + DKDG.dot(DGDX) +\
               DKDL.dot(DLDX) + DKDM.dot(DMDX) + DKDN.dot(DNDX)

        H = (E*N - 2*F*M + G*L)/(2*(E*G-F*F))
        DHDE = self.diags(N/(2*(E*G-F*F)) - (E*N - 2*F*M + G*L)/(2*(E*G-F*F))**2*2*G)
        DHDF = self.diags(-2*M/(2*(E*G-F*F)) + (E*N - 2*F*M + G*L)/(2*(E*G-F*F))**2*4*F)
        DHDG = self.diags(L/(2*(E*G-F*F)) - (E*N - 2*F*M + G*L)/(2*(E*G-F*F))**2*2*E)
        DHDL = self.diags(G/(2*(E*G-F*F)))
        DHDM = self.diags(-2*F/(2*(E*G-F*F)))
        DHDN = self.diags(E/(2*(E*G-F*F)))
        DHDX = DHDE.dot(DEDX) + DHDF.dot(DFDX) + DHDG.dot(DGDX)+\
               DHDL.dot(DLDX) + DHDM.dot(DMDX) + DHDN.dot(DNDX)

        # Assign integration weights for each point
        # 1   for center nodes
        # 1/2 for edge nodes
        # 1/4 for corner nodes
        wt = numpy.zeros_like(n_norm)+1
        wt[self.node_map[iSurf][0,:]] *= 0.5
        wt[self.node_map[iSurf][-1,:]] *= 0.5
        wt[self.node_map[iSurf][:,0]] *= 0.5
        wt[self.node_map[iSurf][:,-1]] *= 0.5
        #Compute discrete area associated with each node
        dS = wt*n_norm
        DdSDX = self.diags(wt).dot(Dn_normDX)

        one = numpy.ones(self.node_map[iSurf].size)

        if self.curvatureType == 'Gaussian':
            # Now compute integral (K**2) over S, equivelent to sum(K**2*dS)
            kS = numpy.dot(one,K*K*dS)
            DkSDX = (self.diags(2*K*dS).dot(DKDX)+self.diags(K*K).dot(DdSDX)).T.dot(one)
            return DkSDX
        elif self.curvatureType == 'mean':
            # Now compute integral (H**2) over S, equivelent to sum(H**2*dS)
            hS = numpy.dot(one,H*H*dS)
            DhSDX = (self.diags(2*H*dS).dot(DHDX)+self.diags(H*H).dot(DdSDX)).T.dot(one)
            return DhSDX
        elif self.curvatureType == 'combined':
            # Now compute dcSDX. Note: cS= sum( (4*H*H-2*K)*dS ), DcSDX = term1 - term2
            # where term1 = sum( 8*H*DHDX*dS + 4*H*H*DdSdX ), term2 = sum( 2*DKDX*dS + 2*K*DdSdX )
            term1 = (self.diags(8*H*dS).dot(DHDX)+self.diags(4*H*H).dot(DdSDX)).T.dot(one)
            term2 = (self.diags(2*dS).dot(DKDX)+self.diags(2*K).dot(DdSDX)).T.dot(one)
            DcSDX = term1 - term2
            return DcSDX
        elif self.curvatureType == 'KSmean':
            sigmaH=numpy.dot(one,numpy.exp(self.KSCoeff*H*H*dS))
            DhSDX = (self.diags(2*H*dS/sigmaH*numpy.exp(self.KSCoeff*H*H*dS)).dot(DHDX)+self.diags(H*H/sigmaH*numpy.exp(self.KSCoeff*H*H*dS)).dot(DdSDX)).T.dot(one)
            return DhSDX
        else:
            raise Error("The curvatureType parameter should be Gaussian, mean, or combined, "
                        "%s is not supported!"%curvatureType)

    def evalCross(self, iSurf, u, v):
        '''
        Evaluate the cross product of two vector fields on the surface
        (n = u x v)
        '''
        n = numpy.zeros_like(self.X[iSurf])
        n[self.X_map[iSurf][:,:,0]] = u[self.X_map[iSurf][:,:,1]]*v[self.X_map[iSurf][:,:,2]] - u[self.X_map[iSurf][:,:,2]]*v[self.X_map[iSurf][:,:,1]]
        n[self.X_map[iSurf][:,:,1]] = -u[self.X_map[iSurf][:,:,0]]*v[self.X_map[iSurf][:,:,2]] + u[self.X_map[iSurf][:,:,2]]*v[self.X_map[iSurf][:,:,0]]
        n[self.X_map[iSurf][:,:,2]] = u[self.X_map[iSurf][:,:,0]]*v[self.X_map[iSurf][:,:,1]] - u[self.X_map[iSurf][:,:,1]]*v[self.X_map[iSurf][:,:,0]]
        return n

    def evalCrossSens(self, iSurf, u, v):
        '''
        Evaluate sensitivity of cross product wrt to the input vectors u and v
        (DnDu, DnDv)
        '''
        # Compute sensitivity wrt v
        ii = []
        jj = []
        data = []
        # Dn[self.X_map[iSurf][:,:,0]]/Dv[self.X_map[iSurf][:,:,2]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,0],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,2],-1))
        data += list(numpy.reshape(u[self.X_map[iSurf][:,:,1]],-1))
        # Dn[self.X_map[iSurf][:,:,0]]/Dv[self.X_map[iSurf][:,:,1]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,0],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,1],-1))
        data += list(numpy.reshape(-u[self.X_map[iSurf][:,:,2]],-1))
        # Dn[self.X_map[iSurf][:,:,1]]/Dv[self.X_map[iSurf][:,:,2]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,1],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,2],-1))
        data += list(numpy.reshape(-u[self.X_map[iSurf][:,:,0]],-1))
        # Dn[self.X_map[iSurf][:,:,1]]/Dv[self.X_map[iSurf][:,:,0]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,1],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,0],-1))
        data += list(numpy.reshape(u[self.X_map[iSurf][:,:,2]],-1))
        # Dn[self.X_map[iSurf][:,:,2]]/Dv[self.X_map[iSurf][:,:,1]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,2],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,1],-1))
        data += list(numpy.reshape(u[self.X_map[iSurf][:,:,0]],-1))
        # Dn[self.X_map[iSurf][:,:,2]]/Dv[self.X_map[iSurf][:,:,0]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,2],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,0],-1))
        data += list(numpy.reshape(-u[self.X_map[iSurf][:,:,1]],-1))


        DnDv = csr_matrix((data,[ii,jj]),shape=(self.X[iSurf].size,self.X[iSurf].size))
        # Now wrt v
        ii = []
        jj = []
        data = []
        # Dn[self.X_map[iSurf][:,:,0]]/Du[self.X_map[iSurf][:,:,1]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,0],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,1],-1))
        data += list(numpy.reshape(v[self.X_map[iSurf][:,:,2]],-1))
        # Dn[self.X_map[iSurf][:,:,0]]/Du[self.X_map[iSurf][:,:,2]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,0],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,2],-1))
        data += list(numpy.reshape(-v[self.X_map[iSurf][:,:,1]],-1))
        # Dn[self.X_map[iSurf][:,:,1]]/Du[self.X_map[iSurf][:,:,0]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,1],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,0],-1))
        data += list(numpy.reshape(-v[self.X_map[iSurf][:,:,2]],-1))
        # Dn[self.X_map[iSurf][:,:,1]]/Du[self.X_map[iSurf][:,:,2]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,1],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,2],-1))
        data += list(numpy.reshape(v[self.X_map[iSurf][:,:,0]],-1))
        # Dn[self.X_map[iSurf][:,:,2]]/Du[self.X_map[iSurf][:,:,0]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,2],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,0],-1))
        data += list(numpy.reshape(v[self.X_map[iSurf][:,:,1]],-1))
        # Dn[self.X_map[iSurf][:,:,2]]/Du[self.X_map[iSurf][:,:,1]]
        ii += list(numpy.reshape(self.X_map[iSurf][:,:,2],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,1],-1))
        data += list(numpy.reshape(-v[self.X_map[iSurf][:,:,0]],-1))


        DnDu = csr_matrix((data,[ii,jj]),shape=(self.X[iSurf].size,self.X[iSurf].size))
        return [DnDu, DnDv]

    def evalNorm(self, iSurf, u):
        '''
        Evaluate the norm of vector field on the surface
         (u o u)**1/2
        '''
        u_norm = numpy.zeros(self.X[iSurf].size//3)
        u_norm[self.node_map[iSurf][:,:]] = numpy.sqrt(u[self.X_map[iSurf][:,:,0]]**2 + \
            u[self.X_map[iSurf][:,:,1]]**2 + u[self.X_map[iSurf][:,:,2]]**2)
        return u_norm

    def evalNormSens(self, iSurf, u):
        '''
        Evaluate the sensitivity of the norm wrt input vector u
        '''
        u_norm = numpy.zeros(self.X[iSurf].size//3)
        u_norm[self.node_map[iSurf][:,:]] = numpy.sqrt(u[self.X_map[iSurf][:,:,0]]**2 + \
            u[self.X_map[iSurf][:,:,1]]**2 + u[self.X_map[iSurf][:,:,2]]**2)
        ii = []
        jj = []
        data = []
        # Du_norm[self.node_map[iSurf][:,:]]Du[self.X_map[iSurf][:,:,0]]
        ii += list(numpy.reshape(self.node_map[iSurf][:,:],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,0],-1))
        data += list(numpy.reshape(u[self.X_map[iSurf][:,:,0]]/u_norm[self.node_map[iSurf][:,:]],-1))

        # Du_norm[self.node_map[iSurf][:,:]]Du[self.X_map[iSurf][:,:,1]]
        ii += list(numpy.reshape(self.node_map[iSurf][:,:],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,1],-1))
        data += list(numpy.reshape(u[self.X_map[iSurf][:,:,1]]/u_norm[self.node_map[iSurf][:,:]],-1))

        # Du_norm[self.node_map[iSurf][:,:]]Du[self.X_map[iSurf][:,:,2]]
        ii += list(numpy.reshape(self.node_map[iSurf][:,:],-1))
        jj += list(numpy.reshape(self.X_map[iSurf][:,:,2],-1))
        data += list(numpy.reshape(u[self.X_map[iSurf][:,:,2]]/u_norm[self.node_map[iSurf][:,:]],-1))

        Du_normDu = csr_matrix((data,[ii,jj]),shape=(u_norm.size,self.X[iSurf].size))
        return Du_normDu

    def evalInProd(self, iSurf, u, v):
        '''
        Evaluate the inner product of two vector fields on the surface
        (ip = u o v)
        '''
        ip = numpy.zeros(self.node_map[iSurf].size)
        for i in range(3):
            ip[self.node_map[iSurf][:,:]] += u[self.X_map[iSurf][:,:,i]]*v[self.X_map[iSurf][:,:,i]]
        return ip

    def evalInProdSens(self, iSurf, u, v):
        '''
        Evaluate sensitivity of inner product wrt to the input vectors u and v
        (DipDu, DipDv)
        '''
        ii = []
        jj = []
        data = []
        for i in range(3):
            # Dip[node_map[:,:]]/Du[self.X_map[iSurf][:,:,i]]
            ii += list(numpy.reshape(self.node_map[iSurf][:,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][:,:,i],-1))
            data += list(numpy.reshape(v[self.X_map[iSurf][:,:,i]],-1))
        DipDu = csr_matrix((data,[ii,jj]),shape=(self.node_map[iSurf].size,self.X_map[iSurf].size))
        ii = []
        jj = []
        data = []
        for i in range(3):
            # Dip[node_map[:,:]]/Dv[self.X_map[iSurf][:,:,i]]
            ii += list(numpy.reshape(self.node_map[iSurf][:,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][:,:,i],-1))
            data += list(numpy.reshape(u[self.X_map[iSurf][:,:,i]],-1))
        DipDv = csr_matrix((data,[ii,jj]),shape=(self.node_map[iSurf].size,self.X_map[iSurf].size))
        return [DipDu, DipDv]

    def evalDiff(self, iSurf, v, wrt):
        '''
        Diferentiate vector field v wrt the parameteric coordinate u or v.
        Second order accurate. Central difference for nodes in the center
        forward/backward difference for nodes on the edge
        '''
        v_wrt = numpy.zeros_like(v)
        if wrt == 'u':
            v_wrt[self.X_map[iSurf][1:-1,:,:]]=(v[self.X_map[iSurf][2:,:,:]]-v[self.X_map[iSurf][0:-2,:,:]])/2.0
            v_wrt[self.X_map[iSurf][0,:,:]]=(-1*v[self.X_map[iSurf][2,:,:]]+4*v[self.X_map[iSurf][1,:,:]]-3*v[self.X_map[iSurf][0,:,:]])/2.0
            v_wrt[self.X_map[iSurf][-1,:,:]]=-(-1*v[self.X_map[iSurf][-3,:,:]]+4*v[self.X_map[iSurf][-2,:,:]]-3*v[self.X_map[iSurf][-1,:,:]])/2.0
        elif wrt == 'v':
            v_wrt[self.X_map[iSurf][:,1:-1,:]]=(v[self.X_map[iSurf][:,2:,:]]-v[self.X_map[iSurf][:,0:-2,:]])/2.0
            v_wrt[self.X_map[iSurf][:,0,:]]=(-1*v[self.X_map[iSurf][:,2,:]]+4*v[self.X_map[iSurf][:,1,:]]-3*v[self.X_map[iSurf][:,0,:]])/2.0
            v_wrt[self.X_map[iSurf][:,-1,:]]=-(-1*v[self.X_map[iSurf][:,-3,:]]+4*v[self.X_map[iSurf][:,-2,:]]-3*v[self.X_map[iSurf][:,-1,:]])/2.0
        return v_wrt

    def evalDiffSens(self, iSurf, wrt):
        '''
        Compute sensitivity of v_wrt with respect to input vector fiel v
        (Dv_wrt/Dv)
        '''
        ii = []
        jj = []
        data = []
        if wrt == 'u':
            # Central Difference

            # Dt_u[X_map[1:-1,:,:]]/DX[X_map[2:,:,:]] = 1/2
            ii += list(numpy.reshape(self.X_map[iSurf][1:-1,:,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][2:,:,:],-1))
            data+=[0.5]*len(numpy.reshape(self.X_map[iSurf][1:-1,:,:],-1))

            # Dt_u[X_map[1:-1,:,:]]/DX[X_map[0:-2,:,:]] = -1/2
            ii += list(numpy.reshape(self.X_map[iSurf][1:-1,:,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][0:-2,:,:],-1))
            data+=[-0.5]*len(numpy.reshape(self.X_map[iSurf][1:-1,:,:],-1))

            # Forward Difference

            # Dt_u[X_map[0,:,:]]/DX[X_map[2,:,:]] = -1/2
            ii += list(numpy.reshape(self.X_map[iSurf][0,:,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][2,:,:],-1))
            data+=[-0.5]*len(numpy.reshape(self.X_map[iSurf][0,:,:],-1))

            # Dt_u[X_map[0,:,:]]/DX[X_map[1,:,:]] = 4/2
            ii += list(numpy.reshape(self.X_map[iSurf][0,:,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][1,:,:],-1))
            data+=[2]*len(numpy.reshape(self.X_map[iSurf][0,:,:],-1))

            # Dt_u[X_map[0,:,:]]/DX[X_map[0,:,:]] = -3/2
            ii += list(numpy.reshape(self.X_map[iSurf][0,:,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][0,:,:],-1))
            data+=[-1.5]*len(numpy.reshape(self.X_map[iSurf][0,:,:],-1))

            # Backward Difference

            # Dt_u[X_map[-1,:,:]]/DX[X_map[-3,:,:]] = 1/2
            ii += list(numpy.reshape(self.X_map[iSurf][-1,:,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][-3,:,:],-1))
            data+=[0.5]*len(numpy.reshape(self.X_map[iSurf][-1,:,:],-1))

            # Dt_u[X_map[-1,:,:]]/DX[X_map[-2,:,:]] = -4/2
            ii += list(numpy.reshape(self.X_map[iSurf][-1,:,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][-2,:,:],-1))
            data+=[-2.0]*len(numpy.reshape(self.X_map[iSurf][-2,:,:],-1))

            # Dt_u[X_map[-1,:,:]]/DX[X_map[-1,:,:]] = 3/2
            ii += list(numpy.reshape(self.X_map[iSurf][-1,:,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][-1,:,:],-1))
            data+=[1.5]*len(numpy.reshape(self.X_map[iSurf][-1,:,:],-1))

        elif wrt == 'v':
            # Central Difference

            # Dt_u[X_map[:,1:-1,:]]/DX[X_map[:,2:,:]] = 1/2
            ii += list(numpy.reshape(self.X_map[iSurf][:,1:-1,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][:,2:,:],-1))
            data+=[0.5]*len(numpy.reshape(self.X_map[iSurf][:,1:-1,:],-1))

            # Dt_u[X_map[:,1:-1,:]]/DX[X_map[:,0:-2,:]] = -1/2
            ii += list(numpy.reshape(self.X_map[iSurf][:,1:-1,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][:,0:-2,:],-1))
            data+=[-0.5]*len(numpy.reshape(self.X_map[iSurf][:,1:-1,:],-1))

            # Forward Difference

            # Dt_u[X_map[:,0,:]]/DX[X_map[:,2,:]] = -1/2
            ii += list(numpy.reshape(self.X_map[iSurf][:,0,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][:,2,:],-1))
            data+=[-0.5]*len(numpy.reshape(self.X_map[iSurf][:,0,:],-1))

            # Dt_u[X_map[:,0,:]]/DX[X_map[:,1,:]] = 4/2
            ii += list(numpy.reshape(self.X_map[iSurf][:,0,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][:,1,:],-1))
            data+=[2]*len(numpy.reshape(self.X_map[iSurf][:,0,:],-1))

            # Dt_u[X_map[:,0,:]]/DX[X_map[:,0,:]] = -3/2
            ii += list(numpy.reshape(self.X_map[iSurf][:,0,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][:,0,:],-1))
            data+=[-1.5]*len(numpy.reshape(self.X_map[iSurf][:,0,:],-1))

            # Backward Difference

            # Dt_u[X_map[:,-1,:]]/DX[X_map[:,-3,:]] = 1/2
            ii += list(numpy.reshape(self.X_map[iSurf][:,-1,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][:,-3,:],-1))
            data+=[0.5]*len(numpy.reshape(self.X_map[iSurf][:,-1,:],-1))

            # Dt_u[X_map[:,-1,:]]/DX[X_map[:,-2,:]] = -4/2
            ii += list(numpy.reshape(self.X_map[iSurf][:,-1,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][:,-2,:],-1))
            data+=[-2.0]*len(numpy.reshape(self.X_map[iSurf][:,-2,:],-1))

            # Dt_u[X_map[:,-1,:]]/DX[X_map[:,-1,:]] = 3/2
            ii += list(numpy.reshape(self.X_map[iSurf][:,-1,:],-1))
            jj += list(numpy.reshape(self.X_map[iSurf][:,-1,:],-1))
            data+=[1.5]*len(numpy.reshape(self.X_map[iSurf][:,-1,:],-1))

        Dv_uDX = csr_matrix((data,[ii,jj]),shape=(self.X[iSurf].size,self.X[iSurf].size))

        return Dv_uDX

    def diags(self, a):
        '''
        A standard vectorized sparse diagnal matrix function. Similar to the above \
        function
        some versions of scipy don't have this function, so this is here to prevent\

        potential import problems.
        '''
        ii=range(len(a))
        return csr_matrix((a,[ii,ii]),(len(a),len(a)))

    def writeTecplot(self, handle1):
        '''
        Write Curvature data on the surface to a tecplot file. Data includes
        mean curvature, H, and Gaussian curvature, K.

        Input:

            tec_file: name of TecPlot file.
        '''
        # we ignore the input handle and use this separated name for curvature constraint tecplot file
        # NOTE: we use this tecplot file to only visualize the local distribution of curctures.
        # The plotted local curvatures are not exactly as that computed in the evalCurvArea function
        handle = open('%s.dat'%self.name,'w')
        handle.write('title = "DVConstraint curvature constraint"\n')
        varbs='variables = "x", "y", "z", "K", "H" "C"'
        handle.write(varbs+'\n')
        for iSurf in range(self.nSurfs):
            [_,K,H,C] = self.evalCurvArea(iSurf)
            handle.write('Zone T=%s_%d\n'% (self.name,iSurf))

            handle.write('Nodes = %d, Elements = %d, f=fepoint, et=quadrilateral\n'% (
                len(self.coords[iSurf]), (self.X_map[iSurf].shape[0]-1)*(self.X_map[iSurf].shape[1]-1)))
            for i in range(self.X_map[iSurf].shape[0]):
                for j in range(self.X_map[iSurf].shape[1]):
                    handle.write('%E %E %E %E %E %E\n'% (self.X[iSurf][self.X_map[iSurf][i, j, 0]], self.X[iSurf][self.X_map[iSurf][i, j, 1]],
                                            self.X[iSurf][self.X_map[iSurf][i, j, 2]],K[self.node_map[iSurf][i,j]],H[self.node_map[iSurf][i,j]],C[self.node_map[iSurf][i,j]]))
            handle.write('\n')
            for i in range(self.X_map[iSurf].shape[0]-1):
                for j in range(self.X_map[iSurf].shape[1]-1):
                    handle.write('%d %d %d %d\n'% (self.node_map[iSurf][i,j]+1, self.node_map[iSurf][i+1,j]+1,self.node_map[iSurf][i+1,j+1]+1,self.node_map[iSurf][i,j+1]+1))
        handle.close()

class GlobalLinearConstraint(object):
    """
    This class is used to represent a set of generic set of linear
    constriants coupling global design variables together.
    """
    def __init__(self, name, key, type, options, lower, upper, DVGeo, config):
        # No error checking here since the calling routine should have
        # already done it.
        self.name = name
        self.key = key
        self.type = type
        self.lower = lower
        self.upper = upper
        self.DVGeo = DVGeo
        self.ncon = 0
        self.jac = {}
        self.config = config
        if self.type == 'monotonic':
            self.setMonotonic(options)

    def evalFunctions(self, funcs):
        """
        Evaluate the function this object has and place in the funcs
        dictionary. Note that this function typically will not need to
        called since these constraints are supplied as a linear
        constraint jacobian they constraints themselves need to be
        revaluated.

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        cons = []
        for key in self.jac:
            cons.extend(self.jac[key].dot(self.DVGeo.DV_listGlobal[key].value))

        funcs[self.name] = numpy.array(cons).real.astype('d')

    def evalFunctionsSens(self, funcsSens):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        funcsSens[self.name] = self.jac

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt. These constraints are added as
        linear constraints.
        """
        if self.ncon > 0:
            for key in self.jac:
                optProb.addConGroup(self.name+'_'+key, self.jac[key].shape[0],
                                    lower=self.lower, upper=self.upper, scale=1.0,
                                    linear=True, wrt=key, jac={key:self.jac[key]})
    def setMonotonic(self, options):
        """
        Set up monotonicity jacobian for the given global design variable
        """
        self.vizConIndices = {}

        if self.config is None or self.config in self.DVGeo.DV_listGlobal[self.key].config:
            ndv = self.DVGeo.DV_listGlobal[self.key].nVal
            start = options['start']
            stop = options['stop']
            if stop == -1:
                stop = ndv

            # Since start and stop are inclusive, we need to add one to stop to
            # account for python indexing
            stop += 1
            ncon = len(numpy.zeros(ndv)[start:stop]) - 1

            jacobian = numpy.zeros((ncon, ndv))
            slope = options['slope']
            for i in range(ncon):
                jacobian[i, start+i] = 1.0*slope
                jacobian[i, start+i+1] = -1.0*slope
            self.jac[self.key] = jacobian
            self.ncon += ncon

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of lete constraints
        to the open file handle
        """

        pass
