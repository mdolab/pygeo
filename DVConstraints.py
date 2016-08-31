# ======================================================================
#         Imports
# ======================================================================
from __future__ import print_function
import numpy
from pygeo import geo_utils
from pyspline import pySpline
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

    """
  
    def __init__(self):
        """
        Create a (empty) DVconstrains object. Specific types of
        constraints will added individually
        """
        self.thickCon = OrderedDict()
        self.locCon = OrderedDict()
        self.volumeCon = OrderedDict()
        self.linearCon = OrderedDict()
        self.volumeCGCon = OrderedDict()
        self.volumeAreaCon = OrderedDict()
        self.gearCon = OrderedDict()
        self.circCon = OrderedDict()
        self.surfAreaCon = OrderedDict()

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
            form pySUmb or from pyTrian.

        Examples
        --------
        >>> CFDsolver = SUMB(comm=comm, options=aeroOptions)
        >>> surf = CFDsolver.getTriangulatedMeshSurface()
        >>> DVCon.setSurface(surf)
        >>> # Or using a pyGeo surface object:
        >>> surf = pyGeo('iges',fileName='wing.igs')
        >>> DVCon.setSurface(surf)

        """
        
        if type(surf) == list:
            self.p0 = numpy.array(surf[0])
            self.v1 = numpy.array(surf[1])
            self.v2 = numpy.array(surf[2])
        else:
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

    def addThicknessConstraints2D(self, leList, teList, nSpan, nChord,
                                  lower=1.0, upper=3.0, scaled=True, scale=1.0,
                                  name=None, addToPyOpt=True):
        """
        Add a set of thickness constraints that span a logically a
        two-dimensional region. A little ASCII art can help here::

          Planform view of the wing: The '+' are the (three dimensional)
          points that are supplied in leList and teList. 

          Physical extent of wing            
                                   \         
          __________________________\_________
          |                                  |   
          +-----------+--o-----------+       |
          |   /                       \      |
          | leList      teList         \     | 
          |                   \         \    |
          +------------------------------+   |
                                             |
          ___________________________________/


        Things to consider:

        * The region given by leList and teList must lie completely
          inside the wing

        * The number of points in leList and teList do not need to be
          the same length.

        * The leading and trailing edges are approximated using
          2-order splines (line segments) and nSpan points are
          interpolated in a linear fashion. Note that the a thickness
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
          entirely line in a plane. This ensure that the projection vectors
          are always exactly normal to this plane.

        * If the surface formed by leList and teList is NOT precisely
          normal, issues can arise near the end of a opensurface (ie
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

        # Create a name 
        if name is None:
            conName = 'thickness_constraints_%d'% len(self.thickCon)
        else:
            conName = name
        self.thickCon[conName] = ThicknessConstraint(
            conName, coords, lower, upper, scaled, scale, self.DVGeo,
            addToPyOpt)
        

    def addThicknessConstraints1D(self, ptList, nCon, axis, 
                                  lower=1.0, upper=3.0, scaled=True,
                                  scale=1.0, name=None,
                                  addToPyOpt=True):
        """
        Add a set of thickness constraints oriented along a poly-line.

        See below for a schematic::

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
            if fail:
                raise Error("There was an error projecting a node "
                            "at (%f, %f, %f) with normal (%f, %f, %f)."% ( 
                                X[i, 0], X[i, 1], X[i, 2], axis[0], axis[1], axis[2]))
            coords[i, 0] = up
            coords[i, 1] = down

        # Create the thickness constraint object:
        coords = coords.reshape((nCon*2, 3))
        if name is None:
            conName = 'thickness_constraints_%d'% len(self.thickCon)
        else:
            conName = name
        self.thickCon[conName] = ThicknessConstraint(
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

        if lower==None:
            lower = X.flatten()
        if upper==None:
            upper = X.flatten()

        # Create the location constraint object
        if name is None:
            conName = 'location_constraints_%d'% len(self.locCon)
        else:
            conName = name
        self.locCon[conName] = LocationConstraint(
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
            if fail:
                raise Error("There was an error projecting a node "
                            "at (%f, %f, %f) with normal (%f, %f, %f)."% ( 
                        X[i, 0], X[i, 1], X[i, 2], axis[0], axis[1], axis[2]))
            coords[i, 0] = up
            coords[i, 1] = down
        
        X = (1-bias)*coords[:, 1] + bias*coords[:, 0]

        # X is now what we want to constrain
        if lower==None:
            lower = X.flatten()
        if upper==None:
            upper = X.flatten()

        # Create the location constraint object
        if name is None:
            conName = 'location_constraints_%d'% len(self.locCon)
        else:
            conName = name
        self.locCon[conName] = LocationConstraint(
            conName, X, lower, upper, scaled, scale, self.DVGeo,
            addToPyOpt)

    def addThicknessToChordConstraints1D(self, ptList, nCon, axis, chordDir, 
                                         lower=1.0, upper=3.0, scale=1.0, 
                                         name=None, addToPyOpt=True):
        """
        Add a set of thickness-to-chord ratio constraints oriented along a poly-line.

        See below for a schematic::

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
        if name is None:
            conName = 'thickness_to_chord_constraints_%d'% len(self.thickCon)
        else:
            conName = name
        self.thickCon[conName] = ThicknessToChordConstraint(
            conName, coords, lower, upper, scale, self.DVGeo, addToPyOpt)

    def addVolumeConstraint(self, leList, teList, nSpan, nChord,
                            lower=1.0, upper=3.0, scaled=True, scale=1.0,
                            name=None, addToPyOpt=True):
        """
        Add a single volume constraint to the wing. The volume
        constraint is defined over a logically two-dimensional region
        as shown below::

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
                                             |
          ___________________________________/

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
        if name is None:
            conName = 'volume_constraint_%d'% len(self.volumeCon)
        else:
            conName = name

        coords = self._generateIntersections(leList, teList, nSpan, nChord)
        coords = coords.reshape((nSpan*nChord*2, 3))

        # Finally add the volume constraint object
        self.volumeCon[conName] = VolumeConstraint(
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
        if name is None:
            conName = 'composite_volume_constraint_%d'% len(self.volumeCon)
        else:
            conName = name

        # Determine the list of volume constraint objects
        volCons = []
        for vol in vols:
            try:
                volCons.append(self.volumeCon[vol])
            except KeyError:
                raise Error("The supplied volume name '%s' has not"
                            " already been added with a call to "
                            "addVolumeConstraint()"% vol)
        self.volumeCon[conName] = CompositeVolumeConstraint(
            conName, volCons, lower, upper, scaled, scale, self.DVGeo,
            addToPyOpt)

    def addLeTeConstraints(self, volID=None, faceID=None,
                           indSetA=None, indSetB=None, name=None, 
                           config=None):
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

        # Now determine what type of specification we have:
        if volID is not None and faceID is not None:
            lIndex = self.DVGeo.getLocalIndex(volID)
            if faceID.lower() == 'ilow':
                indices = lIndex[0, :, :]
            elif faceID.lower() == 'ihigh':
                indices = lIndex[-1, :, :]
            elif faceID.lower() == 'jlow':
                indices = lIndex[:, 0, :]
            elif faceID.lower() == 'jhigh':
                indices = lIndex[:, -1, :]
            elif faceID.lower() == 'klow':
                indices = lIndex[:, :, 0]
            elif faceID.lower() == 'khigh':
                indices = lIndex[:, :, -1]
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
                raise Error("Cannot add leading edge constraints. One (and "
                            "exactly one) of FFD block dimensions on the"
                            " specified face must be 2. The dimensions of "
                            "the selected face are: "
                            "(%d, %d)" % (shp[0], shp[1]))

        elif indSetA is not None and indSetB is not None:
            if len(indSetA) != len(indSetB):
                raise Error("The length of the supplied indices are not "
                            "the same length")
        else:
            raise Error("Incorrect data supplied to addLeTeConstraint. The "
                        "keyword arguments 'volID' and 'faceID' must be "
                        "specified **or** 'indSetA' and 'indSetB'")

        if name is None:
            conName = 'lete_constraint_%d'% len(self.linearCon)
        else:
            conName = name

        # Finally add the volume constraint object
        n = len(indSetA)
        self.linearCon[conName] = LinearConstraint(
            conName, indSetA, indSetB, numpy.ones(n), numpy.ones(n),
            lower=0, upper=0, DVGeo=self.DVGeo, config=config)

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
            conName = 'linear_constraint_%d'% len(self.linearCon)
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
        if name is None:
            conName = 'gear_constraint_%d'% len(self.gearCon)
        else:
            conName = name

        # Project the actual location we were give:
        up, down, fail = geo_utils.projectNode(
            position, axis, self.p0, self.v1, self.v2)
        if fail:
            raise Error("There was an error projecting a node "
                        "at (%f, %f, %f) with normal (%f, %f, %f)."% ( 
                            position))
        
        self.gearCon[conName] = GearPostConstraint(
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

        # Create a name 
        if name is None:
            conName = 'circularity_constraints_%d'% len(self.circCon)
        else:
            conName = name
        self.circCon[conName] = CircularityConstraint(
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
        
        if self.p0==None or self.v1 == None or self.v2 == None:
            raise Error("DVCon surface is not properly defined. Check that setSurface"
                        "is called.")
        
        self._checkDVGeo()

        # Create a name 
        if name is None:
            conName = 'surfaceArea_constraints_%d'% len(self.surfAreaCon)
        else:
            conName = name
        self.surfAreaCon[conName] = SurfaceAreaConstraint(
            conName, self.p0, self.v1, self.v2, lower, upper, scale, scaled, self.DVGeo,
            addToPyOpt)


    def _checkDVGeo(self):

        """check if DVGeo exists"""
        if self.DVGeo is None:
            raise Error("A DVGeometry object must be added to DVCon before "
                        "using a call to DVCon.setDVGeo(DVGeo) before "
                        "constraints can be added.")

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

        for key in self.thickCon:
            self.thickCon[key].addConstraintsPyOpt(optProb)
        for key in self.locCon:
            self.locCon[key].addConstraintsPyOpt(optProb)
        for key in self.volumeCon:
            self.volumeCon[key].addConstraintsPyOpt(optProb)
        for key in self.linearCon:
            self.linearCon[key].addConstraintsPyOpt(optProb)
        for key in self.gearCon:
            self.gearCon[key].addConstraintsPyOpt(optProb)
        for key in self.circCon:
            self.circCon[key].addConstraintsPyOpt(optProb)
        for key in self.surfAreaCon:
            self.surfAreaCon[key].addConstraintsPyOpt(optProb)

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

        for key in self.thickCon:
            self.thickCon[key].evalFunctions(funcs, config)
        for key in self.locCon:
            self.locCon[key].evalFunctions(funcs, config)
        for key in self.volumeCon:
            self.volumeCon[key].evalFunctions(funcs, config)
        for key in self.gearCon:
            self.gearCon[key].evalFunctions(funcs, config)
        for key in self.circCon:
            self.circCon[key].evalFunctions(funcs, config)
        for key in self.surfAreaCon:
            self.surfAreaCon[key].evalFunctions(funcs, config)
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
        for key in self.thickCon:
            self.thickCon[key].evalFunctionsSens(funcsSens, config)
        for key in self.locCon:
            self.locCon[key].evalFunctionsSens(funcsSens, config)
        for key in self.volumeCon:
            self.volumeCon[key].evalFunctionsSens(funcsSens, config)
        for key in self.gearCon:
            self.gearCon[key].evalFunctionsSens(funcsSens, config)
        for key in self.circCon:
            self.circCon[key].evalFunctionsSens(funcsSens, config)
        for key in self.surfAreaCon:
            self.surfAreaCon[key].evalFunctionsSens(funcsSens, config)

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

        for key in self.thickCon:
            self.thickCon[key].writeTecplot(f)
        for key in self.locCon:
            self.locCon[key].writeTecplot(f)
        for key in self.volumeCon:
            self.volumeCon[key].writeTecplot(f)
        for key in self.gearCon:
            self.gearCon[key].writeTecplot(f)
        for key in self.circCon:
            self.circCon[key].writeTecplot(f)
        for key in self.surfAreaCon:
            self.surfAreaCon[key].writeTecplot(f)
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
        # Generate all intersections:
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

                if fail:
                    raise Error('There was an error projecting a node \
                     at (%f, %f, %f) with normal (%f, %f, %f).'% (
                            X[i, j, 0], X[i, j, 1], X[i, j, 2],
                            upVec[0], upVec[1], upVec[2]))
                coords[i, j, 0] = up
                coords[i, j, 1] = down

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

        for i in xrange(nPts):
            newUnitVec = geo_utils.rotVbyW(zeroAxisOrtho, rotAxis, angles[i])
            newUnitVec*=radius
            coords[i,:] = newUnitVec+origin

        return coords

class ThicknessConstraint(object):
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

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addConGroup(self.name, self.nCon, lower=self.lower,
                                upper=self.upper, scale=self.scale,
                                wrt=self.DVGeo.getVarNames())

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

class LocationConstraint(object):
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

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addConGroup(self.name, self.nCon, lower=self.lower,
                                upper=self.upper, scale=self.scale,
                                wrt=self.DVGeo.getVarNames())

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
            handle.write('%d %d\n'% (i, i+1))


class ThicknessToChordConstraint(object):
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

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addConGroup(self.name, self.nCon, lower=self.lower,
                                upper=self.upper, scale=self.scale,
                                wrt=self.DVGeo.getVarNames())

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


class VolumeConstraint(object):
    """
    This class is used to represet a single volume constraint. The
    parameter list is explained in the addVolumeConstaint() of
    the DVConstraints class
    """

    def __init__(self, name, nSpan, nChord, coords, lower, upper, scaled,
                 scale, DVGeo, addToPyOpt):

        self.name = name
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
            
    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addCon(self.name, lower=self.lower, upper=self.upper,
                           scale=self.scale, wrt=self.DVGeo.getVarNames())

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
        V += self.volpym(x0, x2, x4, x6, p)
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
        self.volpym_b(x0, x2, x4, x6, p, x0b, x2b, x4b, x6b, pb)
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


class CompositeVolumeConstraint(object):
    """This class is used to represet a single volume constraints that is a
    group of other VolumeConstraints.
    """
    
    def __init__(self, name, vols, lower, upper, scaled, scale,
                 DVGeo, addToPyOpt):
        self.name = name
        self.vols = vols
        self.scaled = scaled
        self.lower = lower
        self.upper = upper
        self.scaled = scaled
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

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
                    
    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addCon(self.name, lower=self.lower, upper=self.upper,
                           scale=self.scale, wrt=self.DVGeo.getVarNames())
        
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
            cons.extend(self.jac[key].dot(self.DVGeo.DV_listLocal[key].value))
            
        funcs[self.name] = numpy.array(cons).astype('d')
        
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
        
        for key in self.DVGeo.DV_listLocal:
             if self.config is None or self.config in self.DVGeo.DV_listLocal[key].config:

                # Temp is the list of FFD coefficients that are included
                # as shape variables in this localDV "key"
                temp = self.DVGeo.DV_listLocal[key].coefList
                cons = []
                for j in range(len(self.indSetA)):
                    # Try to find this index # in the coefList (temp)
                    up = None
                    down = None

                    # Note: We are doing inefficient double looping here
                    for k in range(len(temp)):
                        if temp[k][0] == self.indSetA[j]:
                            up = k
                        if temp[k][0] == self.indSetB[j]:
                            down = k

                    # If we haven't found up AND down do nothing
                    if up is not None and down is not None:
                        cons.append([up, down])

                # end for (indSet loop)
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



class GearPostConstraint(object):
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

    def writeTecplot(self, handle):
        """
        Write the visualization of this volume constriant
        """
        pass

class CircularityConstraint(object):
    """
    DVConstraints representation of a set of circularity
    constraint. One of these objects is created each time a
    addCircularityConstraints call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, center, coords, lower, upper, scale, DVGeo,
                 addToPyOpt):
        self.name = name
        self.center = numpy.array(center).reshape((3,))
        self.coords = coords
        self.nCon = self.coords.shape[0]-1
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt
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

        #length = 0
#        for j in xrange(3):
            #length += (self.center[j]-self.coords[0,j])*(self.center[j]-self.coords[0,j])
        reflength2 = numpy.sum((self.center-self.coords[0,:])**2)
#        refLength = numpy.sqrt(length2)
        for i in xrange(self.nCon):
            # length = 0
            # for j in xrange(3):
            #     length += (self.center[j]-self.coords[i+1,j])*(self.center[j]-self.coords[i+1,j])
            length2 = numpy.sum((self.center-self.coords[i+1,:])**2)
            self.X[i] = numpy.sqrt(length2/reflength2)
 
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
                                  1,
                                  self.center.shape[0]))

            for con in xrange(self.nCon):
                reflength2 = 0
                for i in xrange(3):
                    reflength2 = reflength2 + (center[i]-coords[0,i])**2

                centerb = dLndCn[con,0,:]#0.0
                coordsb = dLndPt[con,:,:]#0.0
                reflength2b = 0.0
                for i in xrange(self.nCon):
                    length2 = 0
                    for j in xrange(3):
                        length2 = length2 + (center[j]-coords[i+1, j])**2

                    if (length2/reflength2 == 0.0):
                        tempb1 = 0.0
                    else:
                        tempb1 = xb[i]/(2.0*numpy.sqrt(length2/reflength2)*reflength2)

                    length2b = tempb1
                    reflength2b = reflength2b - length2*tempb1/reflength2
                    xb[i] = 0.0
                    for j in reversed(xrange(3)):#DO j=3,1,-1
                        tempb0 = 2*(center[j]-coords[i+1, j])*length2b
                        centerb[j] = centerb[j] + tempb0
                        coordsb[i+1, j] = coordsb[i+1, j] - tempb0
                for i in reversed(xrange(3)):#DO i=3,1,-1
                    tempb = 2*(center[i]-coords[0, i])*reflength2b
                    centerb[i] = centerb[i] + tempb
                    coordsb[0, i] = coordsb[0, i] - tempb
            
                
            tmpPt = self.DVGeo.totalSensitivity(dLndPt, self.name+'coords', config=config)
            tmpCn = self.DVGeo.totalSensitivity(dLndCn, self.name+'center', config=config)
        
            funcsSens[self.name] = tmpPt+tmpCn

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addConGroup(self.name, self.nCon, lower=self.lower,
                                upper=self.upper, scale=self.scale,
                                wrt=self.DVGeo.getVarNames())

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
            handle.write('%d %d\n'% (i, i+1))

        handle.write('Zone T=%s_center\n'% self.name)
        handle.write('Nodes = 2, Elements = 1 ZONETYPE=FELINESEG\n')
        handle.write('DATAPACKING=POINT\n')
        handle.write('%f %f %f\n'% (self.center[0], self.center[1],
                                    self.center[2]))
        handle.write('%f %f %f\n'% (self.center[0], self.center[1],
                                    self.center[2]))

class SurfaceAreaConstraint(object):
    """
    DVConstraints representation of a surface area
    constraint. One of these objects is created each time a
    addSurfaceAreaConstraints call is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, p0, v1, v2, lower, upper, scale,scaled, DVGeo,
                 addToPyOpt):
        self.name = name
        self.nCon = 1
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.scaled = scaled
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt
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

        # compute the refernece area
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
                                 self.p0.shape[0],
                                 self.p0.shape[1]))

            dAdp2 = numpy.zeros((self.nCon, 
                                 self.p0.shape[0],
                                 self.p0.shape[1]))

            for con in xrange(self.nCon):
                p0b = dAdp0[con,:,:]
                p1b = dAdp1[con,:,:]
                p2b = dAdp2[con,:,:]

                areasb = numpy.empty(self.n)
                if self.scaled:
                    areab = areab/self.X0
                areasb[:] = areab/2.
                for i in xrange(n):#DO i=1,n
                    # for j in xrange(3):#DO j=1,3
                    #     v1(i,j) = p1(i, j) - p0(i, j)
                    #     v2(i, j) = p2(i, j) - p0(i, j)
                    v1[i,:] = p1[i,:] - p0[i, :]
                    v2[i,:] = p2[i,:] - p0[i, :]

                    crosses[i, :] = numpy.cross(v1[i, :], v2[i, :])
                    # for j in xrange(3):
                    #     areas(i) = areas(i) + crosses(i, j)**2
                    areas[i] = numpy.sum(crosses[i, :]**2)
                    if (areas[i] == 0.0):
                        areasb[i] = 0.0
                    else:
                        areasb[i] = areasb[i]/(2.0*SQRT(areas[i]))

                    # for j in reversed(xrange(3)):#DO j=3,1,-1
                    #     crossesb(i, j) = crossesb(i, j) + 2*crosses(i, j)*areasb(i)

                    crossesb[i, :] = numpy.sum(2*crosses[i, :]*areasb[i])

                    v1b[i,:],v2b[i,:] = geo_utils.cross_b(v1[i, :], v2[i, :], crossesb[i, :])

                    # for j in reversed(xrange(3)):#DO j=3,1,-1
                    #      p2b(i, j) = p2b(i, j) + v2b(i, j)
                    #      p0b(i, j) = p0b(i, j) - v1b(i, j) - v2b(i, j)
                    #      v2b(i, j) = 0.0
                    #      p1b(i, j) = p1b(i, j) + v1b(i, j)
                    #      v1b(i, j) = 0.0
                    p2b[i, :] = v2b[i, :]
                    p0b[i, :] = - v1b[i, :] - v2b[i, :]
                    p1b[i, :] = p1b[i, :] + v1b[i, :]
            
                
            tmpp0 = self.DVGeo.totalSensitivity(dAdp0, self.name+'p0', config=config)
            tmpp1 = self.DVGeo.totalSensitivity(dAdp1, self.name+'p1', config=config)
            tmpp2 = self.DVGeo.totalSensitivity(dAdp2, self.name+'p2', config=config)

        
            funcsSens[self.name] = tmpp0 + tmpp1 + tmpp2

    def _computeArea(self, p0, p1, p2):
        """
        compute area based on three point arrays
        """
        # convert p1 and p2 to v1 and v2
        v1 = p1- p0
        v2 = p2- p0

        #compute the areas
        areaVec = numpy.cross(v1, v2)

        area = numpy.linalg.norm(areaVec,axis=1)
        
        return numpy.sum(area)/2.0

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addConGroup(self.name, self.nCon, lower=self.lower,
                                upper=self.upper, scale=self.scale,
                                wrt=self.DVGeo.getVarNames())

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write('Zone T=%s_surface\n'% self.name)
        handle.write('Nodes = %d, Elements = %d ZONETYPE=FETRIANGLE\n'% (
            3*self.n, self.n))
        handle.write('DATAPACKING=POINT\n')
        for i in xrange(self.n):
            handle.write('%f %f %f\n'% (self.p0[i, 0], self.p0[i, 1],
                                        self.p0[i, 2]))
        for i in xrange(self.n):
            handle.write('%f %f %f\n'% (self.p1[i, 0], self.p1[i, 1],
                                        self.p1[i, 2]))

        for i in xrange(self.n):
            handle.write('%f %f %f\n'% (self.p2[i, 0], self.p2[i, 1],
                                        self.p2[i, 2]))

        for i in range(self.n):
            handle.write('%d %d %d\n'% (i+1, i+self.n+1, i+self.n*2+1))


# class ProjectedAreaConstraint(object):
#     """
#     DVConstraints representation of a surface area
#     constraint. One of these objects is created each time a
#     addSurfaceAreaConstraints call is made.
#     The user should not have to deal with this class directly.
#     """

#     def __init__(self, name, p0, v1, v2,axis, lower, upper, scale, DVGeo,
#                  addToPyOpt):
#         self.name = name
#         self.center = numpy.array(center).reshape((3,))
#         self.coords = coords
#         self.nCon = len(self.coords.shape[0])-1
#         self.lower = lower
#         self.upper = upper
#         self.scale = scale
#         self.DVGeo = DVGeo
#         self.addToPyOpt = addToPyOpt
#         self.X = numpy.zeros(self.nCon)
#         self.axis = axis
        
#         # The first thing we do is convert v1 and v2 to coords
#         self.p0 = p0
#         self.p1 = v1+p0
#         self.p2 = v2+p0
#         # Now embed the coordinates into DVGeo
#         # with the name provided:
#         self.DVGeo.addPointSet(self.coords, self.name+'p0')
#         self.DVGeo.addPointSet(self.center, self.name+'p1')
#         self.DVGeo.addPointSet(self.center, self.name+'p2')
                
#     def evalFunctions(self, funcs, config):
#         """
#         Evaluate the functions this object has and place in the funcs dictionary

#         Parameters
#         ----------
#         funcs : dict
#             Dictionary to place function values
#         """
#         # Pull out the most recent set of coordinates:
#         self.p0 = self.DVGeo.update(self.name+'p0', config=config)
#         self.p1 = self.DVGeo.update(self.name+'p1', config=config)
#         self.p2 = self.DVGeo.update(self.name+'p2', config=config)

#         # convert p1 and p2 to v1 and v2
#         v1 = self.p1- self.p0
#         v2 = self.p2- self.p0

#         #compute the areas
#         areaVec = numpy.cross(v1, v2)

#         projectedAreas = numpy.sum(areaVec*self.axis,axis=1)

#         projectedAreas[projectedAreas<0] = 0.

#         sumArea = numpy.sum(projectedArea)/2.0

#         funcs[self.name] = sumArea

#     def evalFunctionsSens(self, funcsSens, config):
#         """
#         Evaluate the sensitivity of the functions this object has and
#         place in the funcsSens dictionary

#         Parameters
#         ----------
#         funcsSens : dict
#             Dictionary to place function values
#         """

#         nDV = self.DVGeo.getNDV()
#         if nDV > 0:
#             dLndPt = numpy.zeros((self.nCon, 
#                                  self.coords.shape[0],
#                                  self.coords.shape[1]))
#             dLndCn = numpy.zeros((self.nCon, 
#                                   1,
#                                   self.center.shape[0]))

#             for con in xrange(self.nCon):
#                 reflength2 = 0
#                 for i in xrange(3):
#                     reflength2 = reflength2 + (center[i]-coords[0,i])**2

#                 centerb = dLndCn[con,0,:]#0.0
#                 coordsb = dLndPt[con,:,:]#0.0
#                 reflength2b = 0.0
#                 for i in xrange(self.nCon):
#                     length2 = 0
#                     for j in xrange(3):
#                         length2 = length2 + (center[j]-coords[i+1, j])**2

#                     if (length2/reflength2 == 0.0):
#                         tempb1 = 0.0
#                     else:
#                         tempb1 = xb[i]/(2.0*numpy.sqrt(length2/reflength2)*reflength2)

#                     length2b = tempb1
#                     reflength2b = reflength2b - length2*tempb1/reflength2
#                     xb[i] = 0.0
#                     for j in reversed(xrange(3)):#DO j=3,1,-1
#                         tempb0 = 2*(center[j])-coords[i+1, j])*length2b
#                         centerb[j] = centerb[j] + tempb0
#                         coordsb[i+1, j] = coordsb[i+1, j] - tempb0
#                 for i in reversed(xrange(3)):#DO i=3,1,-1
#                     tempb = 2*(center[i]-coords[0, i])*reflength2b
#                     centerb[i] = centerb[i] + tempb
#                     coordsb[0, i] = coordsb[0, i] - tempb
            
                
#             tmpPt = self.DVGeo.totalSensitivity(dLndPt, self.name+'coords', config=config)
#             tmpCn = self.DVGeo.totalSensitivity(dLndCn, self.name+'center', config=config)
        
#             funcsSens[self.name] = tmpPt+tmpCn

#     def addConstraintsPyOpt(self, optProb):
#         """
#         Add the constraints to pyOpt, if the flag is set
#         """
#         if self.addToPyOpt:
#             optProb.addConGroup(self.name, self.nCon, lower=self.lower,
#                                 upper=self.upper, scale=self.scale,
#                                 wrt=self.DVGeo.getVarNames())

#     def writeTecplot(self, handle):
#         """
#         Write the visualization of this set of thickness constraints
#         to the open file handle
#         """

#         handle.write('Zone T=%s_coords\n'% self.name)
#         handle.write('Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n'% (
#             len(self.coords), len(self.coords)-1))
#         handle.write('DATAPACKING=POINT\n')
#         for i in range(len(self.coords)):
#             handle.write('%f %f %f\n'% (self.coords[i, 0], self.coords[i, 1],
#                                         self.coords[i, 2]))

#         for i in range(len(self.coords)-1):
#             handle.write('%d %d\n'% (i, i+1))

#         handle.write('Zone T=%s_center\n'% self.name)
#         handle.write('Nodes = 2, Elements = 1 ZONETYPE=FELINESEG\n')
#         handle.write('DATAPACKING=POINT\n')
#         handle.write('%f %f %f\n'% (self.center[0], self.center[1],
#                                     self.center[2]))
#         handle.write('%f %f %f\n'% (self.center[0], self.center[1],
#                                     self.center[2]))

