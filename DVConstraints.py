# ======================================================================
#         Extension modules
# ======================================================================
import numpy
from . import geo_utils
from pyspline import pySpline
from collections import OrderedDict

class DVConstraints(object):
    """
    DVConstraints provides a convient way of defining geometric
    constrints for WINGS. This can be very convient for a constrained
    aerodynamic or aerostructural optimization. Three types of
    constraints are supported:

    1. Thickness 'tooth-pick' constraints: Thickness constraints are
    enforced at specific locations. A relative or absolute
    minumum/maximum thickness can be specified. Two variants are
    supplied: a '2d' variant for thickness constraints over an area such
    as a spar box and a '1d' variant for thickness constraints along a
    line 

    2. Volume constraint: This computes and enforces a volume constraint
    over the specified domain. The input is identical to the '2d'
    thickness constraints.
    
    3. LE/TE Constraints: These geometric constraints are required when
    using FFD volumes with shape variables. The leading and trailing
    edges must be fixed wrt the shape variabes so these enforce that the
    coefficients on the leading edge can only move in equal and opposite
    directions

    Analytic sensitivity information is computed for all functions and a
    facility for adding the constrints automatically to a pyOpt
    optimization problem is also provided.
    """
  
    def __init__(self):
        """
        Create a (empty) DVconstrains object. Specific types of
        constraints will added individually
        """
        self.thickCon = OrderedDict()
        self.volumeCon = OrderedDict()
        self.LeTeCon = OrderedDict()
        self.volumeCGCon = OrderedDict()
        self.volumeAreaCon = OrderedDict()
      
        return

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
            self.p0 = numpy.array(wing[0])
            self.v1 = numpy.array(wing[1])
            self.v2 = numpy.array(wing[2])
        else:
            self.p0, self.v1, self.v2 = self._generateDiscreteSurface(wing)
        # end if

    def setDVGeo(self, DVGeo):
        """
        Set the DVGeometry object that will manipulate 'geometry' in
        this object. Note that DVConstraints doesn't **strictly** need
        a DVGeometry object set, but if optimization is desired it is
        required.

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
        upper = self._convertTo2D(upper, nSpan, nChord)
        lower = self._convertTo2D(lower, nSpan, nChord)
        scale = self._convertTo2D(scale, nSpan, nChord)
        coords = self._generateIntersections(leList, teList, nSpan, nChord)

        # Create the thickness constraint object:
        coords = coords.reshape((nSpan*nChord*2,3))

        # Create a name 
        if name is None:
            conName = 'thickness_constraints_%d'%(len(self.thickCon))
        else:
            conName = name
        self.thickCon[conName] = ThicknessConstraint(
            conName, coords, lower, upper, scale, scaled, scale, self.DVGeo,
            addToPyOpt)
        
        return 

    def addThicknessConstraints1D(self, ptList, nCon, axis, 
                                  lower=1.0, upper=3.0, scaled=True,
                                  scale=1.0, name=None,
                                  addToPyOpt=False):
        """
        Add a set of thickness constraint, that are oriented along a
        poly-line given by ptList.

        Parameters
        ----------

        ptList: list or array of size (N x 3) where N >=2
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
            constraint in pyOpt.
            
        addToPyOpt : bool
            Normally this should be left at the default of True. If
            the values need to be processed (modified) BEFORE they are
            given to the optimizer, set this flag to False.
        """

        # Create mesh of itersections
        constr_line = pySpline.curve(X=pt_list,k=2)
        s = numpy.linspace(0,1,nCon)
        X = constr_line(s)
        coords = numpy.zeros((nCon, 2, 3))
        # Project all the points
        for i in xrange(nCon):

            # Project actual node:
            up, down, fail = geo_utils.projectNode(
                X[i], axis, p0, v1, v2)
            if fail:
                raise Error('There was an error projecting a node \
 at (%f, %f, %f) with normal (%f, %f, %f).'%(X[i,j][0], X[i,j][1], X[i,j][2],
                                             up_vec[0], up_vec[1], up_vec[2]))
            coords[i, 0] = up
            coords[i, 1] = down
        # end for

        # Create the thickness constraint object:
        coords = coords.reshape((nCon*2, 3))
        if name is None:
            conName = 'thickness_constraints_%d'%(len(self.thickCon))
        else:
            conName = name
        self.thickCon[conName] = ThicknessConstraint(
            conName, coords, lower, upper, scale, scaled, scale, self.DVGeo,
            addToPyOpt)

        return

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

        if name is None:
            conName = 'volume_constraint_%d'%(len(self.volumeCon))
        else:
            conName = name

        coords = self._generateIntersections(leList, teList, nSpan, nChord)
        # Finally add the volume constraint object
        self.volumeCon[conName] = VolumeConstraint(
            conName, coords, lower, upper, scaled, scale,
            self.DVGeo, addToPyOpt)

        return

    def addLeTeCon(self, up_ind, low_ind):
        """Add Leading Edge and Trailing Edge Constraints to the FFD
        at the indiceis defined by up_ind and low_ind"""

        assert len(up_ind) == len(low_ind),  'up_ind and low_ind are\
 not the same length'

     
        # Check to see if we have local design variables in DVGeo
        if len(DVGeo.DV_listLocal) == 0:
            geo_utils.mpiPrint('Warning: Trying to add Le/Te Constraint when no local variables found')
        # end if

        # Loop over each set of Local Design Variables
        for i in xrange(len(DVGeo.DV_listLocal)):

            # We will assume that each GeoDVLocal only moves on
            # 1,2, or 3 coordinate directions (but not mixed)
            temp = DVGeo.DV_listLocal[i].coef_list
            for j in xrange(len(up_ind)): # Try to find this index
                                          # in the coef_list
                up = None
                down = None
                for k in xrange(len(temp)):
                    if temp[k][0] == up_ind[j]:
                        up = k
                    # end if
                    if temp[k][0] == low_ind[j]:
                        down = k
                    # end for
                # end for
                # If we haven't found up AND down do nothing
                if up is not None and down is not None:
                    self.LeTeCon.append([i, up, down])
                # end if
            # end for
        # end for

        # Finally, unique the list to parse out duplicates. Note:
        # This sort may not be stable however, the order of the
        # LeTeCon list doens't matter
        self.LeTeCon = geo_utils.unique(self.LeTeCon)

        return

    def addConstraintsPyOpt(self, optProb):
        """
        Add all constraints to the optProb object:

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
        for key in self.volumeCon:
            self.volumneCon[key].addConstraintsPyOpt(optProb)
        for key in self.LeTeCon:
            self.LeTeCon[key].addConstraintsPyOpt(optProb)

        return 

    def evalFunctions(self, funcs):
        """
        Evaluate all the 'functions' that this object has. Of course,
        these functions are actually just the desired constraint
        values. These values will be set directly into the funcs
        dictionary.

        Parameters
        ----------
        fucns : dict
            Dictionary into which the function values are placed. 
        """

        for key in self.thickCon:
            self.thickCon[key].evalFunctions(func)
        for key in self.volumeCon:
            self.volumneCon[key].evalFunctions(func)
        for key in self.LeTeCon:
            self.LeTeCon[key].evalFunctions(func)
                    
        return

    def evalFunctionsSens(self, funcsSens):
        """
        Evaluate the derivative of all the 'funcitons' that this
        object has. These functions are just the constraint values.
        Thse values will be set directly in the funcSens dictionary.
        """
        for key in self.thickCon:
            self.thickCon[key].evalFunctionsSens(funcSens)
        for key in self.volumeCon:
            self.volumneCon[key].evalFunctionsSens(funcSens)
        for key in self.LeTeCon:
            self.LeTeCon[key].evalFunctionsSens(funcSens)
            
        return

    def writeTecplot(self, fileName): 
        """
        This function writes a visualization file for all the
        currently added constraints to a tecplotVolume. This is useful
        for publication purposes as well as determine if the
        constraints are *actually* what the user expects them to be.

        Parameters
        ----------
        fileName : str
            File name for tecplot file. Should have a .dat extension or a
            .dat extension will be added automatically. 
        """
        
        f = open(fileName,'w')
        f.write("TITLE = \"DVConstraints Data\"\n")
        f.write("VARIABLES = \"CoordinateX\" \"CoordinateY\" \"CoordinateZ\"\n")

        for key in self.thickCon:
            self.thickCon[key].writeTecplot(f)
        for key in self.volumeCon:
            self.volumeCon[key].writeTecplot(f)
            
        f.close()

        return

    def _convertTo2D(self, value, dim1, dim2):
        """
        Generic function to process 'value'. In the end, it must be dim1
        by dim2. value is already that shape, excellent, otherwise, a
        scalar will be 'upcast' to that size
        """

        if numpy.isscalar:
            return value*numpy.ones((dim1,dim2))
        else:
            temp = numpy.atleast_2d(value)
            if temp.shape[0] == dim1 and tmp.shape[1] == dim2:
                return value
            else:
                raise ValueError('The size of the 2D array was the incorret shape')
                    
    def _convertTo1D(self, value, dim1):
        """
        Generic function to process 'value'. In the end, it must be
        array of size ndim1. value is already that shape, excellent,
        otherwise, a scalar will be 'upcast' to that size
        """

        if numpy.isscalar:
            return value*numpy.ones(ndim1)
        else:
            temp = numpy.atleast_1d(value)
            if temp.shape[0] == dim1:
                return value
            else:
                raise ValueError('The size of the 1D array was the incorret shape')
    
    def _generateIntersections(self, leList, teList, nSpan, nChord):
        """
        Internal function to generate the grid points (nSpan x nChord)
        and to actual perform the intersections. This is in a separate
        functions since addThicknessConstraints2D, and volume based
        constraints use the same conde. The list of projected
        coordinates are returned.
        """
                
        # Create mesh of itersections
        le_s = pySpline.curve(X=leList, k=2)
        te_s = pySpline.curve(X=teList, k=2)
        root_s = pySpline.curve(X=[leList[0], teList[0]], k=2)
        tip_s  = pySpline.curve(X=[leList[-1], teList[-1]], k=2)

        # Generate parametric distances
        span_s = numpy.linspace(0.0, 1.0, nSpan)
        chord_s = numpy.linspace(0.0, 1.0, nChord)
        
        # Generate a 2D region of intersections
        X = geo_utils.tfi_2d(le_s(span_s), te_s(span_s),
                             root_s(chord_s), tip_s(chord_s))
        coords = numpy.zeros((nSpan, nChord, 2, 3))
        # Generate all intersections:
        for i in xrange(nSpan): 
            for j in xrange(nChord):
                # Generate the 'up_vec' from taking the cross product
                # across a quad
                if i == 0:
                    u_vec = X[i+1, j]-X[i, j]
                elif i == nSpan - 1:
                    u_vec = X[i, j] - X[i-1, j]
                else:
                    u_vec = X[i+1, j] - X[i-1, j]
                # end if

                if j == 0:
                    v_vec = X[i, j+1]-X[i, j]
                elif j == nChord - 1:
                    v_vec = X[i, j] - X[i, j-1]
                else:
                    v_vec = X[i, j+1] - X[i, j-1]
                # end if

                up_vec = numpy.cross(u_vec, v_vec)
                
                # Project actual node:
                up, down, fail = geo_utils.projectNode(
                    X[i,j], up_vec, p0, v1, v2)

                if fail:
                    raise Error('There was an error projecting a node \
 at (%f, %f, %f) with normal (%f, %f, %f).'%(X[i,j][0], X[i,j][1], X[i,j][2],
                                             up_vec[0], up_vec[1], up_vec[2]))
                coords[i, j, 0] = up
                coords[i, j, 1] = down
            # end for
        # end for

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
        for isurf in xrange(wing.nSurf):
            surf = wing.surfs[isurf]
            ku = surf.ku
            kv = surf.kv
            tu = surf.tu
            tv = surf.tv
            
            u = geo_utils.fill_knots(tu, ku, level)
            v = geo_utils.fill_knots(tv, kv, level)

            for i in xrange(len(u)-1):
                for j in xrange(len(v)-1):
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

                # end for
            # end for
        # end for
        p0 = numpy.array(p0)
        v1 = numpy.array(v1)
        v2 = numpy.array(v2)

        return p0,v1,v2

class ThicknessConstraint(object):
    """
    DVConstraints representation of a set of thickness
    constraints. One of these objects is created each time a
    addThicknessConstraints2D or addThicknessConstraints1D call is
    made. The user should not have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo,
                 addToPyOpt):
        self.name
        self.coords = coords
        self.nCon = len(self.coords)/2
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
        self.D0 = numpy.zeros(nCon)
        for i in xrange(nCon):
            D0[i] = numpy.linalg.norm(self.coords[2*i] - self.coords[2*i+1])

    def evalFunctions(self, funcs):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name)
        D = numpy.zeros(self.nCon)
        for i in xrange(self.nCon):
            D[i] = numpy.linalg.norm(self.coords[2*i] - self.coords[2*i+1])
            if self.scaled:
                D[i] /= self.D0[i]
        funcs[self.name] = D

    def evalFunctionsSens(self, funcsSens):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo._getNDV()
        dTdx = numpy.zeros(self.nCon, nDV)
        dTdpt = numpy.zeros(self.coords.shape)
        for i in xrange(self.nCon):
            dTdpt[:, :] = 0.0
            p1b, p2b = geo_utils.e_dist_b(
                self.coords[2*i, :], self.coords[2*i+1, :])
        
            dTdpt[2*i  , :] = p1b
            dTdpt[2*i+1, :] = p2b

            if self.scaled:
                dTdpt[2*i  , :] /= self.D0[i]
                dTdpt[2*i+1, :] /= self.D0[i]

            dTdx[i, :] = self.DVGeo.totalSensitivity(dTdpt, name=self.name)
        # end for
        funcsSens[self.name] = {self.DVGeo.varSet:dTdx}

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addConGroup(self.name, self.nCon, lower=self.lower,
                                upper=self.upper, scale=self.scale,
                                wrt=[self.DVGeo.varSet])

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("ZONE T=\"%s\"\n"%(self.name))
        handle.write("Nodes=%d, Elements=%d, ZONETYPE=FELineSeg\n"%(
            self.nCon*2, self.nCon))
        handle.write("DATAPACKING=POINT\n")
        
        for i in xrange(self.nCon*2):
            handle.write('%g %g %g\n'%(self.coords[i,0],
                                       self.coords[i,1],
                                       self.coords[i,2]))
        for i in xrange(self.nCon):
            handle.write('%d %d\n'%(2*i+1,2*i+2))

class VolumeConstraint(object):

    def __init__(name, nSpan, nChord, coords, lower, upper, scaled,
                 scale, DVGeo, addToPyOpt):
        self.name
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
        self.DVvGeo.addPointSet(self.coords, self.name)

        # Now get the reference volume
        self.V0 = self.evalVolume()
                    
    def evalFunctions(self, funcs):
        """
        Evaluate the function this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name)

        V = self.evalVolume()
        if self.scaled:
            V /= self.V0
        funcs[self.name] = V

    def evalFunctionsSens(self, funcsSens):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        dVdPt = self.evalVolumeSens()
        if self.scaled:
            dVdPt /= self.V0

        # Now compute the DVGeo total sensitivity:
        dVdx = DVGeo.totalSensitivity(dVdPt, name=self.name)
        funcsSens[self.name] = {self.DVGeo.varSet:dVdx}

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addCon(self.name, lower=self.lower, upper=self.upper,
                           scale=self.scale, wrt=[self.DVGeo.varSet])

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """
        # Reshape coordinates back to 3D format
        x = self.coords.reshape([self.nSpan, self.nChord, 2, 3])

        handle.write("ZONE T=\"%s\" I=%d J=%d K=%d\n"%(
            self.name, self.nSpan, self.nChord, 2))
        handle.write("DATAPACKING=POINT\n")
        for k in xrange(2):
            for j in xrange(self.nChord):
                for i in xrange(self.nSpan):
                    f.write('%f %f %f\n'%(x[i, j, k, 0],
                                          x[i, j, k, 1],
                                          x[i, j, k, 2]))

    def evalVolume(self):
        """
        Evaluate the total volume of the current coordinates
        """
        Volume = 0.0
        x = self.coords.reshape((self.nSpan, self.nChord, 2, 3))
        for j in xrange(nChord-1):
            for i in xrange(nSpan-1):
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
        for j in xrange(nChord-1):
            for i in xrange(nSpan-1):
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
        
        p = numpy.average([x0, x1, x2, x3, x4, x5, x6, x7])
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

        p = numpy.average([x0, x1, x2, x3, x4, x5, x6, x7])
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

        return
    
    def volpym_b(self, a, b, c, d, p, ab, bb, cb, db, pb):
        """
        Compute the reverse-mode derivative of the square-based
        pyramid. This has been copied from reverse-mode AD'ed tapenade
        fortran code and converted to python to use vectors for the
        points.
        """
        fourth = 1.0/4.0
        volumeb = 1.0
        tempb = ((a[1]-c[1])*(b[2]-d[2])-(a[2]-c[2])*(b[1]-d[1]))*volumeb
        tempb0 = -(fourth*tempb)
        tempb1 = (p[0]-fourth*(a[0]+b[0]+c[0]+d[0]))*volumeb
        tempb2 = ((a[2]-c[2])*(b[0]-d[0])-(a[0]-c[0])*(b[2]-d[2]))*volumeb
        tempb3 = -(fourth*tempb2)
        tempb4 = (p[1]-fourth*(a[1]+b[1]+c[1]+d[1]))*volumeb
        tempb5 = ((a[0]-c[0])*(b[1]-d[1])-(a[1]-c[1])*(b[0]-d[0]))*volumeb
        tempb6 = -(fourth*tempb5)
        tempb7 = (p[2]-fourth*(a[2]+b[2]+c[2]+d[2]))*volumeb
        pb[0] = pb[0] + tempb
        ab[0] = ab[0] + (b[1]-d[1])*tempb7 - (b[2]-d[2])*tempb4 + tempb0
        bb[0] = bb[0] + (a[2]-c[2])*tempb4 - (a[2]-c[1])*tempb7 + tempb0
        cb[0] = cb[0] + (b[2]-d[2])*tempb4 - (b[1]-d[1])*tempb7 + tempb0
        db[0] = db[0] + (a[1]-c[1])*tempb7 - (a[2]-c[2])*tempb4 + tempb0
        ab[1] = ab[1] + tempb3 - (b[0]-d[0])*tempb7 + (b[2]-d[2])*tempb1
        cb[1] = cb[1] + (b[0]-d[0])*tempb7 + tempb3 - (b[2]-d[2])*tempb1
        bb[2] = bb[2] + tempb6 - (a[0]-c[0])*tempb4 + (a[1]-c[1])*tempb1
        db[2] = db[2] + tempb6 + (a[0]-c[0])*tempb4 - (a[1]-c[1])*tempb1
        ab[2] = ab[2] + tempb6 + (b[0]-d[0])*tempb4 - (b[1]-d[1])*tempb1
        cb[2] = cb[2] + tempb6 - (b[0]-d[0])*tempb4 + (b[1]-d[1])*tempb1
        bb[1] = bb[1] + (a[0]-c[0])*tempb7 + tempb3 - (a[2]-c[2])*tempb1
        db[1] = db[1] + tempb3 - (a[0]-c[0])*tempb7 + (a[2]-c[2])*tempb1
        pb[1] = pb[1] + tempb2
        pb[2] = pb[2] + tempb5

        return

class VolumeCGConstraint(VolumeConstraint):

    def __init__(name, nSpan, nChord, coords, lower, upper, scaled,
                 scale, DVGeo, addToPyOpt):

        pass
    def evalFunctions(self, funcs):
        """
        Evaluate the function this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name)

        V = self.evalVolumeGS()


    def evalFunctionsSens(self, funcsSens):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        dVdPt = self.evalVolumeSens()
        if self.scaled:
            dVdPt /= self.V0

        # Now compute the DVGeo total sensitivity:
        dVdx = DVGeo.totalSensitivity(dVdPt, name=self.name)
        funcsSens[self.name] = {self.DVGeo.varSet:dVdx}

# --------------------------------------------------------------------------------
    def _getLeTeConstraints(self):
        """Evaluate the LeTe constraint using the current DVGeo opject"""

        con = numpy.zeros(len(self.LeTeCon))
        for i in xrange(len(self.LeTeCon)):
            dv = self.LeTeCon[i][0]
            up = self.LeTeCon[i][1]
            down = self.LeTeCon[i][2]
            con[i] = self.DVGeo.DV_listLocal[dv].value[up] + \
                DVGeo.DV_listLocal[dv].value[down]
        # end for

        return con

    def _getLeTeSensitivity(self):
        ndv = self.DVGeo._getNDV()
        nlete = len(self.LeTeCon)
        dLeTedx = numpy.zeros([nlete, ndv])

        DVoffset = [self.DVGeo._getNDVGlobal()]
        # Generate offset lift of the number of local variables
        for i in xrange(len(self.DVGeo.DV_listLocal)):
            DVoffset.append(DVoffset[-1] + self.DVGeo.DV_listLocal[i].nVal)

        for i in xrange(len(self.LeTeCon)):
            # Set the two values a +1 and -1 or (+range - range if scaled)
            dv = self.LeTeCon[i][0]
            up = self.LeTeCon[i][1]
            down = self.LeTeCon[i][2]
            dLeTedx[i, DVoffset[dv] + up  ] =  1.0
            dLeTedx[i, DVoffset[dv] + down] =  1.0
        # end for

        return dLeTedx

 
