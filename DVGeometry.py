# ======================================================================
#         Imports
# ======================================================================
from __future__ import print_function
import copy
try:
    from collections import OrderedDict
except ImportError:
    try:
        from ordereddict import OrderedDict
    except ImportError:
        print("Could not find any OrderedDict class. For 2.6 and earlier, "
              "use:\n pip install ordereddict")
import numpy
from scipy import sparse
from mpi4py import MPI
from pyspline import pySpline
from . import pyNetwork, pyBlock, geo_utils
import pdb

class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a explicitly raised exception.
    """
    def __init__(self, message):
        msg = '\n+'+'-'*78+'+'+'\n' + '| DVGeometry Error: '
        i = 19
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
        
class DVGeometry(object):
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
    
    DVGeometry uses the *Free-Form Deformation* approach for goemetry
    manipulation. The basic idea is the coordinates are *embedded* in
    a clear-flexible jelly-like block. Then by stretching moving and
    'poking' the volume, the coordinates that are embedded inside move
    along with overall deformation of the volume. 

    Parameters
    ----------
    fileName : str
       filename of FFD file. This must be a ascii formatted plot3D file
       in fortran ordering. 

    complex : bool
        Make the entire object complex. This should **only** be used when
        debugging the entire tool-chain with the complex step method. 

    child : bool
        Flag to indicate that this object is a child of parent DVGeo object


    Examples
    --------
    The general sequence of operations for using DVGeometry is as follows::
      >>> from pygeo import *
      >>> DVGeo = DVGeometry('FFD_file.fmt')
      >>> # Embed a set of coordinates Xpt into the object
      >>> DVGeo.addPointSet(Xpt, 'myPoints')
      >>> # Associate a 'reference axis' for large-scale manipulation
      >>> DVGeo.addRefAxis('wing_axis', axis_curve)
      >>> # Define a global design variable function:
      >>> def twist(val, geo):
      >>>    geo.rot_z['wing_axis'].coef[:] = val[:]
      >>> # Now add this as a global variable:
      >>> DVGeo.addGeoDVGlobal('wing_twist', 0.0, twist, lower=-10, upper=10)
      >>> # Now add local (shape) variables
      >>> DVGeo.addGeoDVLocal('shape', lower=-0.5, upper=0.5, axis='y')
      >>> 
      """
    def __init__(self, fileName, complex=False, child=False, *args, **kwargs):
        
        self.DV_listGlobal  = OrderedDict() # Global Design Variable List
        self.DV_listLocal = OrderedDict() # Local Design Variable List

        # Flags to determine if this DVGeometry is a parent or child
        self.isChild = child
        self.children = []
        self.iChild = None
        self.nChildren = None
        self.points = OrderedDict()
        self.updated = {}
        self.masks = OrderedDict()
        self.finalized = False
        self.complex = complex
        if self.complex:
            self.dtype = 'D'
        else:
            self.dtype = 'd'

        # Load the FFD file in FFD mode. Also note that args and
        # kwargs are passed through in case additional pyBlock options
        # need to be set. 
        self.FFD = pyBlock('plot3d', fileName=fileName, FFD=True,
                           *args, **kwargs)
        self.origFFDCoef = self.FFD.coef.copy()

        # Jacobians:
        # self.JT: Total transpose jacobian for self.J_name
        self.zeroJacobians()
     
        # Derivatives of Xref and Coef provided by the parent to the
        # children
        self.dXrefdXdvg = None
        self.dCoefdXdvg = None

        self.dXrefdXdvl = None
        self.dCoefdXdvl = None

        # The set of user supplied axis. 
        self.axis = OrderedDict()

    def addRefAxis(self, name, curve=None, xFraction=None, volumes=None,
                   rotType=5, axis='x'):
        """
        This function is used to add a 'reference' axis to the
        DVGeometry object.  Adding a reference axis is only required
        when 'global' design variables are to be used, i.e. variables
        like span, sweep, chord etc --- variables that affect many FFD
        control points.

        There are two different ways that a reference can be
        specified:
 
        #. The first is explicitly a pySpline curve object using the
           keyword argument curve=<curve>.

        #. The second is to specify the xFraction variable. There are
           few caveats with the use of this method. First, DVGeometry
           will try to determine automatically the orientation of the FFD
           volume. Then, a reference axis will consist of the same number
           of span-wise sections as the FFD volume has and will be will
           be oriented in the streamwise (x-direction) according to the
           xPercent keyword argument. 

        Parameters
        ----------
        name : str
            Name of the reference axis. This name is used in the
            user-supplied design variable functions to determine what
            axis operations occur on. 
        
        curve : pySpline curve object
            Supply exactly the desired reference axis

        xFraction : float
            Specifiy the stream-wise extent 

        volumes : list or array or integers
            List of the volume indices, in 0-based ordering that this
            reference axis should manipulate. If xFraction is
            specified, the volumes argument must contain at most 1
            volume. If the volumes is not given, then all volumes are
            taken. 

        rotType : int
            Integer in range 0->6 (inclusive) to determine the order
            that the rotations are made. 

            0. Intrinsic rotation, rot_theta is rotation about axis
            1. x-y-z
            2. x-z-y
            3. y-z-x  
            4. y-x-z
            5. z-x-y  Default (x-streamwise y-up z-out wing)
            6. z-y-x
            7. z-x-y + rot_theta

        axis: str
            Axis along which to project points/control points onto the
            ref axis. Default is 'x' which will project rays. 
            
        Notes
        -----
        One of curve or xFraction must be specified. 
        
        Examples
        --------
        >>> # Simple wing with single volume FFD, reference axis at 1/4 chord:
        >>> DVGeo.addRefAxis('wing', xFraction=0.25)
        >>> # Multiblock FFD, wing is volume 6.
        >>> DVGeo.addRefAxis('wing', xFraction=0.25, volumes=[6])
        >>> # Multiblock FFD, multiple volumes attached refAxis
        >>> DVGeo.addRefAxis('wing', myCurve, volumes=[2,3,4])

        Returns
        -------
        nAxis : int
            The number of control points on the reference axis. 
        """

        # We don't do any of the final processing here; we simply
        # record the information the user has supplied into a
        # dictionary structure.

        if axis.lower() == 'x':
            axis = numpy.array([1, 0, 0], 'd')
        elif axis.lower() == 'y':
            axis = numpy.array([0, 1, 0], 'd')
        elif axis.lower() == 'z':
            axis = numpy.array([0, 0, 1], 'd')

        if curve is not None:
            # Explicit curve has been supplied:
            if self.FFD.symmPlane is None:
                if volumes is None:
                    volumes = numpy.arange(self.FFD.nVol)
                self.axis[name] = {'curve':curve, 'volumes':volumes,
                                   'rotType':rotType, 'axis':axis}

            else:
                # get the direction of the symmetry plane
                if self.FFD.symmPlane.lower() == 'x':
                    index = 0
                elif self.FFD.symmPlane.lower() == 'y':
                    index = 1
                elif self.FFD.symmPlane.lower() == 'z':
                    index = 2

                # mirror the axis and attach the mirrored vols
                if volumes is None:
                    volumes = numpy.arange(self.FFD.nVol/2)

                volumesSymm = []
                for volume in volumes:
                    volumesSymm.append(volume+self.FFD.nVol/2)

                curveSymm = copy.deepcopy(curve)
                curveSymm.reverse()
                for coef in curveSymm.coef:
                    curveSymm.coef[:,index]=-curveSymm.coef[:,index]
                self.axis[name] = {'curve':curve, 'volumes':volumes,
                                   'rotType':rotType, 'axis':axis}
                self.axis[name+'Symm'] = {'curve':curveSymm, 'volumes':volumesSymm,
                                          'rotType':rotType, 'axis':axis}

            nAxis = len(curve.coef)
        elif xFraction is not None:
            raise Error('xFraction specification is not coded yet.')
        else:
            raise Error("One of 'curve' or 'xFraction' must be "
                        "specified for a call to addRefAxis")

        return nAxis
   
    def addPointSet(self, points, ptName, origConfig=True, **kwargs):
        """
        Add a set of coordinates to DVGeometry

        The is the main way that geometry, in the form of a coordinate
        list is given to DVGeoemtry to be manipulated.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed. These cordinates *should* all
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
            exactly what they are doing."""
 
        points = numpy.array(points).real.astype('d')
        self.points[ptName] = points

        # Ensure we project into the undeformed geometry
        if origConfig:
            tmpCoef = self.FFD.coef.copy()
            self.FFD.coef = self.origFFDCoef
            self.FFD._updateVolumeCoef()

        # Project the last set of points into the volume
        if self.isChild:
            coefMask = self.FFD.attachPoints(
                self.points[ptName], ptName, interiorOnly=True, **kwargs)
        else:
            coefMask = self.FFD.attachPoints(
                self.points[ptName], ptName, interiorOnly=False)

        if origConfig:
            self.FFD.coef = tmpCoef
            self.FFD._updateVolumeCoef()

        # Now embed into the children:
        for child in self.children:
            child.addPointSet(points, ptName, origConfig, **kwargs)

        self.masks[ptName] = coefMask
        self.FFD.calcdPtdCoef(ptName)
        self.updated[ptName] = False

    def addChild(self, childDVGeo):
        """Embed a child FFD into this object.

        An FFD child is a 'sub' FFD that is fully contained within
        another, parent FFD. A child FFD is also an instance of
        DVGeometry which may have its own global and/or local design
        variables. Coordinates do **not** need to be added to the
        children. The parent object will take care of that in a call
        to addPointSet(). 

        Parameters
        ----------
        childDVGeo : instance of DVGeometry
            DVGeo object to use as a sub-FFD
        """

        # Make sure the DVGeo being added is flaged as a child:
        if childDVGeo.isChild is False:
            raise Error("Trying to add a child FFD that has NOT been "
                        "created as a child. This operation is illegal.")

        # Extract the coef from the child FFD and ref axis and embed
        # them into the parent and compute their derivatives
        iChild = len(self.children)
        childDVGeo.iChild = iChild
        
        self.FFD.attachPoints(childDVGeo.FFD.coef, 'child%d_coef'%(iChild))
        self.FFD.calcdPtdCoef('child%d_coef'%(iChild))

        # We must finalize the Child here since we need the ref axis
        # coefficients
        childDVGeo._finalizeAxis()
        self.FFD.attachPoints(childDVGeo.refAxis.coef, 'child%d_axis'%(iChild))
        self.FFD.calcdPtdCoef('child%d_axis'%(iChild))

        # Add the child to the parent and return
        self.children.append(childDVGeo)

    def addGeoDVGlobal(self, dvName, value, func, lower=None, upper=None,
                       scale=1.0, config=None):
        """
        Add a global design variable to the DVGeometry object. This
        type of design variable acts on one or more reference axis.

        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group

        value : float, or iterable list of floats
            The starting value(s) for the design variable. This
            parameter may be a single variable or a numpy array
            (or list) if the function requires more than one
            variable. The number of variables is determined by the
            rank (and if rank ==1, the length) of this parameter.

        lower : float, or iterable list of floats
            The lower bound(s) for the variable(s). A single variable
            is permissable even if an array is given for value. However,
            if an array is given for 'lower', it must be the same length
            as 'value'

        func : python function
            The python function handle that will be used to apply the
            design variable

        upper : float, or iterable list of floats
            The upper bound(s) for the variable(s). Same restrictions as
            'lower'

        scale : float, or iterable list of floats
            The scaling of the variables. A good approximate scale to
            start with is approximately 1.0/(upper-lower). This gives
            variables that are of order ~1.0. 
        
        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple 
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.
        """
        if type(config) == str:
            config = [config]
        self.DV_listGlobal[dvName] = geoDVGlobal(
            dvName, value, lower, upper, scale, func, config)

    def addGeoDVLocal(self, dvName, lower=None, upper=None, scale=1.0,
                      axis='y', volList=None, pointSelect=None, config=None):
        """
        Add one or more local design variables ot the DVGeometry
        object. Local variables are used for small shape modifications.

        Parameters
        ----------
        dvName : str
            A unique name to be given to this design variable group

        lower : float
            The lower bound for the variable(s). This will be applied to
            all shape variables

        upper : float
            The upper bound for the variable(s). This will be applied to
            all shape variables

        scale : flot
            The scaling of the variables. A good approximate scale to
            start with is approximately 1.0/(upper-lower). This gives
            variables that are of order ~1.0.
            
        axis : str. Default is 'y'
            The coordinate directions to move. Permissible values are 'x',
            'y' and 'z'. If more than one direction is required, use multiple
            calls to addGeoDVLocal with different axis values
        volList : list
            Use the control points on the volume indicies given in volList
            
        pointSelect : pointSelect object. Default is None Use a
            pointSelect object to select a subset of the total number
            of control points. See the documentation for the
            pointSelect class in geo_utils.

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple 
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.

        Returns
        -------
        N : int
            The number of design variables added. 

        Examples
        --------
        >>> # Add all variables in FFD as local shape variables
        >>> # moving in the y direction, within +/- 1.0 units
        >>> DVGeo.addGeoDVLocal('shape_vars', lower=-1.0, upper= 1.0, axis='y')
        >>> # As above, but moving in the x and y directions.
        >>> nVar = DVGeo.addGeoDVLocal('shape_vars_x', lower=-1.0, upper= 1.0, axis='x')
        >>> nVar = DVGeo.addGeoDVLocal('shape_vars_y', lower=-1.0, upper= 1.0, axis='y')
        >>> # Create a point select to use: (box from (0,0,0) to (10,0,10) with 
        >>> # any point projecting into the point along 'y' axis will be selected.
        >>> PS = geoUtils.pointSelect(type = 'y', pt1=[0,0,0], pt2=[10, 0, 10])
        >>> nVar = DVGeo.addGeoDVLocal('shape_vars', lower=-1.0, upper=1.0, pointSelect=PS)
        """
        if type(config) == str:
            config = [config]

        if pointSelect is not None:
            pts, ind = pointSelect.getPoints(self.FFD.coef)
        elif volList is not None:
            if self.FFD.symmPlane is not None:
                volListTmp = []
                for vol in volList:
                    volListTmp.append(vol)
                for vol in volList:
                    volListTmp.append(vol+self.FFD.nVol/2)
                volList = volListTmp                    

            volList = numpy.atleast_1d(volList).astype('int')
            ind = []
            for iVol in volList:
                ind.extend(self.FFD.topo.lIndex[iVol].flatten())
            ind = geo_utils.unique(ind)
        else:
            # Just take'em all
            ind = numpy.arange(len(self.FFD.coef))

        self.DV_listLocal[dvName] = geoDVLocal(dvName, lower, upper,
                                               scale, axis, ind, config)
            
        return self.DV_listLocal[dvName].nVal

    def getSymmetricCoefList(self,volList=None, pointSelect=None, tol = 1e-8):
        """
        Determine the pairs of coefs that need to be constrained for symmetry.

        Parameters
        ----------
        volList : list
            Use the control points on the volume indicies given in volList
            
        pointSelect : pointSelect object. Default is None Use a
            pointSelect object to select a subset of the total number
            of control points. See the documentation for the
            pointSelect class in geo_utils.
        tol : float
              Tolerance for ignoring nodes around the symmetry plane. These should be
              merged by the network/connectivity anyway

        Returns
        -------
        indSetA : list of ints
                  One half of the coefs to be constrained

        indSetB : list of ints
                  Other half of the coefs to be constrained

        Examples
        --------
 
        """
        
        if self.FFD.symmPlane is None:
            #nothing to be done
            indSetA = []
            indSetB = []
        else:
            # get the direction of the symmetry plane
            if self.FFD.symmPlane.lower() == 'x':
                index = 0
            elif self.FFD.symmPlane.lower() == 'y':
                index = 1
            elif self.FFD.symmPlane.lower() == 'z':
                index = 2

            #get the points to be matched up
            if pointSelect is not None:
                pts, ind = pointSelect.getPoints(self.FFD.coef)
            elif volList is not None:
                volListTmp = []
                for vol in volList:
                    volListTmp.append(vol)
                for vol in volList:
                    volListTmp.append(vol+self.FFD.nVol/2)
                volList = volListTmp                    

                volList = numpy.atleast_1d(volList).astype('int')
                ind = []
                for iVol in volList:
                    ind.extend(self.FFD.topo.lIndex[iVol].flatten())
                ind = geo_utils.unique(ind)
                pts = self.FFD.coef[ind]
            else:
                # Just take'em all
                ind = numpy.arange(len(self.FFD.coef))
                pts = self.FFD.coef

            # Create the base points for the KD tree search. We will take the abs
            # value of the symmetry direction, that way when we search we will get
            # back index pairs which is what we want.
            baseCoords = copy.copy(pts)
            baseCoords[:,index] = abs(baseCoords[:,index])
            
            #now use the baseCoords to create a KD tree
            try:
                from scipy.spatial import cKDTree
            except:
                raise Error("scipy.spatial "
                            "must be available to use detect symmetry")
    
            # Now make a KD-tree so we can use it to find the unique nodes
            tree = cKDTree(baseCoords)
                
            # Now search through the +ve half of the points, ignoring anything within 
            # tol of the symmetry plane to find pairs
            indSetA = []
            indSetB = []
            for pt in pts:
                if pt[index]>tol:
                    # Now find any matching nodes within tol. there should be 2 and
                    # only 2 if the mesh is symmtric
                    Ind =tree.query_ball_point(pt, tol)#should this be a separate tol
                    if not(len(Ind)==2):
                        raise Error("more than 2 coefs found that match pt")
                    else:
                        indSetA.append(Ind[0])
                        indSetB.append(Ind[1])

        return indSetA,indSetB

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

        # Coefficients must be complexifed from here on if complex
        if self.complex:
            self._complexifyCoef()

        for key in dvDict:
            if key in self.DV_listGlobal:
                vals_to_set = numpy.atleast_1d(dvDict[key]).astype('D')
                if len(vals_to_set) != self.DV_listGlobal[key].nVal:
                    raise Error("Incorrect number of design variables "
                                "for DV: %s.\nExpecting %d variables and "
                                "received %d variabes" % (
                                    key, self.DV_listGlobal[key].nVal,
                                    len(vals_to_set)))

                self.DV_listGlobal[key].value = vals_to_set
            
            if key in self.DV_listLocal:
                vals_to_set = numpy.atleast_1d(dvDict[key])
                if len(vals_to_set) != self.DV_listLocal[key].nVal:
                    raise Error('Incorrect number of design variables \
                    for DV: %s.\nExpecting %d variables and received \
                    %d variabes'%(key, self.DV_listLocal[key].nVal,
                                  len(vals_to_set)))
                self.DV_listLocal[key].value = vals_to_set

            # Jacobians are, in general, no longer up to date
            self.zeroJacobians()

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

        # Now call setValues on the children. This way the
        # variables will be set on the children
        for child in self.children:
            child.setDesignVars(dvDict)

    def zeroJacobians(self):
        '''
        set all of the stored jacobians to None
        '''
        self.JT = None # J is no longer up to date
        self.J_name = None # Name is no longer defined
        self.J_attach = None
        self.J_local = None

    def getValues(self):
        """
        Generic routine to return the current set of design
        variables. Values are returned in a dictionary format
        that would be suitable for a subsequent call to setValues()

        Returns
        -------
        dvDict : dict
            Dictionary of design variables
        """
        
        dvDict = {}
        for key in self.DV_listGlobal:
            dvDict[key] = self.DV_listGlobal[key].value
        
        # and now the local DVs
        for key in self.DV_listLocal:
            dvDict[key] = self.DV_listLocal[key].value

        # Now call getValues on the children. This way the
        # returned dictionary will include the variables from
        # the children
        for child in self.children:
            childdvDict = child.getValues()
            dvDict.update(childdvDict)

        return dvDict

    def extractCoef(self, axisID):
        """ Extract the coefficients for the selected reference
        axis. This should be used only inside design variable functions"""

        axisNumber = self._getAxisNumber(axisID)
        C = numpy.zeros((len(self.refAxis.topo.lIndex[axisNumber]), 3),
                        self.coef.dtype)
 
        C[:, 0] = numpy.take(self.coef[:, 0],
                             self.refAxis.topo.lIndex[axisNumber])
        C[:, 1] = numpy.take(self.coef[:, 1],
                             self.refAxis.topo.lIndex[axisNumber])
        C[:, 2] = numpy.take(self.coef[:, 2],
                             self.refAxis.topo.lIndex[axisNumber])

        return C

    def restoreCoef(self, coef, axisID):
        """ Restore the coefficients for the selected reference
        axis. This should be used inside design variable functions"""

        # Reset
        axisNumber = self._getAxisNumber(axisID)
        numpy.put(self.coef[:, 0],
                  self.refAxis.topo.lIndex[axisNumber], coef[:, 0])
        numpy.put(self.coef[:, 1],
                  self.refAxis.topo.lIndex[axisNumber], coef[:, 1])
        numpy.put(self.coef[:, 2],
                  self.refAxis.topo.lIndex[axisNumber], coef[:, 2])

    def extractS(self, axisID):
        """Extract the parametric positions of the control
        points. This is usually used in conjunction with extractCoef()"""
        axisNumber = self._getAxisNumber(axisID)
        return self.refAxis.curves[axisNumber].s.copy()

    def _getAxisNumber(self, axisID):
        """Get the sequential axis number from the name tag axisID"""
        try:
            return self.axis.keys().index(axisID)
        except:
            raise Error("'The 'axisID' was invalid!")
        
    def update(self, ptSetName, childDelta=True, config=None):
        """
        This is the main routine for returning coordinates that have
        been updated by design variables.

        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the
            given in an :func:`addPointSet()` call. 

        childDelta : bool
            Return updates on child as a delta. The user should not
            need to ever change this parameter.
            """
        self.curPtSet = ptSetName
        # We've postposed things as long as we can...do the finialization. 
        self._finalize()
        
        # Make sure coefficients are complex
        self._complexifyCoef()

        # Step 1: Call all the design variables IFF we have ref axis:
        if len(self.axis) > 0:
            # Set all coef Values back to initial values
            if not self.isChild:
                self._setInitialValues()
                
            if self.complex:
                new_pts = numpy.zeros((self.nPtAttach, 3), 'D')
            else:
                new_pts = numpy.zeros((self.nPtAttach, 3), 'd')

            if self.isChild:
                for ipt in xrange(self.nPtAttach):
                    base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])
                    self.links_x[ipt] = self.FFD.coef[self.ptAttachInd[ipt], :] - base_pt

            # Run Global Design Vars
            for key in self.DV_listGlobal:
                self.DV_listGlobal[key](self, config)

            self.refAxis.coef = self.coef.copy()
            self.refAxis._updateCurveCoef()

            for ipt in xrange(self.nPtAttach):
                base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])
                scale = self.scale[self.curveIDNames[ipt]](self.links_s[ipt]) 
                scale_x = self.scale_x[self.curveIDNames[ipt]](self.links_s[ipt]) 
                scale_y = self.scale_y[self.curveIDNames[ipt]](self.links_s[ipt]) 
                scale_z = self.scale_z[self.curveIDNames[ipt]](self.links_s[ipt]) 

                rotType = self.axis[self.curveIDNames[ipt]]['rotType']
                if rotType == 0:
                    deriv = self.refAxis.curves[
                        self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                    deriv /= geo_utils.euclideanNorm(deriv) # Normalize
                    new_vec = -numpy.cross(deriv, self.links_n[ipt])
                    new_vec = geo_utils.rotVbyW(new_vec, deriv, self.rot_x[
                            self.curveIDs[ipt]](self.links_s[ipt])*numpy.pi/180)
                    new_pts[ipt] = base_pt + new_vec*scale

                else:
                    rotX = geo_utils.rotxM(self.rot_x[
                            self.curveIDNames[ipt]](self.links_s[ipt]))
                    rotY = geo_utils.rotyM(self.rot_y[
                            self.curveIDNames[ipt]](self.links_s[ipt]))
                    rotZ = geo_utils.rotzM(self.rot_z[
                            self.curveIDNames[ipt]](self.links_s[ipt]))

                    D = self.links_x[ipt]

                    rotM = self._getRotMatrix(rotX, rotY, rotZ, rotType)
                    D = numpy.dot(rotM, D)
                    if rotType == 7:
                        # only apply the theta rotations in certain cases
                        deriv = self.refAxis.curves[
                            self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                        deriv[0] = 0.0
                        deriv /= geo_utils.euclideanNorm(deriv) # Normalize
                        D = geo_utils.rotVbyW(D, deriv, numpy.pi/180*self.rot_theta[
                                self.curveIDNames[ipt]](self.links_s[ipt]))
                    
                    D[0] *= scale_x
                    D[1] *= scale_y
                    D[2] *= scale_z
                    if self.complex:
                        new_pts[ipt] = base_pt + D*scale
                    else:
                        new_pts[ipt] = numpy.real(base_pt + D*scale)

            if not self.isChild:
                temp = numpy.real(new_pts)
                self.FFD.coef = self.ptAttachFull.copy()
                numpy.put(self.FFD.coef[:, 0], self.ptAttachInd, temp[:, 0])
                numpy.put(self.FFD.coef[:, 1], self.ptAttachInd, temp[:, 1])
                numpy.put(self.FFD.coef[:, 2], self.ptAttachInd, temp[:, 2])
            else:
                oldCoefLocations = self.FFD.coef.copy()

                # Coeffients need to be set with delta values
                temp = numpy.real(new_pts)
                numpy.put(self.FFD.coef[:, 0], self.ptAttachInd, temp[:, 0])
                numpy.put(self.FFD.coef[:, 1], self.ptAttachInd, temp[:, 1])
                numpy.put(self.FFD.coef[:, 2], self.ptAttachInd, temp[:, 2])

                if childDelta:
                    self.FFD.coef -= oldCoefLocations
        else:
            # Since we have no ref axis (and thus no global dvs) we
            # just take the original FFD coefficients. 
            self.FFD.coef = self.origFFDCoef.copy()
#            put child update here for no ref axis case?
        # end for (ref axis)

        for key in self.DV_listLocal:
            self.DV_listLocal[key](self.FFD.coef, config)

        # Update all coef
        self.FFD._updateVolumeCoef()

        # Evaluate coordinates from the parent
        coords = self.FFD.getAttachedPoints(ptSetName)

        # Now loop over the children set the FFD and refAxis control
        # points as evaluated from the parent

        for iChild in xrange(len(self.children)):
            self.children[iChild]._finalize()
            self.children[iChild].FFD.coef = self.FFD.getAttachedPoints(
                'child%d_coef'%(iChild))

            self.children[iChild].coef = self.FFD.getAttachedPoints(
                'child%d_axis'%(iChild))
            self.children[iChild].refAxis.coef = self.children[iChild].coef.copy()
            self.children[iChild].refAxis._updateCurveCoef()

            coords += self.children[iChild].update(ptSetName, childDelta, config=config)
            
        if self.complex:
            if len(self.children) > 0:
                print(' Warning: Complex step NOT TESTED with children yet')
     
            tempCoef = self.ptAttachFull.copy().astype('D')
            numpy.put(tempCoef[:, 0], self.ptAttachInd, new_pts[:, 0])
            numpy.put(tempCoef[:, 1], self.ptAttachInd, new_pts[:, 1])
            numpy.put(tempCoef[:, 2], self.ptAttachInd, new_pts[:, 2])

            # Apply just the complex part of the local varibales
            for key in self.DV_listLocal:
                self.DV_listLocal[key].updateComplex(tempCoef, config)
                     
            coords = coords.astype('D')
            imag_part = numpy.imag(tempCoef)
            imag_j = 1j

            dPtdCoef = self.FFD.embededVolumes[ptSetName].dPtdCoef
            if dPtdCoef is not None:
                for ii in xrange(3):
                    coords[:, ii] += imag_j*dPtdCoef.dot(imag_part[:, ii])

            self._unComplexifyCoef()

        # Finally flag this pointSet as being up to date:
        self.updated[ptSetName] = True

        return coords

    def pointSetUpToDate(self, ptSetName):
        """
        This is used externally to query if the object needs to update
        its pointset or not. Essentially what happens, is when
        update() is called with a point set, it the self.updated dict
        entry for pointSet is flagged as true. Here we just return
        that flag. When design variables are set, we then reset all
        the flags to False since, when DVs are set, nothing (in
        general) will up to date anymore.

        Parameters
        ----------
        ptSetName : str
            The name of the pointset to check.
        """
        if ptSetName in self.updated:
            return self.updated[ptSetName]
        else:
            return True

    def convertSensitivityToDict(self, dIdx, out1D=False):
        """
        This function takes the result of totalSensitivity and
        converts it to a dict for use in pyOptSparse

        Parameters
        ----------
        dIdx : array
           Flattened array of length getNDV(). Generally it comes from
           a call to totalSensitivity()

        out1D : boolean
            If true, creates a 1D array in the dictionary instead of 2D.
            This function is used in the matrix-vector product calculation.

        Returns
        -------
        dIdxDict : dictionary
           Dictionary of the same information keyed by this object's
           design variables
        """

        # compute the various DV offsets
        DVCountGlobal,DVCountLocal = self._getDVOffsets()

        i = DVCountGlobal
        dIdxDict = {}
        for key in self.DV_listGlobal:
            dv = self.DV_listGlobal[key]
            if out1D:
                dIdxDict[dv.name] = numpy.ravel(dIdx[:, i:i+dv.nVal])
            else:
                dIdxDict[dv.name] = dIdx[:, i:i+dv.nVal]
            i += dv.nVal
        i = DVCountLocal
        for key in self.DV_listLocal:
            dv = self.DV_listLocal[key]
            if out1D:
                dIdxDict[dv.name] = numpy.ravel(dIdx[:, i:i+dv.nVal])
            else:
                dIdxDict[dv.name] = dIdx[:, i:i+dv.nVal]

            i += dv.nVal
            
        # Add in child portion
        for iChild in xrange(len(self.children)):
            childdIdx = self.children[iChild].convertSensitivityToDict(dIdx, out1D=out1D)
            # update the total sensitivities with the derivatives from the child
            for key in childdIdx:               
                if key in dIdxDict.keys():
                    dIdxDict[key]+=childdIdx[key]
                else:
                    dIdxDict[key]=childdIdx[key]

        return dIdxDict

    def convertDictToSensitivity(self, dIdxDict):
        """
        This function performs the reverse operation of 
        convertSensitivityToDict(); it transforms the dictionary back 
        into an array. This function is important for the matrix-free 
        interface.

        Parameters
        ----------
        dIdxDict : dictionary
           Dictionary of information keyed by this object's
           design variables

        Returns
        -------
        dIdx : array
           Flattened array of length getNDV(). 
        """

        dIdx = numpy.zeros(self.getNDV(), self.dtype)
        i = 0
        for key in self.DV_listGlobal:
            dv = self.DV_listGlobal[key]
            dIdx[i:i+dv.nVal] = dIdxDict[dv.name]
            i += dv.nVal
        for key in self.DV_listLocal:
            dv = self.DV_listLocal[key]
            dIdx[i:i+dv.nVal] = dIdxDict[dv.name]
            i += dv.nVal
        return dIdx        

    def getVarNames(self):
        """
        Return a list of the design variable names. This is typically
        used when specifying a wrt= argument for pyOptSparse.

        Examples
        --------
        optProb.addCon(.....wrt=DVGeo.getVarNames())
        """
        names = list(self.DV_listGlobal.keys())
        names.extend(list(self.DV_listLocal.keys()))

        # Call the children recursively
        for iChild in xrange(len(self.children)):
            names.extend(self.children[iChild].getVarNames())

        return names

     
    def totalSensitivity(self, dIdpt, ptSetName, comm=None, child=False,
                         nDVStore=0, config=None):
        """
        This function computes sensitivty information.

        Specificly, it computes the following:
        :math:`\\frac{dX_{pt}}{dX_{DV}}^T \\frac{dI}{d_{pt}}

        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)

            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once. 

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place. 

        Returns
        -------
        dIdxDict : dic
            The dictionary containing the derivatives, suitable for
            pyOptSparse
            
        Notes
        -----
        The ``child`` and ``nDVStore`` options are only used
        internally and should not be changed by the user. 
        """

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = numpy.array([dIdpt])
        N = dIdpt.shape[0]

        # generate the total Jacobian self.JT
        if self.JT is None:
            self.computeTotalJacobian(ptSetName,config=config)

        # now that we have self.JT compute the Mat-Mat multiplication
        nDV = self._getNDV()
        dIdx_local = numpy.zeros((N, nDV), 'd')
        for i in range(N):
            dIdx_local[i,:] = self.JT.dot(dIdpt[i,:,:].flatten())

        if comm: # If we have a comm, globaly reduce with sum
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # Now convert to dict:
        dIdx = self.convertSensitivityToDict(dIdx)

        return dIdx

    def totalSensitivityProd(self, vec, ptSetName, comm=None, child=False,
                        nDVStore=0):
        """
        This function computes sensitivty information.

        Specifically, it computes the following:
        :math:`\\frac{dX_{pt}}{dX_{DV}} \\ vec

        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)

            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once. 

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place. 

        Returns
        -------
        dIdxDict : dic
            The dictionary containing the derivatives, suitable for
            pyOptSparse
            
        Notes
        -----
        The ``child`` and ``nDVStore`` options are only used
        internally and should not be changed by the user. 
        """

        self.computeTotalJacobian(ptSetName)

        names = self.getVarNames()
        newvec = numpy.zeros(self.getNDV(),self.dtype)
        i = 0
        for key in names:
            if key in self.DV_listGlobal:
                dv = self.DV_listGlobal[key]
            else:
                dv = self.DV_listLocal[key]

            if key in vec:
                newvec[i:i+dv.nVal] = vec[key]
            
            i += dv.nVal

        # perform the product
        if self.JT is None:
            xsdot = numpy.zeros((0, 3))
        else:
            xsdot = self.JT.T.dot(newvec)
            xsdot.reshape(len(xsdot)/3, 3)

        return xsdot

    def totalSensitivityTransProd(self, vec, ptSetName, comm=None, child=False,
                        nDVStore=0):
        """
        This function computes sensitivty information.

        Specifically, it computes the following:
        :math:`\\frac{dX_{pt}}{dX_{DV}}^T \\ vec

        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)

            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once. 

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place. 

        Returns
        -------
        dIdxDict : dic
            The dictionary containing the derivatives, suitable for
            pyOptSparse
            
        Notes
        -----
        The ``child`` and ``nDVStore`` options are only used
        internally and should not be changed by the user. 
        """

        self.computeTotalJacobian(ptSetName)
        
        # perform the product
        if self.JT == None:
            xsdot = numpy.zeros((0, 3))
        else:
            xsdot = self.JT.dot(numpy.ravel(vec))

        # Pack result into dictionary
        xsdict = {}
        names = self.getVarNames()
        i = 0
        for key in names:
            if key in self.DV_listGlobal:
                dv = self.DV_listGlobal[key]
            else:
                dv = self.DV_listLocal[key]
            xsdict[key] = xsdot[i:i+dv.nVal]
            i += dv.nVal

        return xsdict

    def computeDVJacobian(self,config=None):
        """ 
        return J_temp for a given config
        """
        # This is going to be DENSE in general
        self.J_attach = self._attachedPtJacobian(config=config)
        # This is the sparse jacobian for the local DVs that affect
        # Control points directly.
        if self.J_local is None:
            self.J_local = self._localDVJacobian(config=config)

        # HStack em'
        # Three different possibilities: 
        if self.J_attach is not None and self.J_local is None:
            J_temp = sparse.lil_matrix(self.J_attach)
        elif self.J_local is not None and self.J_attach is None:
            J_temp = sparse.lil_matrix(self.J_local)
        else:
            J_temp = sparse.hstack([self.J_attach, self.J_local], format='lil')

        return J_temp

    def computeTotalJacobian(self, ptSetName, config=None):
        """ Return the total point jacobian in CSR format since we
        need this for TACS"""
        self._finalize()
        self.curPtSet = ptSetName
        # if self.JT is not None and self.J_name == name: # Already computed
        #     return
  
        # compute the design variable Jacobian
        J_temp = self.computeDVJacobian(config=config)
        # This is the FINAL Jacobian for the current geometry
        # point. We need this to be a sparse matrix for TACS. 
        
        if self.FFD.embededVolumes[ptSetName].dPtdCoef is not None:
            dPtdCoef = self.FFD.embededVolumes[ptSetName].dPtdCoef.tocoo()
            # We have a slight problem...dPtdCoef only has the shape
            # functions, so it size Npt x Coef. We need a matrix of
            # size 3*Npt x 3*nCoef, where each non-zero entry of
            # dPtdCoef is replaced by value * 3x3 Identity matrix.

            # Extract IJV Triplet from dPtdCoef
            row = dPtdCoef.row
            col = dPtdCoef.col
            data = dPtdCoef.data

            new_row = numpy.zeros(3*len(row), 'int')
            new_col = numpy.zeros(3*len(row), 'int')
            new_data = numpy.zeros(3*len(row))

            # Loop over each entry and expand:
            for j in xrange(3):
                new_data[j::3] = data
                new_row[j::3] = row*3 + j
                new_col[j::3] = col*3 + j

            # Size of New Matrix:
            Nrow = dPtdCoef.shape[0]*3
            Ncol = dPtdCoef.shape[1]*3

            # Create new matrix in coo-dinate format and convert to csr
            new_dPtdCoef = sparse.coo_matrix(
                (new_data, (new_row, new_col)), shape=(Nrow, Ncol)).tocsr()
            # Do Sparse Mat-Mat multiplaiction and resort indices
            self.JT = (J_temp.T*new_dPtdCoef.T).tocsr()
            self.JT.sort_indices()

            # Add in child portion
            for iChild in xrange(len(self.children)):
                
                # reset control points on child for child link derivatives
                self.children[iChild].FFD.coef = self.FFD.getAttachedPoints(
                    'child%d_coef'%(iChild))

                self.children[iChild].coef = self.FFD.getAttachedPoints(
                    'child%d_axis'%(iChild))
                self.children[iChild].refAxis.coef = (
                    self.children[iChild].coef.copy())
                self.children[iChild].refAxis._updateCurveCoef()
                self.children[iChild].computeTotalJacobian(ptSetName)

                self.JT = self.JT + self.children[iChild].JT
             
        else:
            self.JT = None


    def addVariablesPyOpt(self, optProb, globalVars=True, localVars=True, 
                          ignoreVars=None, freezeVars=None):
        """
        Add the current set of variables to the optProb object.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which variables are added

        globalVars : bool
            Flag specifying whether global variables are to be added

        localVars : bool
            Flag specifying whether local variables are to be added
            
        ignoreVars : list of strings
            List of design variables the user DOESN'T want to use 
            as optimization variables. 

        freezeVars : list of string
            List of design variables the user WANTS to add as optimization
            variables, but to have the lower and upper bounds set at the current
            variable. This effectively eliminates the variable, but it the variable
            is still part of the optimization.
        """
        if ignoreVars is None:
            ignoreVars = set()
        if freezeVars is None:
            freezeVars = set()

        # Add design variables from the master:
        varLists = {'globalVars':self.DV_listGlobal,
                   'localVars':self.DV_listLocal}
        for lst in varLists:
            if lst == 'globalVars' and globalVars or lst=='localVars' and localVars:
                for key in varLists[lst]:
                    if key not in ignoreVars:
                        dv = varLists[lst][key]
                        if key not in freezeVars:
                            optProb.addVarGroup(dv.name, dv.nVal, 'c', value=dv.value,
                                                lower=dv.lower, upper=dv.upper,
                                                scale=dv.scale)
                        else:
                            optProb.addVarGroup(dv.name, dv.nVal, 'c', value=dv.value,
                                                lower=dv.value, upper=dv.value,
                                                scale=dv.scale)
                    
        # Add variables from the children
        for child in self.children:
            child.addVariablesPyOpt(optProb, globalVars, localVars, 
                                    ignoreVars, freezeVars)

    def writeTecplot(self, fileName):
        """Write the (deformed) current state of the FFD's to a tecplot file, 
        including the children

        Parameters
        ----------
        fileName : str
           Filename for tecplot file. Should have a .dat extension
        """

        # Name here doesn't matter, just take the first one
        if len(self.points)>0:
            self.update(self.points.keys()[0], childDelta=False)

        f = pySpline.openTecplot(fileName, 3)
        vol_counter = 0
        # Write master volumes:
        vol_counter += self._writeVols(f, vol_counter)

        # Write children volumes:
        for iChild in xrange(len(self.children)):
            vol_counter += self.children[iChild]._writeVols(f, vol_counter)

        pySpline.closeTecplot(f)
        if len(self.points)>0:
            self.update(self.points.keys()[0], childDelta=True) 
        
    def writeRefAxes(self, fileName):
        """Write the (deformed) current state of the RefAxes to a tecplot file, 
        including the children

        Parameters
        ----------
        fileName : str
           Filename for tecplot file. Should have a no extension,an
           extension will be added.
        """
        # Name here doesnt matter, just take the first one
        self.update(self.points.keys()[0], childDelta=False)
        
        gFileName = fileName+'_parent.dat'
        if not len(self.axis)==0:
            self.refAxis.writeTecplot(gFileName, orig=True, curves=True, coef=True)
        # Write children axes:
        for iChild in xrange(len(self.children)):
            cFileName = fileName+'_child%3d.dat'%iChild
            self.children[iChild].refAxis.writeTecplot(cFileName, orig=True, curves=True, coef=True)

    def writePointSet(self,name,fileName):
        """
        Write a given point set to a tecplot file

        Parameters
        ----------
        name : str
             The name of the point set to write to a file
        fileName : str
           Filename for tecplot file. Should have a no extension,an
           extension will be added
        """
        coords = self.update(name, childDelta=False)
        fileName = fileName+'_%s.dat'%name
        f = pySpline.openTecplot(fileName, 3)
        pySpline.writeTecplot1D(f, name, coords)
        pySpline.closeTecplot(f)

    def writePlot3d(self, fileName):
        """Write the (deformed) current state of the FFD object into a
        plot3D file. This file could then be used as the base-line FFD
        for a subsequent optimization. This function is not typically
        used in a regular basis, but may be useful in certain
        situaions, i.e. a sequence of optimizations

        Parameters
        ----------
        fileName : str
            Filename of the plot3D file to write. Should have a .fmt
            file extension.
            """
        self.FFD.writePlot3dCoef(fileName)

    def getLocalIndex(self, iVol):
        """ Return the local index mapping that points to the global
        coefficient list for a given volume"""
        return self.FFD.topo.lIndex[iVol].copy()
        
# ----------------------------------------------------------------------
#        THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER
# ----------------------------------------------------------------------
    
    def _finalizeAxis(self):
        """
        Internal function that sets up the collection of curve that
        the user has added one at a time. This will create the
        internal pyNetwork object
        """
        if len(self.axis) == 0:
            return

        curves = []
        for axis in self.axis:
            curves.append(self.axis[axis]['curve'])
        
        # Setup the network of reference axis curves
        self.refAxis = pyNetwork(curves)
        # These are the rotations
        self.rot_x = OrderedDict()
        self.rot_y = OrderedDict()
        self.rot_z = OrderedDict()
        self.rot_theta = OrderedDict()
        self.scale = OrderedDict()
        self.scale_x = OrderedDict()
        self.scale_y = OrderedDict()
        self.scale_z = OrderedDict()
        self.coef = self.refAxis.coef # pointer
        self.coef0 = self.coef.copy().astype(self.dtype)

        i = 0
        for key in self.axis:
            # curves in ref axis are indexed sequentially...this is ok
            # since self.axis is an ORDERED dict
            t = self.refAxis.curves[i].t
            k = self.refAxis.curves[i].k
            N = len(self.refAxis.curves[i].coef)
            z = numpy.zeros((N, 1), self.dtype)
            o = numpy.ones((N, 1), self.dtype)
            self.rot_x[key] = pySpline.Curve(t=t, k=k, coef=z.copy())
            self.rot_y[key] = pySpline.Curve(t=t, k=k, coef=z.copy())
            self.rot_z[key] = pySpline.Curve(t=t, k=k, coef=z.copy())
            self.rot_theta[key] = pySpline.Curve(t=t, k=k, coef=z.copy())
            self.scale[key] = pySpline.Curve(t=t, k=k, coef=o.copy())
            self.scale_x[key] = pySpline.Curve(t=t, k=k, coef=o.copy())
            self.scale_y[key] = pySpline.Curve(t=t, k=k, coef=o.copy())
            self.scale_z[key] = pySpline.Curve(t=t, k=k, coef=o.copy())
            i += 1
            
        # Need to keep track of initail scale values
        self.scale0 = self.scale.copy()
        self.scale_x0 = self.scale_x.copy()
        self.scale_y0 = self.scale_y.copy()
        self.scale_z0 = self.scale_z.copy()
        self.rot_x0 = self.rot_x.copy()
        self.rot_y0 = self.rot_y.copy()
        self.rot_z0 = self.rot_z.copy()
        self.rot_theta0 = self.rot_theta.copy()

    def _finalize(self):
        if self.finalized:
            return
        self._finalizeAxis()
        if len(self.axis) == 0:
            self.finalized = True
            self.nPtAttachFull = len(self.FFD.coef)
            return
        # What we need to figure out is which of the control points
        # are connected to an axis, and which ones are not connected
        # to an axis. 

        # Retrieve all the pointset masks
        coefMask = numpy.zeros(len(self.FFD.coef), dtype=bool)
        for ptName in self.masks:
            coefMask += self.masks[ptName] # This is boolean addition.

        self.ptAttachInd = []
        self.ptAttach = []
        curveIDs = []
        s = []
        curveID = 0
        # Loop over the axis we have:
        for key in self.axis:

            vol_list = numpy.atleast_1d(self.axis[key]['volumes']).astype('intc')
            temp = []
            for iVol in vol_list:
                for i in xrange(self.FFD.vols[iVol].nCtlu):
                    for j in xrange(self.FFD.vols[iVol].nCtlv):
                        for k in xrange(self.FFD.vols[iVol].nCtlw):
                            ind = self.FFD.topo.lIndex[iVol][i, j, k]
                            if coefMask[ind] == False:
                                temp.append(ind)

            # Unique the values and append to the master list
            curPtAttach = geo_utils.unique(temp)
            self.ptAttachInd.extend(curPtAttach)

            curPts = self.FFD.coef.take(curPtAttach, axis=0).real
            self.ptAttach.extend(curPts)

            # Now do the projections for *just* the axis defined by my
            # key. 
            if self.axis[key]['axis'] is None:
                tmpIDs, tmpS0 = self.refAxis.projectPoints(curPts, curves=[curveID])
            else:
                tmpIDs, tmpS0 = self.refAxis.projectRays(
                    curPts, self.axis[key]['axis'], curves=[curveID])

            curveIDs.extend(tmpIDs)
            s.extend(tmpS0)
            curveID += 1

        self.ptAttachFull = self.FFD.coef.copy().real
        self.nPtAttach = len(self.ptAttach)
        self.nPtAttachFull = len(self.ptAttachFull)

        self.curveIDs = curveIDs
        self.curveIDNames = []
        axisKeys = list(self.axis.keys())
        for i in range(len(curveIDs)):
            self.curveIDNames.append(axisKeys[self.curveIDs[i]])

        self.links_s = numpy.array(s)
        self.links_x = []
        self.links_n = []

        for i in xrange(self.nPtAttach):
            self.links_x.append(
                self.ptAttach[i] - \
                    self.refAxis.curves[self.curveIDs[i]](s[i]))
            deriv = self.refAxis.curves[
                self.curveIDs[i]].getDerivative(self.links_s[i])
            deriv /= geo_utils.euclideanNorm(deriv) # Normalize
            self.links_n.append(numpy.cross(deriv, self.links_x[-1]))

        self.links_x = numpy.array(self.links_x)
        self.links_s = numpy.array(self.links_s)
        self.finalized = True

    def _setInitialValues(self):
        if len(self.axis) > 0:
            self.coef[:,:] = copy.deepcopy(self.coef0)
            for key in self.axis:
                self.scale[key].coef[:] = copy.deepcopy(self.scale0[key].coef)
                self.scale_x[key].coef[:] = copy.deepcopy(self.scale_x0[key].coef)
                self.scale_y[key].coef[:] = copy.deepcopy(self.scale_y0[key].coef)
                self.scale_z[key].coef[:] = copy.deepcopy(self.scale_z0[key].coef)      
                self.rot_x[key].coef[:] = copy.deepcopy(self.rot_x0[key].coef) 
                self.rot_y[key].coef[:] = copy.deepcopy(self.rot_y0[key].coef)
                self.rot_z[key].coef[:] = copy.deepcopy(self.rot_z0[key].coef)
                self.rot_theta[key].coef[:] = copy.deepcopy(self.rot_theta0[key].coef)

    def _getRotMatrix(self, rotX, rotY, rotZ, rotType):
        if rotType == 1:
            D = numpy.dot(rotZ, numpy.dot(rotY, rotX))
        elif rotType == 2:
            D = numpy.dot(rotY, numpy.dot(rotZ, rotX))
        elif rotType == 3:
            D = numpy.dot(rotX, numpy.dot(rotZ, rotY))
        elif rotType == 4:
            D = numpy.dot(rotZ, numpy.dot(rotX, rotY))
        elif rotType == 5:
            D = numpy.dot(rotY, numpy.dot(rotX, rotZ))
        elif rotType == 6:
            D = numpy.dot(rotX, numpy.dot(rotY, rotZ))
        elif rotType == 7:
            D = numpy.dot(rotY, numpy.dot(rotX, rotZ))

        return D

    def _getNDV(self):
        """Return the actual number of design variables, global +
        local
        """
        return self._getNDVGlobal() + self._getNDVLocal()

    def getNDV(self):
        """
        Return the total number of design variables this object has.

        Returns
        -------
        nDV : int
            Total number of design variables
            """
        return self._getNDV()

    def _getNDVGlobal(self):
        """
        Get total number of global variables, inclding any children
        """
        nDV = 0
        for key in self.DV_listGlobal:
            nDV += self.DV_listGlobal[key].nVal
        
        for child in self.children:
            nDV += child._getNDVGlobal()

        return nDV

    def _getNDVLocal(self):
        """
        Get total number of local variables, inclding any children
        """
        nDV = 0
        for key in self.DV_listLocal:
            nDV += self.DV_listLocal[key].nVal

        for child in self.children:
            nDV += child._getNDVLocal()

        return nDV

    def _getNDVSelf(self):
        """
        Get total number of local and global variables, not including
        children
        """
        return self._getNDVGlobalSelf() + self._getNDVLocalSelf()

    def _getNDVGlobalSelf(self):
        """
        Get total number of global variables, not including
        children
        """
        nDV = 0
        for key in self.DV_listGlobal:
            nDV += self.DV_listGlobal[key].nVal
        
        return nDV

    def _getNDVLocalSelf(self):
        """
        Get total number of local variables, not including
        children
        """
        nDV = 0
        for key in self.DV_listLocal:
            nDV += self.DV_listLocal[key].nVal

        return nDV
        
    def _getDVOffsets(self):
        '''
        return the global and local DV offsets for this FFD
        '''

        # figure out the split between local and global Variables
        # All global Vars at all levels come first 
        # then all Local Vars. 
        # Parent Vars come before child Vars
        nDVTotal = self._getNDV()
        nDVLocalTotal = self._getNDVLocal()
        nDVGlobTotal = self._getNDVGlobal() 

        if self.isChild:
            if self.dXrefdXdvg is not None:
                nDVGlobSummed = self.dXrefdXdvg.shape[1]
            else:
                nDVGlobSummed = 0
            if self.dXrefdXdvl is not None:
                nDVLocalSummed = self.dXrefdXdvl.shape[1]
            else:
                nDVLocalSummed = 0
            DVCountGlobal =  nDVGlobSummed-nDVGlobTotal   
            DVCountLocal = nDVGlobSummed+nDVLocalSummed-nDVLocalTotal
        else:
            nDVGlobSummed = nDVGlobTotal
            nDVLocalSummed = nDVLocalTotal
            DVCountGlobal=0
            DVCountLocal=nDVGlobSummed

        return DVCountGlobal,DVCountLocal
        
    def _update_deriv(self, iDV=0, h=1.0e-40j, oneoverh=1.0/1e-40, config=None):

        """Copy of update function for derivative calc"""
        new_pts = numpy.zeros((self.nPtAttach, 3), 'D')

        # Set all coef Values back to initial values
        if not self.isChild:
            self._setInitialValues()

        self._complexifyCoef()

        if self.isChild:
            self.links_x = self.links_x.astype('D')
            for ipt in xrange(self.nPtAttach):
                base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])
                self.links_x[ipt]=self.FFD.coef[self.ptAttachInd[ipt],:]-base_pt

        if len(self.axis) > 0:
            # Step 1: Call all the design variables
            for key in self.DV_listGlobal:
                self.DV_listGlobal[key](self, config)

            self.refAxis.coef = self.coef.copy()
            self.refAxis._updateCurveCoef()

            for ipt in xrange(self.nPtAttach):
                base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])

                scale = self.scale[self.curveIDNames[ipt]](self.links_s[ipt]) 
                scale_x = self.scale_x[self.curveIDNames[ipt]](self.links_s[ipt]) 
                scale_y = self.scale_y[self.curveIDNames[ipt]](self.links_s[ipt]) 
                scale_z = self.scale_z[self.curveIDNames[ipt]](self.links_s[ipt]) 

                rotType = self.axis[self.curveIDNames[ipt]]['rotType']
                if rotType == 0:
                    deriv = self.refAxis.curves[
                        self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                    deriv /= geo_utils.euclideanNorm(deriv) # Normalize
                    new_vec = -numpy.cross(deriv, self.links_n[ipt])
                    new_vec = geo_utils.rotVbyW(new_vec, deriv, self.rot_x[
                            self.curveIDNames[ipt]](self.links_s[ipt])*numpy.pi/180)
                    new_pts[ipt] = base_pt + new_vec*scale

                else:
                    rotX = geo_utils.rotxM(self.rot_x[
                            self.curveIDNames[ipt]](self.links_s[ipt]))
                    rotY = geo_utils.rotyM(self.rot_y[
                            self.curveIDNames[ipt]](self.links_s[ipt]))
                    rotZ = geo_utils.rotzM(self.rot_z[
                            self.curveIDNames[ipt]](self.links_s[ipt]))

                    D = self.links_x[ipt]

                    rotM = self._getRotMatrix(rotX, rotY, rotZ, rotType)
                    D = numpy.dot(rotM, D)

                    if rotType == 7:
                        # only apply the theta rotations in certain cases
                        deriv = self.refAxis.curves[
                            self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                        deriv[0] = 0.0

                        deriv /= geo_utils.euclideanNorm(deriv) # Normalize

                        D = geo_utils.rotVbyW(D,deriv,numpy.pi/180*self.rot_theta[              
                                self.curveIDNames[ipt]](self.links_s[ipt]))

                    D[0] *= scale_x
                    D[1] *= scale_y
                    D[2] *= scale_z

                    new_pts[ipt] = base_pt + D*scale

            # set the forward effect of the global design vars in each child
            for iChild in xrange(len(self.children)):

                dXrefdCoef = self.FFD.embededVolumes['child%d_axis'%(iChild)].dPtdCoef
                dCcdCoef   = self.FFD.embededVolumes['child%d_coef'%(iChild)].dPtdCoef

                tmp = numpy.zeros(self.FFD.coef.shape,dtype='D')

                numpy.put(tmp[:, 0], self.ptAttachInd, 
                          numpy.imag(new_pts[:,0])*oneoverh)
                numpy.put(tmp[:, 1], self.ptAttachInd, 
                          numpy.imag(new_pts[:,1])*oneoverh)
                numpy.put(tmp[:, 2], self.ptAttachInd, 
                          numpy.imag(new_pts[:,2])*oneoverh)

                dXrefdXdvg = numpy.zeros((dXrefdCoef.shape[0]*3),'D')
                dCcdXdvg   = numpy.zeros((dCcdCoef.shape[0]*3),'D')

                dXrefdXdvg[0::3] = dXrefdCoef.dot(tmp[:, 0])
                dXrefdXdvg[1::3] = dXrefdCoef.dot(tmp[:, 1])
                dXrefdXdvg[2::3] = dXrefdCoef.dot(tmp[:, 2])

                dCcdXdvg[0::3] = dCcdCoef.dot(tmp[:, 0])
                dCcdXdvg[1::3] = dCcdCoef.dot(tmp[:, 1])
                dCcdXdvg[2::3] = dCcdCoef.dot(tmp[:, 2])

                self.children[iChild].dXrefdXdvg[:, iDV] = dXrefdXdvg
                self.children[iChild].dCcdXdvg[:, iDV] = dCcdXdvg
                self.children[iChild].nChildren = len(self.children)

        return new_pts

    def _complexifyCoef(self):
        """Convert coef to complex temporarily"""
        if len(self.axis) > 0:
            for key in self.axis:
                self.rot_x[key].coef = self.rot_x[key].coef.astype('D')
                self.rot_y[key].coef = self.rot_y[key].coef.astype('D')
                self.rot_z[key].coef = self.rot_z[key].coef.astype('D')
                self.rot_theta[key].coef = self.rot_theta[key].coef.astype('D')

                self.scale[key].coef = self.scale[key].coef.astype('D')
                self.scale_x[key].coef = self.scale_x[key].coef.astype('D')
                self.scale_y[key].coef = self.scale_y[key].coef.astype('D')
                self.scale_z[key].coef = self.scale_z[key].coef.astype('D')

            for i in range(self.refAxis.nCurve):
                self.refAxis.curves[i].coef = (
                    self.refAxis.curves[i].coef.real.astype('D'))
            self.coef = self.coef.astype('D')
        
    def _unComplexifyCoef(self):
        """Convert coef back to reals"""
        if len(self.axis) > 0 and not self.complex:
            for key in self.axis:
                self.rot_x[key].coef = self.rot_x[key].coef.real.astype('d')
                self.rot_y[key].coef = self.rot_y[key].coef.real.astype('d')
                self.rot_z[key].coef = self.rot_z[key].coef.real.astype('d')
                self.rot_theta[key].coef = self.rot_theta[key].coef.real.astype('d')

                self.scale[key].coef = self.scale[key].coef.real.astype('d')
                self.scale_x[key].coef = self.scale_x[key].coef.real.astype('d')
                self.scale_y[key].coef = self.scale_y[key].coef.real.astype('d')
                self.scale_z[key].coef = self.scale_z[key].coef.real.astype('d')

            for i in range(self.refAxis.nCurve):
                self.refAxis.curves[i].coef = (
                    self.refAxis.curves[i].coef.real.astype('d'))
            self.coef = self.coef.real.astype('d')

    # def totalSensitivityFD(self, dIdpt, ptSetName, comm=None, nDV_T=None, DVParent=0):
    #     """This function takes the total derivative of an objective, 
    #     I, with respect the points controlled on this processor using FD.
    #     We take the transpose prodducts and mpi_allreduce them to get the
    #     resulting value on each processor. Note that this function is slow
    #     and should eventually be replaced by an analytic version.
    #     """
    #     if self.isChild:
    #         refFFDCoef = copy.copy(self.FFD.coef)
    #     # end if

    #     coords0 = self.update(name).flatten()

    #     h = 1e-6
        
    #     # count up number of DVs
    #     nDV = self._getNDVGlobal() 
    #     if nDV_T==None:
    #         nDV_T = self._getNDV()
    #     # end
    #     dIdx = numpy.zeros(nDV_T)
    #     if self.isChild:
    #         #nDVSummed = self.dXrefdXdvg.shape[1]
    #         DVCount=DVParent#nDVSummed-nDV
    #         DVLocalCount = DVParent+nDV
    #     else:
    #         #nDVSummed = nDV
    #         DVCount=0
    #         DVLocalCount = nDV
    #     # end if

    #     for i in xrange(len(self.DV_listGlobal)):
    #         for j in xrange(self.DV_listGlobal[i].nVal):
    #             if self.isChild:
    #                 self.FFD.coef=  refFFDCoef.copy()
    #             # end if

    #             refVal = self.DV_listGlobal[i].value[j]

    #             self.DV_listGlobal[i].value[j] += h

    #             coordsph = self.update(name).flatten()

    #             deriv = (coordsph-coords0)/h
    #             dIdx[DVCount]=numpy.dot(dIdpt.flatten(),deriv)
    #             DVCount += 1
    #             self.DV_listGlobal[i].value[j] = refVal
    #         # end for
    #     # end for
    #     DVparent=DVCount
        
    #     for i in xrange(len(self.DV_listLocal)):
    #         for j in xrange(self.DV_listLocal[i].nVal):

    #             refVal = self.DV_listLocal[i].value[j]

    #             self.DV_listLocal[i].value[j] += h
    #             coordsph = self.update(name).flatten()

    #             deriv = (coordsph-coords0)/h
    #             dIdx[DVLocalCount]=numpy.dot(dIdpt.flatten(),deriv)
    #             DVLocaLCount += 1
    #             self.DV_listLocal[i].value[j] = refVal
    #         # end for
    #     # end for
        
    #     # reset coords
    #     self.update(name)
        
    #     for iChild in xrange(len(self.children)):
            
    #         self.children[iChild].FFD.coef = self.FFD.getAttachedPoints(
    #             'child%d_coef'%(iChild))
            
    #         self.children[iChild].coef = self.FFD.getAttachedPoints(
    #             'child%d_axis'%(iChild))
    #         self.children[iChild].refAxis.coef =  self.children[iChild].coef.copy()
    #         self.children[iChild].refAxis._updateCurveCoef()

    #     for child in self.children:
    #         dIdx+=child.totalSensitivityFD(dIdpt, comm, name,nDV_T,DVParent)

    #     return dIdx

    # def computeTotalJacobianFD(self, ptSetName, comm=None, nDV_T = None,DVParent=0):
    #     """This function takes the total derivative of an objective, 
    #     I, with respect the points controlled on this processor using FD.
    #     We take the transpose prodducts and mpi_allreduce them to get the
    #     resulting value on each processor. Note that this function is slow
    #     and should eventually be replaced by an analytic version.
    #     """
    #     if self.isChild:
    #         refFFDCoef = copy.copy(self.FFD.coef)
    #     # end if

    #     coords0 = self.update(name).flatten()

    #     h = 1e-6
        
    #     # count up number of DVs
    #     nDV = self._getNDVGlobal() 
    #     if nDV_T==None:
    #         nDV_T = self._getNDV()
    #     # end
    #     dPtdx = numpy.zeros([coords0.shape[0],nDV_T])
    #     if self.isChild:
    #         #nDVSummed = self.dXrefdXdvg.shape[1]
    #         DVCount=DVParent#nDVSummed-nDV
    #         DVLocalCount = DVParent+nDV
    #     else:
    #         #nDVSummed = nDV
    #         DVCount=0
    #         DVLocalCount = nDV
    #     # end if

    #     for i in xrange(len(self.DV_listGlobal)):
    #         for j in xrange(self.DV_listGlobal[i].nVal):
    #             if self.isChild:
    #                 self.FFD.coef=  refFFDCoef.copy()
    #             # end if

    #             refVal = self.DV_listGlobal[i].value[j]

    #             self.DV_listGlobal[i].value[j] += h

    #             coordsph = self.update(name).flatten()

    #             deriv = (coordsph-coords0)/h
    #             dPtdx[:,DVCount]=deriv

    #             DVCount += 1
    #             self.DV_listGlobal[i].value[j] = refVal
    #         # end for
    #     # end for
    #     DVParent=DVCount
        
    #     for i in xrange(len(self.DV_listLocal)):
    #         for j in xrange(self.DV_listLocal[i].nVal):

    #             refVal = self.DV_listLocal[i].value[j]

    #             self.DV_listLocal[i].value[j] += h
    #             coordsph = self.update(name).flatten()

    #             deriv = (coordsph-coords0)/h
    #             dPtdx[:,DVLocalCount]=deriv

    #             DVLocaLCount += 1
    #             self.DV_listLocal[i].value[j] = refVal
    #         # end for
    #     # end for
        
    #     # reset coords
    #     self.update(name)
    #     for iChild in xrange(len(self.children)):
            
    #         self.children[iChild].FFD.coef = self.FFD.getAttachedPoints(
    #             'child%d_coef'%(iChild))
            
    #         self.children[iChild].coef = self.FFD.getAttachedPoints(
    #             'child%d_axis'%(iChild))
    #         self.children[iChild].refAxis.coef =  self.children[iChild].coef.copy()
    #         self.children[iChild].refAxis._updateCurveCoef()
    #     # end
    #     for child in self.children:

    #         dPtdx+=child.computeTotalJacobianFD(comm, name, nDV_T, DVParent)
    #     # end for

    #     return dPtdx
 
    def _attachedPtJacobian(self, config):
        """
        Compute the derivative of the the attached points
        """
        nDV = self._getNDVGlobal()
    
        if self.dXrefdXdvg is not None:
            nDVSummed = self.dXrefdXdvg.shape[1]
        else:
            nDVSummed = nDV
            self.rangeg=None
            iDV = 0
            for key in self.DV_listGlobal:
                nVal = self.DV_listGlobal[key].nVal
                for j in xrange(nVal):
                    iDV+=1

            for iChild in xrange(len(self.children)):
                self.children[iChild].startDVg=iDV
                childnDV = self.children[iChild]._getNDVGlobal()
                iDV+=childnDV

        if nDVSummed == 0:
            return None

        h = 1.0e-40j
        oneoverh = 1.0/1e-40

        # h = 1.0e-6
        # oneoverh = 1.0/1e-6
        # coordref = self._update_deriv().flatten()
        # Just do a CS loop over the coef
        # First sum the actual number of globalDVs
        if nDV != 0:
            Jacobian = numpy.zeros((self.nPtAttachFull*3, nDV))
            # Create the storage arrays for the information that must be
            # passed to the children

            for iChild in xrange(len(self.children)):
                N = self.FFD.embededVolumes['child%d_axis'%(iChild)].N
                self.children[iChild].dXrefdXdvg = numpy.zeros((N*3, nDV))

                N = self.FFD.embededVolumes['child%d_coef'%(iChild)].N
                self.children[iChild].dCcdXdvg = numpy.zeros((N*3, nDV))
                self.children[iChild].rangeg = numpy.zeros(nDV)

            iDV = 0
            for key in self.DV_listGlobal:
                if self.DV_listGlobal[key].config is None or config in self.DV_listGlobal[key].config:
                    nVal = self.DV_listGlobal[key].nVal
                    for j in xrange(nVal):

                        refVal = self.DV_listGlobal[key].value[j]

                        self.DV_listGlobal[key].value[j] += h

                        deriv = oneoverh*numpy.imag(self._update_deriv(iDV,h,oneoverh,config=config)).flatten()
                        #deriv = oneoverh*(self._update_deriv().flatten()-coordref)

                        numpy.put(Jacobian[0::3, iDV], self.ptAttachInd, 
                                  deriv[0::3])
                        numpy.put(Jacobian[1::3, iDV], self.ptAttachInd, 
                                deriv[1::3])
                        numpy.put(Jacobian[2::3, iDV], self.ptAttachInd, 
                                  deriv[2::3])

                        # # save the parent to child jacobians in the child
                        # for iChild in xrange(len(self.children)):

                        #     dXrefdCoef = self.FFD.embededVolumes['child%d_axis'%(iChild)].dPtdCoef
                        #     dCcdCoef   = self.FFD.embededVolumes['child%d_coef'%(iChild)].dPtdCoef

                        #     tmp = numpy.zeros(self.FFD.coef.shape,dtype='D')

                        #     tmp[pt_dv[0],pt_dv[1]] = 1.0

                        #     dXrefdXdvl = numpy.zeros((dXrefdCoef.shape[0]*3),'D')
                        #     dCcdXdvl   = numpy.zeros((dCcdCoef.shape[0]*3),'D')

                        #     dXrefdXdvl[0::3] = dXrefdCoef.dot(tmp[:, 0])
                        #     dXrefdXdvl[1::3] = dXrefdCoef.dot(tmp[:, 1])
                        #     dXrefdXdvl[2::3] = dXrefdCoef.dot(tmp[:, 2])

                        #     dCcdXdvl[0::3] = dCcdCoef.dot(tmp[:, 0])
                        #     dCcdXdvl[1::3] = dCcdCoef.dot(tmp[:, 1])
                        #     dCcdXdvl[2::3] = dCcdCoef.dot(tmp[:, 2])

                        #     self.children[iChild].dXrefdXdvl[:, iDVLocal] = dXrefdXdvl
                        #     self.children[iChild].dCcdXdvl[:, iDVLocal] = dCcdXdvl
                        #     # print('dXref', self.children[iChild].dXrefdXdvl)
                        #     # print('dCc', self.children[iChild].dCcdXdvl)

                        # save the global DV Ranges for the children
                        if self.rangeg==None:
                            for iChild in xrange(len(self.children)):
                                self.children[iChild].rangeg[iDV]=self.DV_listGlobal[key].scale[j]
                        else:
                            for iChild in xrange(len(self.children)):
                                self.children[iChild].rangeg[iDV]=self.rangeg[iDV]

                        iDV += 1

                        self.DV_listGlobal[key].value[j] = refVal
                else:
                    iDV += self.DV_listGlobal[key].nVal
        else:
            Jacobian = None

        if self.dXrefdXdvg is not None:
            # we are now on a child. Add in dependence passed from parent
            temp = numpy.zeros((self.nPtAttachFull*3, nDVSummed))
            startIdx = self.startDVg#nDVSummed-self.nChildren+self.iChild
            endIdx = startIdx+nDV#nDVSummed-self.nChildren+self.iChild+nDV
            #temp[:, nDVSummed - nDV:] = Jacobian
            temp[:, startIdx:endIdx] = Jacobian

            Jacobian = temp

            for iDV in xrange(self.dXrefdXdvg.shape[1]):
                
                self._complexifyCoef()
                self.coef[:,0] +=  self.dXrefdXdvg[0::3, iDV]*h
                self.coef[:,1] +=  self.dXrefdXdvg[1::3, iDV]*h
                self.coef[:,2] +=  self.dXrefdXdvg[2::3, iDV]*h

                self.refAxis.coef = self.coef.copy()
                self.refAxis._updateCurveCoef()

                self.FFD.coef = self.FFD.coef.astype('D')
                tmp1 =numpy.zeros_like(self.FFD.coef,dtype='D')
                tmp1[:,0] = self.dCcdXdvg[0::3, iDV]*h
                tmp1[:,1] = self.dCcdXdvg[1::3, iDV]*h
                tmp1[:,2] = self.dCcdXdvg[2::3, iDV]*h

                self.FFD.coef+=tmp1

                new_pts_child = self._update_deriv(iDV,h,oneoverh, config=config)
                tmp2 = numpy.zeros(self.nPtAttachFull*3,dtype='D')
                numpy.put(tmp2[0::3], self.ptAttachInd, new_pts_child[:,0])
                numpy.put(tmp2[1::3], self.ptAttachInd, new_pts_child[:,1])
                numpy.put(tmp2[2::3], self.ptAttachInd, new_pts_child[:,2])

                tmp3 = numpy.zeros(self.nPtAttachFull*3,dtype='d')
                for index in self.ptAttachInd:
                    for j in xrange(3):
                        idx = index*3+j
                        tmp3[idx]=self.dCcdXdvg[idx,iDV]
               
                Jacobian[:, iDV] += oneoverh*numpy.imag(tmp2)-tmp3
                self.coef = self.coef.astype('d')
                self.FFD.coef = self.FFD.coef.astype('d')

        self._unComplexifyCoef()

        return Jacobian

    def _localDVJacobian(self, config=None):
        """
        Return the derivative of the coefficients wrt the local design 
        variables
        """
        
        # This is relatively straight forward, since the matrix is
        # entirely one's or zeros
        nDV = self._getNDVLocal()

        if self.dXrefdXdvl is not None:
            nDVSummed = self.dXrefdXdvl.shape[1]
        else:
            nDVSummed = nDV
            self.rangel = None
            iDV = 0
            for key in self.DV_listLocal:
                nVal = self.DV_listLocal[key].nVal
                for j in xrange(nVal):
                    iDV+=1

            for iChild in xrange(len(self.children)):
                self.children[iChild].startDVl=iDV
                childnDV = self.children[iChild]._getNDVLocal()
                iDV+=childnDV

        if nDVSummed == 0:
            return None
        h = 1.0e-40j
        oneoverh = 1.0/1e-40

        if nDV != 0:
            Jacobian = sparse.lil_matrix((self.nPtAttachFull*3, nDV))
         
            # Create the storage arrays for the information that must be
            # passed to the children

            for iChild in xrange(len(self.children)):
                N = self.FFD.embededVolumes['child%d_axis'%(iChild)].N
                self.children[iChild].dXrefdXdvl = numpy.zeros((N*3, nDV))

                N = self.FFD.embededVolumes['child%d_coef'%(iChild)].N
                self.children[iChild].dCcdXdvl = numpy.zeros((N*3, nDV))
                self.children[iChild].rangel = numpy.zeros(nDV)

            iDVLocal = 0
            for key in self.DV_listLocal:

                if self.DV_listLocal[key].config is None or config in self.DV_listLocal[key].config:

                    nVal = self.DV_listLocal[key].nVal
                    for j in xrange(nVal):

                        pt_dv = self.DV_listLocal[key].coefList[j] 
                        irow = pt_dv[0]*3 + pt_dv[1]
                        Jacobian[irow, iDVLocal] = 1.0

                        for iChild in xrange(len(self.children)):

                            dXrefdCoef = self.FFD.embededVolumes['child%d_axis'%(iChild)].dPtdCoef
                            dCcdCoef   = self.FFD.embededVolumes['child%d_coef'%(iChild)].dPtdCoef

                            tmp = numpy.zeros(self.FFD.coef.shape,dtype='D')

                            tmp[pt_dv[0],pt_dv[1]] = 1.0

                            dXrefdXdvl = numpy.zeros((dXrefdCoef.shape[0]*3),'D')
                            dCcdXdvl   = numpy.zeros((dCcdCoef.shape[0]*3),'D')

                            dXrefdXdvl[0::3] = dXrefdCoef.dot(tmp[:, 0])
                            dXrefdXdvl[1::3] = dXrefdCoef.dot(tmp[:, 1])
                            dXrefdXdvl[2::3] = dXrefdCoef.dot(tmp[:, 2])

                            dCcdXdvl[0::3] = dCcdCoef.dot(tmp[:, 0])
                            dCcdXdvl[1::3] = dCcdCoef.dot(tmp[:, 1])
                            dCcdXdvl[2::3] = dCcdCoef.dot(tmp[:, 2])

                            self.children[iChild].dXrefdXdvl[:, iDVLocal] = dXrefdXdvl
                            self.children[iChild].dCcdXdvl[:, iDVLocal] = dCcdXdvl
                        # if scaled:
                        #     if self.rangel is None:
                        #         for iChild in xrange(len(self.children)):
                        #             self.children[iChild].rangel[iDVLocal] = self.DV_listLocal[key].scale[j]
                        #     else:
                        #         for iChild in xrange(len(self.children)):
                        #             self.children[iChild].rangel[iDVLocal] = self.rangel[iDVLocal]
                                # end for
                            # end if
                        # end if
                        iDVLocal += 1
                else:
                    iDVLocal += self.DV_listLocal[key].nVal

                # end if config check
            # end for
        else:
            Jacobian = None

        if self.dXrefdXdvl is not None:
            #temp = sparse.lil_matrix((self.nPtAttachFull*3, nDVSummed))
            temp = numpy.zeros((self.nPtAttachFull*3, nDVSummed))
            if Jacobian is not None:
                startIdx = self.startDVl#nDVSummed-self.nChildren+self.iChild
                endIdx = startIdx+nDV#nDVSummed-self.nChildren+self.iChild+nDV
                temp[:, startIdx:endIdx] = Jacobian.todense()
                #temp[:, nDVSummed - nDV:] = Jacobian

            Jacobian = temp

            for iDV in xrange(self.dXrefdXdvl.shape[1]):
                self._complexifyCoef()
                self.coef[:,0] +=  self.dXrefdXdvl[0::3, iDV]*h
                self.coef[:,1] +=  self.dXrefdXdvl[1::3, iDV]*h
                self.coef[:,2] +=  self.dXrefdXdvl[2::3, iDV]*h

                self.refAxis.coef = self.coef.copy()
                self.refAxis._updateCurveCoef()

                self.FFD.coef = self.FFD.coef.astype('D')
                tmp1 =numpy.zeros_like(self.FFD.coef,dtype='D')
                tmp1[:,0] = self.dCcdXdvl[0::3, iDV]*h
                tmp1[:,1] = self.dCcdXdvl[1::3, iDV]*h
                tmp1[:,2] = self.dCcdXdvl[2::3, iDV]*h

                #self.FFD.coef+=tmp1

                new_pts_child = self._update_deriv(iDV, h, oneoverh, config=config)

                tmp2 = numpy.zeros(self.nPtAttachFull*3,dtype='D')
                numpy.put(tmp2[0::3], self.ptAttachInd, new_pts_child[:,0])
                numpy.put(tmp2[1::3], self.ptAttachInd, new_pts_child[:,1])
                numpy.put(tmp2[2::3], self.ptAttachInd, new_pts_child[:,2])

                tmp3 = numpy.zeros(self.nPtAttachFull*3,dtype='d')
                for index in self.ptAttachInd:
                    for j in xrange(3):
                        idx = index*3+j
                        tmp3[idx]=self.dCcdXdvl[idx,iDV]
                        # end
                    # end
                # end
                Jacobian[:, iDV] = Jacobian[:, iDV] + oneoverh*numpy.imag(tmp2)-tmp3
                self.coef = self.coef.astype('d')
                self.FFD.coef = self.FFD.coef.astype('d')
            # end for
        # end if
        self._unComplexifyCoef()
                              
        return sparse.csr_matrix(Jacobian)

    def _writeVols(self, handle, vol_counter):
        for i in xrange(len(self.FFD.vols)):
            pySpline.writeTecplot3D(handle, 'vol%d'%i, self.FFD.vols[i].coef)
            vol_counter += 1

        return vol_counter

    def checkDerivatives(self, ptSetName):
        """
        Run a brute force FD check on ALL design variables

        Parameters
        ----------
        ptSetName : str
            name of the point set to check
            """

        print('Computing Analytic Jacobian...')
        self.zeroJacobians()
        for child in self.children:
            child.zeroJacobians()
        # self.JT = None # J is no longer up to date
        # self.J_name = None # Name is no longer defined
        # self.J_attach = None
        # self.J_local = None
        self.computeTotalJacobian(ptSetName)
       
        Jac = copy.deepcopy(self.JT)
        
        # Global Variables
        print('========================================')
        print('             Global Variables           ')
        print('========================================')
                 
        if self.isChild:
            refFFDCoef = copy.copy(self.FFD.coef)
            refCoef = copy.copy(self.coef)

        coords0 = self.update(ptSetName).flatten()

        h = 1e-6

        # figure out the split between local and global Variables
        DVCountGlob,DVCountLoc = self._getDVOffsets()

        #        DVCount = 0
        for key in self.DV_listGlobal:
            for j in xrange(self.DV_listGlobal[key].nVal):

                print('========================================')
                print('      GlobalVar(%s), Value(%d)'%(key, j))
                print('========================================')

                if self.isChild:
                    self.FFD.coef=  refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = self.coef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listGlobal[key].value[j]

                self.DV_listGlobal[key].value[j] += h

                coordsph = self.update(ptSetName).flatten()

                deriv = (coordsph-coords0)/h

                for ii in xrange(len(deriv)):

                    relErr = (deriv[ii] - Jac[DVCountGlob, ii])/(
                        1e-16 + Jac[DVCountGlob, ii])
                    absErr = deriv[ii] - Jac[DVCountGlob,ii]

                    #if abs(relErr) > h*10 and abs(absErr) > h*10:
                    print(ii, deriv[ii], Jac[DVCountGlob, ii], relErr, absErr)

                DVCountGlob += 1
                self.DV_listGlobal[key].value[j] = refVal

        for key in self.DV_listLocal:
            for j in xrange(self.DV_listLocal[key].nVal):

                print('========================================')
                print('      LocalVar(%s), Value(%d)           '%(key, j))
                print('========================================')
        
                if self.isChild:
                    self.FFD.coef=  refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = self.coef.copy()
                    self.refAxis._updateCurveCoef()
                
                refVal = self.DV_listLocal[key].value[j]

                self.DV_listLocal[key].value[j] += h
                coordsph = self.update(ptSetName).flatten()
             
                deriv = (coordsph-coords0)/h

                for ii in xrange(len(deriv)):
                    relErr = (deriv[ii] - Jac[DVCountLoc, ii])/(
                        1e-16 + Jac[DVCountLoc, ii])
                    absErr = deriv[ii] - Jac[DVCountLoc,ii]

                    #if abs(relErr) > h and abs(absErr) > h:
                    print(ii, deriv[ii], Jac[DVCountLoc, ii], relErr, absErr)

                DVCountLoc += 1
                self.DV_listLocal[key].value[j] = refVal

        for child in self.children:
            child.checkDerivatives(ptSetName)

    def printDesignVariables(self):
        """
        Print a formatted list of design variables to the screen
        """
        for dg in self.DV_listGlobal:
            print('%s'%(dg.name))
            for i in xrange(dg.nVal):
                print('%20.15f'%(dg.value[i]))
  
        for dl in self.DV_listLocal:
            print('%s'%(dl.name))
            for i in xrange(dl.nVal):
                print('%20.15f'%(dl.value[i]))
    
        for child in self.children:
            child.printDesignVariables()
  
class geoDVGlobal(object):
     
    def __init__(self, dv_name, value, lower, upper, scale, function, config):
        """Create a geometric design variable (or design variable group)
        See addGeoDVGlobal in DVGeometry class for more information
        """
        self.name = dv_name
        self.value = numpy.atleast_1d(numpy.array(value)).astype('D')
        self.nVal = len(self.value)
        self.lower = None
        self.upper = None
        self.config = config
        self.function = function
        if lower is not None:
            self.lower = _convertTo1D(lower, self.nVal)
        if upper is not None:
            self.upper = _convertTo1D(upper, self.nVal)
        if scale is not None:
            self.scale = _convertTo1D(scale, self.nVal)

    def __call__(self, geo, config):
        """When the object is called, actually apply the function"""
        # Run the user-supplied function
        d = numpy.dtype(complex)

        if self.config is None or config in self.config:
            # If the geo object is complex, which is indicated by .coef
            # being complex, run with complex numbers. Otherwise, convert
            # to real before calling. This eliminates casting warnings. 
            if geo.coef.dtype == d or geo.complex:
                return self.function(self.value, geo)
            else:
                return self.function(numpy.real(self.value), geo)
    
class geoDVLocal(object):
     
    def __init__(self, dvName, lower, upper, scale, axis, coefList, config):
        
        """Create a set of geometric design variables which change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.
        See addGeoDVLocal for more information
        """
        N = len(axis)
        self.nVal = len(coefList)*N
        self.value = numpy.zeros(self.nVal, 'D')
        self.name = dvName
        self.lower = None
        self.upper = None
        self.config = config
        if lower is not None:
            self.lower = _convertTo1D(lower, self.nVal)
        if upper is not None:
            self.upper = _convertTo1D(upper, self.nVal)
        if scale is not None:
            self.scale = _convertTo1D(scale, self.nVal)
        
        self.coefList = numpy.zeros((self.nVal, 2), 'intc')
        j = 0

        for i in xrange(len(coefList)):
            if 'x' in axis.lower():
                self.coefList[j] = [coefList[i], 0]
                j += 1
            elif 'y' in axis.lower():
                self.coefList[j] = [coefList[i], 1]
                j += 1
            elif 'z' in axis.lower():
                self.coefList[j] = [coefList[i], 2]
                j += 1
   
    def __call__(self, coef, config):
        """When the object is called, apply the design variable values to 
        coefficients"""
        if self.config is None or config in self.config:
            for i in xrange(self.nVal):
                coef[self.coefList[i, 0], self.coefList[i, 1]] += self.value[i].real
      
        return coef

    def updateComplex(self, coef, config):
        if self.config is None or config in self.config:
            for i in xrange(self.nVal):
                coef[self.coefList[i, 0], self.coefList[i, 1]] += self.value[i].imag*1j

        return coef
        

def _convertTo1D(value, dim1):
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
