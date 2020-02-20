# ======================================================================
#         Imports
# ======================================================================
from __future__ import print_function
import copy,time
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
import os

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
    def __init__(self, fileName, complex=False, child=False, faceFreeze=None, *args, **kwargs):

        self.DV_listGlobal  = OrderedDict() # Global Design Variable List
        self.DV_listLocal = OrderedDict() # Local Design Variable List
        self.DV_listSectionLocal = OrderedDict() # Local Normal Design Variable List

        # Coefficient rotation matrix dict for Section Local variables
        self.coefRotM = {}

        # Flags to determine if this DVGeometry is a parent or child
        self.isChild = child
        self.children = []
        self.iChild = None
        self.points = OrderedDict()
        self.updated = {}
        self.masks = None
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
        self.ptSetNames = []
        self.JT = {}
        self.nPts = {}

        # Derivatives of Xref and Coef provided by the parent to the
        # children
        self.dXrefdXdvg = None
        self.dCoefdXdvg = None

        self.dXrefdXdvl = None
        self.dCoefdXdvl = None

        # derivative counters for offsets
        self.nDV_T = None
        self.nDVG_T = None
        self.nDVL_T = None
        self.nDVSL_T = None
        self.nDVG_count = 0
        self.nDVL_count = 0
        self.nDVSL_count = 0

        # The set of user supplied axis.
        self.axis = OrderedDict()

        # Generate coefMask regardless
        coefMask = []
        for iVol in range(self.FFD.nVol):
            coefMask.append(numpy.zeros((self.FFD.vols[iVol].nCtlu,
                                         self.FFD.vols[iVol].nCtlv,
                                         self.FFD.vols[iVol].nCtlw), dtype=bool))
        # Now do the faceFreeze
        if faceFreeze is not None:
            for iVol in range(self.FFD.nVol):
                key = '%d'%iVol
                if key in faceFreeze.keys():
                    if 'iLow' in faceFreeze[key]:
                        coefMask[iVol][0, :, :] = True
                        coefMask[iVol][1, :, :] = True
                    if 'iHigh' in faceFreeze[key]:
                        coefMask[iVol][-1, :, :] = True
                        coefMask[iVol][-2, :, :] = True
                    if 'jLow' in faceFreeze[key]:
                        coefMask[iVol][:, 0, :] = True
                        coefMask[iVol][:, 1, :] = True
                    if 'jHigh' in faceFreeze[key]:
                        coefMask[iVol][:, -1, :] = True
                        coefMask[iVol][:, -2, :] = True
                    if 'kLow' in faceFreeze[key]:
                        coefMask[iVol][:, :, 0] = True
                        coefMask[iVol][:, :, 1] = True
                    if 'kHigh' in faceFreeze[key]:
                        coefMask[iVol][:, :, -1] = True
                        coefMask[iVol][:, :, -2] = True

        # Finally we need to convert coefMask to the flattened global
        # coef type:
        tmp = numpy.zeros(len(self.FFD.coef), dtype=bool)
        for iVol in range(self.FFD.nVol):
            for i in range(coefMask[iVol].shape[0]):
                for j in range(coefMask[iVol].shape[1]):
                    for k in range(coefMask[iVol].shape[2]):
                        ind = self.FFD.topo.lIndex[iVol][i, j, k]
                        if coefMask[iVol][i, j, k]:
                            tmp[ind] = True
        self.masks = tmp

    def addRefAxis(self, name, curve=None, xFraction=None, volumes=None,
                   rotType=5, axis='x', alignIndex=None, rotAxisVar=None,
                   xFractionOrder=2, includeVols=[], ignoreInd=[],
                   raySize=1.5):
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

        #. The second is to specify the xFraction variable. There are a
           few caveats with the use of this method. First, DVGeometry
           will try to determine automatically the orientation of the FFD
           volume. Then, a reference axis will consist of the same number of
           control points as the number of span-wise sections in the FFD volume
           and will be oriented in the streamwise (x-direction) according to the
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
            Specify the stream-wise extent

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
            8. z-x-y + rotation about section axis (to allow for winglet rotation)

        axis: str
            Axis along which to project points/control points onto the
            ref axis. Default is 'x' which will project rays.

        alignIndex: str
            FFD axis along which the reference axis will lie. Can be 'i', 'j',
            or 'k'. Only necessary when using xFraction.

        rotAxisVar: str
            If rotType == 8, then you must specify the name of the section local
            variable which should be used to compute the orientation of the theta
            rotation.

        xFractionOrder : int
            Order of spline used for refaxis curve.

        includeVols : list
            List of additional volumes to add to reference axis after the
            automatic generation of the ref axis based on the volumes list using
            xFraction.

        ignoreInd : list
            List of indices that should be ignored from the volumes that were
            added to this reference axis. This can be handy if you have a single
            volume but you want to link different sets of indices to different
            reference axes.
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
        if axis is None:
            pass
        elif axis.lower() == 'x':
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
            # Some assumptions
            #   - FFD should be a close approximation of geometry surface so that
            #       xFraction roughly corresponds to airfoil LE, TE, or 1/4 chord
            #   - User provides 'i', 'j' or 'k' to specify which block direction
            #       the reference axis should project
            #   - if no volumes are listed, it is assumed that all volumes are
            #       included
            #   - 'x' is streamwise direction

            # This is the block direction along which the reference axis will lie
            # alignIndex = 'K'
            if alignIndex is None:
                raise Error('Must specify alignIndex to use xFraction.')

            # Get index direction along which refaxis will be aligned
            if alignIndex.lower() == 'i':
                alignIndex = 0
                faceCol = 2
            elif alignIndex.lower() == 'j':
                alignIndex = 1
                faceCol = 4
            elif alignIndex.lower() == 'k':
                alignIndex = 2
                faceCol = 0

            if volumes is None:
                volumes = range(self.FFD.nVol)

            # Reorder the volumes in sequential order and check if orientation is correct
            v = list(volumes)
            nVol = len(v)
            volOrd = [v.pop(0)]
            faceLink = self.FFD.topo.faceLink
            for iter in range(nVol):
                for vInd, i in enumerate(v):
                    for pInd, j in enumerate(volOrd):
                        if faceLink[i,faceCol] == faceLink[j,faceCol+1]:
                            volOrd.insert(pInd+1, v.pop(vInd))
                            break
                        elif faceLink[i,faceCol+1] == faceLink[j,faceCol]:
                            volOrd.insert(pInd, v.pop(vInd))
                            break

            if len(volOrd) < nVol:
                raise Error("The volumes are not ordered with matching faces"
                            " in the direction of the reference axis.")

            # Count total number of sections and check if volumes are aligned
            # face to face along refaxis direction
            lIndex = self.FFD.topo.lIndex
            nSections = []
            for i in range(len(volOrd)):
                if i == 0:
                    nSections.append(lIndex[volOrd[i]].shape[alignIndex])
                else:
                    nSections.append(lIndex[volOrd[i]].shape[alignIndex] - 1)

            refaxisNodes = numpy.zeros((sum(nSections), 3))

            # Loop through sections and compute node location
            place = 0
            for j, vol in enumerate(volOrd):
                sectionArr = numpy.rollaxis(lIndex[vol], alignIndex, 0)
                skip = 0
                if j > 0:
                    skip = 1
                for i in range(nSections[j]):
                    LE = numpy.min(self.FFD.coef[sectionArr[i+skip,:,:],0])
                    TE = numpy.max(self.FFD.coef[sectionArr[i+skip,:,:],0])
                    refaxisNodes[place+i,0] = xFraction*(TE - LE) + LE
                    refaxisNodes[place+i,1] = numpy.mean(self.FFD.coef[sectionArr[i+skip,:,:],1])
                    refaxisNodes[place+i,2] = numpy.mean(self.FFD.coef[sectionArr[i+skip,:,:],2])
                place += i + 1

            # Add additional volumes
            for iVol in includeVols:
                if iVol not in volumes:
                    volumes.append(iVol)

            # Generate reference axis pySpline curve
            curve = pySpline.Curve(X=refaxisNodes, k=2)
            nAxis = len(curve.coef)
            self.axis[name] = {'curve':curve, 'volumes':volumes,
                               'rotType':rotType, 'axis':axis,
                               'rotAxisVar':rotAxisVar}
        else:
            raise Error("One of 'curve' or 'xFraction' must be "
                        "specified for a call to addRefAxis")

        # Specify indices to be ignored
        self.axis[name]['ignoreInd'] = ignoreInd

        # Add the raySize multiplication factor for this axis
        self.axis[name]['raySize'] = raySize

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


        # save this name so that we can zero out the jacobians properly
        self.ptSetNames.append(ptName)
        self.zeroJacobians([ptName])
        self.nPts[ptName]=None

        points = numpy.array(points).real.astype('d')
        self.points[ptName] = points

        # Ensure we project into the undeformed geometry
        if origConfig:
            tmpCoef = self.FFD.coef.copy()
            self.FFD.coef = self.origFFDCoef
            self.FFD._updateVolumeCoef()

        # Project the last set of points into the volume
        if self.isChild:
            self.FFD.attachPoints(
                self.points[ptName], ptName, interiorOnly=True, **kwargs)
        else:
            self.FFD.attachPoints(
                self.points[ptName], ptName, interiorOnly=False)

        if origConfig:
            self.FFD.coef = tmpCoef
            self.FFD._updateVolumeCoef()

        # Now embed into the children:
        for child in self.children:
            child.addPointSet(points, ptName, origConfig, **kwargs)

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
            calls to addGeoDVLocal with different axis values.

        volList : list
            Use the control points on the volume indicies given in volList.
            You should use pointSelect = None, otherwise this will not work.

        pointSelect : pointSelect object. Default is None Use a
            pointSelect object to select a subset of the total number
            of control points. See the documentation for the
            pointSelect class in geo_utils. Using pointSelect discards everything in
            volList.

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
        >>> PS = geo_utils.PointSelect(type = 'y', pt1=[0,0,0], pt2=[10, 0, 10])
        >>> nVar = DVGeo.addGeoDVLocal('shape_vars', lower=-1.0, upper=1.0, pointSelect=PS)
        """
        if type(config) == str:
            config = [config]

        if pointSelect is not None:
            if pointSelect.type != 'ijkBounds':
                pts, ind = pointSelect.getPoints(self.FFD.coef)
            else:
                pts, ind = pointSelect.getPoints_ijk(self)
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
                                               scale, axis, ind, self.masks,
                                               config)

        return self.DV_listLocal[dvName].nVal

    def addGeoDVSectionLocal(self, dvName, secIndex, lower=None, upper=None,
                             scale=1.0, axis=1, pointSelect=None, volList=None,
                             orient0=None, orient2='svd', config=None):
        """Add one or more section local design variables to the DVGeometry
        object. Section local variables are used as an alternative to local
        variables when it is desirable to deform a cross-section shape within a
        plane that is consistent with the original cross-section orientation.
        This is helpful in at least two common scenarios:
            1) The original geometry has cross-sections that are not aligned with
                the global coordinate axes. For instance, with a winglet, we want
                the shape variables to deform normal to the winglet surface
                instead of in the x, y, or z directions.
            2) The global design variables cause changes in the geometry that
                rotate the orientation of the original cross-section planes. In
                this case, we want the shape variables to deform in directions
                aligned with the rotated cross-section plane, which may not be
                the x, y, or z directions.

        ** Warnings **
            - Rotations in an upper level (parent) FFD will not propagate down
                to the lower level FFDs due to limitations of the current
                implementation.
            - Section local design variables should not be specified at the same
                time as local design variables. This will most likely not result
                in the desired behavior.

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

        axis : int
            The coordinate directions to move. Permissible values are
                0: longitudinal direction (in section plane)
                1: latitudinal direction (in section plane)
                2: transverse direction (out of section plane)
            If more than one direction is required, use multiple calls to
            addGeoDVSectionLocal with different axis values.

                                    1
                                    ^
                                    |
                o-----o--------o----|----o--------o--------o-----o
                |                   |                            |  j
                |                   x---------> 0                |  ^
                |                  /                             |  |
                o-----o--------o--/------o--------o--------o-----o
                                 /      ----> i
                                /
                               2

        pointSelect : pointSelect object. Default is None Use a
            pointSelect object to select a subset of the total number
            of control points. See the documentation for the
            pointSelect class in geo_utils. Using pointSelect discards everything in
            volList.
            You can create a PointSelect object by using, for instance:
            >>> PS = geo_utils.PointSelect(type = 'y', pt1=[0,0,0], pt2=[10, 0, 10])
            Check the other PointSelect options in geo_utils.py

        volList : list
            Use the control points on the volume indicies given in volList. If
            None, all volumes will be included.
            PointSelect has priority over volList. So if you use PointSelect, the values
            defined in volList will have no effect.

        secIndex : char or list of chars
            For each volume, we need to specify along which index we would like
            to subdivide the volume into sections. Entries in list can be 'i',
            'j', or 'k'. This index will be designated as the transverse (2)
            direction in terms of the direction of perturbation for the 'axis'
            parameter.

        orient0 : None, 'i', 'j', 'k', or numpy vector. Default is None.
            Although secIndex defines the '2' axis, the '0' and '1' axes are still
            free to rotate within the section plane. We will choose the orientation
            of the '0' axis and let '1' be orthogonal. We have three options:
                1. <None> (default) If nothing is prescribed, the 0 direction will
                    be the best fit line through the section points. In the case
                    of an airfoil, this would roughly align with the chord.
                2. <'i','j' or 'k'> In this case, the '0' axis will be aligned
                    with the mean vector between the FFD edges corresponding to
                    this index. In the ascii art above, if 'j' were given for this
                    option, we would average the vectors between the points on the
                    top and bottom surfaces and project this vector on to the
                    section plane as the '0' axis. If a list is given, each index
                    will be applied to its corresponding volume in volList.
                3. <[x, y, z]> If a numpy vector is given, the '0' axis
                    will be aligned with a projection of this vector onto the
                    section plane. If a numpy array of len(volList) x 3 is given,
                    each vector will apply to its corresponding volume.

        orient2: 'svd' or 'ffd. Default is 'svd'
            How to compute the orientation '2' axis. SVD is the
            default bevaviour and is taken from the svd of the plane
            points. 'ffd' Uses the vector along the FFD direction of
            secIndex. This is requied to get consistent normals if you
            have a circular-type FFD when the SVD will swap the
            normals.

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
        >>> # Add all control points in FFD as local shape variables
        >>> # moving in the 1 direction, within +/- 1.0 units
        >>> DVGeo.addGeoDVSectionLocal('shape_vars', secIndex='k', lower=-1, upper=1, axis=1)
        """
        if type(config) == str:
            config = [config]

        # Pick out control points
        if pointSelect is not None:
            if pointSelect.type != 'ijkBounds':
                pts, ind = pointSelect.getPoints(self.FFD.coef)
                volList = numpy.arange(self.FFD.nVol) # Select all volumes
            else:
                pts, ind = pointSelect.getPoints_ijk(self)
                volList = pointSelect.ijkBounds.keys() # Select only volumes used by pointSelect
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
                ind.extend(self.FFD.topo.lIndex[iVol].flatten()) # Get all indices from this block
            ind = geo_utils.unique(ind)
        else:
            # Just take'em all
            volList = numpy.arange(self.FFD.nVol)
            ind = numpy.arange(len(self.FFD.coef))

        secLink = numpy.zeros(self.FFD.coef.shape[0], dtype=int)
        secTransform = [numpy.eye(3)]

        if type(secIndex) is str:
            secIndex = [secIndex]*len(volList)
        elif type(secIndex) is list:
            if len(secIndex) != len(volList):
                raise Error('If a list is given for secIndex, the length must be'
                            ' equal to the length of volList.')

        if orient0 is not None:
            # 'i', 'j', or 'k'
            if type(orient0) is str:
                orient0 = [orient0]*len(volList)
            # ['k', 'k', 'i', etc.]
            elif type(orient0) is list:
                if len(orient0) != len(volList):
                    raise Error('If a list is given for orient0, the length must'
                                ' be equal to the length of volList.')
            # numpy.array([1.0, 0.0, 0.0])
            elif type(orient0) is numpy.ndarray:
                # vector
                if len(orient0.shape) == 1:
                    orient0 = numpy.reshape(orient0, (1,3))
                    orient0 = numpy.repeat(orient0, len(volList), 0)
                elif orient0.shape[0] == 1:
                    orient0 = numpy.repeat(orient0, len(volList), 0)
                elif orient0.shape[0] != len(volList):
                    raise Error('If an array is given for orient0, the row dimension'
                                ' must be equal to the length of volList.')
            for i, iVol in enumerate(volList):
                self.sectionFrame(secIndex[i], secTransform, secLink, iVol,
                                  orient0[i], orient2=orient2)
        else:
            for i, iVol in enumerate(volList):
                self.sectionFrame(secIndex[i], secTransform, secLink, iVol, orient2=orient2)

        self.DV_listSectionLocal[dvName] = geoDVSectionLocal(dvName, lower, upper,
                                               scale, axis, ind, self.masks,
                                               config, secTransform, secLink)

        return self.DV_listSectionLocal[dvName].nVal

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
            self._finalize()
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
                vals_to_set = numpy.atleast_1d(dvDict[key]).astype('D')
                if len(vals_to_set) != self.DV_listLocal[key].nVal:
                    raise Error('Incorrect number of design variables \
                    for DV: %s.\nExpecting %d variables and received \
                    %d variabes'%(key, self.DV_listLocal[key].nVal,
                                  len(vals_to_set)))
                self.DV_listLocal[key].value = vals_to_set

            if key in self.DV_listSectionLocal:
                vals_to_set = numpy.atleast_1d(dvDict[key]).astype('D')
                if len(vals_to_set) != self.DV_listSectionLocal[key].nVal:
                    raise Error('Incorrect number of design variables \
                    for DV: %s.\nExpecting %d variables and received \
                    %d variabes'%(key, self.DV_listSectionLocal[key].nVal,
                                  len(vals_to_set)))
                self.DV_listSectionLocal[key].value = vals_to_set

            # Jacobians are, in general, no longer up to date
            self.zeroJacobians(self.ptSetNames)

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

        # Now call setValues on the children. This way the
        # variables will be set on the children
        for child in self.children:
            child.setDesignVars(dvDict)

    def zeroJacobians(self,ptSetNames):
        '''
        set stored jacobians to None for ptSetNames

        Parameters
        ----------
        ptSetNames : list
            list of ptSetNames to zero the jacobians.
        '''
        for name in ptSetNames:
            self.JT[name] = None # J is no longer up to date

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

        # and now the section local DVs
        for key in self.DV_listSectionLocal:
            dvDict[key] = self.DV_listSectionLocal[key].value


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
            return list(self.axis.keys()).index(axisID)
        except:
            raise Error("'The 'axisID' was invalid!")

    def updateCalculations(self, new_pts, isComplex, config):
        """
        The core update rountine. pulled out here to eliminate duplication between update and
        update_deriv.
        """

        if self.isChild:
            # If this is a child, update the links between the ref axis and the
            # coefficients on the nested FFD now that the nested FFD has been
            # moved.
            # **Important**: this expects the FFD coef to be clean on this level,
            # meaning that the only changes to FFD.coef can be coming from
            # higher levels.
            if isComplex:
                self.links_x = self.links_x.astype('D')
            for ipt in range(self.nPtAttach):
                base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])
                self.links_x[ipt] = self.FFD.coef[self.ptAttachInd[ipt], :] - base_pt

        # Run Global Design Vars
        for key in self.DV_listGlobal:
            self.DV_listGlobal[key](self, config)

        # update the reference axis now that the new global vars have been run
        self.refAxis.coef = self.coef.copy()
        self.refAxis._updateCurveCoef()

        for ipt in range(self.nPtAttach):
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
                new_vec = geo_utils.rotVbyW(new_vec, deriv, self.rot_theta[
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

                # if necessary, assign rotation matrix for each ffd coef
                if self.coefRotM is not None:
                    attachedPoint = self.ptAttachInd[ipt]
                    if isComplex:
                        self.coefRotM[attachedPoint] = rotM
                    else:
                        self.coefRotM[attachedPoint] = numpy.real(rotM)

                D = numpy.dot(rotM, D)
                if rotType == 7:
                    # only apply the theta rotations in certain cases
                    deriv = self.refAxis.curves[
                        self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                    deriv /= geo_utils.euclideanNorm(deriv) # Normalize
                    D = geo_utils.rotVbyW(D, deriv, numpy.pi/180*self.rot_theta[
                            self.curveIDNames[ipt]](self.links_s[ipt]))

                elif rotType == 8:
                    varname = self.axis[self.curveIDNames[ipt]]['rotAxisVar']
                    slVar = self.DV_listSectionLocal[varname]
                    attachedPoint = self.ptAttachInd[ipt]
                    W = slVar.sectionTransform[slVar.sectionLink[attachedPoint]][:,2]
                    D = geo_utils.rotVbyW(D, W, numpy.pi/180*self.rot_theta[
                            self.curveIDNames[ipt]](self.links_s[ipt]))

                D[0] *= scale_x
                D[1] *= scale_y
                D[2] *= scale_z
                if isComplex:
                    new_pts[ipt] = base_pt + D*scale
                else:
                    new_pts[ipt] = numpy.real(base_pt + D*scale)

    def update(self, ptSetName, childDelta=False, config=None):
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

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.

            """
        self.curPtSet = ptSetName
        # We've postponed things as long as we can...do the finialization.
        self._finalize()

        # Make sure coefficients are complex
        self._complexifyCoef()

        # Set all coef Values back to initial values
        if not self.isChild:
            self.FFD.coef = self.origFFDCoef.copy()
            self._setInitialValues()
        else:
            # Update all coef
            self.FFD._updateVolumeCoef()

            # Evaluate starting pointset
            Xstart = self.FFD.getAttachedPoints(ptSetName)

            if self.complex:
                # Now we have to propagate the complex part through Xstart
                tempCoef = self.FFD.coef.copy().astype('D')
                Xstart = Xstart.astype('D')
                imag_part = numpy.imag(tempCoef)
                imag_j = 1j

                dPtdCoef = self.FFD.embededVolumes[ptSetName].dPtdCoef
                if dPtdCoef is not None:
                    for ii in range(3):
                        Xstart[:, ii] += imag_j*dPtdCoef.dot(imag_part[:, ii])

        # Step 1: Call all the design variables IFF we have ref axis:
        if len(self.axis) > 0:
            if self.complex:
                new_pts = numpy.zeros((self.nPtAttach, 3), 'D')
            else:
                new_pts = numpy.zeros((self.nPtAttach, 3), 'd')

            # Apply the global design variables
            self.updateCalculations(new_pts, isComplex=self.complex, config=config)

            # Put the update FFD points in their proper place
            temp = numpy.real(new_pts)
            numpy.put(self.FFD.coef[:, 0], self.ptAttachInd, temp[:, 0])
            numpy.put(self.FFD.coef[:, 1], self.ptAttachInd, temp[:, 1])
            numpy.put(self.FFD.coef[:, 2], self.ptAttachInd, temp[:, 2])

        # Now add in the section local DVs
        for key in self.DV_listSectionLocal:
            self.DV_listSectionLocal[key](self.FFD.coef, self.coefRotM, config)

        # Now add in the local DVs
        for key in self.DV_listLocal:
            self.DV_listLocal[key](self.FFD.coef, config)

        # Update all coef
        self.FFD._updateVolumeCoef()

        # Evaluate coordinates from the parent
        Xfinal = self.FFD.getAttachedPoints(ptSetName)

        # Propagate the complex part through the volume artificially
        if self.complex:
            # Above, we only took the real part of the coef because
            # _updateVolumeCoef gets rid of it anyway. Here, we need to include
            # the complex part because we want to propagate it through
            tempCoef = self.FFD.coef.copy().astype('D')
            if len(self.axis) > 0:
                numpy.put(tempCoef[:, 0], self.ptAttachInd, new_pts[:, 0])
                numpy.put(tempCoef[:, 1], self.ptAttachInd, new_pts[:, 1])
                numpy.put(tempCoef[:, 2], self.ptAttachInd, new_pts[:, 2])

            # Apply just the complex part of the local varibales
            for key in self.DV_listSectionLocal:
                self.DV_listSectionLocal[key].updateComplex(tempCoef, self.coefRotM, config)
            for key in self.DV_listLocal:
                self.DV_listLocal[key].updateComplex(tempCoef, config)

            Xfinal = Xfinal.astype('D')
            imag_part = numpy.imag(tempCoef)
            imag_j = 1j

            dPtdCoef = self.FFD.embededVolumes[ptSetName].dPtdCoef
            if dPtdCoef is not None:
                for ii in range(3):
                    Xfinal[:, ii] += imag_j*dPtdCoef.dot(imag_part[:, ii])

        # Now loop over the children set the FFD and refAxis control
        # points as evaluated from the parent
        for iChild in range(len(self.children)):
            child = self.children[iChild]
            child._finalize()
            self.applyToChild(iChild)

            if self.complex:
                # need to propagate the sensitivity to the children Xfinal here to do this
                # correctly
                child._complexifyCoef()
                child.FFD.coef = child.FFD.coef.astype('D')

                dXrefdCoef = self.FFD.embededVolumes['child%d_axis'%(iChild)].dPtdCoef
                dCcdCoef   = self.FFD.embededVolumes['child%d_coef'%(iChild)].dPtdCoef

                if dXrefdCoef is not None:
                    for ii in range(3):
                        child.coef[:, ii] += imag_j*dXrefdCoef.dot(imag_part[:, ii])

                if dCcdCoef is not None:
                    for ii in range(3):
                        child.FFD.coef[:, ii] += imag_j*dCcdCoef.dot(imag_part[:, ii])
                child.refAxis.coef = child.coef.copy()
                child.refAxis._updateCurveCoef()

            Xfinal += child.update(ptSetName, childDelta=True, config=config)

        self._unComplexifyCoef()

        # Finally flag this pointSet as being up to date:
        self.updated[ptSetName] = True

        if self.isChild and childDelta:
            return Xfinal - Xstart
        else:
            return Xfinal

    def applyToChild(self, iChild):
        """
        This function is used to apply the changes in the parent FFD to the
        child FFD points and child reference axis points.
        """
        child = self.children[iChild]

        # Set FFD points and reference axis points from parent
        child.FFD.coef = self.FFD.getAttachedPoints('child%d_coef'%(iChild))
        child.coef = self.FFD.getAttachedPoints('child%d_axis'%(iChild))

        # Update the reference axes on the child
        child.refAxis.coef = child.coef.copy()
        child.refAxis._updateCurveCoef()

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
        DVCountGlobal, DVCountLocal, DVCountSecLoc = self._getDVOffsets()

        i = DVCountGlobal
        dIdxDict = {}
        for key in self.DV_listGlobal:
            dv = self.DV_listGlobal[key]
            if out1D:
                dIdxDict[dv.name] = numpy.ravel(dIdx[:, i:i+dv.nVal])
            else:
                dIdxDict[dv.name] = dIdx[:, i:i+dv.nVal]
            i += dv.nVal
        i = DVCountSecLoc
        for key in self.DV_listSectionLocal:
            dv = self.DV_listSectionLocal[key]
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
        for iChild in range(len(self.children)):
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
        DVCountGlobal, DVCountLocal, DVCountSecLoc = self._getDVOffsets()
        dIdx = numpy.zeros(self.nDV_T, self.dtype)
        i = DVCountGlobal
        for key in self.DV_listGlobal:
            dv = self.DV_listGlobal[key]
            dIdx[i:i+dv.nVal] = dIdxDict[dv.name]
            i += dv.nVal
        i = DVCountLocal
        for key in self.DV_listLocal:
            dv = self.DV_listLocal[key]
            dIdx[i:i+dv.nVal] = dIdxDict[dv.name]
            i += dv.nVal
        i = DVCountSecLoc
        for key in self.DV_listSectionLocal:
            dv = self.DV_listSectionLocal[key]
            dIdx[i:i+dv.nVal] = dIdxDict[dv.name]
            i += dv.nVal

        #Note: not sure if this works with (multiple) sibling child FFDs
        for iChild in range(len(self.children)):
            childdIdx = self.children[iChild].convertDictToSensitivity(dIdxDict)
            # update the total sensitivities with the derivatives from the child
            dIdx+=childdIdx
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
        names.extend(list(self.DV_listSectionLocal.keys()))

        # Call the children recursively
        for iChild in range(len(self.children)):
            names.extend(self.children[iChild].getVarNames())

        return names


    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        """
        This function computes sensitivty information.

        Specificly, it computes the following:
        :math:`\\frac{dX_{pt}}{dX_{DV}}^T \\frac{dI}{d_{pt}}`

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

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.


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
        self.computeTotalJacobian(ptSetName,config=config)

        # now that we have self.JT compute the Mat-Mat multiplication
        nDV = self._getNDV()
        dIdx_local = numpy.zeros((N, nDV), 'd')
        for i in range(N):
            if self.JT[ptSetName] is not None:
                dIdx_local[i,:] = self.JT[ptSetName].dot(dIdpt[i,:,:].flatten())

        if comm: # If we have a comm, globaly reduce with sum
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # Now convert to dict:
        dIdx = self.convertSensitivityToDict(dIdx)

        return dIdx

    def totalSensitivityProd(self, vec, ptSetName, comm=None, child=False,
                        nDVStore=0, config=None):
        """
        This function computes sensitivty information.

        Specifically, it computes the following:
        :math:`\\frac{dX_{pt}}{dX_{DV}} \\ vec`

        This is useful for forward AD mode.

        Parameters
        ----------
        vec : dictionary whose keys are the design variable names, and whose
              values are the derivative seeds of the corresponding design variable.

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place.

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.

        Returns
        -------
        xsdot : array (Nx3) -> Array with derivative seeds of the surface nodes.

        Notes
        -----
        The ``child`` and ``nDVStore`` options are only used
        internally and should not be changed by the user.
        """

        self.computeTotalJacobian(ptSetName,config=config)

        names = self.getVarNames()
        newvec = numpy.zeros(self.getNDV(),self.dtype)
        i = 0
        for key in names:
            if key in self.DV_listGlobal:
                dv = self.DV_listGlobal[key]
            elif key in self.DV_listSectionLocal:
                dv = self.DV_listSectionLocal[key]
            else:
                dv = self.DV_listLocal[key]

            if key in vec:
                newvec[i:i+dv.nVal] = vec[key]

            i += dv.nVal

        # perform the product
        if self.JT[ptSetName] is None:
            xsdot = numpy.zeros((0, 3))
        else:
            xsdot = self.JT[ptSetName].T.dot(newvec)
            xsdot.reshape(len(xsdot)//3, 3)
            # Maybe this should be:
            #xsdot = xsdot.reshape(len(xsdot)//3, 3)

        return xsdot

    def totalSensitivityTransProd(self, vec, ptSetName, comm=None, child=False,
                        nDVStore=0, config=None):
        """
        This function computes sensitivty information.

        Specifically, it computes the following:
        :math:`\\frac{dX_{pt}}{dX_{DV}}^T \\ vec`

        This is useful for reverse AD mode.

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

        config : str or list
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.

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

        self.computeTotalJacobian(ptSetName,config=config)

        # perform the product
        if self.JT[ptSetName] == None:
            xsdot = numpy.zeros((0, 3))
        else:
            xsdot = self.JT[ptSetName].dot(numpy.ravel(vec))

        # Pack result into dictionary
        xsdict = {}
        names = self.getVarNames()
        i = 0
        for key in names:
            if key in self.DV_listGlobal:
                dv = self.DV_listGlobal[key]
            elif key in self.DV_listSectionLocal:
                dv = self.DV_listSectionLocal[key]
            else:
                dv = self.DV_listLocal[key]
            xsdict[key] = xsdot[i:i+dv.nVal]
            i += dv.nVal

        return xsdict

    def computeDVJacobian(self,config=None):
        """
        return J_temp for a given config
        """
        # These routines are not recursive. They compute the derivatives at this level and
        # pass information down one level for the next pass call from the routine above

        # This is going to be DENSE in general
        J_attach = self._attachedPtJacobian(config=config)

        # Compute local normal jacobian
        J_sectionlocal = self._sectionlocalDVJacobian(config=config)

        # This is the sparse jacobian for the local DVs that affect
        # Control points directly.
        J_local = self._localDVJacobian(config=config)

        # this is the jacobian from accumulated derivative dependence from parent to child
        J_casc = self._cascadedDVJacobian(config=config)

        J_temp = None

        # add them together
        if J_attach is not None:
            J_temp =  sparse.lil_matrix(J_attach)

        if J_sectionlocal is not None:
            if J_temp is None:
                J_temp =  sparse.lil_matrix(J_sectionlocal)
            else:
                J_temp += J_sectionlocal

        if J_local is not None:
            if J_temp is None:
                J_temp =  sparse.lil_matrix(J_local)
            else:
                J_temp += J_local

        if J_casc is not None:
            if J_temp is None:
                J_temp =  sparse.lil_matrix(J_casc)
            else:
                J_temp += J_casc

        return J_temp

    def computeTotalJacobian(self, ptSetName, config=None):
        """ Return the total point jacobian in CSR format since we
        need this for TACS"""

        # Finalize the object, if not done yet
        self._finalize()
        self.curPtSet = ptSetName

        if not(self.JT[ptSetName] is None):
            return

        # compute the derivatives of the coeficients of this level wrt all of the design
        # variables at this level and all levels above
        J_temp = self.computeDVJacobian(config=config)

        # now get the derivative of the points for this level wrt the coefficients(dPtdCoef)
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
            for j in range(3):
                new_data[j::3] = data
                new_row[j::3] = row*3 + j
                new_col[j::3] = col*3 + j

            # Size of New Matrix:
            Nrow = dPtdCoef.shape[0]*3
            Ncol = dPtdCoef.shape[1]*3

            # Create new matrix in coo-dinate format and convert to csr
            new_dPtdCoef = sparse.coo_matrix(
                (new_data, (new_row, new_col)), shape=(Nrow, Ncol)).tocsr()

            # Do Sparse Mat-Mat multiplication and resort indices
            if J_temp is not None:
                self.JT[ptSetName] = (J_temp.T*new_dPtdCoef.T).tocsr()
                self.JT[ptSetName].sort_indices()

            # Add in child portion
            for iChild in range(len(self.children)):

                # Reset control points on child for child link derivatives
                self.applyToChild(iChild)
                self.children[iChild].computeTotalJacobian(ptSetName,config=config)

                if self.JT[ptSetName] is not None:
                    self.JT[ptSetName] = self.JT[ptSetName] + self.children[iChild].JT[ptSetName]
                else:
                    self.JT[ptSetName] = self.children[iChild].JT[ptSetName]
        else:
            self.JT[ptSetName] = None


    def computeTotalJacobianCS(self, ptSetName, config=None):
        """ Return the total point jacobian in CSR format since we
        need this for TACS"""

        print('Calculate DV Jacobian using CS...')
        self._finalize()
        self.curPtSet = ptSetName

        if not(self.JT[ptSetName] is None):
            return

        if self.isChild:
            refFFDCoef = copy.copy(self.FFD.coef)
            refCoef = copy.copy(self.coef)

        if self.nPts[ptSetName] == None:
            self.nPts[ptSetName] = len(self.update(ptSetName).flatten())
        for child in self.children:
            child.nPts[ptSetName] = self.nPts[ptSetName]

        DVGlobalCount, DVLocalCount, DVSecLocCount = self._getDVOffsets()

        h = 1e-40j

        self.JT[ptSetName] = numpy.zeros([self.nDV_T,self.nPts[ptSetName]])
        self._complexifyCoef()
        for key in self.DV_listGlobal:
            for j in range(self.DV_listGlobal[key].nVal):
                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = refCoef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listGlobal[key].value[j]

                self.DV_listGlobal[key].value[j] += h

                deriv = numpy.imag(self._update_deriv_cs(ptSetName,config=config).flatten())/numpy.imag(h)

                self.JT[ptSetName][DVGlobalCount,:]=deriv

                DVGlobalCount += 1
                self.DV_listGlobal[key].value[j] = refVal

        print('local')

        self._unComplexifyCoef()
        for key in self.DV_listSectionLocal:
            for j in range(self.DV_listSectionLocal[key].nVal):
                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = refCoef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listSectionLocal[key].value[j]

                self.DV_listSectionLocal[key].value[j] += h
                deriv = numpy.imag(self._update_deriv_cs(ptSetName,config=config).flatten())/numpy.imag(h)

                self.JT[ptSetName][DVSecLocCount,:] = deriv

                DVSecLocCount += 1
                self.DV_listSectionLocal[key].value[j] = refVal

        for key in self.DV_listLocal:
            for j in range(self.DV_listLocal[key].nVal):
                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = refCoef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listLocal[key].value[j]

                self.DV_listLocal[key].value[j] += h
                deriv = numpy.imag(self._update_deriv_cs(ptSetName,config=config).flatten())/numpy.imag(h)

                self.JT[ptSetName][DVLocalCount,:] = deriv

                DVLocalCount += 1
                self.DV_listLocal[key].value[j] = refVal


        for iChild in range(len(self.children)):
            child = self.children[iChild]
            child._finalize()

            # In the updates applied previously, the FFD points on the children
            # will have been set as deltas. We need to set them as absolute
            # coordinates based on the changes in the parent before moving down
            # to the next level
            self.applyToChild(iChild)

            # Now get jacobian from child and add to parent jacobian
            child.computeTotalJacobianCS(ptSetName,config=config)
            self.JT[ptSetName] = self.JT[ptSetName] + child.JT[ptSetName]

        return

    def addVariablesPyOpt(self, optProb, globalVars=True, localVars=True,
                          sectionlocalVars=True, ignoreVars=None, freezeVars=None):
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
                   'localVars':self.DV_listLocal,
                   'sectionlocalVars':self.DV_listSectionLocal}
        for lst in varLists:
            if lst == 'globalVars' and globalVars or lst=='localVars' and localVars \
               or lst=='sectionlocalVars' and sectionlocalVars:
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
            child.addVariablesPyOpt(optProb, globalVars, localVars, sectionlocalVars,
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
            keyToUpdate = list(self.points.keys())[0]
            self.update(keyToUpdate, childDelta=False)

        f = pySpline.openTecplot(fileName, 3)
        vol_counter = 0

        # Write master volumes:
        vol_counter += self._writeVols(f, vol_counter)

        pySpline.closeTecplot(f)
        if len(self.points)>0:
            self.update(keyToUpdate, childDelta=True)

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
        for iChild in range(len(self.children)):
            cFileName = fileName+'_child{:03d}.dat'.format(iChild)
            self.children[iChild].refAxis.writeTecplot(cFileName, orig=True, curves=True, coef=True)

    def writeLinks(self, fileName):
        """Write the links attaching the control points to the reference axes

        Parameters
        ----------
        fileName : str
            Filename for tecplot file. Should have .dat extension
        """
        self._finalize()
        f = pySpline.openTecplot(fileName, 3)
        f.write('ZONE NODES=%d ELEMENTS=%d ZONETYPE=FELINESEG\n'%(self.nPtAttach*2, self.nPtAttach))
        f.write('DATAPACKING=POINT\n')
        for ipt in range(self.nPtAttach):
            pt1 = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])
            pt2 = self.links_x[ipt] + pt1

            f.write('%.12g %.12g %.12g\n'%(pt1[0], pt1[1], pt1[2]))
            f.write('%.12g %.12g %.12g\n'%(pt2[0], pt2[1], pt2[2]))
        for i in range(self.nPtAttach):
            f.write('%d %d\n'%(2*i+1, 2*i+2))

        pySpline.closeTecplot(f)

    def writePointSet(self,name,fileName):
        """
        Write a given point set to a tecplot file

        Parameters
        ----------
        name : str
             The name of the point set to write to a file
        fileName : str
           Filename for tecplot file. Should have no extension, an
           extension will be added
        """
        if self.isChild:
            raise Error('Must call "writePointSet" from parent DVGeo.')
        else:
            coords = self.update(name, childDelta=True)
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

    def getFlattenedChildren(self):
        """
        Return a flattened list of all DVGeo objects in the family heirarchy.
        """
        flatChildren = [self]
        for child in self.children:
            flatChildren += child.getFlattenedChildren()

        return flatChildren

    def demoDesignVars(self, directory, includeLocal=True, includeGlobal=True,
                       pointSet=None, callBack=None, freq=2):
        """
        This function can be used to "test" the design variable parametrization
        for a given optimization problem. It should be called in the script
        after DVGeo has been set up. The function will loop through all the
        design variables and write out a deformed FFD volume for the upper
        and lower bound of every design variable. It will also write out the
        deformed pointset of choice.

        Parameters
        ----------
        directory : str
            The directory where the FFD files should be written.
        includeLocal : boolean
            False if you don't want to include the shape variables.
        pointSet : str
            Name of the pointset to write out. If this is not specified, it will
            take the first one in the list.
        callBack : function
            This allows the user to perform an additional task at each new design
            variable iteration (e.g. write out a deformed mesh). The callback
            function must take two inputs: 1) the output directory name (str) and
            2) the iteration count (int).
        freq : int
            Number of snapshots to take between the upper and lower bounds of
            a given variable. If greater than 2, will do a sinusoidal sweep.
        """
        # Generate directories
        os.system('mkdir -p {:s}/ffd'.format(directory))
        os.system('mkdir -p {:s}/pointset'.format(directory))

        # Get design variables
        dvDict = self.getValues()

        # Get pointSet
        if pointSet is None:
            writePointSet = False
            if self.ptSetNames:
                pointSet = self.ptSetNames[0]
            else:
                raise Error('DVGeo must have a point set to update for'
                            'demoDesignVars to work.')
        else:
            writePointSet = True

        # Loop through design variables on self and children
        geoList = self.getFlattenedChildren()
        count = 0
        for geo in geoList:
            for key in dvDict:
                lower = []
                if key in geo.DV_listLocal:
                    if not includeLocal:
                        continue
                    lower = geo.DV_listLocal[key].lower
                    upper = geo.DV_listLocal[key].upper
                elif key in geo.DV_listSectionLocal:
                    if not includeLocal:
                        continue
                    lower = geo.DV_listSectionLocal[key].lower
                    upper = geo.DV_listSectionLocal[key].upper
                elif key in geo.DV_listGlobal:
                    if not includeGlobal:
                        continue
                    lower = geo.DV_listGlobal[key].lower
                    upper = geo.DV_listGlobal[key].upper

                if lower is None or upper is None:
                    raise Error('demoDesignVars requires upper and lower bounds'
                                'on all design variables.')

                x = dvDict[key].flatten()
                nDV = len(lower)
                for j in range(nDV):
                    if freq == 2:
                        stops = [lower[j], upper[j]]
                    elif freq > 2:
                        sinusoid = numpy.sin(numpy.linspace(0, numpy.pi, freq))
                        down_swing = x[j] + (lower[j] - x[j]) * sinusoid
                        up_swing = x[j] + (upper[j] - x[j]) * sinusoid
                        stops = numpy.concatenate((down_swing[:-1], up_swing[:-1]))

                    for val in stops:
                        # Add perturbation to the design variable and update
                        old_val = x[j]
                        x[j] = val
                        dvDict.update({key:x})
                        self.setDesignVars(dvDict)
                        X = self.update(pointSet)

                        # Write FFD
                        self.writeTecplot('{}/ffd/iter_{:03d}.dat'.format(directory, count))

                        # Write pointset
                        if writePointSet:
                            self.writePointSet(pointSet, '{}/pointset/iter_{:03d}'.format(directory, count))

                        # Call user function
                        if callBack is not None:
                            callBack(directory, count)

                        # Reset variable
                        x[j] = old_val
                        dvDict.update({key:x})

                        # Iterate counter
                        count += 1

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
        coefMask = self.masks

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
                for i in range(self.FFD.vols[iVol].nCtlu):
                    for j in range(self.FFD.vols[iVol].nCtlv):
                        for k in range(self.FFD.vols[iVol].nCtlw):
                            ind = self.FFD.topo.lIndex[iVol][i, j, k]
                            if coefMask[ind] == False and ind not in self.axis[key]['ignoreInd']:
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
                    curPts, self.axis[key]['axis'], curves=[curveID],
                    raySize=self.axis[key]['raySize'])

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

        for i in range(self.nPtAttach):
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
        elif rotType == 8:
            D = numpy.dot(rotY, numpy.dot(rotX, rotZ))
        return D

    def _getNDV(self):
        """Return the actual number of design variables, global + local
            + section local
        """
        return self._getNDVGlobal() + self._getNDVLocal() + self._getNDVSectionLocal()

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

    def _getNDVSectionLocal(self):
        """
        Get total number of local variables, inclding any children
        """
        nDV = 0
        for key in self.DV_listSectionLocal:
            nDV += self.DV_listSectionLocal[key].nVal

        for child in self.children:
            nDV += child._getNDVSectionLocal()

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

    def _getNDVSectionLocalSelf(self):
        """
        Get total number of local variables, not including
        children
        """
        nDV = 0
        for key in self.DV_listSectionLocal:
            nDV += self.DV_listSectionLocal[key].nVal

        return nDV

    def _getDVOffsets(self):
        '''
        return the global and local DV offsets for this FFD
        '''

        # figure out the split between local and global Variables
        # All global vars at all levels come first
        # then section local vars and then local vars.
        # Parent Vars come before child Vars

        # get the global and local DV numbers on the parents if we don't have them
        if self.nDV_T==None or self.nDVG_T == None or self.nDVL_T==None \
            or self.nDVSL_T==None:
            self.nDV_T = self._getNDV()
            self.nDVG_T = self._getNDVGlobal()
            self.nDVL_T = self._getNDVLocal()
            self.nDVSL_T = self._getNDVSectionLocal()
            self.nDVG_count = 0
            self.nDVSL_count = self.nDVG_T
            self.nDVL_count = self.nDVG_T + self.nDVSL_T

        nDVG = self._getNDVGlobalSelf()
        nDVL = self._getNDVLocalSelf()
        nDVSL = self._getNDVSectionLocalSelf()

        # Set the total number of global and local DVs into any children of this parent
        for child in self.children:
            # now get the numbers for the current parent child

            child.nDV_T = self.nDV_T
            child.nDVG_T = self.nDVG_T
            child.nDVL_T = self.nDVL_T
            child.nDVSL_T = self.nDVSL_T
            child.nDVG_count = self.nDVG_count + nDVG
            child.nDVL_count = self.nDVL_count + nDVL
            child.nDVSL_count = self.nDVSL_count + nDVSL

            # Increment the counters for the children
            nDVG += child._getNDVGlobalSelf()
            nDVL += child._getNDVLocalSelf()
            nDVSL += child._getNDVSectionLocalSelf()

        return self.nDVG_count, self.nDVL_count, self.nDVSL_count

    def _update_deriv(self, iDV=0, h=1.0e-40j, oneoverh=1.0/1e-40, config=None, localDV=False):

        """Copy of update function for derivative calc"""
        new_pts = numpy.zeros((self.nPtAttach, 3), 'D')

        # Step 1: Call all the design variables IFF we have ref axis:
        if len(self.axis) > 0:

            # Recompute changes due to global dvs at current point + h
            self.updateCalculations(new_pts, isComplex=True, config=config)

            # create a vector of the size of the full FFD
            numpy.put(self.FFD.coef[:, 0], self.ptAttachInd, new_pts[:,0])
            numpy.put(self.FFD.coef[:, 1], self.ptAttachInd, new_pts[:,1])
            numpy.put(self.FFD.coef[:, 2], self.ptAttachInd, new_pts[:,2])

            # Add dependence of section variables on the global dv rotations
            for key in self.DV_listSectionLocal:
                self.DV_listSectionLocal[key].updateComplex(self.FFD.coef, self.coefRotM, config)

            # Send values back to new_pts
            new_pts[:,0] = self.FFD.coef[self.ptAttachInd,0]
            new_pts[:,1] = self.FFD.coef[self.ptAttachInd,1]
            new_pts[:,2] = self.FFD.coef[self.ptAttachInd,2]

            # set the forward effect of the global design vars in each child
            for iChild in range(len(self.children)):

                # get the derivative of the child axis and control points wrt the parent
                # control points
                dXrefdCoef = self.FFD.embededVolumes['child%d_axis'%(iChild)].dPtdCoef
                dCcdCoef   = self.FFD.embededVolumes['child%d_coef'%(iChild)].dPtdCoef

                # create a vector with the derivative of the parent control points wrt the
                # parent global variables
                tmp = numpy.zeros(self.FFD.coef.shape, dtype='d')
                numpy.put(tmp[:, 0], self.ptAttachInd,
                          numpy.imag(new_pts[:,0])*oneoverh)
                numpy.put(tmp[:, 1], self.ptAttachInd,
                          numpy.imag(new_pts[:,1])*oneoverh)
                numpy.put(tmp[:, 2], self.ptAttachInd,
                          numpy.imag(new_pts[:,2])*oneoverh)

                # create variables for the total derivative of the child axis and control
                # points wrt the parent global variables
                dXrefdXdv = numpy.zeros((dXrefdCoef.shape[0]*3), 'd')
                dCcdXdv   = numpy.zeros((dCcdCoef.shape[0]*3), 'd')

                # multiply the derivative of the child axis wrt the parent control points
                # by the derivative of the parent control points wrt the parent global vars.
                # this is just chain rule
                dXrefdXdv[0::3] = dXrefdCoef.dot(tmp[:, 0])
                dXrefdXdv[1::3] = dXrefdCoef.dot(tmp[:, 1])
                dXrefdXdv[2::3] = dXrefdCoef.dot(tmp[:, 2])

                # do the same for the child control points
                dCcdXdv[0::3] = dCcdCoef.dot(tmp[:, 0])
                dCcdXdv[1::3] = dCcdCoef.dot(tmp[:, 1])
                dCcdXdv[2::3] = dCcdCoef.dot(tmp[:, 2])
                if localDV and self._getNDVLocalSelf():
                    self.children[iChild].dXrefdXdvl[:, iDV] += dXrefdXdv
                    self.children[iChild].dCcdXdvl[:, iDV] += dCcdXdv
                elif self._getNDVGlobalSelf():
                    self.children[iChild].dXrefdXdvg[:, iDV] += dXrefdXdv.real
                    self.children[iChild].dCcdXdvg[:, iDV] += dCcdXdv.real
        return new_pts

    def _update_deriv_cs(self, ptSetName, config=None):

        """
        A version of the update_deriv function specifically for use
        in the computeTotalJacobianCS function. """
        new_pts = numpy.zeros((self.nPtAttachFull, 3), 'D')

        # Make sure coefficients are complex
        self._complexifyCoef()

        # Set all coef Values back to initial values
        if not self.isChild:
            self.FFD.coef = self.FFD.coef.astype('D')
            self._setInitialValues()
        else:
            # Update all coef
            self.FFD.coef = self.FFD.coef.astype('D')
            self.FFD._updateVolumeCoef()

            # Evaluate starting pointset
            Xstart = self.FFD.getAttachedPoints(ptSetName)

            # Now we have to propagate the complex part through Xstart
            tempCoef = self.FFD.coef.copy().astype('D')
            Xstart = Xstart.astype('D')
            imag_part = numpy.imag(tempCoef)
            imag_j = 1j

            dPtdCoef = self.FFD.embededVolumes[ptSetName].dPtdCoef
            if dPtdCoef is not None:
                for ii in range(3):
                    Xstart[:, ii] += imag_j*dPtdCoef.dot(imag_part[:, ii])

        # Step 1: Call all the design variables IFF we have ref axis:
        if len(self.axis) > 0:

            # Compute changes due to global design vars
            self.updateCalculations(new_pts, isComplex=True, config=config)

            # Put the update FFD points in their proper place
            numpy.put(self.FFD.coef[:, 0], self.ptAttachInd, new_pts[:, 0])
            numpy.put(self.FFD.coef[:, 1], self.ptAttachInd, new_pts[:, 1])
            numpy.put(self.FFD.coef[:, 2], self.ptAttachInd, new_pts[:, 2])

        # Apply the real and complex parts separately
        for key in self.DV_listSectionLocal:
            self.DV_listSectionLocal[key](self.FFD.coef, self.coefRotM, config)
            self.DV_listSectionLocal[key].updateComplex(self.FFD.coef, self.coefRotM, config)

        for key in self.DV_listLocal:
            self.DV_listLocal[key](self.FFD.coef, config)
            self.DV_listLocal[key].updateComplex(self.FFD.coef, config)

        # Update all coef
        self.FFD._updateVolumeCoef()

        # Evaluate coordinates from the parent
        Xfinal = self.FFD.getAttachedPoints(ptSetName)

        # now project derivs through from the coef to the pts
        Xfinal = Xfinal.astype('D')
        imag_part = numpy.imag(self.FFD.coef)
        imag_j = 1j

        dPtdCoef = self.FFD.embededVolumes[ptSetName].dPtdCoef
        if dPtdCoef is not None:
            for ii in range(3):
                Xfinal[:, ii] += imag_j*dPtdCoef.dot(imag_part[:, ii])

        # now do the same for the children
        for iChild in range(len(self.children)):
            #first, update the coef. to their new locations
            child = self.children[iChild]
            child._finalize()
            self.applyToChild(iChild)

            # now cast forward the complex part of the derivative
            child._complexifyCoef()
            child.FFD.coef = child.FFD.coef.astype('D')

            dXrefdCoef = self.FFD.embededVolumes['child%d_axis'%(iChild)].dPtdCoef
            dCcdCoef = self.FFD.embededVolumes['child%d_coef'%(iChild)].dPtdCoef

            if dXrefdCoef is not None:
                for ii in range(3):
                    child.coef[:, ii] += imag_j*dXrefdCoef.dot(imag_part[:, ii])

            if dCcdCoef is not None:
                for ii in range(3):
                    child.FFD.coef[:, ii] += imag_j*dCcdCoef.dot(imag_part[:, ii])
            child.refAxis.coef = child.coef.copy()
            child.refAxis._updateCurveCoef()
            Xfinal += child._update_deriv_cs(ptSetName, config=config)
            child._unComplexifyCoef()

        self.FFD.coef = self.FFD.coef.astype('d')

        if self.isChild:
            return Xfinal - Xstart
        else:
            return Xfinal


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
                    self.refAxis.curves[i].coef.astype('D'))
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

    def computeTotalJacobianFD(self, ptSetName, config=None):
        """This function takes the total derivative of an objective,
        I, with respect the points controlled on this processor using FD.
        We take the transpose prodducts and mpi_allreduce them to get the
        resulting value on each processor. Note that this function is slow
        and should eventually be replaced by an analytic version.
        """

        print('Calculate DV Jacobian using FD...')
        self._finalize()
        self.curPtSet = ptSetName

        if not(self.JT[ptSetName] is None):
            return

        if self.isChild:
            refFFDCoef = copy.copy(self.FFD.coef)
            refCoef = copy.copy(self.coef)

        coords0 = self.update(ptSetName, config=config).flatten()

        if self.nPts[ptSetName] == None:
            self.nPts[ptSetName] = len(coords0.flatten())
        for child in self.children:
            child.nPts[ptSetName] = self.nPts[ptSetName]

        DVGlobalCount, DVLocalCount, DVSecLocCount = self._getDVOffsets()

        h = 1e-6

        self.JT[ptSetName] = numpy.zeros([self.nDV_T,self.nPts[ptSetName]])

        for key in self.DV_listGlobal:
            for j in range(self.DV_listGlobal[key].nVal):
                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = refCoef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listGlobal[key].value[j]

                self.DV_listGlobal[key].value[j] += h

                coordsph = self.update(ptSetName, config=config).flatten()

                deriv = (coordsph-coords0)/h
                self.JT[ptSetName][DVGlobalCount,:]=deriv

                DVGlobalCount += 1
                self.DV_listGlobal[key].value[j] = refVal

        for key in self.DV_listSectionLocal:
            for j in range(self.DV_listSectionLocal[key].nVal):
                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = refCoef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listSectionLocal[key].value[j]

                self.DV_listSectionLocal[key].value[j] += h
                coordsph = self.update(ptSetName, config=config).flatten()

                deriv = (coordsph-coords0)/h
                self.JT[ptSetName][DVSecLocCount,:]=deriv

                DVSecLocCount += 1
                self.DV_listSectionLocal[key].value[j] = refVal

        for key in self.DV_listLocal:
            for j in range(self.DV_listLocal[key].nVal):
                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = refCoef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listLocal[key].value[j]

                self.DV_listLocal[key].value[j] += h
                coordsph = self.update(ptSetName, config=config).flatten()

                deriv = (coordsph-coords0)/h
                self.JT[ptSetName][DVLocalCount,:]=deriv

                DVLocalCount += 1
                self.DV_listLocal[key].value[j] = refVal

        for iChild in range(len(self.children)):
            child = self.children[iChild]
            child._finalize()

            # In the updates applied previously, the FFD points on the children
            # will have been set as deltas. We need to set them as absolute
            # coordinates based on the changes in the parent before moving down
            # to the next level
            self.applyToChild(iChild)

            # Now get jacobian from child and add to parent jacobian
            child.computeTotalJacobianFD(ptSetName, config=config)
            self.JT[ptSetName] = self.JT[ptSetName] + child.JT[ptSetName]

        return

    def _attachedPtJacobian(self, config):
        """
        Compute the derivative of the the attached points
        """
        nDV = self._getNDVGlobalSelf()

        self._getDVOffsets()

        h = 1.0e-40j
        oneoverh = 1.0/1e-40
        # Just do a CS loop over the coef
        # First sum the actual number of globalDVs
        if nDV != 0: # check this
            # create a jacobian the size of nPtAttached full by self.nDV_T, the total number of
            # dvs
            Jacobian = numpy.zeros((self.nPtAttachFull*3, self.nDV_T))

            # Create the storage arrays for the information that must be
            # passed to the children
            for iChild in range(len(self.children)):
                N = self.FFD.embededVolumes['child%d_axis'%(iChild)].N
                # Derivative of reference axis points wrt global DVs at this level
                self.children[iChild].dXrefdXdvg = numpy.zeros((N*3, self.nDV_T))

                N = self.FFD.embededVolumes['child%d_coef'%(iChild)].N
                # derivative of the control points wrt the global DVs at this level
                self.children[iChild].dCcdXdvg = numpy.zeros((N*3, self.nDV_T))

            # We need to save the reference state so that we can always start
            # from the same place when calling _update_deriv
            refFFDCoef = copy.copy(self.FFD.coef)
            refCoef = copy.copy(self.coef)

            iDV = self.nDVG_count
            for key in self.DV_listGlobal:
                if self.DV_listGlobal[key].config is None or \
                   config is None or \
                   any(c0 == config for c0 in self.DV_listGlobal[key].config):
                    nVal = self.DV_listGlobal[key].nVal
                    for j in range(nVal):

                        refVal = self.DV_listGlobal[key].value[j]

                        self.DV_listGlobal[key].value[j] += h

                        # Reset coefficients
                        self.FFD.coef = refFFDCoef.astype('D') # ffd coefficients
                        self.coef = refCoef.astype('D')
                        self.refAxis.coef = refCoef.astype('D')
                        self.refAxis._updateCurveCoef()
                        self._complexifyCoef()  # Make sure coefficients are complex

                        deriv = oneoverh*numpy.imag(self._update_deriv(iDV,h,oneoverh,config=config)).flatten()
                        # reset the FFD and axis
                        self._unComplexifyCoef()
                        self.FFD.coef = self.FFD.coef.real.astype('d')

                        numpy.put(Jacobian[0::3, iDV], self.ptAttachInd,
                                  deriv[0::3])
                        numpy.put(Jacobian[1::3, iDV], self.ptAttachInd,
                                deriv[1::3])
                        numpy.put(Jacobian[2::3, iDV], self.ptAttachInd,
                                  deriv[2::3])

                        iDV += 1

                        self.DV_listGlobal[key].value[j] = refVal
                else:
                    iDV += self.DV_listGlobal[key].nVal
        else:
            Jacobian = None

        return Jacobian

    def _sectionlocalDVJacobian(self, config=None):
        """
        Return the derivative of the coefficients wrt the local normal design
        variables
        """
        # This is relatively straight forward, since the matrix is
        # entirely one's or zeros
        nDV = self._getNDVSectionLocalSelf()
        self._getDVOffsets()

        if nDV != 0:
            Jacobian = sparse.lil_matrix((self.nPtAttachFull*3, self.nDV_T))

            # Create the storage arrays for the information that must be
            # passed to the children

            for iChild in range(len(self.children)):
                N = self.FFD.embededVolumes['child%d_axis'%(iChild)].N
                self.children[iChild].dXrefdXdvl = numpy.zeros((N*3, self.nDV_T))

                N = self.FFD.embededVolumes['child%d_coef'%(iChild)].N
                self.children[iChild].dCcdXdvl = numpy.zeros((N*3, self.nDV_T))

            iDVSectionLocal = self.nDVSL_count
            for key in self.DV_listSectionLocal:
                dv = self.DV_listSectionLocal[key]
                if dv.config is None or config is None or any(c0 == config for c0 in dv.config):
                    nVal = dv.nVal

                    self.DV_listSectionLocal[key](self.FFD.coef, self.coefRotM, config)

                    for j in range(nVal):
                        coef = dv.coefList[j]  # affected control point
                        T = dv.sectionTransform[dv.sectionLink[coef]]
                        inFrame = numpy.zeros((3,1))
                        # Set axis that is being perturbed to 1.0
                        inFrame[dv.axis] = 1.0

                        R = numpy.real(self.coefRotM[coef])
                        rows = range(coef*3,(coef+1)*3)
                        Jacobian[rows, iDVSectionLocal] += R.dot(T.dot(inFrame))

                        for iChild in range(len(self.children)):

                            dXrefdCoef = self.FFD.embededVolumes['child%d_axis'%(iChild)].dPtdCoef
                            dCcdCoef   = self.FFD.embededVolumes['child%d_coef'%(iChild)].dPtdCoef

                            tmp = numpy.zeros(self.FFD.coef.shape, dtype='d')

                            tmp[coef,:] = R.dot(T.dot(inFrame)).flatten()

                            dXrefdXdvl = numpy.zeros((dXrefdCoef.shape[0]*3),'d')
                            dCcdXdvl   = numpy.zeros((dCcdCoef.shape[0]*3),'d')

                            dXrefdXdvl[0::3] = dXrefdCoef.dot(tmp[:, 0])
                            dXrefdXdvl[1::3] = dXrefdCoef.dot(tmp[:, 1])
                            dXrefdXdvl[2::3] = dXrefdCoef.dot(tmp[:, 2])

                            dCcdXdvl[0::3] = dCcdCoef.dot(tmp[:, 0])
                            dCcdXdvl[1::3] = dCcdCoef.dot(tmp[:, 1])
                            dCcdXdvl[2::3] = dCcdCoef.dot(tmp[:, 2])

                            # TODO: the += here is to allow recursion check this with multiple nesting
                            # levels
                            self.children[iChild].dXrefdXdvl[:, iDVSectionLocal] += dXrefdXdvl
                            self.children[iChild].dCcdXdvl[:, iDVSectionLocal] += dCcdXdvl
                        iDVSectionLocal += 1
                else:
                    iDVSectionLocal += self.DV_listSectionLocal[key].nVal

                # end if config check
            # end for
        else:
            Jacobian = None


        return Jacobian

    def _localDVJacobian(self, config=None):
        """
        Return the derivative of the coefficients wrt the local design
        variables
        """

        # This is relatively straight forward, since the matrix is
        # entirely one's or zeros
        nDV = self._getNDVLocalSelf()
        self._getDVOffsets()

        if nDV != 0:
            Jacobian = sparse.lil_matrix((self.nPtAttachFull*3, self.nDV_T))

            # Create the storage arrays for the information that must be
            # passed to the children
            for iChild in range(len(self.children)):
                N = self.FFD.embededVolumes['child%d_axis'%(iChild)].N
                self.children[iChild].dXrefdXdvl = numpy.zeros((N*3, self.nDV_T))

                N = self.FFD.embededVolumes['child%d_coef'%(iChild)].N
                self.children[iChild].dCcdXdvl = numpy.zeros((N*3, self.nDV_T))

            iDVLocal = self.nDVL_count
            for key in self.DV_listLocal:
                if self.DV_listLocal[key].config is None or \
                   config is None or \
                   any(c0 == config for c0 in self.DV_listLocal[key].config):

                    self.DV_listLocal[key](self.FFD.coef, config)

                    nVal = self.DV_listLocal[key].nVal
                    for j in range(nVal):
                        pt_dv = self.DV_listLocal[key].coefList[j]
                        irow = pt_dv[0]*3 + pt_dv[1]
                        Jacobian[irow, iDVLocal] = 1.0

                        for iChild in range(len(self.children)):
                            # Get derivatives of child ref axis and FFD control
                            # points w.r.t. parent's FFD control points
                            dXrefdCoef = self.FFD.embededVolumes['child%d_axis'%(iChild)].dPtdCoef
                            dCcdCoef   = self.FFD.embededVolumes['child%d_coef'%(iChild)].dPtdCoef

                            tmp = numpy.zeros(self.FFD.coef.shape,dtype='d')

                            tmp[pt_dv[0],pt_dv[1]] = 1.0

                            dXrefdXdvl = numpy.zeros((dXrefdCoef.shape[0]*3),'d')
                            dCcdXdvl   = numpy.zeros((dCcdCoef.shape[0]*3),'d')

                            dXrefdXdvl[0::3] = dXrefdCoef.dot(tmp[:, 0])
                            dXrefdXdvl[1::3] = dXrefdCoef.dot(tmp[:, 1])
                            dXrefdXdvl[2::3] = dXrefdCoef.dot(tmp[:, 2])

                            dCcdXdvl[0::3] = dCcdCoef.dot(tmp[:, 0])
                            dCcdXdvl[1::3] = dCcdCoef.dot(tmp[:, 1])
                            dCcdXdvl[2::3] = dCcdCoef.dot(tmp[:, 2])

                            # TODO: the += here is to allow recursion check this with multiple nesting
                            # levels
                            self.children[iChild].dXrefdXdvl[:, iDVLocal] += dXrefdXdvl
                            self.children[iChild].dCcdXdvl[:, iDVLocal] += dCcdXdvl
                        iDVLocal += 1
                else:
                    iDVLocal += self.DV_listLocal[key].nVal

                # end if config check
            # end for
        else:
            Jacobian = None


        return Jacobian

    def _cascadedDVJacobian(self, config=None):
        """
        Compute the cascading derivatives from the parent to the child
        """

        if not self.isChild:
            return None

        # we are now on a child. Add in dependence passed from parent
        Jacobian = sparse.lil_matrix((self.nPtAttachFull*3, self.nDV_T))

        # Save reference values (these are necessary so that we always start
        # from the base state on the current DVGeo, and then apply the design
        # variables from there).
        refFFDCoef = copy.copy(self.FFD.coef)
        refCoef = copy.copy(self.coef)

        h = 1.0e-40j
        oneoverh = 1.0/1e-40
        if self.dXrefdXdvg is not None:
            for iDV in range(self.dXrefdXdvg.shape[1]):
                # print('Coef', self.FFD.coef)
                nz1 = numpy.count_nonzero(self.dXrefdXdvg[:, iDV])
                nz2 = numpy.count_nonzero(self.dCcdXdvg[:, iDV])
                if nz1 + nz2 == 0:
                    continue

                # Complexify all of the coefficients
                self.FFD.coef = refFFDCoef.astype('D')
                self.coef = refCoef.astype('D')
                self._complexifyCoef()

                # Add a complex pertubation representing the change in the child
                # reference axis wrt the parent global DVs
                self.coef[:,0] +=  self.dXrefdXdvg[0::3, iDV]*h
                self.coef[:,1] +=  self.dXrefdXdvg[1::3, iDV]*h
                self.coef[:,2] +=  self.dXrefdXdvg[2::3, iDV]*h

                # insert the new coef into the refAxis
                self.refAxis.coef = self.coef.copy()
                self.refAxis._updateCurveCoef()

                #Complexify the child FFD coords
                tmp1 = numpy.zeros_like(self.FFD.coef,dtype='D')

                # add the effect of the global coordinates on the actual control points
                tmp1[:,0] = self.dCcdXdvg[0::3, iDV]*h
                tmp1[:,1] = self.dCcdXdvg[1::3, iDV]*h
                tmp1[:,2] = self.dCcdXdvg[2::3, iDV]*h

                self.FFD.coef+=tmp1

                #Store the original FFD coordinates so that we can get the delta
                oldCoefLocations = self.FFD.coef.copy()

                # compute the deriv of the child FFD coords wrt the parent by processing
                # the above CS perturbation
                new_pts = self._update_deriv(iDV,h,oneoverh, config=config)

                # insert this result in the the correct locations of a vector the correct
                # size
                numpy.put(self.FFD.coef[:, 0], self.ptAttachInd, new_pts[:, 0])
                numpy.put(self.FFD.coef[:, 1], self.ptAttachInd, new_pts[:, 1])
                numpy.put(self.FFD.coef[:, 2], self.ptAttachInd, new_pts[:, 2])

                # We have to subtract off the oldCoefLocations because we only
                # want the cascading effect on the current design variables. The
                # complex part on oldCoefLocations was already accounted for on
                # the parent.
                self.FFD.coef -= oldCoefLocations

                # sum up all of the various influences
                Jacobian[0::3, iDV] += oneoverh*numpy.imag(self.FFD.coef[:,0:1])
                Jacobian[1::3, iDV] += oneoverh*numpy.imag(self.FFD.coef[:,1:2])
                Jacobian[2::3, iDV] += oneoverh*numpy.imag(self.FFD.coef[:,2:3])

                # decomplexify the coefficients
                self.coef = self.coef.real.astype('d')
                self.FFD.coef = self.FFD.coef.real.astype('d')
                self._unComplexifyCoef()

        if self.dXrefdXdvl is not None:
            # Now repeat for the local variables
            for iDV in range(self.dXrefdXdvl.shape[1]):
                # check if there is any dependence on this DV
                nz1 = numpy.count_nonzero(self.dXrefdXdvl[:, iDV])
                nz2 = numpy.count_nonzero(self.dCcdXdvl[:, iDV])
                if nz1 + nz2 == 0:
                    continue

                # Complexify all of the coefficients
                self.FFD.coef = refFFDCoef.astype('D')
                self.coef = refCoef.astype('D')
                self._complexifyCoef()

                # Add a complex pertubation representing the change in the child
                # reference axis wrt the parent local DVs
                self.coef[:,0] +=  self.dXrefdXdvl[0::3, iDV]*h
                self.coef[:,1] +=  self.dXrefdXdvl[1::3, iDV]*h
                self.coef[:,2] +=  self.dXrefdXdvl[2::3, iDV]*h

                # insert the new coef into the refAxis
                self.refAxis.coef = self.coef.copy()
                self.refAxis._updateCurveCoef()

                #Complexify the child FFD coords
                tmp1 = numpy.zeros_like(self.FFD.coef,dtype='D')

                # add the effect of the global coordinates on the actual control points
                tmp1[:,0] = self.dCcdXdvl[0::3, iDV]*h
                tmp1[:,1] = self.dCcdXdvl[1::3, iDV]*h
                tmp1[:,2] = self.dCcdXdvl[2::3, iDV]*h

                self.FFD.coef += tmp1

                #Store the original FFD coordinates so that we can get the delta
                oldCoefLocations = self.FFD.coef.copy()

                # compute the deriv of the child FFD coords wrt the parent by processing
                # the above CS perturbation
                new_pts = self._update_deriv(iDV, h, oneoverh, config=config,localDV=True)
                numpy.put(self.FFD.coef[:, 0], self.ptAttachInd, new_pts[:, 0])
                numpy.put(self.FFD.coef[:, 1], self.ptAttachInd, new_pts[:, 1])
                numpy.put(self.FFD.coef[:, 2], self.ptAttachInd, new_pts[:, 2])

                # We have to subtract off the oldCoefLocations because we only
                # want the cascading effect on the current design variables. The
                # complex part on oldCoefLocations was already accounted for on
                # the parent.
                self.FFD.coef -= oldCoefLocations

                # sum up all of the various influences
                Jacobian[0::3, iDV] += oneoverh*numpy.imag(self.FFD.coef[:,0:1])
                Jacobian[1::3, iDV] += oneoverh*numpy.imag(self.FFD.coef[:,1:2])
                Jacobian[2::3, iDV] += oneoverh*numpy.imag(self.FFD.coef[:,2:3])

                # decomplexify the coefficients
                self.coef = self.coef.real.astype('d')
                self.FFD.coef = self.FFD.coef.real.astype('d')
                self._unComplexifyCoef()

        return Jacobian

    def _writeVols(self, handle, vol_counter):
        for i in range(len(self.FFD.vols)):
            pySpline.writeTecplot3D(handle, 'vol%d'%i, self.FFD.vols[i].coef)
            vol_counter += 1

        # Write children volumes:
        for iChild in range(len(self.children)):
            vol_counter += self.children[iChild]._writeVols(handle, vol_counter)

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
        self.zeroJacobians(ptSetName)
        for child in self.children:
            child.zeroJacobians(ptSetName)

        self.computeTotalJacobian(ptSetName)
        # self.computeTotalJacobian_fast(ptSetName)

        Jac = copy.deepcopy(self.JT[ptSetName])

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
        DVCountGlob, DVCountLoc, DVCountSecLoc = self._getDVOffsets()

        for key in self.DV_listGlobal:
            for j in range(self.DV_listGlobal[key].nVal):

                print('========================================')
                print('      GlobalVar(%s), Value(%d)'%(key, j))
                print('========================================')

                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = self.coef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listGlobal[key].value[j]

                self.DV_listGlobal[key].value[j] += h

                coordsph = self.update(ptSetName).flatten()

                deriv = (coordsph-coords0)/h

                for ii in range(len(deriv)):

                    relErr = (deriv[ii] - Jac[DVCountGlob, ii])/(
                        1e-16 + Jac[DVCountGlob, ii])
                    absErr = deriv[ii] - Jac[DVCountGlob,ii]

                    if abs(relErr) > h*10 and abs(absErr) > h*10:
                        print(ii, deriv[ii], Jac[DVCountGlob, ii], relErr, absErr)

                DVCountGlob += 1
                self.DV_listGlobal[key].value[j] = refVal

        for key in self.DV_listLocal:
            for j in range(self.DV_listLocal[key].nVal):

                print('========================================')
                print('      LocalVar(%s), Value(%d)           '%(key, j))
                print('========================================')

                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = self.coef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listLocal[key].value[j]

                self.DV_listLocal[key].value[j] += h
                coordsph = self.update(ptSetName).flatten()

                deriv = (coordsph-coords0)/h

                for ii in range(len(deriv)):
                    relErr = (deriv[ii] - Jac[DVCountLoc, ii])/(
                        1e-16 + Jac[DVCountLoc, ii])
                    absErr = deriv[ii] - Jac[DVCountLoc,ii]

                    if abs(relErr) > h and abs(absErr) > h:
                        print(ii, deriv[ii], Jac[DVCountLoc, ii], relErr, absErr)

                DVCountLoc += 1
                self.DV_listLocal[key].value[j] = refVal

        for key in self.DV_listSectionLocal:
            for j in range(self.DV_listSectionLocal[key].nVal):

                print('========================================')
                print('   SectionLocalVar(%s), Value(%d)       '%(key, j))
                print('========================================')

                if self.isChild:
                    self.FFD.coef = refFFDCoef.copy()
                    self.coef = refCoef.copy()
                    self.refAxis.coef = self.coef.copy()
                    self.refAxis._updateCurveCoef()

                refVal = self.DV_listSectionLocal[key].value[j]

                self.DV_listSectionLocal[key].value[j] += h
                coordsph = self.update(ptSetName).flatten()

                deriv = (coordsph-coords0)/h

                for ii in range(len(deriv)):
                    relErr = (deriv[ii] - Jac[DVCountSecLoc, ii])/(
                        1e-16 + Jac[DVCountSecLoc, ii])
                    absErr = deriv[ii] - Jac[DVCountSecLoc,ii]

                    if abs(relErr) > h and abs(absErr) > h:
                        print(ii, deriv[ii], Jac[DVCountSecLoc, ii], relErr, absErr)

                DVCountSecLoc += 1
                self.DV_listSectionLocal[key].value[j] = refVal

        for child in self.children:
            child.checkDerivatives(ptSetName)

    def printDesignVariables(self):
        """
        Print a formatted list of design variables to the screen
        """
        for dg in self.DV_listGlobal:
            print('%s'%(self.DV_listGlobal[dg].name))
            for i in range(self.DV_listGlobal[dg].nVal):
                print('%20.15f'%(self.DV_listGlobal[dg].value[i]))

        for dl in self.DV_listLocal:
            print('%s'%(self.DV_listLocal[dl].name))
            for i in range(self.DV_listLocal[dl].nVal):
                print('%20.15f'%(self.DV_listLocal[dl].value[i]))

        for dsl in self.DV_listSectionLocal:
            print('%s'%(self.DV_listSectionLocal[dsl].name))
            for i in range(self.DV_listSectionLocal[dsl].nVal):
                print('%20.15f'%(self.DV_listSectionLocal[dsl].value[i]))

        for child in self.children:
            child.printDesignVariables()

    def sectionFrame(self, sectionIndex, sectionTransform, sectionLink, ivol=0,
                     orient0=None, orient2='svd'):
        """
        This function computes a unique reference coordinate frame for each
        section of an FFD volume. You can choose which axis of the FFD you would
        like these sections to be defined by. For example, if we have a wing
        with a winglet, the airfoil sections which make up the wing will not all
        lie in parallel planes. We want to find a reference frame for each of
        these airfoil sections so that we can constrain local control points to
        deform within the sectional plane. Let's say the wing FFD is oriented
        with indices:
            'i' - along chord
            'j' - normal to wing surface
            'k' - along span
        If we choose sectionIndex='k', this function will compute a frame which
        has two axes aligned with the k-planes of the FFD volume. This is useful
        because in some cases (as with a winglet), we want to perturb sectional
        control points within the section plane instead of in the global
        coordinate directions.

        Assumptions:
            - the normal direction is computed along the block index with size 2
            - all point for a given sectionIndex lie within a plane

        Parameters
        ----------
        sectionIndex : 'i', 'j', or 'k'
            This the index of the FFD which defines a section plane.

        orient0 : None, 'i', 'j', 'k', or numpy vector. Default is None.
            Although secIndex defines the '2' axis, the '0' and '1' axes are still
            free to rotate within the section plane. We will choose the orientation
            of the '0' axis and let '1' be orthogonal. See addGeoDVSectionLocal
            for a more detailed description.

        ivol : integer
            Volume ID for the volume in which section normals will be computed.

        alignStreamwise : 'x', 'y', or 'z' (optional)
            If given, section frames are rotated about the k-plane normal
            so that the longitudinal axis is parallel with the given streamwise
            direction.

        rootGlobal : list
            List of sections along specified axis that will be fixed to the
            global coordinate frame.

        Output
        ------
        sectionTransform : list of 3x3 arrays
            List of transformation matrices for the sections of a given volume.
            Transformations are set up from local section frame to global frame.
        """
        xyz_2_idx = {'x':0, 'y':1, 'z':2}
        ijk_2_idx = {'i':0, 'j':1, 'k':2}
        lIndex = self.FFD.topo.lIndex[ivol]

        # Get normal index
        orient0idx = False
        orient0vec = False
        if orient0 is not None:
            if type(orient0) is str:
                orient0 = ijk_2_idx[orient0.lower()]
                orient0idx = True
            elif type(orient0) is numpy.ndarray:
                orient0vec = True
            else:
                raise Error('orient0 must be an index (i, j, or k) or a '
                            'vector.')
        # Get section index and number of sections
        sectionIndex = ijk_2_idx[sectionIndex.lower()]
        nSections = lIndex.shape[sectionIndex]

        # Roll lIndex so that 0th index is sectionIndex and 1st index is orient0
        rolledlIndex = numpy.rollaxis(lIndex, sectionIndex, 0)
        if orient0idx:
            if orient0 != 2:
                orient0 += 1
            rolledlIndex = numpy.rollaxis(rolledlIndex, orient0, 1)

        # Length of sectionTransform
        Tcount = len(sectionTransform)

        for i in range(nSections):
            # Compute singular value decomposition of points in section (the
            # U matrix should provide us with a pretty good approximation
            # of the transformation matrix)
            pts = self.FFD.coef[rolledlIndex[i,:,:]]
            nJ, nI = pts.shape[:-1]
            X = numpy.reshape(pts, (nI*nJ, 3))
            c = numpy.mean(X,0)
            A = X - c
            U,S,V = numpy.linalg.svd(A.T)

            # Choose section plane normal axis
            if orient2 == 'svd':
                ax2 = U[:,2]
            elif orient2 == 'ffd':
                # Use a centered FD approximation (first order at the boundaries)
                if i == 0:
                    pt = numpy.mean(self.FFD.coef[rolledlIndex[i,:,:]].reshape(nI*nJ,3), 0)
                    ptp = numpy.mean(self.FFD.coef[rolledlIndex[i+1,:,:]].reshape(nI*nJ,3), 0)
                    ax2 = ptp-pt
                elif i == nSections-1:
                    pt = numpy.mean(self.FFD.coef[rolledlIndex[i,:,:]].reshape(nI*nJ,3), 0)
                    ptm = numpy.mean(self.FFD.coef[rolledlIndex[i-1,:,:]].reshape(nI*nJ,3), 0)
                    ax2 = pt-ptm
                else:
                    ptp = numpy.mean(self.FFD.coef[rolledlIndex[i+1,:,:]].reshape(nI*nJ,3), 0)
                    ptm = numpy.mean(self.FFD.coef[rolledlIndex[i-1,:,:]].reshape(nI*nJ,3), 0)
                    ax2 = ptp-ptm
                ax2 /= numpy.linalg.norm(ax2)
            else:
                raise Error('orient2 must be \'svd\' or \'ffd\'')

            # Options for choosing in-plane axes
            # 1. Align axis '0' with projection of the given vector on section
            #       plane.
            # 2. Align axis '0' with the projection of an average
            #       difference vector between opposing edges of FFD block
            #       section plane
            # 3. Use the default SVD decomposition (in general this will work).
            #       It will choose the chordwise direction as the best fit line
            #       through the section points.
            if orient0vec or orient0idx:
                if orient0vec:
                    u = orient0/numpy.linalg.norm(orient0)
                else:
                    u = numpy.mean((pts[-1,:] - pts[0,:]), axis=0)
                    u = u/numpy.linalg.norm(u)
                ax0 = u - u.dot(ax2)*ax2
                ax1 = numpy.cross(ax2,ax0)
            else:
                ax0 = U[:,0]
                ax1 = U[:,1]

            T = numpy.vstack((ax0,ax1,ax2)).T
            sectionTransform.append(T)
            # Designate section transformation matrix for each control point in
            # section
            sectionLink[rolledlIndex[i,:,:]] = Tcount
            Tcount += 1

            # Need to initialize coefRotM to identity matrix for case with no
            # global design variables
            for slice in rolledlIndex[i,:,:]:
                for coef in slice:
                    self.coefRotM[coef] = numpy.eye(3)

        return nSections

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

        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            # If the geo object is complex, which is indicated by .coef
            # being complex, run with complex numbers. Otherwise, convert
            # to real before calling. This eliminates casting warnings.
            if geo.coef.dtype == d or geo.complex:
                return self.function(self.value, geo)
            else:
                return self.function(numpy.real(self.value), geo)

class geoDVLocal(object):

    def __init__(self, dvName, lower, upper, scale, axis, coefListIn, mask, config):

        """Create a set of geometric design variables which change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.
        See addGeoDVLocal for more information
        """

        coefList = []
        #create a new coefficent list that excludes any values that are masked
        for i in range(len(coefListIn)):
            if mask[coefListIn[i]]==False:
                coefList.append(coefListIn[i])

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

        for i in range(len(coefList)):
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
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(self.nVal):
                coef[self.coefList[i, 0], self.coefList[i, 1]] += self.value[i].real

        return coef

    def updateComplex(self, coef, config):
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(self.nVal):
                coef[self.coefList[i, 0], self.coefList[i, 1]] += self.value[i].imag*1j

        return coef

    def mapIndexSets(self,indSetA,indSetB):
        '''
        Map the index sets from the full coefficient indices to the local set.
        '''
        # Temp is the list of FFD coefficients that are included
        # as shape variables in this localDV "key"
        temp = self.coefList
        cons = []
        for j in range(len(indSetA)):
            # Try to find this index # in the coefList (temp)
            up = None
            down = None

            # Note: We are doing inefficient double looping here
            for k in range(len(temp)):
                if temp[k][0] == indSetA[j]:
                    up = k
                if temp[k][0] == indSetB[j]:
                    down = k

            # If we haven't found up AND down do nothing
            if up is not None and down is not None:
                cons.append([up, down])

        return cons

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


class geoDVSectionLocal(object):

    def __init__(self, dvName, lower, upper, scale, axis, coefListIn, mask,
                config, sectionTransform, sectionLink):
        """
        Create a set of geometric design variables which change the shape
        of a surface.
        See addGeoDVSectionLocal for more information
        """

        self.coefList = []
        #create a new coefficent list that excludes any values that are masked
        for i in range(len(coefListIn)):
            if mask[coefListIn[i]]==False:
                self.coefList.append(coefListIn[i])

        self.nVal = len(self.coefList)
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

        self.sectionTransform = sectionTransform
        self.sectionLink = sectionLink

        self.axis = axis

    def __call__(self, coef, coefRotM, config):
        """When the object is called, apply the design variable values to
        coefficients"""
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(len(self.coefList)):
                T = self.sectionTransform[self.sectionLink[self.coefList[i]]]
                inFrame = numpy.zeros(3)
                inFrame[self.axis] = self.value[i].real

                R = coefRotM[self.coefList[i]].real
                coef[self.coefList[i]] += R.dot(T.dot(inFrame))
        return coef

    def updateComplex(self, coef, coefRotM, config):
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(len(self.coefList)):
                T = self.sectionTransform[self.sectionLink[self.coefList[i]]]
                inFrame = numpy.zeros(3, 'D')
                inFrame[self.axis] = self.value[i]

                R = coefRotM[self.coefList[i]]
                coef[self.coefList[i]] += R.dot(T.dot(inFrame)).imag*1j
        return coef

    def mapIndexSets(self,indSetA,indSetB):
        '''
        Map the index sets from the full coefficient indices to the local set.
        '''
        # Temp is the list of FFD coefficients that are included
        # as shape variables in this localDV "key"
        temp = self.coefList
        cons = []
        for j in range(len(indSetA)):
            # Try to find this index # in the coefList (temp)
            up = None
            down = None

            # Note: We are doing inefficient double looping here
            for k in range(len(temp)):
                if temp[k] == indSetA[j]:
                    up = k
                if temp[k] == indSetB[j]:
                    down = k

            # If we haven't found up AND down do nothing
            if up is not None and down is not None:
                cons.append([up, down])

        return cons
