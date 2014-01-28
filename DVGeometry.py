# =============================================================================
# DVGeometry Deals with all the details of taking user supplied design
# variables and mapping it to the descrete surfaces on the CFD and FEA
# disciplines. In the case of the FE discipline, ALL the FE nodes are
# included (distributed in a volume) and as such this currently only
# works with Free Form Deformation Volumes that can deal with these
# spactialy distributed points. 
# =============================================================================

import copy
from collections import OrderedDict
import numpy
from scipy import sparse
from mpi4py import MPI
from pyspline import pySpline
from pyNetwork import pyNetwork
from pyBlock import pyBlock
import geo_utils

class DVGeometry(object):
    """
    Create a DV Geometry module to handle all design variable
    manipulation

    DVGeometry uses the free-form deformation (FFD) volume
    approach for geometric manipualation.

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
        """

    def __init__(self, fileName, complex=False, child=False, *args, **kwargs):
        
        self.DV_listGlobal  = OrderedDict() # Global Design Variable List
        self.DV_listLocal   = OrderedDict() # Local Design Variable List

        # Flags to determine if this DVGeometry is a parent or child
        self.isChild  = child
        self.children = []
        self.iChild = None
        self.nChildren = None
        self.points = OrderedDict()

        self.complex = kwargs.pop('complex', False)
        if self.complex:
            self.dtype = 'D'
        else:
            self.dtype = 'd'
        # end if

        # Load the FFD file in FFD mode. Also note that args and
        # kwargs are passed through in case aditional pyBlock options
        # need to be set. 
        self.FFD = pyBlock('plot3d', fileName=fileName, FFD=True,
                           *args, **kwargs)

        # Jacobians:
        # self.JT: Total transpose jacobian for self.J_name
        self.JT = None
        self.J_name = None
        self.J_attach = None
        self.J_local  = None

        # Derivatives of Xref and Coef provided by the parent to the
        # children
        self.dXrefdXdvg = None
        self.dCoefdXdvg = None

        self.dXrefdXdvl = None
        self.dCoefdXdvl = None

        self.axis = OrderedDict()
        return 

    def addRefAxis(self, name, curve=None,  xFraction=None, volumes=None, rotType=5,
                   axis='x'):
        """
        This function is used to add a 'reference' axis to the
        DVGeometry object.  Adding a reference axis is only required
        when 'global' design variables are to be used, i.e. variables
        like span, sweep, chord etc --- Variables that affect many FFD
        control points.

        There are two different ways that a reference can be
        specified:
 
        #. The first is explictly a pySpline curve object using the
           keyword argument curve=<curve>.

        #. The second is to specifiy the xFraction variable. There are
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
            Integer in rane 0->6 (inclusive) to determine the order
            that the rotations are made. 

            0. Intrinsic rotation, rot_theta is roation about axis
            1. x-y-z
            2. x-z-y
            3. y-z-x  
            4. y-x-z
            5. z-x-y  Default (x-streamwise y-up z-out wing)
            6. z-y-x

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
        # dictionary structure. All the finalization is performed in
        # finalizeAxis() which will be called automatically when the
        # firt design variables are added. 

        if axis.lower() == 'x':
            axis = [1, 0, 0]
        elif axis.lower() == 'y':
            axis = [0, 1, 0]
        elif axis.lower() == 'z':
            axis = [0, 0, 1]
        # end if
        axis = numpy.array(axis)
        
        if curve is not None:
            # Explict curve has been supplied:
            if volumes is None:
                volumes = numpy.arange(self.FFD.nVol)
                
            self.axis[name] = {'curve':curve, 'volumes':volumes,
                               'rotType':rotType, 'axis':axis}
            nAxis = len(curve.coef)
        elif xFraction is not None:
            raise ValueError('xFraction specification is not coded yet.')
        else:
            raise ValueError('One of \'curve\' or \'xFraction\' must be \
specified for a call to addRefAxis')
        # end if

        return nAxis

    def _finalizeAxis(self):
        """
        Internal function that sets up the collection of curve that
        the user has added one at a time. This will create the
        internal pyNetwork object
        """

        curves = []
        for axis in self.axis:
            curves.append[axis['curve']]

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
            self.rot_x[key] = pySpline.curve(t=t, k=k, coef=z.copy())
            self.rot_y[key] = pySpline.curve(t=t, k=k, coef=z.copy())
            self.rot_z[key] = pySpline.curve(t=t, k=k, coef=z.copy())
            self.rot_theta[key] = pySpline.curve(t=t, k=k, coef=z.copy())
            self.scale[key] = pySpline.curve(t=t, k=k, coef=z.copy())
            self.scale_x[key] = pySpline.curve(t=t, k=k, coef=z.copy())
            self.scale_y[key] = pySpline.curve(t=t, k=k, coef=z.copy())
            self.scale_z[key] = pySpline.curve(t=t, k=k, coef=z.copy())
        # end for

        # Need to keep track of initail scale values
        self.scale0 = self.scale.copy()
        self.scale_x0 = self.scale_x.copy()
        self.scale_y0 = self.scale_y.copy()
        self.scale_z0 = self.scale_z.copy()
        self.rot_x0 = self.rot_x.copy()
        self.rot_y0 = self.rot_y.copy()
        self.rot_z0 = self.rot_z.copy()
        self.rot_theta0 = self.rot_theta.copy()

   
    def addPointSet(self, points, ptName, **kwargs):
        """ Embed a set of points ((N,3) array) with name 'ptName'
        into the DVGeometry object"""
 
        points = numpy.array(points).real.astype('d')
        self.points[ptName] = points

        # Project the last set of points into the volume
        if self.isChild:
            coef_mask = self.FFD.attachPoints(
                self.points[ptName], ptName, interiorOnly=True, **kwargs)
        else:
            coef_mask = self.FFD.attachPoints(
                self.points[ptName], ptName, interiorOnly=False)
        # end if
        self.FFD._calcdPtdCoef(ptName)

        self.ptAttachInd = []
        self.ptAttachPtr = [0]
        for ii in xrange(len(self.vol_list)):
            temp = []
            for iVol in self.vol_list[ii]:
                for i in xrange(self.FFD.vols[iVol].Nctlu):
                    for j in xrange(self.FFD.vols[iVol].Nctlv):
                        for k in xrange(self.FFD.vols[iVol].Nctlw):
                            ind = self.FFD.topo.l_index[iVol][i, j, k]
                            if coef_mask[ind] == False:
                                temp.append(ind)
                            # end if
                        # end for
                    # end for
                # end for
            # end for

            # Unique the values we just added:
            temp = geo_utils.unique(temp)
            self.ptAttachInd.extend(temp)
            self.ptAttachPtr.append(len(self.ptAttachInd))
        # end for

        # Convert the ind list to an array
        self.ptAttachInd = numpy.array(self.ptAttachInd).flatten()

        # Take the subset of the FFD cofficients as what will be
        # attached
        self.ptAttach = self.FFD.coef.take(self.ptAttachInd, axis=0).real
        self.ptAttachFull = self.FFD.coef.copy().real

        # Number of points attached to ref axis
        self.nPtAttach = len(self.ptAttach)
        self.nPtAttachFull = len(self.ptAttachFull)
        
        curveIDs = []
        s = []

        for ii in xrange(len(self.ptAttachPtr)-1):
            pts_to_use = self.ptAttach[
                self.ptAttachPtr[ii]:self.ptAttachPtr[ii+1], :]
            pts_to_use = self.ptAttach
            if self.axis is not None:

                ids, s0 = self.refAxis.projectRays(
                    pts_to_use, self.axis)#, curves=[ii])
            else:
                ids, s0 = self.refAxis.projectPoints(
                    pts_to_use)#, curves=[ii])
            # end for
            curveIDs.extend(ids)
            s.extend(s0)
        # end for

        self.curveIDs = numpy.array(curveIDs)
        self.links_s = s
        self.links_x = []
        self.links_n = []
        for i in xrange(self.nPtAttach):
            self.links_x.append(
                self.ptAttach[i] - \
                    self.refAxis.curves[self.curveIDs[i]](s[i]))
            deriv = self.refAxis.curves[
                self.curveIDs[i]].getDerivative(self.links_s[i])
            deriv /= geo_utils.euclidean_norm(deriv) # Normalize
            self.links_n.append(numpy.cross(deriv, self.links_x[-1]))

        self.links_x = numpy.array(self.links_x)
        self.links_s = numpy.array(self.links_s)

        return
    
    def addChild(self, childDVGeo):
        """Embed a child FFD into this object"""

        # Make sure the DVGeo being added is flaged as a child:
        if childDVGeo.isChild is False:
            print '='*80
            print 'Error: Trying to add a child FFD that has NOT been'
            print 'created as a child. This operation is illegal.'
            print '='*80
            return
        # end if

        # Extract the coef from the child FFD and ref axis and embed
        # them into the parent and compute their derivatives

        iChild = len(self.children)
        childDVGeo.iChild=iChild
        self.FFD.attachPoints(childDVGeo.FFD.coef, 'child%d_coef'%(iChild))
        self.FFD._calcdPtdCoef('child%d_coef'%(iChild))

        self.FFD.attachPoints(childDVGeo.coef, 'child%d_axis'%(iChild))
        self.FFD._calcdPtdCoef('child%d_axis'%(iChild))

        # Add the child to the parent and return
        self.children.append(childDVGeo)

        return

    def _setInitialValues(self):
        self.coef = copy.deepcopy(self.coef0)
        self.scale = copy.deepcopy(self.scale0)
        self.scale_x = copy.deepcopy(self.scale_x0)
        self.scale_y = copy.deepcopy(self.scale_y0)
        self.scale_z = copy.deepcopy(self.scale_z0)      
        self.rot_x = copy.deepcopy(self.rot_x0) 
        self.rot_y = copy.deepcopy(self.rot_y0)
        self.rot_z = copy.deepcopy(self.rot_z0)
        self.rot_theta = copy.deepcopy(self.rot_theta0)   
        
        return

    def addGeoDVGlobal(self, dvName, value, lower, upper, pyFunc):
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

        upper : float, or iterable list of floats
            The upper bound(s) for the variable(s). Same restrictions as
            'lower'

        pyFunc : python function
            The python function handle that will be used to apply the
            design variable
        """

        self.DV_listGlobal[dvName] = geo_utils.geoDVGlobal(\
                dvName, value, lower, upper, pyFunc)

        return

    def addGeoDVLocal(self, dvName, lower, upper, axis='y', pointSelect=None):
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

        axis : str. Default is 'y'
            The coordinate directions to move. Permissible values are 'x',
            'y' and 'z'. If more than one direction is required, use multiple
            calls to addGeoDVLocal with different axis values
            
        pointSelect : pointSelect object. Default is None Use a
            pointSelect object to select a subset of the total number
            of control points. See the documentation for the
            pointSelect class in geo_utils.

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

        if pointSelect is not None:
            pts, ind = pointSelect.getPoints(self.FFD.coef)
        else:
            ind = numpy.arange(self.nPtAttach)
        # end if
        self.DV_listLocal[dvName] = \
        geo_utils.geoDVLocal(dvName, lower, upper, axis, ind)
            
        return self.DV_listLocal[dvName].nVal

    def setValues(self, dvDict):
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
                vals_to_set = numpy.atleast_1d(dv_dict[key]).astype('D')
                assert len(vals_to_set) == self.DV_listGlobal[key].nVal, \
                    'Incorrect number of design variables for DV: %s\nExpecting %d variables\
 received %d variabes'%(key,self.DV_listGlobal[self.DV_namesGlobal[key]].nVal, len(vals_to_set))

                self.DV_listGlobal[key].value = vals_to_set
            # end if
            
            if key in self.DV_listGlobal:
                vals_to_set = numpy.atleast_1d(dv_dict[key])
                assert len(vals_to_set) == self.DV_listLocal[key].nVal, \
                    'Incorrect number of design variables for DV: %s'%(key)
                self.DV_listLocal[self.DV_namesLocal[key]].value = vals_to_set
            # end if

            # Jacobians are, in general, no longer up to date
            self.JT = None 
            self.J_name = None 
            self.J_attach = None
            self.J_local = None
        # end for

        # Now call setValues on the children. This way the
        # variables will be set on the children
        for child in self.children:
            child.setValues(dvDict)

        return

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
            childdvDictt = child.getValues()
            dvDict.update(childdvDict)

        return dvDict

    def _getRotMatrix(self, rotX, rotY, rotZ):
        if self.rot_type == 1:
            D = numpy.dot(rotZ, numpy.dot(rotY, rotX))
        elif self.rot_type == 2:
            D = numpy.dot(rotY, numpy.dot(rotZ, rotX))
        elif self.rot_type == 3:
            D = numpy.dot(rotX, numpy.dot(rotZ, rotY))
        elif self.rot_type == 4:
            D = numpy.dot(rotZ, numpy.dot(rotX, rotY))
        elif self.rot_type == 5:
            D = numpy.dot(rotY, numpy.dot(rotX, rotZ))
        elif self.rot_type == 6:
            D = numpy.dot(rotX, numpy.dot(rotY, rotZ))
        # end if
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
            nDV += self.DV_listLocal[i].nVal

        return nDV
        
    def extractCoef(self, axisID):
        """ Extract the coefficients for the selected reference
        axis. This should be used inside design variable functions"""

        C = numpy.zeros((len(self.refAxis.topo.l_index[axisID]),3),self.coef.dtype)
 
        C[:,0] = numpy.take(self.coef[:,0],self.refAxis.topo.l_index[axisID])
        C[:,1] = numpy.take(self.coef[:,1],self.refAxis.topo.l_index[axisID])
        C[:,2] = numpy.take(self.coef[:,2],self.refAxis.topo.l_index[axisID])

        return C

    def restoreCoef(self, coef, axisID):
        """ Restore the coefficients for the selected reference
        axis. This should be used inside design variable functions"""

        # Reset
        numpy.put(self.coef[:,0],self.refAxis.topo.l_index[axisID],coef[:,0])
        numpy.put(self.coef[:,1],self.refAxis.topo.l_index[axisID],coef[:,1])
        numpy.put(self.coef[:,2],self.refAxis.topo.l_index[axisID],coef[:,2])

        return 

    def update(self, ptSetName, childDelta=True):
        """This is pretty straight forward, perform the operations on
        the ref axis according to the design variables, then return
        the list of points provided. It is up to the user to know what
        to do with the points
        """

        # Set all coef Values back to initial values
        if not self.isChild:
            self._setInitialValues()
        
        # Step 1: Call all the design variables
        if self.complex:
            new_pts = numpy.zeros((self.nPtAttach, 3), 'D')
        else:
            new_pts = numpy.zeros((self.nPtAttach, 3), 'd')
        # end if
        if self.isChild:
            for ipt in xrange(self.nPtAttach):
                base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])
                self.links_x[ipt]=self.FFD.coef[self.ptAttachInd[ipt],:]-base_pt
            # end for
        # end if

        # Run Global Design Vars
        for i in xrange(len(self.DV_listGlobal)):
            self.DV_listGlobal[i](self)

        self.refAxis.coef = self.coef.copy()
        self.refAxis._updateCurveCoef()

        for ipt in xrange(self.nPtAttach):
            base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])

            scale = self.scale[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_x = self.scale_x[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_y = self.scale_y[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_z = self.scale_z[self.curveIDs[ipt]](self.links_s[ipt]) 
            if self.rot_type == 0:
                deriv = self.refAxis.curves[
                    self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                deriv /= geo_utils.euclidean_norm(deriv) # Normalize
                new_vec = -numpy.cross(deriv, self.links_n[ipt])
                new_vec = geo_utils.rotVbyW(new_vec, deriv, self.rot_x[
                        self.curveIDs[ipt]](self.links_s[ipt])*numpy.pi/180)
                new_pts[ipt] = base_pt + new_vec*scale
            # end if
            else:
                rotX = geo_utils.rotxM(self.rot_x[
                        self.curveIDs[ipt]](self.links_s[ipt]))
                rotY = geo_utils.rotyM(self.rot_y[
                        self.curveIDs[ipt]](self.links_s[ipt]))
                rotZ = geo_utils.rotzM(self.rot_z[
                        self.curveIDs[ipt]](self.links_s[ipt]))

                D = self.links_x[ipt]

                rotM = self._getRotMatrix(rotX, rotY, rotZ)
                D = numpy.dot(rotM, D)
                
                deriv = self.refAxis.curves[
                    self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                deriv[0] = 0.0
                deriv /= geo_utils.euclidean_norm(deriv) # Normalize
                D = geo_utils.rotVbyW(D,deriv,numpy.pi/180*self.rot_theta[              
                        self.curveIDs[ipt]](self.links_s[ipt]))
                
                D[0] *= scale_x
                D[1] *= scale_y
                D[2] *= scale_z
                new_pts[ipt] = base_pt + D*scale
            # end if
        # end for

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
            numpy.put(self.FFD.coef[:,0], self.ptAttachInd, temp[:, 0])
            numpy.put(self.FFD.coef[:,1], self.ptAttachInd, temp[:, 1])
            numpy.put(self.FFD.coef[:,2], self.ptAttachInd, temp[:, 2])

            if childDelta:
                self.FFD.coef -= oldCoefLocations
            # end if
        # end if

        for i in xrange(len(self.DV_listLocal)):
            self.DV_listLocal[i](self.FFD.coef)

        # Update all coef
        self.FFD._updateVolumeCoef()

        # Evaluate coordinates from the parent
        coords = self.FFD.getAttachedPoints(name)

        # Now loop over the children set the FFD and refAxis control
        # points as evaluated from the parent

        for iChild in xrange(len(self.children)):

            self.children[iChild].FFD.coef = self.FFD.getAttachedPoints(
                'child%d_coef'%(iChild))

            self.children[iChild].coef = self.FFD.getAttachedPoints(
                'child%d_axis'%(iChild))
            self.children[iChild].refAxis.coef =  self.children[iChild].coef.copy()
            self.children[iChild].refAxis._updateCurveCoef()

            coords += self.children[iChild].update(name, childDelta)
            
        if self.complex:
            if len(self.children) > 0:
                print ' Warning: Complex step NOT TESTED with children yet'
            # end if

            tempCoef = self.ptAttachFull.copy().astype('D')
            numpy.put(tempCoef[:, 0], self.ptAttachInd, new_pts[:, 0])
            numpy.put(tempCoef[:, 1], self.ptAttachInd, new_pts[:, 1])
            numpy.put(tempCoef[:, 2], self.ptAttachInd, new_pts[:, 2])
         
            coords = coords.astype('D')
            imag_part     = numpy.imag(tempCoef)
            imag_j = 1j

            dPtdCoef = self.FFD.embeded_volumes[name].dPtdCoef
            if dPtdCoef is not None:
                for ii in xrange(3):
                    coords[:, ii] += imag_j*dPtdCoef.dot(imag_part[:, ii])
                # end for
            # end if

            self._unComplexifyCoef()

        # end if
        return coords

    def update_deriv(self, iDV=0, h=1.0e-40j, oneoverh=1.0/1e-40):

        """Copy of update function for derivative calc"""
        
        new_pts = numpy.zeros((self.nPtAttach, 3), 'D')

        # Set all coef Values back to initial values
        if not self.isChild:
            self._setInitialValues()
        # end if

        self._complexifyCoef()

        if self.isChild:
            self.links_x = self.links_x.astype('D')
            for ipt in xrange(self.nPtAttach):
                base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])
                self.links_x[ipt]=self.FFD.coef[self.ptAttachInd[ipt],:]-base_pt
            # end for
        # end if

        # Step 1: Call all the design variables
        for i in xrange(len(self.DV_listGlobal)):
            self.DV_listGlobal[i](self)
       
        self.refAxis.coef = self.coef.copy()
        self.refAxis._updateCurveCoef()

        for ipt in xrange(self.nPtAttach):
            base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])

            scale = self.scale[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_x = self.scale_x[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_y = self.scale_y[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_z = self.scale_z[self.curveIDs[ipt]](self.links_s[ipt]) 
            if self.rot_type == 0:
                deriv = self.refAxis.curves[
                    self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                deriv /= geo_utils.euclidean_norm(deriv) # Normalize
                new_vec = -numpy.cross(deriv, self.links_n[ipt])
                new_vec = geo_utils.rotVbyW(new_vec, deriv, self.rot_x[
                        self.curveIDs[ipt]](self.links_s[ipt])*numpy.pi/180)
                new_pts[ipt] = base_pt + new_vec*scale
            # end if
            else:
                rotX = geo_utils.rotxM(self.rot_x[
                        self.curveIDs[ipt]](self.links_s[ipt]))
                rotY = geo_utils.rotyM(self.rot_y[
                        self.curveIDs[ipt]](self.links_s[ipt]))
                rotZ = geo_utils.rotzM(self.rot_z[
                        self.curveIDs[ipt]](self.links_s[ipt]))

                D = self.links_x[ipt]
             
                rotM = self._getRotMatrix(rotX, rotY, rotZ)
                D = numpy.dot(rotM, D)
                deriv = self.refAxis.curves[
                    self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                deriv[0] = 0.0

                deriv /= geo_utils.euclidean_norm(deriv) # Normalize

                D = geo_utils.rotVbyW(D,deriv,numpy.pi/180*self.rot_theta[              
                        self.curveIDs[ipt]](self.links_s[ipt]))

                D[0] *= scale_x
                D[1] *= scale_y
                D[2] *= scale_z

                new_pts[ipt] = base_pt + D*scale
            # end if
        # end for

        # set the forward effect of the global design vars in each child
        for iChild in xrange(len(self.children)):

            dXrefdCoef = self.FFD.embeded_volumes['child%d_axis'%(iChild)].dPtdCoef
            dCcdCoef   = self.FFD.embeded_volumes['child%d_coef'%(iChild)].dPtdCoef
            
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
        # end if

        return new_pts

    def _complexifyCoef(self):
        """Convert coef to complex terporarily"""

        for i in xrange(len(self.refAxis.curves)):
            self.rot_x[i].coef = self.rot_x[i].coef.astype('D')
            self.rot_y[i].coef = self.rot_y[i].coef.astype('D')
            self.rot_z[i].coef = self.rot_z[i].coef.astype('D')
            self.rot_theta[i].coef = self.rot_theta[i].coef.astype('D')
            
            self.scale[i].coef = self.scale[i].coef.astype('D')
            self.scale_x[i].coef = self.scale_x[i].coef.astype('D')
            self.scale_y[i].coef = self.scale_y[i].coef.astype('D')
            self.scale_z[i].coef = self.scale_z[i].coef.astype('D')
            self.refAxis.curves[i].coef = \
                self.refAxis.curves[i].coef.astype('D')
        # end for

        self.coef = self.coef.astype('D')

        return
        
    def _unComplexifyCoef(self):
        """Convert coef back to reals"""

        for i in xrange(len(self.refAxis.curves)):
            self.rot_x[i].coef = self.rot_x[i].coef.real.astype('d')
            self.rot_y[i].coef = self.rot_y[i].coef.real.astype('d')
            self.rot_z[i].coef = self.rot_z[i].coef.real.astype('d')
            self.rot_theta[i].coef = self.rot_theta[i].coef.real.astype('d')
            
            self.scale[i].coef = self.scale[i].coef.real.astype('d')
            self.scale_x[i].coef = self.scale_x[i].coef.real.astype('d')
            self.scale_y[i].coef = self.scale_y[i].coef.real.astype('d')
            self.scale_z[i].coef = self.scale_z[i].coef.real.astype('d')
            self.refAxis.curves[i].coef = \
                self.refAxis.curves[i].coef.real.astype('d')
        # end for

        self.coef = self.coef.real.astype('d')

        return

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, child=False, nDVStore=0):
        """This function takes the total derivative of an objective, 
        I, with respect the points controlled on this processor. We
        take the transpose prodducts and mpi_allreduce them to get the
        resulting value on each processor.  Note we DO NOT want to run
        computeTotalJacobian as this forms the dPt/dXdv jacobian which
        is unnecessary and SLOW!
        """

        # This is going to be DENSE in general -- does not depend on
        # name
        if self.J_attach is None:
            self.J_attach = self._attachedPtJacobian()
           
        # This is the sparse jacobian for the local DVs that affect
        # Control points directly.
        if self.J_local is None:
            self.J_local = self._localDVJacobian()
         
        # HStack em'
        # Three different possibilities: 
        if self.J_attach is not None and self.J_local is None:
            J_temp = self.J_attach
        elif self.J_local is not None and self.J_attach is None:
            J_temp = self.J_local
        else:
            J_temp = sparse.hstack([self.J_attach, self.J_local], format='lil')
        # end if

        # Convert J_temp to CSR Matrix
        J_temp = sparse.csr_matrix(J_temp)

        # Transpose of the point-coef jacobian:
        dPtdCoef = self.FFD.embeded_volumes[name].dPtdCoef
        
        # Store or retreive nDV
        if child:
            nDV = nDVStore
        else:
            nDV = self._getNDV()

        if dPtdCoef is not None:
            dIdcoef = numpy.zeros((self.nPtAttachFull*3))
            if dPtdCoef is not None:
                dIdcoef[0::3] = dPtdCoef.T.dot(dIdpt[:, 0])
                dIdcoef[1::3] = dPtdCoef.T.dot(dIdpt[:, 1])
                dIdcoef[2::3] = dPtdCoef.T.dot(dIdpt[:, 2])
            # end if

            # Now back to design variables:
            dIdx_local = J_temp.T.dot(dIdcoef)
        else:
            # This is an array of zeros of length the number of design
            # variables
            dIdx_local = numpy.zeros(nDV, 'd')
        # end if

        if comm: # If we have a comm, globaly reduce with sum
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local
        # end if

        for iChild in xrange(len(self.children)):
             # reset control points on child for child link derivatives
            self.children[iChild].FFD.coef = self.FFD.getAttachedPoints(
                'child%d_coef'%(iChild))
            
            self.children[iChild].coef = self.FFD.getAttachedPoints(
                'child%d_axis'%(iChild))
            self.children[iChild].refAxis.coef =  self.children[iChild].coef.copy()
            self.children[iChild].refAxis._updateCurveCoef()
            dIdx += self.children[iChild].totalSensitivity(dIdpt, comm, name, True, nDV)
        # end for
        
        # self.computeTotalJacobian(name)
        # #print 'shapes',self.JT.shape,dIdpt.shape
        # dIdx = self.JT.dot(dIdpt.reshape(self.JT.shape[1]))

        return dIdx

    def totalSensitivityFD(self, dIdpt, ptSetName, comm=None, nDV_T=None, DVParent=0):
        """This function takes the total derivative of an objective, 
        I, with respect the points controlled on this processor using FD.
        We take the transpose prodducts and mpi_allreduce them to get the
        resulting value on each processor. Note that this function is slow
        and should eventually be replaced by an analytic version.
        """
        if self.isChild:
            refFFDCoef = copy.copy(self.FFD.coef)
        # end if

        coords0 = self.update(name).flatten()

        h = 1e-6
        
        # count up number of DVs
        nDV = self._getNDVGlobal() 
        if nDV_T==None:
            nDV_T = self._getNDV()
        # end
        dIdx = numpy.zeros(nDV_T)
        if self.isChild:
            #nDVSummed = self.dXrefdXdvg.shape[1]
            DVCount=DVParent#nDVSummed-nDV
            DVLocalCount = DVParent+nDV
        else:
            #nDVSummed = nDV
            DVCount=0
            DVLocalCount = nDV
        # end if

        for i in xrange(len(self.DV_listGlobal)):
            print 'GlobalVar',i,DVCount
            for j in xrange(self.DV_listGlobal[i].nVal):
                if self.isChild:
                    self.FFD.coef=  refFFDCoef.copy()
                # end if

                refVal = self.DV_listGlobal[i].value[j]

                self.DV_listGlobal[i].value[j] += h

                coordsph = self.update(name).flatten()

                deriv = (coordsph-coords0)/h
                dIdx[DVCount]=numpy.dot(dIdpt.flatten(),deriv)
                DVCount += 1
                self.DV_listGlobal[i].value[j] = refVal
            # end for
        # end for
        DVparent=DVCount
        
        for i in xrange(len(self.DV_listLocal)):
            print 'LocalVar',i,DVLocalCount
            for j in xrange(self.DV_listLocal[i].nVal):

                refVal = self.DV_listLocal[i].value[j]

                self.DV_listLocal[i].value[j] += h
                coordsph = self.update(name).flatten()

                deriv = (coordsph-coords0)/h
                dIdx[DVLocalCount]=numpy.dot(dIdpt.flatten(),deriv)
                DVLocaLCount += 1
                self.DV_listLocal[i].value[j] = refVal
            # end for
        # end for
        
        # reset coords
        self.update(name)
        
        for iChild in xrange(len(self.children)):
            
            self.children[iChild].FFD.coef = self.FFD.getAttachedPoints(
                'child%d_coef'%(iChild))
            
            self.children[iChild].coef = self.FFD.getAttachedPoints(
                'child%d_axis'%(iChild))
            self.children[iChild].refAxis.coef =  self.children[iChild].coef.copy()
            self.children[iChild].refAxis._updateCurveCoef()

        for child in self.children:
            dIdx+=child.totalSensitivityFD(dIdpt, comm, name,nDV_T,DVParent)

        return dIdx

    def computeTotalJacobianFD(self, ptSetName, comm=None, nDV_T = None,DVParent=0):
        """This function takes the total derivative of an objective, 
        I, with respect the points controlled on this processor using FD.
        We take the transpose prodducts and mpi_allreduce them to get the
        resulting value on each processor. Note that this function is slow
        and should eventually be replaced by an analytic version.
        """
        if self.isChild:
            refFFDCoef = copy.copy(self.FFD.coef)
        # end if

        coords0 = self.update(name).flatten()

        h = 1e-6
        
        # count up number of DVs
        nDV = self._getNDVGlobal() 
        if nDV_T==None:
            nDV_T = self._getNDV()
        # end
        dPtdx = numpy.zeros([coords0.shape[0],nDV_T])
        if self.isChild:
            #nDVSummed = self.dXrefdXdvg.shape[1]
            DVCount=DVParent#nDVSummed-nDV
            DVLocalCount = DVParent+nDV
        else:
            #nDVSummed = nDV
            DVCount=0
            DVLocalCount = nDV
        # end if

        for i in xrange(len(self.DV_listGlobal)):
            for j in xrange(self.DV_listGlobal[i].nVal):
                if self.isChild:
                    self.FFD.coef=  refFFDCoef.copy()
                # end if

                refVal = self.DV_listGlobal[i].value[j]

                self.DV_listGlobal[i].value[j] += h

                coordsph = self.update(name).flatten()

                deriv = (coordsph-coords0)/h
                dPtdx[:,DVCount]=deriv

                DVCount += 1
                self.DV_listGlobal[i].value[j] = refVal
            # end for
        # end for
        DVParent=DVCount
        
        for i in xrange(len(self.DV_listLocal)):
            for j in xrange(self.DV_listLocal[i].nVal):

                refVal = self.DV_listLocal[i].value[j]

                self.DV_listLocal[i].value[j] += h
                coordsph = self.update(name).flatten()

                deriv = (coordsph-coords0)/h
                dPtdx[:,DVLocalCount]=deriv

                DVLocaLCount += 1
                self.DV_listLocal[i].value[j] = refVal
            # end for
        # end for
        
        # reset coords
        self.update(name)
        for iChild in xrange(len(self.children)):
            
            self.children[iChild].FFD.coef = self.FFD.getAttachedPoints(
                'child%d_coef'%(iChild))
            
            self.children[iChild].coef = self.FFD.getAttachedPoints(
                'child%d_axis'%(iChild))
            self.children[iChild].refAxis.coef =  self.children[iChild].coef.copy()
            self.children[iChild].refAxis._updateCurveCoef()
        # end
        for child in self.children:

            dPtdx+=child.computeTotalJacobianFD(comm, name, nDV_T, DVParent)
        # end for

        return dPtdx

    def computeTotalJacobian(self, ptSetName):
        """ Return the total point jacobian in CSR format since we
        need this for TACS"""

        # if self.JT is not None and self.J_name == name: # Already computed
        #     return
        
        # This is going to be DENSE in general -- does not depend on
        # name

        if self.J_attach is None:
            self.J_attach = self._attachedPtJacobian()

        # This is the sparse jacobian for the local DVs that affect
        # Control points directly.
        if self.J_local is None:
            self.J_local = self._localDVJacobian()
         
        # HStack em'
        # Three different possibilities: 
        if self.J_attach is not None and self.J_local is None:
            J_temp = sparse.lil_matrix(self.J_attach)
        elif self.J_local is not None and self.J_attach is None:
            J_temp = sparse.lil_matrix(self.J_local)
        else:
            J_temp = sparse.hstack([self.J_attach, self.J_local], format='lil')
        # end if

        # This is the FINAL Jacobian for the current geometry
        # point. We need this to be a sparse matrix for TACS. 
        
        if self.FFD.embeded_volumes[name].dPtdCoef is not None:
            dPtdCoef = self.FFD.embeded_volumes[name].dPtdCoef.tocoo()
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
                self.children[iChild].refAxis.coef =  self.children[iChild].coef.copy()
                self.children[iChild].refAxis._updateCurveCoef()
                self.children[iChild].computeTotalJacobian(name)

                self.JT = self.JT + self.children[iChild].JT
                
            # end

        else:
            self.JT = None
        return 

    def _attachedPtJacobian(self):
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
            for i in xrange(len(self.DV_listGlobal)):
                nVal = self.DV_listGlobal[i].nVal
                for j in xrange(nVal):
                    iDV+=1
                # end
            # end
            for iChild in xrange(len(self.children)):
                self.children[iChild].startDVg=iDV
                childnDV = self.children[iChild]._getNDVGlobal()
                iDV+=childnDV
            # end
        # end if

        if nDVSummed == 0:
            return None

        h = 1.0e-40j
        oneoverh = 1.0/1e-40

        # h = 1.0e-6
        # oneoverh = 1.0/1e-6
        # coordref = self.update_deriv().flatten()
        # Just do a CS loop over the coef
        # First sum the actual number of globalDVs
        if nDV <> 0:
            Jacobian = numpy.zeros((self.nPtAttachFull*3, nDV))

            # Create the storage arrays for the information that must be
            # passed to the children

            for iChild in xrange(len(self.children)):
                N = self.FFD.embeded_volumes['child%d_axis'%(iChild)].N
                self.children[iChild].dXrefdXdvg = numpy.zeros((N*3, nDV))

                N = self.FFD.embeded_volumes['child%d_coef'%(iChild)].N
                self.children[iChild].dCcdXdvg = numpy.zeros((N*3, nDV))
                self.children[iChild].rangeg = numpy.zeros(nDV)

            iDV = 0
            for i in xrange(len(self.DV_listGlobal)):
                nVal = self.DV_listGlobal[i].nVal
                for j in xrange(nVal):

                    refVal = self.DV_listGlobal[i].value[j]

                    self.DV_listGlobal[i].value[j] += h

                    deriv = oneoverh*numpy.imag(self.update_deriv(iDV,h,oneoverh)).flatten()
                    #deriv = oneoverh*(self.update_deriv().flatten()-coordref)

                    numpy.put(Jacobian[0::3, iDV], self.ptAttachInd, 
                              deriv[0::3])
                    numpy.put(Jacobian[1::3, iDV], self.ptAttachInd, 
                            deriv[1::3])
                    numpy.put(Jacobian[2::3, iDV], self.ptAttachInd, 
                              deriv[2::3])

                    # save the global DV Ranges for the children
                    if self.rangeg==None:
                        for iChild in xrange(len(self.children)):
                            self.children[iChild].rangeg[iDV]=self.DV_listGlobal[i].range[j]
                        # end for
                    else:
                        for iChild in xrange(len(self.children)):
                            self.children[iChild].rangeg[iDV]=self.rangeg[iDV]
                        # end for
                    # end if

                    iDV += 1

                    self.DV_listGlobal[i].value[j] = refVal
                # end for
            # end for
        else:
            Jacobian = None
        # end if

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

                new_pts_child = self.update_deriv(iDV,h,oneoverh)
                tmp2 = numpy.zeros(self.nPtAttachFull*3,dtype='D')
                numpy.put(tmp2[0::3], self.ptAttachInd, new_pts_child[:,0])
                numpy.put(tmp2[1::3], self.ptAttachInd, new_pts_child[:,1])
                numpy.put(tmp2[2::3], self.ptAttachInd, new_pts_child[:,2])

                tmp3 = numpy.zeros(self.nPtAttachFull*3,dtype='d')
                for index in self.ptAttachInd:
                    for j in xrange(3):
                        idx = index*3+j
                        tmp3[idx]=self.dCcdXdvg[idx,iDV]
                    # end
                # end
                Jacobian[:, iDV] += oneoverh*numpy.imag(tmp2)-tmp3
                self.coef = self.coef.astype('d')
                self.FFD.coef = self.FFD.coef.astype('d')

        self._unComplexifyCoef()

        return Jacobian

    def _localDVJacobian(self, scaled=True):
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
            for i in xrange(len(self.DV_listLocal)):
                nVal = self.DV_listLocal[i].nVal
                for j in xrange(nVal):
                    iDV+=1
                # end
            # end
            for iChild in xrange(len(self.children)):
                self.children[iChild].startDVl=iDV
                childnDV = self.children[iChild]._getNDVLocal()
                iDV+=childnDV
            # end
        # end if
        
        if nDVSummed == 0:
            return None
        h = 1.0e-40j
        oneoverh = 1.0/1e-40

        if nDV <> 0:
            Jacobian = sparse.lil_matrix((self.nPtAttachFull*3, nDV))
            # Create the storage arrays for the information that must be
            # passed to the children

            for iChild in xrange(len(self.children)):
                N = self.FFD.embeded_volumes['child%d_axis'%(iChild)].N
                self.children[iChild].dXrefdXdvl = numpy.zeros((N*3, nDV))

                N = self.FFD.embeded_volumes['child%d_coef'%(iChild)].N
                self.children[iChild].dCcdXdvl = numpy.zeros((N*3, nDV))
                self.children[iChild].rangel = numpy.zeros(nDV)
            # end for

            iDVLocal = 0
            for i in xrange(len(self.DV_listLocal)):
                nVal = self.DV_listLocal[i].nVal
                for j in xrange(nVal):

                    pt_dv = self.DV_listLocal[i].coef_list[j] 
                    irow = pt_dv[0]*3 + pt_dv[1]
                    if not scaled:
                        Jacobian[irow, iDVLocal] = 1.0
                    else:
                        Jacobian[irow, iDVLocal] = self.DV_listLocal[i].range[j]
                    # end if

                    for iChild in xrange(len(self.children)):

                        dXrefdCoef = self.FFD.embeded_volumes['child%d_axis'%(iChild)].dPtdCoef
                        dCcdCoef   = self.FFD.embeded_volumes['child%d_coef'%(iChild)].dPtdCoef
            
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
                    # end for
                    if scaled:
                        if self.rangel is None:
                            for iChild in xrange(len(self.children)):
                                self.children[iChild].rangel[iDVLocal] = self.DV_listLocal[i].range[j]
                            # end for
                        else:
                            for iChild in xrange(len(self.children)):
                                self.children[iChild].rangel[iDVLocal] = self.rangel[iDVLocal]
                            # end for
                        # end if
                    # end if
                    iDVLocal += 1
                # end for
            # end for
        else:
            Jacobian = None
        # end if

        if self.dXrefdXdvl is not None:
            #temp = sparse.lil_matrix((self.nPtAttachFull*3, nDVSummed))
            temp = numpy.zeros((self.nPtAttachFull*3, nDVSummed))
            if Jacobian is not None:
                startIdx = self.startDVl#nDVSummed-self.nChildren+self.iChild
                endIdx = startIdx+nDV#nDVSummed-self.nChildren+self.iChild+nDV
                temp[:, startIdx:endIdx] = Jacobian.todense()
                #temp[:, nDVSummed - nDV:] = Jacobian
            # end if 
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

                self.FFD.coef+=tmp1

                new_pts_child = self.update_deriv(iDV, h, oneoverh)
                tmp2 = numpy.zeros(self.nPtAttachFull*3,dtype='D')
                if scaled:
                    # ptAttachInd is of length nPtAttach, but need to
                    # set the x-y-z coordinates here:
                    numpy.put(tmp2[0::3], self.ptAttachInd, new_pts_child[:,0]*self.rangel[iDV])
                    numpy.put(tmp2[1::3], self.ptAttachInd, new_pts_child[:,1]*self.rangel[iDV])
                    numpy.put(tmp2[2::3], self.ptAttachInd, new_pts_child[:,2]*self.rangel[iDV])
                else:
                    numpy.put(tmp2[0::3], self.ptAttachInd, new_pts_child[:,0])
                    numpy.put(tmp2[1::3], self.ptAttachInd, new_pts_child[:,1])
                    numpy.put(tmp2[2::3], self.ptAttachInd, new_pts_child[:,2])
                # end
                tmp3 = numpy.zeros(self.nPtAttachFull*3,dtype='d')
                for index in self.ptAttachInd:
                    for j in xrange(3):
                        idx = index*3+j
                        if scaled:
                            tmp3[idx]=self.dCcdXdvl[idx,iDV]*self.rangel[iDV]
                        else:
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

    def addVariablesPyOpt(self, optProb, globalVars=True, localVars=True,
                          varSet='geo'):
        """
        Add the current set of variables to the optProb object.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which variables are added

        globalVars : bool
            Flag specifying whether gloabl variables are to be added

        localVars : bool
            Flag specifying whether local variables are to be added

        varSet : str
            Name of the pyOpt variable set for geometric variables. This should
            be left as the default unless multiple DVGeometry objects are used
            within the same optimization problem. 
        """

        # Add design variables from the master:
        if globalVars:
            for key in self.DV_listGlobal:
                dv = DV_listGlobal[key]
                opt_prob.addVarGroup(dv.name, dv.nVal, 'c', 
                                     value=dv.value, lower=dv.lower, upper=dv.upper,
                                     scale = dv.scape, varSet=varSet)
        if localVars:
            for key in self.DV_listLocal:
                dv = DV_listLocal[key]
                opt_prob.addVarGroup(dv.name, dv.nVal, 'c', 
                                     value=dv.value, lower=dv.lower, upper=dv.upper,
                                     scale = dv.scape, varSet=varSet)
        
        # Add variables for children
        for child in self.children:
            child.addVariablesPyOpt(opt_prob)

        return opt_prob

    def writeTecplot(self, fileName):
        """Write the (deformed) current state of the FFD's to a tecplot file, 
        including the children

        Parameters
        ----------
        fileName : str
           Filename for tecplot file. Should have a .dat extension
        """

        # Name here doesnt matter, just take the first one
        self.update(self.points.keys()[0], childDelta=False)

        f = pySpline.openTecplot(fileName, 3)
        vol_counter = 0
        # Write master volumes:
        vol_counter += self._writeVols(f, vol_counter)

        # Write children volumes:
        for iChild in xrange(len(self.children)):
            vol_counter += self.children[iChild]._writeVols(f, vol_counter)
        # end for

        pySpline.closeTecplot(f)

        self.update(self.points.keys()[0], childDelta=True) 

        return

    def _writeVols(self, handle, vol_counter):
        for i in xrange(len(self.FFD.vols)):
            pySpline.writeTecplot3D(handle, 'vol%d'%i, self.FFD.vols[i].coef)
            vol_counter += 1
        # end for

        return vol_counter

    def checkDerivatives(self, ptSetName):
        """
        Run a brute force FD check on ALL design variables

        Parameters
        ----------
        ptSetName : str
            name of the point set to check
        """

        print 'Computing Analytic Jacobian...'
        self.JT = None # J is no longer up to date
        self.J_name = None # Name is no longer defined
        self.J_attach = None
        self.J_local = None
        self.computeTotalJacobian(name)

       
        Jac = copy.deepcopy(self.JT)
        
        # Global Variables
        mpiPrint('========================================')
        mpiPrint('             Global Variables           ')
        mpiPrint('========================================')
                 
        if self.isChild:
            refFFDCoef = copy.copy(self.FFD.coef)
        # end if
        coords0 = self.update(name).flatten()

        h = 1e-6
        
        nDV = self._getNDVGlobal() 
        if self.isChild:
            nDVSummed = self.dXrefdXdvg.shape[1]
            DVCount=nDVSummed-nDV
        else:
            nDVSummed = nDV
            DVCount=0
        # end if
#        DVCount = 0
        for i in xrange(len(self.DV_listGlobal)):
            for j in xrange(self.DV_listGlobal[i].nVal):

                mpiPrint('========================================')
                mpiPrint('      GlobalVar(%d), Value(%d)           '%(i, j))
                mpiPrint('========================================')

                if self.isChild:
                    self.FFD.coef=  refFFDCoef.copy()
                # end if

                refVal = self.DV_listGlobal[i].value[j]

                self.DV_listGlobal[i].value[j] += h

                coordsph = self.update(name).flatten()

                deriv = (coordsph-coords0)/h

                for ii in xrange(len(deriv)):

                    relErr = (deriv[ii] - Jac[DVCount, ii])/(
                        1e-16 + Jac[DVCount, ii])
                    absErr = deriv[ii] - Jac[DVCount,ii]

                    if abs(relErr) > h*10 and abs(absErr) > h*10:
                        print ii, deriv[ii], Jac[DVCount, ii], relErr, absErr
                    # end if
                # end for
                DVCount += 1
                self.DV_listGlobal[i].value[j] = refVal
            # end for
        # end for

        for i in xrange(len(self.DV_listLocal)):
            for j in xrange(self.DV_listLocal[i].nVal):

                mpiPrint('========================================')
                mpiPrint('      LocalVar(%d), Value(%d)           '%(i, j))
                mpiPrint('========================================')

                refVal = self.DV_listLocal[i].value[j]

                self.DV_listLocal[i].value[j] += h
                coordsph = self.update(name).flatten()

                deriv = (coordsph-coords0)/h

                for ii in xrange(len(deriv)):
                    relErr = (deriv[ii] - Jac[DVCount, ii])/(
                        1e-16 + Jac[DVCount, ii])
                    absErr = deriv[ii] - Jac[DVCount,ii]

                    if abs(relErr) > h and abs(absErr) > h:
                        print ii, deriv[ii], Jac[DVCount, ii], relErr, absErr
                    # end if
                # end for
                DVCount += 1
                self.DV_listLocal[i].value[j] = refVal
            # end for
        # end for

        for child in self.children:
            child.checkDerivatives(name)
        # end for
        return

    def printDesignVariables(self):
        """
        Print a formatted list of design variables to the screen
        """
        for dg in self.DV_listGlobal:
            mpiPrint('%s'%(dg.name))
            for i in xrange(dg.nVal):
                mpiPrint('%20.15f'%(dg.value[i]))
  
        for dl in self.DV_listLocal:
            mpiPrint('%s'%(dl.name))
            for i in xrange(dl.nVal):
                mpiPrint('%20.15f'%(dl.value[i]))
    
        for child in self.children:
            child.printDesignVariables()
  
class geoDVGlobal(object):
     
    def __init__(self, dv_name, value, lower, upper, function):
        
        """Create a geometric design variable (or design variable group)
        See addGeoDVGloabl in pyGeo for more information
        """

        self.name = dv_name
        self.value = np.atleast_1d(np.array(value)).astype('D')
        self.nVal = len(self.value)

        low = np.atleast_1d(np.array(lower))
        if len(low) == self.nVal:
            self.lower = low
        else:
            self.lower = np.ones(self.nVal)*lower

        high = np.atleast_1d(np.array(upper))
        if len(high) == self.nVal:
            self.upper = high
        else:
            self.upper = np.ones(self.nVal)*upper
    
        self.function = function

    def __call__(self, geo):

        """When the object is called, actually apply the function"""
        # Run the user-supplied function
        d = np.dtype(complex)

        # If the geo object is complex, which is indicated by .coef
        # being complex, run with complex numbers. Otherwise, convert
        # to real before calling. This eliminates casting warnings. 
        if geo.coef.dtype == d or geo.complex:
            return self.function(self.value, geo)
        else:
            return self.function(np.real(self.value), geo)

class geoDVLocal(object):
     
    def __init__(self, dvName, lower, upper, axis, coefList):
        
        """Create a set of gemoetric design variables whcih change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.
        See addGeoDVLOcal for more information
        """
        N = len(axis)

        self.nVal = len(coef_list)*N
        self.value = np.zeros(self.nVal, 'D')
        self.name = dvName
        self.lower = lower*np.ones(self.nVal)
        self.upper = upper*np.ones(self.nVal)
        self.range    = self.upper-self.lower
       
        self.coefList = np.zeros((self.nVal, 2), 'intc')
        j = 0

        for i in xrange(len(coef_list)):
            if 'x' in axis.lower():
                self.coef_list[j] = [coef_list[i], 0]
                j += 1
            elif 'y' in axis.lower():
                self.coef_list[j] = [coef_list[i], 1]
                j += 1
            elif 'z' in axis.lower():
                self.coef_list[j] = [coef_list[i], 2]
                j += 1
   
    def __call__(self, coef):

        """When the object is called, apply the design variable values to 
        coefficients"""
        for i in xrange(self.nVal):
            coef[self.coef_list[i, 0], self.coef_list[i, 1]] += self.value[i].real
        # end for
      
        return coef
