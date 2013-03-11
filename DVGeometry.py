# =============================================================================
# DVGeometry Deals with all the details of taking user supplied design
# variables and mapping it to the descrete surfaces on the CFD and FEA
# disciplines. In the case of the FE discipline, ALL the FE nodes are
# included (distributed in a volume) and as such this currently only
# works with Free Form Deformation Volumes that can deal with these
# spactialy distributed points. 
# =============================================================================

import sys, copy
import numpy
from scipy import sparse
from mdo_import_helper import MPI, mpiPrint, import_modules
exec(import_modules('geo_utils', 'pySpline', 'pyNetwork'))

class DVGeometry(object):
    
    def __init__(self, points, curves, FFD=None, rot_type=0, child=False, 
                 *args, **kwargs):

        ''' Create a DV Geometry module to handle all design variable
        manipulation 

        Required:
        points: The arbitrary set of points that the design variables 
        will act on.
        curves: A list of pySpline objects that can be used to from a 
        network of "reference axis" which can facilitate the 
        the manipulation of the geometry
        Optional: 
        FFD:    A pyBlock instance that is used in Free-form 
        deformation. Points are embedded inside the volume 
        and the ref axis then act on the FFD control points

        rot_type: integer 0-6 to determine the rotation order
        0 -> None   -> Intrinsic rotation, rot_x is roation about axis
        1 -> x-y-z
        2 -> x-z-y
        3 -> y-z-x  
        4 -> y-x-z
        5 -> z-x-y  -> Default aerosurf (x-streamwise y-up z-out wing)
        6 -> z-y-x
        
        axis:     axis along which to project points/control points onto the
        ref axis

        complex:  True/False flag that can be set to allow complex inputs
        on design variables which show up on output. This is 
        useful for COMPLEX STEP DEBUGGING ONLY. It should not be 
        used in general

        '''
        
        self.DV_listGlobal  = [] # Global Design Variable List
        self.DV_namesGlobal = {} # Names of Global Design Variables
        
        self.DV_listLocal   = [] # Local Design Variable List
        self.DV_namesLocal  = {} # Names of Local Design Variables

        # Flags to determine if this DVGeometry is a parent or child
        self.isChild  = child
        self.children = []

        # Points are the discrete points we must manipulate. We have
        # to be careful here, since this MAY be a list
        self.points = []
        if isinstance(points, list):
            assert 'names' in kwargs, 'Names must be specified if more than\
 one set of points are used'
            self.points = points
            self.pt_names = kwargs['names']
        else:
            self.points = [points]
            self.pt_names = ['default']
        # end if

        self.rot_type = rot_type
        self.complex = kwargs.pop('complex', False)
        self.FFD = FFD

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

        # Setup the network of reference axis curves
        self.refAxis = pyNetwork.pyNetwork(curves, *args, **kwargs)
        self.refAxis.doConnectivity()
   
        # Project Points in to volume
        for i in xrange(len(self.points)):
            if self.isChild:
                coef_mask = self.FFD.attachPoints(
                    self.points[i],self.pt_names[i], interiorOnly=True)
            else:
                coef_mask = self.FFD.attachPoints(
                    self.points[i],self.pt_names[i], interiorOnly=False)
            # end if
            self.FFD._calcdPtdCoef(self.pt_names[i])
        # end for


        if 'vol_list' in kwargs:
            # If the user has specified a vol_list, the curves
            # should only act on some of the volumes. 
            vol_list = kwargs['vol_list']
            assert len(curves)==len(vol_list), \
                'The length of vol_list and curves must be the same'
            # The ptAttach list *MAY* be smaller than the full set
            # of coordinates defining the FFD. Also, the user had
            # told us WHICH volume(s) must be connected to WHICH
            # Axis. It we put all these in a list there's a
            # possibility the curve projects will break this
            # association.
        else:
            vol_list = [numpy.arange(self.FFD.nVol) for i in range(len(curves))]
        # end if

        # So...create ptAttachInd which are the indicies of
        # self.FFD.coef that we are actually manipulating. If

        self.ptAttachInd = []
        self.ptAttachPtr = [0]
        for ii in xrange(len(vol_list)):
            temp = []
            for iVol in vol_list[ii]:
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
        self.ptAttach = self.FFD.coef.take(self.ptAttachInd, axis=0)
        self.ptAttachFull = self.FFD.coef.copy()

        # Number of points attached to ref axis
        self.nPtAttach = len(self.ptAttach)
        self.nPtAttachFull = len(self.ptAttachFull)
        
        # New setup splines for the rotations
        self.rot_x = []
        self.rot_y = []
        self.rot_z = []
        self.rot_theta = []
        self.scale = []
        self.scale_x = []
        self.scale_y = []
        self.scale_z = []
        self.coef = self.refAxis.coef # pointer
        self.coef0 = self.coef.copy()
        for i in xrange(len(self.refAxis.curves)):
            t = self.refAxis.curves[i].t
            k = self.refAxis.curves[i].k
            N = len(self.refAxis.curves[i].coef)
            self.rot_x.append(pySpline.curve(
                    t=t, k=k, coef=numpy.zeros((N, 1), 'd')))
            self.rot_y.append(pySpline.curve(
                    t=t, k=k, coef=numpy.zeros((N, 1), 'd')))
            self.rot_z.append(pySpline.curve(
                    t=t, k=k, coef=numpy.zeros((N, 1), 'd')))
            self.rot_theta.append(pySpline.curve(
                    t=t, k=k, coef=numpy.zeros((N, 1), 'd')))

            self.scale.append(pySpline.curve(
                    t=t, k=k, coef=numpy.ones((N, 1), 'd')))
            self.scale_x.append(pySpline.curve(
                    t=t, k=k, coef=numpy.ones((N, 1), 'd')))
            self.scale_y.append(pySpline.curve(
                    t=t, k=k, coef=numpy.ones((N, 1), 'd')))
            self.scale_z.append(pySpline.curve(
                    t=t, k=k, coef=numpy.ones((N, 1), 'd')))
        # end for
        
        self.scale0 = copy.deepcopy(self.scale)
        self.scale_x0 = copy.deepcopy(self.scale_x)
        self.scale_y0 = copy.deepcopy(self.scale_y)
        self.scale_z0 = copy.deepcopy(self.scale_z)        
        self.rot_x0 = copy.deepcopy(self.rot_x) 
        self.rot_y0 = copy.deepcopy(self.rot_y)
        self.rot_z0 = copy.deepcopy(self.rot_z)
        self.rot_theta0 = copy.deepcopy(self.rot_theta)   
        # Next we will do the point/curve ray/projections. Note we
        # have to take into account the user's desired volume(s)/direction(s)

        if 'axis' in kwargs:
            if kwargs['axis'] == 'x':
                axis = [1, 0, 0]
            elif kwargs['axis'] == 'y':
                axis = [0, 1, 0]
            elif kwargs['axis'] == 'z':
                axis = [0, 0, 1]
            else:
                axis = kwargs['axis']
            # end if
            axis = numpy.array(axis)
            axis = numpy.array([1, 0, 0])
        else:
            axis = None
        # end if

        curveIDs = []
        s = []
     
        for ii in xrange(len(self.ptAttachPtr)-1):
            pts_to_use = self.ptAttach[
                self.ptAttachPtr[ii]:self.ptAttachPtr[ii+1], :]
            pts_to_use = self.ptAttach
            if axis is not None:
                ids, s0 = self.refAxis.projectRays(
                    pts_to_use, axis)#, curves=[ii])
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
        # end for
        self.links_x = numpy.array(self.links_x)
        self.links_s = numpy.array(self.links_s)
        return
    
    def addChild(self, childDVGeo):
        '''Embed a child FFD into this object'''

        # Make sure the DVGeo being added is flaged as a child:
        if childDVGeo.isChild is False:
            print '='*80
            print 'Error: Trying to add a child FFD that has NOT been'
            print 'created as a child. This operation is illegal.'
            print '='*80
            return
        # end if

        # Etract the coef from the child FFD and ref axis and embed
        # them into the parent and compute their derivatives

        iChild = len(self.children)
        self.FFD.attachPoints(childDVGeo.FFD.coef, 'child%d_coef'%(iChild))
        self.FFD._calcdPtdCoef('child%d_coef'%(iChild))

        self.FFD.attachPoints(childDVGeo.coef, 'child%d_axis'%(iChild))
        self.FFD._calcdPtdCoef('child%d_axis'%(iChild))

        # Add the child to the parent and return
        self.children.append(childDVGeo)

        return

    def _setInitialValues(self):

        self.coef = copy.deepcopy(self.coef0).astype('D')
        self.scale = copy.deepcopy(self.scale0)
        self.scale_x = copy.deepcopy(self.scale_x0)
        self.scale_y = copy.deepcopy(self.scale_y0)
        self.scale_z = copy.deepcopy(self.scale_z0)      
        self.rot_x = copy.deepcopy(self.rot_x0) 
        self.rot_y = copy.deepcopy(self.rot_y0)
        self.rot_z = copy.deepcopy(self.rot_z0)
        self.rot_theta = copy.deepcopy(self.rot_theta0)   
        
        return

    def addGeoDVGlobal(self, dv_name, value, lower, upper, function, 
                       useit=True):
        '''Add a global design variable
        Required:
        dv_name: a unquie name for this design variable (group)
        lower: the lower bound for this design variable
        upper: The upper bound for this design variable
        function: the python function for this design variable
        Optional:
        use_it: Boolean flag as to whether to ignore this design variable
        Returns:
        None
        '''
        if not self.DV_listLocal == []:
            mpiPrint('Error: All Global Variables must be set BEFORE\
 setting local Variables')
            sys.exit(1)
        # end if
        
        self.DV_listGlobal.append(geo_utils.geoDVGlobal(\
                dv_name, value, lower, upper, function, useit))
        self.DV_namesGlobal[dv_name] = len(self.DV_listGlobal)-1

        return

    def addGeoDVLocal(self, dv_name, lower, upper, axis='y', pointSelect=None):

        '''Add a local design variable
        Required:
        dv_name: a unique name for this design variable (group)
        lower: the lower bound for this design variable
        upper: The upper bound for this design variable
        Optional:
        axis: The epecified axis direction to move this dof. It can be 
              'x', 'y', 'z' or 'all'. 
        useitg: Boolean flag as to whether to ignore this design variable
        Returns: 
        N: The number of design variables added for this local Set
        '''

        if pointSelect is not None:
            pts, ind = pointSelect.getPoints(self.FFD.coef)
        else:
            ind = numpy.arange(self.nPtAttach)
        # end if
        self.DV_listLocal.append(
            geo_utils.geoDVLocal(dv_name, lower, upper, axis, ind))
            
        self.DV_namesLocal[dv_name] = len(self.DV_listLocal)-1

        return self.DV_listLocal[-1].nVal

    def setValues(self, dvName, value=None, scaled=True):
        ''' This is the generic set values function. It can set values
        in a number of different ways:

        Type One:

        dvName is a STRING and value is the number of values
        associated with this DV

        Type Two: 
        dvName is a DICTIONARY and the argument of each
        dictionary entry is the value to set.

        '''

        # To make the setting generic below, we will simply "up cast"
        # the single DVname as a string, and dvName in a list to a dictionary.

        if type(dvName) == str:
            dv_dict = {dvName:value}
        elif type(dvName) == dict:
            dv_dict = dvName
        else:
            mpiPrint('Error setting values. dvName must be one of\
 string or dict')
            return
        # end 

        for key in dv_dict:
            if key in self.DV_namesGlobal:
                vals_to_set = numpy.atleast_1d(dv_dict[key]).astype('D')
                assert len(vals_to_set) == self.DV_listGlobal[
                    self.DV_namesGlobal[key]].nVal, \
                    'Incorrect number of design variables for DV: %s'%(key)
                if scaled:
                    vals_to_set = vals_to_set * \
                        self.DV_listGlobal[self.DV_namesGlobal[key]].range +\
                        self.DV_listGlobal[self.DV_namesGlobal[key]].lower
                # end if

                self.DV_listGlobal[self.DV_namesGlobal[key]].value = vals_to_set
            # end if
            if key in self.DV_namesLocal:
                vals_to_set = numpy.atleast_1d(dv_dict[key])
                assert len(vals_to_set) == self.DV_listLocal[
                    self.DV_namesLocal[key]].nVal, \
                    'Incorrect number of design variables for DV: %s'%(key)
                if scaled:
                    vals_to_set = vals_to_set * \
                        self.DV_listLocal[self.DV_namesLocal[key]].range +\
                        self.DV_listLocal[self.DV_namesLocal[key]].lower
                # end if

                self.DV_listLocal[self.DV_namesLocal[key]].value = vals_to_set
            # end if
            self.JT = None # J is no longer up to date
            self.J_name = None # Name is no longer defined
            self.J_attach = None
            self.J_local = None
        # end for

        # Now call setValues on the children. This way the
        # variables will be set on the children
        for child in self.children:
            child.setValues(dvName, value, scaled)
        # end for 

        return

    def getValues(self,scaled=True):
        ''' 
        This is the generic get values function. It returns the current values
        of the DVgeometry design variables
        '''
        
        # initialize a dictionary for the DVs
        DVDict = {}

        # loop over the globalDVs
        for key in self.DV_namesGlobal:
            dv_val =self.DV_listGlobal[self.DV_namesGlobal[key]].value
            if scaled:
                dv_val = (dv_val-self.DV_listGlobal[self.DV_namesGlobal[key]].lower)/self.DV_listGlobal[self.DV_namesGlobal[key]].range
            # end
            DVDict[key] = dv_val
        # end
        
        # and now the local DVs
        for key in self.DV_namesLocal:
            dv_val =self.DV_listLocal[self.DV_namesLocal[key]].value
            if scaled:
                dv_val = (dv_val-self.DV_listLocal[self.DV_namesLocal[key]].lower)/self.DV_listLocal[self.DV_namesLocal[key]].range
            # end if
            DVDict[key] = dv_val
        # end for

        # Now call getValues on the children. This way the
        # returned dictionary will include the variables from
        # the children
        for child in self.children:
            DVDict.update(child.getValues(scaled))
        # end for 

        return DVDict

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
        '''Return the actual number of design variables, global +
        local
        '''
        return self._getNDVGlobal() + self._getNDVLocal()

    def _getNDVGlobal(self):
        nDV = 0
        for i in xrange(len(self.DV_listGlobal)):
            nDV += self.DV_listGlobal[i].nVal
        # end for
        
        for child in self.children:
            nDV += child._getNDVGlobal()

        return nDV

    def _getNDVLocal(self):

        nDV = 0
        for i in xrange(len(self.DV_listLocal)):
            nDV += self.DV_listLocal[i].nVal
        # end for

        for child in self.children:
            nDV += child._getNDVLocal()

        return nDV

    def _getNDVSelf(self):
        return self._getNDVGlobalSelf() + self._getNDVLocalSelf()

    def _getNDVGlobalSelf(self):
        nDV = 0
        for i in xrange(len(self.DV_listGlobal)):
            nDV += self.DV_listGlobal[i].nVal
        # end for
        
        return nDV

    def _getNDVLocalSelf(self):

        nDV = 0
        for i in xrange(len(self.DV_listLocal)):
            nDV += self.DV_listLocal[i].nVal
        # end for

        return nDV
        
    
    def extractCoef(self, axisID):
        ''' Extract the coefficients for the selected reference
        axis. This should be used inside design variable functions'''

        C = numpy.zeros((len(self.refAxis.topo.l_index[axisID]),3),self.coef.dtype)
 
        C[:,0] = numpy.take(self.coef[:,0],self.refAxis.topo.l_index[axisID])
        C[:,1] = numpy.take(self.coef[:,1],self.refAxis.topo.l_index[axisID])
        C[:,2] = numpy.take(self.coef[:,2],self.refAxis.topo.l_index[axisID])

        return C

    def restoreCoef(self, coef, axisID):
        ''' Restore the coefficients for the selected reference
        axis. This should be used inside design variable functions'''

        # Reset
        numpy.put(self.coef[:,0],self.refAxis.topo.l_index[axisID],coef[:,0])
        numpy.put(self.coef[:,1],self.refAxis.topo.l_index[axisID],coef[:,1])
        numpy.put(self.coef[:,2],self.refAxis.topo.l_index[axisID],coef[:,2])

        return 

    def update(self, name="default", childDelta=True):

        '''This is pretty straight forward, perform the operations on
        the ref axis according to the design variables, then return
        the list of points provided. It is up to the user to know what
        to do with the points
        '''

        # Set all coef Values back to initial values
        if not self.isChild:
            self._setInitialValues()
        
        # Step 1: Call all the design variables
        if self.complex:
            self._complexifyCoef()
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
        # end for

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
        # end for

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
            
        # end for

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

    def update_deriv(self,iDV=0):

        '''Copy of update function for derivative calc'''
        
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
        # end for
       
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

        # deal with this!!!!!
        h = 1.0e-40j
        oneoverh = 1.0/1e-40
        #just a Hack!!!!
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
#            numpy.put(tmp[:, 0], self.ptAttachInd, new_pts[:,0])
#            numpy.put(tmp[:, 1], self.ptAttachInd, new_pts[:,1])
#            numpy.put(tmp[:, 2], self.ptAttachInd, new_pts[:,2])
 
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
            # complex part onto child ref axis 
            # self.children[iChild].coef = self.children[iChild].coef.astype('D')
            # self.children[iChild].coef[:,0] +=  dXrefdXdvg[0::3]*h
            # self.children[iChild].coef[:,1] +=  dXrefdXdvg[1::3]*h
            # self.children[iChild].coef[:,2] +=  dXrefdXdvg[2::3]*h
            #self.children[iChild].refAxis.coef = self.children[iChild].coef.copy()
            #self.children[iChild].refAxis._updateCurveCoef()

           
        # end if

        return new_pts

    def _complexifyCoef(self):
        '''Convert coef to complex terporarily'''

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
        '''Convert coef back to reals'''
        for i in xrange(len(self.refAxis.curves)):
            self.rot_x[i].coef = self.rot_x[i].coef.astype('d')
            self.rot_y[i].coef = self.rot_y[i].coef.astype('d')
            self.rot_z[i].coef = self.rot_z[i].coef.astype('d')
            self.rot_theta[i].coef = self.rot_theta[i].coef.astype('d')
            
            self.scale[i].coef = self.scale[i].coef.astype('d')
            self.scale_x[i].coef = self.scale_x[i].coef.astype('d')
            self.scale_y[i].coef = self.scale_y[i].coef.astype('d')
            self.scale_z[i].coef = self.scale_z[i].coef.astype('d')
            self.refAxis.curves[i].coef = \
                self.refAxis.curves[i].coef.astype('d')
        # end for

        self.coef = self.coef.astype('d')

        return

    def totalSensitivity(self, dIdpt, comm=None, scaled=True, name='default'):
        '''This function takes the total derivative of an objective, 
        I, with respect the points controlled on this processor. We
        take the transpose prodducts and mpi_allreduce them to get the
        resulting value on each processor.  Note we DO NOT want to run
        computeTotalJacobian as this forms the dPt/dXdv jacobian which
        is unnecessary and SLOW!
        '''

        # This is going to be DENSE in general -- does not depend on
        # name
        if self.J_attach is None:
            self.J_attach = self._attachedPtJacobian(scaled=scaled)
           
        # This is the sparse jacobian for the local DVs that affect
        # Control points directly.
        if self.J_local is None:
            self.J_local = self._localDVJacobian(scaled=scaled)
         
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
            dIdx_local = numpy.zeros(self._getNDV(), 'd')
        # end if

        if comm: # If we have a comm, globaly reduce with sum
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local
        # end if

        for iChild in xrange(len(self.children)):
            dIdx += self.children[iChild].totalSensitivity(dIdpt, comm, scaled, name)
        # end for

        return dIdx

    def computeTotalJacobian(self, name='default', scaled=True):
        ''' Return the total point jacobian in CSR format since we
        need this for TACS'''

        # if self.JT is not None and self.J_name == name: # Already computed
        #     return
        
        # This is going to be DENSE in general -- does not depend on
        # name

        if self.J_attach is None:
            self.J_attach = self._attachedPtJacobian(scaled=scaled)

        # This is the sparse jacobian for the local DVs that affect
        # Control points directly.
        if self.J_local is None:
            self.J_local = self._localDVJacobian(scaled=scaled)
         
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

            for iChild in xrange(len(self.children)):
                
                # reset control points on child for child link derivatives
                self.children[iChild].FFD.coef = self.FFD.getAttachedPoints(
                    'child%d_coef'%(iChild))

                self.children[iChild].coef = self.FFD.getAttachedPoints(
                    'child%d_axis'%(iChild))
                self.children[iChild].refAxis.coef =  self.children[iChild].coef.copy()
                self.children[iChild].refAxis._updateCurveCoef()
                self.children[iChild].computeTotalJacobian(name, scaled)

                self.JT = self.JT+self.children[iChild].JT
                
            # end

        else:
            self.JT = None
        return 

    def _attachedPtJacobian(self, scaled=True):
        '''
        Compute the derivative of the the attached points
        '''

        nDV = self._getNDVGlobal() 
        if nDV == 0:
            return None
      
        if self.dXrefdXdvg is not None:
            nDVSummed = self.dXrefdXdvg.shape[1]
        else:
            nDVSummed = nDV
        # end if

        h = 1.0e-40j
        oneoverh = 1.0/1e-40

        # h = 1.0e-6
        # oneoverh = 1.0/1e-6
        # coordref = self.update_deriv().flatten()
        # Just do a CS loop over the coef
        # First sum the actual number of globalDVs

        Jacobian = numpy.zeros((self.nPtAttachFull*3, nDV))

        # Create the storage arrays for the information that must be
        # passed to the children

        for iChild in xrange(len(self.children)):
            N = self.FFD.embeded_volumes['child%d_axis'%(iChild)].N
            self.children[iChild].dXrefdXdvg = numpy.zeros((N*3, nDV))

            N = self.FFD.embeded_volumes['child%d_coef'%(iChild)].N
            self.children[iChild].dCcdXdvg = numpy.zeros((N*3, nDV))

        iDV = 0
        for i in xrange(len(self.DV_listGlobal)):

            nVal = self.DV_listGlobal[i].nVal
            for j in xrange(nVal):

                refVal = self.DV_listGlobal[i].value[j]

                self.DV_listGlobal[i].value[j] += h
                
                deriv = oneoverh*numpy.imag(self.update_deriv(iDV)).flatten()
                #deriv = oneoverh*(self.update_deriv().flatten()-coordref)
                
                if scaled:
                    # ptAttachInd is of length nPtAttach, but need to
                    # set the x-y-z coordinates here:
                    numpy.put(Jacobian[0::3, iDV], self.ptAttachInd, 
                              deriv[0::3]*self.DV_listGlobal[i].range[j])
                    numpy.put(Jacobian[1::3, iDV], self.ptAttachInd, 
                              deriv[1::3]*self.DV_listGlobal[i].range[j])
                    numpy.put(Jacobian[2::3, iDV], self.ptAttachInd, 
                              deriv[2::3]*self.DV_listGlobal[i].range[j])
                else:
                    numpy.put(Jacobian[0::3, iDV], self.ptAttachInd, 
                              deriv[0::3])
                    numpy.put(Jacobian[1::3, iDV], self.ptAttachInd, 
                              deriv[1::3])
                    numpy.put(Jacobian[2::3, iDV], self.ptAttachInd, 
                              deriv[2::3])
                # end if

                iDV += 1

                self.DV_listGlobal[i].value[j] = refVal
            # end for
        # end for

        if self.dXrefdXdvg is not None:
            temp = numpy.zeros((self.nPtAttachFull*3, nDVSummed))
            temp[:, nDVSummed - nDV:] = Jacobian

            Jacobian = temp

            for i in xrange(self.dXrefdXdvg.shape[1]):
                

                self.coef = self.coef.astype('D')
                self.coef[:,0] +=  self.dXrefdXdvg[0::3, i]*h
                self.coef[:,1] +=  self.dXrefdXdvg[1::3, i]*h
                self.coef[:,2] +=  self.dXrefdXdvg[2::3, i]*h
                self.refAxis.coef = self.coef.copy()
                self.refAxis._updateCurveCoef()
                
                self.FFD.coef = self.FFD.coef.astype('D')
                numpy.put(self.FFD.coef[:,0], self.ptAttachInd,
                          self.dCcdXdvg[0::3, i])
                numpy.put(self.FFD.coef[:,1], self.ptAttachInd,
                          self.dCcdXdvg[1::3, i])
                numpy.put(self.FFD.coef[:,2], self.ptAttachInd,
                          self.dCcdXdvg[2::3, i])

                new_pts_child = self.update_deriv()
                            
                #tmp2 = numpy.zeros(self.FFD.coef.shape,dtype='D')
                # numpy.put(tmp2[:, 0], self.ptAttachInd, new_pts_child[:,0])
                # numpy.put(tmp2[:, 1], self.ptAttachInd, new_pts_child[:,1])
                # numpy.put(tmp2[:, 2], self.ptAttachInd, new_pts_child[:,2])
                tmp2 = numpy.zeros(self.nPtAttachFull*3,dtype='D')
                numpy.put(tmp2[0::3], self.ptAttachInd, new_pts_child[:,0])
                numpy.put(tmp2[1::3], self.ptAttachInd, new_pts_child[:,1])
                numpy.put(tmp2[2::3], self.ptAttachInd, new_pts_child[:,2])

                
                Jacobian[:, i] += oneoverh*numpy.imag(tmp2)
                self.coef = self.coef.astype('d')
                self.FFD.coef = self.FFD.coef.astype('d')

        self._unComplexifyCoef()

        return Jacobian

    def _localDVJacobian(self, scaled=True):
        '''
        Return the derivative of the coefficients wrt the local design 
        variables
        '''
        
        # This is relatively straight forward, since the matrix is
        # entirely one's or zeros

        nDV = self._getNDVLocal()
        if nDV == 0:
            return None
      
        if self.dXrefdXdvl is not None:
            nDVSummed = self.dXrefdXdvl.shape[1]
        else:
            nDVSummed = nDV
        # end if
        
        if nDV == 0:
            return None

        Jacobian = sparse.lil_matrix((self.nPtAttachFull*3, nDV))
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
                iDVLocal += 1
            # end for
        # end for

        if self.dXrefdXdvl is not None:
            temp = numpy.zeros((self.nPtAttachFull*3, nDVSummed))
            temp[:, nDVSummed - nDV:] = Jacobian
            Jacobian = temp

            for i in xrange(self.dXrefdXdvl.shape[1]):
                
                self.coef = self.children[iChild].coef.astype('D')
                self.coef +=  numpy.imag(self.dXrefdXdvl[:, iDV])

                new_pts_child = self.update_deriv()
            
                tmp2 = numpy.zeros(self.FFD.coef.shape,dtype='D')
                numpy.put(tmp2[:, 0], self.ptAttachInd, new_pts_child[:,0])
                numpy.put(tmp2[:, 1], self.ptAttachInd, new_pts_child[:,1])
                numpy.put(tmp2[:, 2], self.ptAttachInd, new_pts_child[:,2])
            
                Jacobian[:, i] = numpy.imag(tmp2 - self.dCoefdXdvl)
            # end for
        # end if

        return Jacobian

    def addVariablesPyOpt(self, opt_prob):
        '''
        Add the current set of global and local design variables to the opt_prob specified
        '''

        # We are going to do our own scaling here...since pyOpt can't
        # do it...

        # Add design variables from the master:
        for dvList in [self.DV_listGlobal, self.DV_listLocal]:
            for dv in dvList:
                if dv.nVal > 1:
                    low = numpy.zeros(dv.nVal)
                    high = numpy.ones(dv.nVal)
                    val = (numpy.real(dv.value)-dv.lower)/(dv.upper-dv.lower)
                    opt_prob.addVarGroup(dv.name, dv.nVal, 'c', 
                                         value=val, lower=low, upper=high)
                else:
                    low = 0.0
                    high = 1.0
                    val = (numpy.real(dv.value)-dv.lower)/(dv.upper-dv.lower)

                    opt_prob.addVar(dv.name, 'c', value=val, 
                                    lower=low, upper=high)
                # end if
            # end for
        # end for

        # Add variables for children
        for child in self.children:
            child.addVariablesPyOpt(opt_prob)
        # end for

        return opt_prob

    def writeTecplot(self, file_name):
        '''Write the (deformed) current state of the FFD's to a file
        including the children'''

        # Name here doesn't matter, just take the first one
        self.update(self.pt_names[0], childDelta=False)

        f = pySpline.openTecplot(file_name, 3)
        vol_counter = 0
        # Write master volumes:
        vol_counter += self._writeVols(f, vol_counter)

        # Write children volumes:
        for iChild in xrange(len(self.children)):
            vol_counter += self.children[iChild]._writeVols(f, vol_counter)
        # end for

        pySpline.closeTecplot(f)

        self.update(self.pt_names[0], childDelta=True) 

        return

    def _writeVols(self, handle, vol_counter):
        for i in xrange(len(self.FFD.vols)):
            pySpline.writeTecplot3D(handle, 'vol%d'%i, self.FFD.vols[i].coef)
            vol_counter += 1
        # end for

        return vol_counter

    def checkDerivatives(self, name='default'):
        '''Run a brute force FD check on ALL design variables'''
        print 'Computing Analytic Jacobian...'
        self.JT = None # J is no longer up to date
        self.J_name = None # Name is no longer defined
        self.J_attach = None
        self.J_local = None
        self.computeTotalJacobian(name, scaled=False)
       
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
                    self.FFD.coef=  refFFDCoef
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
        
        for dg in self.DV_listGlobal:
            mpiPrint('%s'%(dg.name))
            for i in xrange(dg.nVal):
                mpiPrint('%20.15f'%(dg.value[i]))
            # end for
        # end for

        for dl in self.DV_listLocal:
            mpiPrint('%s'%(dl.name))
            for i in xrange(dl.nVal):
                mpiPrint('%20.15f'%(dl.value[i]))
            # end for
        # end for
    
        for child in self.children:
            child.printDesignVariables()
        # end for

        return
