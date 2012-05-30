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
    
    def __init__(self, points, curves, FFD=None, Surface=None, 
                 rot_type=0, *args, **kwargs):

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
        Surface: A pyGeo instance for B-spline geometry. Points are 
        projected onto the surface  and the ref axis then
        operates on the control points of the surface. Since
        this method CAN ONLY DO SURFACES it CANNOT be used 
        for multidisciplinary optimization

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

        # Points are the descrete points we must manipulate. We have
        # to be careful here, since this MAY be a list
        self.points = []
        self.pt_ind = {}
        if isinstance(points, list):
            assert 'names' in kwargs, 'Names must be specified if more than\
 one set of points are used'
            for i in xrange(len(points)):
                self.points.append(points[i])
                self.pt_ind[kwargs['names'][i]] = i
        else:
            self.points = [points]
            self.pt_ind['default'] = 0
        # end if

        self.rot_type = rot_type

        # Jacobians:
        # self.JT: Total transpose jacobian for self.J_name
        self.JT = None
        self.J_name = None
        self.J_attach = None
        self.J_local  = None

        self.complex = kwargs.pop('complex', False)

        if Surface and FFD:
            print 'DVGeometry can only use FFD\'s or surfaces'
            sys.exit(1)

        self.Surface = None
        self.FFD     = None
      
        if FFD:
            self.FFD = FFD
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
                
                # So...create ptAttachInd which are the indicies of
                # self.FFD.coef that we are actually manipulating. If
                # there's no vol_list, then this is just [0, 1, 2, ...N-1]
                self.ptAttachInd = []
                self.ptAttachPtr = [0]

                for ii in xrange(len(kwargs['vol_list'])):
                    temp = []
                    for iVol in kwargs['vol_list'][ii]:
                        for i in xrange(self.FFD.vols[iVol].Nctlu):
                            for j in xrange(self.FFD.vols[iVol].Nctlv):
                                for k in xrange(self.FFD.vols[iVol].Nctlw):
                                    temp.append(
                                        self.FFD.topo.l_index[iVol][i, j, k])


                                # end for
                            # end for
                        # end for
                    # end for
                    # Uniuque the values we just added:
                    temp = geo_utils.unique(temp)
                    self.ptAttachInd.extend(temp)
                    self.ptAttachPtr.append(len(self.ptAttachInd))
                # end for

                # Convert the ind list to an array
                self.ptAttachInd = numpy.array(self.ptAttachInd).flatten()
            else:
                self.ptAttachInd = numpy.arange(len(self.FFD.coef))
                self.ptAttachPtr = [0, len(self.FFD.coef)]
            # end if
     
            # Take the subset of the FFD cofficients as what will be
            # attached
            self.ptAttach = self.FFD.coef.take(self.ptAttachInd, axis=0)
            self.ptAttachFull = self.FFD.coef.copy()

            # Project Points in to volume:
            for i in xrange(len(self.points)):
                self.FFD.attachPoints(self.points[i])
                self.FFD._calcdPtdCoef(i)
            # end for

        elif Surface:
            print 'Not Done yet'
            sys.exit(0)
#             self.Surface = Surface
#             self.ptAttach = self.Surface.coef()
#             self.ptAttachFull = self.surface.coef.copy()
#             self.Surface.attachSurface(self.points)
#             self.Surface._calcdPtdCoef(0)
        else:
            self.ptAttach = points
        # end if

        # Number of points attached to ref axis
        self.nPtAttach = len(self.ptAttach)
        self.nPtAttachFull = len(self.ptAttachFull)

        self.refAxis = pyNetwork.pyNetwork(curves, *args, **kwargs)
        self.refAxis.doConnectivity()
       
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
        self.scale_x0 = copy.deepcopy(self.scale)
        self.scale_y0 = copy.deepcopy(self.scale)
        self.scale_z0 = copy.deepcopy(self.scale)        

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
            deriv /= numpy.linalg.norm(deriv) # Normalize
            self.links_n.append(numpy.cross(deriv, self.links_x[-1]))
        # end for

        return
    
    def _setInitialValues(self):

        self.coef = copy.deepcopy(self.coef0).astype('D')
        self.scale = copy.deepcopy(self.scale0)
        self.scale_x = copy.deepcopy(self.scale_x0)
        self.scale_y = copy.deepcopy(self.scale_y0)
        self.scale_z = copy.deepcopy(self.scale_z0)      
        

    def addGeoDVGlobal(self, dv_name, value, lower, upper, function, useit=True):
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

        # Take the FFD or surface coef. 

        if self.FFD:
            if pointSelect is not None:
                pts, ind = pointSelect.getPoints(self.FFD.coef)
            else:
                ind = numpy.arange(self.nPtAttach)
            # end if
            self.DV_listLocal.append(
                geo_utils.geoDVLocal(dv_name, lower, upper, axis, ind))
            
        if self.Surface:
            self.DV_listLocal.append(geo_utils.geoDVLocal(\
                    dv_name, lower, upper, axis, numpy.arange(
                        len(self.Surface.coef))))
            
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
        #endfor


        return

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
        
        return nDV

    def _getNDVLocal(self):

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

    def update(self, name="default"):

        '''This is pretty straight forward, perform the operations on
        the ref axis according to the design variables, then return
        the list of points provided. It is up to the user to know what
        to do with the points
        '''
        
        # Set all coef Values back to initial values
        self._setInitialValues()
        
        # Step 1: Call all the design variables

        if self.complex:
            self._complexifyCoef()
            new_pts = numpy.zeros((self.nPtAttach, 3), 'D')
        else:
            new_pts = numpy.zeros((self.nPtAttach, 3), 'd')
        # end if

        # Run Global Design Vars
        for i in xrange(len(self.DV_listGlobal)):
            self.DV_listGlobal[i](self)
        # end for

        self.refAxis.coef = self.coef
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
                deriv /= numpy.linalg.norm(deriv) # Normalize
                new_vec = -numpy.cross(deriv, self.links_n[ipt])
                new_vec = rotVbyW(new_vec, deriv, self.rot_x[
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
                deriv /= numpy.linalg.norm(deriv) # Normalize
                D = rotVbyW(D,deriv,numpy.pi/180*self.rot_theta[              
                        self.curveIDs[ipt]](self.links_s[ipt]))
                
                D[0] *= scale_x
                D[1] *= scale_y
                D[2] *= scale_z
                new_pts[ipt] = base_pt + D*scale

            # end if
        # end for

        if self.FFD:
            temp = numpy.real(new_pts)
            self.FFD.coef = self.ptAttachFull.copy()
            numpy.put(self.FFD.coef[:, 0], self.ptAttachInd, temp[:, 0])
            numpy.put(self.FFD.coef[:, 1], self.ptAttachInd, temp[:, 1])
            numpy.put(self.FFD.coef[:, 2], self.ptAttachInd, temp[:, 2])

            for i in xrange(len(self.DV_listLocal)):
                self.DV_listLocal[i](self.FFD.coef)
            # end for

            self.FFD._updateVolumeCoef()
            coords = self.FFD.getAttachedPoints(self.pt_ind[name])


        elif self.Surface:
            pass # Not implemented
            #self.Surface.coef = numpy.real(new_pts)
            #self.Surface._updateSurfaceCoef()
            #coords = self.Surface.getSurfacePoints(0)
        else:
            coords = new_pts
        # end if

        if self.complex:

            tempCoef = self.ptAttachFull.copy().astype('D')
            numpy.put(tempCoef[:, 0], self.ptAttachInd, new_pts[:, 0])
            numpy.put(tempCoef[:, 1], self.ptAttachInd, new_pts[:, 1])
            numpy.put(tempCoef[:, 2], self.ptAttachInd, new_pts[:, 2])
         
            coords = coords.astype('D')
            imag_part     = numpy.imag(tempCoef)
            imag_j = 1j

            if self.FFD:
                dPtdCoef = self.FFD.embeded_volumes[self.pt_ind[name]].dPtdCoef
                for ii in xrange(3):
                    coords[:, ii] += imag_j*dPtdCoef.dot(imag_part[:, ii])


            elif self.Surface:
                coords += imag_j*self.Surface.attached_surfaces[0].\
                    dPtdCoef.dot(imag_part)
             # end if   
            self._unComplexifyCoef()

        # end if
                        
        return coords

    def update_deriv(self):

        '''Copy of update function for derivative calc'''
        
        # Step 1: Call all the design variables

       
        new_pts = numpy.zeros((self.nPtAttach, 3), 'D')

        # Set all coef Values back to initial values
        self._setInitialValues()
        self._complexifyCoef()

        # Step 1: Call all the design variables
        for i in xrange(len(self.DV_listGlobal)):
            self.DV_listGlobal[i](self)
        # end for
       
        self.refAxis.coef = self.coef
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
                deriv /= numpy.linalg.norm(deriv) # Normalize
                new_vec = -numpy.cross(deriv, self.links_n[ipt])
                new_vec = rotVbyW(new_vec, deriv, self.rot_x[
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
                deriv /= numpy.linalg.norm(deriv) # Normalize
                D = rotVbyW(D,deriv,numpy.pi/180*self.rot_theta[              
                        self.curveIDs[ipt]](self.links_s[ipt]))

                D[0] *= scale_x
                D[1] *= scale_y
                D[2] *= scale_z
                new_pts[ipt] = base_pt + D*scale
            # end if
        # end for

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
        # J_attach and no J_local
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
        dPtdCoef = self.FFD.embeded_volumes[self.pt_ind[name]].dPtdCoef

        dIdcoef = numpy.zeros((self.nPtAttachFull*3))
        dIdcoef[0::3] = dPtdCoef.T.dot(dIdpt[:, 0])
        dIdcoef[1::3] = dPtdCoef.T.dot(dIdpt[:, 1])
        dIdcoef[2::3] = dPtdCoef.T.dot(dIdpt[:, 2])

        # Now back to design variables:
        dIdx_local = J_temp.T.dot(dIdcoef)

        # ---------------- OLD --------------------
        # dIdx_local = self.JT.dot(dIdpt.flatten())
        # dIdpt = numpy.numpy.zeros_like(self.points[self.pt_ind[name]])
        # self.computeTotalJacobian(name, scaled)
        # dIdx_local = self.JT.dot(dIdpt.flatten())

        if comm: # If we have a comm, globaly reduce with sum
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local
        # end if

        return dIdx

    def computeTotalJacobian(self, name='default', scaled=True):
        ''' Return the total point jacobian in CSR format since we
        need this for TACS'''

        if self.JT is not None and self.J_name == name: # Already computed
            return
        
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
        # J_attach and no J_local
        if self.J_attach is not None and self.J_local is None:
            J_temp = sparse.lil_matrix(self.J_attach)
        elif self.J_local is not None and self.J_attach is None:
            J_temp = sparse.lil_matrix(self.J_local)
        else:
            J_temp = sparse.hstack([self.J_attach, self.J_local], format='lil')
        # end if

        # This is the FINAL Jacobian for the current geometry
        # point. We need this to be a sparse matrix for TACS. 
        
        if self.FFD:
         
            dPtdCoef = self.FFD.embeded_volumes[
                self.pt_ind[name]].dPtdCoef.tocoo()
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
         
            # ------------- OLD VERY SLOW IMPLEMENTATION -----------
         #    dPtdCoef = self.FFD.embeded_volumes[self.pt_ind[name]].dPtdCoef
#             JT = sparse.lil_matrix((nDV, nPt*3))
#             for i in xrange(nDV):
#                 JT[i, 0::3] = dPtdCoef.dot(J_temp[0::3, i])
#                 JT[i, 1::3] = dPtdCoef.dot(J_temp[1::3, i])
#                 JT[i, 2::3] = dPtdCoef.dot(J_temp[2::3, i])
#             # end for
#             self.JT = JT.tocsr()
#             self.JT.sort_indices()
            # ------------------------------------------------------
        # end if

        return 

    def _attachedPtJacobian(self, scaled=True):
        '''
        Compute the derivative of the the attached points
        '''

        nDV = self._getNDVGlobal()
        if nDV == 0:
            return None

      
        h = 1.0e-40j
        oneoverh = 1.0/1e-40
        # Just do a CS loop over the coef
        # First sum the actual number of globalDVs

        Jacobian = numpy.zeros((self.nPtAttachFull*3, nDV))

        counter = 0
        for i in xrange(len(self.DV_listGlobal)):
            nVal = self.DV_listGlobal[i].nVal
            for j in xrange(nVal):
                refVal = self.DV_listGlobal[i].value[j]

                self.DV_listGlobal[i].value[j] += h
                
                deriv = oneoverh*numpy.imag(self.update_deriv()).flatten()

                if scaled:
                    # ptAttachInd is of length nPtAttach, but need to
                    # set the x-y-z coordinates here:
                    numpy.put(Jacobian[0::3, counter], self.ptAttachInd, 
                              deriv[0::3]*self.DV_listGlobal[i].range[j])
                    numpy.put(Jacobian[1::3, counter], self.ptAttachInd, 
                              deriv[1::3]*self.DV_listGlobal[i].range[j])
                    numpy.put(Jacobian[2::3, counter], self.ptAttachInd, 
                              deriv[2::3]*self.DV_listGlobal[i].range[j])
                else:
                    numpy.put(Jacobian[0::3, counter], self.ptAttachInd, 
                              deriv[0::3])
                    numpy.put(Jacobian[1::3, counter], self.ptAttachInd, 
                              deriv[1::3])
                    numpy.put(Jacobian[2::3, counter], self.ptAttachInd, 
                              deriv[2::3])
                # end if

                counter = counter + 1

                self.DV_listGlobal[i].value[j] = refVal
                
            # end for
        # end for

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

        Jacobian = sparse.lil_matrix((self.nPtAttachFull*3, nDV))
        for i in xrange(len(self.DV_listLocal)):
            nVal = self.DV_listLocal[i].nVal
            for j in xrange(nVal):
                pt_dv = self.DV_listLocal[i].coef_list[j] 
                irow = pt_dv[0]*3 + pt_dv[1]
                if not scaled:
                    Jacobian[irow, j] = 1.0
                else:
                    Jacobian[irow, j] = self.DV_listLocal[i].range[j]
                # end if
            # end for
        # end for

        return Jacobian

    def addVariablesPyOpt(self, opt_prob):
        '''
        Add the current set of global and local design variables to the opt_prob specified
        '''

        # We are going to do our own scaling here...since pyOpt can't
        # do it...

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
                # end
            # end
        # end

        return opt_prob

    def checkDerivatives(self, name='default'):
        '''Run a brute force FD check on ALL design variables'''
        print 'Computing Analytic Jacobian...'
        self.computeTotalJacobian(name, scaled=False)

        Jac = copy.deepcopy(self.JT)
        
        # Global Variables
        mpiPrint('========================================')
        mpiPrint('             Global Variables           ')
        mpiPrint('========================================')
                 
        coords0 = self.update(name).flatten()
        h = 1e-6
        DVCount = 0
        for i in xrange(len(self.DV_listGlobal)):
            for j in xrange(self.DV_listGlobal[i].nVal):

                mpiPrint('========================================')
                mpiPrint('      GlobalVar(%d), Value(%d)           '%(i, j))
                mpiPrint('========================================')

                refVal = self.DV_listGlobal[i].value[j]

                self.DV_listGlobal[i].value[j] += h
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
