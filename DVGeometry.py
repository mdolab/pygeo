# =============================================================================
# DVGeometry Deals with all the details of taking user supplied design
# variables and mapping it to the descrete surfaces on the CFD and FEA
# disciplines. In the case of the FE discipline, ALL the FE nodes are
# included (distributed in a volume) and as such this currently only
# works with Free Form Deformation Volumes that can deal with these
# spactialy distributed points. 
# =============================================================================


import numpy
from numpy import cross,real,imag
from scipy import sparse
from mdo_import_helper import *
exec(import_modules('geo_utils','pySpline','pyNetwork'))

class DVGeometry(object):
    
    def __init__(self,points, curves, FFD=None, Surface=None, 
                 rot_type=0,*args,**kwargs):

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

        # Thes are the descrete points we must manipulate
        self.points = points
        self.rot_type = rot_type
        self.J = None
        if 'complex' in kwargs:
            if kwargs['complex']:
                self.complex = True
            else:
                self.complex = False
            # end if
        else:
            self.complex = False
        # end if

        if Surface and FFD:
            print 'DVGeometry can only use 1 of FFD or Surface'
            sys.exit(1)

        self.Surface = None
        self.FFD     = None

        if FFD:
            self.FFD = FFD
            self.ptAttach = self.FFD.coef
            self.FFD.embedVolume(real(points))
            self.FFD._calcdPtdCoef(0)
        elif Surface:
            self.Surface = Surface
            self.ptAttach = self.Surface.coef
            self.Surface.attachSurface(points)
            self.Surface._calcdPtdCoef(0)
        else:
            self.ptAttach = pts
        # end if

        # Number of points attached to ref axis
        self.nPtAttach = len(self.ptAttach)
        self.nPt = len(self.points)
        self.refAxis = pyNetwork.pyNetwork(curves,*args,**kwargs)
        self.refAxis.doConnectivity()
       
        # New setup splines for the rotations
        self.rot_x = []
        self.rot_y = []
        self.rot_z = []
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
            self.rot_x.append(pySpline.curve(t=t,k=k,coef=zeros((N,1),'d')))
            self.rot_y.append(pySpline.curve(t=t,k=k,coef=zeros((N,1),'d')))
            self.rot_z.append(pySpline.curve(t=t,k=k,coef=zeros((N,1),'d')))

            self.scale.append(pySpline.curve(t=t,k=k,coef=ones((N,1),'d')))
            self.scale_x.append(pySpline.curve(t=t,k=k,coef=ones((N,1),'d')))
            self.scale_y.append(pySpline.curve(t=t,k=k,coef=ones((N,1),'d')))
            self.scale_z.append(pySpline.curve(t=t,k=k,coef=ones((N,1),'d')))
        # end for
        
        self.scale0 = copy.deepcopy(self.scale)
        self.scale_x0 = copy.deepcopy(self.scale)
        self.scale_y0 = copy.deepcopy(self.scale)
        self.scale_z0 = copy.deepcopy(self.scale)        

        if 'axis' in kwargs:
            if kwargs['axis'] == 'x':
                axis = [1,0,0]
            elif kwargs['axis'] == 'y':
                axis = [0,1,0]
            elif kwargs['axis'] == 'z':
                axis = [0,0,1]
            else:
                axis = kwargs['axis']
            # end if
            self.curveIDs,s = self.refAxis.projectRays(self.ptAttach,array(axis))
        else:
            self.curveIDs,s = self.refAxis.projectPoints(self.ptAttach)
        # end if
                
        self.links_s = s
        self.links_x = []
        self.links_n = []
        for i in xrange(self.nPtAttach):
            self.links_x.append(self.ptAttach[i] - self.refAxis.curves[self.curveIDs[i]](s[i]))
            deriv = self.refAxis.curves[self.curveIDs[i]].getDerivative(self.links_s[i])
            deriv /= norm(deriv) # Normalize
            self.links_n.append(cross(deriv,self.links_x[-1]))
        # end for

        return
    

    def _setInitialValues(self):

        self.coef = copy.deepcopy(self.coef0).astype('D')
        self.scale = copy.deepcopy(self.scale0)
        self.scale_x = copy.deepcopy(self.scale_x0)
        self.scale_y = copy.deepcopy(self.scale_y0)
        self.scale_z = copy.deepcopy(self.scale_z0)      
        

    def addGeoDVGlobal(self,dv_name,value,lower,upper,function,useit=True):
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
        self.DV_listGlobal.append(geoDVGlobal(\
                dv_name,value,lower,upper,function,useit))
        self.DV_namesGlobal[dv_name]=len(self.DV_listGlobal)-1

        return

    def addGeoDVLocal(self,dv_name,lower,upper,axis='y',useit=True):
        '''Add a local design variable
        Required:
        dv_name: a unique name for this design variable (group)
        lower: the lower bound for this design variable
        upper: The upper bound for this design variable
        Optional:
        axis: The epecified axis direction to move this dof. It can be 
              'x','y','z' or 'all'. 
        useitg: Boolean flag as to whether to ignore this design variable
        Returns: 
        N: The number of design variables added for this local Set
        '''

        # Take the FFD or surface coef. We dont' have point selects
        # setup yet, so just take everything in FFD.coef of
        # Surface.coef

        if self.FFD:
            self.DV_listLocal.append(geoDVLocal(\
                    dv_name,lower,upper,axis,arange(len(self.FFD.coef))))
            
        if self.Surface:
            self.DV_listLocal.append(geoDVLocal(\
                    dv_name,lower,upper,axis,arange(len(self.Surface.coef))))
            
        self.DV_namesLocal[dv_name] = len(self.DV_listLocal)-1

        return self.DV_listLocal[-1].nVal

    def setValues(self,dvName,value=None,scaled=True):
        ''' This is the generic set values function. It can set values
        in a number of different ways:

        Type One:
        dvName is a STRING and value is the number of values associated with this DV

        Type Two:
        dvName is a DICTIONARY and the argument of each dictionary entry is the value to set.

        '''

        # To make the setting generic below, we will simply "up cast"
        # the single DVname as a string, and dvName in a list to a dictionary.

        if type(dvName) == str:
            dv_dict = {dvName:value}
        elif type(dvName) == dict:
            dv_dict = dvName
        else:
            mpiPrint('Error setting values. dvName must be one of string or dict')
            return
        # end 

        for key in dv_dict:
            if key in self.DV_namesGlobal:
                vals_to_set = atleast_1d(dv_dict[key]).astype('D')
                assert len(vals_to_set) == self.DV_listGlobal[
                    self.DV_namesGlobal[key]].nVal,\
                    'Incorrect number of design variables for DV: %'%(key)
                if scaled:
                    vals_to_set = vals_to_set * \
                        self.DV_listGlobal[self.DV_namesGlobal[key]].range +\
                        self.DV_listGlobal[self.DV_namesGlobal[key]].lower
                # end if

                self.DV_listGlobal[self.DV_namesGlobal[key]].value = vals_to_set
            # end if
            if key in self.DV_namesLocal:
                vals_to_set = atleast_1d(dv_dict[key])
                assert len(vals_to_set) == self.DV_listLocal[self.DV_namesLocal[key]].nVal,\
                    'Incorrect number of design variables for DV: %'%(key)
                if scaled:
                    vals_to_set = vals_to_set * \
                        self.DV_listLocal[self.DV_namesLocal[key]].range +\
                        self.DV_listLocal[self.DV_namesLocal[key]].lower
                # end if

                self.DV_listLocal[self.DV_namesLocal[key]].value = vals_to_set
            # end if
        return

    def _getRotMatrix(self,rotX,rotY,rotZ):
        if self.rot_type == 1:
            D = dot(rotZ,dot(rotY,rotX))
        elif self.rot_type == 2:
            D = dot(rotY,dot(rotZ,rotX))
        elif self.rot_type == 3:
            D = dot(rotX,dot(rotZ,rotY))
        elif self.rot_type == 4:
            D = dot(rotZ,dot(rotX,rotY))
        elif self.rot_type == 5:
            D = dot(rotY,dot(rotX,rotZ))
        elif self.rot_type == 6:
            D = dot(rotX,dot(rotY,rotZ))
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

    def update(self):

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
            new_pts = zeros((self.nPtAttach,3),'D')
        else:
            new_pts = zeros((self.nPtAttach,3),'d')
        # end if
        
        self._complexifyCoef()

        # Run Global Design Vars
        for i in xrange(len(self.DV_listGlobal)):
            self.DV_listGlobal[i](self)

        self.refAxis.coef = self.coef
        self.refAxis._updateCurveCoef()

        for ipt in xrange(self.nPtAttach):
            base_pt = self.refAxis.curves[self.curveIDs[ipt]](self.links_s[ipt])

            scale = self.scale[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_x = self.scale_x[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_y = self.scale_y[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_z = self.scale_z[self.curveIDs[ipt]](self.links_s[ipt]) 
            if self.rot_type == 0:
                deriv = self.refAxis.curves[self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                deriv /= norm(deriv) # Normalize
                new_vec = -cross(deriv,self.links_n[ipt])
                new_vec = rotVbyW(new_vec,deriv,self.rot_x[self.curveIDs[ipt]](self.links_s[ipt])*pi/180)
                new_pts[ipt] = base_pt + new_vec*scale
            # end if
            else:
            
                rotX = rotxM(self.rot_x[self.curveIDs[ipt]](self.links_s[ipt]))
                rotY = rotyM(self.rot_y[self.curveIDs[ipt]](self.links_s[ipt]))
                rotZ = rotzM(self.rot_z[self.curveIDs[ipt]](self.links_s[ipt]))

                D = self.links_x[ipt]
                rotM = self._getRotMatrix(rotX,rotY,rotZ)
                D = dot(rotM,D)

                D[0] *= scale_x
                D[1] *= scale_y
                D[2] *= scale_z
                new_pts[ipt] = base_pt + D*scale
            # end if
        # end for


        if self.FFD or self.Surface:
            # Now run the local DVs
            for i in xrange(len(self.DV_listLocal)):
                self.DV_listLocal[i](new_pts)
            # end for

        if self.FFD:
            self.FFD.coef = real(new_pts)
            self.FFD._updateVolumeCoef()
            coords = self.FFD.getVolumePoints(0)
        elif self.Surface:
            self.Surface.coef = real(new_pts)
            self.Surface._updateSurfaceCoef()
            coords = self.Surface.getSurfacePoints(0)
        else:
            coords = new_pts
        # end if

        if self.complex:
            coords = coords.astype('D')
            imag_part     = imag(new_pts)
            imag_j = 1j
            if self.FFD:
                coords += imag_j*self.FFD.embeded_volumes[0].dPtdCoef.dot(imag_part)
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

       
        new_pts = zeros((self.nPtAttach,3),'D')

        # Set all coef Values back to initial values

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
                deriv = self.refAxis.curves[self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
                deriv /= norm(deriv) # Normalize
                new_vec = -cross(deriv,self.links_n[ipt])
                new_vec = rotVbyW(new_vec,deriv,self.rot_x[self.curveIDs[ipt]](self.links_s[ipt])*pi/180)
                new_pts[ipt] = base_pt + new_vec*scale
            # end if
            else:

                rotX = rotxM(self.rot_x[self.curveIDs[ipt]](self.links_s[ipt]))
                rotY = rotyM(self.rot_y[self.curveIDs[ipt]](self.links_s[ipt]))
                rotZ = rotzM(self.rot_z[self.curveIDs[ipt]](self.links_s[ipt]))

                D = self.links_x[ipt]
                rotM = self._getRotMatrix(rotX,rotY,rotZ)
                D = dot(rotM,D)

                D[0] *= scale_x
                D[1] *= scale_y
                D[2] *= scale_z
                new_pts[ipt] = base_pt + D*scale
            # end if
        # end for

        return new_pts

    def _complexifyCoef(self):
       
        # Convert coef to complex temporairly
        for i in xrange(len(self.refAxis.curves)):
            self.rot_x[i].coef = self.rot_x[i].coef.astype('D')
            self.rot_y[i].coef = self.rot_y[i].coef.astype('D')
            self.rot_z[i].coef = self.rot_z[i].coef.astype('D')
            
            self.scale[i].coef = self.scale[i].coef.astype('D')
            self.scale_x[i].coef = self.scale_x[i].coef.astype('D')
            self.scale_y[i].coef = self.scale_y[i].coef.astype('D')
            self.scale_z[i].coef = self.scale_z[i].coef.astype('D')
            self.refAxis.curves[i].coef = self.refAxis.curves[i].coef.astype('D')
        # end for

        self.coef = self.coef.astype('D')

        

    def _unComplexifyCoef(self):
       
        # Convert coef to complex temporairly
        for i in xrange(len(self.refAxis.curves)):
            self.rot_x[i].coef = self.rot_x[i].coef.astype('d')
            self.rot_y[i].coef = self.rot_y[i].coef.astype('d')
            self.rot_z[i].coef = self.rot_z[i].coef.astype('d')
            
            self.scale[i].coef = self.scale[i].coef.astype('d')
            self.scale_x[i].coef = self.scale_x[i].coef.astype('d')
            self.scale_y[i].coef = self.scale_y[i].coef.astype('d')
            self.scale_z[i].coef = self.scale_z[i].coef.astype('d')
            self.refAxis.curves[i].coef = self.refAxis.curves[i].coef.astype('d')
        # end for

        self.coef = self.coef.astype('d')


    def totalSensitivity(self,dIdpt,comm,scaled=True):
        '''This function takes the total derivative of an objective,
        I, with respect the points controlled on this processor. We
        take the transpose prodducts and mpi_allreduce them to get the
        resulting value on each processor. 
        '''

        self.computeTotalJacobian(scaled)

        dIdx_local = self.J.T.dot(dIdpt.flatten())

        dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)

        return dIdx

    def computeTotalJacobian(self,scaled=True):
        ''' Return the total point jacobian in CSR format since we
        need this for TACS'''

        # This is going to be DENSE in general
        J_attach = self._attachedPtJacobian(scaled=scaled)

        # This is the sparse jacobian for the local DVs that affect
        # Control points directly.

        J_local = self._localDVJacobian(scaled=scaled)

        # HStack em'
        # Three different possibilities: 
        # J_attach and no J_local
        if J_attach is not None and J_local is None:
            J_temp = J_attach
        elif J_local is not None and J_attach is None:
            J_temp = J_local
        else:
            J_temp = sparse.hstack([J_attach,J_local],format='lil')
        # end if

        # This is the FINAL Jacobian for the current geometry
        # point. We need this to be a sparse matrix for TACS. 
        
        # Transpose of JACOBIAN

        nDV = self._getNDV()
        JT = sparse.lil_matrix((nDV,self.nPt*3))

        if self.FFD:
            dPtdCoef = self.FFD.embeded_volumes[0].dPtdCoef
            # We have a slight problem...dPtdCoef only has the shape
            # functions, and not 3 copies for each of the dof. 
            
            for i in xrange(nDV):
                JT[i,0::3] = dPtdCoef.dot(J_temp[0::3,i])
                JT[i,1::3] = dPtdCoef.dot(J_temp[1::3,i])
                JT[i,2::3] = dPtdCoef.dot(J_temp[2::3,i])
            # end for
        # end if

        self.J = JT.tocsr().transpose(copy=True)
        
        return 

    def _attachedPtJacobian(self,scaled=True):
        '''
        Compute the derivative of the the attached points
        '''

        nDV = self._getNDVGlobal()
        if nDV == 0:
            return None

        self._setInitialValues()
        self._complexifyCoef()

        h = 1.0e-40j
        oneoverh = 1.0/1e-40
        # Just do a CS loop over the coef
        # First sum the actual number of globalDVs

        Jacobian = zeros((self.nPtAttach*3,nDV))

        counter = 0
        for i in xrange(len(self.DV_listGlobal)):
            nVal = self.DV_listGlobal[i].nVal
            for j in xrange(nVal):
                refVal = self.DV_listGlobal[i].value[j]

                self.DV_listGlobal[i].value[j] += h
                
                deriv = oneoverh*imag(self.update_deriv()).flatten()

                if scaled:
                    Jacobian[:,counter] = deriv*self.DV_listGlobal[i].range[j]
                else:
                    Jacobian[:,counter] = deriv
                # end if

                counter = counter + 1

                self.DV_listGlobal[i].value[j] = refVal
                
            # end for
        # end for

        self._unComplexifyCoef()

        return Jacobian

    def _localDVJacobian(self,scaled=True):
        '''
        Return the derivative of the coefficients wrt the local design 
        variables
        '''
        
        # This is relatively straight forward, since the matrix is
        # entirely one's or zeros

        nDV = self._getNDVLocal()
        
        if nDV == 0:
            return None

        Jacobian = sparse.lil_matrix((self.nPtAttach*3,nDV))
        for i in xrange(len(self.DV_listLocal)):
            nVal = self.DV_listLocal[i].nVal
            for j in xrange(nVal):
                pt_dv = self.DV_listLocal[i].coef_list[j] 
                irow = pt_dv[0]*3 + pt_dv[1]
                if not scaled:
                    Jacobian[irow,j] = 1.0
                else:
                    Jacobian[irow,j] = self.DV_listLocal[i].range[j]
                # end if
            # end for
        # end for

        return Jacobian

    def addVariablesPyOpt(self,opt_prob):
        '''
        Add the current set of global and local design variables to the opt_prob specified
        '''

        # We are going to do our own scaling here...since pyOpt can't
        # do it...

        for dvList in [self.DV_listGlobal, self.DV_listLocal]:
            for dv in dvList:
                if dv.nVal > 1:
                    low = zeros(dv.nVal)
                    high= ones(dv.nVal)
                    val = (real(dv.value)-dv.lower)/(dv.upper-dv.lower)
                    opt_prob.addVarGroup(dv.name, dv.nVal, 'c', 
                                         value=val, lower=low, upper=high)
                else:
                    low = 0.0
                    high= 1.0
                    val = (real(dv.value)-dv.lower)/(dv.upper-dv.lower)

                    opt_prob.addVar(dv.name, 'c', value=val,
                                    lower=low,upper=high)
                # end
            # end
        # end

        return opt_prob
