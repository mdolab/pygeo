# =============================================================================
# RefAxis utiliy function for pyGeo and pyBlock
# =============================================================================

from numpy import pi,cos,sin,linspace,zeros,where,interp,sqrt,hstack,dot,\
    array,max,min,insert,delete,empty,mod,tan,ones,argsort,mod,sort,\
    arange,copy,floor,fromfile,choose,sign,resize,append,mgrid,average,cross
from numpy.linalg import norm
import string ,sys, copy, pdb, os,time

from mdo_import_helper import *
exec(import_modules('geo_utils','pyNetwork','pySpline'))

class RefAxis(pyNetwork.pyNetwork):

    def __init__(self,curves,points,rot_type=0,*args,**kwargs):

        ''' Create a reference axis network from a pyNetwork object
        
        Reqruied:
            network: The pyNetwork object to use
            points:  The spatial points on which the ref axis acts

        Optional:
        
            rot_type: integer 0-6 to determine the rotation order
                0 -> None   -> Intrinsic rotation, rot_x is roation about axis
                1 -> x-y-z
                2 -> x-z-y
                3 -> y-z-x  -> This example (right body x-streamwise y-out wing z-up)
                4 -> y-x-z
                5 -> z-x-y  -> Default aerosurf (Left body x-streamwise y-up z-out wing)
                6 -> z-y-x
        '''
       
        # We need to define 4 additional scalar splines on the the
        # same basis as each spline in the network object
        
        self.DV_listGlobal  = [] # Global Design Variable List
        self.DV_namesGlobal = {} # Names of Global Design Variables
        
        pyNetwork.pyNetwork.__init__(self,curves,*args,**kwargs)
        self.doConnectivity()
        self.points = points
        self.nPt = len(points)

        # New setup splines for the rotations
        self.rot_x = []
        self.rot_y = []
        self.rot_z = []
        self.scale = []
        self.scale_x = []
        self.scale_y = []
        self.scale_z = []
        for i in xrange(len(self.curves)):
            t = self.curves[i].t
            k = self.curves[i].k
            N = len(self.curves[i].coef)
            self.rot_x.append(pySpline.curve(t=t,k=k,coef=zeros((N,1))))
            self.rot_y.append(pySpline.curve(t=t,k=k,coef=zeros((N,1))))
            self.rot_z.append(pySpline.curve(t=t,k=k,coef=zeros((N,1))))

            self.scale.append(pySpline.curve(t=t,k=k,coef=ones((N,1))))
            self.scale_x.append(pySpline.curve(t=t,k=k,coef=ones((N,1))))
            self.scale_y.append(pySpline.curve(t=t,k=k,coef=ones((N,1))))
            self.scale_z.append(pySpline.curve(t=t,k=k,coef=ones((N,1))))
        # end for

        self.scale0 = copy.deepcopy(self.scale)
        self.scale_x0 = copy.deepcopy(self.scale)
        self.scale_y0 = copy.deepcopy(self.scale)
        self.scale_z0 = copy.deepcopy(self.scale)

        self.rot_type = rot_type
        self.curveIDs,s = self.projectPoints(points)

        self.coef0 = self.coef.copy()

        self.links_s = s
        self.links_x = []
        self.links_n = []
        for i in xrange(self.nPt):
            self.links_x.append(points[i] - self.curves[self.curveIDs[i]](s[i]))
            deriv = self.curves[self.curveIDs[i]].getDerivative(self.links_s[i])
            deriv /= norm(deriv) # Normalize
            self.links_n.append(cross(deriv,self.links_x[-1]))
        # end for
        
        
    def writeTecplot(self,file_name,links=True,*args,**kwargs):

        '''Write the stuff out to a file'''

        f = pySpline.openTecplot(file_name,3)

        # Ref axis curve themselves
        for icurve in xrange(self.nCurve):
            self.curves[icurve]._writeTecplotCurve(f)

        # Ref axis coefficients
        for icurve in xrange(self.nCurve):
            self.curves[icurve]._writeTecplotCoef(f)

        # Write out links
        if links:
            self._writeTecplotLinks(f)
        # end if

        pySpline.closeTecplot(f)

        return

    
    def addGeoDVGlobal(self,dv_name,value,lower,upper,function,useit=True):
        '''Add a global design variable
        Required:
            dv_name: a unquie name for this design variable (group)
            lower: the lower bound for this design variable
            upper: The upper bound for this design variable
            function: the python function for this design variable
        Optional:
            use_it: Boolean flag as to weather to ignore this design variable
        Returns:
            None
            '''


        self.DV_listGlobal.append(geoDVGlobal(\
                dv_name,value,lower,upper,function,useit))
        self.DV_namesGlobal[dv_name]=len(self.DV_listGlobal)-1
        return


    def update(self):

        '''This is pretty straight forward, perform the operations on
        the ref axis according to the design variables, then return
        the list of points provided. It is up to the user to know what
        to do with the points
        '''
        
        # Step 1: Call all the design variables

        for i in xrange(len(self.DV_listGlobal)):
            self.DV_listGlobal[i](self)

        self._updateCurveCoef()

        # Step 2: Update the points
        new_pts = zeros((self.nPt,3))
        for ipt in xrange(self.nPt):
            base_pt = self.curves[self.curveIDs[ipt]](self.links_s[ipt])

            scale = self.scale[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_x = self.scale_x[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_y = self.scale_y[self.curveIDs[ipt]](self.links_s[ipt]) 
            scale_z = self.scale_z[self.curveIDs[ipt]](self.links_s[ipt]) 

            if self.rot_type == 0:
                deriv = self.curves[self.curveIDs[ipt]].getDerivative(self.links_s[ipt])
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


    def ptDeriv(self,X):
        '''
        Compute the derivative of the the attached points





        
        '''

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

#     def _getRotMatrixGlobalToLocal(self,s):
        
#         '''Return the rotation matrix to convert vector from global to
#         local frames'''
#         return     dot(rotyM(self.rotys(s)[0]),dot(rotxM(self.rotxs(s)[0]),\
#                                                     rotzM(self.rotzs(s)[0])))
    
#     def _getRotMatrixLocalToGlobal(self,s):
        
#         '''Return the rotation matrix to convert vector from global to
#         local frames'''
#         return transpose(dot(rotyM(self.rotys(s)[0]),dot(rotxM(self.rotxs(s)[0]),\
#                                                     rotzM(self.rotzs(s)[0]))))

    def _writeTecplotLinks(self,handle):
        '''Write out the surface links. '''

        num_vectors = self.nPt
        coords = zeros((2*num_vectors,3))
        icoord = 0
    
        for i in xrange(self.nPt):
            coords[icoord    ,:] = self.curves[self.curveIDs[i]](self.links_s[i])
            coords[icoord +1 ,:] = self.points[i]
            icoord += 2
        # end for

        icoord = 0
        conn = zeros((num_vectors,2))
        for ivector  in xrange(num_vectors):
            conn[ivector,:] = icoord, icoord+1
            icoord += 2
        # end for

        handle.write('Zone T= %s N= %d ,E= %d\n'%('links',2*num_vectors, num_vectors) )
        handle.write('DATAPACKING=BLOCK, ZONETYPE = FELINESEG\n')

        for n in xrange(3):
            for i in  range(2*num_vectors):
                handle.write('%f\n'%(coords[i,n]))
            # end for
        # end for

        for i in range(num_vectors):
            handle.write('%d %d \n'%(conn[i,0]+1,conn[i,1]+1))
        # end for

        return
