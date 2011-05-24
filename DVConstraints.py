# =============================================================================
# DVConstraints provides a convient way of defining geometric
# constrints for WINGS. This can be very convient for a constrained
# aerodynamic optimization. The basic idea is to produce a "strcuture"
# akin to a rib-spar structure generated in pyLayout and to constrain
# the thickness and/or volume of each of the bays. In fact, pyLayout
# is used to determine this rib/spar type structure. Analytic
# sensitivity information is computed and a facility for adding the
# constrints automatically to a pyOpt optimization problem is also
# provided.
# =============================================================================

import numpy
from mdo_import_helper import *
exec(import_modules('geo_utils','pySpline'))

class DVConstraints(object):

    def __init__(self):

        '''Create a (empty) DVconstrains object. Specific types of
        constraints will added individually'''
        self.nThickCon = 0
        self.thickConPtr = [0]
        self.thickConLower = []
        self.thickConUpper = []
        self.LeTeCon = []
        self.coords = numpy.zeros([0,3],dtype='d')
        self.D0     = numpy.zeros([0  ],dtype='d')
        self.thickConSizes = []
        self.scaled = []
        return

    def addThicknessConstraints(self,wing,le_list,te_list,nSpan,nChord,
                                lower=1.0,upper=3.0,scaled=True):

        '''
        Inputs:

        wing: a pyGeo object representing the wing

        le_list: A list defining the "leading edge" or start of the domain

        te_list: A list defining the "trailing edge" or end of the domain

        nChord: The number values in the chord-wise direction (between le_list and te_list)

        nSpan: The number of span-wise thickness constraints
     
        Lower: The low range for the thickness constraint
        
        Upper: The upper bound for the thickness constraint

        '''
        self.scaled.append(scaled)
        self.thickConPtr.append(self.thickConPtr[-1] + nSpan*nChord)

        # Expand out lower and upper to make them the correct size
        temp = atleast_2d(lower)
        if temp.shape[0] == nSpan and value.shape[1] == nChord:
            lower = temp
        else:
            lower = lower*numpy.ones((nSpan,nChord))
        # end if
                        
        temp = atleast_2d(upper)
        if temp.shape[0] == nSpan and value.shape[1] == nChord:
            upper = temp.flatten()
        else:
            upper = upper*numpy.ones((nSpan,nChord))
        # end if

        xmin,xmax = wing.getBounds()
        scale = e_dist(xmin,xmax)
        
        # Create mesh of itersections

        root_line = [le_list[0],te_list[0]]
        tip_line  = [le_list[-1],te_list[-1]]
        le_s = pySpline.curve(X=le_list,k=2)
        te_s = pySpline.curve(X=te_list,k=2)
        root_s = pySpline.curve(X=[le_list[0],te_list[0]],k=2)
        tip_s  = pySpline.curve(X=[le_list[-1],te_list[-1]],k=2)

        span_s = numpy.linspace(0,1,nSpan)
        chord_s = numpy.linspace(0,1,nChord)

        X = tfi_2d(le_s(span_s),te_s(span_s),root_s(chord_s),tip_s(chord_s))

        p0 = []
        v1 = []
        v2 = []
        level = 0
        for isurf in xrange(wing.nSurf):
            surf = wing.surfs[isurf]
            ku = surf.ku
            kv = surf.kv
            tu = surf.tu
            tv = surf.tv
            
            u = fill_knots(tu,ku,level)
            v = fill_knots(tv,kv,level)

            for i in xrange(len(u)-1):
                for j in xrange(len(v)-1):
                    P0 = surf(u[i  ],v[j  ])
                    P1 = surf(u[i+1],v[j  ])
                    P2 = surf(u[i  ],v[j+1])
                    P3 = surf(u[i+1],v[j+1])

                    p0.append(P0)
                    v1.append(P1-P0)
                    v2.append(P2-P0)

                    p0.append(P3)
                    v1.append(P2-P3)
                    v2.append(P1-P3)

                # end for
            # end for
        # end for
        p0 = array(p0)
        v1 = array(v1)
        v2 = array(v2)

        # Append the new coordinates to self.coords
        coord_offset = len(self.coords)
        D0_offset    = len(self.D0)
        self.coords = numpy.append(self.coords,zeros((nSpan*nChord*2,3)),axis=0)
        self.D0     = numpy.append(self.D0    ,zeros((nSpan*nChord    )),axis=0)
      
        for i in xrange(nSpan): 
            for j in xrange(nChord):
                # Generate the 'up_vec' from taking the cross product across a quad
                if i == 0:
                    u_vec = X[i+1,j]-X[i,j]
                elif i == nSpan - 1:
                    u_vec = X[i,j] - X[i-1,j]
                else:
                    u_vec = X[i+1,j] - X[i-1,j]
                # end if

                if j == 0:
                    v_vec = X[i,j+1]-X[i,j]
                elif j == nChord - 1:
                    v_vec = X[i,j] - X[i,j-1]
                else:
                    v_vec = X[i,j+1] - X[i,j-1]
                # end if

                up_vec = numpy.cross(u_vec,v_vec)*scale

                up,down = projectNode(X[i,j],up_vec,p0,v1,v2)
                self.coords[coord_offset,:] = up
                coord_offset += 1

                self.coords[coord_offset,:] = down
                coord_offset += 1

                # Determine the distance between points
                self.D0[D0_offset] = e_dist(up,down)

                # The constraint will ALWAYS be set as a scaled value,
                # however, it is possible that the user has specified
                # individal values for each location. 
                
                if not scaled:
                    lower[i,j] /= self.D0[D0_offset]
                    upper[i,j] /= self.D0[D0_offset]
                #end
                D0_offset += 1
            # end for
        # end for
        
        # Finally add the thickness constraint values
        self.thickConLower.extend(lower.flatten())
        self.thickConUpper.extend(upper.flatten())
        self.nThickCon += len(lower.flatten())
        self.thickConSizes.append([nSpan,nChord])
        return


    def addLeTeCon(self,DVGeo,up_ind,low_ind):
        '''Add Leading Edge and Trailing Edge Constraints to the FFD
        at the indiceis defined by up_ind and low_ind'''

        assert len(up_ind) == len(low_ind), 'up_ind and low_ind are not the same length'

        if DVGeo.FFD: # Only Setup for FFD's currently
            # Check to see if we have local design variables in DVGeo
            if len(DVGeo.DV_listLocal) == 0:
                mpiPrint('Warning: Trying to add Le/Te Constraint when no local variables found')
            # end if

            # Loop over each set of Local Design Variables
            for i in xrange(len(DVGeo.DV_listLocal)):
                
                # We will assume that each GeoDVLocal only moves on 1,2, or 3 coordinate directions (but not mixed)
                temp = DVGeo.DV_listLocal[i].coef_list # This is already an array
                for j in xrange(len(up_ind)): # Try to find this index in the coef_list
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
                        self.LeTeCon.append([i,up,down])
                    # end if
                # end for
            # end for
            
            # Finally, unique the list to parse out duplicates. Note:
            # This sort may not be stable however, the order of the
            # LeTeCon list doens't matter
            self.LeTeCon = unique(self.LeTeCon)
        else:
            mpiPrint('Warning: addLeTECon is only setup for FFDs')
        # end if

        return

    def getCoordinates(self):
        ''' Return the current set of coordinates used in
        DVConstraints'''

        return self.coords

    def setCoordinates(self,coords):
        ''' Set the new set of coordinates'''

        self.coords = coords.copy()

    def addConstraintsPyOpt(self,opt_prob):
        ''' Add thickness contraints to pyOpt
        
         Input: opt_prob -> optimization problem
                lower    -> Fraction of initial thickness allowed
                upper    -> Fraction of upper thickness allowed
                '''
        if self.nThickCon > 0:
            opt_prob.addConGroup(
                'thickness_constraint',len(self.thickConLower), 'i', 
                lower=self.thickConLower,upper=self.thickConUpper)
        # end if

        if self.LeTeCon:
            # We can just add them individualy
            for i in xrange(len(self.LeTeCon)):
                opt_prob.addCon('LeTeCon%d'%(i),'i',lower=0.0,upper=0.0)
            # end for
        # end if

        return 


    def getLeTeConstraints(self,DVGeo):
        '''Evaluate the LeTe constraint using the current DVGeo opject'''

        con = zeros(len(self.LeTeCon))
        for i in xrange(len(self.LeTeCon)):
            dv = self.LeTeCon[i][0]
            up = self.LeTeCon[i][1]
            down = self.LeTeCon[i][2]
            con[i] = DVGeo.DV_listLocal[dv].value[up] + DVGeo.DV_listLocal[dv].value[down]
        # end for

        return con

    def getLeTeSensitivity(self,DVGeo,scaled=True):
        ndv = DVGeo._getNDV()
        nlete = len(self.LeTeCon)
        dLeTedx = zeros([nlete,ndv])

        DVoffset = [DVGeo._getNDVGlobal()]
        # Generate offset lift of the number of local variables
        for i in xrange(len(DVGeo.DV_listLocal)):
            DVoffset.append(DVoffset[-1] + DVGeo.DV_listLocal[i].nVal)

        for i in xrange(len(self.LeTeCon)):
            # Set the two values a +1 and -1 or (+range - range if scaled)
            dv = self.LeTeCon[i][0]
            up = self.LeTeCon[i][1]
            down = self.LeTeCon[i][2]
            if scaled:
                dLeTedx[i,DVoffset[dv] + up  ] =  DVGeo.DV_listLocal[dv].range[up  ]
                dLeTedx[i,DVoffset[dv] + down] =  DVGeo.DV_listLocal[dv].range[down]
            else:
                dLeTedx[i,DVoffset[dv] + up  ] =  1.0
                dLeTedx[i,DVoffset[dv] + down] =  1.0
            # end if
        # end for

        return dLeTedx

    def getThicknessConstraints(self):
        '''Return the current thickness constraint'''
        D = zeros(self.D0.shape)

        for ii in xrange(len(self.thickConPtr)-1):
            for i in xrange(self.thickConPtr[ii],self.thickConPtr[ii+1]):
                D[i] = e_dist(self.coords[2*i,:],self.coords[2*i+1,:])
                if self.scaled[ii]:
                    D[i]/=self.D0[i]
            # end for
        # end for

        con_value = D

        return con_value


    def getThicknessSensitivity(self,DVGeo,coord_name):

        '''Return the derivative of all the thickness constraints We
        pass in the DVGeo object so this function retuns the full
        appropriate jacobian.
        
        '''

        nDV = DVGeo._getNDV()
        dTdx = zeros((self.nThickCon,nDV))
        dTdpt = zeros(self.coords.shape)

        for ii in xrange(len(self.thickConPtr)-1):
            for i in xrange(self.thickConPtr[ii],self.thickConPtr[ii+1]):

                dTdpt[:,:] = 0.0

                p1b,p2b = e_dist_b(self.coords[2*i,:],self.coords[2*i+1,:])
        
                dTdpt[2*i,:] = p1b
                dTdpt[2*i+1,:] = p2b

                if self.scaled[ii]:
                    dTdpt[2*i,:] /= self.D0[i]
                    dTdpt[2*i+1,:] /= self.D0[i]
                # end if

                dTdx[i,:] = DVGeo.totalSensitivity(dTdpt,name=coord_name)
            # end for
        # end for

        return dTdx
        
