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

    def __init__(self,wing,le_list,te_list,nSpan,nChord):

        '''Create a DVconstrains object.

        Inputs:

        wing: a pyGeo object representing the wing

        domain: This is the same domain object as used in pyLayout. It
        defines the extents obout which the constraints are applied

        nChord: The number of chord-wise thickness constriants

        nSpan: The number of span-wise thickness constraints
     
        '''
        # Create mesh of itersections

        self.nSpan = nSpan
        self.nChord = nChord

        root_line = [le_list[0],te_list[0]]
        tip_line  = [le_list[-1],te_list[-1]]
        le_s = pySpline.curve(X=le_list,k=2)
        te_s = pySpline.curve(X=le_list,k=2)
        root_s = pySpline.curve(X=[le_list[0],te_list[0]],k=2)
        tip_s  = pySpline.curve(X=[le_list[-1],te_list[-1]],k=2)

        span_s = numpy.linspace(0,1,self.nSpan)
        chord_s = numpy.linspace(0,1,self.nChord)

        X = tfi_2d(le_s(span_s),te_s(span_s),root_s(chord_s),tip_s(chord_s))

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

        self.coords = zeros((self.nSpan,self.nChord,2,3),'d')
        
        for i in xrange(self.nSpan): 
            for j in xrange(self.nChord):
                up_vec = array([0.0,1.0,0.0])
                up,down = projectNode(X[i,j],up_vec,p0,v1,v2)
                self.coords[i,j,0,:] = up
                self.coords[i,j,1,:] = down
            # end for
        # end for

        self.D0 = zeros((self.nSpan,self.nChord),'d')
        # Determine the distance between each of the points
        for i in xrange(self.nSpan):
            for j in xrange(self.nChord):
                self.D0[i,j] = e_dist(self.coords[i,j,0,:],self.coords[i,j,1,:])
            # end for
        # end for

    def getCoordinates(self):
        ''' Return the current set of coordinates used for the
        thickness computation'''

        return self.coords.reshape((self.nSpan*self.nChord*2,3))

    def setCoordinates(self,coords):
        ''' Set the new set of surface coordinates '''

        self.coords = coords.reshape((self.nSpan,self.nChord,2,3))

    def addConstraintsPyOpt(self,opt_prob,lower,upper):
        ''' Add thickness contraints to pyOpt
        
         Input: opt_prob -> optimization problem
                lower    -> Fraction of initial thickness allowed
                upper    -> Fraction of upper thickness allowed
                '''
        lower = lower*numpy.ones((self.nSpan,self.nChord),'d').flatten()
        upper = upper*numpy.ones((self.nSpan,self.nChord),'d').flatten()
        
        value = ones((self.nSpan,self.nChord),'d').flatten()
        opt_prob.addConGroup('thickness',self.nSpan*self.nChord, 'i', 
                             value=value, lower=lower, upper=upper)

    def getThicknessConstraints(self):
        '''Return the current thickness constraint'''

        D = zeros((self.nSpan,self.nChord),'d')
        for i in xrange(self.nSpan):
            for j in xrange(self.nChord):
                D[i,j] = e_dist(self.coords[i,j,0,:],self.coords[i,j,1,:])
            # end for
        # end for

        con_value = D/self.D0

        return con_value.flatten()


    def getThicknessSensitivity(self,i,j):

        '''Return the derivative of the i,jth thickness'''

        dTdpt = zeros((self.nChord*self.nSpan*2,3),'d')

        p1b,p2b = e_dist_b(self.coords[i,j,0,:],self.coords[i,j,1,:])
        
        istart = 2*i*self.nChord + 2*j
        dTdpt[istart,:] = p1b/self.D0[i,j]
        dTdpt[istart+1] = p2b/self.D0[i,j]

        return dTdpt
        
