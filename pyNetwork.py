'''
pyNetwork

pyNetwork is the 1D analog of pyGeo (surfaces 2D) and pyBlock (volumes 3D)

The idea is a "network" is a collection of 1D splines that are
connected in some manner. This module provides facility for dealing
with such structures

Copyright (c) 2010 by G. Kenway
All rights reserved. Not to be used for commercial purposes.
Revision: 1.0   $Date: 07/07/2010$


Developers:
-----------
- Gaetan Kenway (GKK)

History
-------
	v. 1.0 - Initial Class Creation (GKK, 2010)
'''
# =============================================================================
# Standard Python modules
# =============================================================================

import os, sys, copy

# =============================================================================
# External Python modules
# =============================================================================
import numpy

# =============================================================================
# Extension modules
# =============================================================================

from mdo_import_helper import mpiPrint
import pyspline as ps
import geo_utils

# =============================================================================
# pyNetwork Class
# =============================================================================

class pyNetwork():
	
    def __init__(self,curves,tol=1e-4,*args,**kwargs):
        
        ''' Create an instance of the netowrk geometry class. 
        
        Input: 
        
        splines: a list of spline objects to be used for the network

        tol: Optional tolerance to determine whether or not then
             end points will be coallased. 

        '''

        # First thing to do is to check if we want totally silent
        # operation i.e. no print statments
        if 'no_print' in kwargs:
            self.NO_PRINT = kwargs['no_print']
        else:
            self.NO_PRINT = False
        # end if

        #------------------- pyNetwork Class Atributes -----------------

        self.topo = None         # The topology of the curves
        self.curves = curves     # The list of curves (pyspline curve)
                                 # objects
        self.nCurve = len(curves)  # The total number of curves
        self.coef  = None        # The global (reduced) set of control
                                 # points
        # --------------------------------------------------------------

        return

# ----------------------------------------------------------------------
#                     Topology Information Functions
# ----------------------------------------------------------------------    

    def doConnectivity(self,node_tol=1e-4):
        '''
        This is the only public edge connectivity function. 
        If file_name exists it loads the file OR it calculates the connectivity
        and saves to that file.
        Optional:
            node_tol: The tolerance for identical nodes
        Returns:
            None
            '''

        self._calcConnectivity(node_tol)
        sizes = []
        for icurve in xrange(self.nCurve):
            sizes.append(self.curves[icurve].Nctl)
        self.topo.calcGlobalNumbering(sizes)
        # end if

        self.coef = numpy.zeros((self.topo.nGlobal,3))
        for i in xrange(len(self.coef)):
            icurve = self.topo.g_index[i][0][0]
            ii     = self.topo.g_index[i][0][1]
            self.coef[i] = self.curves[icurve].coef[ii]
        # end for

        return 

    def _calcConnectivity(self,node_tol):
        '''This function attempts to automatically determine the connectivity
        between the pataches'''
        
        # Calculate the 2 end points

        coords = numpy.zeros((self.nCurve,2,3))
      
        for icurve in xrange(self.nCurve):
            coords[icurve][0] = self.curves[icurve](0)
            coords[icurve][1] = self.curves[icurve](1)
        # end for

        self.topo = geo_utils.CurveTopology(coords=coords)
        return
   
    def printConnectivity(self):
        '''
        Print the Edge connectivity to the screen
        Required:
            None
        Returns:
            None
            '''
        self.topo.printConnectivity()
        return
  
# ----------------------------------------------------------------------
#               Curve Writing Output Functions
# ----------------------------------------------------------------------

    def writeTecplot(self,file_name,orig=False,curves=True,coef=True,
                     curve_labels=False,node_labels=False):

        '''Write the pyNetwork Object to Tecplot dat file
        Required:
            file_name: The filename for the output file
        Optional:
            orig: boolean, write the original data
            curves: boolean, write the interpolated curves
            coef: boolean, write the control points
            curve_labels: boolean, write the curve labels
            node_lables: boolean, write the node labels
            '''

        # Open File and output header
        
        f = ps.pySpline.openTecplot(file_name,3)

        # --------------------------------------
        #    Write out the Interpolated Curves
        # --------------------------------------
        
        if curves == True:
            for icurve in xrange(self.nCurve):
                self.curves[icurve]._writeTecplotCurve(f)

        # -------------------------------
        #    Write out the Control Points
        # -------------------------------
        
        if coef == True:
            for icurve in xrange(self.nCurve):
                self.curves[icurve]._writeTecplotCoef(f)

        # ----------------------------------
        #    Write out the Original Data
        # ----------------------------------
        
        if orig == True:
            for icurve in xrange(self.nCurve):
                self.curves[icurve]._writeTecplotOrigData(f)
                
        # ---------------------------------------------
        #    Write out The Curve and Node Labels
        # ---------------------------------------------
        (dirName,fileName) = os.path.split(file_name)
        (fileBaseName, fileExtension)=os.path.splitext(fileName)

        if curve_labels == True:
            # Split the filename off
            label_filename = dirName+'./'+fileBaseName+'.curve_labels.dat'
            f2 = open(label_filename,'w')
            for icurve in xrange(self.nCurve):
                mid = floor(self.curves[icurve].Nctl/2)
                text_string = 'TEXT CS=GRID3D, X=%f,Y=%f,Z=%f,ZN=%d, T=\"S%d\"\n'%(self.curves[icurve].coef[mid,0],self.curves[icurve].coef[mid,1], self.curves[icurve].coef[mid,2],icurve+1,icurve)
                f2.write('%s'%(text_string))
            # end for 
            f2.close()
        # end if 
        
        if node_labels == True:
            # First we need to figure out where the corners actually *are*
            n_nodes = len(unique(self.topo.node_link.flatten()))
            node_coord = numpy.zeros((n_nodes,3))
            for i in xrange(n_nodes):
                # Try to find node i
                for icurve in xrange(self.nCurve):
                    if self.topo.node_link[icurve][0] == i:
                        coordinate = self.curves[icurve].getValueCorner(0)
                        break
                    elif self.topo.node_link[icurve][1] == i:
                        coordinate = self.curves[icurve].getValueCorner(1)
                        break
                    elif self.topo.node_link[icurve][2] == i:
                        coordinate = self.curves[icurve].getValueCorner(2)
                        break
                    elif self.topo.node_link[icurve][3] == i:
                        coordinate = self.curves[icurve].getValueCorner(3)
                        break
                # end for
                node_coord[i] = coordinate
            # end for
            # Split the filename off

            label_filename = dirName+'./'+fileBaseName+'.node_labels.dat'
            f2 = open(label_filename,'w')

            for i in xrange(n_nodes):
                text_string = 'TEXT CS=GRID3D, X=%f,Y=%f,Z=%f,T=\"n%d\"\n'%(
                    node_coord[i][0],node_coord[i][1],node_coord[i][2],i)
                f2.write('%s'%(text_string))
            # end for 
            f2.close()
        ps.pySpline.closeTecplot(f)
        
        return

# ----------------------------------------------------------------------
#                Update and Derivative Functions
# ----------------------------------------------------------------------

    def _updateCurveCoef(self):
        '''update the coefficents on the pyNetwork update'''
        
        for ii in xrange(len(self.coef)):
            for jj in xrange(len(self.topo.g_index[ii])):
                icurve = self.topo.g_index[ii][jj][0]
                i      = self.topo.g_index[ii][jj][1]
                self.curves[icurve].coef[i] = self.coef[ii]
            # end for
        # end for
        
        return 

    def getBounds(self,curves=None):
        '''Deterine the extents of (a part of) the curves
        Required:
            None:
        Optional:
            curves: a list of curves to include in the calculation
        Returns: xmin and xmin: lowest and highest points
        '''
        if curves==None:
            curves = numpy.arange(self.nCurve)
        # end if
        Xmin0,Xmax0 = self.curves[curves[0]].getBounds()
        for i in xrange(1,len(curves)):
            icurve = curves[i]
            Xmin,Xmax = self.curves[icurve].getBounds()
            # Now check them 
            if Xmin[0] < Xmin0[0]:
                Xmin0[0] = Xmin[0]
            if Xmin[1] < Xmin0[1]:
                Xmin0[1] = Xmin[1]
            if Xmin[2] < Xmin0[2]:
                Xmin0[2] = Xmin[2]
            if Xmax[0] > Xmax0[0]:
                Xmax0[0] = Xmax[0]
            if Xmax[1] > Xmax0[1]:
                Xmax0[1] = Xmax[1]
            if Xmax[2] > Xmax0[2]:
                Xmax0[2] = Xmax[2]
        # end for
        return Xmin0,Xmax0

    def projectRays(self,points,axis,curves=None,*args,**kwargs):
        
        # Lets, cheat, do a point projection:
       
        curveID0,s0 = self.projectPoints(points)
      
        D0 = numpy.zeros((len(s0),3),'d')
        for i in xrange(len(s0)):
            D0[i,:] = self.curves[curveID0[i]](s0[i])-points[i]
        # end for
     
        if curves == None:
            curves = numpy.arange(self.nCurve)
        # end if

        # Now do the same calc as before
        N = len(points)
        S = numpy.zeros((N,len(curves)))
        D = numpy.zeros((N,len(curves),3))     
        
        for i in xrange(len(curves)):
            icurve = curves[i]
            for j in xrange(N):
                ray = ps.pySpline.line(points[j]-axis*1.5*numpy.linalg.norm(D0[j]),
                                    points[j]+axis*1.5*numpy.linalg.norm(D0[j]))
                # end if

                S[j,i],t,D[j,i,:] = self.curves[icurve].projectCurve(
                    ray, Niter=2000)
            # end for
        # end for

        s = numpy.zeros(N)
        curveID = numpy.zeros(N,'intc')

        # Now post-process to get the lowest one
        for i in xrange(N):
            d0 = numpy.linalg.norm((D[i,0]))
            s[i] = S[i,0]
            curveID[i] = curves[0]
            for j in xrange(len(curves)):
                if numpy.linalg.norm(D[i,j]) < d0:
                    d0 = numpy.linalg.norm(D[i,j])
                    s[i] = S[i,j]
                    curveID[i] = curves[j]
                # end for
            # end for
            
        # end for

        return curveID,s

    def projectPoints(self,points,curves=None,*args,**kwargs):
        ''' Project a point(s) onto the nearest curve
        Requires:
            points: points to project (N by 3)
        Optional: 
            curves:  A list of the curves to use
            '''

        if curves == None:
            curves = numpy.arange(self.nCurve)
        # end if
        
        N = len(points)
        S = numpy.zeros((N,len(curves)))
        D = numpy.zeros((N,len(curves),3))     
        for i in xrange(len(curves)):
            icurve = curves[i]
            S[:,i],D[:,i,:] = self.curves[icurve].projectPoint(points,*args,**kwargs)
        # end for
        
        s = numpy.zeros(N)
        curveID = numpy.zeros(N,'intc')

        # Now post-process to get the lowest one
        for i in xrange(N):
            d0 = numpy.linalg.norm((D[i,0]))
            s[i] = S[i,0]
            curveID[i] = curves[0]
            for j in xrange(len(curves)):
                if numpy.linalg.norm(D[i,j]) < d0:
                    d0 = numpy.linalg.norm(D[i,j])
                    s[i] = S[i,j]
                    curveID[i] = curves[j]
                # end for
            # end for
            
        # end for
     
        return curveID,s
#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'

