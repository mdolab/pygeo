'''
pyGeo

pyGeo is a (fairly) complete geometry surfacing engine. It performs
multiple functions including producing surfaces from cross sections,
fitting surfaces and has built-in design variable handling. The actual
b-spline surfaces are of the pySpline surface type. See the individual
functions for additional information

Copyright (c) 2009 by G. Kenway
All rights reserved. Not to be used for commercial purposes.
Revision: 1.0   $Date: 26/05/2009$


Developers:
-----------
- Gaetan Kenway (GKK)
- Graeme Kennedy (GJK)

History
-------
	v. 1.0 - Initial Class Creation (GKK, 2009)
'''
# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, copy

# =============================================================================
# External Python modules
# =============================================================================
import numpy

try:
    from scipy import sparse
    from scipy.sparse.linalg.dsolve import factorized
    USE_SCIPY_SPARSE = True
except:
    USE_SCIPY_SPARSE = False
    print 'There was an error importing scipy scparse tools'

# =============================================================================
# Extension modules
# =============================================================================
from mdo_import_helper import import_modules, mpiPrint
exec(import_modules('geo_utils', 'pySpline'))

# =============================================================================
# pyGeo class
# =============================================================================

class pyGeo():
	
    def __init__(self, init_type, no_print=False,*args, **kwargs):
        
        '''Create an instance of the geometry object. The initialization type, 
        init_type, specifies what type of initialization will be
        used. There are currently 4 initialization types: plot3d, 
        iges, lifting_surface and acdt_geo
        
        Input: 
        
        init_type, string: a key word defining how this geo object
        will be defined. Valid Options/keyword argmuents are:

        'plot3d', file_name = 'file_name.xyz' : Load in a plot3D
        surface patches and use them to create splined surfaces
 
        'iges', file_name = 'file_name.igs': Load the surface patches
        from an iges file to create splined surfaes.

        'lifting_surface', <arguments listed below>

         Mandatory Arguments:
              
              xsections: List of the cross section coordinate files
              scale    : List of the scaling factor for cross sections
              offset   : List of x-y offset to apply BEFORE scaling
              Xsec     : List of spatial coordinates as to the placement of 
                         cross sections
                        OR x=x, y=y and z=z -> coordinates individually
              rot      : List of x-y-z rotations to apply to cross sections
                        OR rot_x=rot_x, rot_y=rot_y or rot_z=rot_z
              Nctl     : Number of control points on each side, chord-wise
              k_span   : The spline order in span-wise direction
              con_file : The file name for the con file
        
         '''
        
        # First thing to do is to check if we want totally silent
        # operation i.e. no print statments
        self.NO_PRINT = no_print
        self.init_type = init_type
        mpiPrint(' ', self.NO_PRINT)
        mpiPrint('-'*48, self.NO_PRINT)
        
        mpiPrint('pyGeo Initialization Type is: %s'%(init_type), 
                 self.NO_PRINT)
        mpiPrint('-'*48, self.NO_PRINT)

        #------------------- pyGeo Class Atributes -----------------
        self.topo = None           # The topology of the surfaces
        self.surfs = []            # The list of surface (pySpline surf)
                                   # objects
        self.nSurf = None          # The total number of surfaces
        self.coef  = None          # The global (reduced) set of control
                                   # points
        self.csm_pre = None
        # --------------------------------------------------------------

        if init_type == 'plot3d':
            self._readPlot3D(*args, **kwargs)
        elif init_type == 'iges':
            self._readIges(*args, **kwargs)
        elif init_type == 'lifting_surface':
            self._init_lifting_surface(*args, **kwargs)
        elif init_type == 'create':  # Don't do anything 
            pass
        else:
            mpiPrint('Unknown init type. Valid Init types are \'plot3d\', \
\'iges\' and \'lifting_surface\'')
            sys.exit(0)

        return

# ----------------------------------------------------------------------------
#               Initialization Type Functions
# ----------------------------------------------------------------------------

    def _readPlot3D(self, file_name, file_type='ascii', order='f'):
        '''Load a plot3D file and create the splines to go with each patch

        file_name: Required string for file
        file_type: 'ascii' or 'binary'
        order: 'f' for fortran ordering (usual), 'c' for c ordering
        '''
        mpiPrint(' ', self.NO_PRINT)
        if file_type == 'ascii':
            mpiPrint('Loading ascii plot3D file: %s ...'%(file_name),
                     self.NO_PRINT)
            binary = False
            f = open(file_name, 'r')
        else:
            mpiPrint('Loading binary plot3D file: %s ...'%(file_name),
                     self.NO_PRINT)
            binary = True
            f = open(file_name, 'rb')
        # end if
        if binary:
            itype = geo_utils.readNValues(f, 1, 'int', binary)[0]
            nSurf = geo_utils.readNValues(f, 1, 'int', binary)[0]
            itype = geo_utils.readNValues(f, 1, 'int', binary)[0] # Need these
            itype = geo_utils.readNValues(f, 1, 'int', binary)[0] # Need these
            sizes   = geo_utils.readNValues(
                f, nSurf*3, 'int', binary).reshape((nSurf, 3))
        else:
            nSurf = geo_utils.readNValues(f, 1, 'int', binary)[0]
            sizes   = geo_utils.readNValues(
                f, nSurf*3, 'int', binary).reshape((nSurf, 3))
        # end if

        # ONE of Patch Sizes index must be one
        nPts = 0
        for i in xrange(nSurf):
            if sizes[i, 0] == 1: # Compress back to indices 0 and 1
                sizes[i, 0] = sizes[i, 1]
                sizes[i, 1] = sizes[i, 2] 
            elif sizes[i, 1] == 1:
                sizes[i, 1] = sizes[i, 2]
            elif sizes[i, 2] == 1:
                pass
            else:
                mpiPrint('Error: One of the plot3d indices must be 1')
            # end if
            nPts += sizes[i, 0]*sizes[i, 1]
        # end for
        mpiPrint(' -> nSurf = %d'%(nSurf), self.NO_PRINT)
        mpiPrint(' -> Surface Points: %d'%(nPts), self.NO_PRINT)

        surfs = []
        for i in xrange(nSurf):
            cur_size = sizes[i, 0]*sizes[i, 1]
            surfs.append(numpy.zeros([sizes[i, 0], sizes[i, 1], 3]))
            for idim in xrange(3):
                surfs[-1][:, :, idim] = geo_utils.readNValues(
                    f, cur_size, 'float', binary).reshape(
                    (sizes[i, 0], sizes[i, 1]), order=order)
            # end for
        # end for

        f.close()

        # Now create a list of spline surface objects:
        self.surfs = []

        # Note This doesn't actually fit the surfaces...just produces
        # the parameterization and knot vectors
        self.nSurf = nSurf
        for isurf in xrange(self.nSurf):
            self.surfs.append(pySpline.surface(X=surfs[isurf], ku=4, kv=4, 
                                               Nctlu=4, Nctlv=4, 
                                               no_print=self.NO_PRINT))
        # end for
        return     

    def _readIges(self, file_name):

        '''Load a Iges file and create the splines to go with each patch'''
        mpiPrint('File Name is: %s'%(file_name), self.NO_PRINT)
        f = open(file_name, 'r')
        Ifile = []
        for line in f:
            line = line.replace(';', ',')  #This is a bit of a hack...
            Ifile.append(line)
        f.close()
        
        start_lines   = int((Ifile[-1][1:8]))
        general_lines = int((Ifile[-1][9:16]))
        directory_lines = int((Ifile[-1][17:24]))
        parameter_lines = int((Ifile[-1][25:32]))

        # Now we know how many lines we have to deal with
        dir_offset  = start_lines + general_lines
        para_offset = dir_offset + directory_lines

        surf_list = []
        # Directory lines is ALWAYS a multiple of 2
        for i in xrange(directory_lines/2): 
            # 128 is bspline surface type
            if int(Ifile[2*i + dir_offset][0:8]) == 128:
                start = int(Ifile[2*i + dir_offset][8:16])
                num_lines = int(Ifile[2*i + 1 + dir_offset][24:32])
                surf_list.append([start, num_lines])
            # end if
        # end for
        self.nSurf = len(surf_list)

        mpiPrint('Found %d surfaces in Iges File.'%(self.nSurf), self.NO_PRINT)

        self.surfs = []

        for isurf in xrange(self.nSurf):  # Loop over our patches
            data = []
            # Create a list of all data
            # -1 is for conversion from 1 based (iges) to python
            para_offset = surf_list[isurf][0]+dir_offset+directory_lines-1 

            for i in xrange(surf_list[isurf][1]):
                aux = Ifile[i+para_offset][0:69].split(',')
                for j in xrange(len(aux)-1):
                    data.append(float(aux[j]))
                # end for
            # end for
            
            # Now we extract what we need
            Nctlu = int(data[1]+1)
            Nctlv = int(data[2]+1)
            ku    = int(data[3]+1)
            kv    = int(data[4]+1)
            
            counter = 10
            tu = data[counter:counter+Nctlu+ku]
            counter += (Nctlu + ku)
            
            tv = data[counter:counter+Nctlv+kv]
            counter += (Nctlv + kv)
            
            weights = data[counter:counter+Nctlu*Nctlv]
            weights = numpy.array(weights)
            if weights.all() != 1:
                mpiPrint('WARNING: Not all weight in B-spline surface are\
 1. A NURBS surface CANNOT be replicated exactly')
            counter += Nctlu*Nctlv

            coef = numpy.zeros([Nctlu, Nctlv, 3])
            for j in xrange(Nctlv):
                for i in xrange(Nctlu):
                    coef[i, j, :] = data[counter:counter +3]
                    counter += 3

            # Last we need the ranges
            prange = numpy.zeros(4)
           
            prange[0] = data[counter    ]
            prange[1] = data[counter + 1]
            prange[2] = data[counter + 2]
            prange[3] = data[counter + 3]

            # Re-scale the knot vectors in case the upper bound is not 1
            tu = numpy.array(tu)
            tv = numpy.array(tv)
            if not tu[-1] == 1.0:
                tu /= tu[-1]

            if not tv[-1] == 1.0:
                tv /= tv[-1]

            self.surfs.append(pySpline.surface(
                ku=ku, kv=kv, tu=tu, tv=tv, coef=coef, no_print=self.NO_PRINT))

            # Generate dummy data for connectivity to work
            u = numpy.linspace(0, 1, 3)
            v = numpy.linspace(0, 1, 3)
            [V, U] = numpy.meshgrid(v, u)
            self.surfs[-1].X = self.surfs[-1](U, V)
            self.surfs[-1].Nu = 3
            self.surfs[-1].Nv = 3
            self.surfs[-1].orig_data = True
         
        return 

    def _init_lifting_surface(self, xsections, X=None, x=None, y=None, z=None, 
                              rot=None, rot_x=None, rot_y=None, rot_z=None, 
                              scale=None, offset=None, Nctl=None, k_span=3, 
                              te_height=None, te_height_scaled=None, thickness=None,
                              blunt_te=False, rounded_te=False, square_te_tip=True,
                              te_scale=0.75, tip='rounded', tip_scale=0.25, 
                              le_offset=.001,
                              te_offset=.001, span_tang=0.5, up_tang=0.5, **kwargs):
        ''' Create a lifting surface by distributing the cross
        sections defined in xsection according to 'X'. Optional
        arguments are as follows:
        
        Supply either: 
        X: Nx3 array of coordinates of each cross section
          or
        x,y,z: Three arrays of length N for each coordiante

        Supply either:
        rot: Nx3 array of x-y-z rotations of each cross section
           or
        rot_x, rot_y, rot_z: Three arrays of length N of each rotation

        scale: list of length xsections to scale each section

        offset: list of length xsections to offset each section in its
                unscaled coordinate frame
        '''
        
        if X is not None:
            Xsec = numpy.array(X)
        else:
            # We have to use x,y,z
            Xsec = numpy.vstack([x,y,z]).T
        # end if
        N = len(Xsec)

        if rot is not None:
            rot = numpy.array(rot)
        else:
            if rot_x is None:
                rot_x = numpy.zeros(N)
            if rot_y is None:
                rot_y = numpy.zeros(N)
            if rot_z is None:
                rot_z = numpy.zeros(N)
            rot = numpy.vstack([rot_x,rot_y,rot_z]).T
        # end if

        if offset is None:
            offset = numpy.zeros((N,2))

        if scale is None:
            scale = numpy.ones(N)

        # Limit k_span to 2 if we only have two cross section
        if len(Xsec) == 2:
            k_span = 2
        # end if
            
        if blunt_te:
            if te_height is None and te_height_scaled is None:
                mpiPrint('Eror: te_height OR te_height_scaled \
must be supplied for blunt_te option')
                sys.exit(1)
            if te_height:
                te_height = numpy.atleast_1d(te_height)
                if len(te_height) == 1:
                    te_height = numpy.ones(N)*te_height
                te_height /= scale

            if te_height_scaled:
                te_height = numpy.atleast_1d(te_height_scaled)
                if len(te_height) == 1:
                    te_height = numpy.ones(N)*te_height
        else:
            te_height = [None for i in xrange(N)]

        # Load in and fit them all 
        curves = []
        knots = []
        for i in xrange(len(xsections)):
            if xsections[i] is not None:
                x, y = geo_utils.read_af2(xsections[i], blunt_te, blunt_thickness=te_height[i])
                weights = numpy.ones(len(x))
                weights[0] = -1
                weights[-1] = -1
                if Nctl is not None:
                    c = pySpline.curve(x=x, y=y, Nctl=Nctl, k=4, weights=weights)
                else:
                    c = pySpline.curve(x=x, y=y, local_interp=True)
                # end if
                curves.append(c)
                knots.append(c.t)
            else:
                curves.append(None)
            # end if
        # end for

        # If we are fitting curves, blend knot vectors and recompute
        if Nctl is not None:
            new_knots = geo_utils.blendKnotVectors(knots, True)
            for i in xrange(len(xsections)):
                if curves[i] is not None:
                    curves[i].t = new_knots.copy()
                    curves[i].recompute(100, computeKnots=False)
                # end if
            # end for

            # If we want a pinched tip will will zero everything here.
            if tip == 'pinched':
            # Just zero out the last section in y
                if curves[-1] is not None:
                    print 'zeroing tip'
                    curves[-1].coef[:,1] = 0
                # end if
            # endif
        else:
            # Otherwise do knot inserions
            orig_knots = [None for i in xrange(N)]
            for i in xrange(N):
                if curves[i] is not None:
                    orig_knots[i] = curves[i].t.copy()
                # end if
            # end for

            # Now for each section go back and insert the required knots
            for i in xrange(N):
                mpiPrint('Inserting knots for curve %d'%(i))
                if curves[i] is not None:
                    for j in xrange(N):
                        if curves[j] is not None:
                            if i<>j:
                               # Add the knots from curve[j] to curve[i]
                                for jj in xrange(len(orig_knots[j])):

                                    found = False
                                    for ii in xrange(len(orig_knots[i])):
                                        if abs(orig_knots[j][jj] - orig_knots[i][ii]) < 1e-12:
                                            found=True
                                            break
                                        # end if
                                    # end for

                                    if not found:
                                        curves[i].insertKnot(orig_knots[j][jj], 1)
                                    # end if
                                # end for
                            # end if
                    # end for
                # end if
            # end for

            # If we want a pinched tip will will zero everything here.
            if tip == 'pinched':
            # Just zero out the last section in y
                if curves[-1] is not None:
                    print 'zeroing tip'
                    curves[-1].coef[:,1] = 0
                # end if
            # endif

            # Finally force ALL curve to have PRECISELY identical knots
            for i in xrange(len(xsections)):
                if curves[i] is not None:
                    curves[i].t = curves[0].t.copy()
            # end for
            new_knots = curves[0].t.copy()
        # end if

        # Generate a curve from X just for the paramterization
        Xcurve = pySpline.curve(X=Xsec, k=k_span)

        # Now blend the missing sections
        mpiPrint('Interpolating missing sections ...')

        for i in xrange(len(xsections)):
            if xsections[i] is None:
                # Fist two cuves bounding this unknown one:
                for j in xrange(i, -1, -1):
                    if xsections[j] is not None:
                        istart = j
                        break
                    # end if
                # end for

                for j in xrange(i, len(xsections), 1):
                    if xsections[j] is not None:
                        iend = j
                        break
                    # end if
                # end for

                # Now generate blending paramter alpha
                s_start = Xcurve.s[istart]
                s_end   = Xcurve.s[iend]
                s       = Xcurve.s[i]

                alpha = (s-s_start)/(s_end-s_start)

                coef = curves[istart].coef*(1-alpha) + \
                    curves[iend].coef*(alpha)

                curves[i] = pySpline.curve(coef=coef, k=4, t=new_knots.copy())
            # end if
        # end for

        # Before we continue the user may want to artifically scale
        # the thickness of the sections. This is useful when a
        # different airfoil thickness is desired than the actual
        # airfoil coordinates. 

        if thickness is not None:
            thickness = numpy.atleast_1d(thickness)
            if len(thickness) == 1:
                thickness = numpy.ones(len(thickness))*thickness
            for i in xrange(N):
                # Only scale the interior control points; not the first and last
                curves[i].coef[1:-1,1] *= thickness[i]
            # end for
        # end if         
        # Now split each curve at u_split which roughly coorsponds to LE
        top_curves = []
        bot_curves = []
        u_split = curves[0].t[(curves[0].Nctl+4-1)/2]

        for i in xrange(len(xsections)):
            c1, c2 = curves[i].splitCurve(u_split)
            top_curves.append(c1)
            c2.reverse()
            bot_curves.append(c2)
        # end for
    
        # Note that the number of control points on the upper and
        # lower surface MAY not be the same. We can fix this by doing
        # more knot insersions. 
        knots_top = top_curves[0].t.copy()
        knots_bot = bot_curves[0].t.copy()

        mpiPrint('Symmetrizing Knot Vectors ...' )
        eps = 1e-12
        for i in xrange(len(knots_top)):
            # Check if knots_top[i] is not in knots_bot to within eps
            found = False
            for j in xrange(len(knots_bot)):
                if abs(knots_top[i] - knots_bot[j]) < eps:
                    found=True
                # end if
            # end for
            if not found:
                # Add to all section
                for ii in xrange(len(xsections)):
                            bot_curves[ii].insertKnot(knots_top[i], 1)
                # end for
            # end if
        # end for

        for i in xrange(len(knots_bot)):
            # Check if knots_bot[i] is not in knots_top to within eps
            found = False
            for j in xrange(len(knots_top)):
                if abs(knots_bot[i] - knots_top[j]) < eps:
                    found=True
                # end if
            # end for
            if not found:
                # Add to all section
                for ii in xrange(len(xsections)):
                    top_curves[ii].insertKnot(knots_bot[i], 1)
                # end for
            # end if
        # end for

        # We now have symmetrized knot vectors for the upper and lower
        # surfaces. We will copy the vectors to make sure they are
        # precisely the same:
        for i in xrange(len(xsections)):
            top_curves[i].t = top_curves[0].t.copy()
            bot_curves[i].t = top_curves[0].t.copy()
        # end for

        # Now we can set the surfaces
        ncoef = top_curves[0].Nctl
        coef_top = numpy.zeros((ncoef, len(xsections), 3))
        coef_bot = numpy.zeros((ncoef, len(xsections), 3))

        for i in xrange(len(xsections)):
            # Scale, rotate and translate the coefficients
            coef_top[:, i, 0] = scale[i]*(
                top_curves[i].coef[:, 0] - offset[i, 0])
            coef_top[:, i, 1] = scale[i]*(
                top_curves[i].coef[:, 1] - offset[i, 1])
            coef_top[:, i, 2] = 0

            coef_bot[:, i, 0] = scale[i]*(
                bot_curves[i].coef[:, 0] - offset[i, 0])
            coef_bot[:, i, 1] = scale[i]*(
                bot_curves[i].coef[:, 1] - offset[i, 1])
            coef_bot[:, i, 2] = 0
            
            for j in xrange(ncoef):
                coef_top[j, i, :] = geo_utils.rotzV(coef_top[j, i, :], 
                                                    rot[i, 2]*numpy.pi/180)
                coef_top[j, i, :] = geo_utils.rotxV(coef_top[j, i, :], 
                                                    rot[i, 0]*numpy.pi/180)
                coef_top[j, i, :] = geo_utils.rotyV(coef_top[j, i, :], 
                                                    rot[i, 1]*numpy.pi/180)

                coef_bot[j, i, :] = geo_utils.rotzV(coef_bot[j, i, :], 
                                                    rot[i, 2]*numpy.pi/180)
                coef_bot[j, i, :] = geo_utils.rotxV(coef_bot[j, i, :], 
                                                    rot[i, 0]*numpy.pi/180)
                coef_bot[j, i, :] = geo_utils.rotyV(coef_bot[j, i, :], 
                                                    rot[i, 1]*numpy.pi/180)
            # end for

            # Finally translate according to  positions specified
            coef_top[:, i, :] += Xsec[i, :]
            coef_bot[:, i, :] += Xsec[i, :]
        # end for

        # Set the two main surfaces
        self.surfs.append(pySpline.surface(
                coef=coef_top, ku=4, kv=k_span, tu=top_curves[0].t, tv=Xcurve.t))
        self.surfs.append(pySpline.surface(
                coef=coef_bot, ku=4, kv=k_span, tu=bot_curves[0].t, tv=Xcurve.t))

        mpiPrint('Computing TE surfaces ...')

        if blunt_te:
            if not rounded_te:
                coef = numpy.zeros((len(xsections), 2, 3), 'd')
                coef[:, 0, :] = coef_top[0, :, :]
                coef[:, 1, :] = coef_bot[0, :, :]
                self.surfs.append(pySpline.surface(
                        coef=coef, ku=k_span, kv=2, tu=Xcurve.t, tv=[0, 0, 1, 1]))
            else:
                coef = numpy.zeros((len(xsections), 4, 3), 'd')
                coef[:, 0, :] = coef_top[0, :, :]
                coef[:, 3, :] = coef_bot[0, :, :]

                # We need to get the tangent for the top and bottom
                # surface, multiply by a scaling factor and this gives
                # us the two inner rows of control points
            
                for j in xrange((len(xsections))):
                    proj_top = (coef_top[0, j]-coef_top[1, j])
                    proj_bot = (coef_bot[0, j]-coef_bot[1, j])
                    proj_top /= numpy.linalg.norm(proj_top)
                    proj_bot /= numpy.linalg.norm(proj_bot)
                    cur_te_thick = numpy.linalg.norm(coef_top[0, j] - coef_bot[0,j])
                    coef[j, 1] = coef[j, 0] + proj_top*0.5*cur_te_thick*te_scale
                    coef[j, 2] = coef[j, 3] + proj_bot*0.5*cur_te_thick*te_scale
                # end for

                self.surfs.append(pySpline.surface(
                        coef=coef, ku=k_span, kv=4, tu=Xcurve.t,
                        tv=[0, 0, 0, 0, 1, 1, 1, 1]))
            # endif
        # end if
        self.nSurf =  len(self.surfs)

        mpiPrint('Computing Tip surfaces ...')
        # # Add on additional surfaces if required for a rounded pinch tip
        if tip == 'rounded':

            # Generate the midpoint of the coefficients
            mid_pts = numpy.zeros([ncoef, 3])
            up_vec  = numpy.zeros([ncoef, 3])
            ds_norm = numpy.zeros([ncoef, 3])
            for j in xrange(ncoef):
                mid_pts[j] = 0.5*(coef_top[j, -1] + coef_bot[j, -1])
                up_vec[j]  = (coef_top[j, -1] - coef_bot[j, -1])
                ds = 0.5*((coef_top[j, -1]-coef_top[j, -2]) + (
                        coef_bot[j, -1]-coef_bot[j, -2]))
                ds_norm[j] = ds/numpy.linalg.norm(ds)
            # end for

            # Generate "average" projection Vector
            proj_vec = numpy.zeros((ncoef, 3), 'd')
            for j in xrange(ncoef):
                offset = te_offset + (float(j)/(ncoef-1))*(
                    le_offset-te_offset)
                proj_vec[j] = ds_norm[j]*(numpy.linalg.norm(
                        up_vec[j]*tip_scale + offset))

            # Generate the tip "line"
            tip_line = numpy.zeros([ncoef, 3])
            for j in xrange(ncoef):
                tip_line[j] =  mid_pts[j] + proj_vec[j]
            # end for

            # Generate a k=4 (cubic) surface
            coef_top_tip = numpy.zeros([ncoef, 4, 3])
            coef_bot_tip = numpy.zeros([ncoef, 4, 3])

            for j in xrange(ncoef):
                coef_top_tip[j, 0] = coef_top[j, -1]
                coef_top_tip[j, 1] = coef_top[j, -1] + \
                    proj_vec[j]*span_tang
                coef_top_tip[j, 2] = tip_line[j] + \
                    up_tang*up_vec[j]
                coef_top_tip[j, 3] = tip_line[j]

                coef_bot_tip[j, 0] = coef_bot[j, -1]
                coef_bot_tip[j, 1] = coef_bot[j, -1] + \
                    proj_vec[j]*span_tang
                coef_bot_tip[j, 2] = tip_line[j] - \
                    up_tang*up_vec[j]
                coef_bot_tip[j, 3] = tip_line[j]
            # end for

            # Modify for square_te_tip... taper over last 20%
            if square_te_tip and not rounded_te:
                tip_dist = geo_utils.e_dist(tip_line[0], tip_line[-1])
                
                for j in xrange(ncoef): # Going from back to front:
                    fraction = geo_utils.e_dist(tip_line[j], tip_line[0]) / tip_dist
                    if fraction < 0.10:
                        fact = (1-fraction/0.10)**2
                        omfact = 1.0-fact
                        coef_top_tip[j, 1] = (
                            fact*((5.0/6.0)*coef_top_tip[j,0] + (1.0/6.0)*coef_bot_tip[j,0]) +
                            omfact*coef_top_tip[j,1])
                        coef_top_tip[j, 2] = (
                            fact*((4.0/6.0)*coef_top_tip[j,0] + (2.0/6.0)*coef_bot_tip[j,0]) + 
                            omfact*coef_top_tip[j,2])
                        coef_top_tip[j, 3] = (
                            fact*((1.0/2.0)*coef_top_tip[j,0] + (1.0/2.0)*coef_bot_tip[j,0]) + 
                            omfact*coef_top_tip[j,3])

                        coef_bot_tip[j, 1] = (
                            fact*((1.0/6.0)*coef_top_tip[j,0] + (5.0/6.0)*coef_bot_tip[j,0]) +
                            omfact*coef_bot_tip[j,1])
                        coef_bot_tip[j, 2] = (
                            fact*((2.0/6.0)*coef_top_tip[j,0] + (4.0/6.0)*coef_bot_tip[j,0]) + 
                            omfact*coef_bot_tip[j,2])
                        coef_bot_tip[j, 3] = (
                            fact*((1.0/2.0)*coef_top_tip[j,0] + (1.0/2.0)*coef_bot_tip[j,0]) + 
                            omfact*coef_bot_tip[j,3])
                    # end if
                # end for
            # end if

            surf_top_tip = pySpline.surface(
                coef=coef_top_tip, ku=4, kv=4, tu=top_curves[0].t, 
                tv=[0, 0, 0, 0, 1, 1, 1, 1])
            surf_bot_tip = pySpline.surface(
                coef=coef_bot_tip, ku=4, kv=4, tu=bot_curves[0].t, 
                tv=[0, 0, 0, 0, 1, 1, 1, 1])
            self.surfs.append(surf_top_tip)
            self.surfs.append(surf_bot_tip)
            self.nSurf += 2

            if blunt_te: 
                # This is the small surface at the trailing edge
                # tip. There are a couple of different things that can
                # happen: If we rounded TE we MUST have a
                # rounded-spherical-like surface (second piece of code
                # below). Otherwise we only have a surface if
                # square_te_tip is false in which case a flat curved
                # surface results. 

                if not rounded_te and not square_te_tip:
                    coef = numpy.zeros((4, 2, 3), 'd')
                    coef[:, 0] = coef_top_tip[0, :]
                    coef[:, 1] = coef_bot_tip[0, :]

                    self.surfs.append(pySpline.surface(
                            coef=coef, ku=4, kv=2, 
                            tu=[0, 0, 0, 0, 1, 1, 1, 1], tv=[0, 0, 1, 1]))
                    self.nSurf += 1
                elif rounded_te:
                    coef = numpy.zeros((4, 4, 3), 'd')
                    coef[:, 0] = coef_top_tip[0, :]
                    coef[:, 3] = coef_bot_tip[0, :]

                    # We will actually recompute the coefficents
                    # on the last sections since we need to do a
                    # couple of more for this surface
                    for i in xrange(4):
                        proj_top = (coef_top_tip[0, i] - coef_top_tip[1, i])
                        proj_bot = (coef_bot_tip[0, i] - coef_bot_tip[1, i])
                        proj_top /= numpy.linalg.norm(proj_top)
                        proj_bot /= numpy.linalg.norm(proj_bot)
                        cur_te_thick = numpy.linalg.norm(coef_top_tip[0, i] - coef_bot_tip[0, i])
                        coef[i, 1] = coef[i, 0] + proj_top*0.5*cur_te_thick*te_scale
                        coef[i, 2] = coef[i, 3] + proj_bot*0.5*cur_te_thick*te_scale

                    self.surfs.append(pySpline.surface(
                            coef=coef, ku=4, kv=4, 
                            tu=[0, 0, 0, 0, 1, 1, 1, 1], 
                            tv=[0, 0, 0, 0, 1, 1, 1, 1]))
                    self.nSurf += 1
                # end if
            # end if blunt_te
        elif tip == 'pinched':
            pass
        else:
            mpiPrint('No tip specified')
        # end if

        # Cheat and make "original data" so that the edge connectivity works
        u = numpy.linspace(0, 1, 3)
        v = numpy.linspace(0, 1, 3)
        [V, U] = numpy.meshgrid(u, v)
        for i in xrange(self.nSurf):
            self.surfs[i].orig_data = True
            self.surfs[i].X = self.surfs[i](U, V)
            self.surfs[i].Nu = 3
            self.surfs[i].Nv = 3
        # end for

        self._calcConnectivity(1e-6, 1e-6)
        sizes = []
        for isurf in xrange(self.nSurf):
            sizes.append([self.surfs[isurf].Nctlu, self.surfs[isurf].Nctlv])
        self.topo.calcGlobalNumbering(sizes)

        self._setSurfaceCoef()

        return

    def fitGlobal(self):
        mpiPrint(' ', self.NO_PRINT)
        mpiPrint('Global Fitting', self.NO_PRINT)
        nCtl = self.topo.nGlobal
        mpiPrint(' -> Copying Topology')
        orig_topo = copy.deepcopy(self.topo)
        
        mpiPrint(' -> Creating global numbering', self.NO_PRINT)
        sizes = []
        for isurf in xrange(self.nSurf):
            sizes.append([self.surfs[isurf].Nu, self.surfs[isurf].Nv])
        # end for
        
        # Get the Globaling number of the original data
        orig_topo.calcGlobalNumbering(sizes) 
        N = orig_topo.nGlobal
        mpiPrint(' -> Creating global point list', self.NO_PRINT)
        pts = numpy.zeros((N, 3))
        for ii in xrange(N):
            pts[ii] = self.surfs[
                orig_topo.g_index[ii][0][0]].X[orig_topo.g_index[ii][0][1], 
                                               orig_topo.g_index[ii][0][2]]
        # end for

        # Get the maximum k (ku, kv for each surf)
        kmax = 2
        for isurf in xrange(self.nSurf):
            if self.surfs[isurf].ku > kmax:
                kmax = self.surfs[isurf].ku
            if self.surfs[isurf].kv > kmax:
                kmax = self.surfs[isurf].kv
            # end if
        # end for
        nnz = N*kmax*kmax
        vals = numpy.zeros(nnz)
        row_ptr = [0]
        col_ind = numpy.zeros(nnz, 'intc')
        mpiPrint(' -> Calculating Jacobian', self.NO_PRINT)
        for ii in xrange(N):
            isurf = orig_topo.g_index[ii][0][0]
            i = orig_topo.g_index[ii][0][1]
            j = orig_topo.g_index[ii][0][2]

            u = self.surfs[isurf].U[i, j]
            v = self.surfs[isurf].V[i, j]

            vals, col_ind = self.surfs[isurf]._getBasisPt(
                u, v, vals, row_ptr[ii], col_ind, self.topo.l_index[isurf])

            kinc = self.surfs[isurf].ku*self.surfs[isurf].kv
            row_ptr.append(row_ptr[-1] + kinc)
        # end for

        # Now we can crop out any additional values in col_ptr and vals
        vals    = vals[:row_ptr[-1]]
        col_ind = col_ind[:row_ptr[-1]]
        # Now make a sparse matrix

        NN = sparse.csr_matrix((vals, col_ind, row_ptr))
        mpiPrint(' -> Multiplying N^T * N', self.NO_PRINT)
        NNT = NN.T
        NTN = NNT*NN
        mpiPrint(' -> Factorizing...', self.NO_PRINT)
        solve = factorized(NTN)
        mpiPrint(' -> Back Solving...', self.NO_PRINT)
        self.coef = numpy.zeros((nCtl, 3))
        for idim in xrange(3):
            self.coef[:, idim] = solve(NNT*pts[:, idim])
        # end for

        mpiPrint(' -> Setting Surface Coefficients...', self.NO_PRINT)
        self._updateSurfaceCoef()
	
        return

# ----------------------------------------------------------------------
#                     Topology Information Functions
# ----------------------------------------------------------------------    

    def doConnectivity(self, file_name=None, node_tol=1e-4, edge_tol=1e-4):
        '''
        This is the only public edge connectivity function. 
        If file_name exists it loads the file OR it calculates the connectivity
        and saves to that file.
        Optional:
            file_name: filename for con file
            node_tol: The tolerance for identical nodes
            edge_tol: The tolerance for midpoints of edge being identical
        Returns:
            None
            '''
        if file_name is not None and os.path.isfile(file_name):
            mpiPrint(' ', self.NO_PRINT)
            mpiPrint('Reading Connectivity File: %s'%(file_name), self.NO_PRINT)
            self.topo = geo_utils.SurfaceTopology(file=file_name)
            if self.init_type != 'iges':
                self._propagateKnotVectors()
            # end if
            sizes = []
            for isurf in xrange(self.nSurf):
                sizes.append([self.surfs[isurf].Nctlu, self.surfs[isurf].Nctlv])
                self.surfs[isurf].recompute()
            self.topo.calcGlobalNumbering(sizes)
        else:
            self._calcConnectivity(node_tol, edge_tol)
            sizes = []
            for isurf in xrange(self.nSurf):
                sizes.append([self.surfs[isurf].Nctlu, self.surfs[isurf].Nctlv])
            self.topo.calcGlobalNumbering(sizes)
            if self.init_type != 'iges':
                self._propagateKnotVectors()
            if file_name is not None:
                mpiPrint('Writing Connectivity File: %s'%(file_name),
                         self.NO_PRINT)
                self.topo.writeConnectivity(file_name)
            # end if
        # end if
        if self.init_type == 'iges':
            self._setSurfaceCoef()

        return 

    def _calcConnectivity(self, node_tol, edge_tol):
        '''This function attempts to automatically determine the connectivity
        between the pataches'''
        
        # Calculate the 4 corners and 4 midpoints for each surface

        coords = numpy.zeros((self.nSurf, 8, 3))
      
        for isurf in xrange(self.nSurf):
            beg, mid, end = self.surfs[isurf].getOrigValuesEdge(0)
            coords[isurf][0] = beg
            coords[isurf][1] = end
            coords[isurf][4] = mid
            beg, mid, end = self.surfs[isurf].getOrigValuesEdge(1)
            coords[isurf][2] = beg
            coords[isurf][3] = end
            coords[isurf][5] = mid
            beg, mid, end = self.surfs[isurf].getOrigValuesEdge(2)
            coords[isurf][6] = mid
            beg, mid, end = self.surfs[isurf].getOrigValuesEdge(3)
            coords[isurf][7] = mid
        # end for

        self.topo = geo_utils.SurfaceTopology(coords=coords, node_tol=node_tol, 
                                              edge_tol=edge_tol)
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
  
    def _propagateKnotVectors(self):
        ''' Propage the knot vectors to make consistent'''
        # First get the number of design groups
        nDG = -1
        ncoef = []
        for i in xrange(self.topo.nEdge):
            if self.topo.edges[i].dg > nDG:
                nDG = self.topo.edges[i].dg
                ncoef.append(self.topo.edges[i].N)
            # end if
        # end for

        nDG += 1
        for isurf in xrange(self.nSurf):
            dg_u = self.topo.edges[self.topo.edge_link[isurf][0]].dg
            dg_v = self.topo.edges[self.topo.edge_link[isurf][2]].dg
            self.surfs[isurf].Nctlu = ncoef[dg_u]
            self.surfs[isurf].Nctlv = ncoef[dg_v]
            if self.surfs[isurf].ku < self.surfs[isurf].Nctlu:
                if self.surfs[isurf].Nctlu > 4:
                    self.surfs[isurf].ku = 4
                else:
                    self.surfs[isurf].ku = self.surfs[isurf].Nctlu
		# endif
            # end if
            if self.surfs[isurf].kv < self.surfs[isurf].Nctlv:
                if self.surfs[isurf].Nctlv > 4:
                    self.surfs[isurf].kv = 4
                else:
                    self.surfs[isurf].kv = self.surfs[isurf].Nctlv
                # end if
            # end if

            self.surfs[isurf]._calcKnots()
            # Now loop over the number of design groups, accumulate all
            # the knot vectors that coorspond to this dg, then merge them all
        # end for

        for idg in xrange(nDG):
            knot_vectors = []
            flip = []
            for isurf in xrange(self.nSurf):
                for iedge in xrange(4):
                    if self.topo.edges[
                        self.topo.edge_link[isurf][iedge]].dg == idg:
                        if self.topo.edge_dir[isurf][iedge] == -1:
                            flip.append(True)
                        else:
                            flip.append(False)
                        # end if
                        if iedge in [0, 1]:
                            knot_vec = self.surfs[isurf].tu
                        elif iedge in [2, 3]:
                            knot_vec = self.surfs[isurf].tv
                        # end if

                        if flip[-1]:
                            knot_vectors.append((1-knot_vec)[::-1].copy())
                        else:
                            knot_vectors.append(knot_vec)
                        # end if
                    # end if
                # end for
            # end for
            
            # Now blend all the knot vectors
            new_knot_vec = geo_utils.blendKnotVectors(knot_vectors, False)
            new_knot_vec_flip = (1-new_knot_vec)[::-1]

            counter = 0
            for isurf in xrange(self.nSurf):
                for iedge in xrange(4):
                    if self.topo.edges[
                        self.topo.edge_link[isurf][iedge]].dg == idg:
                        if iedge in [0, 1]:
                            if flip[counter] == True:
                                self.surfs[isurf].tu = new_knot_vec_flip.copy()
                            else:
                                self.surfs[isurf].tu = new_knot_vec.copy()
                            # end if
                        elif iedge in [2, 3]:
                            if flip[counter] == True:
                                self.surfs[isurf].tv = new_knot_vec_flip.copy()
                            else:
                                self.surfs[isurf].tv = new_knot_vec.copy()
                            # end if
                        # end if
                        counter += 1
                    # end if
                # end for
            # end for
        # end for idg
        return   

# ----------------------------------------------------------------------
#                   Surface Writing Output Functions
# ----------------------------------------------------------------------

    def writeTecplot(self, file_name, orig=False, surfs=True, coef=True, 
                     directions=False, surf_labels=False, edge_labels=False, 
                     node_labels=False):

        '''Write the pyGeo Object to Tecplot dat file
        Required:
            file_name: The filename for the output file
        Optional:
            orig: boolean, write the original data
            surfs: boolean, write the interpolated surfaces 
            coef: boolean, write the control points
            links: boolean, write the coefficient links
            directions: boolean, write the surface direction indicators
            surf_labels: boolean, write the surface labels
            edge_labels: boolean, write the edge labels
            node_lables: boolean, write the node labels
            size: A lenght parameter to control the surface interpolation size
            '''

        # Open File and output header
        
        f = pySpline.openTecplot(file_name, 3)

        # --------------------------------------
        #    Write out the Interpolated Surfaces
        # --------------------------------------
        
        if surfs:
            for isurf in xrange(self.nSurf):
                self.surfs[isurf]._writeTecplotSurface(f)

        # -------------------------------
        #    Write out the Control Points
        # -------------------------------
        
        if coef:
            for isurf in xrange(self.nSurf):
                pySpline.writeTecplot2D(
                    f,'control_pts',self.surfs[isurf].coef)
                
        # ----------------------------------
        #    Write out the Original Data
        # ----------------------------------
        
        if orig:
            for isurf in xrange(self.nSurf):
                pySpline.writeTecplot2D(
                    f,'orig_data',self.surfs[isurf].X)

        # -----------------------------------
        #    Write out The Surface Directions
        # -----------------------------------

        if directions:
            for isurf in xrange(self.nSurf):
                self.surfs[isurf]._writeDirections(f, isurf)
            # end for
        # end if

        # ---------------------------------------------
        #    Write out The Surface, Edge and Node Labels
        # ---------------------------------------------
        (dirName, fileName) = os.path.split(file_name)
        (fileBaseName, fileExtension)=os.path.splitext(fileName)

        if surf_labels:
            # Split the filename off
            label_filename = dirName+'./'+fileBaseName+'.surf_labels.dat'
            f2 = open(label_filename, 'w')
            for isurf in xrange(self.nSurf):
                midu = numpy.floor(self.surfs[isurf].Nctlu/2)
                midv = numpy.floor(self.surfs[isurf].Nctlv/2)
                text_string = 'TEXT CS=GRID3D, X=%f, Y=%f, Z=%f, ZN=%d,\
 T=\"S%d\"\n'% (self.surfs[isurf].coef[midu, midv, 0], 
                self.surfs[isurf].coef[midu, midv, 1], 
                self.surfs[isurf].coef[midu, midv, 2], 
                isurf+1, isurf)
                f2.write('%s'%(text_string))
            # end for 
            f2.close()
        # end if 

        if edge_labels:
            # Split the filename off
            label_filename = dirName+'./'+fileBaseName+'edge_labels.dat'
            f2 = open(label_filename, 'w')
            for iedge in xrange(self.topo.nEdge):
                surfaces =  self.topo.getSurfaceFromEdge(iedge)
                pt = self.surfs[surfaces[0][0]].edge_curves[surfaces[0][1]](0.5)
                text_string = 'TEXT CS=GRID3D X=%f, Y=%f, Z=%f, T=\"E%d\"\n'%(pt[0], pt[1], pt[2], iedge)
                f2.write('%s'%(text_string))
            # end for
            f2.close()
        # end if
        
        if node_labels:
            # First we need to figure out where the corners actually *are*
            n_nodes = len(geo_utils.unique(self.topo.node_link.flatten()))
            node_coord = numpy.zeros((n_nodes, 3))
            for i in xrange(n_nodes):
                # Try to find node i
                for isurf in xrange(self.nSurf):
                    if self.topo.node_link[isurf][0] == i:
                        coordinate = self.surfs[isurf].getValueCorner(0)
                        break
                    elif self.topo.node_link[isurf][1] == i:
                        coordinate = self.surfs[isurf].getValueCorner(1)
                        break
                    elif self.topo.node_link[isurf][2] == i:
                        coordinate = self.surfs[isurf].getValueCorner(2)
                        break
                    elif self.topo.node_link[isurf][3] == i:
                        coordinate = self.surfs[isurf].getValueCorner(3)
                        break
                # end for
                node_coord[i] = coordinate
            # end for
            # Split the filename off

            label_filename = dirName+'./'+fileBaseName+'.node_labels.dat'
            f2 = open(label_filename, 'w')

            for i in xrange(n_nodes):
                text_string = 'TEXT CS=GRID3D, X=%f, Y=%f, Z=%f, T=\"n%d\"\n'% (
                    node_coord[i][0], node_coord[i][1], node_coord[i][2], i)
                f2.write('%s'%(text_string))
            # end for 
            f2.close()
        pySpline.closeTecplot(f)
        
        return

    def writeIGES(self, file_name):
        '''
        Write the surface to IGES format
        Required:
            file_name: filname for writing the iges file
        Returns:
            None
            '''
        f = open(file_name, 'w')

        #Note: Eventually we may want to put the CORRECT Data here
        f.write('                                                                        S      1\n')
        f.write('1H,,1H;,7H128-000,11H128-000.IGS,9H{unknown},9H{unknown},16,6,15,13,15, G      1\n')
        f.write('7H128-000,1.,1,4HINCH,8,0.016,15H19970830.165254, 0.0001,0.,            G      2\n')
        f.write('21Hdennette@wiz-worx.com,23HLegacy PDD AP Committee,11,3,               G      3\n')
        f.write('13H920717.080000,23HMIL-PRF-28000B0,CLASS 1;                            G      4\n')
        
        Dcount = 1
        Pcount = 1

        for isurf in xrange(self.nSurf):
            Pcount, Dcount = self.surfs[isurf]._writeIGES_directory( \
                f, Dcount, Pcount)

        Pcount  = 1
        counter = 1

        for isurf in xrange(self.nSurf):
            Pcount, counter = self.surfs[isurf]._writeIGES_parameters(\
                f, Pcount, counter)

        # Write the terminate statment
        f.write('S%7dG%7dD%7dP%7d%40sT%6s1\n'%(1, 4, Dcount-1, counter-1, ' ', ' '))
        f.close()

        return
        
# ----------------------------------------------------------------------
#                Update and Derivative Functions
# ----------------------------------------------------------------------

    def _updateSurfaceCoef(self):
        '''Copy the pyGeo list of control points back to the surfaces'''
        for ii in xrange(len(self.coef)):
            for jj in xrange(len(self.topo.g_index[ii])):
                isurf = self.topo.g_index[ii][jj][0]
                i     = self.topo.g_index[ii][jj][1]
                j     = self.topo.g_index[ii][jj][2]
                self.surfs[isurf].coef[i, j] = self.coef[ii].astype('d')
            # end for
        # end for
        for isurf in xrange(self.nSurf):
            self.surfs[isurf]._setEdgeCurves()
        return

    def _setSurfaceCoef(self):
        '''Set the surface coef list from the pySpline surfaces'''
        self.coef = numpy.zeros((self.topo.nGlobal, 3))
        for isurf in xrange(self.nSurf):
            surf = self.surfs[isurf]
            for i in xrange(surf.Nctlu):
                for j in xrange(surf.Nctlv):
                    self.coef[self.topo.l_index[isurf][i, j]] = surf.coef[i, j]
                # end for
            # end for
        # end for

        return 

    def getBounds(self, surfs=None):
        '''Deterine the extents of (a part of) the surfaces
        Required:
            None:
        Optional:
            surfs: a list of surfs to include in the calculation
        Returns: xmin and xmin: lowest and highest points
        '''
        if surfs == None:
            surfs = numpy.arange(self.nSurf)
        # end if
        Xmin0, Xmax0 = self.surfs[surfs[0]].getBounds()
        for i in xrange(1, len(surfs)):
            isurf = surfs[i]
            Xmin, Xmax = self.surfs[isurf].getBounds()
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
        return Xmin0, Xmax0

    def projectCurve(self, curve, surfs=None, *args, **kwargs):
        '''
        Project a pySpline curve onto the pyGeo object
        Requires: 
            curve: the pyspline curve for projection
        Optional:
            surfs:  A subset list of surfaces to use
        Returns:
            pachID: The surface id of the intersectin
            u     : u coordiante of intersection
            v     : v coordinate of intersection
        
        Notes: This aglorithim is not efficient at all.  We basically
        do the curve-surface projection agorithim for each surface
        the loop back over them to see which is the best in terms of
        closest distance. This intent is that the curve ACTUALLY
        intersects one of the surfaces.
        '''

        if surfs == None:
            surfs = numpy.arange(self.nSurf)
         # end if
        temp    = numpy.zeros((len(surfs), 4))
        result  = numpy.zeros((len(surfs), 4))
        patchID = numpy.zeros(len(surfs), 'intc')

        for i in xrange(len(surfs)):
            isurf = surfs[i]
            u, v, s, d = self.surfs[isurf].projectCurve(curve, *args, **kwargs)
            temp[i, :] = [u, v, s, numpy.linalg.norm(d)]
        # end for

        # Sort the results by distance 
        index = numpy.argsort(temp[:, 3])
        
        
        for i in xrange(len(surfs)):
            result[i] = temp[index[i]]
            patchID[i] = surfs[index[i]]

        return result, patchID

    def projectPoints(self, points, surfs=None, *args, **kwargs):
        ''' Project a point(s) onto the nearest surface
        Requires:
            points: points to project (N by 3)
        Optional: 
            surfs:  A list of the surfaces to use
            '''

        if surfs == None:
            surfs = numpy.arange(self.nSurf)
        # end if
        
        N = len(points)
        U       = numpy.zeros((N, len(surfs)))
        V       = numpy.zeros((N, len(surfs)))
        D       = numpy.zeros((N, len(surfs), 3))
        for i in xrange(len(surfs)):
            isurf = surfs[i]
            U[:, i], V[:, i], D[:, i, :] = self.surfs[isurf].projectPoint(
                points, *args, **kwargs)
        # end for

        u = numpy.zeros(N)
        v = numpy.zeros(N)
        patchID = numpy.zeros(N, 'intc')

        # Now post-process to get the lowest one
        for i in xrange(N):
            d0 = numpy.linalg.norm((D[i, 0]))
            u[i] = U[i, 0]
            v[i] = V[i, 0]
            patchID[i] = surfs[0]
            for j in xrange(len(surfs)):
                if numpy.linalg.norm(D[i, j]) < d0:
                    d0 = numpy.linalg.norm(D[i, j])
                    u[i] = U[i, j]
                    v[i] = V[i, j]
                    patchID[i] = surfs[j]
                # end for
            # end for
            
        # end for

        return u, v, patchID

#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'

