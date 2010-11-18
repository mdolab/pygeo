# =============================================================================
# Utility Functions for Use in pyGeo
# =============================================================================

from numpy import pi,cos,sin,linspace,zeros,where,interp,sqrt,hstack,dot,\
    array,insert,delete,empty,mod,tan,ones,argsort,mod,sort,\
    arange,copy,floor,fromfile,choose,sign,resize,append,mgrid,average,rank,\
    atleast_1d,atleast_2d,atleast_3d

from numpy.linalg import norm
import string ,sys, copy, pdb, os,time

from mdo_import_helper import *

# --------------------------------------------------------------
#                Rotation Functions
# --------------------------------------------------------------

def rotxM(theta):
    '''Return x rotation matrix'''
    theta = theta*pi/180
    M = [[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]]
    return M

def rotyM(theta):
    ''' Return y rotation matrix'''
    theta = theta*pi/180
    M = [[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]
    return M

def rotzM(theta):
    ''' Return z rotation matrix'''
    theta = theta*pi/180
    M = [[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]]
    return M

def rotxV(x, theta):
    ''' Rotate a coordinate in the local x frame'''
    M = [[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]]
    return dot(M,x)

def rotyV(x, theta):
    '''Rotate a coordinate in the local y frame'''
    M = [[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]
    return dot(M,x)

def rotzV(x,theta):
    '''Roate a coordinate in the local z frame'''
    'rotatez:'
    M = [[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]]
    return dot(M,x)

def rotVbyW(V, W, theta):
    ''' Rotate a vector V, about an axis W by angle theta'''

    ux = W[0]
    uy = W[1]
    uz = W[2]
    
    c = cos(theta)
    s = sin(theta)
    R = zeros((3,3))
    R[0,0] = ux**2 + (1-ux**2)*c
    R[0,1] = ux*uy*(1-c) - uz*s
    R[0,2] = ux*uz*(1-c) + uy*s
    
    R[1,0] = ux*uy*(1-c) + uz*s
    R[1,1] = uy**2 + (1-uy**2)*c
    R[1,2] = uy*uz*(1-c) - ux*s
                    
    R[2,0] = ux*uz*(1-c) - uy*s
    R[2,1] = uy*uz*(1-c) + ux*s
    R[2,2] = uz**2+(1-uz**2)*c

    return dot(R,V)

 # --------------------------------------------------------------
 #                I/O Functions
 # --------------------------------------------------------------

def readNValues(handle, N, type, binary=False):
    '''Read N values of type 'float' or 'int' from file handle'''
    if binary == True:
        sep = ""
    else:
        sep = " "
    # end if
    if type == 'int':
        values = fromfile(handle,dtype='int',count=N,sep=sep)
    else:
        values = fromfile(handle,dtype='float',count=N,sep=sep)
    return values

def writeValues(handle, values, type, binary=False):
    '''Read N values of type 'float' or 'int' from file handle'''
    if binary:
        values.tofile(handle)
    else:
        if type == 'float':
            values.tofile(handle,sep=" ",format="%f")
        elif type == 'int':
            values.tofile(handle,sep=" ",format="%d")
        # end if
    # end if
    return 

def read_af2(filename, blunt_te=False, blunt_scale=0.1):
    ''' Load the airfoil file of type file_type'''
    f = open(filename,'r')
    line  = f.readline() # Read (and ignore) the first line
    r = []
    try:
        r.append([float(s) for s in line.split()])
    except:
        r = []
    # end if

    while 1:
        line = f.readline()
        if not line: break # end of file
        if line.isspace(): break # blank line
        r.append([float(s) for s in line.split()])
    # end while
    r = array(r)
    x = r[:,0]
    y = r[:,1]
    npt = len(x)
    # Check for blunt TE:
    if blunt_te == False:
        if y[0] != y[-1]:
            mpiPrint('Blunt Trailing Edge on airfoil: %s'%(filename))
            mpiPrint('Merging to a point over final %f ...'%(blunt_scale))
            yavg = 0.5*(y[0] + y[-1])
            y_top = y[0]
            y_bot = y[-1]
            # Indices on the TOP surface of the wing
            indices = where(x[0:npt/2]>=(1-blunt_scale))[0]
            for i in xrange(len(indices)):
                fact = (x[indices[i]]- (1-blunt_scale))/blunt_scale
                y[indices[i]] = y[indices[i]]- fact*(y_top-yavg)

            # Indices on the BOTTOM surface of the wing
            indices = where(x[npt/2:]>=(1-blunt_scale))[0]
            indices = indices + npt/2
                    
            for i in xrange(len(indices)):
                fact = (x[indices[i]]- (1-blunt_scale))/blunt_scale
                y[indices[i]] = y[indices[i]]- fact*(y_bot-yavg)
            # end for
        # end if
    # end if
    return x,y

def getCoordinatesFromFile(file_name):
    '''Get a list of coordinates from a file - useful for testing
    Required:
        file_name: filename for file
    Returns:
        coordinates: list of coordinates
    '''

    f = open(file_name,'r')
    coordinates = []
    for line in f:
        aux = string.split(line)
        coordinates.append([float(aux[0]),float(aux[1]),float(aux[2])])
    # end for
    f.close()
    coordinates = transpose(array(coordinates))

    return coordinates
# --------------------------------------------------------------
#            Working with Edges Function
# --------------------------------------------------------------

def e_dist(x1, x2):
    '''Get the eculidean distance between two points'''
    return sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 + (x1[2]-x2[2])**2)

# --------------------------------------------------------------
#             Truly Miscellaneous Functions
# --------------------------------------------------------------

def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def unique(s):
    """Return a list of the elements in s, but without duplicates.

    For example, unique([1,2,3,1,2,3]) is some permutation of [1,2,3],
    unique("abcabc") some permutation of ["a", "b", "c"], and
    unique(([1, 2], [2, 3], [1, 2])) some permutation of
    [[2, 3], [1, 2]].

    For best speed, all sequence elements should be hashable.  Then
    unique() will usually work in linear time.

    If not possible, the sequence elements should enjoy a total
    ordering, and if list(s).sort() doesn't raise TypeError it's
    assumed that they do enjoy a total ordering.  Then unique() will
    usually work in O(N*log2(N)) time.

    If that's not possible either, the sequence elements must support
    equality-testing.  Then unique() will usually work in quadratic
    time.
    """

    n = len(s)
    if n == 0:
        return []

    # Try using a dict first, as that's the fastest and will usually
    # work.  If it doesn't work, it will usually fail quickly, so it
    # usually doesn't cost much to *try* it.  It requires that all the
    # sequence elements be hashable, and support equality comparison.
    u = {}
    try:
        for x in s:
            u[x] = 1
    except TypeError:
        del u  # move on to the next method
    else:
        return u.keys()

    # We can't hash all the elements.  Second fastest is to sort,
    # which brings the equal elements together; then duplicates are
    # easy to weed out in a single pass.
    # NOTE:  Python's list.sort() was designed to be efficient in the
    # presence of many duplicate elements.  This isn't true of all
    # sort functions in all languages or libraries, so this approach
    # is more effective in Python than it may be elsewhere.

    try:
        t = list(s)
        t.sort()
    except TypeError:
        del t  # move on to the next method
    else:
        assert n > 0
        last = t[0]
        lasti = i = 1
        while i < n:
            if t[i] != last:
                t[lasti] = last = t[i]
                lasti += 1
            i += 1
        return t[:lasti]

    # Brute force is all that's left.

    u = []
    for x in s:
        if x not in u:
            u.append(x)
    return u

def unique_index(s, s_hash=None):
    '''
    This function is based on unique

    The idea is to take a list s, and reduce it as per unique.

    However, it additionally calculates a linking index arrary that is
    the same size as the original s, and points to where it ends up in
    the the reduced list

    if s_hash is not specified for sorting, s is used

    '''
    if s_hash!=None:
        ind = argsort(argsort(s_hash))
    else:
        ind = argsort(argsort(s))
    # end if
    n = len(s)
    t = list(s)
    t.sort()
    
    diff = zeros(n,'bool')

    last = t[0]
    lasti = i = 1
    while i < n:
        if t[i] != last:
            t[lasti] = last = t[i]
            lasti += 1
        else:
            diff[i] = True
            pass
        # end if
        i += 1
    # end while
    b = where(diff == True)[0]
    for i in xrange(n):
        ind[i] -= b.searchsorted(ind[i],side='right')
    # end for

    return t[:lasti],ind

def pointReduce(points, node_tol=1e-4):
    '''Given a list of N points in ndim space, with possible
    duplicates, return a list of the unique points AND a pointer list
    for the original points to the reduced set'''

    # First 
    points = array(points)
    N = len(points)
    dists = []
    for ipt in xrange(N): 
        dists.append(sqrt(dot(points[ipt],points[ipt])))
    # end for
    temp = array(dists)
    temp.sort()
    ind = argsort(dists)
    i= 0
    cont = True
    new_points = []
    link = zeros(N,'intc')
    link_counter = 0
   
    while cont:
        cont2 = True
        temp_ind = []
        j = i
        while cont2:
            if abs(dists[ind[i]]-dists[ind[j]])<node_tol:
                temp_ind.append(ind[j])
                j = j + 1
                if j == N: # Overrun check
                    cont2 = False
                # end if
            else:
                cont2 = False
            #end if
        # end while
        sub_points = [] # Copy of the list of sub points with the dists
        for ii in xrange(len(temp_ind)):
            sub_points.append(points[temp_ind[ii]])

        # Brute Force Search them 
        sub_unique_pts,sub_link = pointReduceBruteForce(sub_points,node_tol)
        new_points.extend(sub_unique_pts)

        for ii in xrange(len(temp_ind)):
            link[temp_ind[ii]] = sub_link[ii] + link_counter
        # end if
        link_counter += max(sub_link) + 1

        
        #i = j-1
        #i = i + 1
        i=j-1+1
        if i == N:
            cont = False
        # end if
    # end while
    return array(new_points),array(link)

def pointReduceBruteForce(points, node_tol=1e-4):
    '''Given a list of N points in ndim space, with possible
    duplicates, return a list of the unique points AND a pointer list
    for the original points to the reduced set

    BRUTE FORCE VERSION

    '''
    N = len(points)
    unique_points = [points[0]]
    link = [0]
    for i in xrange(1,N):
        found_it = False
        for j in xrange(len(unique_points)):
            if e_dist(points[i],unique_points[j]) < node_tol:
                link.append(j)
                found_it = True
                break
            # end if
        # end for
        if not found_it:
            unique_points.append(points[i])
            link.append(j+1)
        # end if
    # end for
    return array(unique_points),array(link)

def faceOrientation(f1, f2):
    '''Compare two face orientations f1 and f2 and return the
    transform to get f1 back to f2'''
    
    if [f1[0],f1[1],f1[2],f1[3]] == [f2[0],f2[1],f2[2],f2[3]]:
        return 0
    elif [f1[0],f1[1],f1[2],f1[3]] == [f2[1],f2[0],f2[3],f2[2]]:
        return 1
    elif [f1[0],f1[1],f1[2],f1[3]] == [f2[2],f2[3],f2[0],f2[1]]:
        return 2
    elif [f1[0],f1[1],f1[2],f1[3]] == [f2[3],f2[2],f2[1],f2[0]]:
        return 3
    elif [f1[0],f1[1],f1[2],f1[3]] == [f2[0],f2[2],f2[1],f2[3]]:
        return 4
    elif [f1[0],f1[1],f1[2],f1[3]] == [f2[2],f2[0],f2[3],f2[1]]:
        return 5
    elif [f1[0],f1[1],f1[2],f1[3]] == [f2[1],f2[3],f2[0],f2[2]]:
        return 6
    elif [f1[0],f1[1],f1[2],f1[3]] == [f2[3],f2[1],f2[2],f2[0]]:
        return 7
    else:
        mpiPrint('Error with faceOrientation: Not possible.')
        mpiPrint('Orientation 1 [%d %d %d %d]'%(f1[0],f1[1],f1[2],f1[3]))
        mpiPrint('Orientation 2 [%d %d %d %d]'%(f2[0],f2[1],f2[2],f2[3]))
        raise value

def quadOrientation(pt1, pt2):
    '''Given two sets of 4 points in ndim space, pt1 and pt2,
    determine the orientation of pt2 wrt pt1
    This works for both exact quads and "loosely" oriented quads
    .'''
    dist = zeros((4,4))
    for i in xrange(4):
        for j in xrange(4):
            dist[i,j] = e_dist(pt1[i],pt2[j])
        # end for
    # end for

    # Now compute the 8 distances for the 8 possible orientation
    sum_dist = zeros(8)
    sum_dist[0] = dist[0,0] + dist[1,1] + dist[2,2] + dist[3,3] # corners = [0,1,2,3]
    sum_dist[1] = dist[0,1] + dist[1,0] + dist[2,3] + dist[3,2] # corners = [1,0,3,2]
    sum_dist[2] = dist[0,2] + dist[1,3] + dist[2,0] + dist[3,1] # corners = [2,3,0,1]
    sum_dist[3] = dist[0,3] + dist[1,2] + dist[2,1] + dist[3,0] # corners = [3,2,1,0]
    sum_dist[4] = dist[0,0] + dist[1,2] + dist[2,1] + dist[3,3] # corners = [0,2,1,3]
    sum_dist[5] = dist[0,2] + dist[1,0] + dist[2,3] + dist[3,1] # corners = [2,0,3,1]
    sum_dist[6] = dist[0,1] + dist[1,3] + dist[2,0] + dist[3,2] # corners = [1,3,0,2]
    sum_dist[7] = dist[0,3] + dist[1,1] + dist[2,2] + dist[3,0] # corners = [3,1,2,0]

    index = sum_dist.argmin()

    return index

def orientArray(index, in_array):
    '''Take an input array in_array, and rotate/flip according to the index
    output from quadOrientation'''

    if index == 0:
        out_array = in_array.copy()
    elif index == 1:
        out_array = rotateCCW(in_array)
        out_array = rotateCCW(out_array)
        out_array = reverseRows(out_array)
    elif index == 2:
        out_array = reverseRows(in_array)
    elif index == 3:
        out_array = rotateCCW(in_array) # Verified working
        out_array = rotateCCW(out_array)
    elif index == 4:
        out_array = rotateCW(in_array)
        out_array = reverseRows(out_array)
    elif index == 5:
        out_array = rotateCCW(in_array)
    elif index == 6:
        out_array = rotateCW(in_array)
    elif index == 7:
        out_array = rotateCCW(in_array)
        out_array = reverseRows(out_array)
        
    return out_array

def directionAlongSurface(surface, line, section=None):
    '''Determine the dominate (u or v) direction of line along surface'''
    # Now Do two tests: Take N points in u and test N groups
    # against dn and take N points in v and test the N groups
    # again

    N = 3
    sn = linspace(0,1,N)
    dn = zeros((N,3))
    s = linspace(0,1,N)
    for i in xrange(N):
        dn[i,:] = line.getDerivative(sn[i])
    # end for

    u_dot_tot = 0
    for i in xrange(N):
        for n in xrange(N):
            du,dv = surface.getDerivative(s[i],s[n])
            u_dot_tot += dot(du,dn[n,:])
        # end for
    # end for

    v_dot_tot = 0
    for j in xrange(N):
        for n in xrange(N):
            du,dv = surface.getDerivative(s[n],s[j])
            v_dot_tot += dot(dv,dn[n,:])
        # end for
    # end for

    if abs(u_dot_tot) > abs(v_dot_tot):
        # Its along u now get 
        if u_dot_tot >= 0: 
            return 0 # U same direction
        else:
            return 1 # U opposite direction
    else:
        if v_dot_tot >= 0:
            return 2 # V same direction
        else:
            return 3 # V opposite direction
        # end if
    # end if 

def curveDirection(curve1, curve2):
    '''Determine if the direction of curve 1 is basically in the same
    direction as curve2. Return 1 for same direction, -1 for opposite direction'''

    N = 4
    s = linspace(0,1,N)
    tot = 0
    d_forward = 0
    d_backward = 0
    for i in xrange(N):
        tot += dot(curve1.getDerivative(s[i]),curve2.getDerivative(s[i]))
        d_forward += e_dist(curve1.getValue(s[i]),curve2.getValue(s[i]))
        d_backward+= e_dist(curve1.getValue(s[i]),curve2.getValue(s[N-i-1]))
    # end for

    if tot > 0:
        return tot,d_forward
    else:
        return tot,d_backward

def indexPosition1D(i, N):
    '''This function is a generic function which determines if index
    over a list of length N is an interior point or node 0 or node 1.
    '''
    if i>0 and i < N-1: # Interior
        return 0,None
    elif i == 0: # Node 0
        return 1,0
    elif i == N-1: # Node 1
        return 1,1

def indexPosition2D(i, j, N, M):
    '''This function is a generic function which determines if for a grid
    of data NxM with index i going 0->N-1 and j going 0->M-1, it
    determines if i,j is on the interior, on an edge or on a corner

    The funtion return four values: 
    type: this is 0 for interior, 1 for on an edge and 2 for on a corner
    edge: this is the edge number if type==1
    node: this is the node number if type==2 
    index: this is the value index along the edge of interest -- only defined for edges'''

    if i > 0 and i < N - 1 and j > 0 and j < M-1: # Interior
        return 0,None,None,None
    elif i > 0 and i < N - 1 and j == 0:     # Edge 0
        return 1,0,None,i
    elif i > 0 and i < N - 1 and j == M - 1: # Edge 1
        return 1,1,None,i
    elif i == 0 and j > 0 and j < M - 1:     # Edge 2
        return 1,2,None,j
    elif i == N - 1 and j > 0 and j < M - 1: # Edge 3
        return 1,3,None,j
    elif i == 0 and j == 0:                  # Node 0
        return 2,None,0,None
    elif i == N - 1 and j == 0:              # Node 1
        return 2,None,1,None
    elif i == 0 and j == M -1 :              # Node 2
        return 2,None,2,None
    elif i == N - 1 and j == M - 1:          # Node 3
        return 2,None,3,None

def indexPosition3D(i, j, k, N, M, L):
    '''This function is a generic function which determines if for a
    3dgrid of data NxMXL with index i going 0->N-1 and j going 0->M-1
    k going 0->L-1, it determines if i,j,k is on the interior, on a
    face, on an edge or on a corner

    The funtion return theses values: 
    type: this is 0 for interior, 1 for on an face, 
           3 for an edge and 4 for on a corner
    number: this is the face number if type==1,
            this is the edge number if type==2
            this is the node number if type==3 

    index1: this is the value index along 0th dir the face
        of interest OR edge of interest
    index2: this is the value index along 1st dir the face
        of interest
        '''
    
    # Note to interior->Faces->Edges->Nodes to minimize number of if checks

    if i>0 and i<N-1 and j>0 and j<M-1 and k>0 and k<L-1: # Interior
        return 0,None,None,None

    elif i > 0 and i < N-1 and j > 0 and j < M-1 and k == 0:   # Face 0
        return 1,0,i,j
    elif i > 0 and i < N-1 and j > 0 and j < M-1 and k == L-1: # Face 1
        return 1,1,i,j
    elif i == 0 and j > 0 and j < M-1 and k > 0 and k < L-1:   # Face 2
        return 1,2,j,k
    elif i == N-1 and j > 0 and j < M-1 and k > 0 and k < L-1: # Face 3
        return 1,3,j,k
    elif i > 0 and i < N-1 and j == 0 and k > 0 and k < L-1:   # Face 4
        return 1,4,i,k
    elif i > 0 and i < N-1 and j == M-1 and k > 0 and k < L-1: # Face 5
        return 1,5,i,k

    elif i > 0 and i < N-1 and j == 0 and k == 0:       # Edge 0
        return 2,0,i,None
    elif i > 0 and i < N-1 and j == M-1 and k == 0:     # Edge 1
        return 2,1,i,None
    elif i == 0 and j > 0 and j < M-1 and k == 0:       # Edge 2
        return 2,2,j,None
    elif i == N-1 and j > 0 and j < M-1 and k == 0:     # Edge 3
        return 2,3,j,None
    elif i > 0 and i < N-1 and j == 0 and k == L-1:     # Edge 4
        return 2,4,i,None
    elif i > 0 and i < N-1 and j == M-1 and k == L-1:   # Edge 5
        return 2,5,i,None
    elif i == 0 and j > 0 and j < M-1 and k == L-1:     # Edge 6
        return 2,6,j,None
    elif i == N-1 and j > 0 and j < M-1 and k == L-1:   # Edge 7
        return 2,7,j,None
    elif i == 0 and j == 0 and k > 0 and k < L-1:       # Edge 8
        return 2,8,k,None
    elif i == N-1 and j == 0 and k > 0 and k < L-1:     # Edge 9
        return 2,9,k,None
    elif i == 0 and j == M-1 and k > 0 and k < L-1:     # Edge 10
        return 2,10,k,None
    elif i == N-1 and j == M-1 and k > 0 and k < L-1:   # Edge 11
        return 2,11,k,None

    elif i == 0 and j == 0 and k == 0:                  # Node 0
        return 3,0,None,None
    elif i == N-1 and j == 0 and k == 0:                # Node 1
        return 3,1,None,None
    elif i == 0 and j == M-1 and k == 0:                # Node 2
        return 3,2,None,None
    elif i == N-1 and j == M-1 and k == 0:              # Node 3
        return 3,3,None,None
    elif i == 0 and j == 0 and k == L-1:                # Node 4
        return 3,4,None,None
    elif i == N-1 and j == 0 and k == L-1:              # Node 5
        return 3,5,None,None
    elif i == 0 and j == M-1 and k == L-1:              # Node 6
        return 3,6,None,None
    elif i == N-1 and j == M-1 and k == L-1:            # Node 7
        return 3,7,None,None

# --------------------------------------------------------------
#                     Node/Edge Functions
# --------------------------------------------------------------

def edgeFromNodes(n1, n2):
    '''Return the edge coorsponding to nodes n1,n2'''
    if (n1 == 0 and n2 == 1) or (n1 == 1 and n2 == 0):
        return 0
    elif (n1 == 0 and n2 == 2) or (n1 == 2 and n2 == 0):
        return 2
    elif (n1 == 3 and n2 == 1) or (n1 == 1 and n2 == 3):
        return 3
    elif (n1 == 3 and n2 == 2) or (n1 == 2 and n2 == 3):
        return 1

def edgesFromNode(n):
    ''' Return the two edges coorsponding to node n'''
    if n == 0:
        return 0,2
    if n == 1:
        return 0,3
    if n == 2:
        return 1,2
    if n == 3:
        return 1,3

def edgesFromNodeIndex(n, N, M):
    ''' Return the two edges coorsponding to node n AND return the index
of the node on the edge according to the size (N,M)'''
    if n == 0:
        return 0,2,0,0
    if n == 1:
        return 0,3,N-1,0
    if n == 2:
        return 1,2,0,M-1
    if n == 3:
        return 1,3,N-1,M-1

def nodesFromEdge(edge):
    '''Return the nodes on either edge of a standard edge'''
    if edge == 0:
        return 0,1
    elif edge == 1:
        return 2,3
    elif edge == 2:
        return 0,2
    elif edge == 3:
        return 1,3
    elif edge == 4:
        return 4,5
    elif edge == 5:
        return 6,7
    elif edge == 6:
        return 4,6
    elif edge == 7:
        return 5,7
    elif edge == 8:
        return 0,4
    elif edge == 9:
        return 1,5
    elif edge == 10:
        return 2,6
    elif edge == 11:
        return 3,7

# Volume Face/edge functions
def nodesFromFace(face):
    if face == 0:
        return [0,1,2,3]
    elif face == 1:
        return [4,5,6,7]
    elif face == 2:
        return [0,2,4,6]
    elif face == 3:
        return [1,3,5,7]
    elif face == 4:
        return [0,1,4,5]
    elif face == 5:
        return [2,3,6,7]

# --------------------------------------------------------------
#                  Knot Vector Manipulation Functions
# --------------------------------------------------------------
    
def blendKnotVectors(knot_vectors, sym):
    '''Take in a list of knot vectors and average them'''

    nVec = len(knot_vectors)
 
    if sym: # Symmetrize each knot vector first
        for i in xrange(nVec):
            cur_knot_vec = knot_vectors[i].copy()
            if mod(len(cur_knot_vec),2) == 1: #its odd
                mid = (len(cur_knot_vec) -1)/2
                beg1 = cur_knot_vec[0:mid]
                beg2 = (1-cur_knot_vec[mid+1:])[::-1]
                # Average
                beg = 0.5*(beg1+beg2)
                cur_knot_vec[0:mid] = beg
                cur_knot_vec[mid+1:] = (1-beg)[::-1]
                cur_knot_vec[mid] = 0.5
            else: # its even
                mid = len(cur_knot_vec)/2
                beg1 = cur_knot_vec[0:mid]
                beg2 = (1-cur_knot_vec[mid:])[::-1]
                beg = 0.5*(beg1+beg2)
                cur_knot_vec[0:mid] = beg
                cur_knot_vec[mid:] = (1-beg)[::-1]
            # end if
            knot_vectors[i] = cur_knot_vec
        # end for
    # end if

    # Now average them all
   
    new_knot_vec = zeros(len(knot_vectors[0]))
    for i in xrange(nVec):
        new_knot_vec += knot_vectors[i]
    # end if

    new_knot_vec /= nVec
    return new_knot_vec

class point_select(object):

    def __init__(self,type,*args,**kwargs):

        '''Initialize a control point selection class. There are several ways
        to initialize this class depending on the 'type' qualifier:

        Inputs:
        
        type: string which inidicates the initialization type:
        
        'x': Define two corners (pt1=,pt2=) on a plane parallel to the
        x=0 plane

        'y': Define two corners (pt1=,pt2=) on a plane parallel to the
        y=0 plane

        'z': Define two corners (pt1=,pt2=) on a plane parallel to the
        z=0 plane

        'quad': Define FOUR corners (pt1=,pt2=,pt3=,pt4=) in a
        COUNTER-CLOCKWISE orientation 

        '''
        
        if type == 'x' or type == 'y' or type == 'z':
            assert 'pt1' in kwargs and 'pt2' in kwargs,'Error:, two points \
must be specified with initialization type x,y, or z. Points are specified \
with kwargs pt1=[x1,y1,z1],pt2=[x2,y2,z2]'

        elif type == 'quad':
            assert 'pt1' in kwargs and 'pt2' in kwargs and 'pt3' in kwargs \
                and 'pt4' in kwargs,'Error:, four points \
must be specified with initialization type quad. Points are specified \
with kwargs pt1=[x1,y1,z1],pt2=[x2,y2,z2],pt3=[x3,y3,z3],pt4=[x4,y4,z4]'
        
        # end if
        corners = zeros([4,3])
        if type == 'x':
            corners[0] = kwargs['pt1']

            corners[1][1] = kwargs['pt2'][1]
            corners[1][2] = kwargs['pt1'][2]

            corners[2][1] = kwargs['pt1'][1]
            corners[2][2] = kwargs['pt2'][2]

            corners[3] = kwargs['pt2']

            corners[:,0] = 0.5*(kwargs['pt1'][0] + kwargs['pt2'][0])

        elif type == 'y':
            corners[0] = kwargs['pt1']

            corners[1][0] = kwargs['pt2'][0]
            corners[1][2] = kwargs['pt1'][2]

            corners[2][0] = kwargs['pt1'][0]
            corners[2][2] = kwargs['pt2'][2]

            corners[3] = kwargs['pt2']

            corners[:,1] = 0.5*(kwargs['pt1'][1] + kwargs['pt2'][1])

        elif type == 'z':
            corners[0] = kwargs['pt1']

            corners[1][0] = kwargs['pt2'][0]
            corners[1][1] = kwargs['pt1'][1]

            corners[2][0] = kwargs['pt1'][0]
            corners[2][1] = kwargs['pt2'][1]

            corners[3] = kwargs['pt2']

            corners[:,2] = 0.5*(kwargs['pt1'][2] + kwargs['pt2'][2])

        elif type == 'quad':
            corners[0] = kwargs['pt1']
            corners[1] = kwargs['pt2']
            corners[2] = kwargs['pt4'] # Note the switch here from CC orientation
            corners[3] = kwargs['pt3']
        # end if

        X = reshape(corners,[2,2,3])

        self.box=pySpline.bilinear_surface(X=X)
        self.type = type

        return

    def getPoints(self, points):

        '''Take in a list of points and return the ones that statify
        the point select class.'''
        pt_list = []
        ind_list = []
        for i in xrange(len(points)):
            u0,v0,D,converged = self.box.projectPoint(points[i])
            if u0>0 and u0<1 and v0>0 and v0<1: #Its Inside
                pt_list.append(points[i])
                ind_list.append(i)
            # end if
        # end for

        return pt_list,ind_list

class topology(object):
    '''
    The base topology class from which the BlockTopology,
    SurfaceTology and CuveTopology classes inherit from
    
    The topology object contains all the info required for the block
    topology (most complex) however, simpiler topologies are handled
    accordingly.

    Class Attributes:
        nVol : The number of volumes in the topology (may be 0)
        nFace: The number of unique faces on the topology (may be 0)
        nEdge: The number of uniuqe edges on the topology 
        nNode: The number of unique nodes on the topology

        nEnt: The number of "entities" in the topology class. This may
        be curves, faces or volumes

        mNodeEnt: The number of NODES per entity. For curves it's 2, for
        surfaces 4 and for volumes 8.

        mEdgeEnt: The number of EDGES per entity. For curves it's 1,
        for surfaces, 4 and for volumes, 12

        mFaceEnt: The number of faces per entity. For curves its's 0,
        for surfaces, 1 and for volumes,6

        mVolEnt: The number of volumes per entity. For curves it's 0,
        for surfaces, 0 and for volumnes, 1

        node_link: The array of size nEnt x mNodesEnt which points
                   to the node for each entity
        edge_link: The array of size nEnt x mEdgeEnt which points
                   to the edge for each edge of entity
        face_link: The array of size nEnt x mFaceEnt which points to 
                   the face of each face on an entity

        edge_dir:  The array of size nEnt x mEdgeEnt which detrmines
                   if the intrinsic direction of this edge is
                   opposite of the direction as recorded in the
                   edge list. edge_dir[entity#][#] = 1 means same direction;
                   -1 is opposite direction.
                  
        face_dir:  The array of size nFace x 6 which determines the 
                   intrinsic direction of this face. It is one of 0->7
                   
        l_index:   The local->global list of arrays for each volue
        g_index:   The global->local list points for the entire topology
        edges:     The list of edge objects defining the topology
        simple    : A flag to determine of this is a "simple" topology which means
                   there are NO degernate Edges, NO multiple edges sharing the same
                   nodes and NO edges which loop back and have the same nodes
                   MUST BE SIMPLE
    '''

    def __init__(self):
        # Not sure what should go here...
        return
    def _calcDGs(self,edges,edge_link,edge_link_sorted,edge_link_ind):

        dg_counter = -1
        for i in xrange(self.nEdge):
            if edges[i][2] == -1: # Not set yet
                dg_counter += 1
                edges[i][2] = dg_counter
                self._addDGEdge(i,edges,edge_link,edge_link_sorted,edge_link_ind)
            # end if
        # end for
        self.nDG = dg_counter + 1
   
    def _addDGEdge(self,i,edges,edge_link,edge_link_sorted,edge_link_ind):
        left  = edge_link_sorted.searchsorted(i,side='left')
        right = edge_link_sorted.searchsorted(i,side='right')
        res   = edge_link_ind[slice(left,right)]

        for j in xrange(len(res)):
            ient = res[j]/self.mEdgeEnt #Integer Division
            iedge = mod(res[j],self.mEdgeEnt)

            pEdges = self._getParallelEdges(iedge)
            oppositeEdges = []
            for iii in xrange(len(pEdges)):
                oppositeEdges.append(edge_link[self.mEdgeEnt*ient + pEdges[iii]])
            
            for ii in xrange(len(pEdges)):
                if edges[oppositeEdges[ii]][2] == -1:
                    edges[oppositeEdges[ii]][2] = edges[i][2]
                    if not edges[oppositeEdges[ii]][0] == edges[oppositeEdges[ii]][1]:
                        self._addDGEdge(oppositeEdges[ii],edges,edge_link,edge_link_sorted,edge_link_ind)
                # end if
            # end if
        # end for

    def _getParallelEdges(self, iedge):
        '''Return parallel edges for surfaces and volumes'''

        if self.topo_type == 'surface':
            if iedge == 0: return [1]
            if iedge == 1: return [0]
            if iedge == 2: return [3]
            if iedge == 3: return [2]

        if self.topo_type == 'volume':
            if iedge == 0: return [1,4,5]
            if iedge == 1: return [0,4,5]
            if iedge == 2: return [3,6,7]
            if iedge == 3: return [2,6,7]
            if iedge == 4: return [0,1,5]
            if iedge == 5: return [0,1,4]
            if iedge == 6: return [2,3,7]
            if iedge == 7: return [2,3,6]
            if iedge == 8: return [9,10,11]
            if iedge == 9: return [8,10,11]
            if iedge == 10: return [8,9,11]
            if iedge == 11: return [8,9,10]
        if self.topo_type == 'curve':
            return None

    def printConnectivity(self):
        '''Print the Edge Connectivity to the screen'''

        mpiPrint('------------------------------------------------------------------------')
        mpiPrint('%4d  %4d  %4d  %4d  %4d '%(self.nNode,self.nEdge,self.nFace,self.nVol,self.nDG))
        N_list = self._getDGList()
        mpiPrint('Design Group | Number')
        for i in xrange(self.nDG):
            mpiPrint('%5d        | %5d       '%(i,N_list[i]))
        # end for

        # Always have edges!
        mpiPrint('Edge Number    |   n0  |   n1  |  Cont | Degen | Intsct|   DG   |  N     |')
        for i in xrange(len(self.edges)):
            self.edges[i].write_info(i,sys.stdout)
        # end for

        print '%9s Num |'%(self.topo_type),
        for i in xrange(self.mNodeEnt):
            print ' n%2d|'%(i),
        for i in xrange(self.mEdgeEnt):
            print ' e%2d|'%(i),
        print ' ' # Get New line
            
        for i in xrange(self.nEnt):
            print ' %5d        |'%(i),
            for j in xrange(self.mNodeEnt):
                print '%4d|'%self.node_link[i][j],
            # end for
            for j in xrange(self.mEdgeEnt):
                print '%4d|'%(self.edge_link[i][j]*self.edge_dir[i][j]),
            # end for
            print ' '
        # end for
        print('------------------------------------------------------------------------')

        if self.topo_type == 'volume':
            mpiPrint('Vol Number | f0 | f1 | f2 | f3 | f4 | f5 |f0dir|f1dir|f2dir|f3dir|f4dir|f5dir|')
            for i in xrange(self.nVol):
                mpiPrint(' %5d     |%4d|%4d|%4d|%4d|%4d|%4d|%5d|%5d|%5d|%5d|%5d|%5d|'\
                             %(i,self.face_link[i][0],self.face_link[i][1],
                               self.face_link[i][2],self.face_link[i][3],
                               self.face_link[i][3],self.face_link[i][5],
                               self.face_dir[i][0],self.face_dir[i][1],
                               self.face_dir[i][2],self.face_dir[i][3],
                               self.face_dir[i][4],self.face_dir[i][5])) 
            # end for
        # end if
        return

    def writeConnectivity(self, file_name):
        '''Write the full edge connectivity to a file file_name'''
        f = open(file_name,'w')
        f.write('%4d  %4d  %4d   %4d  %4d\n'%(self.nNode,self.nEdge,self.nFace,self.nVol,self.nDG))
        f.write('Design Group |  Number\n')
        # Write out the design groups and their number parameter
        N_list = self._getDGList()
        for i in xrange(self.nDG):
            f.write('%5d        | %5d       \n'%(i,N_list[i]))
        # end for

        f.write('Edge Number    |   n0  |   n1  |  Cont | Degen | Intsct|   DG   |  N     |\n')
        for i in xrange(len(self.edges)):
            self.edges[i].write_info(i,f)
        # end for

        f.write('%9s Num |'%(self.topo_type))
        for i in xrange(self.mNodeEnt):
            f.write(' n%2d|'%(i))
        for i in xrange(self.mEdgeEnt):
            f.write(' e%2d|'%(i))
        f.write('\n')
            
        for i in xrange(self.nEnt):
            f.write(' %5d        |'%(i))
            for j in xrange(self.mNodeEnt):
                f.write('%4d|'%self.node_link[i][j])
            # end for
            for j in xrange(self.mEdgeEnt):
                f.write('%4d|'%(self.edge_link[i][j]*self.edge_dir[i][j]))
            # end for
            f.write('\n')
        # end for

        if self.topo_type == 'volume':

            f.write('Vol Number | f0 | f1 | f2 | f3 | f4 | f5 |f0dir|f1dir|f2dir|f3dir|f4dir|f5dir|\n')
            for i in xrange(self.nVol):
                f.write(' %5d     |%4d|%4d|%4d|%4d|%4d|%4d|%5d|%5d|%5d|%5d|%5d|%5d|\n'\
                            %(i,self.face_link[i][0],self.face_link[i][1],
                              self.face_link[i][2],self.face_link[i][3],
                              self.face_link[i][4],self.face_link[i][5],
                              self.face_dir[i][0],self.face_dir[i][1],
                              self.face_dir[i][2],self.face_dir[i][3],
                              self.face_dir[i][4],self.face_dir[i][5])) 
  
        f.close()
        
        return

    def readConnectivity(self, file_name):
        '''Read the full edge connectivity from a file file_name'''
        # We must be able to populate the following:
        #nNode,nEdge,nFace,nVol,node_link,edge_link,face_link,edge_dir,face_dir

        f = open(file_name,'r')
        aux = string.split(f.readline())
        self.nNode = int(aux[0])
        self.nEdge = int(aux[1])
        self.nFace = int(aux[2])
        self.nVol  = int(aux[3])
        self.nDG   = int(aux[4])
        self.edges = []
        
        if self.topo_type == 'volume':
            self.nEnt = self.nVol
        elif self.topo_type == 'surface':
            self.nEnt = self.nFace
        elif self.topo_type == 'curve':
            self.nEnt = self.nEdge
        # end if

        f.readline() # This is the header line so ignore

        N_list = zeros(self.nDG,'intc')
        for i in xrange(self.nDG):
            aux = string.split(f.readline(),'|')
            N_list[i] = int(aux[1])
        # end for

        f.readline() # Second Header line

        for i in xrange(self.nEdge):
            aux = string.split(f.readline(),'|')
            self.edges.append(edge(int(aux[1]),int(aux[2]),int(aux[3]),
                                       int(aux[4]),int(aux[5]),int(aux[6]),int(aux[7])))
        # end for
        f.readline() # This is the third header line so ignore

        self.edge_link = zeros((self.nEnt,self.mEdgeEnt),'intc')
        self.node_link = zeros((self.nEnt,self.mNodeEnt),'intc')
        self.edge_dir  = zeros((self.nEnt,self.mEdgeEnt),'intc')
        
        for i in xrange(self.nEnt):
            aux = string.split(f.readline(),'|')
            for j in xrange(self.mNodeEnt):
                self.node_link[i][j] = int(aux[j+1])
            for j in xrange(self.mEdgeEnt):
                self.edge_dir[i][j]  = sign(int(aux[j+1+self.mNodeEnt]))
                self.edge_link[i][j] = int(aux[j+1+self.mNodeEnt])*self.edge_dir[i][j]

            # end for
        # end for

        if self.topo_type == 'volume':
            f.readline() # This the fourth header line so ignore

            self.face_link = zeros((self.nVol,6),'intc')
            self.face_dir  = zeros((self.nVol,6),'intc')
            for ivol in xrange(self.nVol):
                aux = string.split(f.readline(),'|')
                self.face_link[ivol] = [int(aux[i]) for i in xrange(1,7)]
                self.face_dir[ivol]  = [int(aux[i]) for i in xrange(7,13)]
            # end for
        # end if
      
        # Set the N_list to the edges
        for iedge in xrange(self.nEdge):
            self.edges[iedge].N = N_list[self.edges[iedge].dg]
            
        return

    def _getDGList(self):
        '''After calcGlobalNumbering is called with the size
        parameters, we can now produce a list of length ndg with the
        each entry coorsponing to the number N associated with that DG'''

        # This can be run in linear time...just loop over each edge
        # and add to dg list
        N_list = zeros(self.nDG,'intc')
        for iedge in xrange(self.nEdge):
            N_list[self.edges[iedge].dg] = self.edges[iedge].N
        # end for
            
        return N_list

class CurveTopology(topology):
    '''
    See topology class for more information
    '''
    def __init__(self, coords=None, file=None, node_tol=1e-4):
        '''Initialize the class with data required to compute the topology'''
        topology.__init__(self)
        self.mNodeEnt = 2
        self.mEdgeEnt = 1
        self.mfaceEnt = 0
        self.mVolEnt  = 0
        self.nVol = 0
        self.nFace = 0
        self.topo_type = 'curve'
        self.g_index = None
        self.l_index = None
        self.nGlobal = None
        if file != None:
            self.readConnectivity(file)
            return
        # end if
        
        self.edges = None
        self.simple = True

        # Must have curves
        # Get the end points of each curve

        self.nEdge = len(coords)
        coords = coords.reshape((self.nEdge*2,3))
        node_list,self.node_link = pointReduce(coords)
        self.node_link = self.node_link.reshape((self.nEdge,2))
        self.nNode = len(node_list)
        self.edges = []
        self.edge_link = zeros((self.nEdge,1),'intc')
        for iedge in xrange(self.nEdge):
            self.edge_link[iedge][0] = iedge
        # end for
        self.edge_dir  = zeros((self.nEdge,1),'intc')

        for iedge in xrange(self.nEdge):
            n1 = self.node_link[iedge][0]
            n2 = self.node_link[iedge][1]
            if n1<n2:
                self.edges.append(edge(n1,n2,0,0,0,iedge,2))
                self.edge_dir[iedge][0] = 1
            else:
                self.edges.append(edge(n2,n1,0,0,0,iedge,2))
                self.edge_dir[iedge][1] = -1
            # end if
        # end for
        self.nDG = self.nEdge
        self.nEnt = self.nEdge
        return

    def calcGlobalNumbering(self, sizes, curve_list=None):
        '''Internal function to calculate the global/local numbering for each curve'''
        for i in xrange(len(sizes)):
            self.edges[self.edge_link[i][0]].N = sizes[i]
        # end for
        if curve_list == None:
            curve_list = range(self.nEdge)
        # end if
        
        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        l_index = []

        assert len(sizes) == len(curve_list),'Error: The list of sizes and \
the list of surfaces must be the same length'

        # Assign unique numbers to the corners -> Corners are indexed sequentially
        node_index = arange(self.nNode)
        counter = len(node_index)
        edge_index = [ [] for i in xrange(len(self.edges))]
     
        # Assign unique numbers to the edges

        for ii in xrange(len(curve_list)):
            cur_size = [sizes[ii]]
            icurve = curve_list[ii]
            for iedge in xrange(1):
                edge = self.edge_link[ii][iedge]
                    
                if edge_index[edge] == []:# Not added yet
                    for jj in xrange(cur_size[iedge]-2):
                        edge_index[edge].append(counter)
                        counter += 1
                    # end for
                # end if
            # end for
        # end for

        g_index = [ [] for i in xrange(counter)] # We must add [] for each global node

        for ii in xrange(len(curve_list)):
            icurve = curve_list[ii]
            N = sizes[ii]
            l_index.append(-1*ones(N,'intc'))

            for i in xrange(N):
                type,node = indexPosition1D(i,N)

                if type == 1: # Node
                    cur_node = self.node_link[ii][node]
                    l_index[ii][i] = node_index[cur_node]
                    g_index[node_index[cur_node]].append([icurve,i])
                else:
                    if self.edge_dir[ii][0] == -1:
                        cur_index = edge_index[self.edge_link[ii][0]][N-i-2]
                    else:
                        cur_index = edge_index[self.edge_link[ii][0]][i-1]
                    # end if
                    l_index[ii][i] = cur_index
                    g_index[cur_index].append([icurve,i])
                # end if
            # end for
        # end for
        self.nGlobal = len(g_index)
        self.g_index = g_index
        self.l_index = l_index
        
        return 


class SurfaceTopology(topology):
    '''
    See topology class for more information
    '''
    def __init__(self, coords=None, face_con=None, file=None, node_tol=1e-4,
                 edge_tol=1e-4):
        '''Initialize the class with data required to compute the topology'''
        topology.__init__(self)
        self.mNodeEnt = 4
        self.mEdgeEnt = 4
        self.mfaceEnt = 1
        self.mVolEnt  = 0
        self.nVol = 0
        self.topo_type = 'surface'
        self.g_index = None
        self.l_index = None
        self.nGlobal = None
        if file != None:
            self.readConnectivity(file)
            return
        # end if
        
        self.edges = None
        self.face_index = None
        self.simple = False

        if not face_con == None: 
            face_con = array(face_con)
            midpoints = None
            self.nFace = len(face_con)
            self.nEnt = self.nFace
            self.simple = True
            # Check to make sure nodes are sequential
            self.nNode = len(unique(face_con.flatten()))
            if self.nNode != max(face_con.flatten())+1:
                mpiPrint('Error: The Node numbering in faceCon is not sequential. There are \
missing nodes')
                sys.exit(1)
            # end if
            
            edges = []
            edge_hash = []
            for iface in xrange(self.nFace):
                #             n1                ,n2               ,dg,n,degen
                edges.append([face_con[iface][0],face_con[iface][1],-1,0,0])
                edges.append([face_con[iface][2],face_con[iface][3],-1,0,0])
                edges.append([face_con[iface][0],face_con[iface][2],-1,0,0])
                edges.append([face_con[iface][1],face_con[iface][3],-1,0,0])
            # end for
            edge_dir = ones(len(edges),'intc')
            for iedge in xrange(self.nFace*4):
                if edges[iedge][0] > edges[iedge][1]:
                    temp = edges[iedge][0]
                    edges[iedge][0] = edges[iedge][1]
                    edges[iedge][1] = temp
                    edge_dir[iedge] = -1
                # end if
                edge_hash.append(edges[iedge][0]*4*self.nFace + edges[iedge][1])
            # end for

            edges,edge_link = unique_index(edges,edge_hash)

            self.nEdge = len(edges)
            self.edge_link = array(edge_link).reshape((self.nFace,4))
            self.node_link = array(face_con)
            self.edge_dir  = array(edge_dir).reshape((self.nFace,4))

            edge_link_sorted = sort(edge_link)
            edge_link_ind    = argsort(edge_link)

        elif not coords == None:
            self.nFace = len(coords)
            self.nEnt  = self.nFace
            # We can use the pointReduce algorithim on the nodes
            node_list,node_link = pointReduce(coords[:,0:4,:].reshape((self.nFace*4,3)))
            node_link = node_link.reshape((self.nFace,4))

            # Next Calculate the EDGE connectivity. -- This is Still Brute Force

            edges = []
            midpoints = []
            edge_link = -1*ones(self.nFace*4,'intc')
            edge_dir  = zeros((self.nFace,4),'intc')

            for iface in xrange(self.nFace):
                for iedge in xrange(4):
                    n1,n2 = nodesFromEdge(iedge) # nodesFromEdge in geo_utils
                    n1 = node_link[iface][n1]
                    n2 = node_link[iface][n2] 
                    midpoint = coords[iface][iedge + 4]
                    if len(edges) == 0:
                        edges.append([n1,n2,-1,0,0])
                        midpoints.append(midpoint)
                        edge_link[4*iface + iedge] = 0
                        edge_dir [iface][iedge] = 1
                    else:
                        found_it = False
                        for i in xrange(len(edges)):
                            if [n1,n2] == edges[i][0:2] and n1 != n2:
                                if e_dist(midpoint,midpoints[i]) < edge_tol:
                                    edge_link[4*iface + iedge] = i
                                    edge_dir [iface][iedge] = 1
                                    found_it = True
                                # end if
                            elif [n2,n1] == edges[i][0:2] and n1 != n2:
                                if e_dist(midpoint,midpoints[i]) < edge_tol:
                                    edge_link[4*iface + iedge] = i
                                    edge_dir[iface][iedge] = -1
                                    found_it = True
                                # end if
                            # end if
                        # end for

                        # We went all the way though the list so add it at end and return index
                        if not found_it:
                            edges.append([n1,n2,-1,0,0])
                            midpoints.append(midpoint)
                            edge_link[4*iface + iedge] = i+1
                            edge_dir [iface][iedge] = 1
                    # end if
                # end for
            # end for

            self.nEdge = len(edges)
            self.edge_link = array(edge_link).reshape((self.nFace,4))
            self.node_link = array(node_link)
            self.nNode = len(unique(self.node_link.flatten()))
            self.edge_dir = edge_dir

            edge_link_sorted = sort(edge_link.flatten())
            edge_link_ind    = argsort(edge_link.flatten())

        # end if
            
        # Next Calculate the Design Group Information
        self._calcDGs(edges,edge_link,edge_link_sorted,edge_link_ind)

        # Set the edge ojects
        self.edges = []
        for i in xrange(self.nEdge): # Create the edge objects
            if midpoints: # If they exist
                if edges[i][0] == edges[i][1] and \
                        e_dist(midpoints[i],node_list[edges[i][0]]) < node_tol:
                    self.edges.append(edge(edges[i][0],edges[i][1],0,1,0,edges[i][2],edges[i][3]))
                else:
                    self.edges.append(edge(edges[i][0],edges[i][1],0,0,0,edges[i][2],edges[i][3]))
                # end if
            else:
                self.edges.append(edge(edges[i][0],edges[i][1],0,0,0,edges[i][2],edges[i][3]))
            # end if
        # end for

        return

    def calcGlobalNumbering(self, sizes, surface_list=None):
        '''Internal function to calculate the global/local numbering for each surface'''
        for i in xrange(len(sizes)):
            self.edges[self.edge_link[i][0]].N = sizes[i][0]
            self.edges[self.edge_link[i][1]].N = sizes[i][0]
            self.edges[self.edge_link[i][2]].N = sizes[i][1]
            self.edges[self.edge_link[i][3]].N = sizes[i][1]

        if surface_list == None:
            surface_list = range(0,self.nFace)
        # end if
        
        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        g_index = []
        l_index = []

        assert len(sizes) == len(surface_list),'Error: The list of sizes and \
the list of surfaces must be the same length'

        # Assign unique numbers to the corners -> Corners are indexed sequentially
        node_index = arange(self.nNode)
        counter = len(node_index)
        edge_index = [ [] for i in xrange(len(self.edges))]
     
        # Assign unique numbers to the edges

        for ii in xrange(len(surface_list)):
            cur_size = [sizes[ii][0],sizes[ii][0],sizes[ii][1],sizes[ii][1]]
            isurf = surface_list[ii]
            for iedge in xrange(4):
                edge = self.edge_link[ii][iedge]
                    
                if edge_index[edge] == []:# Not added yet
                    if self.edges[edge].degen == 1:
                        # Get the counter value for this "node"
                        index = node_index[self.edges[edge].n1]
                        for jj in xrange(cur_size[iedge]-2):
                            edge_index[edge].append(index)
                        # end for
                    else:
                        for jj in xrange(cur_size[iedge]-2):
                            edge_index[edge].append(counter)
                            counter += 1
                        # end for
                    # end if
                # end if
            # end for
        # end for
     
        g_index = [ [] for i in xrange(counter)] # We must add [] for each global node
        l_index = []

        # Now actually fill everything up

        for ii in xrange(len(surface_list)):
            isurf = surface_list[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            l_index.append(-1*ones((N,M),'intc'))

            for i in xrange(N):
                for j in xrange(M):
                    
                    type,edge,node,index = indexPosition2D(i,j,N,M)

                    if type == 0:           # Interior
                        l_index[ii][i,j] = counter
                        g_index.append([[isurf,i,j]])
                        counter += 1
                    elif type == 1:         # Edge
                       
                        if edge in [0,1]:
                            if self.edge_dir[ii][edge] == -1: # Its a reverse dir
                                cur_index = edge_index[self.edge_link[ii][edge]][N-i-2]
                            else:  
                                cur_index = edge_index[self.edge_link[ii][edge]][i-1]
                            # end if
                        else: # edge in [2,3]
                            if self.edge_dir[ii][edge] == -1: # Its a reverse dir
                                cur_index = edge_index[self.edge_link[ii][edge]][M-j-2]
                            else:  
                                cur_index = edge_index[self.edge_link[ii][edge]][j-1]
                            # end if
                        # end if
                        l_index[ii][i,j] = cur_index
                        g_index[cur_index].append([isurf,i,j])
                            
                    else:                  # Node
                        cur_node = self.node_link[ii][node]
                        l_index[ii][i,j] = node_index[cur_node]
                        g_index[node_index[cur_node]].append([isurf,i,j])
                    # end for
                # end for (j)
            # end for (i)
        # end for (ii)

        # Reorder the indices with a greedy scheme

        new_indices = zeros(len(g_index),'intc')
        new_indices[:] = -1
        new_g_index = [[] for i in xrange(len(g_index))]
        counter = 0

        # Re-order the l_index
        for ii in xrange(len(surface_list)):
            isurf = surface_list[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            for i in xrange(N):
                for j in xrange(M):
                    if new_indices[l_index[ii][i,j]] == -1:
                        new_indices[l_index[ii][i,j]] = counter
                        l_index[ii][i,j] = counter 
                        counter += 1
                    else:
                        l_index[ii][i,j] = new_indices[l_index[ii][i,j]]
                    # end if
                # end for
            # end for
        # end for
       
        # Re-order the g_index
        for ii in xrange(len(g_index)):
            isurf = g_index[ii][0][0]
            i     = g_index[ii][0][1]
            j     = g_index[ii][0][2]
            pt = l_index[isurf][i,j]
            new_g_index[pt] = g_index[ii]
            # end for
        # end for
            
        self.nGlobal = len(g_index)
        self.g_index = new_g_index
        self.l_index = l_index
        
        return 

    def getSurfaceFromEdge(self, edge):
        '''Determine the surfaces and their edge_link index that points to edge iedge'''
        # Its not efficient but it works - scales with Nface not constant
        surfaces = []
        for isurf in xrange(self.nFace):
            for iedge in xrange(4):
                if self.edge_link[isurf][iedge] == edge:
                    surfaces.append([isurf,iedge])
                # end if
            # end for
        # end for

        return surfaces

class BlockTopology(topology):
    '''
    See Topology base class for more information
    '''

    def __init__(self,coords=None, node_tol=1e-4, edge_tol=1e-4, file=None):
        '''Initialize the class with data required to compute the topology'''
        
        topology.__init__(self)
        self.mNodeEnt = 8
        self.mEdgeEnt = 12
        self.mFaceEnt = 6
        self.mVolEnt  = 1
        self.topo_type = 'volume'
        self.g_index = None
        self.l_index = None
        self.nGlobal = None
        if file != None:
            self.readConnectivity(file)
            return
        # end if

        coords = atleast_2d(coords)
        nVol = len(coords)
        
        if coords.shape[1] == 8: # Just the corners are given
            self.simple = True
            # This is initialized with a list of coordiantes -> Corners
            # Point reduce them
            un,node_link = pointReduceBruteForce(coords.reshape((8*nVol,3)))
            node_link = node_link.reshape((nVol,8))

            # We can now generate vol_con AND face_con
            vol_con = []
            face_con = []
            for ivol in xrange(nVol):
                vol_con.append(zeros(8,'intc'))
                for icorner in xrange(8):
                    vol_con[-1][icorner] = node_link[ivol][icorner]
                # end for
                face_con.append([node_link[ivol][0],node_link[ivol][1],
                                 node_link[ivol][2],node_link[ivol][3]])
                
                face_con.append([node_link[ivol][4],node_link[ivol][5],
                                 node_link[ivol][6],node_link[ivol][7]])

                face_con.append([node_link[ivol][0],node_link[ivol][2],
                                 node_link[ivol][4],node_link[ivol][6]])
                
                face_con.append([node_link[ivol][1],node_link[ivol][3],
                                 node_link[ivol][5],node_link[ivol][7]])
                
                face_con.append([node_link[ivol][0],node_link[ivol][1],
                                 node_link[ivol][4],node_link[ivol][5]])
                
                face_con.append([node_link[ivol][2],node_link[ivol][3],
                                 node_link[ivol][6],node_link[ivol][7]])
            # end for

            # Uniqify the Faces 
            face_hash = []
            for iface in xrange(nVol * 6):
                temp = sorted(face_con[iface])
                face_hash.append((temp[0]*6*nVol*3 + temp[1]*6*nVol*2 +
                              temp[2]*6*nVol*1 + temp[3]))
                
            uf,face_link = unique_index(face_hash) # Only need the lengh of uf and face_link
            face_link = array(face_link).reshape((nVol,6))
            nFace = len(uf)
            used = zeros(nFace,'bool')
            reduced_face_con = zeros((nFace,4),'intc')
            for ivol in xrange(nVol):
                for iface in xrange(6):
                    if used[face_link[ivol][iface]] == False:
                        used[face_link[ivol][iface]] = True
                        reduced_face_con[face_link[ivol][iface]] = \
                            face_con[ivol*6+iface]
                    # end if
                # end for

            # Now get the directions for faces
            face_dir = zeros((nVol,6),'intc')
            for ivol in xrange(nVol):
                for iface in xrange(6):
                    face_dir[ivol][iface] = faceOrientation(reduced_face_con[face_link[ivol][iface]],
                                            face_con[ivol*6+iface])
                # end for
            # end for

            # Uniqify the Edges - Get the edge_hash
            edges = []
            edge_hash = []
            for ivol in xrange(nVol):
                #                 #             n1                ,n2               ,dg,n,degen
                # k = 0 plane
                edges.append([vol_con[ivol][0],vol_con[ivol][1],-1,0,0])
                edges.append([vol_con[ivol][2],vol_con[ivol][3],-1,0,0])
                edges.append([vol_con[ivol][0],vol_con[ivol][2],-1,0,0])
                edges.append([vol_con[ivol][1],vol_con[ivol][3],-1,0,0])

                # k = max plane
                edges.append([vol_con[ivol][4],vol_con[ivol][5],-1,0,0])
                edges.append([vol_con[ivol][6],vol_con[ivol][7],-1,0,0])
                edges.append([vol_con[ivol][4],vol_con[ivol][6],-1,0,0])
                edges.append([vol_con[ivol][5],vol_con[ivol][7],-1,0,0])

                # Vertical ones
                edges.append([vol_con[ivol][0],vol_con[ivol][4],-1,0,0])
                edges.append([vol_con[ivol][1],vol_con[ivol][5],-1,0,0])
                edges.append([vol_con[ivol][2],vol_con[ivol][6],-1,0,0])
                edges.append([vol_con[ivol][3],vol_con[ivol][7],-1,0,0])
            # end for

            edge_dir = ones(len(edges),'intc')
            for iedge in xrange(len(edges)):
                if edges[iedge][0] > edges[iedge][1]:
                    temp = edges[iedge][0]
                    edges[iedge][0] = edges[iedge][1]
                    edges[iedge][1] = temp
                    edge_dir[iedge] = -1
                # end if
                edge_hash.append(edges[iedge][0]*12*nVol + edges[iedge][1])
            # end for

            ue,edge_link = unique_index(edges,edge_hash)

        else:
            # Now it is much more complex, since the nodes no longer
            # uniqely define edges, and in general, faces are not
            # uniqely defined by edges.

            # Coords is now assumed to be of size nvol,(8 + 12 + 6),3
            #                 8 corners + 12 edge midpoints + 6 face midpoints

            un,node_link = pointReduceBruteForce(coords[:,0:8,:].reshape((nVol*8,3)))
            node_link = node_link.reshape((nVol,8))
            # Next Calculate the EDGE connectivity. -- This is Still Brute Force

            ue = []
            midpoints = []
            edge_link = -1*ones(nVol*12,'intc')
            edge_dir  = zeros((nVol,12),'intc')

            for ivol in xrange(nVol):
                for iedge in xrange(12):
                    n1,n2 = nodesFromEdge(iedge)

                    n1 = node_link[ivol][n1]
                    n2 = node_link[ivol][n2]
                    midpoint = coords[ivol][iedge + 8]

                    if len(ue) == 0:
                        ue.append([n1,n2,-1,0,0])
                        midpoints.append(midpoint)
                        edge_link[12*ivol + iedge] = 0
                        edge_dir [ivol][iedge] = 1
                    else:
                        found_it = False
                        for i in xrange(len(ue)):
                            if [n1,n2] == ue[i][0:2]:
                                if e_dist(midpoint,midpoints[i]) < edge_tol:
                                    edge_link[12*ivol + iedge] = i
                                    edge_dir [ivol][iedge] = 1
                                    found_it = True
                                # end if
                            elif [n2,n1] == ue[i][0:2]:
                                if e_dist(midpoint,midpoints[i]) < edge_tol:
                                    edge_link[12*ivol + iedge] = i
                                    edge_dir[ivol][iedge] = -1
                                    found_it = True
                                # end if
                            # end if
                        # end for

                        # We went all the way though the list so add it at end and return index
                        if not found_it:
                            ue.append([n1,n2,-1,0,0])
                            midpoints.append(midpoint)
                            edge_link[12*ivol + iedge] = i+1
                            edge_dir [ivol][iedge] = 1
                    # end if
                # end for
            # end for
      
            # Next Calculate the FACE connectivity. -- This is Still Brute Force

            uf = []
            midpoints = []
            face_link = -1*ones(nVol*6,'intc')
            face_dir  = zeros((nVol,6),'intc')

            for ivol in xrange(nVol):
                for iface in xrange(6):
                    n1,n2,n3,n4 = nodesFromFace(iface)
                    n1 = node_link[ivol][n1]
                    n2 = node_link[ivol][n2] 
                    n3 = node_link[ivol][n3]
                    n4 = node_link[ivol][n4] 

                    midpoint = coords[ivol][iface + 8 + 12]

                    if len(uf) == 0:
                        uf.append([n1,n2,n3,n4])
                        midpoints.append(midpoint)
                        face_link[6*ivol + iface] = 0
                        face_dir [ivol][iface] = 0
                    else:
                        found_it = False
                        for i in xrange(len(uf)):
                            if sorted([n1,n2,n3,n4]) == sorted(uf[i]):
                                # We have the same face, but not not necessairly the same orientation

                                if e_dist(midpoint,midpoints[i]) < edge_tol:
                                    face_link[6*ivol + iface] = i
                                    face_dir[ivol][iface] = faceOrientation(uf[i],[n1,n2,n3,n4])
                                    found_it = True
                                # end if
                            # end if
                        # end for

                        # We went all the way though the list so add it at end and return index
                        if not found_it:
                            uf.append([n1,n2,n3,n4])
                            midpoints.append(midpoint)
                            face_link[6*ivol + iface] = i+1
                            face_dir [ivol][iface] = 0
                    # end if
                # end for
            # end for
            face_link = face_link.reshape([nVol,6])
        # end if 

        # --------- Set the Requried Data for this class ------------
        self.nNode = len(un) # Done
        self.nEdge = len(ue) # Done
        self.nFace = len(uf) # Done
        self.nVol  = len(coords) # Done
        self.nEnt  = self.nVol # Done

        self.node_link = node_link # Done
        self.edge_link = array(edge_link).reshape((nVol,12)) # Done
        self.face_link = face_link # Done

        self.edge_dir  = array(edge_dir).reshape((nVol,12))  # Done
        self.face_dir  = face_dir # Done
        
       
        # ------------------------------------------------------------

        # Next Calculate the Design Group Information
        edge_link_sorted = sort(edge_link)
        edge_link_ind    = argsort(edge_link)

        self._calcDGs(ue,edge_link,edge_link_sorted,edge_link_ind)

        # Set the edge ojects
        self.edges = []
        for i in xrange(self.nEdge): # Create the edge objects
            self.edges.append(edge(ue[i][0],ue[i][1],0,0,0,ue[i][2],ue[i][3]))
        # end for

        return

    def calcGlobalNumbering(self, sizes=None, volume_list=None):
        '''Internal function to calculate the global/local numbering for each surface'''

        if sizes != None:
            for i in xrange(len(sizes)):
                self.edges[self.edge_link[i][0]].N = sizes[i][0]
                self.edges[self.edge_link[i][1]].N = sizes[i][0]
                self.edges[self.edge_link[i][4]].N = sizes[i][0]
                self.edges[self.edge_link[i][5]].N = sizes[i][0]

                self.edges[self.edge_link[i][2]].N = sizes[i][1]
                self.edges[self.edge_link[i][3]].N = sizes[i][1]
                self.edges[self.edge_link[i][6]].N = sizes[i][1]
                self.edges[self.edge_link[i][7]].N = sizes[i][1]

                self.edges[self.edge_link[i][8]].N = sizes[i][2]
                self.edges[self.edge_link[i][9]].N = sizes[i][2]
                self.edges[self.edge_link[i][10]].N = sizes[i][2]
                self.edges[self.edge_link[i][11]].N = sizes[i][2]
            # end for
        else: # N is already set in the edge objects, use them
            sizes = zeros((self.nVol,3),'intc')
            for ivol in xrange(self.nVol):
                sizes[ivol][0] = self.edges[self.edge_link[ivol][0]].N
                sizes[ivol][1] = self.edges[self.edge_link[ivol][2]].N
                sizes[ivol][2] = self.edges[self.edge_link[ivol][8]].N
            # end for
        # end if

        if volume_list == None:
            volume_list = range(0,self.nVol)
        # end if
        
        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        g_index = []
        l_index = []
    
        assert len(sizes) == len(volume_list),'Error: The list of sizes and \
the list of volumes must be the same length'

        # Assign unique numbers to the corners -> Corners are indexed sequentially
        node_index = arange(self.nNode)
        counter = len(node_index)

        edge_index = [ empty((0),'intc') for i in xrange(self.nEdge)]
        face_index = [ empty((0,0),'intc') for i in xrange(self.nFace)]
        # Assign unique numbers to the edges

        for ii in xrange(len(volume_list)):
            cur_size_e = [sizes[ii][0],sizes[ii][0],sizes[ii][1],sizes[ii][1],
                          sizes[ii][0],sizes[ii][0],sizes[ii][1],sizes[ii][1],
                          sizes[ii][2],sizes[ii][2],sizes[ii][2],sizes[ii][2]]  

            cur_size_f = [[sizes[ii][0],sizes[ii][1]],
                          [sizes[ii][0],sizes[ii][1]],
                          [sizes[ii][1],sizes[ii][2]],
                          [sizes[ii][1],sizes[ii][2]],
                          [sizes[ii][0],sizes[ii][2]],
                          [sizes[ii][0],sizes[ii][2]]]

            ivol = volume_list[ii]
            for iedge in xrange(12):
                edge = self.edge_link[ii][iedge]
                if edge_index[edge].shape == (0,):# Not added yet
                    edge_index[edge] = resize(edge_index[edge],cur_size_e[iedge]-2)
                    if self.edges[edge].degen == 1:
                        # Get the counter value for this "node"
                        index = node_index[self.edges[edge].n1]
                        for jj in xrange(cur_size_e[iedge]-2):
                            edge_index[edge][jj] = index
                        # end for
                    else:
                        for jj in xrange(cur_size_e[iedge]-2):
                            edge_index[edge][jj] = counter
                            counter += 1
                        # end for
                    # end if
                # end if
            # end for
            for iface in xrange(6):
                face = self.face_link[ii][iface]
                if face_index[face].shape == (0,0):
                    face_index[face] = resize(face_index[face],
                                               [cur_size_f[iface][0]-2,cur_size_f[iface][1]-2])
                    for iii in xrange(cur_size_f[iface][0]-2):
                        for jjj in xrange(cur_size_f[iface][1]-2):
                            face_index[face][iii,jjj] = counter
                            counter += 1
                        # end for
                    # end for
                # end if
            # end for
        # end for

        g_index = [ [] for i in xrange(counter)] # We must add [] for each global node
        l_index = []

        def addNode(i, j, k, N, M, L):
            type,number,index1,index2 = indexPosition3D(i,j,k,N,M,L)
            
            if type == 1:         # Face 

                if number in [0,1]:
                    icount = i;imax = N
                    jcount = j;jmax = M
                elif number in [2,3]:
                    icount = j;imax = M
                    jcount = k;jmax = L
                elif number in [4,5]:
                    icount = i;imax = N
                    jcount = k;jmax = L
                # end if

                if self.face_dir[ii][number] == 0:
                    cur_index = face_index[self.face_link[ii][number]][icount-1,jcount-1]
                elif self.face_dir[ii][number] == 1:
                    cur_index = face_index[self.face_link[ii][number]][imax-icount-2,jcount-1]
                elif self.face_dir[ii][number] == 2:
                    cur_index = face_index[self.face_link[ii][number]][icount-1,jmax-jcount-2]
                elif self.face_dir[ii][number] == 3:
                    cur_index = face_index[self.face_link[ii][number]][imax-icount-2,jmax-jcount-2]
                elif self.face_dir[ii][number] == 4:
                    cur_index = face_index[self.face_link[ii][number]][jcount-1,icount-1]
                elif self.face_dir[ii][number] == 5:
                    cur_index = face_index[self.face_link[ii][number]][jmax-jcount-2,icount-1]
                elif self.face_dir[ii][number] == 6:
                    cur_index = face_index[self.face_link[ii][number]][jcount-1,imax-icount-2]
                elif self.face_dir[ii][number] == 7:
                    cur_index = face_index[self.face_link[ii][number]][jmax-jcount-2,imax-icount-2]
                    
                l_index[ii][i,j,k] = cur_index
                g_index[cur_index].append([ivol,i,j,k])
                            
            elif type == 2:         # Edge
                        
                if number in [0,1,4,5]:
                    if self.edge_dir[ii][number] == -1: # Its a reverse dir
                        cur_index = edge_index[self.edge_link[ii][number]][N-i-2]
                    else:  
                        cur_index = edge_index[self.edge_link[ii][number]][i-1]
                    # end if
                elif number in [2,3,6,7]:
                    if self.edge_dir[ii][number] == -1: # Its a reverse dir
                        cur_index = edge_index[self.edge_link[ii][number]][M-j-2]
                    else:  
                        cur_index = edge_index[self.edge_link[ii][number]][j-1]
                    # end if
                elif number in [8,9,10,11]:
                    if self.edge_dir[ii][number] == -1: # Its a reverse dir
                        cur_index = edge_index[self.edge_link[ii][number]][L-k-2]
                    else:  
                        cur_index = edge_index[self.edge_link[ii][number]][k-1]
                    # end if
                # end if
                l_index[ii][i,j,k] = cur_index
                g_index[cur_index].append([ivol,i,j,k])
                            
            elif type == 3:                  # Node
                cur_node = self.node_link[ii][number]
                l_index[ii][i,j,k] = node_index[cur_node]
                g_index[node_index[cur_node]].append([ivol,i,j,k])
            # end if type
        # end for (volume loop)
        # Now actually fill everything up
        for ii in xrange(len(volume_list)):
            ivol = volume_list[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            L = sizes[ii][2]
            l_index.append(-1*ones((N,M,L),'intc'))

            # DO the 6 planes
            for i in xrange(N):
                for j in xrange(M):
                    for k in [0,L-1]:
                        addNode(i,j,k,N,M,L)
            for i in xrange(N):
                for j in [0,M-1]:
                    for k in xrange(1,L-1):
                        addNode(i,j,k,N,M,L)
            for i in [0,N-1]:
                for j in xrange(1,M-1):
                    for k in xrange(1,L-1):
                        addNode(i,j,k,N,M,L)
            
        # end for (ii)

        full = True
        if full: # Add the remainder
            for ii in xrange(len(volume_list)):
                ivol = volume_list[ii]
                N = sizes[ii][0]
                M = sizes[ii][1]
                L = sizes[ii][2]
                for i in xrange(1,N-1):
                    for j in xrange(1,M-1):
                        for k in xrange(1,L-1):
                            l_index[ii][i,j,k] = counter
                            g_index.append([[ivol,i,j,k]])
                            counter += 1
                        # end for
                    # end for
                # end for
            # end for
        # end if

        # Reorder the indices with a greedy scheme
            
        new_indices = zeros(len(g_index),'intc')
        new_indices[:] = -1
        new_g_index = [[] for i in xrange(len(g_index))]
        counter = 0

        # Re-order the l_index
        for ii in xrange(len(volume_list)):
            ivol = volume_list[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            L = sizes[ii][2]
            for i in xrange(N):
                for j in xrange(M):
                    for k in xrange(L):
                        if new_indices[l_index[ii][i,j,k]] == -1:
                            new_indices[l_index[ii][i,j,k]] = counter
                            l_index[ii][i,j,k] = counter 
                            counter += 1
                        else:
                            l_index[ii][i,j,k] = new_indices[l_index[ii][i,j,k]]
                        # end if
                    # end for
                # end for
            # end for
        # end for
       
        # Re-order the g_index
        for ii in xrange(len(g_index)):
            ivol  = g_index[ii][0][0]
            i     = g_index[ii][0][1]
            j     = g_index[ii][0][2]
            k     = g_index[ii][0][3]
            pt = l_index[ivol][i,j,k]
            new_g_index[pt] = g_index[ii]
            # end for
        # end for
            
        self.nGlobal = len(g_index)
        self.g_index = new_g_index
        self.l_index = l_index

        return 

class edge(object):
    '''A class for edge objects'''

    def __init__(self, n1, n2, cont, degen, intersect, dg, N):
        self.n1        = n1        # Integer for node 1
        self.n2        = n2        # Integer for node 2
        self.cont      = cont      # Integer: 0 for c0 continuity, 1 for c1 continuity
        self.degen     = degen     # Integer: 1 for degenerate, 0 otherwise
        self.intersect = intersect # Integer: 1 for an intersected edge, 0 otherwise
        self.dg        = dg        # Design Group index
        self.N         = N         # Number of control points for this edge

    def write_info(self,i,handle):
        handle.write('  %5d        | %5d | %5d | %5d | %5d | %5d |  %5d |  %5d |\n'\
                     %(i,self.n1,self.n2,self.cont,self.degen,self.intersect,self.dg,self.N))

# --------------------------------------------------------------
#                Array Rotation and Flipping Functions
# --------------------------------------------------------------

def rotateCCW(input):
    '''Rotate the input array 90 degrees CCW'''
    rows = input.shape[0]
    cols = input.shape[1]
    output = empty([cols,rows],input.dtype)
 
    for row in xrange(rows):
        for col in xrange(cols):
            output[cols-col-1][row] = input[row][col]
        # end for
    # end for

    return output

def rotateCW(input):
    '''Rotate the input array 90 degrees CW'''
    rows = input.shape[0]
    cols = input.shape[1]
    output = empty([cols,rows],input.dtype)
 
    for row in xrange(rows):
        for col in xrange(cols):
            output[col][rows-row-1] = input[row][col]
        # end for
    # end for

    return output

def reverseRows(input):
    '''Flip Rows (horizontally)'''
    rows = input.shape[0]
    cols = input.shape[1]
    output = empty([rows,cols],input.dtype)
    for row in xrange(rows):
        output[row] = input[row][::-1].copy()
    # end for

    return output

def reverseCols(input):
    '''Flip Cols (vertically)'''
    rows = input.shape[0]
    cols = input.shape[1]
    output = empty([rows,cols],input.dtype)
    for col in xrange(cols):
        output[:,col] = input[:,col][::-1].copy()
    # end for

    return output

def getBiLinearMap(edge0, edge1, edge2, edge3):
    '''Get the UV coordinates on a square defined from spacing on the edges'''

    assert len(edge0)==len(edge1),'Error, getBiLinearMap: The len of edge0 and edge1 are \
not the same'
    assert len(edge2)==len(edge3),'Error, getBiLinearMap: The len of edge2 and edge3 are \
not the same'

    N = len(edge0)
    M = len(edge2)

    UV = zeros((N,M,2))

    UV[:,0,0] = edge0
    UV[:,0,1] = 0.0

    UV[:,-1,0] = edge1
    UV[:,-1,1] = 1.0

    UV[0,:,0] = 0.0
    UV[0,:,1] = edge2

    UV[-1,:,0] = 1.0
    UV[-1,:,1] = edge3
   
    for i in xrange(1,N-1):
        x1 = edge0[i]
        y1 = 0.0

        x2 = edge1[i]
        y2 = 1.0

        for j in xrange(1,M-1):
            x3 = 0
            y3 = edge2[j]

            x4 = 1.0
            y4 = edge3[j]

            UV[i,j] = calc_intersection(x1,y1,x2,y2,x3,y3,x4,y4)
            
        # end for
    # end for
  
    return UV

def calc_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calc the intersection between two line segments defined by
    # (x1,y1) to (x2,y2) and (x3,y3) to (x4,y4)

    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1);
    ua = ((x4-x3)*(y1-y3)-(y4-y3)*(x1-x3))/denom;
    xi = x1 + ua*(x2-x1);
    yi = y1 + ua*(y2-y1);

    return xi,yi

def checkInput(input, input_name, data_type, data_rank, data_shape=None):
    '''This is a generic function to check the data type and sizes of
    inputs in functions where the user must supply proper
    values. Since Python does not do type checking on Inputs, this is
    required

    input: The input argument

    input_name: A string with the argument's name to be used in an 
                output error message

    data_type: The requested numpy data type. Up-casting will be done
               automatically, warnings will be issued if downcasting or 
               doing a float to int conversion etc. A value of None 
               indicates the data_type is not be checked.

    rank     : The desired rank of the array. It is 0 for scalars, 1 for 
               vectors, 2 for matrices etc. 

    data_shape: The required shape of the data. A value of 0 indicates
                a scalar. 1D arrays are specified with a value =>
                1. Higher dimensional arrays sizes are specified with
                a list such as [dim0,dim1,...,dimN]. A value of None
                indicates the data shape is not to be checked
          
    Output : Returns the input value iff it conforms to the specified 
             data_type and data_shape. Execption is raised otherwise. 
             
''' 

    # Checking the depth is the first and easiest thing to do.
    rank = checkRank(input)
    if not(rank == data_rank):
        if data_rank == 0:
            mpiPrint('Error: \'%s\' must be a scalar, rank=0. Input of rank %d was given.'%(input_name,rank))
            sys.exit(0)
        elif data_rank == 1:
            mpiPrint('Error: \'%s\' must be a vector, rank=1. Input of rank %d was given.'%(input_name,rank))
            sys.exit(0)
        elif data_rank == 2:
            mpiPrint('Error: \'%s\' must be a matrix, rank=2. Input of rank %d was given.'%(input_name,rank))
            sys.exit(0)
        else:
            mpiPrint('Error: \'%s\' must be of rank %d. Input of rank %d was given.'%(input_name,data_rank,rank))
            sys.exit(0)
        # end if
    # end if

    # Now we know the rank is what we expect it to be

    if rank == 0: #Scalar Case
        if data_type == None: # No need to check type and data_shape is irrelevant 
            return input
        # end if

        input_type = type(input)

        if data_type == complex: # We can upcast-ANYthing to complex
            input = complex(input)
        elif data_type == float:
            if isinstance(input,complex):
                mpiPrint('Error: \'%s\' must be a given a \'float\' value, not a \'%s\' value'%(input_name,input_type.__name__))
                sys.exit(0)
            else:
                input = float(input)
            # end if
        elif data_type == int:
            if isinstance(input,(complex,float)):
                mpiPrint('Error: \'%s\' must be a given a \'int\' value, not a \'%s\' value'%(input_name,input_type.__name__))
                sys.exit(0)
            else:
                input = int(input)
            # end if
        elif data_type == bool:
            if isinstance(input,(complex,float,int)):
                mpiPrint('Error: \'%s\' must be a given a \'bool\' value, not a \'%s\' value'%(input_name,input_type.__name__))
                sys.exit(0)
            else:
                input = bool(input)
            # end if
        # end if

        return input
    else: # We have array-like objects
        try:
            input = array(input)
        except:
            mpiPrint('Error: Rank>1 object must be cast-able to numpy array for checkInput to work')
        # end try

        if not(data_type == None): # Now check the data type
            if rank == 1: test_val=input[0]
            if rank == 2: test_val=input[0][0]
            if rank == 3: test_val=input[0][0][0]
            if rank == 4: test_val=input[0][0][0][0]
            if rank == 5: test_val=input[0][0][0][0][0]

            input_type = type(test_val)

            if data_type == complex: # We can upcast-ANYthing to complex
                input = input.astype('D')
            elif data_type == float:
                if isinstance(test_val,complex):
                    mpiPrint('Error: \'%s\' must be of type \'float\', not type \'%s\'.'%(input_name,input_type.__name__))
                    sys.exit(0)
                else:
                    input = input.astype('d')
                # end if
            elif data_type == int:
                if isinstance(test_val,(complex,float)):
                    mpiPrint('Error: \'%s\' must be of type \'int\', not type \'%s\'.'%(input_name,input_type.__name__))
                    sys.exit(0)
                else:
                    input = input.astype('intc')
                # end if
            elif data_type == bool:
                if isinstance(test_val,(complex,float,int)):
                    mpiPrint('Error: \'%s\' must be of type \'bool\', not type \'%s\'.'%(input_name,input_type.__name__))
                    sys.exit(0)
                else:
                    input = input.astype('bool')
                # end if
            # end if
        # end if

        if not(data_shape) == None:
            # Check the size of each rank
            if isinstance(data_shape,int): # Make sure data_shape is iterable
                data_shape = array([data_shape])
            # end if

            array_shape = input.shape
            for irank in xrange(rank):
                if not(array_shape[irank] == data_shape[irank]):
                    mpiPrint('Error: \'%s\' must have a length of %d in rank %d, not %d.'%(input_name,data_shape[irank],irank,array_shape[irank]))
                    sys.exit(0)
                # end if
            # end for
        # end if

        return input # If we made it to the end, just return input
    # end if

def checkRank(input):
    if not(hasattr(input,'__iter__')):
        return 0
    else:
        return 1 + checkRank(input[0])


class geoDVGlobal(object):
     
    def __init__(self, dv_name, value, lower, upper, function, useit=True):
        
        '''Create a geometric design variable (or design variable group)
        See addGeoDVGloabl in pyGeo for more information
        '''

        self.name = dv_name
        self.value = atleast_1d(array(value)).astype('D')
        self.nVal = len(self.value)
        self.lower    = float(lower)
        self.upper    = float(upper)
        self.range    = upper-lower
        self.function = function
        self.useit    = useit

        return

    def __call__(self, ref_axis):

        '''When the object is called, actually apply the function'''
        # Run the user-supplied function
        return self.function(self.value,ref_axis)


class geoDVLocal(object):
     
    def __init__(self, dv_name, lower, upper, axis, coef_list, useit=True):
        
        '''Create a set of gemoetric design variables whcih change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.
        See addGeoDVLOcal for more information
        '''
        if axis.lower() in ['x','y','z']:
            N = 1
        else:
            N = 3
        # end if

        self.nVal = len(coef_list)*N

        self.value = zeros(self.nVal,'D')
        self.name = dv_name
        self.lower = lower*ones(self.nVal)
        self.upper = upper*ones(self.nVal)
        self.range    = upper-lower
        self.coef_list = zeros((self.nVal,2),'intc')
        
        j = 0

        for i in xrange(len(coef_list)):
            if axis.lower() == 'x':
                self.coef_list[j] = [coef_list[i],0]
                j += 1
            elif axis.lower() == 'y':
                self.coef_list[j] = [coef_list[i],1]
                j += 1
            elif axis.lower() == 'z':
                self.coef_list[j] = [coef_list[i],2]
                j += 1
            else:
                self.coef_list[j] = [coef_list[i],0]
                j += 1
                self.coef_list[j] = [coef_list[i],1]
                j += 1
                self.coef_list[j] = [coef_list[i],2]
                j += 1
            # end if
        # end for
        
        return

    def __call__(self, coef):

        '''When the object is called, apply the design variable values to 
        coefficients'''
        
        for i in xrange(self.nVal):
            coef[self.coef_list[i,0],self.coef_list[i,1]] += self.value[i]
        # end for
      
        return coef

    
# def createTriPanMesh(geo,tripan_name,wake_name,surfaces=None,specs_file=None,default_size = 0.1):

#     '''Create a TriPanMesh from a pyGeo Object'''
    
#     if MPI: # Only run this on Root Prosessor if MPI
#         if MPI.Comm.Get_rank( MPI.WORLD ) == 0:
#             pass
#             # end if
#         else:
#             return
#         # end if
#     # end if

#     if surfaces == None:
#         surfaces = arange(geo.topo.nFace)
#     # end if

#     # Create a sub_topology, which MAY be the same as the original one
#     topo = geo.topo.createSubTopology(surfaces)

#     nEdge = topo.nEdge
#     nFace = topo.nFace
    
#     Edge_Number = -1*ones(nEdge,'intc')
#     Edge_Type = [ '' for i in xrange(nEdge)]
#     wakeEdges = []
#     if specs_file:
#         f = open(specs_file,'r')
#         f.readline()
#         for iedge in xrange(nEdge):
#             aux = string.split(f.readline())
#             Edge_Number[iedge] = int(aux[1])
#             Edge_Type[iedge]   = aux[2]
#             if int(aux[5]) == 1:
#                 wakeEdges.append(iedge)
#             # end if
#         # end for
#         f.close()
#     else:
#         default_size = float(default_size)
#         # First Get the default number on each edge
    
#         for iface in xrange(nFace):
#             for iedge in xrange(4):
#                 # First check if we even have to do it
#                 if Edge_Number[topo.edge_link[iface][iedge]] == -1:
#                     edge_length = geo.surfs[topo.sub_to_master_faces[iface]].getEdgeLength(iedge)
#                     Edge_Number[topo.edge_link[iface][iedge]] = int(floor(edge_length/default_size))+2
#                     Edge_Type[topo.edge_link[iface][iedge]] = 'linear'
#                 # end if
#             # end for
#         # end for
#     # end if
    
#     # Create the sizes Geo for the make consistent function
#     sizes = []
#     order = []
#     for iface in xrange(nFace):
#         sizes.append([Edge_Number[topo.edge_link[iface][0]],Edge_Number[topo.edge_link[iface][2]]])
#         order.append(0)
#     # end for
#     sizes,Edge_Number = topo.makeSizesConsistent(sizes,order)

#     # Now create the global numbering scheme
    
#     # Now we need to get the edge parameter spacing for each edge
#     topo.calcGlobalNumbering(sizes) # This gets g_index,l_index and counter

#     # Now calculate the intrinsic spacing for each edge:
#     edge_para = []
#     for iedge in xrange(nEdge):
#         if Edge_Type[iedge] == 'linear':
#             edge_para.append(linspace(0,1,Edge_Number[iedge]))
#         elif Edge_Type[iedge] == 'full_cos':
#             edge_para.append(0.5*(1-cos(linspace(0,pi,Edge_Number[iedge]))))
#         else:
#             mpiPrint('Warning: Edge type not understood. Using a linear type')
#             edge_para.append(0,1,Edge_Number[iedge])
#         # end if
#     # end for

#     # Get the number of panels
#     nPanels = 0
#     for iface in xrange(nFace):
#         nPanels += (sizes[iface][0]-1)*(sizes[iface][1]-1)
#     # end for

#     # Open the outputfile
#     fp = open(tripan_name,'w')

#     # Write he number of points and panels
#     fp.write( '%5d %5d \n'%(topo.counter,nPanels))
   
#     # Output the Points First
#     UV = []
#     for iface in xrange(nFace):
        
#         uv= getBiLinearMap(edge_para[topo.edge_link[iface][0]],
#                            edge_para[topo.edge_link[iface][1]],
#                            edge_para[topo.edge_link[iface][2]],
#                            edge_para[topo.edge_link[iface][3]])
#         UV.append(uv)

#     # end for
    
#     for ipt in xrange(len(topo.g_index)):
#         iface = topo.g_index[ipt][0][0]
#         i     = topo.g_index[ipt][0][1]
#         j     = topo.g_index[ipt][0][2]
#         pt = geo.surfs[topo.sub_to_master_faces[iface]].getValue(UV[iface][i,j][0],UV[iface][i,j][1])
#         fp.write( '%12.10e %12.10e %12.10e \n'%(pt[0],pt[1],pt[2]))
#     # end for

#     # Output the connectivity Next
#     count = 0
#     for iface in xrange(nFace):
#         for i in xrange(sizes[iface][0]-1):
#             for j in xrange(sizes[iface][1]-1):
#                 count += 1
#                 fp.write('%d %d %d %d \n'%(topo.l_index[iface][i  ,j],
#                                            topo.l_index[iface][i,j+1],
#                                            topo.l_index[iface][i+1,j+1],
#                                            topo.l_index[iface][i+1  ,j]))
#             # end for
#         # end for
#     # end for
#     fp.write('\n')
#     fp.close()

#     # Output the wake file

#     fp = open(wake_name,'w')
#     fp.write('%d\n'%(len(wakeEdges)))
#     print 'wakeEdges:',wakeEdges
#     for edge in wakeEdges:
#         # Get a surface/edge for this edge
#         surfaces = topo.getSurfaceFromEdge(edge)
#         iface = surfaces[0][0]
#         iedge = surfaces[0][1]
#         print 'iface,iedge:',iface,iedge
#         if iedge == 0:
#             indices = topo.l_index[iface][:,0]
#         elif iedge == 1:
#             indices = topo.l_index[iface][:,-1]
#         elif iedge == 2:
#             indices = topo.l_index[iface][0,:]
#         elif iedge == 3:
#             indices = topo.l_index[iface][-1,:]
#         # end if
        
#         fp.write('%d\n'%(len(indices)))

#         for i in xrange(len(indices)):
#             fp.write('%d %d\n'%(indices[len(indices)-1-i],3))
#         # end for
#     # end for

#     fp.close()

#     # Write out the default specFile
#     (dirName,fileName) = os.path.split(tripan_name)
#     (fileBaseName, fileExtension)=os.path.splitext(fileName)
#     if dirName != '':
#         new_specs_file = dirName+'/'+fileBaseName+'.specs'
#     else:
#         new_specs_file = fileBaseName+'.specs'
#     # end if
#     if specs_file == None:
#         if os.path.isfile(new_specs_file):
#             mpiPrint('Error: Attempting to write the specs file %s, but it already exists. Please\
#             delete this file and re-run'%(new_specs_file))
#             sys.exit(1)
#         # end if
#     # end if
#     specs_file = new_specs_file
#     f = open(specs_file,'w')
#     f.write('Edge Number #Node Type     Start Space   End Space   WakeEdge\n') 
#     for iedge in xrange(nEdge):
#         if iedge in wakeEdges:
#             f.write( '  %4d    %5d %10s %10.4f %10.4f  %1d \n'%(\
#                 topo.sub_to_master_edges[iedge],Edge_Number[iedge],Edge_Type[iedge],.1,.1,1))
#         else:
#             f.write( '  %4d    %5d %10s %10.4f %10.4f  %1d \n'%(\
#             topo.sub_to_master_edges[iedge],Edge_Number[iedge],Edge_Type[iedge],.1,.1,0))
#         # end if

#         # end for
#     # end for
#     f.close()

#     return
 
    



    # def makeSizesConsistent(self,sizes,order):
    #     '''Take a given list of [Nu x Nv] for each surface and return
    #     the sizes list such that all sizes are consistent

    #     prescedence is given according to the order list: 0 is highest
    #     prescedence, 1 is next highest ect.

    #     '''

    #     # First determine how many "order" loops we have
    #     nloops = max(order)+1
    #     edge_number = -1*ones(self.nDG,'intc')
    #     for iedge in xrange(self.nEdge):
    #         self.edges[iedge].N = -1
    #     # end for
    
    #     for iloop in xrange(nloops):
    #         for iface in xrange(self.nFace):
    #             if order[iface] == iloop: # Set this edge
    #                 for iedge in xrange(4):
    #                     if edge_number[self.edges[self.edge_link[iface][iedge]].dg] == -1:
    #                         if iedge in [0,1]:
    #                             edge_number[self.edges[self.edge_link[iface][iedge]].dg] = sizes[iface][0]
    #                         else:
    #                             edge_number[self.edges[self.edge_link[iface][iedge]].dg] = sizes[iface][1]
    #                         # end if
    #                     # end if
    #                 # end if
    #             # end for
    #         # end for
    #     # end for

    #     # Now repoluative the sizes:
    #     for iface in xrange(self.nFace):
    #         for i in [0,1]:
    #             sizes[iface][i] = edge_number[self.edges[self.edge_link[iface][i*2]].dg]
    #         # end for
    #     # end for

    #     # And return the number of elements on each actual edge
    #     nEdge = []
    #     for iedge in xrange(self.nEdge):
    #         self.edges[iedge].N = edge_number[self.edges[iedge].dg]
    #         nEdge.append(edge_number[self.edges[iedge].dg])
    #     # end if
    #     return sizes,nEdge


#     def createSubTopology(self,face_list):
#         '''Produce another insistance of the topology class which
#         contains a subset of the faces on this topology class'''

#         # Empty Topology Class
#         sub_topo = Topology()

#         # Also create the master-to-sub index lists
#         sub_topo.master_to_sub_nodes = -1*ones(self.nNode,'intc')
#         sub_topo.master_to_sub_edges = -1*ones(self.nEdge,'intc')
#         sub_topo.master_to_sub_faces = -1*ones(self.nFace,'intc')

#         face_con = zeros(4*len(face_list),'intc')
#         edge_link = zeros(4*len(face_list),'intc') 
#         edge_dir  = zeros((len(face_list),4),'intc')

#         for i in xrange(len(face_list)):
#             face_con[4*i:4*i+4] = self.node_link[face_list[i]]
#             edge_link[4*i:4*i+4] = self.edge_link[face_list[i]]
#             edge_dir[i] = self.edge_dir[face_list[i]]
#         # end for

#         # We have to make the nodes sequential
#         nodes,face_con = unique_index(face_con)
#         sub_topo.node_link = array(face_con).reshape((len(face_list),4))

#         sub_topo.nFace = len(sub_topo.node_link)
#         sub_topo.nNode = len(nodes)

#         sub_topo.sub_to_master_nodes = array(nodes)
#         sub_topo.sub_to_master_faces = array(face_list)
        
#         # Now set the correct entries in face and node master-to-sub arrays
#         counter = 0
#         for inode in xrange(sub_topo.nNode):
#             sub_topo.master_to_sub_nodes[sub_topo.sub_to_master_nodes[inode]] = counter
#             counter += 1
#         # end for

#         counter = 0
#         for iface in xrange(sub_topo.nFace):
#             sub_topo.master_to_sub_faces[sub_topo.sub_to_master_faces[iface]] = counter
#             counter += 1
#         # end for
     
#         # Now for the edges...

#         old_to_new_edge = []
#         new_to_old_edge = []
#         counter = -1
#         edges = []

#         nodes = array(nodes)
#         for iedge in xrange(self.nEdge):
#             # Check to see if both nodes are in our nodes

#             loc1 = where(nodes == self.edges[iedge].n1)[0]
#             loc2 = where(nodes == self.edges[iedge].n2)[0]
#             surfaces = self.getSurfaceFromEdge(iedge)

#             # Determine if ANY surfaces are still around
#             found_surf = False
#             for i in xrange(len(surfaces)):
#                 if surfaces[i][0] in sub_topo.sub_to_master_faces:
#                     found_surf = True
#                 # end if
#             # end for
            
#             if len(loc1)>0 and len(loc2)>0  and found_surf: 
#                 counter += 1
#                 old_to_new_edge.append(counter)

#                 edges.append([loc1[0],loc2[0],-1])

#                 #edges.append([self.edges[iedge].n1,self.edges[iedge].n2,-1])
#             else:
#                 old_to_new_edge.append(-1)
#             # end if
#         # end for

#         sub_topo.nEdge = len(edges)
#         # Now we can do edge_link:
#         for iface in xrange(sub_topo.nFace):
#             for iedge in xrange(4):
#                 edge_link[4*iface + iedge] = old_to_new_edge[edge_link[4*iface + iedge]]
#             # end for
#         # end for

#         sub_topo.edge_link = array(edge_link).reshape((sub_topo.nFace,4))
#         sub_topo.edge_dir = edge_dir

#         edge_link_sorted = sort(edge_link.flatten())
#         edge_link_ind    = argsort(edge_link.flatten())
#         sub_topo._calcDGs(edges,edge_link,edge_link_sorted,edge_link_ind)

#         # Now actually set all the edge objects
#         sub_topo.edges = []
#         for iedge in xrange(sub_topo.nEdge):
#             sub_topo.edges.append(edge(edges[iedge][0],edges[iedge][1],0,0,0,edges[iedge][2],0))
#         # end for

#         sub_topo.master_to_sub_edges = array(old_to_new_edge)
#         sub_topo.sub_to_master_edges = zeros(sub_topo.nEdge,'intc')
#         # Lastly we need sub_to_master_edges 

#         counter = 0

#         for iedge in  xrange(len(sub_topo.master_to_sub_edges)):
#             if sub_topo.master_to_sub_edges[iedge] != -1:
#                 sub_topo.sub_to_master_edges[counter] = iedge
#                 counter += 1
#             # end if
#         # end for
# #         print 'sub_to_master_nodes:',sub_topo.sub_to_master_nodes
# #         print 'master_to_sub_nodes:',sub_topo.master_to_sub_nodes
# #         print 'sub_to_master_faces:',sub_topo.sub_to_master_faces
# #         print 'master_to_sub_faces:',sub_topo.master_to_sub_faces
# #         print 'sub_to_master_edges:',sub_topo.sub_to_master_edges
# #         print 'master_to_sub_edges:',sub_topo.master_to_sub_edges
     
#         return sub_topo
