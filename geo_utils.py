# =============================================================================
# Utility Functions for Use in pyGeo
# =============================================================================

from numpy import pi,cos,sin,linspace,zeros,where,interp,sqrt,hstack,dot,\
    array,max,min,insert,delete,empty
import string ,sys

sys.path.append('../pySpline')
import pySpline

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

def rotxV(x,theta):
    ''' Rotate a coordinate in the local x frame'''
    M = [[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]]
    return dot(M,x)

def rotyV(x,theta):
    '''Rotate a coordinate in the local y frame'''
    M = [[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]
    return dot(M,x)

def rotzV(x,theta):
    '''Roate a coordinate in the local z frame'''
    'rotatez:'
    M = [[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]]
    return dot(M,x)


 # --------------------------------------------------------------
 #                I/O Functions
 # --------------------------------------------------------------

def readNValues(handle,N,type):
    '''Read N values of type 'float' or 'int' from file handle'''
    if type == 'int':
        values = zeros(N,'intc')
    elif type == 'float':
        values = zeros(N)
    else:
        print 'Error: type is not known. MUST be \'int\' or \'float\''
# end if
        
    counter = 0
    while counter < N:
        aux = string.split(handle.readline())
        if type == 'int':
            for i in xrange(len(aux)):
                values[counter] = int(aux[i])
                counter += 1
            # end for
        else:
            for i in xrange(len(aux)):
                values[counter] = float(aux[i])
                counter += 1
            # end for
        # end if
    # end while
    return values


def read_af(filename,file_type='xfoil',N=35):
    ''' Load the airfoil file of type file_type'''

    # Interpolation Format
    s_interp = 0.5*(1-cos(linspace(0,pi,N)))
   
    if file_type == 'precomp':
        f = open(filename,'r')

        aux = string.split(f.readline())
        npts = int(aux[0]) 

        xnodes = zeros(npts)
        ynodes = zeros(npts)

        f.readline()
        f.readline()
        f.readline()

        for i in xrange(npts):
            aux = string.split(f.readline())
            xnodes[i] = float(aux[0])
            ynodes[i] = float(aux[1])
        # end for
        f.close()
    
        # -------------
        # Upper Surfce
        # -------------

        # Find the trailing edge point
        index = where(xnodes == 1)
        te_index = index[0]
        n_upper = te_index+1   # number of nodes on upper surface
        n_lower = int(npts-te_index)+1 # nodes on lower surface

        # upper Surface Nodes
        x_u = xnodes[0:n_upper]
        y_u = ynodes[0:n_upper]

        # -------------
        # Lower Surface
        # -------------
        x_l = xnodes[te_index:npts]
        y_l = ynodes[te_index:npts]
        x_l = hstack([x_l,0])
        y_l = hstack([y_l,0])

    elif file_type == 'xfoil':

        f = open(filename,'r')
        line  = f.readline() # Read (and ignore) the first line
        
        r = []
        while 1:
            line = f.readline()
            if not line: break # end of file
            if line.isspace(): break # blank line
            r.append([float(s) for s in line.split()])
        # end while
        r = array(r)
        x = r[:,0]
        y = r[:,1]

        # Check for blunt TE:
        if y[0] != y[-1]:
            print 'Blunt Trailing Edge on airfoil: %s'%(filename)
            print 'Merging to a point...'
            yavg = 0.5*(y[0] + y[-1])
            y[0]  = yavg
            y[-1] = yavg
        # end if
        ntotal = len(x)

        # Find the LE Point
        xmin = min(x)
        index = where(x == xmin)[0]

        if len(index > 1): # We don't have a clearly defined LE node

            # Merge the two 
            
            xavg = 0.5*(x[index[0]] + x[index[1]])
            yavg = 0.5*(y[index[0]] + y[index[1]])
     
            x = delete(x,[index[0],index[-1]])
            y = delete(y,[index[0],index[-1]])
            
            x = insert(x,index[0],xavg)
            y = insert(y,index[0],yavg)
            
            ntotal = len(x)
        # end if
        
        le_index = index[0]

        n_upper = le_index + 1
        n_lower = ntotal - le_index
       
        # upper Surface Nodes
        x_u = x[0:n_upper]
        y_u = y[0:n_upper]

        # lower Surface Nodes
        x_l = x[n_upper-1:]
        y_l = y[n_upper-1:]

        # now reverse their directions to be consistent with other formats

        x_u = x_u[::-1].copy()
        y_u = y_u[::-1].copy()
        x_l = x_l[::-1].copy()
        y_l = y_l[::-1].copy()
    else:

        print 'file_type is unknown. Supported file_type is \'xfoil\' \
and \'precomp\''
        sys.exit(1)
    # end if

    # ---------------------- Common Processing -----------------------
    # Now determine the upper surface 's' parameter

    s = zeros(n_upper)
    for j in xrange(n_upper-1):
        s[j+1] = s[j] + sqrt((x_u[j+1]-x_u[j])**2 + (y_u[j+1]-y_u[j])**2)
    # end for
    s = s/s[-1] #Normalize s

    # linearly interpolate to find the points at the positions we want
    X_u = interp(s_interp,s,x_u)
    Y_u = interp(s_interp,s,y_u)

    # Now determine the lower surface 's' parameter

    s = zeros(n_lower)
    for j in xrange(n_lower-1):
        s[j+1] = s[j] + sqrt((x_l[j+1]-x_l[j])**2 + (y_l[j+1]-y_l[j])**2)
    # end for
    s = s/s[-1] #Normalize s
    
    # linearly interpolate to find the points at the positions we want

    X_l = interp(s_interp,s,x_l)
    Y_l = interp(s_interp,s,y_l)

    return X_u,Y_u,X_l,Y_l

# --------------------------------------------------------------
#            Working with Edges Function
# --------------------------------------------------------------

def test_edge(surf1,surf2,i,j,edge_tol):

    '''Test edge i on surf1 with edge j on surf2'''
    # First get the values at the beginning, middle and end of each segment

    val1_beg,val1_mid,val1_end = surf1.getOrigValuesEdge(i)
    val2_beg,val2_mid,val2_end = surf2.getOrigValuesEdge(j)

    # Second we wil check to see if either edge is degenerate:

    degen1,val_degen1 = surf1.checkDegenerateEdge(i)
    degen2,val_degen2 = surf2.checkDegenerateEdge(j)

    coinc = False # Default return is False
    dir_flag  = 1
    side  = 0 
    type = None
    if not degen1 and not degen2: # This is the 'regular' case where
                                  # we have two non degenerate edges
        #Three things can happen:

        # Beginning and End match (same sense)
        if e_dist(val1_beg,val2_beg) < edge_tol and \
               e_dist(val1_end,val2_end) < edge_tol:
            if e_dist(val1_mid,val2_mid) < edge_tol:
                coinc = True
            else:
                coinc = False
            dir_flag = 1

        # Beginning and End match (opposite sense)
        elif e_dist(val1_beg,val2_end) < edge_tol and \
               e_dist(val1_end,val2_beg) < edge_tol:

            if e_dist(val1_mid,val2_mid) < edge_tol:
                coinc = True
            else:
                coinc = False

            dir_flag = -1
        # end if
        type = 1 # Standard edge to edge connection
        return coinc,dir_flag

    return coinc,dir_flag

def test_node(surf1,surf2,i,j,node_tol):

    '''Test edge i on surf1 with edge j on surf2'''
    # First get the two values

    val1 = surf1.getOrigValueCorner(i)
    val2 = surf2.getOrigValueCorner(j)
    
    if e_dist(val1,val2) < node_tol:
        return True
    else:
        return False
    
def e_dist(x1,x2):
    '''Get the eculidean distance between two points'''
    return sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 + (x1[2]-x2[2])**2)


# --------------------------------------------------------------
#             Truly Miscellaneous Functions
# --------------------------------------------------------------
   
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

def directionAlongSurface(surface,line):
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

# --------------------------------------------------------------
#             Python Surface Mesh Warping Implementation
# --------------------------------------------------------------

def delI(i,j,vals):
    return sqrt( ( vals[i,j,0]-vals[i-1,j,0]) ** 2 + \
                  (vals[i,j,1]-vals[i-1,j,1]) ** 2 + \
                  (vals[i,j,2]-vals[i-1,j,2]) ** 2)

def delJ(i,j,vals):
    return sqrt( ( vals[i,j,0]-vals[i,j-1,0]) ** 2 + \
                  (vals[i,j,1]-vals[i,j-1,1]) ** 2 + \
                  (vals[i,j,2]-vals[i,j-1,2]) ** 2)

def parameterizeFace(Nu,Nv,coef):

    '''Parameterize a pyGeo surface'''
    S = zeros([Nu,Nv,2])

    for i in xrange(1,Nu):
        for j in xrange(1,Nv):
            S[i,j,0] = S[i-1,j  ,0] + delI(i,j,coef)
            S[i,j,1] = S[i  ,j-1,1] + delJ(i,j,coef)

    for i in xrange(1,Nu):
        S[i,0,0] = S[i-1,0,0] + delI(i,0,coef)
    for j in xrange(1,Nv):
        S[0,j,1] = S[0,j-1,1] + delJ(0,j,coef)

    # Do a no-check normalization
    for i in xrange(Nu):
        for j in xrange(Nv):
            S[i,j,0] /= S[-1,j,0]
            S[i,j,1] /= S[i,-1,1]

    return S

def warp_face(Nu,Nv,S,dface):
    '''Run the warp face algorithim'''

    # Edge 0
    for i in xrange(1,Nu):
        j = 0
        WTK2 = S[i,j,0]
        WTK1 = 1.0-WTK2
        dface[i,j] = WTK1 * dface[0,j] + WTK2 * dface[-1,j]

    # Edge 1
    for i in xrange(1,Nu):
        j = -1
        WTK2 = S[i,j,0]
        WTK1 = 1.0-WTK2
        dface[i,j] = WTK1 * dface[0,j] + WTK2 * dface[-1,j]

    # Edge 1
    for j in xrange(1,Nv):
        i=0
        WTK2 = S[i,j,1]
        WTK1 = 1.0-WTK2
        dface[i,j] = WTK1 * dface[i,0] + WTK2 * dface[i,-1]

    # Edge 1
    for j in xrange(1,Nv):
        i=-1
        WTK2 = S[i,j,1]
        WTK1 = 1.0-WTK2
        dface[i,j] = WTK1 * dface[i,0] + WTK2 * dface[i,-1]

    eps = 1.0e-14
   
    for i in xrange(1,Nu-1):
        for j in xrange(1,Nv-1):
            WTI2 = S[i,j,0]
            WTI1 = 1.0-WTI2
            WTJ2 = S[i,j,1]
            WTJ1 = 1.0-WTJ2
            deli = WTI1 * dface[0,j,0] + WTI2 * dface[-1,j,0]
            delj = WTJ1 * dface[i,0,0] + WTJ2 * dface[i,-1,0]

            dface[i,j,0] = (abs(deli)*deli + abs(delj)*delj)/  \
                max( ( abs (deli) + abs(delj),eps))

            deli = WTI1 * dface[0,j,1] + WTI2 * dface[-1,j,1]
            delj = WTJ1 * dface[i,0,1] + WTJ2 * dface[i,-1,1]

            dface[i,j,1] = (abs(deli)*deli + abs(delj)*delj)/ \
                max( ( abs (deli) + abs(delj),eps))
        # end for
    # end for

    return dface

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
          
def edgeFromNodes(n1,n2):
    '''Return the edge coorsponding to nodes n1,n2'''
    if (n1 == 0 and n2 == 1) or (n1 == 1 and n2 == 0):
        return 0
    elif (n1 == 0 and n2 == 2) or (n1 == 2 and n2 == 0):
        return 2
    elif (n1 == 3 and n2 == 1) or (n1 == 1 and n2 == 3):
        return 3
    elif (n1 == 3 and n2 == 2) or (n1 == 2 and n2 == 3):
        return 1

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

def flipEdge(edge):
    '''Return the edge on a surface, opposite to given edge'''
    if edge == 0: return 1
    if edge == 1: return 0
    if edge == 2: return 3
    if edge == 3: return 2
    else:
        return None


def indexPosition(i,j,N,M):
    '''This function is a generic function which determines if for a grid
    of data NxM with index i going 0->N-1 and j going 0->M-1, it
    determines if i,j is on the interior, on an edge or on a corner

    The funtion return three values: 
    type: this is 0 for interior, 1 for on an edge and 2 for on a corner
    edge: this is the edge number if type==1
    node: this is the node number if type==2 '''

    if i > 0 and i < N - 1 and j > 0 and j < M-1: # Interior
        return 0,None,None                   
    elif i > 0 and i < N - 1 and j == 0:     # Edge 0
        return 1,0,None
    elif i > 0 and i < N - 1 and j == M - 1: # Edge 1
        return 1,1,None
    elif i == 0 and j > 0 and j < M - 1:     # Edge 2
        return 1,2,None
    elif i == N - 1 and j > 0 and j < M - 1: # Edge 3
        return 1,3,None
    elif i == 0 and j == 0:                  # Node 0
        return 2,None,0
    elif i == N - 1 and j == 0:              # Node 1
        return 2,None,1
    elif i == 0 and j == M -1 :              # Node 2
        return 2,None,2
    elif i == N - 1 and j == M - 1:          # Node 3
        return 2,None,3
