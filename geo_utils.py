# =============================================================================
# Utility Functions for Use in pyGeo
# =============================================================================

from numpy import pi,cos,sin,linspace,zeros,where,interp,sqrt,hstack,dot,\
    array,max,min,insert,delete,empty,mod,tan,ones,argsort,lexsort,mod,sort,\
    arange
import string ,sys

try:
    import mpi4py
    from mpi4py import MPI
except:
    MPI = None
# end try

# =============================================================================
# Global Functions
# =============================================================================

def mpiPrint(string,NO_PRINT=False):
    if NO_PRINT:
        return
    else:
        if MPI:
            if MPI.Comm.Get_rank( MPI.WORLD ) == 0:
                print string
            # end if
        else:
            print string
        # end if
    # end if

    return

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

        if len(index) > 1: # We don't have a clearly defined LE node
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
    print 'Uniuqe'
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


def unique_edges(s,ind):
    """Return a list of the elements in s, but without duplicates.
    See unique for full description 
    Specific for edge reduction
    """
    #print 's:',s
    n = len(s)
    t = list(s)
    t.sort()
    #print 't:',t
        
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
    #print 'd:',diff
    b = where(diff == True)[0]
    #print 'b:',b
    #print 'i, ind, search'
    for i in xrange(n):
        ind[i] -= b.searchsorted(ind[i],side='right')
    # end for
    return t[:lasti],ind


def directionAlongSurface(surface,line,section=None):
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

def curveDirection(curve1,curve2):
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
    
        


def indexPosition(i,j,N,M):
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

def convertCSRtoCSC_one(nrow,ncol,Ap,Aj,Ax):
    '''Convert a one-based CSR format to a one-based CSC format'''
    nnz = Ap[-1]-1
    Bp = zeros(ncol+1,'int')
    Bi = zeros(nnz,'int')
    Bx = zeros(nnz)

    # compute number of non-zero entries per column of A 
    nnz_per_col = zeros(ncol) # temp array

    for i in xrange(nnz):
        nnz_per_col[Aj[i]]+=1
    # end for
    # cumsum the nnz_per_col to get Bp[]
    cumsum = 0
    for i in xrange(ncol):
        Bp[i] = cumsum+1
        cumsum += nnz_per_col[i]
        nnz_per_col[i] = 1
    # end for
    Bp[ncol] = nnz+1

    for i in xrange(nrow):
        row_start = Ap[i]
        row_end = Ap[i+1]
        for j  in xrange(row_start,row_end):
            col = Aj[j-1]
            k = Bp[col] + nnz_per_col[col] - 1
            Bi[k-1] = i+1
            Bx[k-1] = Ax[j-1]
            nnz_per_col[col]+=1
        # end for
    # end for
    return Bp,Bi,Bx

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
          
# --------------------------------------------------------------
#                     Node/Edge Functions
# --------------------------------------------------------------

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

def edgesFromNodeIndex(n,N,M):
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

def flipEdge(edge):
    '''Return the edge on a surface, opposite to given edge'''
    if edge == 0: return 1
    if edge == 1: return 0
    if edge == 2: return 3
    if edge == 3: return 2
    else:
        return None

# --------------------------------------------------------------
#                  Knot Vector Manipulation Functions
# --------------------------------------------------------------
    
def blendKnotVectors(knot_vectors,sym):
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


class Topology(object):
    '''
    The topology class contains the data and functions assocatied with
    at set of connected quadrilaterials. The quadraliterals may be
    edgenerate or may have edges beginning and ending at the same
    node. Non-manifold topologies (3 or more surfaces sharing an edge
    are fully supported
    '''


    def __init__(self, face_con,edges=None):
        '''Initialize the class with data required to compute the topology'''

        # First Get the edge list and unique-ify

        self.nFace = len(face_con)

        if edges == None:
            edges = []

            edge_hash = []
            for iface in xrange(self.nFace):
                #             n1                ,n2               ,dg,n,degen
                edges.append([face_con[iface][0],face_con[iface][1],-1,0,0])
                edges.append([face_con[iface][2],face_con[iface][3],-1,0,0])
                edges.append([face_con[iface][0],face_con[iface][2],-1,0,0])
                edges.append([face_con[iface][1],face_con[iface][3],-1,0,0])
            # end for
        # end if
        
        edge_dir = ones(len(edges),'intc')
        nEdge = len(edges)
        for iedge in xrange(nEdge):
            if edges[iedge][0] > edges[iedge][1]:
                temp = edges[iedge][0]
                edges[iedge][0] = edges[iedge][1]
                edges[iedge][1] = temp
                edge_dir[iedge] = -1
            # end if
            edge_hash.append(edges[iedge][0]*nEdge + edges[iedge][1])
        # end for

        indices = argsort(argsort(edge_hash))
        u_edges,edge_link = unique_edges(edges,indices.copy())

        self.edges = u_edges
        nEdge = len(u_edges)
        self.edge_dir = array(edge_dir).reshape((self.nFace,4))
        self.edge_link = array(edge_link).reshape((self.nFace,4))
        self.flat_edge_link = edge_link
        self.flat_edge_link_sorted = sort(edge_link)
        self.flat_edge_link_ind    = (argsort(edge_link))
        self.node_link = face_con
        self.node_link_flat = array(face_con).flatten()
      
     
        # Next Calculate the Design Group Information
        print 'Calculating the Design Groups'
        dg_counter = -1
        for i in xrange(nEdge):
            if self.edges[i][2] == -1: # Not set yet
                dg_counter += 1
                self.edges[i][2] = dg_counter
                self.addDGEdge(i)
            # end if
        # end for
        self.nDG = dg_counter + 1
        
        return

    def addDGEdge(self,i):

        left  = self.flat_edge_link_sorted.searchsorted(i,side='left')
        right = self.flat_edge_link_sorted.searchsorted(i,side='right')
        res   = self.flat_edge_link_ind[slice(left,right)]

        for j in xrange(len(res)):
            iface = res[j]/4 #Integer Division
            iedge = mod(res[j],4)
            if iedge in [0,1]:
                self.edges[i][3] = 0# self.surfs[iface].Nctlu
            else:
                self.edges[i][3] = 0# self.surfs[iface].Nctlv
            # end if
            
            oppositeEdge = self.edge_link[iface][flipEdge(iedge)]

            if self.edges[oppositeEdge][2] == -1:
                self.edges[oppositeEdge][2] = self.edges[i][2]
                if not self.edges[oppositeEdge][0] == self.edges[oppositeEdge][1]:
                    self.addDGEdge(oppositeEdge)
                # end if
            # end if
        # end for

    def calcGlobalNumbering(self,sizes,surface_list=None):
        '''Internal function to calculate the global/local numbering for each surface'''

        if surface_list == None:
            surface_list = range(0,self.nFace)
        # end if
        
        nNode = len(unique(self.node_link_flat))
      
        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        g_index = []
        l_index = []

        assert len(sizes) == len(surface_list),'Error: The list of sizes and \
the list of surfaces must be the same length'

        # Assign unique numbers to the corners -> Corners are indexed sequentially
        node_index = arange(nNode)
        counter = len(node_index)
        edge_index = [ [] for i in xrange(len(self.edges))]

        # Assign unique numbers to the edges

        for ii in xrange(len(surface_list)):
            cur_size = [sizes[ii][0],sizes[ii][0],sizes[ii][1],sizes[ii][1]]
            isurf = surface_list[ii]
            for iedge in xrange(4):
                edge = self.edge_link[ii][iedge]
                    
                if edge_index[edge] == []:# Not added yet
                    if self.edges[edge][4] == 1:
                        # Get the counter value for this "node"
                        index = node_index[self.edges[edge][0]]
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
                    
                    type,edge,node,index = indexPosition(i,j,N,M)

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

        return counter,g_index,l_index


    def makeSizesConsistent(self,sizes,order):
        '''Take a given list of [Nu x Nv] for each surface and return
        the sizes list such that all sizes are consistent

        prescedence is given according to the order list: 0 is highest
        prescedence, 1 is next highest ect.

        '''
        # First determine how many "order" loops we have
        nloops = max(order)+1
        edge_number = -1*ones(self.nDG)

        for iloop in xrange(nloops):
            for iface in xrange(self.nFace):
                for iedge in xrange(4):
                    if order[iface] == iloop: # Set this edge
                        if edge_number[self.edge_list[self.edge_link[iface][iedge]][2]] == -1:
                            if iedge in [0,1]:
                                edge_number[self.edge_list[self.edge_link[iface][iedge]][2]] = sizes[iface][0]
                            else:
                                edge_number[self.edge_list[self.edge_link[iface][iedge]][2]] = sizes[iface][1]
                            # end if
                        # end if
                    # end if
                # end for
            # end for
        # end for

        # Now repoluative the sizes:

        for iface in xrange(self.nFace):
            for i in [0,1]:
                sizes[iface][i] = edge_number[edge_list[edge_link[iface][i*2]][2]]
            # end for
        # end for

        return sizes







# --------------------------------------------------------------
#                     pyACDT Interface
# --------------------------------------------------------------

# def createGeometryFromACDT(ac,LiftingSurface,BodySurface,Airfoil,pyGeo):
#     '''Create a list of pyGeo objects coorsponding to the pyACDT geometry specified in ac'''
#     dtor = pi*180

#     Components = ac['_components']
#     nComp = len(Components)
#     geo_objects = []

#     for icomp in xrange(nComp):
#         print 'Processing Component: %s'%(ac['_components'][icomp].Name)
#         # Determine Type -> Lifting Surface or Body Surface
#         if isinstance(ac['_components'][icomp],BodySurface):
#             print 'Body Surfaces are not yet supported'
            

#         elif isinstance(ac['_components'][icomp],LiftingSurface):
#             nSubComp = len(Components[icomp])
#             # Get the Key Data for this object
#             xrLE = ac['_components'][icomp].xrLE
#             yrLE = ac['_components'][icomp].yrLE
#             zrLE = ac['_components'][icomp].zrLE
#             xRot = ac['_components'][icomp].xRot
#             yRot = ac['_components'][icomp].yRot
#             zRot = ac['_components'][icomp].zRot

#             for jcomp in xrange(nSubComp):
                
#                 # Get the data for this section
#                 Area = ac['_components'][icomp][jcomp].Area #used
#                 Span = ac['_components'][icomp][jcomp].Span #used
#                 Taper= ac['_components'][icomp][jcomp].Taper #used
#                 SweepLE = ac['_components'][icomp][jcomp].SweepLE #used
#                 Dihedreal = ac['_components'][icomp][jcomp].Dihedral #used
#                 xc_offset = ac['_components'][icomp][jcomp].xc_offset  # Not sure how to use this yet

#                 root_Incidence = ac['_components'][icomp][jcomp].root_Incidence
#                 root_Thickness = ac['_components'][icomp][jcomp].root_Thickness
#                 root_Airfoil_type = ac['_components'][icomp][jcomp].root_Airfoil_type
#                 root_Airfoil_ID   = ac['_components'][icomp][jcomp].root_Airfoil_ID

#                 tip_Incidence = ac['_components'][icomp][jcomp].tip_Incidence
#                 tip_Thickness = ac['_components'][icomp][jcomp].tip_Thickness
#                 tip_Airfoil_type = ac['_components'][icomp][jcomp].tip_Airfoil_type
#                 tip_Airfoil_ID   = ac['_components'][icomp][jcomp].tip_Airfoil_ID

#                 # First Deduce the le position of root and tip
                
#                 Xsec = zeros([2,3])
#                 Xsec[0,:] = [xrLE,yrLE,zrLE]
#                 Xsec[1,:] = [xrLE + Span*tan(SweepLE*dtor),
#                              yrLE + Span*tan(Dihedreal*dtor),
#                              zrLE + Span]
                
#                 # Next Deduce the chords
#                 root_chord = 2*Area/(Span*(1+Taper))
#                 tip_chord  = root_chord*Taper

#                 # Next Get the airfoil data
#                 # ----------- Root ------------------
#         	xxID_root = string.find(root_Airfoil_ID,'xx')
#                 tc_r = int(round(root_Thickness*100))
#                 if (tc_r < 10):
#                     tcID_r = str('0') + str(tc_r)
#                 else:
#                     tcID_r = str(tc_r)
#                 # end if
#                 root_Airfoil_ID = root_Airfoil_ID[:xxID_root] + tcID_r + root_Airfoil_ID[xxID_root+2:]    
#                 root_points = Airfoil(root_Airfoil_type,root_Airfoil_ID).shapepoints

#                 # ----------- Tip -------------------
#         	xxID_tip = string.find(tip_Airfoil_ID,'xx')
#                 tc_r = int(round(tip_Thickness*100))
#                 if (tc_r < 10):
#                     tcID_r = str('0') + str(tc_r)
#                 else:
#                     tcID_r = str(tc_r)
#                 # end if
#                 tip_Airfoil_ID = tip_Airfoil_ID[:xxID_tip] + tcID_r + tip_Airfoil_ID[xxID_tip+2:]    
#                 tip_points = Airfoil(tip_Airfoil_type,tip_Airfoil_ID).shapepoints

#                 # Post Process the airfoil data to get what we need

#                 N = (len(root_points)-1)/2

#                 root_upper = root_points[0:N+1]
#                 root_lower = root_points[N:]
#                 tip_upper  = tip_points[0:N+1]
#                 tip_lower  = tip_points[N:]

#                 X = zeros((2,N+1,2,3)) # 2 for upper/lower, N for the
#                                      # points on each side, 2 for root
#                                      # and tip, and 3 spatially

#                 X[0,:,0,0:2] = root_upper # Set only the x and y components, z is curently 0
#                 X[1,:,1,0:2] = root_lower
#                 X[0,:,0,0:2] = tip_upper
#                 X[1,:,1,0:2] = tip_lower

#                 X[:,:,0] *= root_chord
#                 X[:,:,1] *= tip_chord

#                 X[:,:,0] += Xsec[0]
#                 X[:,:,1] += Xsec[1]
                
#                 # Create an empty geo object
#                 cur_geo = pyGeo.pyGeo('create')
#                 cur_geo.surfs.append(pySpline.surf_spline('interpolate',ku=4,kv=4,X=X[0]))
#                 cur_geo.surfs.append(pySpline.surf_spline('interpolate',ku=4,kv=4,X=X[1]))
#                 cur_geo.nSurf = 2

#                 # Add the geo object to the list
                
#                 geo_objects.append(cur_geo)
#             # end for (sub Comp)
#         # end if (lifting/body type)
#     # end if (Comp Loop)

#     return geo_objects
                
#   def addDGEdge(self,i):
#         # Find surfs with edges of 'i'
#         for iface in xrange(self.nFace):
#             for iedge in xrange(4):
#                 edge_num = self.edge_link[iface][iedge]
#                 if edge_num == i:
#                     if iedge in [0,1]:
#                         self.edges[i][3] = 0# self.surfs[iface].Nctlu
#                     else:
#                         self.edges[i][3] = 0# self.surfs[iface].Nctlv

#                     oppositeEdge = self.edge_link[iface][flipEdge(iedge)]

#                     if self.edges[oppositeEdge][2] == -1:
#                         self.edges[oppositeEdge][2] = self.edges[i][2]

#                         # Check if the "oppositeEdge is degenerate" since DON't recursively add for them
#                         #if not self.edges[oppositeEdge][0] == self.edges[oppositeEdge][1]:
#                         self.addDGEdge(oppositeEdge)
#                         # end if
#                     # end if
#                 # end if
#             # end for
#         # end for
#         return 
