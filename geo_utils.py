# =============================================================================
# Utility Functions for Use in pyGeo
# =============================================================================

from numpy import pi,cos,sin,linspace,zeros,where,interp,sqrt,hstack,dot,\
    array,max,min,insert,delete,empty,mod,tan,ones,argsort,lexsort,mod,sort,\
    arange,copy,floor
import string ,sys, copy, pdb, os

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
            try:
                if MPI.Comm.Get_rank( MPI.WORLD ) == 0:
                    print string
                # end if
            except:
                print string
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

def flatten(x):
    if not isinstance(x,list):
        return x
    elif len(x) is 0:
        return []
    elif isinstance(x[0],list):
        return flatten(x[0]) + flatten(x[1:])
    else:
        return [x[0]] + flatten(x[1:])


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


def unique_index(s,s_hash=None):
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

    Class Attributes:

        nFace: The number of faces on the topology
        nNode: The number of unique nodes on the topology
        nEdge: The number of uniuqe edges on the topology
        node_link: The array of size nFace x 4 which points
                   to the node for each corner of a face
        edge_link: The array of size nFace x 4 which points
                   to the edge for each edge on a face
        edge_dir:  The array of size nFace x 4 which detrmines
                   if the intrinsic direction of this edge is
                   opposite of the direction as recorded in the
                   edge list. edge_dir[face][#] = 1 means same direction,
                   edge_dir[face][#] = -1 means opposite direction
        l_index:   The local->global list of arrays for each face
        g_index:   The global->local list points for the entire topology
        edges:     The list of edge objects defining the topology
        face_index:A list which points to which the original faces
                   on a higher level topology. It is None for the highest level
                   topology.
        simple    : A flag to determine of this is a "simple" topology which means
                   there are NO degernate Edges, NO multiple edges sharing the same
                   nodes and NO edges which loop back and have the same nodes
        sub_topo  : A flag to determine if this topology is a sub-topology of another
                    If so, face, edge and node references are available to faciliate
                    the use of both topologies
    '''
    def __init__(self,coords=None,face_con=None,file=None,node_tol=1e-4,edge_tol=1e-4):
        '''Initialize the class with data required to compute the topology'''

        self.nFace = None
        self.nNode = None
        self.nEdge = None
        self.node_link = None
        self.edge_link = None
        self.edge_dir  = None
        
        self.l_index = None
        self.g_index = None
        
        self.edges = None
        self.face_index = None
        self.simple = False

        self.sub_topo = False

        # Thse are only set if a topology is a sub topology
        self.sub_to_master_nodes = None
        self.master_to_sub_nodes = None

        self.sub_to_master_edges = None
        self.master_to_sub_edges = None

        self.sub_to_master_faces = None
        self.master_to_sub_faces = None

        if not face_con == None: 
            face_con = array(face_con)
            midpoints = None
            self.nFace = len(face_con)
            self.simple = True
            # Check to make sure nodes are sequential
            self.nNode = len(unique(face_con.flatten()))
            if self.nNode != max(face_con.flatten())+1:
                print 'Error: The Node numbering in faceCon is not sequential. There are \
missing nodes'
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

            node_list= [coords[0][0]] # Physical Coordinates of the Nodes
            node_link = [[]]

            for iface in xrange(self.nFace):
                if iface !=0: node_link.append([])
                
                for inode in xrange(4): 
                    found_it = False
                    for i in xrange(len(node_list)):
                        if e_dist(coords[iface][inode],node_list[i]) < node_tol:
                            node_link[iface].append(i)
                            found_it = True
                            break
                        # end if
                    # end for
                    if not found_it:
                        node_list.append(coords[iface][inode])
                        node_link[iface].append(i+1)
                    # end if
                # end for
            # end for

        # Next Calculate the EDGE connectivity. 
                        
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

        elif not file==None:
            self.readEdgeConnectivity(file)
            return
        else:
            mpiPrint('Empty Topology Class Creation')
           
            return
        # end if

        # Next Calculate the Design Group Information
        self._calcDGs(edges,edge_link,edge_link_sorted,edge_link_ind)

        # Set the edge ojects
        self._setEdgeObjects(edges,midpoints)
        
        return

    def _calcDGs(self,edges,edge_link,edge_link_sorted,edge_link_ind):
        dg_counter = -1
        for i in xrange(self.nEdge):
            if edges[i][2] == -1: # Not set yet
                dg_counter += 1
                edges[i][2] = dg_counter
                self.addDGEdge(i,edges,edge_link,edge_link_sorted,edge_link_ind)
            # end if
        # end for
        self.nDG = dg_counter + 1

    def _setEdgeObjects(self,edges,midpoints):
        self.edges = []
        for i in xrange(self.nEdge): # Create the edge objects
            if midpoints: # If they exist
                if edges[i][0] == edges[i][1] and \
                       e_dist(edges[i][2],edges[edges[i][0]]) < node_tol:
                    self.edges.append(edge(edges[i][0],edges[i][1],0,1,0,edges[i][2],edges[i][3]))
                else:
                    self.edges.append(edge(edges[i][0],edges[i][1],0,0,0,edges[i][2],edges[i][3]))
                # end if
            else:
                self.edges.append(edge(edges[i][0],edges[i][1],0,0,0,edges[i][2],edges[i][3]))
            # end if
        # end for
        return
    
    def addDGEdge(self,i,edges,edge_link,edge_link_sorted,edge_link_ind):

        left  = edge_link_sorted.searchsorted(i,side='left')
        right = edge_link_sorted.searchsorted(i,side='right')
        res   = edge_link_ind[slice(left,right)]

        for j in xrange(len(res)):
            iface = res[j]/4 #Integer Division
            iedge = mod(res[j],4)
                    
            oppositeEdge = edge_link[4*iface+flipEdge(iedge)]

            if edges[oppositeEdge][2] == -1:
                edges[oppositeEdge][2] = edges[i][2]
                if not edges[oppositeEdge][0] == edges[oppositeEdge][1]:
                    self.addDGEdge(oppositeEdge,edges,edge_link,edge_link_sorted,edge_link_ind)
                # end if
            # end if
        # end for

    def calcGlobalNumbering(self,sizes,surface_list=None):
        '''Internal function to calculate the global/local numbering for each surface'''

        for i in xrange(len(sizes)):
            self.edges[self.edge_link[i][0]].Nctl = sizes[i][0]
            self.edges[self.edge_link[i][1]].Nctl = sizes[i][0]
            self.edges[self.edge_link[i][2]].Nctl = sizes[i][1]
            self.edges[self.edge_link[i][3]].Nctl = sizes[i][1]

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
        self.counter = counter
        self.g_index = g_index
        self.l_index = l_index
        
        return 

    def makeSizesConsistent(self,sizes,order):
        '''Take a given list of [Nu x Nv] for each surface and return
        the sizes list such that all sizes are consistent

        prescedence is given according to the order list: 0 is highest
        prescedence, 1 is next highest ect.

        '''

        # First determine how many "order" loops we have
        nloops = max(order)+1
        edge_number = -1*ones(self.nDG,'intc')
        for iedge in xrange(self.nEdge):
            self.edges[iedge].Nctl = -1
        # end for
    
        for iloop in xrange(nloops):
            for iface in xrange(self.nFace):
                if order[iface] == iloop: # Set this edge
                    for iedge in xrange(4):
                        if edge_number[self.edges[self.edge_link[iface][iedge]].dg] == -1:
                            if iedge in [0,1]:
                                edge_number[self.edges[self.edge_link[iface][iedge]].dg] = sizes[iface][0]
                            else:
                                edge_number[self.edges[self.edge_link[iface][iedge]].dg] = sizes[iface][1]
                            # end if
                        # end if
                    # end if
                # end for
            # end for
        # end for

        # Now repoluative the sizes:
        for iface in xrange(self.nFace):
            for i in [0,1]:
                sizes[iface][i] = edge_number[self.edges[self.edge_link[iface][i*2]].dg]
            # end for
        # end for

        # And return the number of elements on each actual edge
        nEdge = []
        for iedge in xrange(self.nEdge):
            self.edges[iedge].Nctl = edge_number[self.edges[iedge].dg]
            nEdge.append(edge_number[self.edges[iedge].dg])
        # end if
        return sizes,nEdge

    def printEdgeConnectivity(self):
        '''Print the Edge Connectivity to the screen'''


        mpiPrint('------------------------------------------------------------------------')
        mpiPrint('%3d   %3d'%(self.nEdge,self.nFace))
        mpiPrint('Edge Number    |  n0  |  n1  | Cont | Degen|Intsct|  DG  | Nctl |')
        for i in xrange(len(self.edges)):
            self.edges[i].write_info(i,sys.stdout)
        # end for
        mpiPrint('Surface Number |  n0  |  n1  |  n2  |  n3  |  e0  |  e1  |  e2  |  e3  | dir0 | dir1 | dir2 | dir3 |')
        for i in xrange(self.nFace):
            mpiPrint('    %3d        |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d '\
                     %(i,self.node_link[i][0],self.node_link[i][1],self.node_link[i][2],
                       self.node_link[i][3],self.edge_link[i][0],self.edge_link[i][1],
                       self.edge_link[i][2],self.edge_link[i][3],self.edge_dir[i][0],
                       self.edge_dir[i][1],self.edge_dir[i][2],self.edge_dir[i][3]))
        # end for
        mpiPrint('------------------------------------------------------------------------')
        return

    def writeEdgeConnectivity(self,file_name):
        '''Write the full edge connectivity to a file file_name'''
        f = open(file_name,'w')
        f.write('%3d %3d\n'%(self.nEdge,self.nFace))
        f.write('Edge Number    |  n0  |  n1  | Cont | Degen|Intsct|  DG  | Nctl |\n')
        for i in xrange(self.nEdge):
            self.edges[i].write_info(i,f)
        # end for
        f.write('Surface Number |  n0  |  n1  |  n2  |  n3  |  e0  |  e1  |  e2  |  e3  | dir0 | dir1 | dir2 | dir3 | \n')
        for i in xrange(self.nFace):
            f.write('    %3d        |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d \n'\
                    %(i,self.node_link[i][0],self.node_link[i][1],self.node_link[i][2],
                      self.node_link[i][3],self.edge_link[i][0],self.edge_link[i][1],
                      self.edge_link[i][2],self.edge_link[i][3],self.edge_dir[i][0],
                      self.edge_dir[i][1],self.edge_dir[i][2],self.edge_dir[i][3]))
        # end for
        f.close()
        
        return

    def readEdgeConnectivity(self,file_name):
        '''Read the full edge connectivity from a file file_name'''
        f = open(file_name,'r')
        aux = string.split(f.readline())
        self.nEdge = int(aux[0])
        self.nFace = int(aux[1])
        self.edges = []
        
        f.readline() # This is the header line so ignore
        for i in xrange(self.nEdge):
            aux = string.split(f.readline(),'|')
            self.edges.append(edge(int(aux[1]),int(aux[2]),int(aux[3]),
                                       int(aux[4]),int(aux[5]),int(aux[6]),int(aux[7])))
        # end for

        f.readline() # This the second header line so ignore

        self.edge_link = zeros((self.nFace,4),'intc')
        self.node_link = zeros((self.nFace,4),'intc')
        self.edge_dir  = zeros((self.nFace,4),'intc')
        
        for i in xrange(self.nFace):
            aux = string.split(f.readline(),'|')
            
            for j in xrange(4):
                self.node_link[i][j] = int(aux[j+1])
                self.edge_link[i][j] = int(aux[j+1+4])
                self.edge_dir[i][j]  = int(aux[j+1+8])
            # end for
        # end for
                
        self.nNode = len(unique(self.node_link.flatten()))
        # Get the number of design groups
        dgs = []
        for iedge in xrange(self.nEdge):
            dgs.append(self.edges[iedge].dg)
        # end for
        self.nDG = max(dgs)+ 1
        return

    def createSubTopology(self,face_list):
        '''Produce another insistance of the topology class which
        contains a subset of the faces on this topology class'''

        # Empty Topology Class
        sub_topo = Topology()

        # Also create the master-to-sub index lists
        sub_topo.master_to_sub_nodes = -1*ones(self.nNode,'intc')
        sub_topo.master_to_sub_edges = -1*ones(self.nEdge,'intc')
        sub_topo.master_to_sub_faces = -1*ones(self.nFace,'intc')

        face_con = zeros(4*len(face_list),'intc')
        edge_link = zeros(4*len(face_list),'intc') 
        edge_dir  = zeros((len(face_list),4),'intc')

        for i in xrange(len(face_list)):
            face_con[4*i:4*i+4] = self.node_link[face_list[i]]
            edge_link[4*i:4*i+4] = self.edge_link[face_list[i]]
            edge_dir[i] = self.edge_dir[face_list[i]]
        # end for

        # We have to make the nodes sequential
        nodes,face_con = unique_index(face_con)
        sub_topo.node_link = array(face_con).reshape((len(face_list),4))

        sub_topo.nFace = len(sub_topo.node_link)
        sub_topo.nNode = len(nodes)

        sub_topo.sub_to_master_nodes = array(nodes)
        sub_topo.sub_to_master_faces = array(face_list)
        
        # Now set the correct entries in face and node master-to-sub arrays
        counter = 0
        for inode in xrange(sub_topo.nNode):
            sub_topo.master_to_sub_nodes[sub_topo.sub_to_master_nodes[inode]] = counter
            counter += 1
        # end for

        counter = 0
        for iface in xrange(sub_topo.nFace):
            sub_topo.master_to_sub_faces[sub_topo.sub_to_master_faces[iface]] = counter
            counter += 1
        # end for
     
        # Now for the edges...

        old_to_new_edge = []
        new_to_old_edge = []
        counter = -1
        edges = []

        nodes = array(nodes)
        for iedge in xrange(self.nEdge):
            # Check to see if both nodes are in our nodes

            loc1 = where(nodes == self.edges[iedge].n1)[0]
            loc2 = where(nodes == self.edges[iedge].n2)[0]
            surfaces = self.getSurfaceFromEdge(iedge)

            # Determine if ANY surfaces are still around
            found_surf = False
            for i in xrange(len(surfaces)):
                if surfaces[i][0] in sub_topo.sub_to_master_faces:
                    found_surf = True
                # end if
            # end for
            
            if len(loc1)>0 and len(loc2)>0  and found_surf: 
                counter += 1
                old_to_new_edge.append(counter)

                edges.append([loc1[0],loc2[0],-1])

                #edges.append([self.edges[iedge].n1,self.edges[iedge].n2,-1])
            else:
                old_to_new_edge.append(-1)
            # end if
        # end for

        sub_topo.nEdge = len(edges)
        # Now we can do edge_link:
        for iface in xrange(sub_topo.nFace):
            for iedge in xrange(4):
                edge_link[4*iface + iedge] = old_to_new_edge[edge_link[4*iface + iedge]]
            # end for
        # end for

        sub_topo.edge_link = array(edge_link).reshape((sub_topo.nFace,4))
        sub_topo.edge_dir = edge_dir

        edge_link_sorted = sort(edge_link.flatten())
        edge_link_ind    = argsort(edge_link.flatten())
        sub_topo._calcDGs(edges,edge_link,edge_link_sorted,edge_link_ind)

        # Now actually set all the edge objects
        sub_topo.edges = []
        for iedge in xrange(sub_topo.nEdge):
            sub_topo.edges.append(edge(edges[iedge][0],edges[iedge][1],0,0,0,edges[iedge][2],0))
        # end for

        sub_topo.master_to_sub_edges = array(old_to_new_edge)
        sub_topo.sub_to_master_edges = zeros(sub_topo.nEdge,'intc')
        # Lastly we need sub_to_master_edges 

        counter = 0

        for iedge in  xrange(len(sub_topo.master_to_sub_edges)):
            if sub_topo.master_to_sub_edges[iedge] != -1:
                sub_topo.sub_to_master_edges[counter] = iedge
                counter += 1
            # end if
        # end for
#         print 'sub_to_master_nodes:',sub_topo.sub_to_master_nodes
#         print 'master_to_sub_nodes:',sub_topo.master_to_sub_nodes
#         print 'sub_to_master_faces:',sub_topo.sub_to_master_faces
#         print 'master_to_sub_faces:',sub_topo.master_to_sub_faces
#         print 'sub_to_master_edges:',sub_topo.sub_to_master_edges
#         print 'master_to_sub_edges:',sub_topo.master_to_sub_edges
     
        return sub_topo

    def getSurfaceFromEdge(self,edge):
        '''Determine the surfaces and their edge_link index that points to edge iedge'''
        # Its not efficient but it works
        surfaces = []
        for isurf in xrange(self.nFace):
            for iedge in xrange(4):
                if self.edge_link[isurf][iedge] == edge:
                    surfaces.append([isurf,iedge])
                # end if
            # end for
        # end for

        return surfaces


        
class edge(object):
    '''A class for edge objects'''

    def __init__(self,n1,n2,cont,degen,intersect,dg,Nctl):
        self.n1        = n1        # Integer for node 1
        self.n2        = n2        # Integer for node 2
        self.cont      = cont      # Integer: 0 for c0 continuity, 1 for c1 continuity
        self.degen     = degen     # Integer: 1 for degenerate, 0 otherwise
        self.intersect = intersect # Integer: 1 for an intersected edge, 0 otherwise
        self.dg        = dg        # Design Group index
        self.Nctl      = Nctl      # Number of control points for this edge

    def write_info(self,i,handle):
        handle.write('  %3d          |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |\n'\
                     %(i,self.n1,self.n2,self.cont,self.degen,self.intersect,self.dg,self.Nctl))


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


def getBiLinearMap(edge0,edge1,edge2,edge3):
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

def calc_intersection(x1,y1,x2,y2,x3,y3,x4,y4):
    # Calc the intersection between two line segments defined by
    # (x1,y1) to (x2,y2) and (x3,y3) to (x4,y4)

    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1);
    ua = ((x4-x3)*(y1-y3)-(y4-y3)*(x1-x3))/denom;
    xi = x1 + ua*(x2-x1);
    yi = y1 + ua*(y2-y1);

    return xi,yi
 
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


# ----------------------- Auto TRI-Pan Mesh Creation --------------------

def createTriPanMesh(geo,tripan_name,wake_name,surfaces=None,specs_file=None,default_size = 0.1):

    '''Create a TriPanMesh from a pyGeo Object'''
    
    if MPI: # Only run this on Root Prosessor if MPI
        if MPI.Comm.Get_rank( MPI.WORLD ) == 0:
            pass
            # end if
        else:
            return
        # end if
    # end if

    if surfaces == None:
        surfaces = arange(geo.topo.nFace)
    # end if

    # Create a sub_topology, which MAY be the same as the original one
    topo = geo.topo.createSubTopology(surfaces)

    nEdge = topo.nEdge
    nFace = topo.nFace
    
    Edge_Number = -1*ones(nEdge,'intc')
    Edge_Type = [ '' for i in xrange(nEdge)]
    wakeEdges = []
    if specs_file:
        f = open(specs_file,'r')
        f.readline()
        for iedge in xrange(nEdge):
            aux = string.split(f.readline())
            Edge_Number[iedge] = int(aux[1])
            Edge_Type[iedge]   = aux[2]
            if int(aux[5]) == 1:
                wakeEdges.append(iedge)
            # end if
        # end for
        f.close()
    else:
        default_size = float(default_size)
        # First Get the default number on each edge
    
        for iface in xrange(nFace):
            for iedge in xrange(4):
                # First check if we even have to do it
                if Edge_Number[topo.edge_link[iface][iedge]] == -1:
                    edge_length = geo.surfs[topo.sub_to_master_faces[iface]].getEdgeLength(iedge)
                    Edge_Number[topo.edge_link[iface][iedge]] = int(floor(edge_length/default_size))+2
                    Edge_Type[topo.edge_link[iface][iedge]] = 'linear'
                # end if
            # end for
        # end for
    # end if
    
    # Create the sizes Geo for the make consistent function
    sizes = []
    order = []
    for iface in xrange(nFace):
        sizes.append([Edge_Number[topo.edge_link[iface][0]],Edge_Number[topo.edge_link[iface][2]]])
        order.append(0)
    # end for
    sizes,Edge_Number = topo.makeSizesConsistent(sizes,order)

    # Now create the global numbering scheme
    
    # Now we need to get the edge parameter spacing for each edge
    topo.calcGlobalNumbering(sizes) # This gets g_index,l_index and counter

    # Now calculate the intrinsic spacing for each edge:
    edge_para = []
    for iedge in xrange(nEdge):
        if Edge_Type[iedge] == 'linear':
            edge_para.append(linspace(0,1,Edge_Number[iedge]))
        elif Edge_Type[iedge] == 'full_cos':
            edge_para.append(0.5*(1-cos(linspace(0,pi,Edge_Number[iedge]))))
        else:
            mpiPrint('Warning: Edge type not understood. Using a linear type')
            edge_para.append(0,1,Edge_Number[iedge])
        # end if
    # end for

    # Get the number of panels
    nPanels = 0
    for iface in xrange(nFace):
        nPanels += (sizes[iface][0]-1)*(sizes[iface][1]-1)
    # end for

    # Open the outputfile
    fp = open(tripan_name,'w')

    # Write he number of points and panels
    fp.write( '%5d %5d \n'%(topo.counter,nPanels))
   
    # Output the Points First
    UV = []
    for iface in xrange(nFace):
        
        uv= getBiLinearMap(edge_para[topo.edge_link[iface][0]],
                           edge_para[topo.edge_link[iface][1]],
                           edge_para[topo.edge_link[iface][2]],
                           edge_para[topo.edge_link[iface][3]])
        UV.append(uv)

    # end for
    
    for ipt in xrange(len(topo.g_index)):
        iface = topo.g_index[ipt][0][0]
        i     = topo.g_index[ipt][0][1]
        j     = topo.g_index[ipt][0][2]
        pt = geo.surfs[topo.sub_to_master_faces[iface]].getValue(UV[iface][i,j][0],UV[iface][i,j][1])
        fp.write( '%12.10e %12.10e %12.10e \n'%(pt[0],pt[1],pt[2]))
    # end for

    # Output the connectivity Next
    count = 0
    for iface in xrange(nFace):
        for i in xrange(sizes[iface][0]-1):
            for j in xrange(sizes[iface][1]-1):
                count += 1
                fp.write('%d %d %d %d \n'%(topo.l_index[iface][i  ,j],
                                           topo.l_index[iface][i,j+1],
                                           topo.l_index[iface][i+1,j+1],
                                           topo.l_index[iface][i+1  ,j]))
            # end for
        # end for
    # end for
    fp.write('\n')
    fp.close()

    # Output the wake file

    fp = open(wake_name,'w')
    fp.write('%d\n'%(len(wakeEdges)))
    print 'wakeEdges:',wakeEdges
    for edge in wakeEdges:
        # Get a surface/edge for this edge
        surfaces = topo.getSurfaceFromEdge(edge)
        iface = surfaces[0][0]
        iedge = surfaces[0][1]
        print 'iface,iedge:',iface,iedge
        if iedge == 0:
            indices = topo.l_index[iface][:,0]
        elif iedge == 1:
            indices = topo.l_index[iface][:,-1]
        elif iedge == 2:
            indices = topo.l_index[iface][0,:]
        elif iedge == 3:
            indices = topo.l_index[iface][-1,:]
        # end if
        
        fp.write('%d\n'%(len(indices)))

        for i in xrange(len(indices)):
            fp.write('%d %d\n'%(indices[len(indices)-1-i],3))
        # end for
    # end for

    fp.close()

    # Write out the default specFile
    (dirName,fileName) = os.path.split(tripan_name)
    (fileBaseName, fileExtension)=os.path.splitext(fileName)
    if dirName != '':
        new_specs_file = dirName+'/'+fileBaseName+'.specs'
    else:
        new_specs_file = fileBaseName+'.specs'
    # end if
    if specs_file == None:
        if os.path.isfile(new_specs_file):
            mpiPrint('Error: Attempting to write the specs file %s, but it already exists. Please\
            delete this file and re-run'%(new_specs_file))
            sys.exit(1)
        # end if
    # end if
    specs_file = new_specs_file
    f = open(specs_file,'w')
    f.write('Edge Number #Node Type     Start Space   End Space   WakeEdge\n') 
    for iedge in xrange(nEdge):
        if iedge in wakeEdges:
            f.write( '  %4d    %5d %10s %10.4f %10.4f  %1d \n'%(\
                topo.sub_to_master_edges[iedge],Edge_Number[iedge],Edge_Type[iedge],.1,.1,1))
        else:
            f.write( '  %4d    %5d %10s %10.4f %10.4f  %1d \n'%(\
            topo.sub_to_master_edges[iedge],Edge_Number[iedge],Edge_Type[iedge],.1,.1,0))
        # end if

        # end for
    # end for
    f.close()

    return
