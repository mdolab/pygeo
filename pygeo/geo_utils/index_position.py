# --------------------------------------------------------------
#             Index position functions
# --------------------------------------------------------------


def indexPosition1D(i, N):
    """This function is a generic function which determines if index
    over a list of length N is an interior point or node 0 or node 1.
    """
    if 0 < i < N - 1:  # Interior
        return 0, None
    elif i == 0:  # Node 0
        return 1, 0
    elif i == N - 1:  # Node 1
        return 1, 1


def indexPosition2D(i, j, N, M):
    """This function is a generic function which determines if for a grid
    of data NxM with index i going 0->N-1 and j going 0->M-1, it
    determines if i,j is on the interior, on an edge or on a corner

    The function return four values:
    type: this is 0 for interior, 1 for on an edge and 2 for on a corner
    edge: this is the edge number if type==1
    node: this is the node number if type==2
    index: this is the value index along the edge of interest --
    only defined for edges
    """

    if 0 < i < N - 1 and 0 < j < M - 1:  # Interior
        return 0, None, None, None
    elif 0 < i < N - 1 and j == 0:  # Edge 0
        return 1, 0, None, i
    elif 0 < i < N - 1 and j == M - 1:  # Edge 1
        return 1, 1, None, i
    elif i == 0 and 0 < j < M - 1:  # Edge 2
        return 1, 2, None, j
    elif i == N - 1 and 0 < j < M - 1:  # Edge 3
        return 1, 3, None, j
    elif i == 0 and j == 0:  # Node 0
        return 2, None, 0, None
    elif i == N - 1 and j == 0:  # Node 1
        return 2, None, 1, None
    elif i == 0 and j == M - 1:  # Node 2
        return 2, None, 2, None
    elif i == N - 1 and j == M - 1:  # Node 3
        return 2, None, 3, None


def indexPosition3D(i, j, k, N, M, L):
    """This function is a generic function which determines if for a
    3D grid of data NxMXL with index i going 0->N-1 and j going 0->M-1
    k going 0->L-1, it determines if i,j,k is on the interior, on a
    face, on an edge or on a corner

    Returns
    -------
    type : int
        this is 0 for interior, 1 for on an face, 3 for an edge and 4 for on a corner
    number : int
        this is the face number if type==1,
        this is the edge number if type==2,
        this is the node number if type==3

    index1 : int
        this is the value index along 0th dir the face of interest OR edge of interest
    index2 : int
        this is the value index along 1st dir the face of interest
    """

    # Note to interior->Faces->Edges->Nodes to minimize number of if checks

    # Interior:
    if 0 < i < N - 1 and 0 < j < M - 1 and 0 < k < L - 1:
        return 0, None, None, None

    elif 0 < i < N - 1 and 0 < j < M - 1 and k == 0:  # Face 0
        return 1, 0, i, j
    elif 0 < i < N - 1 and 0 < j < M - 1 and k == L - 1:  # Face 1
        return 1, 1, i, j
    elif i == 0 and 0 < j < M - 1 and 0 < k < L - 1:  # Face 2
        return 1, 2, j, k
    elif i == N - 1 and 0 < j < M - 1 and 0 < k < L - 1:  # Face 3
        return 1, 3, j, k
    elif 0 < i < N - 1 and j == 0 and 0 < k < L - 1:  # Face 4
        return 1, 4, i, k
    elif 0 < i < N - 1 and j == M - 1 and 0 < k < L - 1:  # Face 5
        return 1, 5, i, k

    elif 0 < i < N - 1 and j == 0 and k == 0:  # Edge 0
        return 2, 0, i, None
    elif 0 < i < N - 1 and j == M - 1 and k == 0:  # Edge 1
        return 2, 1, i, None
    elif i == 0 and 0 < j < M - 1 and k == 0:  # Edge 2
        return 2, 2, j, None
    elif i == N - 1 and 0 < j < M - 1 and k == 0:  # Edge 3
        return 2, 3, j, None
    elif 0 < i < N - 1 and j == 0 and k == L - 1:  # Edge 4
        return 2, 4, i, None
    elif 0 < i < N - 1 and j == M - 1 and k == L - 1:  # Edge 5
        return 2, 5, i, None
    elif i == 0 and 0 < j < M - 1 and k == L - 1:  # Edge 6
        return 2, 6, j, None
    elif i == N - 1 and 0 < j < M - 1 and k == L - 1:  # Edge 7
        return 2, 7, j, None
    elif i == 0 and j == 0 and 0 < k < L - 1:  # Edge 8
        return 2, 8, k, None
    elif i == N - 1 and j == 0 and 0 < k < L - 1:  # Edge 9
        return 2, 9, k, None
    elif i == 0 and j == M - 1 and 0 < k < L - 1:  # Edge 10
        return 2, 10, k, None
    elif i == N - 1 and j == M - 1 and 0 < k < L - 1:  # Edge 11
        return 2, 11, k, None

    elif i == 0 and j == 0 and k == 0:  # Node 0
        return 3, 0, None, None
    elif i == N - 1 and j == 0 and k == 0:  # Node 1
        return 3, 1, None, None
    elif i == 0 and j == M - 1 and k == 0:  # Node 2
        return 3, 2, None, None
    elif i == N - 1 and j == M - 1 and k == 0:  # Node 3
        return 3, 3, None, None
    elif i == 0 and j == 0 and k == L - 1:  # Node 4
        return 3, 4, None, None
    elif i == N - 1 and j == 0 and k == L - 1:  # Node 5
        return 3, 5, None, None
    elif i == 0 and j == M - 1 and k == L - 1:  # Node 6
        return 3, 6, None, None
    elif i == N - 1 and j == M - 1 and k == L - 1:  # Node 7
        return 3, 7, None, None
