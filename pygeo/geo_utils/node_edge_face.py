# Local modules
from .norm import eDist
from .rotation import orientArray

# --------------------------------------------------------------
#                     Node/Edge Functions
# --------------------------------------------------------------


def edgeFromNodes(n1, n2):
    """Return the edge coorsponding to nodes n1, n2"""
    if (n1 == 0 and n2 == 1) or (n1 == 1 and n2 == 0):
        return 0
    elif (n1 == 0 and n2 == 2) or (n1 == 2 and n2 == 0):
        return 2
    elif (n1 == 3 and n2 == 1) or (n1 == 1 and n2 == 3):
        return 3
    elif (n1 == 3 and n2 == 2) or (n1 == 2 and n2 == 3):
        return 1


def edgesFromNode(n):
    """Return the two edges coorsponding to node n"""
    if n == 0:
        return 0, 2
    if n == 1:
        return 0, 3
    if n == 2:
        return 1, 2
    if n == 3:
        return 1, 3


def edgesFromNodeIndex(n, N, M):
    """Return the two edges coorsponding to node n AND return the index
    of the node on the edge according to the size (N, M)"""
    if n == 0:
        return 0, 2, 0, 0
    if n == 1:
        return 0, 3, N - 1, 0
    if n == 2:
        return 1, 2, 0, M - 1
    if n == 3:
        return 1, 3, N - 1, M - 1


def nodesFromEdge(edge):
    """Return the nodes on either edge of a standard edge"""
    if edge == 0:
        return 0, 1
    elif edge == 1:
        return 2, 3
    elif edge == 2:
        return 0, 2
    elif edge == 3:
        return 1, 3
    elif edge == 4:
        return 4, 5
    elif edge == 5:
        return 6, 7
    elif edge == 6:
        return 4, 6
    elif edge == 7:
        return 5, 7
    elif edge == 8:
        return 0, 4
    elif edge == 9:
        return 1, 5
    elif edge == 10:
        return 2, 6
    elif edge == 11:
        return 3, 7


# Volume Face/edge functions
def nodesFromFace(face):
    if face == 0:
        return [0, 1, 2, 3]
    elif face == 1:
        return [4, 5, 6, 7]
    elif face == 2:
        return [0, 2, 4, 6]
    elif face == 3:
        return [1, 3, 5, 7]
    elif face == 4:
        return [0, 1, 4, 5]
    elif face == 5:
        return [2, 3, 6, 7]


def edgesFromFace(face):
    if face == 0:
        return [0, 1, 2, 3]
    elif face == 1:
        return [4, 5, 6, 7]
    elif face == 2:
        return [2, 6, 8, 10]
    elif face == 3:
        return [3, 7, 9, 11]
    elif face == 4:
        return [0, 4, 8, 9]
    elif face == 5:
        return [1, 5, 10, 11]


def setNodeValue(arr, value, nodeIndex):
    # Set "value" in 3D array "arr" at node "nodeIndex":
    if nodeIndex == 0:
        arr[0, 0, 0] = value
    elif nodeIndex == 1:
        arr[-1, 0, 0] = value
    elif nodeIndex == 2:
        arr[0, -1, 0] = value
    elif nodeIndex == 3:
        arr[-1, -1, 0] = value

    if nodeIndex == 4:
        arr[0, 0, -1] = value
    elif nodeIndex == 5:
        arr[-1, 0, -1] = value
    elif nodeIndex == 6:
        arr[0, -1, -1] = value
    elif nodeIndex == 7:
        arr[-1, -1, -1] = value

    return arr


def setEdgeValue(arr, values, edgeDir, edgeIndex):

    if edgeDir == -1:  # Reverse values
        values = values[::-1]

    if edgeIndex == 0:
        arr[1:-1, 0, 0] = values
    elif edgeIndex == 1:
        arr[1:-1, -1, 0] = values
    elif edgeIndex == 2:
        arr[0, 1:-1, 0] = values
    elif edgeIndex == 3:
        arr[-1, 1:-1, 0] = values

    elif edgeIndex == 4:
        arr[1:-1, 0, -1] = values
    elif edgeIndex == 5:
        arr[1:-1, -1, -1] = values
    elif edgeIndex == 6:
        arr[0, 1:-1, -1] = values
    elif edgeIndex == 7:
        arr[-1, 1:-1, -1] = values

    elif edgeIndex == 8:
        arr[0, 0, 1:-1] = values
    elif edgeIndex == 9:
        arr[-1, 0, 1:-1] = values
    elif edgeIndex == 10:
        arr[0, -1, 1:-1] = values
    elif edgeIndex == 11:
        arr[-1, -1, 1:-1] = values

    return arr


def setFaceValue(arr, values, faceDir, faceIndex):

    # Orient the array first according to the dir:

    values = orientArray(faceDir, values)

    if faceIndex == 0:
        arr[1:-1, 1:-1, 0] = values
    elif faceIndex == 1:
        arr[1:-1, 1:-1, -1] = values
    elif faceIndex == 2:
        arr[0, 1:-1, 1:-1] = values
    elif faceIndex == 3:
        arr[-1, 1:-1, 1:-1] = values
    elif faceIndex == 4:
        arr[1:-1, 0, 1:-1] = values
    elif faceIndex == 5:
        arr[1:-1, -1, 1:-1] = values

    return arr


def setFaceValue2(arr, values, faceDir, faceIndex):

    # Orient the array first according to the dir:

    values = orientArray(faceDir, values)

    if faceIndex == 0:
        arr[1:-1, 1:-1, 0] = values
    elif faceIndex == 1:
        arr[1:-1, 1:-1, -1] = values
    elif faceIndex == 2:
        arr[0, 1:-1, 1:-1] = values
    elif faceIndex == 3:
        arr[-1, 1:-1, 1:-1] = values
    elif faceIndex == 4:
        arr[1:-1, 0, 1:-1] = values
    elif faceIndex == 5:
        arr[1:-1, -1, 1:-1] = values

    return arr


def getFaceValue(arr, faceIndex, offset):
    # Return the values from 'arr' on faceIndex with offset of offset:

    if faceIndex == 0:
        values = arr[:, :, offset]
    elif faceIndex == 1:
        values = arr[:, :, -1 - offset]
    elif faceIndex == 2:
        values = arr[offset, :, :]
    elif faceIndex == 3:
        values = arr[-1 - offset, :, :]
    elif faceIndex == 4:
        values = arr[:, offset, :]
    elif faceIndex == 5:
        values = arr[:, -1 - offset, :]

    return values.copy()


# --------------------------------------------------------------
#                     Other edge and face classes
# --------------------------------------------------------------


class Edge:
    """A class for edge objects"""

    def __init__(self, n1, n2, cont, degen, intersect, dg, N):
        self.n1 = n1  # Integer for node 1
        self.n2 = n2  # Integer for node 2
        self.cont = cont  # Integer: 0 for c0 continuity, 1
        # for c1 continuity
        self.degen = degen  # Integer: 1 for degenerate, 0 otherwise
        self.intersect = intersect  # Integer: 1 for an intersected
        # edge, 0 otherwise
        self.dg = dg  # Design Group index
        self.N = N  # Number of control points for this edge

    def writeInfo(self, i, handle):
        handle.write(
            "  %5d        | %5d | %5d | %5d | %5d | %5d |  %5d |  %5d |\n"
            % (i, self.n1, self.n2, self.cont, self.degen, self.intersect, self.dg, self.N)
        )


class EdgeCmpObject:
    """A temporary class for sorting edge objects"""

    def __init__(self, n1, n2, n1o, n2o, midPt, tol):
        self.n1 = n1
        self.n2 = n2
        self.nodes = [n1o, n2o]
        self.midPt = midPt
        self.tol = tol

    def __repr__(self):
        return "Node1: %d Node2: %d MidPt: %f %f %f" % (self.n1, self.n2, self.midPt[0], self.midPt[1], self.midPt[2])

    def __lt__(self, other):

        if self.n1 != other.n1:
            return self.n1 < other.n1

        if self.n2 != other.n2:
            return self.n2 < other.n2

        if eDist(self.midPt, other.midPt) < self.tol:
            return False
        else:
            if self.midPt[0] != other.midPt[0]:
                return self.midPt[0] < other.midPt[0]
            if self.midPt[1] != other.midPt[1]:
                return self.midPt[1] < other.midPt[1]
            if self.midPt[2] != other.midPt[2]:
                return self.midPt[2] < other.midPt[2]
            return False

    def __eq__(self, other):
        if self.n1 == other.n1 and self.n2 == other.n2 and eDist(self.midPt, other.midPt) < self.tol:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class FaceCmpObject:
    """A temporary class for sorting edge objects"""

    def __init__(self, n1, n2, n3, n4, n1o, n2o, n3o, n4o, midPt, tol):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.nodes = [n1o, n2o, n3o, n4o]
        self.midPt = midPt
        self.tol = tol

    def __repr__(self):
        return "n1: %d n2: %d n3: %d n4: %d MidPt: %f %f %f" % (
            self.n1,
            self.n2,
            self.n3,
            self.n4,
            self.midPt[0],
            self.midPt[1],
            self.midPt[2],
        )

    def __lt__(self, other):

        if self.n1 != other.n1:
            return self.n1 < other.n1

        if self.n2 != other.n2:
            return self.n2 < other.n2

        if self.n3 != other.n3:
            return self.n3 < other.n3

        if self.n4 != other.n4:
            return self.n4 < other.n4

        if eDist(self.midPt, other.midPt) < self.tol:
            return False
        else:
            if self.midPt[0] != other.midPt[0]:
                return self.midPt[0] < other.midPt[0]
            if self.midPt[1] != other.midPt[1]:
                return self.midPt[1] < other.midPt[1]
            if self.midPt[2] != other.midPt[2]:
                return self.midPt[2] < other.midPt[2]
            return False

    def __eq__(self, other):
        if (
            self.n1 == other.n1
            and self.n2 == other.n2
            and self.n3 == other.n3
            and self.n4 == other.n4
            and eDist(self.midPt, other.midPt) < self.tol
        ):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
