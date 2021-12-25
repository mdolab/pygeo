import numpy as np
import sys
from .geo_utils.norm import eDist
from .geo_utils.node_edge_face import (
    Edge,
    setNodeValue,
    setEdgeValue,
    setFaceValue,
    EdgeCmpObject,
    nodesFromEdge,
    nodesFromFace,
    FaceCmpObject,
)
from .geo_utils.orientation import edgeOrientation, faceOrientation
from .geo_utils.remove_duplicates import unique, uniqueIndex, pointReduce
from .geo_utils.index_position import indexPosition1D, indexPosition2D, indexPosition3D

# --------------------------------------------------------------
#                Topology classes
# --------------------------------------------------------------


class Topology:
    """
    The base topology class from which the BlockTopology, SurfaceTology and CurveTopology classes inherit from.
    The topology object contains all the info required for the block topology (most complex) however,
    simpler topologies are handled accordingly.

    Attributes
    ----------
        nVol : int
            The number of volumes in the topology (may be 0)
        nFace : int
            The number of unique faces on the topology (may be 0)
        nEdge : int
            The number of unique edges on the topology
        nNode : int
            The number of unique nodes on the topology

        nEnt : int
            The number of "entities" in the topology class. This may be curves, faces or volumes.
        mNodeEnt : int
            The number of NODES per entity. For curves it's 2, for surfaces 4 and for volumes 8.
        mEdgeEnt : int
            The number of EDGES per entity. For curves it's 1, for surfaces, 4 and for volumes, 12.
        mFaceEnt : int
            The number of faces per entity. For curves its's 0, for surfaces, 1 and for volumes, 6.
        mVolEnt : int
            The number of volumes per entity. For curves it's 0, for surfaces, 0 and for volumes, 1.

        nodeLink : ndarray[nEnt, mFaceEnt]
            The array of size nEnt x mNodesEnt which points to the node for each entity
        edgeLink : ndarray[nEnt, mFaceEnt]
            The array of size nEnt x mEdgeEnt which points to the edge for each edge of entity
        faceLink : ndarray[nEnt, mFaceEnt]
            The array of size nEnt x mFaceEnt which points to the face of each face on an entity

        edgeDir : ndarray[nEnt, mEdgeEnt]
            The array of size nEnt x mEdgeEnt which determines if the intrinsic direction of this edge is
            opposite of the direction as recorded in the edge list.
            ``edgeDir[entity#][#] = 1`` means same direction; -1 is opposite direction.
        faceDir : ndarray[nFace, 6]
            The array of size nFace x 6 which determines the intrinsic direction of this face. It is one of 0->7.

        lIndex : ndarray
            The local->global list of arrays for each volume
        gIndex : ndarray
            The global->local list points for the entire topology
        edges : ndarray
            The list of edge objects defining the topology
        simple : bool
            A flag to determine if this is a "simple" topology which means there are NO degenerate Edges,
            NO multiple edges sharing the same nodes, and NO edges which loop back and have the same nodes.
    """

    def __init__(self):
        # Not sure what should go here...

        self.nVol = None
        self.nFace = None
        self.nEdge = None
        self.nNode = None
        self.nExt = None
        self.mNodeEnt = None
        self.mEdgeEnt = None
        self.mFaceEnt = None
        self.nodeLink = None
        self.edgeLink = None
        self.faceLink = None
        self.edgeDir = None
        self.faceDir = None
        self.lIndex = None
        self.gIndex = None
        self.edges = None
        self.simple = None
        self.topoType = None
        self.nDG = None
        self.nEnt = None

    def _calcDGs(self, edges, edgeLink, edgeLinkSorted, edgeLinkInd):

        dgCounter = -1
        for i in range(self.nEdge):
            if edges[i][2] == -1:  # Not set yet
                dgCounter += 1
                edges[i][2] = dgCounter
                self._addDGEdge(i, edges, edgeLink, edgeLinkSorted, edgeLinkInd)

        self.nDG = dgCounter + 1

    def _addDGEdge(self, i, edges, edgeLink, edgeLinkSorted, edgeLinkInd):
        left = edgeLinkSorted.searchsorted(i, side="left")
        right = edgeLinkSorted.searchsorted(i, side="right")
        res = edgeLinkInd[slice(left, right)]

        for j in range(len(res)):
            ient = res[j] // self.mEdgeEnt
            iedge = np.mod(res[j], self.mEdgeEnt)

            pEdges = self._getParallelEdges(iedge)
            oppositeEdges = []
            for iii in range(len(pEdges)):
                oppositeEdges.append(edgeLink[self.mEdgeEnt * ient + pEdges[iii]])

            for ii in range(len(pEdges)):
                if edges[oppositeEdges[ii]][2] == -1:
                    edges[oppositeEdges[ii]][2] = edges[i][2]
                    if not edges[oppositeEdges[ii]][0] == edges[oppositeEdges[ii]][1]:
                        self._addDGEdge(oppositeEdges[ii], edges, edgeLink, edgeLinkSorted, edgeLinkInd)

    def _getParallelEdges(self, iedge):
        """Return parallel edges for surfaces and volumes"""

        if self.topoType == "surface":
            if iedge == 0:
                return [1]
            if iedge == 1:
                return [0]
            if iedge == 2:
                return [3]
            if iedge == 3:
                return [2]

        if self.topoType == "volume":
            if iedge == 0:
                return [1, 4, 5]
            if iedge == 1:
                return [0, 4, 5]
            if iedge == 2:
                return [3, 6, 7]
            if iedge == 3:
                return [2, 6, 7]
            if iedge == 4:
                return [0, 1, 5]
            if iedge == 5:
                return [0, 1, 4]
            if iedge == 6:
                return [2, 3, 7]
            if iedge == 7:
                return [2, 3, 6]
            if iedge == 8:
                return [9, 10, 11]
            if iedge == 9:
                return [8, 10, 11]
            if iedge == 10:
                return [8, 9, 11]
            if iedge == 11:
                return [8, 9, 10]
        if self.topoType == "curve":
            return None

    def printConnectivity(self):
        """Print the Edge Connectivity to the screen"""

        print("------------------------------------------------------------------------")
        print("%4d  %4d  %4d  %4d  %4d " % (self.nNode, self.nEdge, self.nFace, self.nVol, self.nDG))
        nList = self._getDGList()
        print("Design Group | Number")
        for i in range(self.nDG):
            print("%5d        | %5d       " % (i, nList[i]))

        # Always have edges!
        print("Edge Number    |   n0  |   n1  |  Cont | Degen | Intsct|   DG   |  N     |")
        for i in range(len(self.edges)):
            self.edges[i].writeInfo(i, sys.stdout)

        print("%9s Num |" % (self.topoType))
        for i in range(self.mNodeEnt):
            print(" n%2d|" % (i))
        for i in range(self.mEdgeEnt):
            print(" e%2d|" % (i))
        print(" ")  # Get New line

        for i in range(self.nEnt):
            print(" %5d        |" % (i))
            for j in range(self.mNodeEnt):
                print("%4d|" % self.nodeLink[i][j])

            for j in range(self.mEdgeEnt):
                print("%4d|" % (self.edgeLink[i][j] * self.edgeDir[i][j]))
            print(" ")

        print("------------------------------------------------------------------------")

        if self.topoType == "volume":
            print("Vol Number | f0 | f1 | f2 | f3 | f4 | f5 |f0dir|f1dir|f2dir|f3dir|f4dir|f5dir|")
            for i in range(self.nVol):
                print(
                    " %5d     |%4d|%4d|%4d|%4d|%4d|%4d|%5d|%5d|%5d|%5d|%5d|%5d|"
                    % (
                        i,
                        self.faceLink[i][0],
                        self.faceLink[i][1],
                        self.faceLink[i][2],
                        self.faceLink[i][3],
                        self.faceLink[i][3],
                        self.faceLink[i][5],
                        self.faceDir[i][0],
                        self.faceDir[i][1],
                        self.faceDir[i][2],
                        self.faceDir[i][3],
                        self.faceDir[i][4],
                        self.faceDir[i][5],
                    )
                )

    def writeConnectivity(self, fileName):
        """Write the full edge connectivity to a file fileName"""

        f = open(fileName, "w")
        f.write("%4d  %4d  %4d   %4d  %4d\n" % (self.nNode, self.nEdge, self.nFace, self.nVol, self.nDG))
        f.write("Design Group |  Number\n")
        # Write out the design groups and their number parameter
        nList = self._getDGList()
        for i in range(self.nDG):
            f.write("%5d        | %5d       \n" % (i, nList[i]))

        f.write("Edge Number    |   n0  |   n1  |  Cont | Degen | Intsct|   DG   |  N     |\n")
        for i in range(len(self.edges)):
            self.edges[i].writeInfo(i, f)

        f.write("%9s Num |" % (self.topoType))
        for i in range(self.mNodeEnt):
            f.write(" n%2d|" % (i))
        for i in range(self.mEdgeEnt):
            f.write(" e%2d|" % (i))
        f.write("\n")

        for i in range(self.nEnt):
            f.write(" %5d        |" % (i))
            for j in range(self.mNodeEnt):
                f.write("%4d|" % self.nodeLink[i][j])

            for j in range(self.mEdgeEnt):
                f.write("%4d|" % (self.edgeLink[i][j] * self.edgeDir[i][j]))

            f.write("\n")

        if self.topoType == "volume":

            f.write("Vol Number | f0 | f1 | f2 | f3 | f4 | f5 |f0dir|f1dir|f2dir|f3dir|f4dir|f5dir|\n")
            for i in range(self.nVol):
                f.write(
                    " %5d     |%4d|%4d|%4d|%4d|%4d|%4d|%5d|%5d|%5d|%5d|%5d|%5d|\n"
                    % (
                        i,
                        self.faceLink[i][0],
                        self.faceLink[i][1],
                        self.faceLink[i][2],
                        self.faceLink[i][3],
                        self.faceLink[i][4],
                        self.faceLink[i][5],
                        self.faceDir[i][0],
                        self.faceDir[i][1],
                        self.faceDir[i][2],
                        self.faceDir[i][3],
                        self.faceDir[i][4],
                        self.faceDir[i][5],
                    )
                )
            f.close()

    def readConnectivity(self, fileName):
        """Read the full edge connectivity from a file fileName"""
        # We must be able to populate the following:
        # nNode, nEdge, nFace,nVol,nodeLink,edgeLink,
        # faceLink,edgeDir,faceDir

        f = open(fileName)
        aux = f.readline().split()
        self.nNode = int(aux[0])
        self.nEdge = int(aux[1])
        self.nFace = int(aux[2])
        self.nVol = int(aux[3])
        self.nDG = int(aux[4])
        self.edges = []

        if self.topoType == "volume":
            self.nEnt = self.nVol
        elif self.topoType == "surface":
            self.nEnt = self.nFace
        elif self.topoType == "curve":
            self.nEnt = self.nEdge

        f.readline()  # This is the header line so ignore

        nList = np.zeros(self.nDG, "intc")
        for i in range(self.nDG):
            aux = f.readline().split("|")
            nList[i] = int(aux[1])

        f.readline()  # Second Header line

        for _i in range(self.nEdge):
            aux = f.readline().split("|")
            self.edges.append(
                Edge(int(aux[1]), int(aux[2]), int(aux[3]), int(aux[4]), int(aux[5]), int(aux[6]), int(aux[7]))
            )

        f.readline()  # This is the third header line so ignore

        self.edgeLink = np.zeros((self.nEnt, self.mEdgeEnt), "intc")
        self.nodeLink = np.zeros((self.nEnt, self.mNodeEnt), "intc")
        self.edgeDir = np.zeros((self.nEnt, self.mEdgeEnt), "intc")

        for i in range(self.nEnt):
            aux = f.readline().split("|")
            for j in range(self.mNodeEnt):
                self.nodeLink[i][j] = int(aux[j + 1])
            for j in range(self.mEdgeEnt):
                self.edgeDir[i][j] = np.sign(int(aux[j + 1 + self.mNodeEnt]))
                self.edgeLink[i][j] = int(aux[j + 1 + self.mNodeEnt]) * self.edgeDir[i][j]

        if self.topoType == "volume":
            f.readline()  # This the fourth header line so ignore

            self.faceLink = np.zeros((self.nVol, 6), "intc")
            self.faceDir = np.zeros((self.nVol, 6), "intc")
            for ivol in range(self.nVol):
                aux = f.readline().split("|")
                self.faceLink[ivol] = [int(aux[i]) for i in range(1, 7)]
                self.faceDir[ivol] = [int(aux[i]) for i in range(7, 13)]

        # Set the nList to the edges
        for iedge in range(self.nEdge):
            self.edges[iedge].N = nList[self.edges[iedge].dg]

    def _getDGList(self):
        """After calcGlobalNumbering is called with the size
        parameters, we can now produce a list of length ndg with the
        each entry coorsponing to the number N associated with that DG"""

        # This can be run in linear time...just loop over each edge
        # and add to dg list
        nList = np.zeros(self.nDG, "intc")
        for iedge in range(self.nEdge):
            nList[self.edges[iedge].dg] = self.edges[iedge].N

        return nList


class CurveTopology(Topology):
    """
    See topology class for more information
    """

    def __init__(self, coords=None, file=None):
        """Initialize the class with data required to compute the topology"""
        Topology.__init__(self)
        self.mNodeEnt = 2
        self.mEdgeEnt = 1
        self.mfaceEnt = 0
        self.mVolEnt = 0
        self.nVol = 0
        self.nFace = 0
        self.topoType = "curve"
        self.gIndex = None
        self.lIndex = None
        self.nGlobal = None
        if file is not None:
            self.readConnectivity(file)
            return

        self.edges = None
        self.simple = True

        # Must have curves
        # Get the end points of each curve
        self.nEdge = len(coords)
        coords = coords.reshape((self.nEdge * 2, 3))
        nodeList, self.nodeLink = pointReduce(coords)
        self.nodeLink = self.nodeLink.reshape((self.nEdge, 2))
        self.nNode = len(nodeList)
        self.edges = []
        self.edgeLink = np.zeros((self.nEdge, 1), "intc")
        for iedge in range(self.nEdge):
            self.edgeLink[iedge][0] = iedge

        self.edgeDir = np.zeros((self.nEdge, 1), "intc")

        for iedge in range(self.nEdge):
            n1 = self.nodeLink[iedge][0]
            n2 = self.nodeLink[iedge][1]
            if n1 < n2:
                self.edges.append(Edge(n1, n2, 0, 0, 0, iedge, 2))
                self.edgeDir[iedge][0] = 1
            else:
                self.edges.append(Edge(n2, n1, 0, 0, 0, iedge, 2))
                self.edgeDir[iedge][0] = -1

        self.nDG = self.nEdge
        self.nEnt = self.nEdge

    def calcGlobalNumbering(self, sizes, curveList=None):
        """Internal function to calculate the global/local numbering
        for each curve"""
        for i in range(len(sizes)):
            self.edges[self.edgeLink[i][0]].N = sizes[i]

        if curveList is None:
            curveList = np.arange(self.nEdge)

        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        lIndex = []

        if len(sizes) != len(curveList):
            raise ValueError("The list of sizes and the list of surfaces must be the same length")

        # Assign unique numbers to the corners -> Corners are indexed
        # sequentially
        nodeIndex = np.arange(self.nNode)
        counter = len(nodeIndex)
        edgeIndex = [[] for i in range(len(self.edges))]

        # Assign unique numbers to the edges
        for ii in range(len(curveList)):
            curSize = [sizes[ii]]
            icurve = curveList[ii]
            for iedge in range(1):
                edge = self.edgeLink[ii][iedge]

                if edgeIndex[edge] == []:  # Not added yet
                    for _jj in range(curSize[iedge] - 2):
                        edgeIndex[edge].append(counter)
                        counter += 1

        gIndex = [[] for i in range(counter)]  # We must add [] for
        # each global node

        for ii in range(len(curveList)):
            icurve = curveList[ii]
            N = sizes[ii]
            lIndex.append(-1 * np.ones(N, "intc"))

            for i in range(N):
                _type, node = indexPosition1D(i, N)

                if _type == 1:  # Node
                    curNode = self.nodeLink[ii][node]
                    lIndex[ii][i] = nodeIndex[curNode]
                    gIndex[nodeIndex[curNode]].append([icurve, i])
                else:
                    if self.edgeDir[ii][0] == -1:
                        curIndex = edgeIndex[self.edgeLink[ii][0]][N - i - 2]
                    else:
                        curIndex = edgeIndex[self.edgeLink[ii][0]][i - 1]

                    lIndex[ii][i] = curIndex
                    gIndex[curIndex].append([icurve, i])

        self.nGlobal = len(gIndex)
        self.gIndex = gIndex
        self.lIndex = lIndex


class SurfaceTopology(Topology):
    """
    See topology class for more information
    """

    def __init__(self, coords=None, faceCon=None, fileName=None, nodeTol=1e-4, edgeTol=1e-4):
        """Initialize the class with data required to compute the topology"""
        Topology.__init__(self)
        self.mNodeEnt = 4
        self.mEdgeEnt = 4
        self.mfaceEnt = 1
        self.mVolEnt = 0
        self.nVol = 0
        self.topoType = "surface"
        self.gIndex = None
        self.lIndex = None
        self.nGlobal = None
        if fileName is not None:
            self.readConnectivity(fileName)
            return

        self.edges = None
        self.faceIndex = None
        self.simple = False

        if faceCon is not None:
            faceCon = np.array(faceCon)
            midpoints = None
            self.nFace = len(faceCon)
            self.nEnt = self.nFace
            self.simple = True
            # Check to make sure nodes are sequential
            self.nNode = len(unique(faceCon.flatten()))
            if self.nNode != max(faceCon.flatten()) + 1:
                # We don't have sequential nodes
                print("Error: Nodes are not sequential")
                sys.exit(1)

            edges = []
            edgeHash = []
            for iface in range(self.nFace):
                #             n1                ,n2               ,dg,n,degen
                edges.append([faceCon[iface][0], faceCon[iface][1], -1, 0, 0])
                edges.append([faceCon[iface][2], faceCon[iface][3], -1, 0, 0])
                edges.append([faceCon[iface][0], faceCon[iface][2], -1, 0, 0])
                edges.append([faceCon[iface][1], faceCon[iface][3], -1, 0, 0])

            edgeDir = np.ones(len(edges), "intc")
            for iedge in range(self.nFace * 4):
                if edges[iedge][0] > edges[iedge][1]:
                    temp = edges[iedge][0]
                    edges[iedge][0] = edges[iedge][1]
                    edges[iedge][1] = temp
                    edgeDir[iedge] = -1

                edgeHash.append(edges[iedge][0] * 4 * self.nFace + edges[iedge][1])

            edges, edgeLink = uniqueIndex(edges, edgeHash)

            self.nEdge = len(edges)
            self.edgeLink = np.array(edgeLink).reshape((self.nFace, 4))
            self.nodeLink = np.array(faceCon)
            self.edgeDir = np.array(edgeDir).reshape((self.nFace, 4))

            edgeLinkSorted = np.sort(edgeLink)
            edgeLinkInd = np.argsort(edgeLink)

        elif coords is not None:
            self.nFace = len(coords)
            self.nEnt = self.nFace
            # We can use the pointReduce algorithim on the nodes
            nodeList, nodeLink = pointReduce(coords[:, 0:4, :].reshape((self.nFace * 4, 3)), nodeTol)
            nodeLink = nodeLink.reshape((self.nFace, 4))

            # Next Calculate the EDGE connectivity. -- This is Still
            # Brute Force

            edges = []
            midpoints = []
            edgeLink = -1 * np.ones(self.nFace * 4, "intc")
            edgeDir = np.zeros((self.nFace, 4), "intc")

            for iface in range(self.nFace):
                for iedge in range(4):
                    n1, n2 = nodesFromEdge(iedge)
                    n1 = nodeLink[iface][n1]
                    n2 = nodeLink[iface][n2]
                    midpoint = coords[iface][iedge + 4]
                    if len(edges) == 0:
                        edges.append([n1, n2, -1, 0, 0])
                        midpoints.append(midpoint)
                        edgeLink[4 * iface + iedge] = 0
                        edgeDir[iface][iedge] = 1
                    else:
                        foundIt = False
                        for i in range(len(edges)):
                            if [n1, n2] == edges[i][0:2] and n1 != n2:
                                if eDist(midpoint, midpoints[i]) < edgeTol:
                                    edgeLink[4 * iface + iedge] = i
                                    edgeDir[iface][iedge] = 1
                                    foundIt = True

                            elif [n2, n1] == edges[i][0:2] and n1 != n2:
                                if eDist(midpoint, midpoints[i]) < edgeTol:
                                    edgeLink[4 * iface + iedge] = i
                                    edgeDir[iface][iedge] = -1
                                    foundIt = True
                        # end for

                        # We went all the way though the list so add
                        # it at end and return index
                        if not foundIt:
                            edges.append([n1, n2, -1, 0, 0])
                            midpoints.append(midpoint)
                            edgeLink[4 * iface + iedge] = i + 1
                            edgeDir[iface][iedge] = 1
            # end for (iFace)

            self.nEdge = len(edges)
            self.edgeLink = np.array(edgeLink).reshape((self.nFace, 4))
            self.nodeLink = np.array(nodeLink)
            self.nNode = len(unique(self.nodeLink.flatten()))
            self.edgeDir = edgeDir

            edgeLinkSorted = np.sort(edgeLink.flatten())
            edgeLinkInd = np.argsort(edgeLink.flatten())
        # end if

        # Next Calculate the Design Group Information
        self._calcDGs(edges, edgeLink, edgeLinkSorted, edgeLinkInd)

        # Set the edge ojects
        self.edges = []
        for i in range(self.nEdge):  # Create the edge objects
            if midpoints:  # If they exist
                if edges[i][0] == edges[i][1] and eDist(midpoints[i], nodeList[edges[i][0]]) < nodeTol:
                    self.edges.append(Edge(edges[i][0], edges[i][1], 0, 1, 0, edges[i][2], edges[i][3]))
                else:
                    self.edges.append(Edge(edges[i][0], edges[i][1], 0, 0, 0, edges[i][2], edges[i][3]))
            else:
                self.edges.append(Edge(edges[i][0], edges[i][1], 0, 0, 0, edges[i][2], edges[i][3]))

    def calcGlobalNumberingDummy(self, sizes, surfaceList=None):
        """Internal function to calculate the global/local numbering for each surface"""
        for i in range(len(sizes)):
            self.edges[self.edgeLink[i][0]].N = sizes[i][0]
            self.edges[self.edgeLink[i][1]].N = sizes[i][0]
            self.edges[self.edgeLink[i][2]].N = sizes[i][1]
            self.edges[self.edgeLink[i][3]].N = sizes[i][1]

        if surfaceList is None:
            surfaceList = np.arange(0, self.nFace)

        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        if len(sizes) != len(surfaceList):
            raise ValueError("The list of sizes and the list of surfaces must be the same length")

        # Assign unique numbers to the corners -> Corners are indexed
        # sequentially
        nodeIndex = np.arange(self.nNode)
        counter = len(nodeIndex)
        edgeIndex = [[] for i in range(len(self.edges))]

        # Assign unique numbers to the edges
        for ii in range(len(surfaceList)):
            iSurf = surfaceList[ii]
            curSize = [sizes[iSurf][0], sizes[iSurf][0], sizes[iSurf][1], sizes[iSurf][1]]

            for iedge in range(4):
                edge = self.edgeLink[ii][iedge]

                if edgeIndex[edge] == []:  # Not added yet
                    if self.edges[edge].degen == 1:
                        # Get the counter value for this "node"
                        index = nodeIndex[self.edges[edge].n1]
                        for _jj in range(curSize[iedge] - 2):
                            edgeIndex[edge].append(index)
                    else:
                        for _jj in range(curSize[iedge] - 2):
                            edgeIndex[edge].append(counter)
                            counter += 1

        lIndex = []
        # Now actually fill everything up
        for ii in range(len(surfaceList)):
            iSurf = surfaceList[ii]
            N = sizes[iSurf][0]
            M = sizes[iSurf][1]
            lIndex.append(-1 * np.ones((N, M), "intc"))

        self.lIndex = lIndex

    def calcGlobalNumbering(self, sizes, surfaceList=None):
        """Internal function to calculate the global/local numbering for each surface"""
        for i in range(len(sizes)):
            self.edges[self.edgeLink[i][0]].N = sizes[i][0]
            self.edges[self.edgeLink[i][1]].N = sizes[i][0]
            self.edges[self.edgeLink[i][2]].N = sizes[i][1]
            self.edges[self.edgeLink[i][3]].N = sizes[i][1]

        if surfaceList is None:
            surfaceList = np.arange(0, self.nFace)

        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        gIndex = []
        lIndex = []

        if len(sizes) != len(surfaceList):
            raise ValueError("The list of sizes and the list of surfaces must be the same length")

        # Assign unique numbers to the corners -> Corners are indexed
        # sequentially
        nodeIndex = np.arange(self.nNode)
        counter = len(nodeIndex)
        edgeIndex = [[] for i in range(len(self.edges))]

        # Assign unique numbers to the edges
        for ii in range(len(surfaceList)):
            curSize = [sizes[ii][0], sizes[ii][0], sizes[ii][1], sizes[ii][1]]
            isurf = surfaceList[ii]
            for iedge in range(4):
                edge = self.edgeLink[ii][iedge]

                if edgeIndex[edge] == []:  # Not added yet
                    if self.edges[edge].degen == 1:
                        # Get the counter value for this "node"
                        index = nodeIndex[self.edges[edge].n1]
                        for _jj in range(curSize[iedge] - 2):
                            edgeIndex[edge].append(index)
                    else:
                        for _jj in range(curSize[iedge] - 2):
                            edgeIndex[edge].append(counter)
                            counter += 1

        gIndex = [[] for i in range(counter)]  # We must add [] for
        # each global node
        lIndex = []
        # Now actually fill everything up
        for ii in range(len(surfaceList)):
            isurf = surfaceList[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            lIndex.append(-1 * np.ones((N, M), "intc"))

            for i in range(N):
                for j in range(M):

                    _type, edge, node, index = indexPosition2D(i, j, N, M)

                    if _type == 0:  # Interior
                        lIndex[ii][i, j] = counter
                        gIndex.append([[isurf, i, j]])
                        counter += 1
                    elif _type == 1:  # Edge
                        if edge in [0, 1]:
                            # Its a reverse dir
                            if self.edgeDir[ii][edge] == -1:
                                curIndex = edgeIndex[self.edgeLink[ii][edge]][N - i - 2]
                            else:
                                curIndex = edgeIndex[self.edgeLink[ii][edge]][i - 1]
                        else:  # edge in [2, 3]
                            # Its a reverse dir
                            if self.edgeDir[ii][edge] == -1:
                                curIndex = edgeIndex[self.edgeLink[ii][edge]][M - j - 2]
                            else:
                                curIndex = edgeIndex[self.edgeLink[ii][edge]][j - 1]
                        lIndex[ii][i, j] = curIndex
                        gIndex[curIndex].append([isurf, i, j])

                    else:  # Node
                        curNode = self.nodeLink[ii][node]
                        lIndex[ii][i, j] = nodeIndex[curNode]
                        gIndex[nodeIndex[curNode]].append([isurf, i, j])
        # end for (surface loop)

        # Reorder the indices with a greedy scheme
        newIndices = np.zeros(len(gIndex), "intc")
        newIndices[:] = -1
        newGIndex = [[] for i in range(len(gIndex))]
        counter = 0

        # Re-order the lIndex
        for ii in range(len(surfaceList)):
            isurf = surfaceList[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            for i in range(N):
                for j in range(M):
                    if newIndices[lIndex[ii][i, j]] == -1:
                        newIndices[lIndex[ii][i, j]] = counter
                        lIndex[ii][i, j] = counter
                        counter += 1
                    else:
                        lIndex[ii][i, j] = newIndices[lIndex[ii][i, j]]

        # Re-order the gIndex
        for ii in range(len(gIndex)):
            isurf = gIndex[ii][0][0]
            i = gIndex[ii][0][1]
            j = gIndex[ii][0][2]
            pt = lIndex[isurf][i, j]
            newGIndex[pt] = gIndex[ii]

        self.nGlobal = len(gIndex)
        self.gIndex = newGIndex
        self.lIndex = lIndex

    def getSurfaceFromEdge(self, edge):
        """Determine the surfaces and their edgeLink index that points to edge iedge"""
        # Its not efficient but it works - scales with Nface not constant
        surfaces = []
        for isurf in range(self.nFace):
            for iedge in range(4):
                if self.edgeLink[isurf][iedge] == edge:
                    surfaces.append([isurf, iedge])

        return surfaces

    def makeSizesConsistent(self, sizes, order):
        """
        Take a given list of [Nu x Nv] for each surface and return
        the sizes list such that all sizes are consistent

        prescedence is given according to the order list: 0 is highest
        prescedence,  1 is next highest ect.
        """

        # First determine how many "order" loops we have
        nloops = max(order) + 1
        edgeNumber = -1 * np.ones(self.nDG, "intc")
        for iedge in range(self.nEdge):
            self.edges[iedge].N = -1

        for iloop in range(nloops):
            for iface in range(self.nFace):
                if order[iface] == iloop:  # Set this edge
                    for iedge in range(4):
                        dg = self.edges[self.edgeLink[iface][iedge]].dg
                        if edgeNumber[dg] == -1:
                            if iedge in [0, 1]:
                                edgeNumber[dg] = sizes[iface][0]
                            else:
                                edgeNumber[dg] = sizes[iface][1]

        # Now re-populate the sizes:
        for iface in range(self.nFace):
            for i in [0, 1]:
                dg = self.edges[self.edgeLink[iface][i * 2]].dg
                sizes[iface][i] = edgeNumber[dg]

        # And return the number of elements on each actual edge
        nEdge = []
        for iedge in range(self.nEdge):
            self.edges[iedge].N = edgeNumber[self.edges[iedge].dg]
            nEdge.append(edgeNumber[self.edges[iedge].dg])

        return sizes, nEdge


class BlockTopology(Topology):
    """
    See Topology base class for more information
    """

    def __init__(self, coords=None, nodeTol=1e-4, edgeTol=1e-4, fileName=None):
        """Initialize the class with data required to compute the topology"""

        Topology.__init__(self)
        self.mNodeEnt = 8
        self.mEdgeEnt = 12
        self.mFaceEnt = 6
        self.mVolEnt = 1
        self.topoType = "volume"
        self.gIndex = None
        self.lIndex = None
        self.nGlobal = None
        if fileName is not None:
            self.readConnectivity(fileName)
            return

        coords = np.atleast_2d(coords)
        nVol = len(coords)

        if coords.shape[1] == 8:  # Just the corners are given --- Just
            # put in np.zeros for the edge and face
            # mid points
            temp = np.zeros((nVol, (8 + 12 + 6), 3))
            temp[:, 0:8, :] = coords
            coords = temp.copy()

        # ----------------------------------------------------------
        #                     Unique Nodes
        # ----------------------------------------------------------

        # Do the pointReduce Agorithm on the corners
        un, nodeLink = pointReduce(coords[:, 0:8, :].reshape((nVol * 8, 3)), nodeTol=nodeTol)
        nodeLink = nodeLink.reshape((nVol, 8))

        # ----------------------------------------------------------
        #                     Unique Edges
        # ----------------------------------------------------------
        # Now determine the unique edges:
        edgeObjs = []
        origEdges = []
        for ivol in range(nVol):
            for iedge in range(12):
                # Node number on volume
                n1, n2 = nodesFromEdge(iedge)

                # Actual Global Node Number
                n1 = nodeLink[ivol][n1]
                n2 = nodeLink[ivol][n2]

                # Midpoint
                midpoint = coords[ivol][iedge + 8]

                # Sorted Nodes:
                ns = sorted([n1, n2])

                # Append the new edgeCmp Object
                edgeObjs.append(EdgeCmpObject(ns[0], ns[1], n1, n2, midpoint, edgeTol))

                # Keep track of original edge orientation---needed for
                # face direction
                origEdges.append([n1, n2])

        # Generate unique set of edges
        uniqueEdgeObjs, edgeLink = uniqueIndex(edgeObjs)

        edgeDir = []
        for i in range(len(edgeObjs)):  # This is nVol * 12
            edgeDir.append(edgeOrientation(origEdges[i], uniqueEdgeObjs[edgeLink[i]].nodes))
        # ----------------------------------------------------------
        #                     Unique Faces
        # ----------------------------------------------------------

        faceObjs = []
        origFaces = []
        for ivol in range(nVol):
            for iface in range(6):
                # Node number on volume
                n1, n2, n3, n4 = nodesFromFace(iface)

                # Actual Global Node Number
                n1 = nodeLink[ivol][n1]
                n2 = nodeLink[ivol][n2]
                n3 = nodeLink[ivol][n3]
                n4 = nodeLink[ivol][n4]

                # Midpoint --> May be [0, 0, 0] -> This is OK
                midpoint = coords[ivol][iface + 8 + 12]

                # Sort the nodes before they go into the faceObject
                ns = sorted([n1, n2, n3, n4])
                faceObjs.append(FaceCmpObject(ns[0], ns[1], ns[2], ns[3], n1, n2, n3, n4, midpoint, 1e-4))
                # Keep track of original face orientation---needed for
                # face direction
                origFaces.append([n1, n2, n3, n4])

        # Generate unique set of faces
        uniqueFaceObjs, faceLink = uniqueIndex(faceObjs)

        faceDir = []
        faceDirRev = []
        for i in range(len(faceObjs)):  # This is nVol * 12
            faceDir.append(faceOrientation(uniqueFaceObjs[faceLink[i]].nodes, origFaces[i]))
            faceDirRev.append(faceOrientation(origFaces[i], uniqueFaceObjs[faceLink[i]].nodes))

        # --------- Set the Requried Data for this class ------------
        self.nNode = len(un)
        self.nEdge = len(uniqueEdgeObjs)
        self.nFace = len(uniqueFaceObjs)
        self.nVol = len(coords)
        self.nEnt = self.nVol

        self.nodeLink = nodeLink
        self.edgeLink = np.array(edgeLink).reshape((nVol, 12))
        self.faceLink = np.array(faceLink).reshape((nVol, 6))

        self.edgeDir = np.array(edgeDir).reshape((nVol, 12))
        self.faceDir = np.array(faceDir).reshape((nVol, 6))
        self.faceDirRev = np.array(faceDirRev).reshape((nVol, 6))

        # Next Calculate the Design Group Information
        edgeLinkSorted = np.sort(edgeLink.flatten())
        edgeLinkInd = np.argsort(edgeLink.flatten())

        ue = []
        for i in range(len(uniqueEdgeObjs)):
            ue.append([uniqueEdgeObjs[i].nodes[0], uniqueEdgeObjs[i].nodes[1], -1, 0, 0])

        self._calcDGs(ue, edgeLink, edgeLinkSorted, edgeLinkInd)

        # Set the edge ojects
        self.edges = []
        for i in range(self.nEdge):  # Create the edge objects
            self.edges.append(Edge(ue[i][0], ue[i][1], 0, 0, 0, ue[i][2], ue[i][3]))

    def calcGlobalNumbering(self, sizes=None, volumeList=None, greedyReorder=False, gIndex=True):
        """Internal function to calculate the global/local numbering for each volume"""

        if sizes is not None:
            for i in range(len(sizes)):
                self.edges[self.edgeLink[i][0]].N = sizes[i][0]
                self.edges[self.edgeLink[i][1]].N = sizes[i][0]
                self.edges[self.edgeLink[i][4]].N = sizes[i][0]
                self.edges[self.edgeLink[i][5]].N = sizes[i][0]

                self.edges[self.edgeLink[i][2]].N = sizes[i][1]
                self.edges[self.edgeLink[i][3]].N = sizes[i][1]
                self.edges[self.edgeLink[i][6]].N = sizes[i][1]
                self.edges[self.edgeLink[i][7]].N = sizes[i][1]

                self.edges[self.edgeLink[i][8]].N = sizes[i][2]
                self.edges[self.edgeLink[i][9]].N = sizes[i][2]
                self.edges[self.edgeLink[i][10]].N = sizes[i][2]
                self.edges[self.edgeLink[i][11]].N = sizes[i][2]
        else:  # N is already set in the edge objects, use them
            sizes = np.zeros((self.nVol, 3), "intc")
            for ivol in range(self.nVol):
                sizes[ivol][0] = self.edges[self.edgeLink[ivol][0]].N
                sizes[ivol][1] = self.edges[self.edgeLink[ivol][2]].N
                sizes[ivol][2] = self.edges[self.edgeLink[ivol][8]].N

        if volumeList is None:
            volumeList = np.arange(0, self.nVol)

        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        gIndex = []
        lIndex = []

        if len(sizes) != len(volumeList):
            raise ValueError("The list of sizes and the list of volumes must be the same length")

        # Assign unique numbers to the corners -> Corners are indexed
        # sequentially
        nodeIndex = np.arange(self.nNode)
        counter = len(nodeIndex)

        edgeIndex = [np.empty((0), "intc") for i in range(self.nEdge)]
        faceIndex = [np.empty((0, 0), "intc") for i in range(self.nFace)]
        # Assign unique numbers to the edges

        for ii in range(len(volumeList)):
            curSizeE = [
                sizes[ii][0],
                sizes[ii][0],
                sizes[ii][1],
                sizes[ii][1],
                sizes[ii][0],
                sizes[ii][0],
                sizes[ii][1],
                sizes[ii][1],
                sizes[ii][2],
                sizes[ii][2],
                sizes[ii][2],
                sizes[ii][2],
            ]

            curSizeF = [
                [sizes[ii][0], sizes[ii][1]],
                [sizes[ii][0], sizes[ii][1]],
                [sizes[ii][1], sizes[ii][2]],
                [sizes[ii][1], sizes[ii][2]],
                [sizes[ii][0], sizes[ii][2]],
                [sizes[ii][0], sizes[ii][2]],
            ]

            ivol = volumeList[ii]
            for iedge in range(12):
                edge = self.edgeLink[ii][iedge]
                if edgeIndex[edge].shape == (0,):  # Not added yet
                    edgeIndex[edge] = np.resize(edgeIndex[edge], curSizeE[iedge] - 2)
                    if self.edges[edge].degen == 1:
                        # Get the counter value for this "node"
                        index = nodeIndex[self.edges[edge].n1]
                        for jj in range(curSizeE[iedge] - 2):
                            edgeIndex[edge][jj] = index
                    else:
                        for jj in range(curSizeE[iedge] - 2):
                            edgeIndex[edge][jj] = counter
                            counter += 1

            for iface in range(6):
                face = self.faceLink[ii][iface]
                if faceIndex[face].shape == (0, 0):
                    faceIndex[face] = np.resize(faceIndex[face], [curSizeF[iface][0] - 2, curSizeF[iface][1] - 2])
                    for iii in range(curSizeF[iface][0] - 2):
                        for jjj in range(curSizeF[iface][1] - 2):
                            faceIndex[face][iii, jjj] = counter
                            counter += 1

        # end for (volume list)

        gIndex = [[] for i in range(counter)]  # We must add [] for
        # each global node
        lIndex = []

        def addNode(i, j, k, N, M, L):
            _type, number, _, _ = indexPosition3D(i, j, k, N, M, L)

            if _type == 1:  # Face

                if number in [0, 1]:
                    icount = i
                    imax = N
                    jcount = j
                    jmax = M
                elif number in [2, 3]:
                    icount = j
                    imax = M
                    jcount = k
                    jmax = L
                elif number in [4, 5]:
                    icount = i
                    imax = N
                    jcount = k
                    jmax = L

                if self.faceDir[ii][number] == 0:
                    curIndex = faceIndex[self.faceLink[ii][number]][icount - 1, jcount - 1]
                elif self.faceDir[ii][number] == 1:
                    curIndex = faceIndex[self.faceLink[ii][number]][imax - icount - 2, jcount - 1]
                elif self.faceDir[ii][number] == 2:
                    curIndex = faceIndex[self.faceLink[ii][number]][icount - 1, jmax - jcount - 2]
                elif self.faceDir[ii][number] == 3:
                    curIndex = faceIndex[self.faceLink[ii][number]][imax - icount - 2, jmax - jcount - 2]
                elif self.faceDir[ii][number] == 4:
                    curIndex = faceIndex[self.faceLink[ii][number]][jcount - 1, icount - 1]
                elif self.faceDir[ii][number] == 5:
                    curIndex = faceIndex[self.faceLink[ii][number]][jmax - jcount - 2, icount - 1]
                elif self.faceDir[ii][number] == 6:
                    curIndex = faceIndex[self.faceLink[ii][number]][jcount - 1, imax - icount - 2]
                elif self.faceDir[ii][number] == 7:
                    curIndex = faceIndex[self.faceLink[ii][number]][jmax - jcount - 2, imax - icount - 2]

                lIndex[ii][i, j, k] = curIndex
                gIndex[curIndex].append([ivol, i, j, k])

            elif _type == 2:  # Edge

                if number in [0, 1, 4, 5]:
                    if self.edgeDir[ii][number] == -1:  # Its a reverse dir
                        curIndex = edgeIndex[self.edgeLink[ii][number]][N - i - 2]
                    else:
                        curIndex = edgeIndex[self.edgeLink[ii][number]][i - 1]

                elif number in [2, 3, 6, 7]:
                    if self.edgeDir[ii][number] == -1:  # Its a reverse dir
                        curIndex = edgeIndex[self.edgeLink[ii][number]][M - j - 2]
                    else:
                        curIndex = edgeIndex[self.edgeLink[ii][number]][j - 1]

                elif number in [8, 9, 10, 11]:
                    if self.edgeDir[ii][number] == -1:  # Its a reverse dir
                        curIndex = edgeIndex[self.edgeLink[ii][number]][L - k - 2]
                    else:
                        curIndex = edgeIndex[self.edgeLink[ii][number]][k - 1]

                lIndex[ii][i, j, k] = curIndex
                gIndex[curIndex].append([ivol, i, j, k])

            elif _type == 3:  # Node
                curNode = self.nodeLink[ii][number]
                lIndex[ii][i, j, k] = nodeIndex[curNode]
                gIndex[nodeIndex[curNode]].append([ivol, i, j, k])

        # end for (volume loop)

        # Now actually fill everything up
        for ii in range(len(volumeList)):
            ivol = volumeList[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            L = sizes[ii][2]
            lIndex.append(-1 * np.ones((N, M, L), "intc"))

            # DO the 6 planes
            for k in [0, L - 1]:
                for i in range(N):
                    for j in range(M):
                        addNode(i, j, k, N, M, L)
            for j in [0, M - 1]:
                for i in range(N):
                    for k in range(1, L - 1):
                        addNode(i, j, k, N, M, L)

            for i in [0, N - 1]:
                for j in range(1, M - 1):
                    for k in range(1, L - 1):
                        addNode(i, j, k, N, M, L)

        # Add the remainder
        for ii in range(len(volumeList)):
            ivol = volumeList[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            L = sizes[ii][2]

            NN = sizes[ii][0] - 2
            MM = sizes[ii][1] - 2
            LL = sizes[ii][2] - 2

            toAdd = NN * MM * LL

            lIndex[ii][1 : N - 1, 1 : M - 1, 1 : L - 1] = np.arange(counter, counter + toAdd).reshape((NN, MM, LL))

            counter = counter + toAdd
            A = np.zeros((toAdd, 1, 4), "intc")
            A[:, 0, 0] = ivol
            A[:, 0, 1:] = np.mgrid[1 : N - 1, 1 : M - 1, 1 : L - 1].transpose((1, 2, 3, 0)).reshape((toAdd, 3))
            gIndex.extend(A)

        # Set the following as atributes
        self.nGlobal = len(gIndex)
        self.gIndex = gIndex
        self.lIndex = lIndex

        if greedyReorder:

            # Reorder the indices with a greedy scheme
            newIndices = np.zeros(len(gIndex), "intc")
            newIndices[:] = -1
            newGIndex = [[] for i in range(len(gIndex))]
            counter = 0

            # Re-order the lIndex
            for ii in range(len(volumeList)):
                ivol = volumeList[ii]
                N = sizes[ii][0]
                M = sizes[ii][1]
                L = sizes[ii][2]
                for i in range(N):
                    for j in range(M):
                        for k in range(L):
                            if newIndices[lIndex[ii][i, j, k]] == -1:
                                newIndices[lIndex[ii][i, j, k]] = counter
                                lIndex[ii][i, j, k] = counter
                                counter += 1
                            else:
                                lIndex[ii][i, j, k] = newIndices[lIndex[ii][i, j, k]]

            # Re-order the gIndex
            for ii in range(len(gIndex)):
                ivol = gIndex[ii][0][0]
                i = gIndex[ii][0][1]
                j = gIndex[ii][0][2]
                k = gIndex[ii][0][3]
                pt = lIndex[ivol][i, j, k]
                newGIndex[pt] = gIndex[ii]

            self.gIndex = newGIndex
            self.lIndex = lIndex
        # end if (greedy reorder)

    def calcGlobalNumbering2(self, sizes=None, gIndex=True, volumeList=None, greedyReorder=False):
        """Internal function to calculate the global/local numbering for each volume"""
        if sizes is not None:
            for i in range(len(sizes)):
                self.edges[self.edgeLink[i][0]].N = sizes[i][0]
                self.edges[self.edgeLink[i][1]].N = sizes[i][0]
                self.edges[self.edgeLink[i][4]].N = sizes[i][0]
                self.edges[self.edgeLink[i][5]].N = sizes[i][0]

                self.edges[self.edgeLink[i][2]].N = sizes[i][1]
                self.edges[self.edgeLink[i][3]].N = sizes[i][1]
                self.edges[self.edgeLink[i][6]].N = sizes[i][1]
                self.edges[self.edgeLink[i][7]].N = sizes[i][1]

                self.edges[self.edgeLink[i][8]].N = sizes[i][2]
                self.edges[self.edgeLink[i][9]].N = sizes[i][2]
                self.edges[self.edgeLink[i][10]].N = sizes[i][2]
                self.edges[self.edgeLink[i][11]].N = sizes[i][2]
        else:  # N is already set in the edge objects, use them
            sizes = np.zeros((self.nVol, 3), "intc")
            for ivol in range(self.nVol):
                sizes[ivol][0] = self.edges[self.edgeLink[ivol][0]].N
                sizes[ivol][1] = self.edges[self.edgeLink[ivol][2]].N
                sizes[ivol][2] = self.edges[self.edgeLink[ivol][8]].N

        if volumeList is None:
            volumeList = np.arange(0, self.nVol)

        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        lIndex = []

        if len(sizes) != len(volumeList):
            raise ValueError("The list of sizes and the list of volumes must be the same length")

        # Assign unique numbers to the corners -> Corners are indexed
        # sequentially
        nodeIndex = np.arange(self.nNode)
        counter = len(nodeIndex)

        edgeIndex = [np.empty((0), "intc") for i in range(self.nEdge)]
        faceIndex = [np.empty((0, 0), "intc") for i in range(self.nFace)]
        # Assign unique numbers to the edges

        for ii in range(len(volumeList)):
            curSizeE = [
                sizes[ii][0],
                sizes[ii][0],
                sizes[ii][1],
                sizes[ii][1],
                sizes[ii][0],
                sizes[ii][0],
                sizes[ii][1],
                sizes[ii][1],
                sizes[ii][2],
                sizes[ii][2],
                sizes[ii][2],
                sizes[ii][2],
            ]

            curSizeF = [
                [sizes[ii][0], sizes[ii][1]],
                [sizes[ii][0], sizes[ii][1]],
                [sizes[ii][1], sizes[ii][2]],
                [sizes[ii][1], sizes[ii][2]],
                [sizes[ii][0], sizes[ii][2]],
                [sizes[ii][0], sizes[ii][2]],
            ]

            ivol = volumeList[ii]
            for iedge in range(12):
                edge = self.edgeLink[ii][iedge]
                if edgeIndex[edge].shape == (0,):  # Not added yet
                    edgeIndex[edge] = np.resize(edgeIndex[edge], curSizeE[iedge] - 2)
                    if self.edges[edge].degen == 1:
                        # Get the counter value for this "node"
                        index = nodeIndex[self.edges[edge].n1]
                        for jj in range(curSizeE[iedge] - 2):
                            edgeIndex[edge][jj] = index
                    else:
                        edgeIndex[edge][:] = np.arange(counter, counter + curSizeE[iedge] - 2)
                        counter += curSizeE[iedge] - 2

            for iface in range(6):
                face = self.faceLink[ii][iface]
                if faceIndex[face].shape == (0, 0):
                    faceIndex[face] = np.resize(faceIndex[face], [curSizeF[iface][0] - 2, curSizeF[iface][1] - 2])
                    N = curSizeF[iface][0] - 2
                    M = curSizeF[iface][1] - 2
                    faceIndex[face] = np.arange(counter, counter + N * M).reshape((N, M))
                    counter += N * M
        # end for (volume list)

        lIndex = []

        # Now actually fill everything up
        for ii in range(len(volumeList)):
            iVol = volumeList[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            L = sizes[ii][2]
            lIndex.append(-1 * np.ones((N, M, L), "intc"))

            # 8 Corners
            for iNode in range(8):
                curNode = self.nodeLink[iVol][iNode]
                lIndex[ii] = setNodeValue(lIndex[ii], nodeIndex[curNode], iNode)

            # 12 Edges
            for iEdge in range(12):
                curEdge = self.edgeLink[iVol][iEdge]
                edgeDir = self.edgeDir[iVol][iEdge]
                lIndex[ii] = setEdgeValue(lIndex[ii], edgeIndex[curEdge], edgeDir, iEdge)
            # 6 Faces
            for iFace in range(6):
                curFace = self.faceLink[iVol][iFace]
                faceDir = self.faceDirRev[iVol][iFace]
                lIndex[ii] = setFaceValue(lIndex[ii], faceIndex[curFace], faceDir, iFace)
            # Interior
            toAdd = (N - 2) * (M - 2) * (L - 2)

            lIndex[ii][1 : N - 1, 1 : M - 1, 1 : L - 1] = np.arange(counter, counter + toAdd).reshape(
                (N - 2, M - 2, L - 2)
            )
            counter = counter + toAdd
        # end for

        if gIndex:
            # We must add [] for each global node
            gIndex = [[] for i in range(counter)]

            for ii in range(len(volumeList)):
                iVol = volumeList[ii]
                N = sizes[ii][0]
                M = sizes[ii][1]
                L = sizes[ii][2]

                for i in range(N):
                    for j in range(M):
                        for k in range(L):
                            gIndex[lIndex[ii][i, j, k]].append([iVol, i, j, k])
        else:
            gIndex = None

        self.nGlobal = counter
        self.gIndex = gIndex
        self.lIndex = lIndex

        if greedyReorder:

            # Reorder the indices with a greedy scheme
            newIndices = np.zeros(len(gIndex), "intc")
            newIndices[:] = -1
            newGIndex = [[] for i in range(len(gIndex))]
            counter = 0

            # Re-order the lIndex
            for ii in range(len(volumeList)):
                ivol = volumeList[ii]
                N = sizes[ii][0]
                M = sizes[ii][1]
                L = sizes[ii][2]
                for i in range(N):
                    for j in range(M):
                        for k in range(L):
                            if newIndices[lIndex[ii][i, j, k]] == -1:
                                newIndices[lIndex[ii][i, j, k]] = counter
                                lIndex[ii][i, j, k] = counter
                                counter += 1
                            else:
                                lIndex[ii][i, j, k] = newIndices[lIndex[ii][i, j, k]]

            # Re-order the gIndex
            for ii in range(len(gIndex)):
                ivol = gIndex[ii][0][0]
                i = gIndex[ii][0][1]
                j = gIndex[ii][0][2]
                k = gIndex[ii][0][3]
                pt = lIndex[ivol][i, j, k]
                newGIndex[pt] = gIndex[ii]

            self.gIndex = newGIndex
            self.lIndex = lIndex
        # end if (greedy reorder)

    def reOrder(self, reOrderList):
        """This function takes as input a permutation list which is used to reorder the entities in the topology object"""

        # Class atributates that possible need to be modified
        for i in range(8):
            self.nodeLink[:, i] = self.nodeLink[:, i].take(reOrderList)

        for i in range(12):
            self.edgeLink[:, i] = self.edgeLink[:, i].take(reOrderList)
            self.edgeDir[:, i] = self.edgeDir[:, i].take(reOrderList)

        for i in range(6):
            self.faceLink[:, i] = self.faceLink[:, i].take(reOrderList)
            self.faceDir[:, i] = self.faceDir[:, i].take(reOrderList)
