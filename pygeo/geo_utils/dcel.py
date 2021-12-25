import numpy as np
import functools
from .misc import hangle, isLeft

# --------------------------------------------------------------
# 2D Doubly connected edge list implementation.
# Copyright 2008, Angel Yanguas-Gil
# --------------------------------------------------------------


class DCELEdge:
    def __init__(self, v1, v2, X, PID, uv, tag):
        # Create a representation of a surface edge that contains the
        # required information to be able to construct a trimming
        # curve on the orignal skin surfaces

        self.X = X
        self.PID = PID
        self.uv = uv
        tmp = tag.split("-")
        self.tag = tmp[0]
        if len(tmp) > 1:
            self.seg = tmp[1]
        else:
            self.seg = None

        self.v1 = v1
        self.v2 = v2
        if X is not None:
            self.x1 = 0.5 * (X[0, 0] + X[0, 1])
            self.x2 = 0.5 * (X[-1, 0] + X[-1, 1])

        self.con = [v1, v2]

    def __repr__(self):

        str1 = f"v1: {self.v1[0]:f} {self.v1[1]:f}\nv2: {self.v2[0]:f} {self.v2[1]:f}"
        return str1

    def midPt(self):
        # return [0.5*(self.v1.x + self.v2.x), 0.5*(self.v1.y + self.v2.y)]
        return 0.5 * self.x1 + 0.5 * self.x2


class DCELVertex:
    """Minimal implementation of a vertex of a 2D dcel"""

    def __init__(self, uv, X):
        self.x = uv[0]
        self.y = uv[1]
        self.X = X
        self.hedgelist = []

    def sortincident(self):
        key = functools.cmp_to_key(self.hsort)
        self.hedgelist.sort(key=key, reverse=True)

    def hsort(self, h1, h2):
        """Sorts two half edges counterclockwise"""

        if h1.angle < h2.angle:
            return -1
        elif h1.angle > h2.angle:
            return 1
        else:
            return 0


class DCELHedge:
    """Minimal implementation of a half-edge of a 2D dcel"""

    def __init__(self, v1, v2, PID, uv, tag=None):
        # The origin is defined as the vertex it points to
        self.origin = v2
        self.twin = None
        self.face = None
        self.sface = None
        self.uv = uv
        self.PID = PID
        self.nexthedge = None
        self.angle = hangle(v2.x - v1.x, v2.y - v1.y)
        self.prevhedge = None
        self.length = np.sqrt((v2.x - v1.x) ** 2 + (v2.y - v1.y) ** 2)
        self.tag = tag


class DCELFace:
    """Implements a face of a 2D dcel"""

    def __init__(self):
        self.wedge = None
        self.data = None
        self.external = None
        self.tag = "EXTERNAL"
        self.id = None
        self.centroid = None
        self.spatialCentroid = None

    def area(self):
        h = self.wedge
        a = 0
        while h.nexthedge is not self.wedge:
            p1 = h.origin
            p2 = h.nexthedge.origin
            a += p1.x * p2.y - p2.x * p1.y
            h = h.nexthedge

        p1 = h.origin
        p2 = self.wedge.origin
        a = (a + p1.x * p2.y - p2.x * p1.y) / 2.0
        return a

    def calcCentroid(self):
        h = self.wedge
        center = np.zeros(2)
        center += [h.origin.x, h.origin.y]
        counter = 1
        while h.nexthedge is not self.wedge:
            counter += 1
            h = h.nexthedge
            center += [h.origin.x, h.origin.y]

        self.centroid = center / counter

    def calcSpatialCentroid(self):

        h = self.wedge
        center = np.zeros(3)
        center += h.origin.X
        counter = 1
        while h.nexthedge is not self.wedge:
            counter += 1
            h = h.nexthedge
            center += h.origin.X

        self.spatialCentroid = center / counter

    def perimeter(self):
        h = self.wedge
        p = 0
        while h.nexthedge is not self.wedge:
            p += h.length
            h = h.nexthedge
        return p

    def vertexlist(self):
        h = self.wedge
        pl = [h.origin]
        while h.nexthedge is not self.wedge:
            h = h.nexthedge
            pl.append(h.origin)
        return pl

    def isinside(self, P):
        """Determines whether a point is inside a face using a
        winding formula"""
        pl = self.vertexlist()
        V = []
        for i in range(len(pl)):
            V.append([pl[i].x, pl[i].y])
        V.append([pl[0].x, pl[0].y])

        wn = 0
        # loop through all edges of the polygon
        for i in range(len(V) - 1):  # edge from V[i] to V[i+1]
            if V[i][1] <= P[1]:  # start y <= P[1]
                if V[i + 1][1] > P[1]:  # an upward crossing
                    if isLeft(V[i], V[i + 1], P) > 0:  # P left of edge
                        wn += 1  # have a valid up intersect
            else:  # start y > P[1] (no test needed)
                if V[i + 1][1] <= P[1]:  # a downward crossing
                    if isLeft(V[i], V[i + 1], P) < 0:  # P right of edge
                        wn -= 1  # have a valid down intersect
        if wn == 0:
            return False
        else:
            return True


class DCEL:
    """
    Implements a doubly-connected edge list
    """

    def __init__(self, vl=None, el=None, fileName=None):
        self.vertices = []
        self.hedges = []
        self.faces = []
        self.faceInfo = None
        if vl is not None and el is not None:
            self.vl = vl
            self.el = el
            self.buildDcel()
        elif fileName is not None:
            self.loadDCEL(fileName)
            self.buildDcel()
        else:
            # The user is going to manually create the hedges and
            # faces
            pass

    def buildDcel(self):
        """
        Creates the dcel from the list of vertices and edges
        """

        # Do some pruning first:
        self.vertices = self.vl
        ii = 0
        while ii < 1000:  # Trim at most 1000 edges
            ii += 1

            mult = np.zeros(self.nvertices(), "intc")
            for e in self.el:
                mult[e.con[0]] += 1
                mult[e.con[1]] += 1

            multCheck = mult < 2

            if np.any(multCheck):

                # We need to do a couple of things:
                # 1. The bad vertices need to be removed from the vertex list
                # 2. Remaning vertices must be renamed
                # 3. Edges that reference deleted vertices must be popped
                # 4. Remaining edges must have connectivity info updated

                # First generate new mapping:
                count = 0
                mapping = -1 * np.ones(self.nvertices(), "intc")
                deletedVertices = []
                for i in range(self.nvertices()):
                    if multCheck[i]:
                        # Vertex must be removed
                        self.vertices.pop(i - len(deletedVertices))
                        deletedVertices.append(i)
                    else:
                        mapping[i] = count  # Other wise get the mapping count:
                        count += 1

                # Now prune the edges:
                nEdgeDeleted = 0
                for i in range(len(self.el)):
                    if (
                        self.el[i - nEdgeDeleted].con[0] in deletedVertices
                        or self.el[i - nEdgeDeleted].con[1] in deletedVertices
                    ):
                        # Edge must be deleted
                        self.el.pop(i - nEdgeDeleted)
                        nEdgeDeleted += 1
                    else:
                        # Mapping needs to be updated:
                        curCon = self.el[i - nEdgeDeleted].con
                        self.el[i - nEdgeDeleted].con[0] = mapping[curCon[0]]
                        self.el[i - nEdgeDeleted].con[1] = mapping[curCon[1]]
            else:
                break

        # end while

        # Step 2: hedge list creation. Assignment of twins and
        # vertices

        self.hedges = []
        appendCount = 0

        for e in self.el:

            h1 = DCELHedge(self.vertices[e.con[0]], self.vertices[e.con[1]], e.PID, e.uv, e.tag)
            h2 = DCELHedge(self.vertices[e.con[1]], self.vertices[e.con[0]], e.PID, e.uv, e.tag)

            h1.twin = h2
            h2.twin = h1
            self.vertices[e.con[1]].hedgelist.append(h1)
            self.vertices[e.con[0]].hedgelist.append(h2)
            appendCount += 2
            self.hedges.append(h2)
            self.hedges.append(h1)

        # Step 3: Identification of next and prev hedges
        for v in self.vertices:
            v.sortincident()
            length = len(v.hedgelist)

            for i in range(length - 1):
                v.hedgelist[i].nexthedge = v.hedgelist[i + 1].twin
                v.hedgelist[i + 1].prevhedge = v.hedgelist[i]

            v.hedgelist[length - 1].nexthedge = v.hedgelist[0].twin
            v.hedgelist[0].prevhedge = v.hedgelist[length - 1]

        # Step 4: Face assignment
        provlist = self.hedges[:]
        nf = 0
        nh = len(self.hedges)

        while nh > 0:
            h = provlist.pop()
            nh -= 1

            # We check if the hedge already points to a face
            if h.face is None:
                f = DCELFace()
                nf += 1
                # We link the hedge to the new face
                f.wedge = h
                f.wedge.face = f
                # And we traverse the boundary of the new face
                while h.nexthedge is not f.wedge:
                    h = h.nexthedge
                    h.face = f
                self.faces.append(f)
        # And finally we have to determine the external face
        for f in self.faces:
            f.external = f.area() < 0
            f.calcCentroid()
            f.calcSpatialCentroid()

        if self.faceInfo is not None:
            for i in range(len(self.faceInfo)):
                self.faces[i].tag = self.faceInfo[i]

    def writeTecplot(self, fileName):

        f = open(fileName, "w")
        f.write('VARIABLES = "X","Y"\n')
        for i in range(len(self.el)):
            f.write('Zone T="edge%d" I=%d\n' % (i, 2))
            f.write("DATAPACKING=POINT\n")
            v1 = self.el[i].con[0]
            v2 = self.el[i].con[1]

            f.write(f"{self.vl[v1].x:g} {self.vl[v1].y:g}\n")

            f.write(f"{self.vl[v2].x:g} {self.vl[v2].y:g}\n")

        f.close()

    def findpoints(self, pl, onetoone=False):
        """Given a list of points pl, returns a list of
        with the corresponding face each point belongs to and
        None if it is outside the map.

        """

        ans = []
        if onetoone:
            fl = self.faces[:]
            for p in pl:
                found = False
                for f in fl:
                    if f.external:
                        continue
                    if f.isinside(p):
                        fl.remove(f)
                        found = True
                        ans.append(f)
                        break
                if not found:
                    ans.append(None)

        else:
            for p in pl:
                found = False
                for f in self.faces:
                    if f.external:
                        continue
                    if f.isinside(p):
                        found = True
                        ans.append(f)
                        break
                if not found:
                    ans.append(None)

        return ans

    def areas(self):
        return [f.area() for f in self.faces if not f.external]

    def perimeters(self):
        return [f.perimeter() for f in self.faces if not f.external]

    def nfaces(self):
        return len(self.faces)

    def nvertices(self):
        return len(self.vertices)

    def nedges(self):
        return len(self.hedges) // 2

    def saveDCEL(self, fileName):

        f = open(fileName, "w")
        f.write("%d %d %d\n" % (self.nvertices(), self.nedges(), self.nfaces()))
        for i in range(self.nvertices()):
            f.write(
                "%g %g %g %g %g \n"
                % (
                    self.vertices[i].x,
                    self.vertices[i].y,
                    self.vertices[i].X[0],
                    self.vertices[i].X[1],
                    self.vertices[i].X[2],
                )
            )

        for i in range(self.nedges()):
            if self.el[i].seg is not None:
                f.write(
                    "%d %d %g %g %g %g %g %g %s-%s\n"
                    % (
                        self.el[i].con[0],
                        self.el[i].con[1],
                        self.el[i].x1[0],
                        self.el[i].x1[1],
                        self.el[i].x1[2],
                        self.el[i].x2[0],
                        self.el[i].x2[1],
                        self.el[i].x2[2],
                        self.el[i].tag,
                        self.el[i].seg,
                    )
                )
            else:
                f.write(
                    "%d %d %g %g %g %g %g %g %s\n"
                    % (
                        self.el[i].con[0],
                        self.el[i].con[1],
                        self.el[i].x1[0],
                        self.el[i].x1[1],
                        self.el[i].x1[2],
                        self.el[i].x2[0],
                        self.el[i].x2[1],
                        self.el[i].x2[2],
                        self.el[i].tag,
                    )
                )

        for i in range(self.nfaces()):
            f.write("%s\n" % (self.faces[i].tag))
        f.close()

    def loadDCEL(self, fileName):

        f = open(fileName)
        # Read sizes
        tmp = f.readline().split()
        nvertices = int(tmp[0])
        nedges = int(tmp[1])
        nfaces = int(tmp[2])

        self.vl = []
        self.el = []
        self.faceInfo = []
        for _i in range(nvertices):
            a = f.readline().split()
            self.vl.append(DCELVertex([float(a[0]), float(a[1])], np.array([float(a[2]), float(a[3]), float(a[4])])))

        for _i in range(nedges):
            a = f.readline().split()
            self.el.append(DCELEdge(int(a[0]), int(a[1]), None, None, None, a[8]))
            self.el[-1].x1 = np.array([float(a[2]), float(a[3]), float(a[4])])
            self.el[-1].x2 = np.array([float(a[5]), float(a[6]), float(a[7])])

        for _i in range(nfaces):
            a = f.readline().split()
            self.faceInfo.append(a[0])

        f.close()
