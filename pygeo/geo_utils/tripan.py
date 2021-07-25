import numpy as np
import os
from .bilinear_map import getBiLinearMap

# --------------------------------------------------------------
#                Tripan-related class
# --------------------------------------------------------------


def createTriPanMesh(geo, tripanFile, wakeFile, specsFile=None, defaultSize=0.1):
    """
    Create a TriPan mesh from a pyGeo object.

    geo:          The pyGeo object
    tripanFile:  The name of the TriPan File
    wakeFile:    The name of the wake file
    specsFile:   The specification of panels/edge and edge types
    defaultSize: If no specsFile is given, attempt to make edges with
    defaultSize-length panels

    This cannot be run in parallel!
    """

    # Use the topology of the entire geo object
    topo = geo.topo

    nEdge = topo.nEdge
    nFace = topo.nFace

    # Face orientation
    faceOrientation = [1] * nFace
    # edgeNumber == number of panels along a given edge
    edgeNumber = -1 * np.ones(nEdge, "intc")
    # edgeType == what type of parametrization to use along an edge
    edgeType = ["linear"] * nEdge
    wakeEdges = []
    wakeDir = []

    if specsFile:
        f = open(specsFile, "r")
        line = f.readline().split()
        if int(line[0]) != nFace:
            print("Number of faces do not match in specs file")
        if int(line[1]) != nEdge:
            print("Number of edges do not match in specs file")
        # Discard a line
        f.readline()
        # Read in the face info
        for iface in range(nFace):
            aux = f.readline().split()
            faceOrientation[iface] = int(aux[1])
        f.readline()
        # Read in the edge info
        for iedge in range(nEdge):
            aux = f.readline().split()
            edgeNumber[iedge] = int(aux[1])
            edgeType[iedge] = aux[2]
            if int(aux[5]) > 0:
                wakeEdges.append(iedge)
                wakeDir.append(1)
            elif int(aux[5]) < 0:
                wakeEdges.append(iedge)
                wakeDir.append(-1)
        f.close()
    else:
        defaultSize = float(defaultSize)
        # First Get the default number on each edge

        for iface in range(nFace):
            for iedge in range(4):
                # First check if we even have to do it
                if edgeNumber[topo.edgeLink[iface][iedge]] == -1:
                    # Get the physical length of the edge
                    edgeLength = geo.surfs[iface].edgeCurves[iedge].getLength()

                    # Using defaultSize calculate the number of panels
                    # along this edge
                    edgeNumber[topo.edgeLink[iface][iedge]] = int(np.floor(edgeLength / defaultSize)) + 2
    # end if

    # Create the sizes Geo for the make consistent function
    sizes = []
    order = [0] * nFace
    for iface in range(nFace):
        sizes.append([edgeNumber[topo.edgeLink[iface][0]], edgeNumber[topo.edgeLink[iface][2]]])

    sizes, edgeNumber = topo.makeSizesConsistent(sizes, order)

    # Now we need to get the edge parameter spacing for each edge
    topo.calcGlobalNumbering(sizes)  # This gets gIndex,lIndex and counter

    # Now calculate the intrinsic spacing for each edge:
    edgePara = []
    for iedge in range(nEdge):
        if edgeType[iedge] == "linear":
            spacing = np.linspace(0.0, 1.0, edgeNumber[iedge])
            edgePara.append(spacing)
        elif edgeType[iedge] == "cos":
            spacing = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, edgeNumber[iedge])))
            edgePara.append(spacing)
        elif edgeType[iedge] == "hyperbolic":
            x = np.linspace(0.0, 1.0, edgeNumber[iedge])
            beta = 1.8
            spacing = x - beta * x * (x - 1.0) * (x - 0.5)
            edgePara.append(spacing)
        else:
            print(
                "Warning: Edge type %s not understood. \
Using a linear type"
                % (edgeType[iedge])
            )
            edgePara.append(np.linspace(0, 1, edgeNumber[iedge]))

    # Get the number of panels
    nPanels = 0
    nNodes = len(topo.gIndex)
    for iface in range(nFace):
        nPanels += (sizes[iface][0] - 1) * (sizes[iface][1] - 1)

    # Open the outputfile
    fp = open(tripanFile, "w")

    # Write he number of points and panels
    fp.write("%5d %5d\n" % (nNodes, nPanels))

    # Output the Points First
    UV = []
    for iface in range(nFace):
        UV.append(
            getBiLinearMap(
                edgePara[topo.edgeLink[iface][0]], edgePara[topo.edgeLink[iface][1]], edgePara[topo.edgeLink[iface][2]], edgePara[topo.edgeLink[iface][3]],
            )
        )

    for ipt in range(len(topo.gIndex)):
        iface = topo.gIndex[ipt][0][0]
        i = topo.gIndex[ipt][0][1]
        j = topo.gIndex[ipt][0][2]
        pt = geo.surfs[iface].getValue(UV[iface][i, j][0], UV[iface][i, j][1])
        fp.write("%12.10e %12.10e %12.10e \n" % (pt[0], pt[1], pt[2]))

    # Output the connectivity Next
    for iface in range(nFace):
        if faceOrientation[iface] >= 0:
            for i in range(sizes[iface][0] - 1):
                for j in range(sizes[iface][1] - 1):
                    fp.write(
                        "%d %d %d %d \n"
                        % (topo.lIndex[iface][i, j], topo.lIndex[iface][i + 1, j], topo.lIndex[iface][i + 1, j + 1], topo.lIndex[iface][i, j + 1],)
                    )
        else:
            for i in range(sizes[iface][0] - 1):
                for j in range(sizes[iface][1] - 1):
                    fp.write(
                        "%d %d %d %d \n"
                        % (topo.lIndex[iface][i, j], topo.lIndex[iface][i, j + 1], topo.lIndex[iface][i + 1, j + 1], topo.lIndex[iface][i + 1, j],)
                    )

    fp.write("\n")
    fp.close()

    # Output the wake file
    fp = open(wakeFile, "w")
    fp.write("%d\n" % (len(wakeEdges)))
    print("wakeEdges:", wakeEdges)

    for k in range(len(wakeEdges)):
        # Get a surface/edge for this edge
        surfaces = topo.getSurfaceFromEdge(wakeEdges[k])
        iface = surfaces[0][0]
        iedge = surfaces[0][1]
        if iedge == 0:
            indices = topo.lIndex[iface][:, 0]
        elif iedge == 1:
            indices = topo.lIndex[iface][:, -1]
        elif iedge == 2:
            indices = topo.lIndex[iface][0, :]
        elif iedge == 3:
            indices = topo.lIndex[iface][-1, :]

        fp.write("%d\n" % (len(indices)))

        if wakeDir[k] > 0:
            for i in range(len(indices)):
                # A constant in TriPan to indicate projected wake
                teNodeType = 3
                fp.write("%d %d\n" % (indices[i], teNodeType))
        else:
            for i in range(len(indices)):
                teNodeType = 3
                fp.write("%d %d\n" % (indices[len(indices) - 1 - i], teNodeType))
    # end for
    fp.close()

    # Write out the default specFile
    if specsFile is None:
        (dirName, fileName) = os.path.split(tripanFile)
        (fileBaseName, fileExtension) = os.path.splitext(fileName)
        if dirName != "":
            newSpecsFile = dirName + "/" + fileBaseName + ".specs"
        else:
            newSpecsFile = fileBaseName + ".specs"

        specsFile = newSpecsFile

    if not os.path.isfile(specsFile):
        f = open(specsFile, "w")
        f.write("%d %d Number of faces and number of edges\n" % (nFace, nEdge))
        f.write(
            "Face number   Normal (1 for regular, -1 for\
 reversed orientation\n"
        )
        for iface in range(nFace):
            f.write("%d %d\n" % (iface, faceOrientation[iface]))
        f.write(
            "Edge Number #Node Type     Start Space   End Space\
   WakeEdge\n"
        )
        for iedge in range(nEdge):
            if iedge in wakeEdges:
                f.write("  %4d    %5d %10s %10.4f %10.4f  %1d \n" % (iedge, edgeNumber[iedge], edgeType[iedge], 0.1, 0.1, 1))
            else:
                f.write("  %4d    %5d %10s %10.4f %10.4f  %1d \n" % (iedge, edgeNumber[iedge], edgeType[iedge], 0.1, 0.1, 0))
        f.close()
