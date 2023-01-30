# External modules
import numpy as np
from pyspline.utils import bilinearSurface


class PointSelect:
    def __init__(self, psType, *args, **kwargs):

        """Initialize a control point selection class. There are several ways
        to initialize this class depending on the 'type' qualifier:

        Parameters
        ----------
        psType : {'x', 'y', 'z', 'quad', 'ijkBounds', 'corners', 'list'}
            indicates the initialization type:

            - 'x': Define two corners (pt1=,pt2=) on a plane parallel to the x=0 plane
            - 'y': Define two corners (pt1=,pt2=) on a plane parallel to the y=0 plane
            - 'z': Define two corners (pt1=,pt2=) on a plane parallel to the z=0 plane
            - 'quad': Define four corners (pt1=,pt2=,pt3=,pt4=) in a counter-clockwise orientation
            - 'ijkBounds': Dictionary of int[3x2] defining upper and lower block indices to which we will apply the DVs.
              It should follow this format:

              .. code-blocks: python
                ijkBounds = {volID:[[ilow, ihigh],
                                    [jlow, jhigh],
                                    [klow, khigh]]}

              volID is the same block identifier used in volList.
              If the user provides none, then we will apply the normal DVs to all FFD nodes.
              This is how you call PointSelect for ijkBounds:

              .. code-blocks: python
                ps = PointSelect('ijkBounds',ijkBounds =  {volID:[[ilow, ihigh],
                                                                 [jlow, jhigh],
                                                                 [klow, khigh]]})

              Then to get the point indices you need to use ``ps.getPointsijk(FFD)``
            - 'list': Define the indices of a list that will be used to extract the points
        """

        if psType == "x" or psType == "y" or psType == "z":
            if not ("pt1" in kwargs and "pt2" in kwargs):
                raise ValueError(
                    "Two points must be specified with initialization type x, y, or z. "
                    + "Points are specified with kwargs pt1=[x1,y1,z1], pt2=[x2,y2,z2]"
                )
        elif psType == "quad":
            if not ("pt1" in kwargs and "pt2" in kwargs and "pt3" in kwargs and "pt4" in kwargs):
                raise ValueError(
                    "Four points must be specified with initialization type quad. "
                    + "Points are specified with kwargs pt1=[x1,y1,z1], pt2=[x2,y2,z2], pt3=[x3,y3,z3], pt4=[x4,y4,z4]"
                )

        elif psType == "ijkBounds":
            if not ("ijkBounds" in kwargs):
                raise ValueError(
                    "ijkBounds selection method requires a dictonary with the specific ijkBounds for each volume."
                )

        corners = np.zeros([4, 3])
        if psType in ["x", "y", "z", "corners"]:
            if psType == "x":
                corners[0] = kwargs["pt1"]

                corners[1][1] = kwargs["pt2"][1]
                corners[1][2] = kwargs["pt1"][2]

                corners[2][1] = kwargs["pt1"][1]
                corners[2][2] = kwargs["pt2"][2]

                corners[3] = kwargs["pt2"]

                corners[:, 0] = 0.5 * (kwargs["pt1"][0] + kwargs["pt2"][0])

            elif psType == "y":
                corners[0] = kwargs["pt1"]

                corners[1][0] = kwargs["pt2"][0]
                corners[1][2] = kwargs["pt1"][2]

                corners[2][0] = kwargs["pt1"][0]
                corners[2][2] = kwargs["pt2"][2]

                corners[3] = kwargs["pt2"]

                corners[:, 1] = 0.5 * (kwargs["pt1"][1] + kwargs["pt2"][1])

            elif psType == "z":
                corners[0] = kwargs["pt1"]

                corners[1][0] = kwargs["pt2"][0]
                corners[1][1] = kwargs["pt1"][1]

                corners[2][0] = kwargs["pt1"][0]
                corners[2][1] = kwargs["pt2"][1]

                corners[3] = kwargs["pt2"]

                corners[:, 2] = 0.5 * (kwargs["pt1"][2] + kwargs["pt2"][2])

            elif psType == "quad":
                corners[0] = kwargs["pt1"]
                corners[1] = kwargs["pt2"]
                corners[2] = kwargs["pt4"]  # Note the switch here from
                # CC orientation
                corners[3] = kwargs["pt3"]

            X = corners

            self.box = bilinearSurface(X)
            self.type = "box"

        elif psType == "list":
            self.box = None
            self.type = "list"
            self.indices = np.array(args[0])

            # Check if the list is unique:
            if len(self.indices) != len(np.unique(self.indices)):
                raise ValueError("The indices provided to pointSelect are not unique.")

        elif psType == "ijkBounds":

            self.ijkBounds = kwargs["ijkBounds"]  # Store the ijk bounds dictionary
            self.type = "ijkBounds"

    def getPoints(self, points):

        """Take in a list of points and return the ones that statify
        the point select class."""
        ptList = []
        indList = []
        if self.type == "box":
            for i in range(len(points)):
                u0, v0, _ = self.box.projectPoint(points[i])
                if 0 < u0 < 1 and 0 < v0 < 1:  # Its Inside
                    ptList.append(points[i])
                    indList.append(i)

        elif self.type == "list":
            for i in range(len(self.indices)):
                ptList.append(points[self.indices[i]])

            indList = self.indices.copy()

        elif self.type == "ijkBounds":
            raise NameError(
                "Use PointSelect.getPoints_ijk() to return indices of an object initialized with ijkBounds."
            )

        return ptList, indList

    def getPoints_ijk(self, DVGeo):

        """Receives a DVGeo object (with an embedded FFD) and uses the ijk bounds specified in the initialization to extract
        the corresponding indices.

        You can only use this method if you initialized PointSelect with 'ijkBounds' option.

        DVGeo : DVGeo object"""

        # Initialize list to hold indices in the DVGeo ordering
        indList = []

        # Loop over every dictionary entry to get cooresponding indices
        for iVol in self.ijkBounds:

            # Get current bounds
            ilow = self.ijkBounds[iVol][0][0]
            ihigh = self.ijkBounds[iVol][0][1]
            jlow = self.ijkBounds[iVol][1][0]
            jhigh = self.ijkBounds[iVol][1][1]
            klow = self.ijkBounds[iVol][2][0]
            khigh = self.ijkBounds[iVol][2][1]

            # Retrieve current points
            indList.extend(DVGeo.FFD.topo.lIndex[iVol][ilow:ihigh, jlow:jhigh, klow:khigh].flatten())

        # Now get the corresponding coordinates
        ptList = [DVGeo.FFD.coef[ii] for ii in indList]

        return ptList, indList
