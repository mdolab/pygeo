# ======================================================================
#         Imports
# ======================================================================
import numpy as np
from pygeo.geo_utils import convertTo1D


class geoDVLocal(object):
    def __init__(self, dvName, lower, upper, scale, axis, coefListIn, mask, config):

        """Create a set of geometric design variables which change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.
        See addLocalDV for more information
        """

        coefList = []
        # create a new coefficent list that excludes any values that are masked
        for i in range(len(coefListIn)):
            if not mask[coefListIn[i]]:
                coefList.append(coefListIn[i])

        N = len(axis)
        self.nVal = len(coefList) * N
        self.value = np.zeros(self.nVal, "D")
        self.name = dvName
        self.lower = None
        self.upper = None
        self.config = config
        if lower is not None:
            self.lower = convertTo1D(lower, self.nVal)
        if upper is not None:
            self.upper = convertTo1D(upper, self.nVal)
        if scale is not None:
            self.scale = convertTo1D(scale, self.nVal)

        self.coefList = np.zeros((self.nVal, 2), "intc")
        j = 0

        for i in range(len(coefList)):
            if "x" in axis.lower():
                self.coefList[j] = [coefList[i], 0]
                j += 1
            elif "y" in axis.lower():
                self.coefList[j] = [coefList[i], 1]
                j += 1
            elif "z" in axis.lower():
                self.coefList[j] = [coefList[i], 2]
                j += 1

    def __call__(self, coef, config):
        """When the object is called, apply the design variable values to
        coefficients"""
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(self.nVal):
                coef[self.coefList[i, 0], self.coefList[i, 1]] += self.value[i].real

        return coef

    def updateComplex(self, coef, config):
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(self.nVal):
                coef[self.coefList[i, 0], self.coefList[i, 1]] += self.value[i].imag * 1j

        return coef

    def mapIndexSets(self, indSetA, indSetB):
        """
        Map the index sets from the full coefficient indices to the local set.
        """
        # Temp is the list of FFD coefficients that are included
        # as shape variables in this localDV "key"
        temp = self.coefList
        cons = []
        for j in range(len(indSetA)):
            # Try to find this index # in the coefList (temp)
            up = None
            down = None

            # Note: We are doing inefficient double looping here
            for k in range(len(temp)):
                if temp[k][0] == indSetA[j]:
                    up = k
                if temp[k][0] == indSetB[j]:
                    down = k

            # If we haven't found up AND down do nothing
            if up is not None and down is not None:
                cons.append([up, down])

        return cons


class geoDVSpanwiseLocal(geoDVLocal):
    def __init__(self, dvName, lower, upper, scale, axis, vol_dv_to_coefs, mask, config):

        """Create a set of geometric design variables which change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.
        See addLocalDV for more information
        """

        self.dv_to_coefs = []

        # add all the coefs to a flat array, but check that it isn't masked first
        for ivol in range(len(vol_dv_to_coefs)):
            for loc_dv in range(len(vol_dv_to_coefs[ivol])):
                coefs = vol_dv_to_coefs[ivol][loc_dv]

                loc_dv_to_coefs = []

                # loop through each of coefs to see if it is masked
                for coef in coefs:
                    if not mask[coef]:
                        loc_dv_to_coefs.append(coef)

                self.dv_to_coefs.append(loc_dv_to_coefs)

        if "x" == axis.lower():
            self.axis = 0
        elif "y" == axis.lower():
            self.axis = 1
        elif "z" == axis.lower():
            self.axis = 2
        else:
            raise NotImplementedError

        self.nVal = len(self.dv_to_coefs)
        self.value = np.zeros(self.nVal, "D")

        self.name = dvName
        self.lower = None
        self.upper = None
        self.config = config

        if lower is not None:
            self.lower = convertTo1D(lower, self.nVal)
        if upper is not None:
            self.upper = convertTo1D(upper, self.nVal)
        if scale is not None:
            self.scale = convertTo1D(scale, self.nVal)

    def __call__(self, coef, config):
        """When the object is called, apply the design variable values to
        coefficients"""
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(self.nVal):
                coef[self.dv_to_coefs[i], self.axis] += self.value[i].real

        return coef

    def updateComplex(self, coef, config):
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(self.nVal):
                coef[self.dv_to_coefs[i], self.axis] += self.value[i].imag * 1j

        return coef

    def mapIndexSets(self, indSetA, indSetB):
        """
        Map the index sets from the full coefficient indices to the local set.
        """

        cons = []
        for j in range(len(indSetA)):
            # Try to find this index # in the coefList (temp)
            up = None
            down = None

            # Note: We are doing inefficient double looping here
            for idx_dv, coefs in enumerate(self.dv_to_coefs):

                for coef in coefs:

                    if coef == indSetA[j]:
                        up = idx_dv
                    if coef == indSetB[j]:
                        down = idx_dv

            # If we haven't found up AND down do nothing
            if up is not None and down is not None:
                cons.append([up, down])

        return cons


class geoDVSectionLocal(object):
    def __init__(self, dvName, lower, upper, scale, axis, coefListIn, mask, config, sectionTransform, sectionLink):
        """
        Create a set of geometric design variables which change the shape
        of a surface.
        See `addLocalSectionDV` for more information
        """

        self.coefList = []
        # create a new coefficent list that excludes any values that are masked
        for i in range(len(coefListIn)):
            if not mask[coefListIn[i]]:
                self.coefList.append(coefListIn[i])

        self.nVal = len(self.coefList)
        self.value = np.zeros(self.nVal, "D")
        self.name = dvName
        self.lower = None
        self.upper = None
        self.config = config
        if lower is not None:
            self.lower = convertTo1D(lower, self.nVal)
        if upper is not None:
            self.upper = convertTo1D(upper, self.nVal)
        if scale is not None:
            self.scale = convertTo1D(scale, self.nVal)

        self.sectionTransform = sectionTransform
        self.sectionLink = sectionLink

        self.axis = axis

    def __call__(self, coef, coefRotM, config):
        """When the object is called, apply the design variable values to
        coefficients"""
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(len(self.coefList)):
                T = self.sectionTransform[self.sectionLink[self.coefList[i]]]
                inFrame = np.zeros(3)
                inFrame[self.axis] = self.value[i].real

                R = coefRotM[self.coefList[i]].real
                coef[self.coefList[i]] += R.dot(T.dot(inFrame))
        return coef

    def updateComplex(self, coef, coefRotM, config):
        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            for i in range(len(self.coefList)):
                T = self.sectionTransform[self.sectionLink[self.coefList[i]]]
                inFrame = np.zeros(3, "D")
                inFrame[self.axis] = self.value[i]

                R = coefRotM[self.coefList[i]]
                coef[self.coefList[i]] += R.dot(T.dot(inFrame)).imag * 1j
        return coef

    def mapIndexSets(self, indSetA, indSetB):
        """
        Map the index sets from the full coefficient indices to the local set.
        """
        # Temp is the list of FFD coefficients that are included
        # as shape variables in this localDV "key"
        temp = self.coefList
        cons = []
        for j in range(len(indSetA)):
            # Try to find this index # in the coefList (temp)
            up = None
            down = None

            # Note: We are doing inefficient double looping here
            for k in range(len(temp)):
                if temp[k] == indSetA[j]:
                    up = k
                if temp[k] == indSetB[j]:
                    down = k

            # If we haven't found up AND down do nothing
            if up is not None and down is not None:
                cons.append([up, down])

        return cons