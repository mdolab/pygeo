# External modules
import numpy as np

# Local modules
from ..geo_utils import convertTo1D


class geoDV:
    def __init__(self, name, value, nVal, lower, upper, scale):
        self.name = name
        self.value = value
        self.nVal = nVal
        self.lower = self.upper = self.scale = None

        if lower is not None:
            self.lower = convertTo1D(lower, self.nVal)
        if upper is not None:
            self.upper = convertTo1D(upper, self.nVal)
        if scale is not None:
            self.scale = convertTo1D(scale, self.nVal)


class geoDVGlobal(geoDV):
    def __init__(self, name, value, lower, upper, scale, function, config):
        """
        Create a geometric design variable (or design variable group)
        See addGlobalDV in DVGeometry class for more information
        """
        value = np.atleast_1d(np.array(value)).astype("D")
        super().__init__(
            name=name,
            value=value,
            nVal=len(value),
            lower=lower,
            upper=upper,
            scale=scale,
        )

        self.config = config
        self.function = function

    def __call__(self, geo, config):
        """When the object is called, actually apply the function"""
        # Run the user-supplied function
        d = np.dtype(complex)

        if self.config is None or config is None or any(c0 == config for c0 in self.config):
            # If the geo object is complex, which is indicated by .coef
            # being complex, run with complex numbers. Otherwise, convert
            # to real before calling. This eliminates casting warnings.
            if geo.coef.dtype == d or geo.complex:
                return self.function(self.value, geo)
            else:
                return self.function(np.real(self.value), geo)


class geoDVLocal(geoDV):
    def __init__(self, name, lower, upper, scale, axis, coefListIn, mask, config):
        """
        Create a set of geometric design variables which change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.
        See addLocalDV for more information
        """

        coefList = []
        # create a new coefficient list that excludes any values that are masked
        for i in range(len(coefListIn)):
            if not mask[coefListIn[i]]:
                coefList.append(coefListIn[i])

        N = len(axis)
        nVal = len(coefList) * N
        super().__init__(name=name, value=np.zeros(nVal, "D"), nVal=nVal, lower=lower, upper=upper, scale=scale)

        self.config = config
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


class geoDVSpanwiseLocal(geoDV):
    def __init__(self, name, lower, upper, scale, axis, vol_dv_to_coefs, mask, config):
        """
        Create a set of geometric design variables which change the shape
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

        nVal = len(self.dv_to_coefs)
        super().__init__(name=name, value=np.zeros(nVal, "D"), nVal=nVal, lower=lower, upper=upper, scale=scale)

        if "x" == axis.lower():
            self.axis = 0
        elif "y" == axis.lower():
            self.axis = 1
        elif "z" == axis.lower():
            self.axis = 2
        else:
            raise NotImplementedError

        self.config = config

    def __call__(self, coef, config):
        """
        When the object is called, apply the design variable values to coefficients
        """
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


class geoDVSectionLocal(geoDV):
    def __init__(self, name, lower, upper, scale, axis, coefListIn, mask, config, sectionTransform, sectionLink):
        """
        Create a set of geometric design variables which change the shape
        of a surface.
        See `addLocalSectionDV` for more information
        """
        self.coefList = []
        # create a new coefficient list that excludes any values that are masked
        for i in range(len(coefListIn)):
            if not mask[coefListIn[i]]:
                self.coefList.append(coefListIn[i])

        nVal = len(self.coefList)
        super().__init__(name=name, value=np.zeros(nVal, "D"), nVal=nVal, lower=lower, upper=upper, scale=scale)

        self.config = config

        self.sectionTransform = sectionTransform
        self.sectionLink = sectionLink

        self.axis = axis

    def __call__(self, coef, coefRotM, config):
        """
        When the object is called, apply the design variable values to coefficients
        """
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


class geoDVComposite(geoDV):
    def __init__(self, name, value, nVal, u, scale=1.0, s=None):
        """
        Create a set of design variables which are linear combinations of existing design variables.
        """
        super().__init__(name=name, value=value, nVal=nVal, lower=None, upper=None, scale=convertTo1D(scale, nVal))
        self.u = u
        self.s = s


class espDV(geoDV):
    def __init__(self, csmDesPmtr, name, value, lower, upper, scale, rows, cols, dh, globalstartind):
        """
        Internal class for storing ESP design variable information
        """
        nVal = len(rows) * len(cols)
        super().__init__(name=name, value=np.array(value), nVal=nVal, lower=lower, upper=upper, scale=scale)

        self.csmDesPmtr = csmDesPmtr
        self.rows = rows
        self.cols = cols
        self.nVal = len(rows) * len(cols)
        self.dh = dh
        self.globalStartInd = globalstartind


class vspDV(geoDV):
    def __init__(self, parmID, dvName, component, group, parm, value, lower, upper, scale, dh):
        """
        Internal class for storing VSP design variable information
        """
        super().__init__(
            name=dvName, value=np.atleast_1d(np.array(value)), nVal=1, lower=lower, upper=upper, scale=scale
        )
        self.parmID = parmID
        self.component = component
        self.group = group
        self.parm = parm
        self.dh = dh


class cstDV(geoDV):
    def __init__(self, name, value, nVal, lower, upper, scale, dvType):
        """
        Internal class for storing CST design variable information
        """
        super().__init__(name=name, value=value, nVal=nVal, lower=lower, upper=upper, scale=scale)
        self.type = dvType
