# External modules
import numpy as np

# Local modules
from .. import geo_utils
from .baseConstraint import GeometricConstraint


class ThicknessConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of thickness
    constraints. One of these objects is created each time a
    addThicknessConstraints2D or addThicknessConstraints1D call is
    made. The user should not have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords) // 2, lower, upper, scale, DVGeo, addToPyOpt)

        self.coords = coords
        self.scaled = scaled

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths
        self.D0 = np.zeros(self.nCon)
        for i in range(self.nCon):
            self.D0[i] = geo_utils.norm.euclideanNorm(self.coords[2 * i] - self.coords[2 * i + 1])

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        D = np.zeros(self.nCon)
        for i in range(self.nCon):
            D[i] = geo_utils.norm.euclideanNorm(self.coords[2 * i] - self.coords[2 * i + 1])
            if self.scaled:
                D[i] /= self.D0[i]
        funcs[self.name] = D

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dTdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))

            for i in range(self.nCon):
                p1b, p2b = geo_utils.eDist_b(self.coords[2 * i, :], self.coords[2 * i + 1, :])
                if self.scaled:
                    p1b /= self.D0[i]
                    p2b /= self.D0[i]
                dTdPt[i, 2 * i, :] = p1b
                dTdPt[i, 2 * i + 1, :] = p2b

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dTdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) // 2))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class ProjectedThicknessConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of projected thickness
    constraints. One of these objects is created each time a
    addThicknessConstraints2D or addThicknessConstraints1D call is
    made. The user should not have to deal with this class directly.

    This is different from ThicknessConstraints becuase it measures the projected
    thickness along the orginal direction of the constraint.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords) // 2, lower, upper, scale, DVGeo, addToPyOpt)

        self.coords = coords
        self.scaled = scaled

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths and directions
        self.D0 = np.zeros(self.nCon)
        self.dir_vec = np.zeros((self.nCon, 3))
        for i in range(self.nCon):
            vec = self.coords[2 * i] - self.coords[2 * i + 1]
            self.D0[i] = geo_utils.norm.euclideanNorm(vec)
            self.dir_vec[i] = vec / self.D0[i]

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        D = np.zeros(self.nCon)
        for i in range(self.nCon):
            vec = self.coords[2 * i] - self.coords[2 * i + 1]

            # take the dot product with the direction vector
            D[i] = vec[0] * self.dir_vec[i, 0] + vec[1] * self.dir_vec[i, 1] + vec[2] * self.dir_vec[i, 2]

            if self.scaled:
                D[i] /= self.D0[i]

        funcs[self.name] = D

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dTdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))
            for i in range(self.nCon):
                D_b = 1.0

                # the reverse mode seeds still need to be scaled
                if self.scaled:
                    D_b /= self.D0[i]

                # d(dot(vec,n))/d(vec) = n
                # where vec = thickness vector
                #   and  n = the reference direction
                #  This is easier to see if you write out the dot product
                # dot(vec, n) = vec_1*n_1 + vec_2*n_2 + vec_3*n_3
                # d(dot(vec,n))/d(vec_1) = n_1
                # d(dot(vec,n))/d(vec_2) = n_2
                # d(dot(vec,n))/d(vec_3) = n_3
                vec_b = self.dir_vec[i] * D_b

                # the reverse mode of calculating vec is just scattering the seed of vec_b to the coords
                # vec = self.coords[2 * i] - self.coords[2 * i + 1]
                # we just set the coordinate seeds directly into the jacobian
                dTdPt[i, 2 * i, :] = vec_b
                dTdPt[i, 2 * i + 1, :] = -vec_b

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dTdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) // 2))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))

        # create a seperate zone to plot the projected direction for each thickness constraint
        handle.write("Zone T=%s_ref_directions\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.dir_vec) * 2, len(self.dir_vec)))
        handle.write("DATAPACKING=POINT\n")

        for i in range(self.nCon):
            pt1 = self.coords[i * 2 + 1]
            pt2 = pt1 + self.dir_vec[i]
            handle.write(f"{pt1[0]:f} {pt1[1]:f} {pt1[2]:f}\n")
            handle.write(f"{pt2[0]:f} {pt2[1]:f} {pt2[2]:f}\n")

        for i in range(self.nCon):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class ThicknessToChordConstraint(GeometricConstraint):
    """
    ThicknessToChordConstraint represents of a set of
    thickess-to-chord ratio constraints. One of these objects is
    created each time a addThicknessToChordConstraints2D or
    addThicknessToChordConstraints1D call is made. The user should not
    have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords) // 4, lower, upper, scale, DVGeo, addToPyOpt)
        self.coords = coords

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths
        self.ToC0 = np.zeros(self.nCon)
        for i in range(self.nCon):
            t = np.linalg.norm(self.coords[4 * i] - self.coords[4 * i + 1])
            c = np.linalg.norm(self.coords[4 * i + 2] - self.coords[4 * i + 3])
            self.ToC0[i] = t / c

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        ToC = np.zeros(self.nCon)
        for i in range(self.nCon):
            t = geo_utils.eDist(self.coords[4 * i], self.coords[4 * i + 1])
            c = geo_utils.eDist(self.coords[4 * i + 2], self.coords[4 * i + 3])
            ToC[i] = (t / c) / self.ToC0[i]

        funcs[self.name] = ToC

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dToCdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))

            for i in range(self.nCon):
                t = geo_utils.eDist(self.coords[4 * i], self.coords[4 * i + 1])
                c = geo_utils.eDist(self.coords[4 * i + 2], self.coords[4 * i + 3])

                p1b, p2b = geo_utils.eDist_b(self.coords[4 * i, :], self.coords[4 * i + 1, :])
                p3b, p4b = geo_utils.eDist_b(self.coords[4 * i + 2, :], self.coords[4 * i + 3, :])

                dToCdPt[i, 4 * i, :] = p1b / c / self.ToC0[i]
                dToCdPt[i, 4 * i + 1, :] = p2b / c / self.ToC0[i]
                dToCdPt[i, 4 * i + 2, :] = (-p3b * t / c**2) / self.ToC0[i]
                dToCdPt[i, 4 * i + 3, :] = (-p4b * t / c**2) / self.ToC0[i]

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dToCdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) // 2))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class ProximityConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of proximity
    constraints. The user should not have to deal with this
    class directly.
    """

    def __init__(
        self,
        name,
        coordsA,
        coordsB,
        pointSetKwargsA,
        pointSetKwargsB,
        lower,
        upper,
        scaled,
        scale,
        DVGeo,
        addToPyOpt,
        compNames,
    ):
        super().__init__(name, len(coordsA), lower, upper, scale, DVGeo, addToPyOpt)

        self.coordsA = coordsA
        self.coordsB = coordsB
        self.scaled = scaled

        # First thing we can do is embed the coordinates into the DVGeo.
        # ptsets A and B get different kwargs
        self.DVGeo.addPointSet(self.coordsA, f"{self.name}_A", compNames=compNames, **pointSetKwargsA)
        self.DVGeo.addPointSet(self.coordsB, f"{self.name}_B", compNames=compNames, **pointSetKwargsB)

        # Now get the reference lengths
        self.D0 = np.zeros(self.nCon)
        for i in range(self.nCon):
            self.D0[i] = geo_utils.norm.euclideanNorm(self.coordsA[i] - self.coordsB[i])

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coordsA = self.DVGeo.update(f"{self.name}_A", config=config)
        self.coordsB = self.DVGeo.update(f"{self.name}_B", config=config)
        D = np.zeros(self.nCon)
        for i in range(self.nCon):
            D[i] = geo_utils.norm.euclideanNorm(self.coordsA[i] - self.coordsB[i])
            if self.scaled:
                D[i] /= self.D0[i]
        funcs[self.name] = D

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dTdPtA = np.zeros((self.nCon, self.nCon, 3))
            dTdPtB = np.zeros((self.nCon, self.nCon, 3))

            for i in range(self.nCon):
                pAb, pBb = geo_utils.eDist_b(self.coordsA[i], self.coordsB[i])
                if self.scaled:
                    pAb /= self.D0[i]
                    pBb /= self.D0[i]
                dTdPtA[i, i, :] = pAb
                dTdPtB[i, i, :] = pBb

            funcSensA = self.DVGeo.totalSensitivity(dTdPtA, f"{self.name}_A", config=config)
            funcSensB = self.DVGeo.totalSensitivity(dTdPtB, f"{self.name}_B", config=config)

            funcsSens[self.name] = {}
            for key, value in funcSensA.items():
                funcsSens[self.name][key] = value
            for key, value in funcSensB.items():
                if key in funcsSens[self.name]:
                    funcsSens[self.name][key] += value
                else:
                    funcsSens[self.name][key] = value

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coordsA) * 2, len(self.coordsA)))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coordsA)):
            handle.write(f"{self.coordsA[i, 0]:f} {self.coordsA[i, 1]:f} {self.coordsA[i, 2]:f}\n")
            handle.write(f"{self.coordsB[i, 0]:f} {self.coordsB[i, 1]:f} {self.coordsB[i, 2]:f}\n")

        for i in range(len(self.coordsA)):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))

class MaxThicknessToChordConstraint(GeometricConstraint):
    """
    MaxThicknessToChordConstraint represents of a set of
    absolute max thickess-to-chord ratio constraints.

    The user should not have to deal with this class directly.
    """

    def __init__(self, name, ncon, coords, lower, upper, scale,
                 DVGeo, addToPyOpt, compNames):
        
        super().__init__(name, ncon, lower, upper, scale, DVGeo, addToPyOpt)
        self.coords = coords

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths
        self.ToC0 = np.zeros(self.nCon)
        for i in range(self.nCon):
            t = np.linalg.norm(self.coords[4 * i] - self.coords[4 * i + 1])
            c = np.linalg.norm(self.coords[4 * i + 2] - self.coords[4 * i + 3])
            self.ToC0[i] = t / c

        self.KSCoeff = 300.
        

    def evalFunctions(self, funcs, config) :
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """

        # init KS function value and coefficients
        KS = np.zeros((self.nCon))
        KSCoeff = self.KSCoeff
        
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        
        nchord = int(len(self.coords) / self.nCon / 4)
        
        # Loop over the spanwise stations
        for i in range(self.nCon):
            
            # Loop over the chordwise stations
            for j in range(nchord) :

                idx = 4*(j+i*nchord)
            
                t = geo_utils.eDist(self.coords[idx], self.coords[idx + 1])
                c = geo_utils.eDist(self.coords[idx + 2], self.coords[idx + 3])

                KS[i] += np.exp(KSCoeff * (t/c) )
                
            KS[i] = np.log(KS[i]) / KSCoeff
            
        funcs[self.name] = KS

        
    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        # KS Coeff
        KSCoeff = self.KSCoeff

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dKSdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))

            nchord = int(len(self.coords) / self.nCon / 4)

            for i in range(self.nCon):
                KS = 0.0
                dToCdPt = np.zeros((4, 3))
                
                # Loop over the chordwise stations
                for j in range(nchord) :
                    
                    idx = 4*(j+i*nchord)

                    # Thickness and chord valuves
                    t = geo_utils.eDist(self.coords[idx], self.coords[idx + 1])
                    c = geo_utils.eDist(self.coords[idx + 2], self.coords[idx + 3])

                    # KS function value
                    val = np.exp(KSCoeff * (t/c) )
                    KS += val
                    
                    # Derivatives of points
                    p1b, p2b = geo_utils.eDist_b(self.coords[idx, :], self.coords[idx + 1, :])
                    p3b, p4b = geo_utils.eDist_b(self.coords[idx + 2, :], self.coords[idx + 3, :])

                    # Derivative of ratios
                    dToCdPt[0, :] = p1b / c 
                    dToCdPt[1, :] = p2b / c
                    dToCdPt[2, :] = (-p3b * t / c**2)
                    dToCdPt[3, :] = (-p4b * t / c**2)

                    # Intermediate derivative of the KS function
                    dKSdPt[i, idx:idx+4, :] += val * dToCdPt
                    
                # Finalize the KS derivative
                dKSdPt[i, :, :] /= KS

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dKSdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) // 2))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))
