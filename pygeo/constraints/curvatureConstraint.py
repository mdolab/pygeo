# ======================================================================
#         Imports
# ======================================================================
import numpy as np
from mpi4py import MPI
from scipy.sparse import csr_matrix
from baseclasses.utils import Error
from .baseConstraint import GeometricConstraint


class CurvatureConstraint1D(GeometricConstraint):
    """
    DVConstraints representation of a set of 1D curvature
    constraints. One of these objects is created each time a
    addCurvatureConstraints1D call is made.
    The user should not have to deal with this class directly.
    NOTE: the output is actually the square of the curvature
    """

    def __init__(
        self, name, curvatureType, coords, axis, eps, KSCoeff, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames
    ):
        super().__init__(name, 1, lower, upper, scale, DVGeo, addToPyOpt)
        self.curvatureType = curvatureType
        self.coords = coords
        self.axis = axis
        self.eps = eps
        self.KSCoeff = KSCoeff
        self.nPts = len(self.coords)
        self.scaled = scaled

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Calculate the reference curvatures
        self.C, self.KSC2Ref, self.meanC2Ref, self.maxC2 = self.calcCurvature2(
            self.coords, self.axis, self.nPts, self.eps, self.KSCoeff
        )

        if MPI.COMM_WORLD.rank == 0:
            print(
                "Reference squared-curvatures: KS: %f, mean: %f, max: %f" % (self.KSC2Ref, self.meanC2Ref, self.maxC2)
            )

    def calcCurvature2(self, coords, axis, nPts, eps, KSCoeff):
        """
        Calculate the curvature
        C = d^2y/dx^2
        Where C is the curvature, x is the distance between the points projected on to the design surface,
        y is the coordinates of the points projected to the axis direction.
        We calculate two types of curvatures,
        mean:       meanC2 = avg(C*C)
        aggregated: KSC2   = log(sum(KSCoeff*C*C))/KSCoeff
        We also print out the max curvature for these samples points. You need to tweak the KSCoeff to make sure
        The max curvature should be slightly lower than the KS curvature.
        NOTE: we always do a square of the curvatures to make sure they are positive
        """
        # project the coordinates to the axis direction, coordsP is a scalar now
        coordsP = np.zeros(nPts)
        for i in range(nPts):
            coordsP[i] = np.dot(coords[i], axis)

        # init KS, mean, and max curvatures
        KSC2 = 0.0
        meanC2 = 0.0
        maxC2 = -1e16
        C = np.zeros(nPts)

        # calculate curvatures
        # NOTE: we do not calculate the curvatures at the end points!!
        # this treatment allows us to use central FD for all points
        for i in range(1, nPts - 1):
            cFDC = (coordsP[i + 1] - 2 * coordsP[i] + coordsP[i - 1]) / eps / eps
            C[i] = cFDC
            KSC2 += np.exp(KSCoeff * C[i] * C[i])
            meanC2 += C[i] * C[i]
            if (C[i] * C[i]) > maxC2:
                maxC2 = C[i] * C[i]

        # Assign the values for the end points for visualization purpose. The curvature at the end points
        # are not used in the computation!!
        C[0] = C[1]
        C[nPts - 1] = C[nPts - 2]

        KSC2 = np.log(KSC2) / KSCoeff
        meanC2 = meanC2 / (nPts - 2)

        return C, KSC2, meanC2, maxC2

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

        self.C, KSC2, meanC2, self.maxC2 = self.calcCurvature2(
            self.coords, self.axis, self.nPts, self.eps, self.KSCoeff
        )

        if MPI.COMM_WORLD.rank == 0:
            print("Curvature squared-curvatures: KS: %f, mean: %f, max: %f" % (KSC2, meanC2, self.maxC2))

        if self.scaled:
            KSC2 /= self.KSC2Ref + 1e-16
            meanC2 /= self.meanC2Ref + 1e-16

            if MPI.COMM_WORLD.rank == 0:
                print("Normalized squared-curvatures: KS: %f, mean: %f, max: %f" % (KSC2, meanC2, self.maxC2))

        if self.curvatureType == "mean":
            funcs[self.name] = meanC2
        elif self.curvatureType == "aggregated":
            funcs[self.name] = KSC2
        else:
            raise Error("curvatureType=%s not supported! Options are: mean or aggregated" % self.curvatureType)

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        # we need to hand derive the derivatives of the curvature with respect to the projected points
        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dC2dPt = np.zeros((self.coords.shape[0], self.coords.shape[1]))
            if self.curvatureType == "mean":
                tmp = 2 / (self.nPts - 2) / self.eps / self.eps
                for i in range(self.nPts):
                    for j in range(3):
                        if i == 0:
                            dC2dPt[i][j] = self.axis[j] * tmp * (self.C[i + 1])
                        elif i == 1:
                            dC2dPt[i][j] = self.axis[j] * tmp * (-2 * self.C[i] + self.C[i + 1])
                        elif i == self.nPts - 1:
                            dC2dPt[i][j] = self.axis[j] * tmp * (self.C[i - 1])
                        elif i == self.nPts - 2:
                            dC2dPt[i][j] = self.axis[j] * tmp * (-2 * self.C[i] + self.C[i - 1])
                        else:
                            dC2dPt[i][j] = self.axis[j] * tmp * (self.C[i + 1] - 2 * self.C[i] + self.C[i - 1])
            elif self.curvatureType == "aggregated":
                eSum = 0.0
                for i in range(1, self.nPts - 1):
                    eSum += np.exp(self.KSCoeff * self.C[i] * self.C[i])
                tmp = 2 / eSum / self.eps / self.eps
                for i in range(self.nPts):
                    for j in range(3):
                        if i == 0:
                            dC2dPt[i][j] = (
                                self.axis[j]
                                * tmp
                                * (self.C[i + 1] * np.exp(self.KSCoeff * self.C[i + 1] * self.C[i + 1]))
                            )
                        elif i == 1:
                            dC2dPt[i][j] = (
                                self.axis[j]
                                * tmp
                                * (
                                    -2 * self.C[i] * np.exp(self.KSCoeff * self.C[i] * self.C[i])
                                    + self.C[i + 1] * np.exp(self.KSCoeff * self.C[i + 1] * self.C[i + 1])
                                )
                            )
                        elif i == self.nPts - 1:
                            dC2dPt[i][j] = (
                                self.axis[j]
                                * tmp
                                * (self.C[i - 1] * np.exp(self.KSCoeff * self.C[i - 1] * self.C[i - 1]))
                            )
                        elif i == self.nPts - 2:
                            dC2dPt[i][j] = (
                                self.axis[j]
                                * tmp
                                * (
                                    -2 * self.C[i] * np.exp(self.KSCoeff * self.C[i] * self.C[i])
                                    + self.C[i - 1] * np.exp(self.KSCoeff * self.C[i - 1] * self.C[i - 1])
                                )
                            )
                        else:
                            dC2dPt[i][j] = (
                                self.axis[j]
                                * tmp
                                * (
                                    self.C[i + 1] * np.exp(self.KSCoeff * self.C[i + 1] * self.C[i + 1])
                                    - 2 * self.C[i] * np.exp(self.KSCoeff * self.C[i] * self.C[i])
                                    + self.C[i - 1] * np.exp(self.KSCoeff * self.C[i - 1] * self.C[i - 1])
                                )
                            )
            else:
                raise Error("curvatureType=%s not supported! Options are: mean or aggregated" % self.curvatureType)

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dC2dPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of the projected points on the design surface
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) - 1))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) - 1):
            handle.write("%d %d\n" % (i + 1, i + 2))

        # NOTE: in addition to write the points, we create a new file to write the actual curvature value
        f = open("%s.dat" % self.name, "w")
        f.write('TITLE = "DVConstraints Data"\n')
        f.write('VARIABLES = "CoordinateX" "CoordinateY" "CoordinateZ" "Curvature"\n')
        f.write("Zone T=%s\n" % self.name)
        f.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) - 1))
        f.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            f.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f} {self.C[i]:f} \n")

        for i in range(len(self.coords) - 1):
            f.write("%d %d\n" % (i + 1, i + 2))
        f.close()


class CurvatureConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of the curvature constraint.
    One of these objects is created each time a addCurvatureConstraint is made.
    The user should not have to deal with this class directly.
    """

    def __init__(self, name, surfs, curvatureType, lower, upper, scaled, scale, KSCoeff, DVGeo, addToPyOpt, compNames):
        super().__init__(name, 1, lower, upper, scale, DVGeo, addToPyOpt)

        self.nSurfs = len(surfs)  # we support multiple surfaces (plot3D files)
        self.X = []
        self.X_map = []
        self.node_map = []
        self.coords = []
        for iSurf in range(self.nSurfs):
            # A list of the coordinates arrays for each surface, flattened in order
            # to vectorize operations
            self.X += [np.reshape(surfs[iSurf].X, -1)]
            # A list of maping arrays used to translate from the structured index
            # to the flatten index number of X
            # For example: X[iSurf][X_map[iSurf][i,j,2]] gives the z coordinate
            # of the node in the i-th row and j-th column on surface iSurf
            self.X_map += [np.reshape(np.array(range(surfs[iSurf].X.size)), surfs[iSurf].X.shape)]
            # A list of maping arrays used to provide a unique node number for
            # every node on each surface
            # For example: node_map[iSurf][i,j] gives the node number
            # of the node in the i-th row and j-th column on surface iSurf
            self.node_map += [
                np.reshape(
                    np.array(range(surfs[iSurf].X.size // 3)), (surfs[iSurf].X.shape[0], surfs[iSurf].X.shape[1])
                )
            ]
            # A list of the coordinates arrays for each surface, in the shape that DVGeo expects (N_nodes,3)
            self.coords += [np.reshape(self.X[iSurf], (surfs[iSurf].X.shape[0] * surfs[iSurf].X.shape[1], 3))]

        self.curvatureType = curvatureType
        self.scaled = scaled
        self.KSCoeff = KSCoeff
        if self.KSCoeff is None:
            # set KSCoeff to be the number of points in the plot 3D files
            self.KSCoeff = 0.0
            for i in range(len(self.coords)):
                self.KSCoeff += len(self.coords[i])

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided. We need to add a point set for each surface:
        for iSurf in range(self.nSurfs):
            self.DVGeo.addPointSet(self.coords[iSurf], self.name + "%d" % (iSurf), compNames=compNames)

        # compute the reference curvature for normalization
        self.curvatureRef = 0.0
        for iSurf in range(self.nSurfs):
            self.curvatureRef += self.evalCurvArea(iSurf)[0]

        if MPI.COMM_WORLD.rank == 0:
            print("Reference curvature: ", self.curvatureRef)

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates for each surface:
        funcs[self.name] = 0
        for iSurf in range(self.nSurfs):
            self.coords[iSurf] = self.DVGeo.update(self.name + "%d" % (iSurf), config=config)
            self.X[iSurf] = np.reshape(self.coords[iSurf], -1)
            if self.scaled:
                funcs[self.name] += self.evalCurvArea(iSurf)[0] / self.curvatureRef
            else:
                funcs[self.name] += self.evalCurvArea(iSurf)[0]

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
            # Add the sensitivity of the curvature integral over all surfaces
            for iSurf in range(self.nSurfs):
                DkSDX = self.evalCurvAreaSens(iSurf)
                if self.scaled:
                    DkSDX /= self.curvatureRef
                # Reshape the Xpt sensitivity to the shape DVGeo is expecting
                DkSDpt = np.reshape(DkSDX, self.coords[iSurf].shape)
                if iSurf == 0:
                    funcsSens[self.name] = self.DVGeo.totalSensitivity(
                        DkSDpt, self.name + "%d" % (iSurf), config=config
                    )
                else:
                    tmp = self.DVGeo.totalSensitivity(DkSDpt, self.name + "%d" % (iSurf), config=config)
                    for key in funcsSens[self.name]:
                        funcsSens[self.name][key] += tmp[key]

    def evalCurvArea(self, iSurf):
        """
        Evaluate the integral K**2 over the surface area of the wing.
        Where K is the Gaussian curvature.
        """
        # Evaluate the derivative of the position vector of every point on the
        # surface wrt to the parameteric corrdinate u and v
        t_u = self.evalDiff(iSurf, self.X[iSurf], "u")
        t_v = self.evalDiff(iSurf, self.X[iSurf], "v")
        # Compute the normal vector by taking the cross product of t_u and t_v
        n = self.evalCross(iSurf, t_u, t_v)
        # Compute the norm of tu_ x tv
        n_norm = self.evalNorm(iSurf, n)
        # Normalize the normal vector
        n_hat = np.zeros_like(n)
        n_hat[self.X_map[iSurf][:, :, 0]] = n[self.X_map[iSurf][:, :, 0]] / n_norm[self.node_map[iSurf][:, :]]
        n_hat[self.X_map[iSurf][:, :, 1]] = n[self.X_map[iSurf][:, :, 1]] / n_norm[self.node_map[iSurf][:, :]]
        n_hat[self.X_map[iSurf][:, :, 2]] = n[self.X_map[iSurf][:, :, 2]] / n_norm[self.node_map[iSurf][:, :]]
        # Evaluate the second derivatives of the position vector wrt u and v
        t_uu = self.evalDiff(iSurf, t_u, "u")
        t_vv = self.evalDiff(iSurf, t_v, "v")
        t_uv = self.evalDiff(iSurf, t_v, "u")
        # Compute the components of the first fundamental form of a parameteric
        # surface
        E = self.evalInProd(iSurf, t_u, t_u)
        F = self.evalInProd(iSurf, t_v, t_u)
        G = self.evalInProd(iSurf, t_v, t_v)
        # Compute the components of the second fundamental form of a parameteric
        # surface
        L = self.evalInProd(iSurf, t_uu, n_hat)
        M = self.evalInProd(iSurf, t_uv, n_hat)
        N = self.evalInProd(iSurf, t_vv, n_hat)
        # Compute Gaussian and mean curvature (K and H)
        K = (L * N - M * M) / (E * G - F * F)
        H = (E * N - 2 * F * M + G * L) / (2 * (E * G - F * F))
        # Compute the combined curvature (C)
        C = 4.0 * H * H - 2.0 * K
        # Assign integration weights for each point
        # 1   for center nodes
        # 1/2 for edge nodes
        # 1/4 for corner nodes
        wt = np.zeros_like(n_norm) + 1
        wt[self.node_map[iSurf][0, :]] *= 0.5
        wt[self.node_map[iSurf][-1, :]] *= 0.5
        wt[self.node_map[iSurf][:, 0]] *= 0.5
        wt[self.node_map[iSurf][:, -1]] *= 0.5
        # Compute discrete area associated with each node
        dS = wt * n_norm
        one = np.ones(self.node_map[iSurf].size)

        if self.curvatureType == "Gaussian":
            # Now compute integral (K**2) over S, equivalent to sum(K**2*dS)
            kS = np.dot(one, K * K * dS)
            return [kS, K, H, C]
        elif self.curvatureType == "mean":
            # Now compute integral (H**2) over S, equivalent to sum(H**2*dS)
            hS = np.dot(one, H * H * dS)
            return [hS, K, H, C]
        elif self.curvatureType == "combined":
            # Now compute integral C over S, equivalent to sum(C*dS)
            cS = np.dot(one, C * dS)
            return [cS, K, H, C]
        elif self.curvatureType == "KSmean":
            # Now compute the KS function for mean curvature, equivalent to KS(H*H*dS)
            sigmaH = np.dot(one, np.exp(self.KSCoeff * H * H * dS))
            KSmean = np.log(sigmaH) / self.KSCoeff
            if MPI.COMM_WORLD.rank == 0:
                print("Max curvature: ", max(H * H * dS))
            return [KSmean, K, H, C]
        else:
            raise Error(
                "The curvatureType parameter should be Gaussian, mean, or combined, "
                "%s is not supported!" % self.curvatureType
            )

    def evalCurvAreaSens(self, iSurf):
        """
        Compute sensitivity of the integral K**2 wrt the coordinate
        locations X
        """
        # Evaluate the derivative of the position vector of every point on the
        # surface wrt to the parameteric corrdinate u and v
        t_u = self.evalDiff(iSurf, self.X[iSurf], "u")
        Dt_uDX = self.evalDiffSens(iSurf, "u")
        t_v = self.evalDiff(iSurf, self.X[iSurf], "v")
        Dt_vDX = self.evalDiffSens(iSurf, "v")
        # Compute the normal vector by taking the cross product of t_u and t_v
        n = self.evalCross(iSurf, t_u, t_v)
        [DnDt_u, DnDt_v] = self.evalCrossSens(iSurf, t_u, t_v)
        DnDX = DnDt_u.dot(Dt_uDX) + DnDt_v.dot(Dt_vDX)
        # Compute the norm of tu_ x tv
        n_norm = self.evalNorm(iSurf, n)
        Dn_normDn = self.evalNormSens(iSurf, n)
        Dn_normDX = Dn_normDn.dot(DnDX)
        # Normalize the normal vector
        n_hat = np.zeros_like(n)
        n_hat[self.X_map[iSurf][:, :, 0]] = n[self.X_map[iSurf][:, :, 0]] / n_norm[self.node_map[iSurf][:, :]]
        n_hat[self.X_map[iSurf][:, :, 1]] = n[self.X_map[iSurf][:, :, 1]] / n_norm[self.node_map[iSurf][:, :]]
        n_hat[self.X_map[iSurf][:, :, 2]] = n[self.X_map[iSurf][:, :, 2]] / n_norm[self.node_map[iSurf][:, :]]

        ii = []
        data = []
        for i in range(3):
            # Dn_hat[self.X_map[iSurf][:,:,i]]/Dn[self.X_map[iSurf][:,:,i]]
            ii += list(np.reshape(self.X_map[iSurf][:, :, i], -1))
            data += list(np.reshape(n_norm[self.node_map[iSurf][:, :]] ** -1, -1))
        Dn_hatDn = csr_matrix((data, [ii, ii]), shape=(self.X[iSurf].size, self.X[iSurf].size))

        ii = []
        jj = []
        data = []
        for i in range(3):
            # Dn_hat[self.X_map[iSurf][:,:,i]]/Dn_norm[self.node_map[iSurf][:,:]]
            ii += list(np.reshape(self.X_map[iSurf][:, :, i], -1))
            jj += list(np.reshape(self.node_map[iSurf][:, :], -1))
            data += list(np.reshape(-n[self.X_map[iSurf][:, :, i]] / (n_norm[self.node_map[iSurf][:, :]] ** 2), -1))
        Dn_hatDn_norm = csr_matrix((data, [ii, jj]), shape=(n_hat.size, n_norm.size))

        Dn_hatDX = Dn_hatDn.dot(DnDX) + Dn_hatDn_norm.dot(Dn_normDX)
        # Evaluate the second derivatives of the position vector wrt u and v
        t_uu = self.evalDiff(iSurf, t_u, "u")
        Dt_uuDt_u = self.evalDiffSens(iSurf, "u")
        Dt_uuDX = Dt_uuDt_u.dot(Dt_uDX)

        t_vv = self.evalDiff(iSurf, t_v, "v")
        Dt_vvDt_v = self.evalDiffSens(iSurf, "v")
        Dt_vvDX = Dt_vvDt_v.dot(Dt_vDX)

        t_uv = self.evalDiff(iSurf, t_v, "u")
        Dt_uvDt_v = self.evalDiffSens(iSurf, "u")
        Dt_uvDX = Dt_uvDt_v.dot(Dt_vDX)
        # Compute the components of the first fundamental form of a parameteric
        # surface
        E = self.evalInProd(iSurf, t_u, t_u)
        [DEDt_u, _] = self.evalInProdSens(iSurf, t_u, t_u)
        DEDt_u *= 2
        DEDX = DEDt_u.dot(Dt_uDX)

        F = self.evalInProd(iSurf, t_v, t_u)
        [DFDt_v, DFDt_u] = self.evalInProdSens(iSurf, t_v, t_u)
        DFDX = DFDt_v.dot(Dt_vDX) + DFDt_u.dot(Dt_uDX)

        G = self.evalInProd(iSurf, t_v, t_v)
        [DGDt_v, _] = self.evalInProdSens(iSurf, t_v, t_v)
        DGDt_v *= 2
        DGDX = DGDt_v.dot(Dt_vDX)

        # Compute the components of the second fundamental form of a parameteric
        # surface
        L = self.evalInProd(iSurf, t_uu, n_hat)
        [DLDt_uu, DLDn_hat] = self.evalInProdSens(iSurf, t_uu, n_hat)
        DLDX = DLDt_uu.dot(Dt_uuDX) + DLDn_hat.dot(Dn_hatDX)

        M = self.evalInProd(iSurf, t_uv, n_hat)
        [DMDt_uv, DMDn_hat] = self.evalInProdSens(iSurf, t_uv, n_hat)
        DMDX = DMDt_uv.dot(Dt_uvDX) + DMDn_hat.dot(Dn_hatDX)

        N = self.evalInProd(iSurf, t_vv, n_hat)
        [DNDt_vv, DNDn_hat] = self.evalInProdSens(iSurf, t_vv, n_hat)
        DNDX = DNDt_vv.dot(Dt_vvDX) + DNDn_hat.dot(Dn_hatDX)

        # Compute Gaussian and mean curvature (K and H)
        K = (L * N - M * M) / (E * G - F * F)
        DKDE = self.diags(-(L * N - M * M) / (E * G - F * F) ** 2 * G)
        DKDF = self.diags((L * N - M * M) / (E * G - F * F) ** 2 * 2 * F)
        DKDG = self.diags(-(L * N - M * M) / (E * G - F * F) ** 2 * E)
        DKDL = self.diags(N / (E * G - F * F))
        DKDM = self.diags(2 * M / (E * G - F * F))
        DKDN = self.diags(L / (E * G - F * F))
        DKDX = DKDE.dot(DEDX) + DKDF.dot(DFDX) + DKDG.dot(DGDX) + DKDL.dot(DLDX) + DKDM.dot(DMDX) + DKDN.dot(DNDX)

        H = (E * N - 2 * F * M + G * L) / (2 * (E * G - F * F))
        DHDE = self.diags(N / (2 * (E * G - F * F)) - (E * N - 2 * F * M + G * L) / (2 * (E * G - F * F)) ** 2 * 2 * G)
        DHDF = self.diags(
            -2 * M / (2 * (E * G - F * F)) + (E * N - 2 * F * M + G * L) / (2 * (E * G - F * F)) ** 2 * 4 * F
        )
        DHDG = self.diags(L / (2 * (E * G - F * F)) - (E * N - 2 * F * M + G * L) / (2 * (E * G - F * F)) ** 2 * 2 * E)
        DHDL = self.diags(G / (2 * (E * G - F * F)))
        DHDM = self.diags(-2 * F / (2 * (E * G - F * F)))
        DHDN = self.diags(E / (2 * (E * G - F * F)))
        DHDX = DHDE.dot(DEDX) + DHDF.dot(DFDX) + DHDG.dot(DGDX) + DHDL.dot(DLDX) + DHDM.dot(DMDX) + DHDN.dot(DNDX)

        # Assign integration weights for each point
        # 1   for center nodes
        # 1/2 for edge nodes
        # 1/4 for corner nodes
        wt = np.zeros_like(n_norm) + 1
        wt[self.node_map[iSurf][0, :]] *= 0.5
        wt[self.node_map[iSurf][-1, :]] *= 0.5
        wt[self.node_map[iSurf][:, 0]] *= 0.5
        wt[self.node_map[iSurf][:, -1]] *= 0.5
        # Compute discrete area associated with each node
        dS = wt * n_norm
        DdSDX = self.diags(wt).dot(Dn_normDX)

        one = np.ones(self.node_map[iSurf].size)

        if self.curvatureType == "Gaussian":
            # Now compute integral (K**2) over S, equivalent to sum(K**2*dS)
            # kS = np.dot(one, K * K * dS)
            DkSDX = (self.diags(2 * K * dS).dot(DKDX) + self.diags(K * K).dot(DdSDX)).T.dot(one)
            return DkSDX
        elif self.curvatureType == "mean":
            # Now compute integral (H**2) over S, equivalent to sum(H**2*dS)
            # hS = np.dot(one, H * H * dS)
            DhSDX = (self.diags(2 * H * dS).dot(DHDX) + self.diags(H * H).dot(DdSDX)).T.dot(one)
            return DhSDX
        elif self.curvatureType == "combined":
            # Now compute dcSDX. Note: cS= sum( (4*H*H-2*K)*dS ), DcSDX = term1 - term2
            # where term1 = sum( 8*H*DHDX*dS + 4*H*H*DdSdX ), term2 = sum( 2*DKDX*dS + 2*K*DdSdX )
            term1 = (self.diags(8 * H * dS).dot(DHDX) + self.diags(4 * H * H).dot(DdSDX)).T.dot(one)
            term2 = (self.diags(2 * dS).dot(DKDX) + self.diags(2 * K).dot(DdSDX)).T.dot(one)
            DcSDX = term1 - term2
            return DcSDX
        elif self.curvatureType == "KSmean":
            sigmaH = np.dot(one, np.exp(self.KSCoeff * H * H * dS))
            DhSDX = (
                self.diags(2 * H * dS / sigmaH * np.exp(self.KSCoeff * H * H * dS)).dot(DHDX)
                + self.diags(H * H / sigmaH * np.exp(self.KSCoeff * H * H * dS)).dot(DdSDX)
            ).T.dot(one)
            return DhSDX
        else:
            raise Error(
                "The curvatureType parameter should be Gaussian, mean, or combined, "
                "%s is not supported!" % self.curvatureType
            )

    def evalCross(self, iSurf, u, v):
        """
        Evaluate the cross product of two vector fields on the surface
        (n = u x v)
        """
        n = np.zeros_like(self.X[iSurf])
        n[self.X_map[iSurf][:, :, 0]] = (
            u[self.X_map[iSurf][:, :, 1]] * v[self.X_map[iSurf][:, :, 2]]
            - u[self.X_map[iSurf][:, :, 2]] * v[self.X_map[iSurf][:, :, 1]]
        )
        n[self.X_map[iSurf][:, :, 1]] = (
            -u[self.X_map[iSurf][:, :, 0]] * v[self.X_map[iSurf][:, :, 2]]
            + u[self.X_map[iSurf][:, :, 2]] * v[self.X_map[iSurf][:, :, 0]]
        )
        n[self.X_map[iSurf][:, :, 2]] = (
            u[self.X_map[iSurf][:, :, 0]] * v[self.X_map[iSurf][:, :, 1]]
            - u[self.X_map[iSurf][:, :, 1]] * v[self.X_map[iSurf][:, :, 0]]
        )
        return n

    def evalCrossSens(self, iSurf, u, v):
        """
        Evaluate sensitivity of cross product wrt to the input vectors u and v
        (DnDu, DnDv)
        """
        # Compute sensitivity wrt v
        ii = []
        jj = []
        data = []
        # Dn[self.X_map[iSurf][:,:,0]]/Dv[self.X_map[iSurf][:,:,2]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 0], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 2], -1))
        data += list(np.reshape(u[self.X_map[iSurf][:, :, 1]], -1))
        # Dn[self.X_map[iSurf][:,:,0]]/Dv[self.X_map[iSurf][:,:,1]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 0], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 1], -1))
        data += list(np.reshape(-u[self.X_map[iSurf][:, :, 2]], -1))
        # Dn[self.X_map[iSurf][:,:,1]]/Dv[self.X_map[iSurf][:,:,2]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 1], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 2], -1))
        data += list(np.reshape(-u[self.X_map[iSurf][:, :, 0]], -1))
        # Dn[self.X_map[iSurf][:,:,1]]/Dv[self.X_map[iSurf][:,:,0]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 1], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 0], -1))
        data += list(np.reshape(u[self.X_map[iSurf][:, :, 2]], -1))
        # Dn[self.X_map[iSurf][:,:,2]]/Dv[self.X_map[iSurf][:,:,1]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 2], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 1], -1))
        data += list(np.reshape(u[self.X_map[iSurf][:, :, 0]], -1))
        # Dn[self.X_map[iSurf][:,:,2]]/Dv[self.X_map[iSurf][:,:,0]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 2], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 0], -1))
        data += list(np.reshape(-u[self.X_map[iSurf][:, :, 1]], -1))

        DnDv = csr_matrix((data, [ii, jj]), shape=(self.X[iSurf].size, self.X[iSurf].size))
        # Now wrt v
        ii = []
        jj = []
        data = []
        # Dn[self.X_map[iSurf][:,:,0]]/Du[self.X_map[iSurf][:,:,1]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 0], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 1], -1))
        data += list(np.reshape(v[self.X_map[iSurf][:, :, 2]], -1))
        # Dn[self.X_map[iSurf][:,:,0]]/Du[self.X_map[iSurf][:,:,2]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 0], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 2], -1))
        data += list(np.reshape(-v[self.X_map[iSurf][:, :, 1]], -1))
        # Dn[self.X_map[iSurf][:,:,1]]/Du[self.X_map[iSurf][:,:,0]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 1], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 0], -1))
        data += list(np.reshape(-v[self.X_map[iSurf][:, :, 2]], -1))
        # Dn[self.X_map[iSurf][:,:,1]]/Du[self.X_map[iSurf][:,:,2]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 1], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 2], -1))
        data += list(np.reshape(v[self.X_map[iSurf][:, :, 0]], -1))
        # Dn[self.X_map[iSurf][:,:,2]]/Du[self.X_map[iSurf][:,:,0]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 2], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 0], -1))
        data += list(np.reshape(v[self.X_map[iSurf][:, :, 1]], -1))
        # Dn[self.X_map[iSurf][:,:,2]]/Du[self.X_map[iSurf][:,:,1]]
        ii += list(np.reshape(self.X_map[iSurf][:, :, 2], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 1], -1))
        data += list(np.reshape(-v[self.X_map[iSurf][:, :, 0]], -1))

        DnDu = csr_matrix((data, [ii, jj]), shape=(self.X[iSurf].size, self.X[iSurf].size))
        return [DnDu, DnDv]

    def evalNorm(self, iSurf, u):
        """
        Evaluate the norm of vector field on the surface
         (u o u)**1/2
        """
        u_norm = np.zeros(self.X[iSurf].size // 3)
        u_norm[self.node_map[iSurf][:, :]] = np.sqrt(
            u[self.X_map[iSurf][:, :, 0]] ** 2 + u[self.X_map[iSurf][:, :, 1]] ** 2 + u[self.X_map[iSurf][:, :, 2]] ** 2
        )
        return u_norm

    def evalNormSens(self, iSurf, u):
        """
        Evaluate the sensitivity of the norm wrt input vector u
        """
        u_norm = np.zeros(self.X[iSurf].size // 3)
        u_norm[self.node_map[iSurf][:, :]] = np.sqrt(
            u[self.X_map[iSurf][:, :, 0]] ** 2 + u[self.X_map[iSurf][:, :, 1]] ** 2 + u[self.X_map[iSurf][:, :, 2]] ** 2
        )
        ii = []
        jj = []
        data = []
        # Du_norm[self.node_map[iSurf][:,:]]Du[self.X_map[iSurf][:,:,0]]
        ii += list(np.reshape(self.node_map[iSurf][:, :], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 0], -1))
        data += list(np.reshape(u[self.X_map[iSurf][:, :, 0]] / u_norm[self.node_map[iSurf][:, :]], -1))

        # Du_norm[self.node_map[iSurf][:,:]]Du[self.X_map[iSurf][:,:,1]]
        ii += list(np.reshape(self.node_map[iSurf][:, :], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 1], -1))
        data += list(np.reshape(u[self.X_map[iSurf][:, :, 1]] / u_norm[self.node_map[iSurf][:, :]], -1))

        # Du_norm[self.node_map[iSurf][:,:]]Du[self.X_map[iSurf][:,:,2]]
        ii += list(np.reshape(self.node_map[iSurf][:, :], -1))
        jj += list(np.reshape(self.X_map[iSurf][:, :, 2], -1))
        data += list(np.reshape(u[self.X_map[iSurf][:, :, 2]] / u_norm[self.node_map[iSurf][:, :]], -1))

        Du_normDu = csr_matrix((data, [ii, jj]), shape=(u_norm.size, self.X[iSurf].size))
        return Du_normDu

    def evalInProd(self, iSurf, u, v):
        """
        Evaluate the inner product of two vector fields on the surface
        (ip = u o v)
        """
        ip = np.zeros(self.node_map[iSurf].size)
        for i in range(3):
            ip[self.node_map[iSurf][:, :]] += u[self.X_map[iSurf][:, :, i]] * v[self.X_map[iSurf][:, :, i]]
        return ip

    def evalInProdSens(self, iSurf, u, v):
        """
        Evaluate sensitivity of inner product wrt to the input vectors u and v
        (DipDu, DipDv)
        """
        ii = []
        jj = []
        data = []
        for i in range(3):
            # Dip[node_map[:,:]]/Du[self.X_map[iSurf][:,:,i]]
            ii += list(np.reshape(self.node_map[iSurf][:, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][:, :, i], -1))
            data += list(np.reshape(v[self.X_map[iSurf][:, :, i]], -1))
        DipDu = csr_matrix((data, [ii, jj]), shape=(self.node_map[iSurf].size, self.X_map[iSurf].size))
        ii = []
        jj = []
        data = []
        for i in range(3):
            # Dip[node_map[:,:]]/Dv[self.X_map[iSurf][:,:,i]]
            ii += list(np.reshape(self.node_map[iSurf][:, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][:, :, i], -1))
            data += list(np.reshape(u[self.X_map[iSurf][:, :, i]], -1))
        DipDv = csr_matrix((data, [ii, jj]), shape=(self.node_map[iSurf].size, self.X_map[iSurf].size))
        return [DipDu, DipDv]

    def evalDiff(self, iSurf, v, wrt):
        """
        Differentiate vector field v wrt the parameteric coordinate u or v.
        Second order accurate. Central difference for nodes in the center
        forward/backward difference for nodes on the edge
        """
        v_wrt = np.zeros_like(v)
        if wrt == "u":
            v_wrt[self.X_map[iSurf][1:-1, :, :]] = (
                v[self.X_map[iSurf][2:, :, :]] - v[self.X_map[iSurf][0:-2, :, :]]
            ) / 2.0
            v_wrt[self.X_map[iSurf][0, :, :]] = (
                -1 * v[self.X_map[iSurf][2, :, :]]
                + 4 * v[self.X_map[iSurf][1, :, :]]
                - 3 * v[self.X_map[iSurf][0, :, :]]
            ) / 2.0
            v_wrt[self.X_map[iSurf][-1, :, :]] = (
                -(
                    -1 * v[self.X_map[iSurf][-3, :, :]]
                    + 4 * v[self.X_map[iSurf][-2, :, :]]
                    - 3 * v[self.X_map[iSurf][-1, :, :]]
                )
                / 2.0
            )
        elif wrt == "v":
            v_wrt[self.X_map[iSurf][:, 1:-1, :]] = (
                v[self.X_map[iSurf][:, 2:, :]] - v[self.X_map[iSurf][:, 0:-2, :]]
            ) / 2.0
            v_wrt[self.X_map[iSurf][:, 0, :]] = (
                -1 * v[self.X_map[iSurf][:, 2, :]]
                + 4 * v[self.X_map[iSurf][:, 1, :]]
                - 3 * v[self.X_map[iSurf][:, 0, :]]
            ) / 2.0
            v_wrt[self.X_map[iSurf][:, -1, :]] = (
                -(
                    -1 * v[self.X_map[iSurf][:, -3, :]]
                    + 4 * v[self.X_map[iSurf][:, -2, :]]
                    - 3 * v[self.X_map[iSurf][:, -1, :]]
                )
                / 2.0
            )
        return v_wrt

    def evalDiffSens(self, iSurf, wrt):
        """
        Compute sensitivity of v_wrt with respect to input vector field v
        (Dv_wrt/Dv)
        """
        ii = []
        jj = []
        data = []
        if wrt == "u":
            # Central Difference

            # Dt_u[X_map[1:-1,:,:]]/DX[X_map[2:,:,:]] = 1/2
            ii += list(np.reshape(self.X_map[iSurf][1:-1, :, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][2:, :, :], -1))
            data += [0.5] * len(np.reshape(self.X_map[iSurf][1:-1, :, :], -1))

            # Dt_u[X_map[1:-1,:,:]]/DX[X_map[0:-2,:,:]] = -1/2
            ii += list(np.reshape(self.X_map[iSurf][1:-1, :, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][0:-2, :, :], -1))
            data += [-0.5] * len(np.reshape(self.X_map[iSurf][1:-1, :, :], -1))

            # Forward Difference

            # Dt_u[X_map[0,:,:]]/DX[X_map[2,:,:]] = -1/2
            ii += list(np.reshape(self.X_map[iSurf][0, :, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][2, :, :], -1))
            data += [-0.5] * len(np.reshape(self.X_map[iSurf][0, :, :], -1))

            # Dt_u[X_map[0,:,:]]/DX[X_map[1,:,:]] = 4/2
            ii += list(np.reshape(self.X_map[iSurf][0, :, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][1, :, :], -1))
            data += [2] * len(np.reshape(self.X_map[iSurf][0, :, :], -1))

            # Dt_u[X_map[0,:,:]]/DX[X_map[0,:,:]] = -3/2
            ii += list(np.reshape(self.X_map[iSurf][0, :, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][0, :, :], -1))
            data += [-1.5] * len(np.reshape(self.X_map[iSurf][0, :, :], -1))

            # Backward Difference

            # Dt_u[X_map[-1,:,:]]/DX[X_map[-3,:,:]] = 1/2
            ii += list(np.reshape(self.X_map[iSurf][-1, :, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][-3, :, :], -1))
            data += [0.5] * len(np.reshape(self.X_map[iSurf][-1, :, :], -1))

            # Dt_u[X_map[-1,:,:]]/DX[X_map[-2,:,:]] = -4/2
            ii += list(np.reshape(self.X_map[iSurf][-1, :, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][-2, :, :], -1))
            data += [-2.0] * len(np.reshape(self.X_map[iSurf][-2, :, :], -1))

            # Dt_u[X_map[-1,:,:]]/DX[X_map[-1,:,:]] = 3/2
            ii += list(np.reshape(self.X_map[iSurf][-1, :, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][-1, :, :], -1))
            data += [1.5] * len(np.reshape(self.X_map[iSurf][-1, :, :], -1))

        elif wrt == "v":
            # Central Difference

            # Dt_u[X_map[:,1:-1,:]]/DX[X_map[:,2:,:]] = 1/2
            ii += list(np.reshape(self.X_map[iSurf][:, 1:-1, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][:, 2:, :], -1))
            data += [0.5] * len(np.reshape(self.X_map[iSurf][:, 1:-1, :], -1))

            # Dt_u[X_map[:,1:-1,:]]/DX[X_map[:,0:-2,:]] = -1/2
            ii += list(np.reshape(self.X_map[iSurf][:, 1:-1, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][:, 0:-2, :], -1))
            data += [-0.5] * len(np.reshape(self.X_map[iSurf][:, 1:-1, :], -1))

            # Forward Difference

            # Dt_u[X_map[:,0,:]]/DX[X_map[:,2,:]] = -1/2
            ii += list(np.reshape(self.X_map[iSurf][:, 0, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][:, 2, :], -1))
            data += [-0.5] * len(np.reshape(self.X_map[iSurf][:, 0, :], -1))

            # Dt_u[X_map[:,0,:]]/DX[X_map[:,1,:]] = 4/2
            ii += list(np.reshape(self.X_map[iSurf][:, 0, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][:, 1, :], -1))
            data += [2] * len(np.reshape(self.X_map[iSurf][:, 0, :], -1))

            # Dt_u[X_map[:,0,:]]/DX[X_map[:,0,:]] = -3/2
            ii += list(np.reshape(self.X_map[iSurf][:, 0, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][:, 0, :], -1))
            data += [-1.5] * len(np.reshape(self.X_map[iSurf][:, 0, :], -1))

            # Backward Difference

            # Dt_u[X_map[:,-1,:]]/DX[X_map[:,-3,:]] = 1/2
            ii += list(np.reshape(self.X_map[iSurf][:, -1, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][:, -3, :], -1))
            data += [0.5] * len(np.reshape(self.X_map[iSurf][:, -1, :], -1))

            # Dt_u[X_map[:,-1,:]]/DX[X_map[:,-2,:]] = -4/2
            ii += list(np.reshape(self.X_map[iSurf][:, -1, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][:, -2, :], -1))
            data += [-2.0] * len(np.reshape(self.X_map[iSurf][:, -2, :], -1))

            # Dt_u[X_map[:,-1,:]]/DX[X_map[:,-1,:]] = 3/2
            ii += list(np.reshape(self.X_map[iSurf][:, -1, :], -1))
            jj += list(np.reshape(self.X_map[iSurf][:, -1, :], -1))
            data += [1.5] * len(np.reshape(self.X_map[iSurf][:, -1, :], -1))

        Dv_uDX = csr_matrix((data, [ii, jj]), shape=(self.X[iSurf].size, self.X[iSurf].size))

        return Dv_uDX

    def diags(self, a):
        """
        A standard vectorized sparse diagonal matrix function. Similar to the above function
        some versions of scipy don't have this function, so this is here to prevent
        potential import problems.
        """
        ii = range(len(a))
        return csr_matrix((a, [ii, ii]), (len(a), len(a)))

    def writeTecplot(self, handle):
        """
        Write Curvature data on the surface to a tecplot file. Data includes
        mean curvature, H, and Gaussian curvature, K.
        """
        # we ignore the input handle and use this separated name for curvature constraint tecplot file
        # NOTE: we use this tecplot file to only visualize the local distribution of curvatures.
        # The plotted local curvatures are not exactly as that computed in the evalCurvArea function
        with open(f"{self.name}.dat", "w") as f:
            f.write('title = "DVConstraint curvature constraint"\n')
            varbs = 'variables = "x", "y", "z", "K", "H" "C"'
            f.write(varbs + "\n")
            for iSurf in range(self.nSurfs):
                [_, K, H, C] = self.evalCurvArea(iSurf)
                f.write("Zone T=%s_%d\n" % (self.name, iSurf))

                f.write(
                    "Nodes = %d, Elements = %d, f=fepoint, et=quadrilateral\n"
                    % (len(self.coords[iSurf]), (self.X_map[iSurf].shape[0] - 1) * (self.X_map[iSurf].shape[1] - 1))
                )
                for i in range(self.X_map[iSurf].shape[0]):
                    for j in range(self.X_map[iSurf].shape[1]):
                        f.write(
                            "%E %E %E %E %E %E\n"
                            % (
                                self.X[iSurf][self.X_map[iSurf][i, j, 0]],
                                self.X[iSurf][self.X_map[iSurf][i, j, 1]],
                                self.X[iSurf][self.X_map[iSurf][i, j, 2]],
                                K[self.node_map[iSurf][i, j]],
                                H[self.node_map[iSurf][i, j]],
                                C[self.node_map[iSurf][i, j]],
                            )
                        )
                f.write("\n")
                for i in range(self.X_map[iSurf].shape[0] - 1):
                    for j in range(self.X_map[iSurf].shape[1] - 1):
                        f.write(
                            "%d %d %d %d\n"
                            % (
                                self.node_map[iSurf][i, j] + 1,
                                self.node_map[iSurf][i + 1, j] + 1,
                                self.node_map[iSurf][i + 1, j + 1] + 1,
                                self.node_map[iSurf][i, j + 1] + 1,
                            )
                        )
