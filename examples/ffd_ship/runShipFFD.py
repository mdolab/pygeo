import numpy as np
from pygeo import DVGeometry, pyGeo
from pyspline.utils import openTecplot, closeTecplot, writeTecplot3D

IGES_FILE = "KCS_half_hull_SVA.igs"
FFD_FILE = "KCS_ffd.xyz"

# Bulge definition. The bow is at high x; the design waterline is near z = 10.8 m.
# We select forward control points at/below the waterline and push them outboard.
BOW_X_MIN = 215.0  # only stations forward of this (forward ~10% of Lpp)
WATERLINE_Z_MAX = 12.0  # only control points at/below the design waterline
BULGE_OUTBOARD = 0.8  # transverse (+y) offset applied to the selected points [m]

STERN_X_MAX = 240.0 * 0.2  # only stations aft of this

# Point-cloud sampling resolution per surface (parametric u x v grid). Higher
# values capture more hull curvature at the cost of larger output files.
NU_SAMPLE = 40
NV_SAMPLE = 40


def sampleHullPointCloud(geo, nu=NU_SAMPLE, nv=NV_SAMPLE):
    """Sample the baseline (undeformed) hull on an nu x nv parametric grid.

    Evaluates each surface's NURBS definition once and stacks all surface points
    into a single ``(nSurf*nu*nv, 3)`` array. This is the only place the
    expensive NURBS evaluation (``getValue``) happens; the returned cloud is
    embedded in the FFD and thereafter deformed directly by ``DVGeo.update``.
    """
    u, v = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
    u, v = u.flatten(), v.flatten()
    allPts = [
        geo.surfs[ii].getValue(u, v) for ii in range(geo.nSurf)
    ]  # each (nu*nv, 3)
    return np.vstack(allPts)


def writeDeformedHull(
    DVGeo, fileName, nSurf, nu=NU_SAMPLE, nv=NV_SAMPLE, ptName="hull", iframe=0
):
    """Deform the embedded hull cloud with the current design variables and write
    it as a structured Tecplot surface (one zone per hull surface).

    ``DVGeo.update`` returns the baseline cloud deformed by the current DVs in the
    order it was embedded: surface-major, then the v-major ``u x v`` grid from
    ``sampleHullPointCloud``. Each surface block therefore reshapes back to an
    ``(nv, nu)`` structured patch, which ``writeTecplot3D`` expects as an
    ``(nx, ny, nz, ndim)`` array (here ``(nv, nu, 1, 3)``). No NURBS re-evaluation.
    """
    coords = DVGeo.update(ptName)
    npts = nu * nv
    f = openTecplot(fileName, 3)
    for ii in range(nSurf):
        patch = coords[ii * npts : (ii + 1) * npts].reshape(nv, nu, 1, 3)
        writeTecplot3D(f, f"hull{ii}", patch)
    closeTecplot(f)

    # Also write FFD
    DVGeo.writeTecplot(f"KCS_FFD_{iframe}.dat")


def main():
    # ---------------------- Load the CAD hull (IGES) ---------------------- #
    geo = pyGeo(fileName=IGES_FILE, initType="iges")
    geo.doConnectivity()

    # ------------------------- Set up the FFD ----------------------------- #
    DVGeo = DVGeometry(FFD_FILE)
    DVGeo.addLocalDV("localShape", lower=-3.0, upper=3.0, axis="y", scale=1.0)

    # Sample the baseline hull once and embed that point cloud in the
    # (undeformed) FFD. From here on the cloud is deformed directly by the FFD
    cloud = sampleHullPointCloud(geo)
    DVGeo.addPointSet(cloud, "hull")

    # Save the undeformed surface for side-by-side comparison (DVs are still zero).
    writeDeformedHull(DVGeo, "KCS_original.dat", geo.nSurf)

    # ----------------- Select the bow and stern control points ---------------------- #
    localIndex = DVGeo.getLocalIndex(0)
    coef = DVGeo.FFD.coef

    isForward = coef[:, 0] >= BOW_X_MIN
    isAft = coef[:, 0] <= STERN_X_MAX
    isBelowWaterline = coef[:, 2] <= WATERLINE_Z_MAX
    bowSelectedFFDs = isForward & isBelowWaterline
    sternFFDs = isAft & isBelowWaterline

    # Never move the centerline points, so the symmetry plane stays at y=0.
    CLFFDs = localIndex[
        :, 0:4, :
    ].flatten()  # pinning the first 4 FFD control points because this is a cubic spline
    bowSelectedFFDs[CLFFDs] = False
    sternFFDs[CLFFDs] = False

    print(
        f"Bulging {bowSelectedFFDs.sum()} bow control points outboard by {BULGE_OUTBOARD} m"
    )

    # ----------------------- Apply the deformation ------------------------ #
    iframe = 0
    shape = DVGeo.getValues()["localShape"].copy()
    DVGeo.setDesignVars({"localShape": shape})
    nframes = 10
    wave = BULGE_OUTBOARD * np.sin(np.linspace(0, 2 * np.pi, nframes))

    # --- Bow deformation ---
    print("=" * 30)
    print("Bow deformation")
    print("=" * 30)
    for ii, bump in enumerate(wave):
        print(f"bump: {bump}")
        # breakpoint()
        shape[bowSelectedFFDs] += bump
        DVGeo.setDesignVars({"localShape": shape})

        # ----------------------- Write the outputs ---------------------------- #
        # Write the FFD-deformed hull as a structured Tecplot surface.
        writeDeformedHull(
            DVGeo, f"KCS_pointcloud_{iframe}.dat", geo.nSurf, iframe=iframe
        )

        # Reset DVs
        shape[bowSelectedFFDs] -= bump
        DVGeo.setDesignVars({"localShape": shape})
        iframe += 1

    # --- Stern def ---
    print("=" * 30)
    print("Stern deformation")
    print("=" * 30)
    for ii, bump in enumerate(wave):
        print(f"bump: {bump}")
        shape[sternFFDs] += bump
        DVGeo.setDesignVars({"localShape": shape})

        # ----------------------- Write the outputs ---------------------------- #
        # Write the FFD-deformed hull as a structured Tecplot surface.
        writeDeformedHull(
            DVGeo, f"KCS_pointcloud_{iframe}.dat", geo.nSurf, iframe=iframe
        )

        # Reset DVs
        shape[sternFFDs] -= bump
        DVGeo.setDesignVars({"localShape": shape})
        iframe += 1


if __name__ == "__main__":
    main()
