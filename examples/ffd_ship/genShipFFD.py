import argparse

import numpy as np

from pygeo import pyGeo
from pygeo.geo_utils import createFittedHullFFD, write_wing_FFD_file

IGES_FILE = "KCS_half_hull_SVA.igs"

# Switch between the body-fitted FFD (True) and the simple rectangular box (False).
BODY_FITTED = True

# Number of control points in each FFD direction per half hull.
# i -> longitudinal, j -> transverse (y), k -> vertical (z).
N_LONGITUDINAL = 22
N_TRANSVERSE = 6
N_VERTICAL = 8

# --- Body-fitted parameters ------------------------------------------------- #
# Margins that grow the FFD outward so it fully encloses the hull, given as
# [longitudinal, transverse, vertical]. Absolute margins are in metres; relative
# margins are fractions of hull length, local half-beam, and hull depth. The
# transverse inner face stays on the y=0 centreline regardless of these.
ABS_MARGINS = [1.0, 5.0, 2.0]
REL_MARGINS = [0.01, 0.1, 0.1]

# --- Rectangular-box parameters --------------------------------------------- #
# FFD box extents: hull bounding box plus margins (full-scale metres).
# The inner transverse face sits exactly on the centreline plane (y=0).
X_MIN, X_MAX = -9.0, 240.0  # longitudinal (stern -> bow)
Y_MIN, Y_MAX = -0.01, 17.0  # transverse (centreline -> outboard of max beam)
Z_MIN, Z_MAX = -0.5, 24.0  # vertical (below keel -> above deck)


def generate_body_fitted(fileName):
    """Write a body-fitted FFD that conforms to the hull surface."""
    geo = pyGeo(fileName=IGES_FILE, initType="iges")
    geo.doConnectivity()

    createFittedHullFFD(
        geo,
        "point-vector",
        fileName,
        nLongitudinal=N_LONGITUDINAL,
        nTransverse=N_TRANSVERSE,
        nVertical=N_VERTICAL,
        absMargins=ABS_MARGINS,
        relMargins=REL_MARGINS,
        xDist="cosine",
    )


def generate_box(fileName):
    """Write a simple rectangular FFD box around the hull bounding box."""

    # write_wing_FFD_file builds the box from two end "slices" (stern and bow),
    # each holding the four cross-section corners. The slice array is indexed
    # [slice, a, b, xyz] where a selects the transverse (y) corner and b the
    # vertical (z) corner. We march between the slices along the longitudinal
    # direction (dim 0).
    def cross_section(xStation):
        return [
            # a = 0 -> y inner (centreline)
            [[xStation, Y_MIN, Z_MIN], [xStation, Y_MIN, Z_MAX]],
            # a = 1 -> y outer
            [[xStation, Y_MAX, Z_MIN], [xStation, Y_MAX, Z_MAX]],
        ]

    slices = np.array([cross_section(X_MIN), cross_section(X_MAX)])

    # axes = ["i", "j", "k"] maps the FFD i-index to the longitudinal slice
    # direction, j to the transverse (y) corners, and k to the vertical (z)
    # corners. So getLocalIndex(0) has shape (N_LONGITUDINAL, N_TRANSVERSE,
    # N_VERTICAL) and j=0 is the centreline plane.
    axes = ["i", "j", "k"]

    # Cluster control sections toward bow and stern along the longitudinal axis.
    dist = [["cosine", "linear", "linear"]]

    write_wing_FFD_file(
        fileName,
        slices,
        N0=N_LONGITUDINAL,
        N1=N_TRANSVERSE,
        N2=N_VERTICAL,
        axes=axes,
        dist=dist,
    )


def read_ffd(fileName):
    """Read a single-block ASCII PLOT3D FFD file into an (Ni, Nj, Nk, 3) array."""
    with open(fileName) as f:
        nBlocks = int(f.readline())
        if nBlocks != 1:
            raise ValueError(f"{fileName} has {nBlocks} blocks; expected a single-block FFD")
        Ni, Nj, Nk = (int(n) for n in f.readline().split())
        data = np.array(f.read().split(), dtype=float)
    # The writer loops ell -> k -> j -> i, so the flat data reshapes to
    # (3, Nk, Nj, Ni) and transposes back to (Ni, Nj, Nk, 3).
    return data.reshape(3, Nk, Nj, Ni).transpose(3, 2, 1, 0)


def write_ffd(fileName, coords):
    """Write an (Ni, Nj, Nk, 3) lattice as a single-block ASCII PLOT3D FFD file."""
    Ni, Nj, Nk, _ = coords.shape
    with open(fileName, "w") as f:
        f.write("1\n")
        f.write(f"{Ni} {Nj} {Nk}\n")
        for ell in range(3):
            for kk in range(Nk):
                for jj in range(Nj):
                    for ii in range(Ni):
                        f.write("%.15f " % (coords[ii, jj, kk, ell]))
                    f.write("\n")


def mirror_ffd(halfFileName, fullFileName):
    """Mirror the half-hull FFD in ``halfFileName`` about y=0 and write the
    full-beam FFD to ``fullFileName``.

    The half FFD's inboard planes sit at negative y (the j=0 plane is pinned at
    y=-ABS_MARGINS[1], and interior planes also go negative at slender bow/stern
    stations), so simply concatenating a reflected copy would tangle the lattice.
    Instead the transverse control points at every station and level are rebuilt
    as ``linspace(-yOuter, +yOuter, 2*Nj - 1)`` from that (i, k)'s outer-face y,
    which preserves the body-fitted outer envelope on both sides and puts the
    middle j-plane exactly on the centerline.
    """
    half = read_ffd(halfFileName)
    nTransverse = half.shape[1]
    nTransverseFull = 2 * nTransverse - 1

    # The symmetric rebuild keeps x and z from the half lattice, which is only
    # valid because they do not vary across j.
    for dim in (0, 2):
        if not np.allclose(half[:, :, :, dim], half[:, :1, :, dim]):
            raise ValueError(f"{halfFileName}: x/z vary across the transverse index; cannot mirror")

    full = np.repeat(half[:, :1, :, :], nTransverseFull, axis=1)
    yOuter = half[:, -1, :, 1]  # (Ni, Nk) outer-face half-beam plus margins
    tt = np.linspace(-1.0, 1.0, nTransverseFull)
    full[:, :, :, 1] = yOuter[:, None, :] * tt[None, :, None]

    write_ffd(fullFileName, full)


def generate(fileName="KCS_ffd.xyz", mirrorHull=False, fullFileName="KCS_full_ffd.xyz"):
    """Write the KCS hull FFD box to ``fileName`` in PLOT3D format.

    With ``mirrorHull=True``, also mirror it about y=0 into a full-beam FFD
    written to ``fullFileName``.
    """
    if BODY_FITTED:
        generate_body_fitted(fileName)
        kind = "body-fitted"
    else:
        generate_box(fileName)
        kind = "rectangular-box"
    print(f"Wrote {fileName}: {N_LONGITUDINAL} x {N_TRANSVERSE} x {N_VERTICAL} {kind} FFD control points")

    if mirrorHull:
        mirror_ffd(fileName, fullFileName)
        print(
            f"Wrote {fullFileName}: {N_LONGITUDINAL} x {2 * N_TRANSVERSE - 1} x {N_VERTICAL} "
            f"mirrored full-beam {kind} FFD control points"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mirror_hull",
        help="Mirror the hull geometry",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    generate(mirrorHull=args.mirror_hull)
