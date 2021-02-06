import numpy as np
import argparse
from mpi4py import MPI
from pygeo import DVGeometryVSP
import vsp


def printCurrentDVs(DVGeo):
    x = DVGeo.getValues()
    print("Current DVs:")
    for key in x:
        print(key, ":", x[key])


def sample_uv(nu, nv):
    # function to create sample uv from the surface and save these points.
    u = np.linspace(0, 1, nu + 1)
    v = np.linspace(0, 1, nv + 1)
    uu, vv = np.meshgrid(u, v)
    # print (uu.flatten(), vv.flatten())
    uv = np.array((uu.flatten(), vv.flatten()))
    return uv


parser = argparse.ArgumentParser()
parser.add_argument("--dhfd", type=float, default=1e-6)
parser.add_argument("--dhvsp", type=float, default=1e-6)
parser.add_argument("--nint", type=int, default=10)  # number of intervals in each direction of the  quad mesh
args = parser.parse_args()

vspFile = "naca0012.vsp3"

vsp.ClearVSPModel()
vsp.ReadVSPFile(vspFile)
geoms = vsp.FindGeoms()

comps = []

DVGeo = DVGeometryVSP(vspFile)

comp = "WingGeom"

# loop over sections
# normally, there are 9 sections so we should loop over range(9) for the full test
# to have it run faster, we just pick 2 sections
for i in [0, 5]:  # range(9):

    # Twist
    DVGeo.addVariable(comp, "XSec_%d" % i, "Twist", lower=-10.0, upper=10.0, scale=1e-2, scaledStep=False, dh=args.dhvsp)

    # loop over coefs
    # normally, there are 7 coeffs so we should loop over range(7) for the full test
    # to have it run faster, we just pick 2 sections
    for j in [2, 3]:  # range(7):
        # CST Airfoil shape variables
        group = "UpperCoeff_%d" % i
        var = "Au_%d" % j
        DVGeo.addVariable(comp, group, var, lower=-0.1, upper=0.5, scale=1e-3, scaledStep=False, dh=args.dhvsp)
        group = "LowerCoeff_%d" % i
        var = "Al_%d" % j
        DVGeo.addVariable(comp, group, var, lower=-0.5, upper=0.1, scale=1e-3, scaledStep=False, dh=args.dhvsp)

# now lets generate ourselves a quad mesh of these cubes.
uv_g = sample_uv(args.nint, args.nint)

# total number of points
ntot = uv_g.shape[1]

# rank on this proc
rank = MPI.COMM_WORLD.rank

# first, equally divide
nuv = ntot // MPI.COMM_WORLD.size
# then, add the remainder
if rank < ntot % MPI.COMM_WORLD.size:
    nuv += 1

# allocate the uv array on this proc
uv = np.zeros((2, nuv))

# print how mant points we have
print("[%d] npts on this proc: %d" % (rank, uv.shape[1]), flush=True)
MPI.COMM_WORLD.Barrier()

# loop over the points and save all that this proc owns
ii = 0
for i in range(ntot):
    if i % MPI.COMM_WORLD.size == rank:
        uv[:, ii] = uv_g[:, i]
        ii += 1

# get the coordinates
nNodes = len(uv[0, :])
ptVecA = vsp.CompVecPnt01(geoms[0], 0, uv[0, :], uv[1, :])

# extract node coordinates and save them in a numpy array
coor = np.zeros((nNodes, 3))
for i in range(nNodes):
    pnt = vsp.CompPnt01(geoms[0], 0, uv[0, i], uv[1, i])
    coor[i, :] = (pnt.x(), pnt.y(), pnt.z())

# Add this pointSet to DVGeo
DVGeo.addPointSet(coor, "test_points")

# set some design variables
if rank == 0:
    printCurrentDVs(DVGeo)

# Now we will test gradients...

# We will have nNodes*3 many functions of interest...
dIdpt = np.zeros((nNodes * 3, nNodes, 3))

# set the seeds to one in the following fashion:
# first function of interest gets the first coordinate of the first point
# second func gets the second coord of first point etc....
for i in range(nNodes):
    for j in range(3):
        dIdpt[i * 3 + j, i, j] = 1

# first get the dvgeo result
funcSens = DVGeo.totalSensitivity(dIdpt.copy(), "test_points")

# now perturb the design with finite differences and compute FD gradients
DVs = DVGeo.getValues()

funcSensFD = {}

for x in DVs:

    if MPI.COMM_WORLD.rank == 0:
        print("perturbing DV", x)

    # perturb the design
    xRef = DVs[x].copy()
    DVs[x] += args.dhfd
    DVGeo.setDesignVars(DVs)

    # get the new points
    coorNew = DVGeo.update("test_points")

    # calculate finite differences
    funcSensFD[x] = (coorNew.flatten() - coor.flatten()) / args.dhfd

    # set back the DV
    DVs[x] = xRef.copy()

# now loop over the values and compare
# when this is run with multiple procs, VSP sometimes has a bug
# that leads to different procs having different spanwise
# u-v distributions. as a result, the final values can differ up to 1e-5 levels
# this issue does not come up if this tests is ran with a single proc
for x in DVs:
    err = np.array(funcSens[x].squeeze()) - np.array(funcSensFD[x])

    # get the L2 norm of error
    print("[%d] L2   error norm for DV %s: " % (rank, x), np.linalg.norm(err))

    # get the L_inf norm
    print("[%d] Linf error norm for DV %s: " % (rank, x), np.linalg.norm(err, ord=np.inf))
