# External modules
import numpy as np
from pyspline import Curve

# First party modules
from pygeo import pyGeo

# ==============================================================================
# Start of Script
# ==============================================================================
naf = 20
n0012 = "naca0012.dat"

airfoil_list = [n0012 for i in range(naf)]
for i in range(1, naf - 1):
    airfoil_list[i] = None

# Use the digitize it data for the planform:
le = np.array(np.loadtxt("bwb_le.out"))
te = np.array(np.loadtxt("bwb_te.out"))
front_up = np.array(np.loadtxt("bwb_front_up.out"))
front_low = np.array(np.loadtxt("bwb_front_low.out"))

le[0, :] = 0
te[0, 0] = 0
front_up[0, 0] = 0
front_low[0, 0] = 0

# Now make a ONE-DIMENSIONAL spline for each of the le and trailing edes
le_spline = Curve(X=le[:, 1], s=le[:, 0], nCtl=11, k=4)
te_spline = Curve(X=te[:, 1], s=te[:, 0], nCtl=11, k=4)
up_spline = Curve(X=front_up[:, 1], s=front_up[:, 0], nCtl=11, k=4)
low_spline = Curve(X=front_low[:, 1], s=front_low[:, 0], nCtl=11, k=4)

# Generate consistent equally spaced spline data
span = np.linspace(0, 1, naf)

le = le_spline(span)
te = te_spline(span)
up = up_spline(span)
low = low_spline(span)

ref_span = 138
chord = te - le
x = le
z = span * ref_span
mid_y = (up[0] + low[0]) / 2.0
y = -(up + low) / 2 + mid_y

# Scale the thicknesses
toc = -(up - low) / chord
thickness = toc / 0.12

rot_x = np.zeros(naf)
rot_y = np.zeros(naf)
rot_z = np.zeros(naf)
offset = np.zeros((naf, 2))

bwb = pyGeo(
    "liftingSurface",
    xsections=airfoil_list,
    scale=chord,
    offset=offset,
    thickness=thickness,
    bluntTe=True,
    teHeight=0.05,
    tip="rounded",
    x=x,
    y=y,
    z=z,
)

bwb.writeIGES("bwb.igs")
