import numpy as np
from pygeo import pyGeo

# ==============================================================================
# Start of Script
# ==============================================================================
naf = 3
airfoil_list = ["naca2412.dat", "naca2412.dat", "naca2412.dat"]
chord = [1.67, 1.67, 1.18]
x = [0, 0, 0.125 * 1.18]
y = [0, 0, 0]
z = [0, 2.5, 10.58 / 2]
rot_x = [0, 0, 0]
rot_y = [0, 0, 0]
rot_z = [0, 0, 2]
offset = np.zeros((naf, 2))

# There are several examples that follow showing many of the different
# combinations of tip/trailing edge options that are available.

# --------- Sharp Trailing Edge / Rounded Tip -------
wing = pyGeo(
    "liftingSurface",
    xsections=airfoil_list,
    scale=chord,
    offset=offset,
    x=x,
    y=y,
    z=z,
    rotX=rot_x,
    rotY=rot_y,
    rotZ=rot_z,
    kSpan=2,
    tip="rounded",
)

wing.writeTecplot("c172_sharp_te_rounded_tip.dat")
wing.writeIGES("c172_sharp_te_rounded_tip.igs")

# --------- Sharp Trailing Edge / Pinched Tip -------
wing = pyGeo(
    "liftingSurface",
    xsections=airfoil_list,
    scale=chord,
    offset=offset,
    x=x,
    y=y,
    z=z,
    rotX=rot_x,
    rotY=rot_y,
    rotZ=rot_z,
    kSpan=2,
    tip="pinched",
)

wing.writeTecplot("c172_sharp_te_pinched_tip.dat")
wing.writeIGES("c172_sharp_te_pinched_tip.igs")

# --------- Sharp Trailing Edge / Rounded Tip with Fitting -------
# This option shouldn't be used except to match previously generated
# geometries
wing = pyGeo(
    "liftingSurface",
    xsections=airfoil_list,
    nCtl=29,
    scale=chord,
    offset=offset,
    x=x,
    y=y,
    z=z,
    rotX=rot_x,
    rotY=rot_y,
    rotZ=rot_z,
    kSpan=2,
    tip="rounded",
)

wing.writeTecplot("c172_sharp_te_rounded_tip_fitted.dat")
wing.writeIGES("c172_sharp_te_rounded_tip_fitted.igs")

# --------- Blunt Trailing (Flat) / Rounded Tip -------

# This is the normal way of producing blunt TE geometries. The
# thickness of the trailing edge is specified with 'te_height', either
# a constant value or an array of length naf. This is in physical
# units. Alternatively, 'te_height_scaled' can be specified to have a
# scaled thickness. This option is specified as a fraction of initial
# chord, so te_height_scaled=0.002 will give a 0.2% trailing edge
# thickness
wing = pyGeo(
    "liftingSurface",
    xsections=airfoil_list,
    scale=chord,
    offset=offset,
    x=x,
    y=y,
    z=z,
    rotX=rot_x,
    rotY=rot_y,
    rotZ=rot_z,
    bluntTe=True,
    teHeightScaled=0.002,
    kSpan=2,
    tip="rounded",
)

wing.writeTecplot("c172_blunt_te_rounded_tip.dat")
wing.writeIGES("c172_blunt_te_rounded_tip.igs")

# --------- Blunt Trailing (Rounded) / Rounded Tip -------

# Alternative way of producing rounded trailing edges that can be easier
# to mesh and extrude with pyHyp.
wing = pyGeo(
    "liftingSurface",
    xsections=airfoil_list,
    scale=chord,
    offset=offset,
    x=x,
    y=y,
    z=z,
    rotX=rot_x,
    rotY=rot_y,
    rotZ=rot_z,
    bluntTe=True,
    roundedTe=True,
    teHeightScaled=0.002,
    kSpan=2,
    tip="rounded",
)

wing.writeTecplot("c172_rounded_te_rounded_tip.dat")
wing.writeIGES("c172_rounded_te_rounded_tip.igs")
