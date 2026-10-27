"""
==============================================================================
Recovering an airfoil's NACA parameters from its CST representation
==============================================================================
This example demonstrates the DVGeometry least-squares fitting capability. We take a NACA 2412 airfoil represented with
a CST parameterization and fit an ESP NACA model to it, recovering the airfoil's NACA 4-digit parameters (camber,
location of maximum camber, and thickness).
"""

# Standard Python modules
import os

# External modules
import matplotlib

matplotlib.use("Agg")  # use a non-interactive backend so the example runs headless (e.g. in CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# First party modules
from pygeo import DVGeometryCST, DVGeometryESP  # noqa: E402

curDir = os.path.dirname(__file__)

# rst ref (beg)
# Build the reference geometry: the CST representation of a NACA 2412 airfoil. This stands in for a geometry we want to
# match, for example the result of an FFD-based optimization.
airfoilFile = os.path.join(curDir, "naca2412.dat")
refDVGeo = DVGeometryCST(airfoilFile, numCST=8)
refDVGeo.addDV("upper", dvType="upper")
refDVGeo.addDV("lower", dvType="lower")

# Use the airfoil coordinates themselves as the point set (z is unused for the 2D CST parameterization)
points = []
with open(airfoilFile, "r") as f:
    for line in f:
        points.append([float(n) for n in line.split()])
points = np.hstack((np.array(points), np.zeros((len(points), 1))))

ptName = "airfoil"
refDVGeo.addPointSet(points, ptName)
# rst ref (end)

# rst fit setup (beg)
# Build the model we want to fit: an ESP NACA airfoil parameterized by camber, location of maximum camber, and
# thickness. These ESP design parameters are exactly the NACA 4-digit definition, so the fit can recover them.
fitDVGeo = DVGeometryESP(os.path.join(curDir, "naca0012.csm"))
fitDVGeo.addVariable("camber_dv")
fitDVGeo.addVariable("maxloc_dv")
fitDVGeo.addVariable("thickness_dv")
# rst fit setup (end)

# rst fit (beg)
# Fit the ESP model's design variables so that its geometry matches the reference CST airfoil
fitDVs, result = fitDVGeo.fitDVGeo(refDVGeo, ptName)
# rst fit (end)

# rst results (beg)
print(f"Recovered camber    = {float(fitDVs['camber_dv']):.4f}  (NACA 2412 -> 0.02)")
print(f"Recovered maxloc    = {float(fitDVs['maxloc_dv']):.4f}  (NACA 2412 -> 0.40)")
print(f"Recovered thickness = {float(fitDVs['thickness_dv']):.4f}  (NACA 2412 -> 0.12)")
# rst results (end)

# rst plot (beg)
# Compare the target CST geometry with the fitted ESP geometry. fitDVGeo embeds the reference's points into a new
# point set named "<ptName>_fit", which is the one the ESP model was fit on.
targetCoords = refDVGeo.update(ptName)
fitCoords = fitDVGeo.update(f"{ptName}_fit")

fig, ax = plt.subplots(figsize=[10, 3])
ax.plot(targetCoords[:, 0], targetCoords[:, 1], "o", markerfacecolor="none", label="Target (CST NACA 2412)")
ax.plot(fitCoords[:, 0], fitCoords[:, 1], ".", label="Fitted ESP model")
ax.set_aspect("equal")
ax.legend()
ax.set_title("Airfoil fitting: recovered NACA parameters from a CST airfoil")
fig.savefig(os.path.join(curDir, "airfoil_fitting.svg"))
# rst plot (end)
