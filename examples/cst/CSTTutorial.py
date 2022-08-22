from pygeo import DVGeometryCST
import numpy as np
import matplotlib.pyplot as plt
import os

# rst Plot func (beg)
def plot_points(points, filename=None):
    fig, ax = plt.subplots(figsize=[10, 3])
    ax.plot(points[:, 0], points[:, 1], "-o")
    ax.set_aspect("equal")
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])

    if filename:
        fig.savefig(filename)

    return fig, ax
    # rst Plot func (end)


# rst Init (beg)
# Initialize the DVGeometryCST object
curDir = os.path.dirname(__file__)  # directory of this script
airfoilFile = os.path.join(curDir, "naca2412.dat")
nCoeff = 4  # number of CST coefficients on each surface

DVGeo = DVGeometryCST(airfoilFile, numCST=nCoeff)
# rst Init (end)

# rst DV (beg)
# Add design variables that we can perturb
DVGeo.addDV("upper_shape", dvType="upper", lowerBound=-0.1, upperBound=0.5)
DVGeo.addDV("lower_shape", dvType="lower", lowerBound=-0.5, upperBound=0.1)
DVGeo.addDV("class_shape_n1", dvType="N1")
DVGeo.addDV("class_shape_n2", dvType="N2")
DVGeo.addDV("chord", dvType="chord")
# rst DV (end)

# rst Create pointset (beg)
# For this case, we'll just use the points in the airfoil file as the pointset
points = []
with open(airfoilFile, "r") as f:
    for line in f:
        points.append([float(n) for n in line.split()])

points = np.array(points)
points = np.hstack((points, np.zeros((points.shape[0], 1))))  # add 0s for z coordinates (unused)
# rst Create pointset (end)

# rst Add pointset (beg)
ptName = "pointset"
DVGeo.addPointSet(points, ptName=ptName)
# rst Add pointset (end)

# Show current geometry
points = DVGeo.update(ptName)
fig, ax = plot_points(points, filename=os.path.join(curDir, "original_points.svg"))
plt.show()
plt.close(fig)

# rst Perturb one (beg)
DVGeo.setDesignVars(
    {
        "upper_shape": np.array([0.3, 0.7, -0.1, 0.6]),
        "lower_shape": np.array([-0.1, 0.1, 0.1, -0.3]),
    }
)
points = DVGeo.update(ptName)
# rst Perturb one (end)

# Show current geometry
fig, ax = plot_points(points, filename=os.path.join(curDir, "perturbed_coeff.svg"))
plt.show()
plt.close(fig)

# rst Perturb two (beg)
DVGeo.setDesignVars(
    {
        "class_shape_n1": np.array([0.6]),
        "class_shape_n2": np.array([0.8]),
        "chord": np.array([2.0]),
    }
)
points = DVGeo.update(ptName)
# rst Perturb two (end)

# Show current geometry
fig, ax = plot_points(points, filename=os.path.join(curDir, "perturbed_class_chord.svg"))
plt.show()
plt.close(fig)
