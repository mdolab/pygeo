from pygeo import DVGeometry
import numpy as np
import matplotlib.pyplot as plt

# create a pointset. pointsets are of shape npts by 3 (the second dim is xyz coordinate)
# we'll generate a cylinder in parametric coordinates
# the circular portion is in the x-y plane
# the depth is along the z axis
t = np.linspace(0, 2 * np.pi, 100)
Xpt = np.zeros([200, 3])
Xpt[:100, 0] = 0.5 * np.cos(t) + 0.5
Xpt[:100, 1] = 0.5 * np.sin(t)
Xpt[:100, 2] = 0.0
Xpt[100:, 0] = 0.5 * np.cos(t) + 0.5
Xpt[100:, 1] = 0.5 * np.sin(t)
Xpt[100:, 2] = 1.0

# rst create DVGeo
# The Plot3D file ffdbox.xyz contains the coordinates of the free-form deformation (FFD)volume
# we will be using for this problem. It's a cube with sides of length 1 centered on (0, 0,0.5).
# The "i" direction of the cube consists of 10 points along the x axis
# The "j" direction of the cube is 2 points up and down (y axis direction)
# The "k" direction of the cube is 2 points into the page (z axis direction)
FFDfile = "ffdbox.xyz"

# initialize the DVGeometry object with the FFD file
DVGeo = DVGeometry(FFDfile)

# rst add pointset
# add the cylinder pointset to the FFD under the name 'cylinder'
DVGeo.addPointSet(Xpt.copy(), "cylinder")
DVGeo.writePointSet("cylinder", "pointset")

# rst add shape DV
# Now that we have pointsets added, we should parameterize the geometry.

# Adding local geometric design to make local modifications to FFD box
# This option will perturb all the control points but only the y (up-down) direction
DVGeo.addLocalDV("shape", lower=-0.5, upper=0.5, axis="y", scale=1.0)

# rst getLocalIndex
# The control points of the FFD are the same as the coordinates of the points in the input file
# but they will be in a jumbled order because of the internal spline representation of the volume.
# Let's put them in a sensible order for plotting.

# the raw array of FFD control points (size n_control_pts x 3)
FFD = DVGeo.FFD.coef

# we can use the getLocalIndex method to put the coefs in contiguous order
# the FFD block has i,j,k directions
# in this problem i is left/right, j is up down, and k is into the page (along the cyl)
# Let's extract a ring of the front-face control points in contiguous order.
# We'll add these as a pointset as well so we can visualize them.
# (Don't worry too much about the details)
FFDptset = np.concatenate(
    [
        FFD[DVGeo.getLocalIndex(0)[:, 0, 0]],
        FFD[DVGeo.getLocalIndex(0)[::-1, 1, 0]],
        FFD[DVGeo.getLocalIndex(0)[0, 0, 0]].reshape((1, 3)),
    ]
).reshape(21, 3)

# Add these control points to the FFD volume. This is only for visualization purposes in this demo.
# Under normal circumstances you don't need to worry about adding the FFD points as a pointset
DVGeo.addPointSet(FFDptset, "ffd")

# Print the indices and coordinates of the FFD points for informational purposes
print("FFD Indices:")
print(DVGeo.getLocalIndex(0)[:, 0, 0])
print("FFD Coordinates:")
print(FFD[DVGeo.getLocalIndex(0)[:, 0, 0]])

# Create tecplot output that contains the FFD control points, embedded volume, and pointset
DVGeo.writeTecplot(fileName="undeformed_embedded.dat", solutionTime=1)

# rst perturb geometry
# Now let's deform the geometry.
# We want to set the front and rear control points the same so we preserve symmetry along the z axis
# and we ues the getLocalIndex function to accomplish this
lower_front_idx = DVGeo.getLocalIndex(0)[:, 0, 0]
lower_rear_idx = DVGeo.getLocalIndex(0)[:, 0, 1]
upper_front_idx = DVGeo.getLocalIndex(0)[:, 1, 0]
upper_rear_idx = DVGeo.getLocalIndex(0)[:, 1, 1]

currentDV = DVGeo.getValues()["shape"]
newDV = currentDV.copy()

# add a constant offset (upward) to the lower points, plus a linear ramp and a trigonometric local change
# this will shrink the cylinder height-wise and make it wavy
# set the front and back points the same to keep the cylindrical sections square along that axis
for idx in [lower_front_idx, lower_rear_idx]:
    const_offset = 0.3 * np.ones(10)
    local_perturb = np.cos(np.linspace(0, 4 * np.pi, 10)) / 10 + np.linspace(-0.05, 0.05, 10)
    newDV[idx] = const_offset + local_perturb

# add a constant offset (downward) to the upper points, plus a linear ramp and a trigonometric local change
# this will shrink the cylinder height-wise and make it wavy
for idx in [upper_front_idx, upper_rear_idx]:
    const_offset = -0.3 * np.ones(10)
    local_perturb = np.sin(np.linspace(0, 4 * np.pi, 10)) / 20 + np.linspace(0.05, -0.10, 10)
    newDV[idx] = const_offset + local_perturb

# we've created an array with design variable perturbations. Now set the FFD control points with them
# and update the point sets so we can see how they changed
DVGeo.setDesignVars({"shape": newDV.copy()})

Xmod = DVGeo.update("cylinder")
FFDmod = DVGeo.update("ffd")

# Create tecplot output that contains the FFD control points, embedded volume, and pointset
DVGeo.writeTecplot(fileName="deformed_embedded.dat", solutionTime=1)

# rst plot
# cast the 3D pointsets to 2D for plotting (ignoring depth)
FFDplt = FFDptset[:, :2]
FFDmodplt = FFDmod[:, :2]
Xptplt = Xpt[:, :2]
Xmodplt = Xmod[:, :2]

# plot the new and deformed pointsets and control points
plt.figure()
plt.title("Applying FFD deformations to a cylinder")

plt.plot(Xptplt[:, 0], Xptplt[:, 1], color="#293bff")
plt.plot(FFDplt[:, 0], FFDplt[:, 1], color="#d6daff", marker="o")

plt.plot(Xmodplt[:, 0], Xmodplt[:, 1], color="#ff0000")
plt.plot(FFDmodplt[:, 0], FFDmodplt[:, 1], color="#ffabab", marker="o")

plt.xlabel("x")
plt.ylabel("y")
# plt.xlim([-0.7,1.2])
plt.axis("equal")
legend = plt.legend(
    ["original shape", "original FFD ctl pts", "deformed shape", "deformed FFD ctl pts"],
    loc="lower right",
    framealpha=0.0,
)
legend.get_frame().set_facecolor("none")
plt.tight_layout()
plt.savefig("deformed_cylinder.png")
