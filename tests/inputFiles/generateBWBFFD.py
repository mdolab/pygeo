import numpy as np

# rst Dimensions
# Bounding box for root airfoil
x_root_range = [0.00, 30.01]
y_root_range = [-3.1, 3.0]
z_root = -23.5

# Bounding box for tip airfoil
x_tip_range = [0.00, 30.01]
y_tip_range = [-3.1, 3.0]
z_tip = 23.5

# Number of FFD control points per dimension
nX = 4  # streamwise
nY = 2  # perpendicular to wing planform
nZ = 4  # spanwise
# rst Compute
# Compute grid points
span_dist = np.linspace(0, 1, nZ) ** 0.8
z_sections = span_dist * (z_tip - z_root) + z_root

x_te = span_dist * (x_tip_range[0] - x_root_range[0]) + x_root_range[0]
x_le = span_dist * (x_tip_range[1] - x_root_range[1]) + x_root_range[1]
y_coords = np.vstack(
    (
        span_dist * (y_tip_range[0] - y_root_range[0]) + y_root_range[0],
        span_dist * (y_tip_range[1] - y_root_range[1]) + y_root_range[1],
    )
)

X = np.zeros((nY * nZ, nX))
Y = np.zeros((nY * nZ, nX))
Z = np.zeros((nY * nZ, nX))
row = 0
for k in range(nZ):
    for j in range(nY):
        X[row, :] = np.linspace(x_te[k], x_le[k], nX)
        Y[row, :] = np.ones(nX) * y_coords[j, k]
        Z[row, :] = np.ones(nX) * z_sections[k]
        row += 1
# rst Write
# Write FFD to file
filename = "bwb.xyz"
f = open(filename, "w")
f.write("\t\t1\n")
f.write("\t\t%d\t\t%d\t\t%d\n" % (nX, nY, nZ))
for set in [X, Y, Z]:
    for row in set:
        vals = tuple(row)
        f.write("\t%3.8f\t%3.8f\t%3.8f\t%3.8f\n" % vals)
f.close()
