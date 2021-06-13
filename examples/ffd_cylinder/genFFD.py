import numpy as np

nffd = 10
FFDbox = np.zeros((nffd, 2, 2, 3))
xslice = np.zeros(nffd)
yupper = np.zeros(nffd)
ylower = np.zeros(nffd)

xmargin = 0.001
ymargin = 0.02
yu = 0.5
yl = -0.5

# construct the i-j (x-y) plane grid of control points 10 x 2
# we'll copy this along the k (z) axis later to make a cube
for i in range(nffd):
    xtemp = i * 1.0 / (nffd - 1.0)
    xslice[i] = -1.0 * xmargin + (1 + 2.0 * xmargin) * xtemp
    yupper[i] = yu + ymargin
    ylower[i] = yl - ymargin

# create the FFD box topology
# 1st dim = plot3d i dimension
# 2nd = plot3d j dimension
# 3rd = plot3d k dimension
# 4th = xyz coordinate dimension
# the result is a three-axis tensor of points in R3
FFDbox[:, 0, 0, 0] = xslice[:].copy()
FFDbox[:, 1, 0, 0] = xslice[:].copy()
# Y
# lower
FFDbox[:, 0, 0, 1] = ylower[:].copy()
# upper
FFDbox[:, 1, 0, 1] = yupper[:].copy()
# copy
FFDbox[:, :, 1, :] = FFDbox[:, :, 0, :].copy()
# Z
FFDbox[:, :, 0, 2] = 0.0
# Z
FFDbox[:, :, 1, 2] = 1.0

# write the result to disk in plot3d format
# the i dimension is on the rows
# j and k are on newlines
# k changes slower than j
# xyz changes slowest of all
f = open("ffdbox.xyz", "w")
f.write("1\n")
# header row with block topology n x 2 x 2
f.write(str(nffd) + " 2 2\n")
for ell in range(3):
    for k in range(2):
        for j in range(2):
            for i in range(nffd):
                f.write("%.15f " % (FFDbox[i, j, k, ell]))
            f.write("\n")
f.close()
