import numpy as np

nstreamwise = 10
nspanwise = 8
FFDbox = np.zeros((nstreamwise, 2, nspanwise, 3))

zspan1 = np.linspace(-0.001, 2.5, 4)
zspan2 = np.linspace(2.5, 5.4, 5)[1:]
chord1 = 1.70 * np.ones(4)
chord2 = np.linspace(1.70, 1.20, 5)[1:]
xoff1 = -0.01 * np.ones(4)
xoff2 = np.linspace(-0.01, 0.1375, 5)[1:]
toverc = np.ones(8) * 0.16

chords = np.concatenate([chord1, chord2])
xle = np.concatenate([xoff1, xoff2])
xte = xle + chords
yl = -toverc * chords / 3
yu = 2 * toverc * chords / 3
z = np.concatenate([zspan1, zspan2])

for k in range(nspanwise):
    # create the FFD box topology
    # 1st dim = plot3d i dimension
    # 2nd = plot3d j dimension
    # 3rd = plot3d k dimension
    # 4th = xyz coordinate dimension
    # the result is a three-axis tensor of points in R3
    FFDbox[:, 0, k, 0] = np.linspace(xle[k], xte[k], nstreamwise)
    FFDbox[:, 1, k, 0] = np.linspace(xle[k], xte[k], nstreamwise)
    # Y
    # lower
    FFDbox[:, 0, k, 1] = yl[k]
    # upper
    FFDbox[:, 1, k, 1] = yu[k]
    # Z
    FFDbox[:, :, k, 2] = z[k]

# write the result to disk in plot3d format
# the i dimension is on the rows
# j and k are on newlines
# k changes slower than j
# xyz changes slowest of all
f = open("ffdbox.xyz", "w")
f.write("1\n")
# header row with block topology n x 2 x 8
f.write(str(nstreamwise) + " 2 8\n")
for ell in range(3):
    for k in range(8):
        for j in range(2):
            for i in range(nstreamwise):
                f.write("%.15f " % (FFDbox[i, j, k, ell]))
            f.write("\n")
f.close()
