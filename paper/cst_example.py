import os

import matplotlib.pyplot as plt
import numpy as np
from pygeo.parameterization.DVGeoCST import DVGeometryCST

cur_dir = os.path.dirname(os.path.abspath(__file__))
out_file = os.path.join(cur_dir, "cst_example.pdf")

# Airfoil shape inputs
x = np.linspace(0, 1, 10000)
N1 = 0.5
N2 = 1.0
yte = 0.0
num_cst = 3
coeff = [
    {"upper": np.full(num_cst, 0.5), "lower": np.full(num_cst, 0.5)},
    {"upper": np.array([1.5, 0.5, 0.5]), "lower": np.full(num_cst, 0.5)},
]

# Line styles
colors = ["#0CAAEF", "#003268"]
poly_linewidth = 0.8
alpha = 0.5

fig, axs = plt.subplots(1, 2, figsize=[5.5, 2])

for i_foil, c in enumerate(coeff):
    for surf, color in zip(["upper", "lower"], colors):
        for i in range(num_cst):
            w = np.zeros(num_cst)
            w[i] = c[surf][i]
            y = DVGeometryCST.computeCSTCoordinates(x, N1, N2, w, yte)
            y = -y if surf == "lower" else y
            axs[i_foil].plot(x, y, color=color, linewidth=poly_linewidth, alpha=alpha)

        y = DVGeometryCST.computeCSTCoordinates(x, N1, N2, c[surf], yte)
        y = -y if surf == "lower" else y
        axs[i_foil].plot(x, y, color=color)

    axs[i_foil].set_aspect("equal")
    axs[i_foil].axis("off")
    axs[i_foil].set_ylim([-0.2, 0.45])

plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0)

fig.savefig(out_file)
