import copy

import numpy as np

from pygeo import DVGeometry

FFDFile = "ffd.xyz"
DVGeo = DVGeometry(FFDFile)

nRefAxPts = DVGeo.addRefAxis("wing", xFraction=0.25, alignIndex="k")


def dihedral(val, geo):
    C = geo.extractCoef("wing")
    for i in range(1, nRefAxPts):
        C[i, 1] += val[i - 1]
    geo.restoreCoef(C, "wing")


# rst Twist
def twist(val, geo):
    for i in range(1, nRefAxPts):
        geo.rot_z["wing"].coef[i] = val[i - 1]


# rst Taper
def taper(val, geo):
    s = geo.extractS("wing")
    slope = (val[1] - val[0]) / (s[-1] - s[0])
    for i in range(nRefAxPts):
        geo.scale_x["wing"].coef[i] = slope * (s[i] - s[0]) + val[0]


nTwist = nRefAxPts - 1
DVGeo.addGlobalDV(dvName="dihedral", value=[0] * nTwist, func=dihedral, lower=-10, upper=10, scale=1)
DVGeo.addGlobalDV(dvName="twist", value=[0] * nTwist, func=twist, lower=-10, upper=10, scale=1)
DVGeo.addGlobalDV(dvName="taper", value=[1] * 2, func=taper, lower=0.5, upper=1.5, scale=1)

# Comment out one or the other
DVGeo.addLocalDV("local", lower=-0.5, upper=0.5, axis="y", scale=1)

dvDict = DVGeo.getValues()
dvDictCopy = copy.deepcopy(dvDict)
dvDictCopy["twist"] = np.linspace(0, 50, nRefAxPts)[1:]
DVGeo.setDesignVars(dvDictCopy)
DVGeo.writePlot3d("ffd_deformed-twist.xyz")

dvDictCopy = copy.deepcopy(dvDict)
dvDictCopy["dihedral"] = np.linspace(0, 3, nRefAxPts)[1:]
DVGeo.setDesignVars(dvDictCopy)
DVGeo.writePlot3d("ffd_deformed-dihedral.xyz")

dvDictCopy = copy.deepcopy(dvDict)
dvDictCopy["taper"] = np.array([1.2, 0.5])
DVGeo.setDesignVars(dvDictCopy)
DVGeo.writePlot3d("ffd_deformed-taper.xyz")
