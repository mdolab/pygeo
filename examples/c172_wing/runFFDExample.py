# External modules
import numpy as np
from stl import mesh

# First party modules
from pygeo import DVGeometry


def create_fresh_dvgeo():
    # The Plot3D file ffdbox.xyz contains the coordinates of the free-form deformation (FFD) volume
    # The "i" direction of the cube consists of 10 points along the x (streamwise) axis
    # The "j" direction of the cube is 2 points up and down (y axis direction)
    # The "k" direction of the cube is 8 along the span (z axis direction)
    FFDfile = "ffdbox.xyz"

    # initialize the DVGeometry object with the FFD file
    DVGeo = DVGeometry(FFDfile)

    stlmesh = mesh.Mesh.from_file("baseline_wing.stl")
    # create a pointset. pointsets are of shape npts by 3 (the second dim is xyz coordinate)
    # already have the wing mesh as a triangulated surface (stl file)
    # each vertex set is its own pointset, so we actually add three pointsets
    DVGeo.addPointSet(stlmesh.v0, "mesh_v0")
    DVGeo.addPointSet(stlmesh.v1, "mesh_v1")
    DVGeo.addPointSet(stlmesh.v2, "mesh_v2")
    return DVGeo, stlmesh


# rst add local DV
# Now that we have pointsets added, we should parameterize the geometry.
# Adding local geometric design variables to make local modifications to FFD box
# This option will perturb all the control points but only the y (up-down) direction
DVGeo, stlmesh = create_fresh_dvgeo()
DVGeo.addLocalDV("shape", lower=-0.5, upper=0.5, axis="y", scale=1.0)

# Perturb some local variables and observe the effect on the surface
dvdict = DVGeo.getValues()
dvdict["shape"][DVGeo.getLocalIndex(0)[:, 1, 5]] += 0.15
dvdict["shape"][DVGeo.getLocalIndex(0)[3, 1, 1]] += 0.15
DVGeo.setDesignVars(dvdict)

# write the perturbed wing surface and ffd to output files
stlmesh.vectors[:, 0, :] = DVGeo.update("mesh_v0")
stlmesh.vectors[:, 1, :] = DVGeo.update("mesh_v1")
stlmesh.vectors[:, 2, :] = DVGeo.update("mesh_v2")
stlmesh.save("local_wing.stl")
DVGeo.writeTecplot("local_ffd.dat")

# rst ref axis
DVGeo, stlmesh = create_fresh_dvgeo()
# add a reference axis named 'c4' to the FFD volume
# it will go in the spanwise (k) direction and be located at the quarter chord line
nrefaxpts = DVGeo.addRefAxis("c4", xFraction=0.25, alignIndex="k")
# note that the number of reference axis points is the same as the number
# of FFD nodes in the alignIndex direction
print("Num ref axis pts: ", str(nrefaxpts), " Num spanwise FFD: 8")

# can write the ref axis geometry to a Tecplot file for visualization
DVGeo.writeRefAxes("local")

# rst twist
# global design variable functions are callbacks that take two inputs:
# a design variable value from the optimizer, and
# the DVGeometry object itself

# the rot_z attribute defines rotations about the z axis
# along the ref axis


def twist(val, geo):
    for i in range(nrefaxpts):
        geo.rot_z["c4"].coef[i] = val[i]


# now create global design variables using the callback functions
# we just defined
DVGeo.addGlobalDV("twist", func=twist, value=np.zeros(nrefaxpts), lower=-10, upper=10, scale=0.05)

# rst sweep
# sweeping back the wing requires modifying the actual
# location of the reference axis, not just defining rotations or stretching
# about the axis


def sweep(val, geo):
    # the extractCoef method gets the unperturbed ref axis control points
    C = geo.extractCoef("c4")
    C_orig = C.copy()
    # we will sweep the wing about the first point in the ref axis
    sweep_ref_pt = C_orig[0, :]

    theta = -val[0] * np.pi / 180
    rot_mtx = np.array([[np.cos(theta), 0.0, -np.sin(theta)], [0.0, 1.0, 0.0], [np.sin(theta), 0.0, np.cos(theta)]])

    # modify the control points of the ref axis
    # by applying a rotation about the first point in the x-z plane
    for i in range(nrefaxpts):
        # get the vector from each ref axis point to the wing root
        vec = C[i, :] - sweep_ref_pt
        # need to now rotate this by the sweep angle and add back the wing root loc
        C[i, :] = sweep_ref_pt + rot_mtx @ vec
    # use the restoreCoef method to put the control points back in the right place
    geo.restoreCoef(C, "c4")


DVGeo.addGlobalDV("sweep", func=sweep, value=0.0, lower=0, upper=45, scale=0.05)

# rst set DV
# set a twist distribution from -10 to +20 degrees along the span
dvdict = DVGeo.getValues()
dvdict["twist"] = np.linspace(-10.0, 20.0, nrefaxpts)
DVGeo.setDesignVars(dvdict)
# write out the twisted wing and FFD
stlmesh.vectors[:, 0, :] = DVGeo.update("mesh_v0")
stlmesh.vectors[:, 1, :] = DVGeo.update("mesh_v1")
stlmesh.vectors[:, 2, :] = DVGeo.update("mesh_v2")
stlmesh.save("twist_wing.stl")
DVGeo.writeTecplot("twist_ffd.dat")

# rst set DV 2
# now add some sweep and change the twist a bit
dvdict = DVGeo.getValues()
dvdict["sweep"] = 30.0
dvdict["twist"] = np.linspace(0.0, 20.0, nrefaxpts)
DVGeo.setDesignVars(dvdict)

# write out the swept / twisted wing and FFD
stlmesh.vectors[:, 0, :] = DVGeo.update("mesh_v0")
stlmesh.vectors[:, 1, :] = DVGeo.update("mesh_v1")
stlmesh.vectors[:, 2, :] = DVGeo.update("mesh_v2")
stlmesh.save("sweep_wing.stl")
DVGeo.writeTecplot("sweep_ffd.dat")
DVGeo.writeRefAxes("sweep")

# rst set DV 3
# we can change the chord distribution by using the
# scale_x attribute which stretches/shrinks the pointset
# about the ref axis in the x direction


def chord(val, geo):
    for i in range(nrefaxpts):
        geo.scale_x["c4"].coef[i] = val[i]


# rst set DV 4
# set up a new DVGeo with all three global design vars plus local thickness
DVGeo, stlmesh = create_fresh_dvgeo()
nrefaxpts = DVGeo.addRefAxis("c4", xFraction=0.25, alignIndex="k")
DVGeo.addGlobalDV("twist", func=twist, value=np.zeros(nrefaxpts), lower=-10, upper=10, scale=0.05)
DVGeo.addGlobalDV("chord", func=chord, value=np.ones(nrefaxpts), lower=0.01, upper=2.0, scale=0.05)
DVGeo.addGlobalDV("sweep", func=sweep, value=0.0, lower=0, upper=45, scale=0.05)
DVGeo.addLocalDV("thickness", axis="y", lower=-0.5, upper=0.5)

# change everything and the kitchen sink
dvdict = DVGeo.getValues()
dvdict["twist"] = np.linspace(0.0, 20.0, nrefaxpts)
# scale_x should be set to 1 at baseline, unlike the others which perturb about 0
# the following will produce a longer wing root and shorter wing tip
dvdict["chord"] = np.linspace(1.2, 0.2, nrefaxpts)
# randomly perturbing the local variables should make a cool wavy effect
dvdict["thickness"] = np.random.uniform(-0.1, 0.1, 160)
dvdict["sweep"] = 30.0
DVGeo.setDesignVars(dvdict)

# write out to data files for visualization
stlmesh.vectors[:, 0, :] = DVGeo.update("mesh_v0")
stlmesh.vectors[:, 1, :] = DVGeo.update("mesh_v1")
stlmesh.vectors[:, 2, :] = DVGeo.update("mesh_v2")
stlmesh.save("all_wing.stl")
DVGeo.writeTecplot("all_ffd.dat")
DVGeo.writeRefAxes("all")
