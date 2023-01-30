# External modules
from stl import mesh

# First party modules
from pygeo import DVGeometryESP

# rst create dvgeo start
# initialize the DVGeometry object with the csm file
csm_file = "naca0012.csm"
DVGeo = DVGeometryESP(csm_file)
# rst create dvgeo end

# rst add pointset start
stlmesh = mesh.Mesh.from_file("esp_airfoil.stl")
# create a pointset. pointsets are of shape npts by 3 (the second dim is xyz coordinate)
# already have the airfoil mesh as a triangulated surface (stl file)
# each vertex set is its own pointset, so we actually add three pointsets
DVGeo.addPointSet(stlmesh.v0, "mesh_v0")
DVGeo.addPointSet(stlmesh.v1, "mesh_v1")
DVGeo.addPointSet(stlmesh.v2, "mesh_v2")
# rst add pointset end

# rst add DV start
# Now that we have pointsets added, we should parameterize the geometry.
# Adding geometric design variable to make modifications to the airfoil geometry directly
DVGeo.addVariable("camber_dv")
DVGeo.addVariable("maxloc_dv")
DVGeo.addVariable("thickness_dv")
# rst add DV end

# rst perturb DV start
# Perturb some local variables and observe the effect on the surface
dvdict = DVGeo.getValues()
dvdict["camber_dv"] = 0.09
dvdict["maxloc_dv"] = 0.8
dvdict["thickness_dv"] = 0.2
DVGeo.setDesignVars(dvdict)

# write out the new ESP model
DVGeo.writeCSMFile("modified.csm")
# rst perturb DV end
