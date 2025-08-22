"""
This script demonstrates the deformation of a geometry object using FFD and
the process for exporting the geometry as a tecplot or IGES file.
"""

# Standard Python modules
import argparse

# External modules
import numpy as np

# First party modules
from pygeo import DVGeometry, pyGeo

input_files = "../../input_files/"


def deform_liftingsurface():
    # =========================================================================
    # Set Up pyGeo Object
    # =========================================================================
    # rst LiftingSurface
    # ---------------------  Lifting Surface Definition --------------------- #
    # Airfoil file
    naf = 10  # number of airfoils
    airfoil_list = ["./geo/rae2822.dat"] * naf

    # Airfoil leading edge positions
    x = np.linspace(0.0, 7.5, naf)
    y = np.linspace(0.0, 0.0, naf)
    z = np.linspace(0.0, 14.0, naf)

    offset = np.zeros((naf, 2))  # x-y offset applied to airfoil position before scaling

    # Airfoil rotations
    rot_x = [0.0] * naf
    rot_y = [0.0] * naf
    rot_z = [0.0] * naf

    # Airfoil scaling
    chord = np.linspace(5.0, 1.5, naf)

    # Run pyGeo
    geo = pyGeo(
        "liftingSurface",
        xsections=airfoil_list,
        scale=chord,
        offset=offset,
        x=x,
        y=y,
        z=z,
        rotX=rot_x,
        rotY=rot_y,
        rotZ=rot_z,
        tip="rounded",
        bluntTe=True,
        squareTeTip=True,
        teHeight=0.25 * 0.0254,
    )
    # rst LiftingSurface (end)

    # Deform Geometry Object and Output
    deform_DVGeo(geo)


def deform_iges():
    # =========================================================================
    # Set Up pyGeo Object
    # =========================================================================
    # rst IGES
    # ------------------------------ IGES File ------------------------------ #
    geo = pyGeo(fileName=input_files + "deform_geometry_wing.igs", initType="iges")
    geo.doConnectivity()
    # rst IGES (end)

    # Deform Geometry Object and Output
    deform_DVGeo(geo)


def deform_plot3d():
    # =========================================================================
    # Set Up pyGeo Object
    # =========================================================================
    # rst plot3d
    # ----------------------------- Plot3D File ----------------------------- #
    geo = pyGeo(fileName=input_files + "deform_geometry_wing.xyz", initType="plot3d")
    geo.doConnectivity()
    geo.fitGlobal()
    # rst plot3d (end)

    # Deform Geometry Object and Output
    deform_DVGeo(geo)


def deform_DVGeo(geo):
    # =========================================================================
    # Setup DVGeometry object
    # =========================================================================
    # rst DVGeometry
    DVGeo = DVGeometry(input_files + "deform_geometry_ffd.xyz")

    # Create reference axis
    nRefAxPts = DVGeo.addRefAxis("wing", xFraction=0.25, alignIndex="k")

    # Set the Twist Variable
    def twist(val, geo):
        for i in range(nRefAxPts):
            geo.rot_z["wing"].coef[i] = val[i]

    # Add the Twist Design Variable to DVGeo
    DVGeo.addGlobalDV(dvName="twist", value=[0] * nRefAxPts, func=twist, lower=-10, upper=10, scale=1.0)

    # Get Design Variables
    dvDict = DVGeo.getValues()

    # Set First Twist Section to 5deg
    dvDict["twist"][0] = 5

    # Set Design Variables
    DVGeo.setDesignVars(dvDict)
    # rst DVGeometry (end)

    # =========================================================================
    # Update pyGeo Object and output result
    # =========================================================================
    # rst UpdatePyGeo
    DVGeo.updatePyGeo(geo, "tecplot", "wingNew", nRefU=10, nRefV=10)
    # rst UpdatePyGeo (end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", type=str, default="iges", choices=["iges", "plot3d", "liftingsurface"])
    args = parser.parse_args()
    if args.input_type == "liftingsurface":
        deform_liftingsurface()
    elif args.input_type == "iges":
        deform_iges()
    elif args.input_type == "plot3d":
        deform_plot3d()
