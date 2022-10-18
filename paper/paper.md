---
title: 'pyGeo: A geometry package for multidisciplinary design optimization'
tags:
  - geometry
  - Python
authors:
#   - name: Neil Wu
#     orcid: 0000-0001-8856-9661
#     affiliation: 1
#   - name: Hannah Hajdik
#     affiliation: 1
#   - name: Ben Brelje
#     affiliation: 1
#   - name: Gaetan Kenway
#     affiliation: 1
#   - name: Charles A. Mader
#     affiliation: 1
  - name: Joaquim R. R. A. Martins
    affiliation: 1
affiliations:
 - name: Department of Aerospace Engineering, University of Michigan
   index: 1
date: October 18, 2022
bibliography: paper.bib
---

# Summary
In the field of aerodynamic shape optimization, the geometry of an object is often modified by an optimization algorithm in order to improve its performance.
A common example is the shape optimization of an aircraft wing, where the aerodynamic drag is minimized by adjusting the external shape of the wing.
In a multidisciplinary design optimization context, aerodynamics and structural mechanics are considered and optimized simultaneously, which often provides additional benefit over optimizing only a single discipline.
In such cases, the geometry takes on an even greater significance in ensuring that multiple disciplines have a consistent and unified geometry representation.

pyGeo is a geometry package for aerodynamic and multidisciplinary design optimization.
It provides some basic geometry generation capabilities, several methods for geometry parameterization, numerous geometric constraints, and some utility functions for geometry manipulation.
The parameterizations and constraints are also differentiated to enable the use of gradient-based optimizers.

# Features


## Integrations
<!-- Integration with OM and MACH -->
## FFD


## Parametric Geometry Tools

The flexibility and ease of setup of the FFD method make it preferable for some applications.
In other applications, however, it can be beneficial to have the geometry defined in a more commonly accepted engineering format, such as a computer-aided design (CAD) model or other parametric definition of the geometry.
CAD is the industry standard, so if manufacturing of a design is desired then a CAD model defining it is required. 
If the geometry is designed parametrically, the relationships between design variables and geometry is defined in the model itself.
In an FFD model of a box, for example, the FFD points could represent the four corners of the box, but then the user would be required to define the planes in which points move to change the length, width, and height of the box.
In a parametric modelling tool, the user would create a box by defining its initial length, width, and height.
For either case, the length, width, and height (or a subset) can be controlled in the optimization process as design variables.

### Engineering Sketch Pad

The Engineering Sketch Pad (ESP) [@Haimes2013a] is an open-source CAD software for creating parametric geometries. 
ESP can be used to create general CAD models for applications ranging from conceptual to detailed design.
These geometries can then be used in external analysis tools. 
pyGeo contains the module DVGeoESP which translates an ESP model into a form usable for the MACH framework and updates it with the changes throughout the optimization. 


### OpenVSP

OpenVSP [@McDonald2022a] is a tool for creating 3D parametric geometries. 
Typically used for conceptual design, OpenVSP can be used to create geometries commonly used in aircraft vehicle applications. 
These geometries can then be used in external analysis tools. 
The DVGeoVSP module in pyGeo tranlates an OpenVSP model for use within the MACH framework and keeps it updated as the design variables are changed in the optimization. 




## Class Shape Transformation



The class shape transformation [@Kulfan2008] is a widely used airfoil parameterization. 

## Constraints

# One Dimension

# Two Dimensions

# Three Dimensions

# Statement of Need


# Acknowledgements


# References
