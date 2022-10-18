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


# Integrations

pyGeo was originally developed to use FFDs in MACH [@Kenway2010b].

pyGeo can be used as the basis for the geometry within the MDO framework MDO of Aircraft Configurations at High Fidelity (MACH) [@Kenway2014a], [@Kenway2014c].
Through MPhys, a wrapper for MACH, pyGeo's features can also be used within another MDO framework, OpenMDAO [@Gray2019a].

The package pyOptSparse [@Wu2020a] is used to interface with the optimizer directly. 
pyGeo's modules are used to send design variables and constraints to pyOptSparse rather than the user handling these interactions.

# Geometry Generation

# Geometry Parameterization

pyGeo handles geometry manipulation through DVGeo objects. 
There are different types of DVGeo objects for different methods of geometry parameterization, but all use the same interface and create design variables which are passed to the rest of the framework for optimization. 

## Free-form deformation

The free-form deformation (FFD) method [Sederberg1986] is one of the most popular three-dimensional geometry parameterization approaches.
In this approach, the entire geometry is embedded in a flexible jelly-like block, and manipulated together with the control points of the block.
By introducing different densities of control points, a high degree of geometry control can be obtained.

Compared to other parameterizations, the FFD method has several key advantages.
Since the entire geometry is embedded, there is no need to start with or reverse-engineer a parametric geometry representation as commonly done with B-spline-based methods, where a least-squares fit is needed to generate a B-spline surface representation.
Rather than parameterizing the geometry directly, the geometric _deformation_ is parameterized instead.
This decoupling of geometry definition from geometric deformation allows for control and refinement of the deformation independently of the original geometry.
When working with multiple geometries, for example an optimization involving an aerodynamic and structural surface simultaneously, both surfaces can be embedded into the same FFD volume.
As both surfaces would be manipulated by the same volume, coincident surfaces remain coincident after deformations and this approach ensures consistency between disparate geometries.


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
pyGeo contains a module, DVGeoCST, that can be used for airfoil optimization. 
This module contains analytic derivatives for each design variable available in CST optimization. 
Unlike other DVGeo modules, this one can only be used for 2D problems, such as airfoil optimization. 

## Constraints

pyGeo also includes geometric constraints through the DVCon module. 
Constraints are all differentiated in order to use within gradient-based optimization. 
DVCon creates constraint objects which are passed to pyOptSparse. 

The main type of geometric constraint in DVCon is a dimensional constraint. 
In 1D this is thickness, for 2D area, and for 3D volume.
In each case the dimension of the geometry is constrained from deviating from the initial value by some amount set by the user. 
For example, a common constraint is to prevent the internal volume of a wing from falling below some set threshold.

<!-- Triangulated surface constraint -->

# Statement of Need


# Acknowledgements


# References
