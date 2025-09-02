---
title: 'pyGeo: A geometry package for multidisciplinary design optimization'
tags:
  - geometry
  - Python
authors:
  - name: Hannah M. Hajdik
    orcid: 0000-0002-5103-7159
    affiliation: 1
  - name: Anil Yildirim
    orcid: 0000-0002-1814-9191
    affiliation: 1
  - name: Ella Wu
    orcid: 0000-0001-8856-9661
    affiliation: 1
  - name: Benjamin J. Brelje
    orcid: 0000-0002-9819-3028
    affiliation: 1
  - name: Sabet Seraj
    orcid: 0000-0002-7364-0071
    affiliation: 1
  - name: Marco Mangano
    orcid: 0000-0001-8495-3578
    affiliation: 1
  - name: Joshua L. Anibal
    orcid: 0000-0002-7795-2523
    affiliation: 1
  - name: Eirikur Jonsson
    orcid: 0000-0002-5166-3889
    affiliation: 1
  - name: Eytan J. Adler
    orcid: 0000-0002-8716-1805
    affiliation: 1
  - name: Charles A. Mader
    orcid: 0000-0002-2744-1151
    affiliation: 1
  - name: Gaetan K. W. Kenway
    orcid: 0000-0003-1352-5458
    affiliation: 1
  - name: Joaquim R. R. A. Martins
    orcid: 0000-0003-2143-1478
    affiliation: 1
affiliations:
 - name: Department of Aerospace Engineering, University of Michigan
   index: 1
date: October 18, 2022
bibliography: paper.bib
header-includes: \usepackage{subcaption}
---

# Summary
Geometry parameterization is a key challenge in shape optimization.
Parameterizations must accurately capture the design intent and perform well in optimization.
In multidisciplinary design optimization (MDO), the parameterization must additionally represent the shape consistently across each discipline.

pyGeo is a geometry package for three-dimensional shape manipulation tailored for aerodynamic and multidisciplinary design optimization.
It provides several methods for geometry parameterization, geometric constraints, and utility functions for geometry manipulation.
pyGeo computes derivatives for all parameterization methods and constraints, facilitating efficient gradient-based optimization.

# Features

## Integrations
pyGeo is the geometry manipulation engine within the MDO of Aircraft Configurations at High Fidelity (MACH) framework [@Kenway2014a; @Kenway2014c], specializing in high-fidelity aerostructural optimization.
pyGeo can be used as a stand-alone package and is integrated into MPhys[^1], a more general framework for high-fidelity multiphysics problems built with OpenMDAO [@Gray2019a].
MACH and MPhys use pyOptSparse [@Wu2020a] to interface with optimization algorithms.

[^1]: \url{https://github.com/OpenMDAO/mphys}

pyGeo's interface for design variables and constraints is independent of which disciplinary models access the geometry.
This means that pyGeo can interact with different disciplines (such as structures and aerodynamics) in the same way.
This also facilitates a direct comparison of the behavior or performance of two alternative models for a discipline using the same geometry parameterization [@Adler2022c].

## Geometry Parameterization with pyGeo
pyGeo contains several options for parameterizing geometry: variations on the free-form deformation (FFD) method, interfaces to external parametric modeling tools, and an analytic parameterization.
Because each parameterization method uses a common interface for interacting with the rest of the MACH framework, any surface parameterization can be used in place of another within an optimization setup [@Hajdik2023a].
The choice of parameterization depends on the user's experience, the geometry details, and whether the user needs the final design in a specific format.

### Free-form Deformation
The FFD method [@Sederberg1986] is one of the most popular three-dimensional geometry parameterization approaches [@Zhang2018a].
This approach embeds the entire reference geometry in a parameterized volume.
The set of control points that determines the shape of the volume is displaced to manipulate the points inside.
The user can have a high degree of control over the geometry by selecting different control point densities and locations.

Individual control points can be moved to obtain local shape modifications.
In pyGeo, these are referred to as _local_ design variables because a single control point is affected.
Conversely, it is also common to define geometric operations involving a collection of control points across the entire FFD block.
These are referred to as _global_ design variables in pyGeo.
For example, wing twist variables can be defined as rotations of the control points about a reference axis that runs along the wing.
\autoref{fig:FFD_DV} shows a few common planform design variables for an aircraft wing.

Design variables formulated from groupings of FFD control points often exhibit ill-conditioning.
A parameterization based on singular value decomposition is also possible within pyGeo to alleviate this issue [@Wu2022b].

![Examples of common wing planform design variables.\label{fig:FFD_DV}](ffd_dvs.pdf)

In addition to the basic FFD implementation, pyGeo offers two additional features: hierarchical FFD and multi-component FFD.

#### Hierarchical FFD
FFD objects can be organized in a hierarchical structure within pyGeo.
Dependent, "child" FFD blocks can be embedded in the main, "parent" FFD block to enable modifications on a subset of the entire geometry.
pyGeo first propagates the parent deformations to both the geometry and the child control points and then propagates the deformations of the child control points to their subset of the geometry.
One of the advantages of using this approach is that each FFD block can have its own independent reference axis to be used for global design variables such as rotations and scaling.
\autoref{fig:ffd_child} shows a case where the parent FFD block is used to manipulate the shape of a blended wing body aircraft while its control surface is deformed using a child FFD block.

![Example of parameterization through parent-child FFD blocks [@Lyu2014c]. \label{fig:ffd_child}](child_ffd.pdf)

#### Multi-component FFD
The basic FFD implementation lacks flexibility when the geometry has intersecting components.
In such cases, pyGeo can parameterize each component using FFD and ensure a watertight surface representation at the component intersections using an inverse-distance surface deformation method [@Yildirim2021b].
\autoref{fig:ffd_multi} shows an example of a component-based FFD setup for a supersonic transport aircraft.

![Example of FFD parameterization with intersecting components [@Seraj2022a]. \label{fig:ffd_multi}](ffd_multi.png){ width=75% }

### Parametric Geometry Tools
pyGeo contains interfaces to two parametric geometry tools, the Engineering Sketch Pad (ESP) [@Haimes2013a] and OpenVSP [@McDonald2022a].
ESP is CAD-based, while OpenVSP is a conceptual design tool.
The two packages have different capabilities, but both directly define the geometry with design variables, and the created geometry can be used in external analysis tools.

pyGeo interfaces with ESP and OpenVSP in similar ways.
In both cases, pyGeo takes an instance of the model, and its points are associated with coordinates in a mesh from a solver in the MACH framework.
For ESP (\autoref{fig:esp_example}) and OpenVSP models (\autoref{fig:vsp_example}), the pyGeo interface to the respective software stores the model in a form usable within the MACH framework and updates it as design variables are changed throughout the optimization.

![Example of ESP models of hydrogen tanks used through pyGeo [@Brelje2021a]. \label{fig:esp_example}](esp_example.png){ width=75% }

![Example of an OpenVSP aircraft model used through OpenVSP's pyGeo interface [@Yildirim2022a]. \label{fig:vsp_example}](vsp_example.png){ width=85% }

### Class Shape Transformation
The class shape transformation (CST) [@Kulfan2008] is a popular airfoil parameterization.
It generates a shape using Bernstein polynomials to scale a class function, which is most often a base airfoil shape.
pyGeo contains an implementation of this airfoil parameterization that supports design variables for the Bernstein polynomial weightings, the class function parameters, and the airfoil chord length.
pyGeo's CST implementation can only be used for 2D problems, such as airfoil optimization (\autoref{fig:cst_example}).

![Airfoil defined by three CST coefficients on each surface undergoing a perturbation in one Bernstein polynomial. \label{fig:cst_example}](cst_example.pdf){ width=85% }

## Constraints
pyGeo also includes geometric constraints to prevent geometrically undesirable designs.
The most commonly-used class of geometry constraints in pyGeo involves tracking one or more linear dimensions on the optimized object's surface.
These constraints are created by specifying a single point, a line, or an array of points, along with a normal direction, then computing two line-surface intersection points.
Some commonly used geometric constraints in shape optimization, such as thickness, area, and volume constraints (\autoref{fig:constraint}) can be computed using variations on this approach, which is computationally cheap and robust [@Brelje2020a].

![Thickness and volume constraints demonstrated on an wing section [@Brelje2020a]. \label{fig:constraint}](constraints_3d.pdf){ width=75% }

If a more complex geometry needs to be integrated into an optimized surface, pyGeo supports an alternative geometric constraint formulation based on arbitrary triangulated surfaces as illustrated in \autoref{fig:trisurf} [@Brelje2020a].

![Triangulated surface constraint used to optimize an aeroshell around a complex geometry [@Brelje2020a].\label{fig:trisurf}](trisurfcon.pdf)


# Parallelism
pyGeo can optionally work under distributed memory parallelism using MPI (Message Passing Interface), which is required when interfacing with large-scale computational fluid dynamics (CFD) applications.
For example, the computational mesh may be partitioned and distributed among many processors by the CFD solver, and each processor may be aware of only its portion of the mesh.
pyGeo can handle such scenarios by independently manipulating the geometry on each processor and aggregating the constraints across all processors when communicating with the optimizer.


# Derivative Computation
In addition to geometry manipulation and constraints, pyGeo can compute derivatives of these operations with respect to design variables.
For the geometric deformation, pyGeo can compute the Jacobian matrix
\begin{equation*}
\frac{\mathrm{d}X_s}{\mathrm{d}x} ,
\end{equation*}
where $X_s$ is the vector of surface mesh coordinates, and $x$ is the vector of geometric design variables.

Similarly, pyGeo can compute the constraint Jacobian matrix
\begin{equation*}
\frac{\mathrm{d}g}{\mathrm{d}x} ,
\end{equation*}
where $g$ is the vector of geometric constraints.

For the FFD parameterization, these derivatives are computed using analytic methods [@Martins2021, Sec. 6.6] and the complex-step method [@Martins2003a].
For the interfaces to OpenVSP and ESP, the derivatives are computed with parallel finite differences.
The CST derivatives are computed analytically.


# Statement of Need
Few open-source packages exist with comparable functionalities.
To the authors' best knowledge, the only other open-source CFD-based optimization framework that contains a geometry parameterization method is SU2 [@Economon2016a].
It supports Hicks--Henne bump functions [@Hicks1978] for airfoil optimizations and the FFD method for 3D cases.
However, this geometry parameterization cannot be used with other solvers (aerodynamic or otherwise) because it is tightly integrated into SU2.

While both OpenVSP and ESP can be used directly in optimization without using pyGeo, they lack the capabilities needed for high-fidelity MDO when used as stand-alone tools.
pyGeo fills these gaps through parallelism, efficient gradients, and geometric constraints.
It keeps OpenVSP and ESP in the optimization loop and provides a standard interface to these tools for use with external solvers.

pyGeo is maintained and developed by the MDO Lab[^2] at the University of Michigan and is actively used for MDO applications in research and industry.
Having different parameterization choices in pyGeo has been useful because the best parameterization depends on the type of problem.
pyGeo's standard FFD implementation is the most commonly used parameterization [@Kenway2014c; @Bons2019a].
The hierarchical FFD method was used to optimize a blended wing body aircraft [@Lyu2014c], a hydrofoil [@Liao2021a], and a wind turbine [@Madsen2019a].
The method for using multiple FFD blocks has been used to optimize a conventional aircraft [@Yildirim2021b], a T-shaped hydrofoil [@Liao2022], and a supersonic transport aircraft [@Seraj2022a].
The interface to ESP made it possible to parameterize hydrogen tanks within a combined aerostructural and packaging optimization [@Brelje2021a].
pyGeo's OpenVSP interface was used in aeropropulsive optimizations [@Yildirim2022a].
The implementation of CST airfoil parameterization was used to compare methods for airfoil optimization [@Adler2022c].

<!-- HH: Removed overarching sentence because this last paragraph acts as that sentence and this section is a statement of need, not a conclusion. This is in line with the format of pyOptSparse. -->


[^2]: \url{https://mdolab.engin.umich.edu}


# Acknowledgements
We are grateful to the numerous pyGeo users who have contributed their time to the code and its maintenance over the years.

# References
