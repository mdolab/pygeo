---
title: 'pyGeo: A geometry package for multidisciplinary design optimization'
tags:
  - geometry
  - Python
authors:
  - name: Hannah Hajdik
    orcid: 0000-0002-5103-7159
    affiliation: 1
  - name: Anil Yildirim
    orcid: 0000-0002-1814-9191
    affiliation: 1
  - name: Neil Wu
    orcid: 0000-0001-8856-9661
    affiliation: 1
  - name: Ben Brelje
    orcid: 0000-0002-9819-3028
    affiliation: 1
  - name: Sabet Seraj
    orcid: 0000-0002-7364-0071
    affiliation: 1
  - name: Marco Mangano
    orcid: 0000-0001-8495-3578
    affiliation: 1
  - name: Josh Anibal
    orcid: 0000-0002-7795-2523
    affiliation: 1
  - name: Eirikur Jonsson
    orcid: 0000-0002-5166-3889
    affiliation: 1
  - name: Eytan Adler
    orcid: 0000-0002-8716-1805
    affiliation: 1
  - name: Charles A. Mader
    orcid: 0000-0002-2744-1151
    affiliation: 1
  - name: Gaetan Kenway
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

In shape optimization, an algorithm modifies a body's geometry to improve its performance.
A common shape optimization example is adjusting the geometry of an aircraft wing to minimize aerodynamic drag computed via computational fluid dynamics (CFD).
Multidisciplinary design optimization (MDO) couples multiple disciplines, such as aerodynamics and structural mechanics, to optimize them simultaneously.
In MDO, the geometry must be represented consistently across multiple disciplines.

pyGeo is a geometry package for three-dimensional shape manipulation tailored for aerodynamic and multidisciplinary design optimization.
It provides several methods for geometry parameterization, geometric constraints, and utility functions for geometry manipulation.
pyGeo computes derivatives for all parameterization methods and constraints, facilitating efficient gradient-based optimization.

# Features

## Integrations

pyGeo is the geometry manipulation engine within the MDO of Aircraft Configurations at High Fidelity (MACH) framework [@Kenway2014a; @Kenway2014c], which specializes in high-fidelity aerostructural optimization.
pyGeo is also integrated into MPhys[^1], a more general framework for high-fidelity multiphysics problems built with OpenMDAO [@Gray2019a].
Both MACH and MPhys use pyOptSparse [@Wu2020a] to interface with optimization algorithms.

[^1]: \url{https://github.com/OpenMDAO/mphys}

pyGeo's interface for design variables and constraints is independent of which solvers are accessing the geometry.
This means that pyGeo geometries can interact with different types of solvers, such as structures and aerodynamics, in the same way.
This also allows direct comparison of the behavior or performance of two different solvers within the same discipline using the same geometric parameterization for each, such as two different flow solvers [@Adler2022c].

## Geometry Parameterization with pyGeo


pyGeo contains several options for parameterizing geometry: variations on the FFD method, interfaces to external parametric modeling tools, and an analytic parameterization.
Because each parameterization method uses a common interface for interacting with the rest of the MACH framework, any surface parameterization can be used in place of another within an optimization setup [@Hajdik2023a].
The choice of parameterization depends on the user's experience, the geometry details, and whether the user needs the final design in a specific format.

### Free-form Deformation

The free-form deformation (FFD) method [@Sederberg1986] is one of the most popular three-dimensional geometry parameterization approaches [@Zhang2018a].
This approach embeds the entire reference geometry in a parameterized volume. 
The set of control points that determine the shape of the volume are displaced to manipulate the points inside. 
The user can have a high degree of control over the geometry by selecting different control point densities and locations.

Individual control points can be moved to obtain local shape modifications.
In pyGeo, these are referred to as _local_ design variables because a single control point is affected.
Conversely, it is also common to define geometric operations involving a collection of control points across the entire FFD block.
These are referred to as _global_ design variables in pyGeo.
For example, wing twist variables can be defined as rotations of the control points about a reference axis that runs along the wing. 
\autoref{fig:FFD_DV} shows a few common planform design variables for an aircraft wing.

Design variables formulated from groupings of FFD control points often exhibit ill conditioning. 
To alleviate this, a parameterization based on singular value decomposition is also possible within pyGeo [@Wu2022b].

![Examples of common wing planform design variables.\label{fig:FFD_DV}](ffd_dvs.pdf)

In addition to the basic FFD implementation, pyGeo offers two additional features: hierarchical FFD and multi-component FFD.

#### Hierarchical FFD
FFD objects can be organized in a hierarchical structure within pyGeo.
Dependent, "child" FFD blocks can be embedded in the main, "parent" FFD block to enable modifications on a subset of the full geometry.
pyGeo first propagates the parent deformations to both the geometry and the child control points and then propagates the deformations of the child control points to their subset of the geometry. <!--MM: I would like to double check this sentence with Anil-->
One of the advantages of using this approach is that each FFD block can have its own independent reference axis to be used for global design variables such as rotations and scaling.
This has facilitated the definition of control surface deflections [@Lyu2014c] and hydrofoil design [@Liao2021a].
\autoref{fig:ffd_child} from the former paper shows a case where the parent FFD block is used to manipulate the shape of an entire wing of a blended wing body aircraft while the control surface on that wing is deformed using a child FFD block.

![Example of parameterization through parent-child FFD blocks [@Lyu2014c]. \label{fig:ffd_child}](child_ffd.pdf)

#### Multi-component FFD

The basic FFD implementation lacks flexibility when the geometry has intersecting components.
In such cases, pyGeo can parameterize each component using FFD and ensure a watertight surface representation at the component intersections using an inverse-distance surface deformation method [@Yildirim2021b].
\autoref{fig:ffd_multi} shows an example of a component-based FFD setup for a supersonic transport aircraft.

![Example of FFD parameterization with intersecting components [@Seraj2022a]. \label{fig:ffd_multi}](ffd_multi.png)

### Parametric Geometry Tools

pyGeo contains interfaces to two parametric geometry tools, the Engineering Sketch Pad (ESP) [@Haimes2013a] and OpenVSP [@McDonald2022a].
ESP is a CAD software, while OpenVSP is a conceptual design tool.
The two packages have different capabilities, but both directly define the geometry with design variables, and the created geometry can be used in external analysis tools.

pyGeo interfaces with ESP and OpenVSP in similar ways.
In both cases, pyGeo takes an instance of the model and its points are associated with coordinates in a mesh from a solver in the MACH framework.
For ESP and OpenVSP models, the pyGeo interface to the respective software stores the model in a form usable within the MACH framework and updates it as design variables are changed throughout the optimization.
The pyGeo interface to ESP was used by [@Brelje2021a] to parameterize hydrogen tanks (\autoref{fig:esp_example}) that were packaged within an aircraft wing as part of an aerostructural optimization.
pyGeo's OpenVSP interface was used to parameterize the full aircraft configuration (\autoref{fig:vsp_example}) studied in the aeropropulsive optimization work in [@Yildirim2022a].

![Example of ESP models of hydrogen tanks used through pyGeo [@Brelje2021a]. \label{fig:esp_example}](esp_example.png)

![Example of a VSP aircraft model used through VSP's pyGeo interface [@Yildirim2022a]. \label{fig:vsp_example}](vsp_example.png)

### Class Shape Transformation

The class shape transformation (CST) methodology [@Kulfan2008] is a popular airfoil parameterization.
It generates a shape using Bernstein polynomials to scale a class function, which is most often a base airfoil shape.
pyGeo contains an implementation of this airfoil parameterization that supports design variables for the Bernstein polynomial weightings, the class function parameters, and the airfoil chord length.
pyGeo's CST implementation can only be used for 2D problems, such as airfoil optimization (\autoref{fig:cst_example}).

![Airfoil defined by three CST coefficients on each surface undergoing a perturbation in one Bernstein polynomial. \label{fig:cst_example}](cst_example.pdf)

## Constraints

pyGeo also includes geometric constraints to prevent geometrically undesirable designs.
The most commonly-used class of geometry constraints in pyGeo involves tracking one or more linear dimensions on the optimized object's surface.
These constraints are created by specifying a single point, a line, or an array of points, along with a normal direction, then computing two line-surface intersection points.
Some commonly used geometric constraints in shape optimization, such as thickness, area, and volume constraints (\autoref{fig:constraint}) can be computed using variations on this approach, which is computationally cheap and robust [@Brelje2020a].

![Thickness and volume constraints demonstrated on an wing section [@Brelje2020a]. \label{fig:constraint}](constraints_3d.pdf)

If a more complex geometry needs to be integrated into an optimized surface, pyGeo supports an alternative geometric constraint formulation based on arbitrary triangulated surfaces as described in [@Brelje2020a] and illustrated in \autoref{fig:trisurf}.

![Triangulated surface constraint used to optimize an aeroshell around a complex geometry [@Brelje2020a].\label{fig:trisurf}](trisurfcon.pdf)


# Parallelism
pyGeo can optionally work under distributed memory parallelism using MPI, which is a requirement when interfacing with large-scale CFD applications.
For example, the computational mesh may be partitioned and distributed among many processors by the CFD solver, and each processor may be aware of only its portion of the mesh.
pyGeo can handle such scenarios by independently manipulating the geometry on each processor and aggregating the constraints across all processors when communicating with the optimizer.


# Derivative Computation
In addition to geometry manipulation and constraints, pyGeo can compute derivatives of these operations with respect to design variables.
For the geometric deformation, pyGeo can compute the Jacobian
\begin{equation*}
\frac{\mathrm{d}X_s}{\mathrm{d}x} ,
\end{equation*}
where $X_s$ is the vector of surface mesh coordinates, and $x$ is the vector of geometric design variables.

Similarly, pyGeo can compute the constraint Jacobian
\begin{equation*}
\frac{\mathrm{d}g}{\mathrm{d}x} ,
\end{equation*}
where $g$ is the vector of geometric constraints.

For the FFD parameterization, these derivatives are computed using a combination of analytic methods [@Martins2021] and the complex-step method [@Martins2003a].
For the interfaces to OpenVSP and ESP, the derivatives are computed with parallel finite differences.
The CST derivatives are computed analytically.


# Statement of Need
Few open-source packages exist with comparable functionalities.
To the authors' best knowledge, the only other open-source CFD-based optimization framework that contains geometry parameterization is SU2 [@Economon2016a].
<!-- NW: changed to 'CFD-based framework' since in other fields there may be some geometry-related stuff -->
It supports Hicks--Henne bump functions [@Hicks1978] for airfoil optimizations and the FFD method for three-dimensional cases.
However, it cannot be used with other solvers because it is tightly integrated into the CFD solver.


While both OpenVSP and ESP can be used directly in optimization without using pyGeo, they lack capabilities needed for high-fidelity MDO when used as stand-alone tools.
pyGeo fills in these gaps through parallelism, efficient gradients, and geometric constraints.
It keeps OpenVSP and ESP in the optimization loop and provides a standard interface to these tools for their use with external solvers.

pyGeo has been used extensively in aerodynamic and aerostructural optimizations in aircraft, hydrofoil, and wind turbine applications.
<!-- [] TODO SS-: We should add a few citations for the basic FFD functionality. -->
<!-- HMH: Any ideas on which would be representative? Neil suggested uCRM, maybe we want a wind turbine and/or hydrofoil paper as well for ~range~  -->
<!-- MM: we can re-use the citations above for non-aircraft examples. For aero stuff, one of the latest Nick Bons' paprs could also be a good fit-->
The different parameterizations within pyGeo have all been necessary for different optimization problems, depending on the geometry involved.
The interface to ESP made it possible to parameterize hydrogen tanks within a combined aerostructural and packaging optimization [@Brelje2021a].
pyGeo's OpenVSP interface was used in aeropropulsive optimizations [@Yildirim2022a].

The implementation of CST airfoil parameterization was used to compare methods for airfoil optimization [@Adler2022c].
The method for using multiple FFD volumes has been used to optimize a conventional aircraft [@Yildirim2021b], a T-shaped hydrofoil [@Liao2022], and a supersonic transport aircraft [@Seraj2022a].

pyGeo is maintained and developed by the MDO Lab[^2] at the University of Michigan and is actively used for MDO applications in both research and industry.
The geometry parameterization capabilities provided by pyGeo have facilitated the development of environmentally sustainable aircraft through design optimization.

[^2]: \url{https://mdolab.engin.umich.edu}


# Acknowledgements
We are grateful to the numerous pyGeo users who have contributed their time to the code and its maintenance over the years.

# References
