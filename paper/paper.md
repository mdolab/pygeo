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
In the field of aerodynamic shape optimization, an object's geometry is modified by an optimization algorithm to improve its performance.
A common shape optimization example is adjusting the external shape of an aircraft wing to minimize the aerodynamic drag computed via computational fluid dynamics (CFD).
In a multidisciplinary design optimization (MDO) context, aerodynamics and structural mechanics are considered and optimized simultaneously, which often provides additional benefit over optimizing only a single discipline.
In such cases, the geometry must be represented consistently across multiple disciplines.

pyGeo is a geometry package for three-dimensional shape manipulation, tailored for aerodynamic and multidisciplinary design optimization.
It provides some basic geometry generation capabilities, several methods for geometry parameterization, numerous geometric constraints, and some utility functions for geometry manipulation.
The code provides derivatives for all parameterization methods and constraint functions, enabling their use in gradient-based optimization.
<!--MM: I am team parameTRIzation-->

# Features

## Integrations

pyGeo was originally developed to implement and use free-form deformation (FFD) to manipulate 3D geometries in CFD-based optimization [@Kenway2010b].
More details on this implementation as well as recent extensions are summarized in the section "Free-form Deformation".

pyGeo is the geometry manipulation engine within the MDO of Aircraft Configurations at High Fidelity (MACH) framework [@Kenway2014a], [@Kenway2014c], which specializes in high-fidelity aerostructural optimization.
pyGeo, together with the other core MACH modules, are integrated in MPhys[^1], a more general-use MDO tool based in OpenMDAO [@Gray2019a].

[^1]: https://github.com/OpenMDAO/mphys


Both frameworks mentioned above use pyOptSparse [@Wu2020a] to interface with the optimization algorithm.
pyGeo sends design variables and constraints to pyOptSparse directly, reducing user effort.

pyGeo's interface for both design variables and constraints is independent of which discipline solvers are accessing the geometry.
This means that pyGeo geometries can interact with different types of solvers, such as structures and aerodynamics, in the same way.
This also allows for a direct comparison of the behavior or performance of two different solvers within the same discipline by using the same geometric parameterization for each, such as two different flow solvers [@Adler2022c].

## Geometry Generation

pyGeo can create simple geometries in the IGES file format, a common type readable by computer-aided design (CAD) tools.
One method of geometry generation in pyGeo is generating a surface by lofting between user-specified cross-sections, which is most commonly used to create a wing from airfoil cross-sections (\autoref{fig:geo_gen}).
In particular, rounded or pinched wingtip geometries can be generated easily.
These features rely on the open-source package pySpline[^2], which handles the underlying B-spline implementation.

![Example of a wing generated with pyGeo and the airfoil used for its cross-sections.\label{fig:geo_gen}](geo_gen.pdf)

[^2]: https://github.com/mdolab/pyspline

## Geometry Parameterization with pyGeo

pyGeo contains several options for parameterizing geometry: variations on the FFD method, interfaces to external parametric modeling tools, and an analytic parameterization.
Because each of these use a common interface for interacting with the rest of the MACH framework, any parameterization can be used in place of another within an optimization setup.
Which parameterization is used depends on the experience of the user, the details of the geometry involved, and whether the user needs the final design in a specific format.

<!-- MM: Should we include a list of modules/classes here for easier reference later on?-->

### Free-form Deformation
<!--
TODO:
- talk less about FFDs and more about capabilities
- ref axis and complex geometric operations for a wing
- [x] child FFD
- [x] multi FFD
- [x] redo planform DV picture to be less pixelated
- [x] ESP pic
- [x] VSP pic
- [] other pic?
-->
The free-form deformation (FFD) method [@Sederberg1986] is one of the most popular three-dimensional geometry parameterization approaches.
In this approach, the entire reference geometry is embedded in a parameterized volume. 
The set of control points that determine the shape of the volume can be displaced to change the location of the points inside. 
<!--JLA: The control points do not have to be (and often are not) on the surface of the embedding volume -->
A high degree of geometry control can be realized by the user by selecting different control point densities and locations.

Individual control points can be moved to obtain local shape modifications.
In pyGeo, these are referred to as local design variables because a single control point is affected.
It is also common to define geometric operations involving a collection of control points.
In pyGeo, these are referred to as global design variables because control points across the entire FFD block can be affected.
For example, twist variables can be defined as rotations of the control points about a reference axis which runs along the wing. <!-- MM: this is a good example but I feel we need to specify what twist is to a non-aerospace audience-->

\autoref{fig:FFD_DV} shows a few common planform design variables for an aircraft wing.
Parameterizations based on the singular value decomposition are also possible [@Wu2022b].
<!-- talk about ref axis more? -->

<!-- Compared to other parameterizations, the FFD method has several key advantages.
Since the entire geometry is embedded, there is no need to start with or reverse-engineer a parametric geometry representation as commonly done with B-spline-based methods, where a least-squares fit is needed to generate a B-spline surface representation.
Rather than parameterizing the geometry directly, the geometric _deformation_ is parameterized instead.
This decoupling of geometry definition from geometric deformation allows for control and refinement of the deformation independently of the original geometry.
When working with multiple geometries, for example an optimization involving an aerodynamic and structural surface simultaneously, both surfaces can be embedded into the same FFD volume.
As both surfaces would be manipulated by the same volume, coincident surfaces remain coincident after deformations and this approach ensures consistency between disparate geometries. -->

![Examples of common planform design variables.\label{fig:FFD_DV}](ffd_dvs.pdf)

In addition to the basic FFD implementation, pyGeo offers two additional features: hierarchical FFD and multi-component FFD.

#### Hierarchical FFD
<!-- HMH: changing to child FFD - plural child FFDs -->
<!--MM: not sold on the subsection titles I made, pls provide input-->
<!--SS: Changed subsections from 'Child FFD' to 'Hierarchical FFD' and 'Multi-FFD' to 'Multi-component FFD'. Also using 'FFD blocks' instead of 'FFDs'. -->
FFD objects can be organized in a hierarchical structure within pyGeo.
Dependent, "child" FFD blocks can be embedded in the main, "parent" FFD block to enable modifications on a subset of the full geometry.
<!-- The user can define local and global variables on both objects independently.-->
<!--  HMH: do we explain the difference between local and global variables somewhere?-->
pyGeo will first propagate the parent deformations to both the geometry and the child control points, then propagate the deformations of the child control points to their subset of the geometry. <!--MM: I would like to double check this sentence with Anil-->
One of the advantages of using this approach is that each FFD block can have its own independent reference axis to be used for global design variables such as rotations and scaling.
This facilitates, for example, the definition of independent leading and trailing edge wing deformations [@Mangano2021a], wind turbine blade parametrization [@Mangano2022a],[@Madsen2019a], and hydrofoil design [@Liao2021a].
\autoref{fig:ffd_child} from the latter paper shows a case where the parent FFD block is used for scaling the chord of a hydrofoil using a reference axis at the trailing-edge, whereas the twist and sweep variables are defined on the child FFD block with its reference axis at the quarter-chord.

![Example of parametrization through parent-child FFD blocks [@Liao2021a] \label{fig:ffd_child}](ffd_child.png)

#### Multi-component FFD

The basic FFD implementation lacks flexibility when the geometry has intersecting components.
In such cases, pyGeo can parameterize each component using FFD and ensure a watertight surface representation at the component intersections using an inverse-distance surface deformation method [@Yildirim2021b].
This method relies on the open source pySurf package [@Secco2018b] to compute intersections between components, perform projections, and remesh curves.
\autoref{fig:ffd_multi} shows an example of a component-based FFD setup for a supersonic transport aircraft.

![Example of FFD parameterization with intersecting components [@Seraj2022a]. \label{fig:ffd_multi}](ffd_multi.png)

### Parametric Geometry Tools

The flexibility and ease of setup of the FFD method make it preferable for some applications.
In other applications, however, it can be beneficial to define the geometry in a more commonly accepted engineering format, such as a CAD model or other parametric definition.
For example, a CAD model is usually required to manufacture a design.

<!-- [X] TODO SS-HMH: If we are looking to cut text, this paragraph could be a candidate. I think one line making the point that FFD defines the deformation, whereas CAD directly defines the geometry would be sufficient. -->
<!-- [X] TODO SS-HMH: 'designed parametrically or 'defined parametrically' ? -->
If the geometry is defined parametrically, the relationships between design variables and geometry is defined in the model itself.
An FFD block only defines the deformation, while parametric geometry tools directly define the geometry.
<!-- In an FFD model of a box, for example, the FFD points could represent the four corners of the box, but then the user would be required to define the planes in which points move to change the length, width, and height of the box.
In a parametric modeling tool, the user would create a box by defining its initial length, width, and height.
For either case, the length, width, and height (or a subset) can be controlled in the optimization process as design variables. -->

<!-- [X] TODO SS-HMH: The transition here is a bit abrupt without any introduction to what ESP and OpenVSP are. -->
pyGeo contains interfaces to two parametric geometry tools, the Engineering Sketch Pad (ESP) [@Haimes2013a] and OpenVSP [@McDonald2022a].
ESP is a CAD software, while OpenVSP is a concpetual design tool.
The two packages have different capabilities, but in both design variables directly define the geometry and the created geometry can be used in external analysis tools.

<!-- [X] TODO SS-HMH: There is enough repeated text between the ESP and OpenVSP sections that we might want to describe both under 'Parametric Geometry Tools' rather than have separate subsections. -->
pyGeo interfaces with ESP and OpenVSP in similar ways.
In both cases, an instance of the model is read into pyGeo and its points are associated with coordinates in a mesh from a solver in the MACH framework.
The design variables built into the ESP or OpenVSP model by the designer are also read into pyGeo.
For both ESP and OpenVSP models, the pyGeo interface to the respective software stores the model in a form usable within the MACH framework and updates it as design variables are changed throughout the optimization.
The pyGeo interface to ESP was used by [@Brelje2021a] to parameterize hydrogen tanks (\autoref{fig:esp_example}) that were packaged within an aircraft wing as part of an aerostructural optimization.
pyGeo's OpenVSP interface was used to parameterize the full aircraft configuration (\autoref{fig:vsp_example}) studied in the aeropropulsive optimization work in [@Yildirim2022a].

![Example of ESP models of hydrogen tanks used through pyGeo from [@Brelje2021a]. \label{fig:esp_example}](esp_example.png)

![Example of a VSP model used through VSP's pyGeo interface from [@Yildirim2022a]. \label{fig:vsp_example}](vsp_example.png)

### Class Shape Transformation

The class shape transformation (CST) methodology [@Kulfan2008] is a popular airfoil parameterization.
It generates a shape by using Bernstein polynomials to scale a class function, which is most often a base airfoil shape.
The class function is modified with two parameters, and the number of Bernstein polynomials is adjustable.
pyGeo contains a module that implements this airfoil parameterization.
The implementation supports design variables for the Bernstein polynomial weightings, the class function parameters, and the airfoil chord length.
<!-- SS: We don't mention derivatives for each method, so I'm commenting the following line out. -->
<!-- It includes methods to analytically compute derivatives of the airfoil's surface coordinates with respect to the design variables, which is useful for gradient-based optimization. -->
pyGeo's CST implementation can only be used for 2D problems, such as airfoil optimization (\autoref{fig:cst_example}).

![Airfoil defined by 3 CST coefficients on each surface undergoing a perturbation in one Bernstein polynomial. \label{fig:cst_example}](cst_example.pdf)

## Constraints

pyGeo also includes geometric constraints.
<!--
Constraints are all differentiated in order to use within gradient-based optimization.
DVCon creates constraint objects which are passed to pyOptSparse.
-->
Some commonly used geometric constraints in shape optimization are thickness, area, and volume constraints.
Thickness constraints control the distance between two points.
<!-- [] TODO SS-: Almost all the constraints can be described by the line below. Should this section focus on why these constraints are useful or just describe them generally? -->
<!-- HMH: Neil suggested listing more of the constraints we use, I think we could also outline why they are useful but if we are short on words that could be tricky -->
Area and volume constraints constrain the geometry from deviating from the initial design by some relative or absolute measure.

<!-- list out more constraints -->
<!-- Triangulated surface constraint -->
<!-- [] TODO SS-: Add pictures of some of these constraints? -->
<!-- HMH: working on a figure for thickness and volume constrints -->

# Parallelism
pyGeo can optionally work under distributed memory parallelism with MPI, which can be helpful when interfacing with CFD applications.
For example, the computational mesh may be partitioned and distributed among many processors, and each processor may be aware of only its portion of the mesh.
pyGeo can handle such scenarios by independently manipulating the geometry on each processor and aggregating the constraints across all processors when communicating with the optimizer.

# Derivative Computation
In addition to geometry manipulation and constraints, pyGeo is able to compute derivatives of these operations with respect to design variables.
For the geometric deformation, pyGeo can compute the Jacobian
\begin{equation*}
\frac{\mathrm{d}X_s}{\mathrm{d}x}
\end{equation*}
where $X_s$ is the vector of surface mesh coordinates, and $x$ the vector of geometric design variables.

Similarly for the constraints, the Jacobian
\begin{equation*}
\frac{\mathrm{d}g}{\mathrm{d}x}
\end{equation*}
can be computed, where $g$ is the vector of geometric constraints.

For the FFD parameterization, these derivatives are computed using a combination of analytic methods and the complex-step method [@Martins2003a].
<!-- [] TODO SS-: Should we mention how derivatives for other methods are computed? -->
<!-- HMH: my thought is no because then we'd have to mention finite differences but I'd rather leave FFD out than have that be the only one mentioned -->

# Statement of Need
Very few open-source packages exist with comparable functionalities.
To the best knowledge of the authors, the only other optimization framework that contains geometry parameterization is SU2 [@Economon2016a].
It supports Hicks--Henne bump functions for airfoil optimizations and the FFD method for three-dimensional cases.
However, it cannot be used with other solvers because it is directly integrated into the CFD solver SU2.

It is worth noting here that both OpenVSP and ESP can be used directly in optimization without using pyGeo.
However, when used as stand-alone tools these parameterization methods lack capabilities needed for high-fidelity MDO.
pyGeo enables high-fidelity MDO with these tools through parallelism, efficient gradients, and geometric constraints, all while keeping the original tool in the optimization loop.
It provides an interface to OpenVSP and ESP that allows for their use with solvers beyond those which they are natively tied to.

pyGeo has been used extensively in aerodynamic and aerostructural optimizations within aerospace engineering and related fields.
<!-- [] TODO SS-: We should add a few citations for the basic FFD functionality. -->
<!-- HMH: Any ideas on which would be representative? Neil suggested uCRM, maybe we want a wind turbine and/or hydrofoil paper as well for ~range~  -->
Its different parametrization options have all been necessary for different optimization problems depending on the geometry involved.
The interface to ESP made it possible to parameterize hydrogen tanks within a combined aerostructural and packaging optimization [@Brelje2021a].
pyGeo's OpenVSP interface was used in the aeropropulsive optimization of a podded electric turbofan [@Yildirim2021c].
The implementation of CST airfoil parameterization was used to compare methods for airfoil optimization [@Adler2022c].
The method for using multiple FFD volumes has been used to optimize a conventional aircraft [@Yildirim2021b], a T-shaped hydrofoil [@Liao2022], and a supersonic transport aircraft [@Seraj2022a].

# Acknowledgements

We acknowledge the numerous pyGeo users who have contributed their time to the code and its maintenance over the years.

# References
