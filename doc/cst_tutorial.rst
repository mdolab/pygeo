.. _csttutorial:

====================
CST Airfoil Geometry
====================

pyGeo includes a Class-Shape Transformation\ :footcite:p:`CST` (CST) implementation for airfoil parameterization called DVGeometryCST.
In addition to CST coefficients, it allows the user to add design variables for the chord length and class shape parameters.

Setting up a CST parameterization requires only an airfoil dat file, so initialization is straightforward.
After the DVGeometryCST object is initialized with the airfoil dat file, we add design variables.
Next, we add pointsets that will be deformed when the airfoil shape is updated.
Finally, we can change the variables to update the airfoil shape and associated pointsets.

Note that while the CST parameterization is easy to set up and use, it may not be the best choice for all applications.
As we increase the number of CST coefficients that parameterize the airfoil shape, the parameterization becomes poorly conditioned.
This may result in unexpected optimizer behavior, such as adjacent CST design variables ending up at very large equal and opposite values.

This parameterization makes the following assumptions:

- The initial airfoil dat file is ordered continuously around the airfoil and the beginning and end of the list is the trailing edge (no jumping around, but clockwise vs. counterclockwise does not matter)
- The pointset geometry is exclusively an extruded shape (no spanwise changes allowed)
- Tirfoil's leading edge is on the left (minimum :math:`x`) and trailing edge is on the right (maximum :math:`x`)
- The airfoil is not rotated (trailing edge and leading edge are close to :math:`y = 0`)

.. footbibliography::
