.. _fitting_tutorial:

==================================
Fitting a DVGeometry to a Geometry
==================================

A common task is to take a geometry produced by one parameterization and represent it with a different one.
For example, you might optimize a wing using an FFD and then want a CAD model of the result, or you might want to compare two parameterizations of the same shape.
pyGeo provides a least-squares fitting capability for exactly this: ``fitDVGeo`` finds the design variables of one ``DVGeometry`` object that make its geometry match the point set of another.

This tutorial demonstrates the capability by recovering an airfoil's NACA parameters.
We represent a NACA 2412 airfoil with a CST parameterization and then fit an ESP NACA model to it.
Because the ESP ``naca`` user-defined primitive is parameterized by the exact NACA 4-digit definition (camber, location of maximum camber, and thickness), the fit recovers those parameters.

The full script for this tutorial is located at ``examples/airfoil_fitting/airfoil_fitting.py``.

--------------------------
Setting up the reference
--------------------------

We first build the reference geometry that we want to match.
Here we use a :class:`.DVGeometryCST` object initialized from a NACA 2412 dat file, and we use the airfoil coordinates themselves as the point set.
In practice this reference geometry could come from anywhere, such as the result of an FFD-based optimization.

.. literalinclude:: ../examples/airfoil_fitting/airfoil_fitting.py
    :start-after: # rst ref (beg)
    :end-before: # rst ref (end)

--------------------------
Setting up the model to fit
--------------------------

Next we build the model whose design variables we want to solve for.
We use a :class:`.DVGeometryESP` object initialized from an ESP model that builds a NACA airfoil from three design parameters: camber, location of maximum camber, and thickness.

.. literalinclude:: ../examples/airfoil_fitting/airfoil_fitting.py
    :start-after: # rst fit setup (beg)
    :end-before: # rst fit setup (end)

--------------------------
Performing the fit
--------------------------

With both objects set up, the fit is a single call.
``fitDVGeo`` takes the reference ``DVGeometry`` object and the name of the point set to match.
It embeds the reference geometry's original points into the model being fit, then solves a least-squares problem for the design variables that best match the reference coordinates.
It returns the fitted design variables and the SciPy ``OptimizeResult`` from the optimization.

.. literalinclude:: ../examples/airfoil_fitting/airfoil_fitting.py
    :start-after: # rst fit (beg)
    :end-before: # rst fit (end)

The fit recovers the NACA 2412 parameters: a camber of 0.02, a location of maximum camber of 0.4, and a thickness of 0.12.

.. literalinclude:: ../examples/airfoil_fitting/airfoil_fitting.py
    :start-after: # rst results (beg)
    :end-before: # rst results (end)

--------------------------
Comparing the geometries
--------------------------

Finally, we compare the fitted ESP geometry with the target CST geometry.
``fitDVGeo`` embeds the reference's points into a new point set named ``"<ptName>_fit"``, which is the one the model was fit on.

.. literalinclude:: ../examples/airfoil_fitting/airfoil_fitting.py
    :start-after: # rst plot (beg)
    :end-before: # rst plot (end)

-------
Summary
-------

In this tutorial, you learned how to use pyGeo's least-squares fitting capability to recover the design variables of one parameterization that best match the geometry of another.
The same approach works between any pair of parameterizations.
A lower-level method, ``fitPointset``, is also available for fitting a point set to an arbitrary set of coordinates that need not come from another ``DVGeometry`` object.
