.. _install:

Installation
============

Installation requires a working copy of the ``pyspline`` package, which requires a Fortran compiler and is available on `GitHub <https://github.com/mdolab/pyspline/>`_.

To install ``pyGeo``, first clone the `repo <https://github.com/mdolab/pygeo/>`_, then go into the root directory and type::

   pip install .

The tests require unique dependencies ``numpy-stl`` and ``parameterized``.
These and additional, generic, testing dependencies can be installed by using::
    
    pip install .[testing]

For stability we recommend cloning or checking out a tagged release.
