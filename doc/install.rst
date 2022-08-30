.. _install:

============
Installation
============

Installation requires a working copy of the ``pyspline`` package, which requires a Fortran compiler and is available on `GitHub <https://github.com/mdolab/pyspline/>`_.
Because of this dependency, pyGeo is only supported on Linux. 
On other platforms, a Docker image that includes pyGeo and other MDO Lab codes can be used following `these instructions <https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/installInstructions/dockerInstructions.html#initialize-docker-container>`_.

To install ``pyGeo``, first clone the `repo <https://github.com/mdolab/pygeo/>`_, then go into the root directory and type::

   pip install .

The tests require unique dependencies ``numpy-stl`` and ``parameterized``.
These and additional, generic, testing dependencies can be installed by using::
    
    pip install .[testing]

For stability we recommend cloning or checking out a tagged release.

-------------
pyGeo and ESP
-------------
The simplest way to install ESP so that it works with pyGeo is to use the Docker image mentioned above, which will have pyGeo and ESP both installed. 
The ESP GUI can then be installed on your local machine for Mac, Windows, and Linux following the instructions in their README to view ESP models.
Our currently supported version is 1.20, which can be installed from the `archive page <https://acdl.mit.edu/ESP/archive/>`_. 
