# pyGeo
[![Build Status](https://dev.azure.com/mdolab/Public/_apis/build/status/mdolab.pygeo?branchName=main)](https://dev.azure.com/mdolab/Public/_build/latest?definitionId=17&branchName=main)
[![Documentation Status](https://readthedocs.com/projects/mdolab-pygeo/badge/?version=latest)](https://mdolab-pygeo.readthedocs-hosted.com/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mdolab/pygeo/branch/main/graph/badge.svg?token=N2L58WGCDI)](https://codecov.io/gh/mdolab/pygeo)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05319/status.svg)](https://doi.org/10.21105/joss.05319)

pyGeo is an object oriented geometry manipulation framework for multidisciplinary design optimization (MDO).
It can be used for MDO within the [MACH framework](https://github.com/mdolab/MACH-Aero) and within [OpenMDAO](https://github.com/OpenMDAO/OpenMDAO) through [MPhys](https://github.com/OpenMDAO/mphys).
Its parameterization options include a free form deformation (FFD) implementation, an interface to NASA's [OpenVSP](https://openvsp.org/) parametric geometry tool, and an interface to the CAD package [ESP](https://acdl.mit.edu/ESP/).
It also includes geometric constraints and utility functions for geometry manipulation.

![](doc/images/DPW4_FFD-27745.gif)

## Documentation
Please see the [documentation](https://mdolab-pygeo.readthedocs-hosted.com/en/latest/) for installation details and API documentation.

To locally build the documentation, enter the `doc` folder and enter `make html` in terminal.
You can then view the built documentation in the `_build` folder.


## Citation
Please cite pyGeo in any publication for which you find it useful.
Citation information can be found [here](https://mdolab-pygeo.readthedocs-hosted.com/en/latest/citation.html).


## License
pyGeo is licensed under the Apache License, Version 2.0 (the "License"). See `LICENSE` for the full license.

## Copyright
Copyright (c) 2012 University of Toronto
Copyright (c) 2014 University of Michigan
Additional copyright (c) 2014 Gaetan K. W. Kenway, Charles A. Mader, and Joaquim R. R. A. Martins
All rights reserved.
