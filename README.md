# pyGeo
[![Build Status](https://dev.azure.com/mdolab/Public/_apis/build/status/mdolab.pygeo?branchName=main)](https://dev.azure.com/mdolab/Public/_build/latest?definitionId=17&branchName=main)
[![Documentation Status](https://readthedocs.com/projects/mdolab-pygeo/badge/?version=latest)](https://mdolab-pygeo.readthedocs-hosted.com/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mdolab/pygeo/branch/main/graph/badge.svg?token=N2L58WGCDI)](https://codecov.io/gh/mdolab/pygeo)

pyGeo is an object oriented geometry manipulation framework for multidisciplinary design optimization.
It provides a free form deformation (FFD) based geometry manipulation object, an interface to NASA's Vehicle Sketch Pad geometry engine, a simple geometric constraint formulation object, and some utility functions for geometry manipulation.

![](doc/images/DPW4_FFD-27745.gif)

## Documentation

Please see the [documentation](https://mdolab-pygeo.readthedocs-hosted.com/en/latest/) for installation details and API documentation.
This link requires credentials currently only available to MDO Lab members.

To locally build the documentation, enter the `doc` folder and enter `make html` in terminal.
You can then view the built documentation in the `_build` folder.

## Citation

Please cite pyGeo in any publication for which you find it useful.
For more background, theory, and figures, see the [this paper](http://umich.edu/~mdolaboratory/pdf/Kenway2010b.pdf).

G. K. W. Kenway, Kennedy, G. J., and Martins, J. R. R. A., “A CAD-Free Approach to High-Fidelity Aerostructural Optimization”, in Proceedings of the 13th AIAA/ISSMO Multidisciplinary Analysis Optimization Conference, Fort Worth, TX, 2010.
```
@conference {Kenway:2010:C,
	title = {A {CAD}-Free Approach to High-Fidelity Aerostructural Optimization},
	booktitle = {Proceedings of the 13th AIAA/ISSMO Multidisciplinary Analysis Optimization Conference},
	year = {2010},
	note = {AIAA 2010-9231},
	month = {September},
	address = {Fort Worth,~TX},
	author = {Gaetan K. W. Kenway and Graeme J. Kennedy and Joaquim R. R. A. Martins}
}
```

## License

Copyright 2019 MDO Lab. See the LICENSE file for details.
