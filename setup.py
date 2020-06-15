from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('pygeo/__init__.py').read(),
)[0]

setup(name='pygeo',
      version=__version__,


      description="pyGeo is an object oriented geometry manipulation framework for multidisciplinary design optimization",
      long_description="""It provides a free form deformation (FFD) based geometry manipulation object, an interface to NASA's Vehicle Sketch Pad geometry engine, a simple geometric constraint formulation object, and some utility functions for geometry manipulation.


      ![](images/DPW4_FFD-27745.gif)

      Documentation
      -------------

      Please see the [documentation](http://mdolab.engin.umich.edu/docs/packages/pygeo/doc/index.html) for installation details and API documentation.
      This link requires credentials currently only available to MDO Lab members.

      To locally build the documentation, enter the `doc` folder and enter `make html` in terminal.
      You can then view the built documentation in the `_build` folder.

      Citation
      --------

      Please cite pyGeo in any publication for which you find it useful.
      For more background, theory, and figures, see the [this paper](http://mdolab.engin.umich.edu/sites/default/files/mao2010_final.pdf).

      G. K. W. Kenway, Kennedy, G. J., and Martins, J. R. R. A., “A CAD-Free Approach to High-Fidelity Aerostructural Optimization”, in Proceedings of the 13th AIAA/ISSMO Multidisciplinary Analysis Optimization Conference, Fort Worth, TX, 2010.

      
      @conference {Kenway:2010:C,
        title = {A {CAD}-Free Approach to High-Fidelity Aerostructural Optimization},
        booktitle = {Proceedings of the 13th AIAA/ISSMO Multidisciplinary Analysis Optimization Conference},
        year = {2010},
        note = {AIAA 2010-9231},
        month = {September},
        address = {Fort Worth,~TX},
        author = {Gaetan K. W. Kenway and Graeme J. Kennedy and Joaquim R. R. A. Martins}
      }
      
      """,
      long_description_content_type="text/markdown",
      keywords='geometry FFD optimization',
      author='',
      author_email='',
      url='https://github.com/mdolab/pygeo',
      license='Apache License Version 2.0',
      packages=[
          'pygeo',
      ],
      package_data={
          'pygeo': ['*.so']
      },
      install_requires=[
            'numpy>=1.16.4',
            'pyspline>=1.1.0',
            'scipy>=1.2.1'

      ],
      classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python"]
      )
