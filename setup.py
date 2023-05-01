from setuptools import setup, find_packages
import re
import os

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("pygeo/__init__.py").read(),
)[0]

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pygeo",
    version=__version__,
    description="pyGeo is an object oriented geometry manipulation framework for multidisciplinary design optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="geometry FFD optimization",
    author="",
    author_email="",
    url="https://github.com/mdolab/pygeo",
    license="Apache License Version 2.0",
    packages=find_packages(include=["pygeo*"]),
    install_requires=["numpy>=1.16", "pyspline>=1.1", "scipy>=1.2", "mpi4py>=3.0", "mdolab-baseclasses", "packaging"],
    extras_require={
        "testing": ["numpy-stl", "parameterized", "testflo"],
        "mphys": ["openmdao"],
        "openvsp": ["openvsp>=3.28"],
    },
    classifiers=["Operating System :: OS Independent", "Programming Language :: Python"],
)
