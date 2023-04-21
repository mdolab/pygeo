# Standard Python modules
import os
import sys

# External modules
from sphinx_mdolab_theme.config import *

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("./_exts"))

# -- Project information -----------------------------------------------------
project = "pyGeo"

# -- General configuration -----------------------------------------------------
# Built-in Sphinx extensions are already contained in the imported variable
# here we add external extensions, which must also be added to requirements.txt
# so that RTD can import and use them
extensions.extend(
    [
        "sphinx_mdolab_theme.ext.embed_code",
        "sphinxcontrib.bibtex",
    ]
)

# mock import for autodoc
autodoc_mock_imports = [
    "numpy",
    "mpi4py",
    "scipy",
    "pyspline",
    "baseclasses",
    "pysurf",
    "prefoil",
    "pyOCSM",
    "openvsp",
    "openmdao",
]

# This sets the bibtex bibliography file(s) to reference in the documentation
bibtex_bibfiles = ["ref.bib"]
