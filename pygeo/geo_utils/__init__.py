# =============================================================================
# Utility Functions for Use in pyNetwork, pyGeo, pyBlock, DVGeometry,
# and pyLayout
# =============================================================================


import sys

# This __init__ file imports every methods in pygeo/geo_utils
from .bilinear_map import *  # noqa: F401, F403
from .dcel import *  # noqa: F401, F403
from .ffd_generation import *  # noqa: F401, F403
from .file_io import *  # noqa: F401, F403
from .index_position import *  # noqa: F401, F403
from .knotvector import *  # noqa: F401, F403
from .mesh_generation import *  # noqa: F401, F403
from .misc import *  # noqa: F401, F403
from .node_edge_face import *  # noqa: F401, F403
from .norm import *  # noqa: F401, F403
from .orientation import *  # noqa: F401, F403
from .pointselect import *  # noqa: F401, F403
from .polygon import *  # noqa: F401, F403
from .projection import *  # noqa: F401, F403
from .remove_duplicates import *  # noqa: F401, F403
from .rotation import *  # noqa: F401, F403
from .split_quad import *  # noqa: F401, F403

# Set a (MUCH) larger recursion limit. For meshes with extremely large
# numbers of blocs (> 5000) the recursive edge propagation may hit a
# recursion limit.
sys.setrecursionlimit(10000)
