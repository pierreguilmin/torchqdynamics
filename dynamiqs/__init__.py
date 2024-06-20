from importlib.metadata import version

from . import dark
from .options import *
from .plots import *
from .qarray import *
from .result import *
from .solvers import *
from .sparse_qarray import *
from .time_array import *
from .utils import *

# get version from pyproject.toml
__version__ = version(__package__)

# set default matmul precision to 'highest'
set_matmul_precision('highest')
