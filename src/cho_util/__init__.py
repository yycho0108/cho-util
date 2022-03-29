from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("cho_util")
except PackageNotFoundError:
    # package is not installed
    pass

from . import cam
from . import math
from . import viz
from . import app
