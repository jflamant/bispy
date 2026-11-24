from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bispy-polar")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

from . import utils
from . import qfft
from . import timefrequency
from . import spectral
from . import filters
from . import signals
