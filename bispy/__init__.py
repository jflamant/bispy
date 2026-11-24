from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bispy-polar")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

from . import filters, qfft, signals, spectral, timefrequency, utils

__all__ = ["filters", "qfft", "signals", "spectral", "timefrequency", "utils", "__version__"]
