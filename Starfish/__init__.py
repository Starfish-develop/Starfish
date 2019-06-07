# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
__version__ = "0.3.0-dev"

from .spectrum import Spectrum

__all__ = [
    "constants",
    "emulator",
    "grid_tools",
    "models",
    "samplers",
    "spectrum",
    "Spectrum",
    "transforms",
    "utils",
]
