# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
try:
    __STARFISH_SETUP__
except NameError:
    __STARFISH_SETUP__ = False

__version__ = "0.3.1-dev"

if not __STARFISH_SETUP__:

    from .spectrum import Spectrum
    from .emulator import Emulator

    __all__ = [
        "constants",
        "emulator",
        "Emulator",
        "grid_tools",
        "models",
        "samplers",
        "spectrum",
        "Spectrum",
        "utils",
    ]

