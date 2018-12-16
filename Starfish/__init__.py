# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
try:
    __STARFISH_SETUP__
except NameError:
    __STARFISH_SETUP__ = False

__version__ = '0.2'

if not __STARFISH_SETUP__:
    import os
    import warnings

    from ._config import Config

    base_dir = os.path.dirname(os.path.dirname(__file__))
    default_config_file = os.path.join(base_dir, "config.yaml")

    if os.path.exists("config.yaml") and os.path.abspath("config.yaml") != default_config_file:
        config = Config("config.yaml")
    else:
        warnings.warn("Using the default config file located at {}. This is likely NOT what you want. Please "
                      "create a similar 'config.yaml' file in your current working directory.".format(
            default_config_file), UserWarning)
        config = Config(default_config_file)

    __all__ = ["spectrum", "model", "grid_tools", "constants", "covariance", "utils", "emulator", "samplers", "config"]
