# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
try:
    __STARFISH_SETUP__
except NameError:
    __STARFISH_SETUP__ = False

__version__ = '0.2.3'

if not __STARFISH_SETUP__:
    import os
    import warnings

    starfish_dir = os.path.dirname(__file__)
    DEFAULT_CONFIG_FILE = os.path.join(starfish_dir, "config.yaml")

    from ._config import Config

    if os.path.exists("config.yaml") and os.path.abspath("config.yaml") != DEFAULT_CONFIG_FILE:
        config = Config("config.yaml")
    else:
        warnings.warn(
            "Using the default config file located at {}. This is likely NOT what you want and "
            "you will not be able to change any of the config values. Please use config.copy_file(<path>) to copy a "
            "version of the default config for your own project.".format(DEFAULT_CONFIG_FILE),
            UserWarning)
        config = Config(DEFAULT_CONFIG_FILE)

    __all__ = ['config', 'constants', 'covariance', 'emulator', 'grid_tools', 'model', 'samplers', 'spectrum', 'utils']

