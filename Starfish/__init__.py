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

    base_dir = os.path.dirname(os.path.dirname(__file__))
    DEFAULT_CONFIG_FILE = os.path.join(base_dir, "config.yaml")

    from ._config import Config

    if os.path.exists("config.yaml") and os.path.abspath("config.yaml") != DEFAULT_CONFIG_FILE:
        config = Config("config.yaml")
    else:
        warnings.warn(
            f"Using the default config file located at {DEFAULT_CONFIG_FILE}. This is likely NOT what you want and "
            f"you will not be able to change any of the config values. Please use config.copy_file(<path>) to copy a "
            f"version of the default config for your own project.",
            UserWarning)
        config = Config(DEFAULT_CONFIG_FILE)

    __all__ = ["spectrum", "model", "grid_tools", "constants", "covariance", "utils", "emulator", "samplers", "config"]
