# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
try:
    __STARFISH_SETUP__
except NameError:
    __STARFISH_SETUP__ = False

__version__ = '0.3'

if not __STARFISH_SETUP__:
    import os
    import warnings
    import logging

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

    level = "INFO"
    log_file = "starfish.log"
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    if "logging" in config:
        level = config["logging"].get("level", level)
        log_file = config["logging"].get("file", log_file)
        log_format = config["logging"].get("format", log_format)

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )

    logging.debug("Initialized logger")

    __all__ = ["spectrum", "model", "grid_tools", "constants", "covariance", "utils", "emulator", "samplers", "config"]
