__version__ = '0.1'
# Read the users config.yaml file.
# If it doesn't exist, print a useful help message
import os
import warnings

import yaml

from .config import Config

if os.path.exists("config.yaml"):
    config = Config("config.yaml")
else:
    base_dir = os.path.dirname(__file__)
    default = os.path.join(base_dir, "config.yaml")
    warnings.warn("Using the default config.yaml file located at {0}. This is likely NOT what you want. Please create a similar 'config.yaml' file in your current working directory.".format(default), UserWarning)
    config = Config(default)



__all__ = ["spectrum", "model", "grid_tools", "constants", "covariance", "utils", "emulator", "samplers", "config"]