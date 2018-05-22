__version__ = '0.1'
__all__ = ["spectrum", "model", "grid_tools", "constants", "covariance", "utils", "emulator", "samplers"]

# Read the users config.yaml file.
# If it doesn't exist, print a useful help message

import yaml

try:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
except FileNotFoundError as e:
    default = __file__[:-11]+"config.yaml"
    import warnings
    warnings.warn("Using the default config.yaml file located at {0}. This is likely NOT what you want. Please create a similar 'config.yaml' file in your current working directory.".format(default), UserWarning)
    with open(default) as f:
         config = yaml.safe_load(f)

# Format string for saving/reading orders
specfmt = "s{}_o{}"

# Read the YAML variables into package-level dictionaries to be used by the other programs.
name = config["name"]
outdir = config["outdir"]

grid = config["grid"]
parname = grid["parname"]

# PCA = config["PCA"]
PCA = config.get("PCA", None)

data = config["data"]
instruments = data["instruments"]
