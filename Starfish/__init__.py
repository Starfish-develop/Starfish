__version__ = '1.0'
__all__ = ["spectrum", "model", "grid_tools", "constants", "covariance", "utils", "emulator", "em_cov"]

# Read the users config.yaml file. If it doesn't exist, print a useful help message

import yaml

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    print("Please create a 'config.yaml' file in your current working directory."\
    " An example file can be found in", __file__[:-11]+"config.yaml")
    raise(e)

# Read the YAML variables into package-level dictionaries to be used by the other programs.
grid = config["grid"]
parname = grid["parname"]

PCA = config["PCA"]

data = config["data"]
instruments = data["instruments"]
