__all__ = ["model", "grid_tools", "MCMC", "constants"]

import os
import yaml

#Could also hardcode all of the constants used throughout?

fn = os.path.join(os.path.dirname(__file__), 'config.yaml')

f = open(fn)
default_config = yaml.load(f)
f.close()

#Set data directories


#Set library directories