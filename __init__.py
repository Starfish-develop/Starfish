__all__ = ["model", "grid_tools", "MCMC"]

import os
import yaml
#import grid_tools

#Presumably config directory should be a string, hardcoded here

#Could also hardcode all of the constants used throughout?

fn = os.path.join(os.path.dirname(__file__), 'config.yaml')
print(fn)

f = open(fn)
default_config = yaml.load(f)
f.close()

#Set data directories


#Set library directories