import jinja2
import yaml
import sys
import glob
import os
import numpy as np

__author__ = 'ian'

'''Takes the output of a run and generates a viewable webpage. To be called by MCMC.py, likely.'''


if len(sys.argv) > 1:
    confname= sys.argv[1]
else:
    confname = 'config.yaml'
f = open(confname)
config = yaml.load(f)
f.close()

# In this case, we will load templates off the filesystem.
# This means we must construct a FileSystemLoader object.
# The search path can be used to make finding templates by
#   relative paths much easier.
templateLoader = jinja2.FileSystemLoader(searchpath="_templates")
# An environment provides the data necessary to read and
#   parse our templates.  We pass in the loader object here.
templateEnv = jinja2.Environment(loader=templateLoader)
# Read the template file using the environment object.
# This also constructs our Template object.
template = templateEnv.get_template('run_output.jinja')

#location of where flatchain.npy is
base_dir = 'output/' + config['name'] + '/'

hist_param = 'hist_param.png'
nuisance_images = [os.path.relpath(i,base_dir) for i in glob.glob(base_dir + 'nuisance/*.png')]
sample_names = [os.path.relpath(i,base_dir + 'visualize/') for i in glob.glob(base_dir + 'visualize/sample*')]

#for each order_dir in each sample_dir, create a list of images
sample_dict = {}
for i in sample_names:
    sample_dict[i] = [os.path.relpath(i,base_dir) for i in glob.glob(base_dir + 'visualize/' + i + '/*.png')]

#for each order_dir in each sample_dir, create a list of param and probabilities
sample_dict_prob = {}
for i in sample_names:
    sample_dict_prob[i] = [np.load(i) for i in glob.glob(base_dir + 'visualize/' + i + '/*.npy')]

print(sample_dict_prob)

#Need to find the directory relative to the base_dir

# Specify any input variables to the template as a dictionary.
templateVars = { 'title': config['name'], "hist_param" : hist_param, 'config' : config, 'nuisance_images':nuisance_images,
                 'sample_dict':sample_dict}


# Finally, process the template to produce our final text.
outputText = template.render( templateVars )
f = open(base_dir + 'index.html','w')
f.write(outputText)
f.close()