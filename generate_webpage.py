import jinja2
import yaml
import sys
import glob
import os
import subprocess
import numpy as np


'''Takes the output of a run and generates a viewable webpage. To be called by MCMC.py at the end of a run.'''

if len(sys.argv) > 1:
    confname= sys.argv[1]
else:
    confname = 'config.yaml'
f = open(confname)
config = yaml.load(f)
f.close()

templateLoader = jinja2.FileSystemLoader(searchpath="templates")
# An environment provides the data necessary to read and
# parse our templates.  We pass in the loader object here.
templateEnv = jinja2.Environment(loader=templateLoader)
template = templateEnv.get_template('run_output.jinja')

#location of where flatchain.npy is
base_dir = 'output/' + config['name'] + '/'

hist_param = 'hist_param.png'
nuisance_images = [os.path.relpath(i,base_dir) for i in glob.glob(base_dir + 'nuisance/*.png')]
marginal_images = [os.path.relpath(i,base_dir) for i in glob.glob(base_dir + 'marginals/*.png')]
sample_names = [os.path.relpath(i,base_dir + 'visualize/') for i in glob.glob(base_dir + 'visualize/sample*')]

sample_dict = {}
for i in sample_names:
    lnp = np.load(base_dir + 'visualize/' + i + '/lnp.npy')
    p = np.load(base_dir + 'visualize/' + i + '/p.npy')
    # create a list of order images in each sample_dir
    images = [os.path.relpath(i,base_dir) for i in glob.glob(base_dir + 'visualize/' + i + '/*.png')]
    sample_dict[i] = {"lnp":lnp, "p":p, "images":images}

# Specify any input variables to the template as a dictionary.
templateVars = { 'title': config['name'], "hist_param" : hist_param, 'config' : config,
                 'nuisance_images':nuisance_images, 'marginal_images':marginal_images, 'sample_dict':sample_dict}


# Finally, process the template to produce our final text.
outputText = template.render(templateVars)
f = open(base_dir + 'index.html', 'w')
f.write(outputText)
f.close()

import generate_web_index

#run rsync to update the web output, don't copy the chain files
subprocess.call('rsync -avr --exclude "*.npy" output/ ../web', shell=True)
