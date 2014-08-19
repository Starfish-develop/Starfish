#!/usr/bin/env python

'''
Given a run output directory, and nothing else, this script attempts to create a blog post and place it in the Jekyll
_posts category so that it can be rendered and appear on the website.

Assumes that the output of the run will always have stellar parameters and at least one order, containing at least
Chebyshev polynomials.

It is optional that it may contain more things, like covariance kernels and region kernels, and this script should
attempt to find these things in a error-proof manner.


'''

#Give the run directory as a command-line argument
import argparse
parser = argparse.ArgumentParser(prog="generate_post.py", description="Generate a Jekyll blog post to summarize a run output.")
parser.add_argument("run", help="path to the run. Ex: output/WASP14/PHOENIX/22/run00/")
args = parser.parse_args()

import jinja2
import yaml
import os
import shutil
from datetime import date

templateLoader = jinja2.FileSystemLoader(searchpath="templates")
templateEnv = jinja2.Environment(loader=templateLoader)
template = templateEnv.get_template('post.jinja')

#Scan the output directory to make sure we aren't overwriting anything. 
#Increment the post number until we have a free index
datestr = date.isoformat(date.today())
outdir = 'web/_posts/'
num = 0
base = outdir + datestr + "-{:0>2}.md"
while os.path.exists(base.format(num)) and (num < 20):
    print(base.format(num), "exists")
    num += 1

numstr = "{:0>2}".format(num)
outfile = base.format(num)
print("Using", outfile)


#Make a subdirectory in the assets folder
path_list = args.run.split("/")
asset_path = "web/assets/" + "/".join(path_list[1:])

#Clean out any files that may be there under the same name
if os.path.exists(asset_path):
    shutil.rmtree(asset_path)
#Copy every image we have over to the assets folder
shutil.copytree(args.run, asset_path,  ignore=shutil.ignore_patterns("*.json", "*.yaml", "*.hdf5"))


jekyll_asset_path = "{{ site.url }}/assets/" + "/".join(path_list[1:]) + "/"


#Open the yaml file in this directory
yaml_file = args.run + "/input.yaml"
f = open(yaml_file)
config = yaml.load(f)
f.close()

#Use the model_final.json to figure out how many orders there are
from StellarSpectra.model import Model
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface

#Figure out what the relative path is to base
import StellarSpectra
base = StellarSpectra.__file__[:-26]

myDataSpectrum = DataSpectrum.open(base + config['data'], orders=config['orders'])
myInstrument = TRES()
myHDF5Interface = HDF5Interface(base + config['HDF5_path'])

myModel = Model.from_json(args.run + "/model_final.json", myDataSpectrum, myInstrument, myHDF5Interface)
orders = [orderModel.order for orderModel in myModel.OrderModels]

flot_plots = {22:"Hi"}

#If the Jinja templater is going to work, it needs a list of orders. It also needs a list of how many regions
# are in each order
# each order, there is dictionary
#of global

#Set the categories as the decomposition of the run directory, excluding 
#output and the "run00" directory. 
#For example, output/WASP14/Kurucz/22/run01 becomes categories="WASP14 Kurucz 22"
categories = " ".join(args.run.split("/")[1:-1])


templateVars = {'num':numstr, 'date':datestr, 'categories':categories, 'jekyll_asset_path':jekyll_asset_path,
                'orders': orders, 'flot_plots':flot_plots}

outputText = template.render(templateVars)

f = open(outfile, 'w')
f.write(outputText)
f.close()

