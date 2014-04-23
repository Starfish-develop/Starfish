#!/usr/bin/env python
from StellarSpectra.model import Model
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
import StellarSpectra.constants as C
import numpy as np
import yaml
import json

import bokeh
from bokeh.plotting import *
from bokeh.objects import Range1d

#Use argparse to determine if we've specified a config file
import argparse
parser = argparse.ArgumentParser(prog="plot_model.py", description="Plot the model.")
parser.add_argument("json", help="*.json file describing the model.")
parser.add_argument("params", help="*.yaml file specifying run parameters.")
parser.add_argument("-o", "--output", help="*.html file for output")
args = parser.parse_args()


if args.json: #
    #assert that we actually specified a *.json file
    if ".json" not in args.json:
        import sys
        sys.exit("Must specify a *.json file.")

if args.params: #
    #assert that we actually specified a *.yaml file
    if ".yaml" in args.params:
        yaml_file = args.params
        f = open(args.params)
        config = yaml.load(f)
        f.close()

    else:
        import sys
        sys.exit("Must specify a *.yaml file.")
        yaml_file = args.params


myDataSpectrum = DataSpectrum.open(config['data_dir'], orders=config['orders'])
myInstrument = TRES()
myHDF5Interface = HDF5Interface(config['HDF5_path'])

stellar_Starting = config['stellar_params']
stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)

cheb_Starting = config['cheb_params']
cheb_tuple = ("logc0", "c1", "c2", "c3")

cov_Starting = config['cov_params']
cov_tuple = C.dictkeys_to_cov_global_tuple(cov_Starting)

region_tuple = ("h", "loga", "mu", "sigma")
region_MH_cov = np.array([0.05, 0.04, 0.02, 0.02])**2 * np.identity(len(region_tuple))


myModel = Model.from_json(args.json, myDataSpectrum, myInstrument, myHDF5Interface)

model = myModel.OrderModels[0]

#Get the data
wl, fl = model.get_data()

#Get the model flux
flm = model.get_spectrum()

#Get residuals
residuals = model.get_residuals()

#Get the Chebyshev spectrum
cheb = model.get_Cheb()

#Get the covariance matrix
S = model.get_Cov()

filename = args.output if args.output else "image.html"

output_file(filename, title="plot_model.py WASP14")

min_x, max_x = np.min(wl), np.max(wl)
min_y, max_y = np.min(wl), np.max(wl)
xy_range = Range1d(start=min_x, end=max_x)

hold()

figure(title="WASP-14",
       tools="pan,wheel_zoom,box_zoom,reset,previewsave,select",
       plot_width=800, plot_height=300)

plot0 = line(wl, fl, line_width=1.5, legend="WASP14", x_range=xy_range, color="blue")
plot1 = line(wl, flm, line_width=1.5, legend="model", x_range=xy_range, color="red")

figure(title="Residuals",
       tools="pan,wheel_zoom,box_zoom,reset,previewsave,select",
       plot_width=800, plot_height=300)

plot0 = line(wl, residuals, line_width=1.5, legend="residuals", x_range=xy_range)

figure(title="Covariance",
       tools="pan,wheel_zoom,box_zoom,reset,previewsave,select",
       plot_width=800, plot_height=800)

img = np.log10(S.todense() + 1e-29)

image(image=[img],
      x=[min_x],
      y=[min_y],
      dw=[max_x-min_x],
      dh=[max_y-min_y],
      palette=["Spectral-11"],
      x_range = xy_range,
      y_range = xy_range,
      title="Covariance",
      tools="pan,wheel_zoom,box_zoom,reset,previewsave",
      plot_width=800,
      plot_height=800
)

figure(title="Chebyshev",
       tools="pan,wheel_zoom,box_zoom,reset,previewsave,select",
       plot_width=800, plot_height=300)

plot0 = line(wl, cheb, line_width=1.5, legend="Chebyshev", x_range=xy_range)

show()
