#!/usr/bin/env python

#Use argparse to determine if we've specified a config file
import argparse
parser = argparse.ArgumentParser(prog="plotly_model.py", description="Plot the model and residuals using plot.ly")
parser.add_argument("json", help="*.json file describing the model.")
parser.add_argument("params", help="*.yaml file specifying run parameters.")
# parser.add_argument("-o", "--output", help="*.html file for output")
args = parser.parse_args()

import json
import yaml

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

import numpy as np
from StellarSpectra.model import Model
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
import StellarSpectra.constants as C

myDataSpectrum = DataSpectrum.open_npy(config['data_dir'], orders=config['orders'])
myInstrument = TRES()
myHDF5Interface = HDF5Interface(config['HDF5_path'])

# stellar_Starting = config['stellar_params']
# stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)
#
# cheb_Starting = config['cheb_params']
# cheb_tuple = ("logc0", "c1", "c2", "c3")
#
# cov_Starting = config['cov_params']
# cov_tuple = C.dictkeys_to_cov_global_tuple(cov_Starting)
#
# region_tuple = ("h", "loga", "mu", "sigma")
# region_MH_cov = np.array([0.05, 0.04, 0.02, 0.02])**2 * np.identity(len(region_tuple))

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
# S = model.get_Cov()

# filename = args.output if args.output else "image.html"


import plotly.plotly as py
from plotly.graph_objs import *

dspec = Scatter( #data spectrum
    x=wl,
    y=fl,
)
mspec = Scatter( #model spectrum
    x=wl,
    y=flm,
    xaxis="x2",
    yaxis="y2"
)
# rspec = Scatter( #residual spectrum
#     x=wl,
#     y=residuals,
#     xaxis="x3",
#     yaxis='y3'
# )
data = Data([dspec, mspec]) #, rspec])
layout = Layout(
    # xaxis=XAxis(
    #     title="Wavelength (AA)"
    # ),
    xaxis=XAxis(
    ),
    yaxis=YAxis(
    ),

    xaxis2=XAxis(
        anchor='y2'
    ),
    yaxis2=YAxis(
    ),

    # xaxis3=XAxis(
    #     anchor='y3'
    # ),

    # yaxis3=YAxis(
    # )
)
fig = Figure(data=data, layout=layout)

plot_url = py.plot(fig, filename='WASP 14')
