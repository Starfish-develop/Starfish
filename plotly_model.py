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
import plotly.plotly as py
from plotly.graph_objs import *

def plotly_order(name, wl, fl, flm, residuals):

    dspec = Scatter( #data spectrum
                     x=wl,
                     y=fl,
                     name="Data",
                     marker=Marker(
                         color="blue",
                         ),
                     )
    mspec = Scatter( #model spectrum
                     x=wl,
                     y=flm,
                     name="Model",
                     marker=Marker(
                         color="red",
                         ),
                     )
    rspec = Scatter( #residual spectrum
                     x=wl,
                     y=residuals,
                     name="Residuals",
                     yaxis='y2',
                     marker=Marker(
                         color="black",
                         ),

                     )
    data = Data([dspec, mspec, rspec])
    layout = Layout(
        title=name,
        xaxis=XAxis(
            title="Wavelength (AA)",
        ),
        yaxis=YAxis(
            domain=[0.65, 1],
            exponentformat='power',
            showexponent="All",
            title="Flux (flam)"
        ),
        yaxis2=YAxis(
            domain=[0.1, 0.45],
            exponentformat='power',
            showexponent="All",
            ),
        )
    fig = Figure(data=data, layout=layout)

    plot_url = py.plot(fig, filename=name)

from StellarSpectra.model import Model
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface

#Figure out what the relative path is to base
import StellarSpectra
base = StellarSpectra.__file__[:-26]

myDataSpectrum = DataSpectrum.open(base + config['data'], orders=config['orders'])
myInstrument = TRES()
myHDF5Interface = HDF5Interface(base + config['HDF5_path'])

myModel = Model.from_json(args.json, myDataSpectrum, myInstrument, myHDF5Interface)

for model in myModel.OrderModels:

    #Get the data
    wl, fl = model.get_data()

    #Get the model flux
    flm = model.get_spectrum()

    #Get residuals
    residuals = model.get_residuals()

    name = "Order {}".format(model.order)

    plotly_order(name, wl, fl, flm, residuals)

    #Get the Chebyshev spectrum
    # cheb = model.get_Cheb()

    #Get the covariance matrix
    # S = model.get_Cov()

