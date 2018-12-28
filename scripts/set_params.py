#!/usr/bin/env python

import argparse

import Starfish.plot_utils

parser = argparse.ArgumentParser(description="Use the last runs to set the nuisance Chebyshev and covariance parameters.")
parser.add_argument("rundir", help="The relative path to the output directory containing the samples.")
args = parser.parse_args()

from Starfish import config
from Starfish.model import PhiParam
from Starfish import utils

# Determine all of the orders we will be fitting
spectra = Starfish.data["files"]
orders = Starfish.data["orders"]


for spectrum_id in range(len(spectra)):
    for order in orders:

        npoly = config["cheb_degree"]

        if order == orders[-1]:
            # Use cheb degree - 1 for the last order
            npoly -= 1

        fname_phi = Starfish.specfmt.format(spectrum_id, order) + "phi.json"
        phi = PhiParam.load(fname_phi)

        fname_mc = args.rundir + "/" + Starfish.specfmt.format(spectrum_id, order) + "/mc.hdf5"
        flatchain = Starfish.plot_utils.h5read(fname_mc)

        pars = flatchain[-1,:]

        phi.cheb = pars[:npoly]
        phi.sigAmp = float(pars[npoly])
        phi.logAmp = float(pars[npoly + 1])
        phi.l = float(pars[npoly + 2])

        phi.save()
