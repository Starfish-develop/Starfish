#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(prog="splot.py", description="Plot JSON files.")
parser.add_argument("file", help="JSON file containing the data, model, and residual.")
parser.add_argument("--regions", action="store_true", help="Optionally plot any instantiated regions on top.")
parser.add_argument("--matplotlib", action="store_true", help="Plot the image using matplotlib")
parser.add_argument("--noise", action="store_true", help="Plot random draws using the phi parameters.")
args = parser.parse_args()

from Starfish import config
import json
import numpy as np

from Starfish.covariance import get_dense_C, make_k_func, make_k_func_region
from Starfish.model import ThetaParam, PhiParam
from Starfish.utils import random_draws, std_envelope

#Colorbrewer bands
s3 = '#fee6ce'
s2 = '#fdae6b'
s1 = '#e6550d'

f = open(args.file, "r")
read = json.load(f) # read is a dictionary
f.close()

wl = np.array(read["wl"])
data = np.array(read["data"])
model = np.array(read["model"])
sigma = np.array(read["sigma"])
resid = np.array(read["resid"])
spectrum_id = read["spectrum_id"]
order = read["order"]



if args.regions:
    fname = config.specfmt.format(spectrum_id, order) + "regions.json"
    f = open(fname, "r")
    read = json.load(f) # read is a dictionary
    f.close()

    mus = np.array(read["mus"])

if args.noise:

    sigma_mat =  sigma**2 * np.eye(len(sigma))

    fname = config.specfmt.format(spectrum_id, order) + "phi.json"
    phi = PhiParam.load(fname)

    if phi.regions is not None:
        region_func = make_k_func_region(phi)
        max_r = 4.0 * np.max(phi.regions, axis=0)[2]
        region_mat = get_dense_C(wl, k_func=region_func, max_r=max_r)
    else:
        region_mat = 0.0

    max_r = 6.0 * phi.l # [km/s]
    # Create a partial function which returns the proper element.
    k_func = make_k_func(phi)

    data_mat = get_dense_C(wl, k_func=k_func, max_r=max_r) + phi.sigAmp*sigma_mat + region_mat

    # Get many random draws from data_mat
    draws = random_draws(data_mat, num=4)
    min_spec, max_spec = std_envelope(draws)


if args.matplotlib:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=2, figsize=(10, 8), sharex=True)

    ax[0].plot(wl, data, "b", label="data")
    ax[0].plot(wl, model, "r", label="model")

    ax[0].set_ylabel(r"$f_\lambda$")
    ax[0].legend(loc="lower right")

    if args.noise:
        ax[1].fill_between(wl, 3*min_spec, 3*max_spec, zorder=0, color=s3)
        ax[1].fill_between(wl, min_spec, max_spec, zorder=0, color=s1)

    ax[1].plot(wl, resid, "k", label="residual")
    ax[1].set_xlabel(r"$\lambda$ [AA]")
    ax[1].set_ylabel(r"$f_\lambda$")
    ax[1].legend(loc="lower right")

    ax[1].set_xlim(wl[0], wl[-1])

    if args.regions:
        for a in ax:
            for mu in mus:
                a.axvline(mu, color="0.7", zorder=-100)



    # fig.subplots_adjust()
    fig.savefig(config["plotdir"] + args.file + ".png")
