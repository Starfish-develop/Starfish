#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(prog="splot.py", description="Plot JSON files.")
parser.add_argument("file", help="JSON file containing the data, model, and residual.")
parser.add_argument("--regions", help="Optionally plot any instantiated regions on top.")
parser.add_argument("--matplotlib", action="store_true", help="Plot the image using matplotlib")
args = parser.parse_args()

import Starfish
import json
import numpy as np

f = open(args.file, "r")
read = json.load(f) # read is a dictionary
f.close()

wl = np.array(read["wl"])
data = np.array(read["data"])
model = np.array(read["model"])
resid = np.array(read["resid"])

if args.regions:
    f = open(args.regions, "r")
    read = json.load(f) # read is a dictionary
    f.close()

    mus = np.array(read["mus"])

if args.matplotlib:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=2, figsize=(10, 8), sharex=True)

    ax[0].plot(wl, data, "b", label="data")
    ax[0].plot(wl, model, "r", label="model")

    ax[0].set_ylabel(r"$f_\lambda$")
    ax[0].legend(loc="lower right")

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
    fig.savefig(Starfish.config["plotdir"] + args.file + ".png")
