#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description="Create and manipulate a PCA decomposition of the synthetic spectral library.")
parser.add_argument("--create", action="store_true", help="Create a PCA decomposition.")

parser.add_argument("--plot", choices=["reconstruct", "priors", "emcee",
                                       "emulator"], help="""
                    reconstruct: plot the original synthetic spectra vs. the PCA
                    reconstructed spectra.\n
                    priors: plot the chosen priors on the parameters
                    emcee: plot the triangle diagram for the result of the emcee
                    optimization.
                    emulator: plot weight interpolations""")

parser.add_argument("--optimize", choices=["fmin", "emcee"], help="""Optimize the
                    emulator using either a downhill simplex algorithm or the
                    emcee ensemble sampler algorithm.""")

parser.add_argument("--continue", action="store_true", help="""Designed to be used
                    with the --optimize flag to continue from the previous set
                    of parameters. If this is left off, the chain will start
                    from your initial guess specified in config.yaml.
                    """)

args = parser.parse_args()

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import Starfish
from Starfish.grid_tools import HDF5Interface
from Starfish.emulator import PCAGrid


if args.create:

    myHDF5 = HDF5Interface()
    my_pca = PCAGrid.create(myHDF5)
    my_pca.write()

if args.plot == "reconstruct":


    my_HDF5 = HDF5Interface()
    my_pca = PCAGrid.open()

    recon_fluxes = my_pca.reconstruct_all()

    # we need to apply the same normalization to the synthetic fluxes that we
    # used for the reconstruction
    fluxes = np.empty((my_pca.M, my_pca.npix))
    for i, spec in enumerate(my_HDF5.fluxes):
        fluxes[i,:] = spec

    # Normalize all of the fluxes to an average value of 1
    # In order to remove uninteresting correlations
    fluxes = fluxes/np.average(fluxes, axis=1)[np.newaxis].T
    
    data = zip(my_HDF5.grid_points, fluxes, recon_fluxes)

    # Define the plotting function
    def plot(data):
        par, real, recon = data
        fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
        ax[0].plot(my_pca.wl, real)
        ax[0].plot(my_pca.wl, recon)
        ax[0].set_ylabel(r"$f_\lambda$")

        ax[1].plot(my_pca.wl, real - recon)
        ax[1].set_xlabel(r"$\lambda$ [AA]")
        ax[1].set_ylabel(r"$f_\lambda$")

        fmt = "_".join(["{:.2f}" for i in range(len(Starfish.parname))])
        name = fmt.format(*[p for p in par])
        ax[0].set_title(name)
        fig.savefig(Starfish.config["plotdir"] + "PCA_" + name + ".png")
        plt.close("all")

    p = mp.Pool(mp.cpu_count())
    p.map(plot, data)
