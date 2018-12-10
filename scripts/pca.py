import argparse

import Starfish
from Starfish.grid_tools import HDF5Interface
from Starfish.spectral_emulator import *
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create and manipulate a PCA decomposition of the synthetic spectral library.")
    parser.add_argument("--create", action="store_true", help="Create a PCA decomposition.")
    parser.add_argument("--plot", choices=["reconstruct", "eigenspectra", "priors", "emcee", "emulator"],
                        help="reconstruct: plot the original synthetic spectra vs. the PCA reconstructed spectra.\n "
                             "priors: plot the chosen priors on the parameters emcee: plot the triangle diagram for "
                             "the result of the emcee optimization. emulator: plot weight interpolations")
    parser.add_argument("--optimize", choices=["min", "emcee"],
                        help="Optimize the emulator using either a downhill simplex algorithm or the emcee ensemble "
                             "sampler algorithm.")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to run the emcee ensemble sampler.")
    args = parser.parse_args()

    if args.create:
        grid = HDF5Interface()
        pca_grid = PCAGrid.create(grid)
        pca_grid.write()
    else:
        pca_grid = PCAGrid.open()

    # Create plotting directory if not already created
    if args.plot and not os.path.isdir(Starfish.config['plotdir']):
        os.makedirs(os.path.expandvars(Starfish.config['plotdir']))

    if args.plot == 'reconstruct':
        plot_reconstructed()

    if args.plot == 'eigenspectra':
        plot_eigenspectra()

    if args.plot == 'priors':
        plot_priors()

    if args.plot == 'emcee':
        plot_corner()

    if args.plot == 'emulator':
        plot_emulator()

    if args.optimize == 'min':
        pca_grid.optimize(method='min')
    elif args.optimize == 'emcee':
        pca_grid.optimize(method='emcee', nburn=int(args.samples / 4), nsamples=args.samples)
