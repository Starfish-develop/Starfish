import os
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import Starfish
from Starfish.grid_tools import HDF5Interface
from .pca import PCAGrid, _prior


def plot_reconstructed(show=False, save=True, parallel=True):
    """
    Plot the reconstructed spectra at each grid point.

    :param show: If True, will show each plot. (Default is False)
    :type show: bool
    :param save: If True, will save each plot into the ``config["plotdir"]`` from ``config.yaml``. (Default is True)
    :param parallel: If True, will pool the creation of the plots. (Default is True)
    :type parallel: bool
    """
    grid = HDF5Interface()
    pca_grid = PCAGrid.open()

    recon_fluxes = pca_grid.reconstruct_all()

    # we need to apply the same normalization to the synthetic fluxes that we
    # used for the reconstruction
    fluxes = np.empty((pca_grid.M, pca_grid.npix))
    for i, spec in enumerate(grid.fluxes):
        fluxes[i, :] = spec

    # Normalize all of the fluxes to an average value of 1
    # In order to remove uninteresting correlations
    fluxes = fluxes / np.average(fluxes, axis=1)[np.newaxis].T

    data = zip(grid.grid_points, fluxes, recon_fluxes)

    plotdir = os.path.expandvars(Starfish.config["plotdir"])

    # Define the plotting function
    def plot(data):
        par, real, recon = data
        fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
        ax[0].plot(pca_grid.wl, real)
        ax[0].plot(pca_grid.wl, recon)
        ax[0].set_ylabel(r"$f_\lambda$")

        ax[1].plot(pca_grid.wl, real - recon)
        ax[1].set_xlabel(r"$\lambda$ [AA]")
        ax[1].set_ylabel(r"$f_\lambda$")

        fmt = "=".join(["{:.2f}" for _ in range(len(Starfish.parname))])
        name = fmt.format(*[p for p in par])
        ax[0].set_title(name)
        plt.tight_layout()
        if save:
            filename = os.path.join(plotdir, "PCA_{}.png".format(name))
            fig.savefig(filename)

    if parallel:
        with mp.Pool() as p:
            p.map(plot, data)
    else:
        map(plot, data)

    if show:
        plt.show()
    else:
        plt.close("all")


def plot_eigenspectra(show=False, save=True):
    """
    Plot the eigenspectra for a PCA Grid

    :param show: If True, will show the plot. (Default is False)
    :type show: bool
    :param save: If True, will save the plot into the ``config["plotdir"]`` from ``config.yaml``. (Default is True)
    :type save: bool
    """
    pca_grid = PCAGrid.open()

    row_height = 3  # in
    margin = 0.5  # in

    fig_height = pca_grid.m * (row_height + margin) + margin
    fig_width = 14  # in

    fig = plt.figure(figsize=(fig_width, fig_height))

    for i in range(pca_grid.m):
        ax = plt.subplot2grid((pca_grid.m, 4), (i, 0), colspan=3)
        ax.plot(pca_grid.wl, pca_grid.eigenspectra[i])
        ax.set_xlabel(r"$\lambda$ [AA]")
        ax.set_ylabel(r"$\xi_{}$".format(i))

        ax = plt.subplot2grid((pca_grid.m, 4), (i, 3))
        ax.hist(pca_grid.w[i], histtype="step", normed=True)
        ax.set_xlabel(r"$w_{}$".format(i))
        ax.set_ylabel("count")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3, left=0.1, right=0.98, bottom=0.1, top=0.98)
    plotdir = os.path.expandvars(Starfish.config["plotdir"])
    if save:
        fig.savefig(os.path.join(plotdir, "eigenspectra.png"))
    if show:
        plt.show()
    else:
        plt.close("all")


def plot_priors(show=False, save=True):
    """
    Plot the gamma priors for the PCA optimization problem.

    .. seealso:: :func:`PCAGrid.optimize`

    :param show: If True, will show the plot. (Default is False)
    :type show: bool
    :param save: If True, will save the plot into the ``config["plotdir"]`` from ``config.yaml``. (Default is True)
    :type save: bool
    """
    # Read the priors on each of the parameters from Starfish config.yaml
    priors = Starfish.PCA["priors"]
    plotdir = os.path.expandvars(Starfish.config["plotdir"])
    for i, par in enumerate(Starfish.parname):
        s, r = priors[i]
        mu = s / r
        x = np.linspace(0.01, 2 * mu)
        prob = _prior(x, s, r)
        plt.plot(x, prob)
        plt.xlabel(par)
        plt.ylabel("Probability")
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(plotdir, "prior_{}.png".format(par)))

    if show:
        plt.show()
    else:
        plt.close("all")
