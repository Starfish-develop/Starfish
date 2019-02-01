import itertools
import multiprocessing as mp
import os

import corner
import matplotlib.pyplot as plt
import numpy as np

from Starfish import config
from Starfish.grid_tools import HDF5Interface
from .emulator import Emulator
from .utils import PCAGrid, _prior


def plot_reconstructed(pca_filename=config.PCA["path"], save=True, parallel=True):
    """
    Plot the reconstructed spectra and residual at each grid point.

    :param parallel: If True, will pool the creation of the plots. (Default is True)
    :type parallel: bool

    Example of a reconstructed spectrum at [7200, 5.5, 0.5]

    .. figure:: assets/PCA_7200.00_5.50_0.50.png
        :align: center
        :width: 80%
    """
    grid = HDF5Interface()
    pca_grid = PCAGrid.open(pca_filename)

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

    plotdir = os.path.expandvars(config["plotdir"])

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

        fmt = "=".join(["{:.2f}" for _ in range(len(config.grid["parname"]))])
        name = fmt.format(*[p for p in par])
        ax[0].set_title(name)
        plt.tight_layout()
        filename = os.path.join(plotdir, "PCA_{}.png".format(name))
        fig.savefig(filename)
        plt.close()

    if parallel:
        with mp.Pool() as p:
            p.map(plot, data)
    else:
        list(map(plot, data))


def plot_eigenspectra(pca_filename=config.PCA["path"], show=False, save=True):
    """
    Plot the eigenspectra for a PCA Grid

    :param show: If True, will show the plot. (Default is False)
    :type show: bool
    :param save: If True, will save the plot into the ``config["plotdir"]`` from ``config.yaml``. (Default is True)
    :type save: bool

    Example of a deconstructed set of eigenspectra

    .. figure:: assets/eigenspectra.png
        :align: center

    """
    if not show and not save:
        raise ValueError("If you don't save OR show the plots nothing will happen.")
    pca_grid = PCAGrid.open(pca_filename)

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
    plotdir = os.path.expandvars(config["plotdir"])
    if save:
        fig.savefig(os.path.join(plotdir, "eigenspectra.png"))
    if show:
        plt.show()
    else:
        plt.close("all")


def plot_priors(show=False, save=True):
    """
    Plot the gamma priors for the PCA optimization problem.

    :param show: If True, will show the plot. (Default is False)
    :type show: bool
    :param save: If True, will save the plot into the ``config["plotdir"]`` from ``config.yaml``. (Default is True)
    :type save: bool

    Example prior plot

    .. figure:: assets/prior.png
        :align: center

    .. seealso:: :func:`PCAGrid.optimize`
    """
    if not show and not save:
        raise ValueError("If you don't save OR show the plots nothing will happen.")
    # Read the priors on each of the parameters from config.yaml
    priors = config.PCA["priors"]
    plotdir = os.path.expandvars(config["plotdir"])
    fig, axes = plt.subplots(1, len(config.grid["parname"]), sharey=True, figsize=(4 * len(config.grid["parname"]), 4))
    axes[0].set_ylabel("Probability")
    for i, par in enumerate(config.grid["parname"]):
        s, r = priors[i]
        mu = s / r
        x = np.linspace(0.01, 2 * mu)
        prob = _prior(x, s, r)
        axes[i].plot(x, prob, label='s={:.2f}, r={:.2f}'.format(s, r))
        axes[i].axvline(mu, ls='--', c='k', label='$\mu$={:.1f}'.format(mu))
        axes[i].set_xlabel(par)
        axes[i].legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    if show:
        plt.show()
    if save:
        plt.savefig(os.path.join(plotdir, "prior.png"))


def plot_corner(pca_filename=config.PCA["path"], show=False, save=True):
    """
    Plot the corner plots for the ``emcee`` PCA hyper parameters

    :param show: If True, will show the plot. (Default is False)
    :type show: bool
    :param save: If True, will save the plot into the ``config["plotdir"]`` from ``config.yaml``. (Default is True)
    :type save: bool

    Example corner plot

    .. figure:: assets/triangle_0.png
        :width: 90%
        :align: center
    """
    if not show and not save:
        raise ValueError("If you don't save OR show the plots nothing will happen.")
    # Load in file
    pca_grid = PCAGrid.open(pca_filename)
    flatchain = pca_grid.emcee_chain

    # figure out how many separate triangle plots we need to make
    npar = len(config.grid["parname"]) + 1
    labels = ["amp"] + config.grid["parname"]

    # Make a histogram of lambda xi
    plt.hist(flatchain[:, 0], histtype="step", normed=True)
    plt.title(r"$\lambda_\xi$")
    plt.xlabel(r"$\lambda_\xi$")
    plt.ylabel("prob")
    plt.tight_layout()
    plotdir = os.path.expandvars(config["plotdir"])
    if save:
        plt.savefig(os.path.join(plotdir, "triangle_lambda_xi.png"))

    # Make a triangle plot for each eigenspectrum independently
    for i in range(pca_grid.m):
        start = 1 + i * npar
        end = 1 + (i + 1) * npar
        figure = corner.corner(flatchain[:, start:end], quantiles=[0.16, 0.5, 0.84],
                               plot_contours=True, plot_datapoints=False, show_titles=True, labels=labels)
        plt.tight_layout()
        if save:
            figure.savefig(os.path.join(plotdir, "triangle_{}.png".format(i)))

    if show:
        plt.show()
    else:
        plt.close("all")


def plot_emulator(pca_filename=config.PCA["path"], parallel=True):
    """
    Plot the optimized fits for the weights of each eigenspectrum for each parameter.

    :param parallel: If True, will pool the creation of the plots. (Default is True)
    :type parallel: bool

    Example Plot of the weights for the second eigenspectra for temperature when logg=4.0 and [Fe/H]=-0.5

    .. figure:: assets/w2temp4.0-0.5.png
        :width: 60%
        :align: center
    """
    pca_grid = PCAGrid.open(pca_filename)

    # Print out the emulator parameters in an easily-readable format
    eparams = pca_grid.eparams
    lambda_xi = eparams[0]
    hparams = eparams[1:].reshape((pca_grid.m, -1))

    print("Emulator parameters are:")
    print("lambda_xi", lambda_xi)
    for row in hparams:
        print(row)

    emulator = Emulator(pca_grid, eparams)

    # We will want to produce interpolated plots spanning each parameter dimension,
    # for each eigenspectrum.

    # Create a list of parameter blocks.
    # Go through each parameter, and create a list of all parameter combination of
    # the other two parameters.
    unique_points = [np.unique(pca_grid.gparams[:, i]) for i in range(len(config.grid["parname"]))]
    blocks = []
    for ipar, pname in enumerate(config.grid["parname"]):
        upars = unique_points.copy()
        dim = upars.pop(ipar)
        ndim = len(dim)

        # use itertools.product to create permutations of all possible values
        par_combos = itertools.product(*upars)

        # Now, we want to create a list of parameters in the original order.
        for static_pars in par_combos:
            par_list = []
            for par in static_pars:
                par_list.append(par * np.ones((ndim,)))

            # Insert the changing dim in the right location
            par_list.insert(ipar, dim)

            blocks.append(np.vstack(par_list).T)

    # Now, this function takes a parameter block and plots all of the eigenspectra.
    npoints = 40  # How many points to include across the active dimension
    ndraw = 8  # How many draws from the emulator to use

    def plot_block(block):
        # block specifies the parameter grid points
        # fblock defines a parameter grid that is finer spaced than the gridpoints

        # Query for the weights at the grid points.
        ww = np.empty((len(block), pca_grid.m))
        for i, param in enumerate(block):
            weights = pca_grid.get_weights(param)
            ww[i, :] = weights

        # Determine the active dimension by finding the one that has unique > 1
        uni = np.array([len(np.unique(block[:, i])) for i in range(len(config.grid["parname"]))])
        active_dim = np.where(uni > 1)[0][0]

        ublock = block.copy()
        ablock = ublock[:, active_dim]
        ublock = np.delete(ublock, active_dim, axis=1)
        nactive = len(ablock)

        fblock = []
        for par in ublock[0, :]:
            # Create a space of the parameter the same length as the active
            fblock.append(par * np.ones((npoints,)))

        # find min and max of active dim. Create a linspace of `npoints` spanning from
        # min to max
        active = np.linspace(ablock[0], ablock[-1], npoints)

        fblock.insert(active_dim, active)
        fgrid = np.vstack(fblock).T

        # Draw multiple times at the location.
        weight_draws = []
        for i in range(ndraw):
            weight_draws.append(emulator.draw_many_weights(fgrid))

        # Now make all of the plots
        for eig_i in range(pca_grid.m):
            fig, ax = plt.subplots(nrows=1, figsize=(6, 6))

            x0 = block[:, active_dim]  # x-axis
            # Weight values at grid points
            y0 = ww[:, eig_i]
            ax.plot(x0, y0, "bo")

            x1 = fgrid[:, active_dim]
            for i in range(ndraw):
                y1 = weight_draws[i][:, eig_i]
                ax.plot(x1, y1)

            ax.set_ylabel(r"$w_{:}$".format(eig_i))
            ax.set_xlabel(config.grid["parname"][active_dim])
            plt.tight_layout()

            fstring = "w{:}".format(eig_i) + config.grid["parname"][active_dim] + "".join(
                ["{:.1f}".format(ub) for ub in ublock[0, :]])
            plotdir = os.path.expandvars(config["plotdir"])
            fig.savefig(os.path.join(plotdir, fstring + ".png"))
            plt.close()

    # Create a pool of workers and map the plotting to these.
    if parallel:
        with mp.Pool(mp.cpu_count() - 1) as p:
            p.map(plot_block, blocks)
    else:
        list(map(plot_block, blocks))

    plt.close('all')
