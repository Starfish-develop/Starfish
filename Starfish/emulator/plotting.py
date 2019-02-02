import itertools
import multiprocessing as mp
import os

import corner
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from Starfish import config
from Starfish.grid_tools import HDF5Interface
from .emulator import Emulator


def plot_reconstructed(emulator, grid, folder):
    """
    Plot the reconstructed spectra and residual at each grid point.

    Parameters
    ----------
    emulator : :class:`Emulaor`
    grid : :class:`Starfish.grid_tools.GridInterface`
    folder : str or path-like

    Example of a reconstructed spectrum at [6200, 4.5, 0.5]

    .. figure:: assets/6200_4.50_0.00.png
        :align: center
        :width: 80%
    """
    weights = emulator.draw_many_weights(grid.grid_points)
    recon_fluxes = weights @ emulator.eigenspectra
    fluxes = np.array(list(grid.fluxes))
    data = zip(grid.grid_points, fluxes, recon_fluxes)

    plotdir = os.path.expandvars(folder)
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    # Define the plotting function
    def plot(data):
        par, real, recon = data
        fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
        plt.style.use('seaborn')
        ax[0].plot(emulator.wl, real, lw=1, label='Original')
        ax[0].plot(emulator.wl, recon, lw=1, label='Reconstructed')
        ax[0].set_yscale('log')
        ax[0].set_ylabel(r"$f_\lambda$ [erg/cm^2/s/A]")
        ax[0].legend()

        ax[1].plot(emulator.wl, real - recon, lw=1, label='Residuals')
        ax[1].set_xlabel(r"$\lambda$ [A]")
        ax[1].set_ylabel(r"$\Delta f_\lambda$")
        ax[1].legend()

        name = 'T={} logg={} Z={}'.format(*par)
        ax[0].set_title(name)
        plt.tight_layout()
        filename = os.path.join(plotdir, '{:.0f}_{}_{}.png'.format(*par))
        plt.savefig(filename)
        plt.close()

    list(map(plot, data))


def plot_eigenspectra(emulator, params, filename=None):
    """

    Parameters
    ----------
    emulator
    params
    filename : str or path-like, optional
        If provided, will save the plot at the given filename

    Example of a deconstructed set of eigenspectra

    .. figure:: assets/eigenspectra.png
        :align: center
    """
    mu, cov = emulator(params)
    weights = np.random.multivariate_normal(mu, cov)
    reconstructed = weights @ emulator.eigenspectra

    height = int(emulator.ncomps) * 1.25
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(8, height))
    gs = gridspec.GridSpec(int(emulator.ncomps) + 1, 1, height_ratios=[3] + list(np.ones(int(emulator.ncomps))))
    ax = plt.subplot(gs[0])
    ax.plot(emulator.wl, reconstructed, lw=1)
    ax.set_ylabel('$f_\lambda$ [erg/cm^2/s/A]')
    plt.setp(ax.get_xticklabels(), visible=False)
    for i in range(emulator.ncomps):
        ax = plt.subplot(gs[i + 1], sharex=ax)
        ax.plot(emulator.wl, emulator.eigenspectra[i], c='0.4', lw=1)
        ax.set_ylabel(rf'$\xi_{i}$')
        if i < emulator.ncomps - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.legend([rf'$w_{i}$ = {weights[i]:.2e}'])
    plt.xlabel('Wavelength (A)')
    plt.tight_layout(h_pad=0.2)

    plt.show()

    if filename is not None:
        plt.savefig(os.path.expandvars(filename))

#
# def plot_emulator(pca_filename=config.PCA["path"], parallel=True):
#     """
#     Plot the optimized fits for the weights of each eigenspectrum for each parameter.
#
#     :param parallel: If True, will pool the creation of the plots. (Default is True)
#     :type parallel: bool
#
#     Example Plot of the weights for the second eigenspectra for temperature when logg=4.0 and [Fe/H]=-0.5
#
#     .. figure:: assets/w2temp4.0-0.5.png
#         :width: 60%
#         :align: center
#     """
#     pca_grid = PCAGrid.open(pca_filename)
#
#     # Print out the emulator parameters in an easily-readable format
#     eparams = pca_grid.eparams
#     lambda_xi = eparams[0]
#     hparams = eparams[1:].reshape((pca_grid.m, -1))
#
#     print("Emulator parameters are:")
#     print("lambda_xi", lambda_xi)
#     for row in hparams:
#         print(row)
#
#     emulator = Emulator(pca_grid, eparams)
#
#     # We will want to produce interpolated plots spanning each parameter dimension,
#     # for each eigenspectrum.
#
#     # Create a list of parameter blocks.
#     # Go through each parameter, and create a list of all parameter combination of
#     # the other two parameters.
#     unique_points = [np.unique(pca_grid.gparams[:, i]) for i in range(len(config.grid["parname"]))]
#     blocks = []
#     for ipar, pname in enumerate(config.grid["parname"]):
#         upars = unique_points.copy()
#         dim = upars.pop(ipar)
#         ndim = len(dim)
#
#         # use itertools.product to create permutations of all possible values
#         par_combos = itertools.product(*upars)
#
#         # Now, we want to create a list of parameters in the original order.
#         for static_pars in par_combos:
#             par_list = []
#             for par in static_pars:
#                 par_list.append(par * np.ones((ndim,)))
#
#             # Insert the changing dim in the right location
#             par_list.insert(ipar, dim)
#
#             blocks.append(np.vstack(par_list).T)
#
#     # Now, this function takes a parameter block and plots all of the eigenspectra.
#     npoints = 40  # How many points to include across the active dimension
#     ndraw = 8  # How many draws from the emulator to use
#
#     def plot_block(block):
#         # block specifies the parameter grid points
#         # fblock defines a parameter grid that is finer spaced than the gridpoints
#
#         # Query for the weights at the grid points.
#         ww = np.empty((len(block), pca_grid.m))
#         for i, param in enumerate(block):
#             weights = pca_grid.get_weights(param)
#             ww[i, :] = weights
#
#         # Determine the active dimension by finding the one that has unique > 1
#         uni = np.array([len(np.unique(block[:, i])) for i in range(len(config.grid["parname"]))])
#         active_dim = np.where(uni > 1)[0][0]
#
#         ublock = block.copy()
#         ablock = ublock[:, active_dim]
#         ublock = np.delete(ublock, active_dim, axis=1)
#         nactive = len(ablock)
#
#         fblock = []
#         for par in ublock[0, :]:
#             # Create a space of the parameter the same length as the active
#             fblock.append(par * np.ones((npoints,)))
#
#         # find min and max of active dim. Create a linspace of `npoints` spanning from
#         # min to max
#         active = np.linspace(ablock[0], ablock[-1], npoints)
#
#         fblock.insert(active_dim, active)
#         fgrid = np.vstack(fblock).T
#
#         # Draw multiple times at the location.
#         weight_draws = []
#         for i in range(ndraw):
#             weight_draws.append(emulator.draw_many_weights(fgrid))
#
#         # Now make all of the plots
#         for eig_i in range(pca_grid.m):
#             fig, ax = plt.subplots(nrows=1, figsize=(6, 6))
#
#             x0 = block[:, active_dim]  # x-axis
#             # Weight values at grid points
#             y0 = ww[:, eig_i]
#             ax.plot(x0, y0, "bo")
#
#             x1 = fgrid[:, active_dim]
#             for i in range(ndraw):
#                 y1 = weight_draws[i][:, eig_i]
#                 ax.plot(x1, y1)
#
#             ax.set_ylabel(r"$w_{:}$".format(eig_i))
#             ax.set_xlabel(config.grid["parname"][active_dim])
#             plt.tight_layout()
#
#             fstring = "w{:}".format(eig_i) + config.grid["parname"][active_dim] + "".join(
#                 ["{:.1f}".format(ub) for ub in ublock[0, :]])
#             plotdir = os.path.expandvars(config["plotdir"])
#             fig.savefig(os.path.join(plotdir, fstring + ".png"))
#             plt.close()
#
#     # Create a pool of workers and map the plotting to these.
#     if parallel:
#         with mp.Pool(mp.cpu_count() - 1) as p:
#             p.map(plot_block, blocks)
#     else:
#         list(map(plot_block, blocks))
#
#     plt.close('all')
