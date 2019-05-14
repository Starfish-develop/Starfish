import itertools
import multiprocessing as mp
import os

import corner
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from Starfish.grid_tools import HDF5Interface
from .emulator import Emulator

plt.style.use('seaborn')


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
    recon_fluxes = []
    recon_err = []
    for params in grid.grid_points:
        flux, var = emulator.load_flux(params, full_cov=False)
        recon_fluxes.append(flux)
        recon_err.append(np.sqrt(var))
    fluxes = np.array(list(grid.fluxes))
    data = zip(grid.grid_points, fluxes, recon_fluxes, recon_err)

    plotdir = os.path.expandvars(folder)
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    # Define the plotting function
    def plot(datum):
        par, real, recon, err = datum
        fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
        ax[0].plot(emulator.wl, real, lw=1, label='Original')
        ax[0].plot(emulator.wl, recon, lw=1, label='Reconstructed')
        ax[0].fill_between(emulator.wl, recon - 2 * err, recon + 2 * err, color='C1', alpha=0.4)
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
    weights = emulator.draw_weights(params)
    reconstructed = weights @ emulator.eigenspectra

    height = int(emulator.ncomps) * 1.25
    fig = plt.figure(figsize=(8, height))
    gs = gridspec.GridSpec(int(emulator.ncomps) + 1, 1, height_ratios=[3] + list(np.ones(int(emulator.ncomps))))
    ax = plt.subplot(gs[0])
    ax.plot(emulator.wl, reconstructed, lw=1)
    ax.set_ylabel('$f_\lambda$ [erg/cm^2/s/A]')
    plt.setp(ax.get_xticklabels(), visible=False)
    for i in range(emulator.ncomps):
        ax = plt.subplot(gs[i + 1], sharex=ax)
        ax.plot(emulator.wl, emulator.eigenspectra[i], c='0.4', lw=1)
        ax.set_ylabel(r'$\xi_{}$'.format(i))
        if i < emulator.ncomps - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.legend([r'$w_{}$ = {:.2e}'.format(i, weights[i])])
    plt.xlabel('Wavelength (A)')
    plt.tight_layout(h_pad=0.2)

    plt.show()

    if filename is not None:
        plt.savefig(os.path.expandvars(filename))


def plot_emulator(emulator):
    import itertools
    T = np.unique(emulator.grid_points[:, 0])
    logg = np.unique(emulator.grid_points[:, 1])
    Z = np.unique(emulator.grid_points[:, 2])
    params = np.array(list(itertools.product(T, logg[:1], Z[:1])))
    idxs = np.array([emulator.get_index(p) for p in params])
    weights = emulator.weights[idxs.astype('int')].T
    if emulator.ncomps < 4:
        fix, axes = plt.subplots(emulator.ncomps, 1, sharex=True, figsize=(8, (emulator.ncomps - 1) * 2))
    else:
        fix, axes = plt.subplots(int(np.ceil(emulator.ncomps / 2)), 2, sharex=True,
                                 figsize=(13, (emulator.ncomps - 1) * 2))
    axes = np.ravel(np.array(axes).T)
    [ax.set_ylabel('$w_{}$'.format(i)) for i, ax in enumerate(axes)]
    for i, w in enumerate(weights):
        axes[i].plot(T, w, 'o')

    Ttest = np.linspace(T.min(), T.max(), 100)
    Xtest = np.array(list(itertools.product(Ttest, logg[:1], Z[:1])))
    mus = []
    covs = []
    for X in Xtest:
        m, c = emulator(X)
        mus.append(m)
        covs.append(c)
    mus = np.array(mus)
    covs = np.array(covs)
    sigs = np.sqrt(np.diagonal(covs, axis1=-2, axis2=-1))
    for i, (m, s), in enumerate(zip(mus.T, sigs.T)):
        axes[i].plot(Ttest, m, 'C1')
        axes[i].fill_between(Ttest, m - 2 * s, m + 2 * s, color='C1', alpha=0.4)
    axes[-1].set_xlabel('T (K)')
    plt.suptitle('Weights for fixed $\log g={:.2f}$, $[Fe/H]={:.2f}$'.format(logg[0], Z[0]), fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
