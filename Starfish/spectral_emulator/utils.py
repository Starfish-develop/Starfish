import os
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

import Starfish
from Starfish.grid_tools import HDF5Interface
from .pca import PCAGrid


def Phi(eigenspectra, M):
    """
    Warning: for any spectra of real-world dimensions, this routine will
    likely over flow memory.

    :param eigenspectra:
    :type eigenspectra: 2D array
    :param M: number of spectra in the synthetic library
    :type M: int
    Calculate the matrix Phi using the kronecker products.
    """

    return np.hstack([np.kron(np.eye(M), eigenspectrum[np.newaxis].T) for eigenspectrum in eigenspectra])


def get_w_hat(eigenspectra, fluxes, M):
    """
    Since we will overflow memory if we actually calculate Phi, we have to
    determine w_hat in a memory-efficient manner.

    """
    m = len(eigenspectra)
    out = np.empty((M * m,))
    for i in range(m):
        for j in range(M):
            out[i * M + j] = eigenspectra[i].T.dot(fluxes[j])

    PhiPhi = np.linalg.inv(skinny_kron(eigenspectra, M))

    return PhiPhi.dot(out)


def skinny_kron(eigenspectra, M):
    """
    Compute Phi.T.dot(Phi) in a memory efficient manner.

    eigenspectra is a list of 1D numpy arrays.
    """
    m = len(eigenspectra)
    out = np.zeros((m * M, m * M))

    # Compute all of the dot products pairwise, beforehand
    dots = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            dots[i, j] = eigenspectra[i].T.dot(eigenspectra[j])

    for i in range(M * m):
        for jj in range(m):
            ii = i // M
            j = jj * M + (i % M)
            out[i, j] = dots[ii, jj]
    return out


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
