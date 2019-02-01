import logging
import os

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.decomposition import NMF

from Starfish.grid_tools.utils import determine_chunk_log
from Starfish.utils import calculate_dv

from ._covariance import Sigma, V12, V22, V12m, V22m
from .utils import skinny_kron, get_w_hat

log = logging.getLogger(__name__)


class Emulator:
    def __init__(self, grid_points, wavelength, weights, eigenspectra, w_hat, lambda_xi=1, variance=None, lengthscales=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.grid_points = grid_points
        self.wavelength = wavelength
        self.weights = weights
        self.eigenspectra = eigenspectra

        self.dv = calculate_dv(wavelength)
        self.ncomps = eigenspectra.shape[0]

        lengthscale_shape = (self.ncomps, grid_points.shape[-1])

        self.lambda_xi = lambda_xi
        self.variance = variance if variance is not None else np.ones(self.ncomps)
        self.lengthscales = lengthscales if lengthscales is not None else np.ones(lengthscale_shape)

        # Determine the minimum and maximum bounds of the grid
        self.min_params = grid_points.min(axis=0)
        self.max_params = grid_points.max(axis=0)

        # TODO find better variable names for the following
        self.iPhiPhi = (1. / self.lambda_xi) * np.linalg.inv(skinny_kron(self.eigenspectra, self.grid_points.shape[0]))
        self.v11_cho = cho_factor(self.iPhiPhi + Sigma(self.grid_points, self.h2params))
        self.w_hat = w_hat

    @classmethod
    def open(cls, filename):
        """
        Create an Emulator object from an HDF5 file.
        """
        filename = os.path.expandvars(filename)
        pass

    def save(self, filename):
        filename = os.path.expandvars(filename)
        pass

    @classmethod
    def from_grid(cls, grid, ncomps=6):
        """
        Create an Emulator using NMF decomposition from a GridInterface.

        Parameters
        ----------
        grid : :class:`GridInterface`
            The grid interface to decompose
        ncomps : int, optional
            The number of eigenspectra to use for NMF. The larger this number, the less reconstruction error.
            Default is 6.

        See Also
        --------
        sklearn.decomposition.NMF
        """
        fluxes = np.array(list(grid.fluxes))
        nmf = NMF(n_components=ncomps)
        weights = nmf.fit_transform(fluxes)
        eigenspectra = nmf.components_
        # This is basically the mean square error of the reconstruction
        log.info('NMF completed with reconstruction error {}'.format(nmf.reconstruction_err_))
        w_hat = get_w_hat(eigenspectra, grid.fluxes, len(grid.grid_points))
        return cls(grid.grid_points, grid.wl, weights, eigenspectra, w_hat)

    def reconstruct(self, weights):
        """
        Reconstruct spectra given weights.

        Parameters
        ----------
        weights : array_like
            The weights for reconstruction.

        Returns
        -------
        numpy.ndarray
            If weights.ndim == 1, will return a single spectrum, otherwise will return len(weights) spectra
        """
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        return weights @ self.eigenspectra

    def __call__(self, params):
        """
        Gets the mu and cov matrix for a given set of params

        Parameters
        ----------
        params : array_like
            The parameters to sample at. Should be consistent with the shapes of the original grid points.

        Returns
        -------
        mu : numpy.ndarray (len(params),)
        cov : numpy.ndarray (len(params), len(params))

        Raises
        ------
        ValueError
            If querying the emulator outside of its trained grid points
        """
        params = np.array(params)
        # If the pars is outside of the range of emulator values, raise a ModelError
        if np.any(params < self.min_params) or np.any(params > self.max_params):
            raise ValueError('Querying emulator outside of original parameter range.')

        # Do this according to R&W eqn 2.18, 2.19
        # Recalculate V12, V21, and V22.
        v12 = V12(params, self.grid_points, self.h2params, self.ncomps)
        v22 = V22(params, self.h2params, self.ncomps)

        # Recalculate the covariance

        mu = v12.T @ cho_solve(self.v11_cho, self.w_hat).squeeze()
        cov = v22 - v12.T @ cho_solve(self.v11_cho, v12)
        return mu, cov

    def load_flux(self, params, full_cov=False):
        """
        Interpolate a model given any parameters within the grid's parameter range using eigenspectrum reconstruction
        by sampling from the weight distributions.

        :param params: The parameters to sample at. Should have same length as ``grid["parname"]`` in ``config.yaml``
        :type: iterable
        :param full_cov: If true, will return the full covariance matrix for the weights
        :type full_cov: bool
        :return: tuple of (mu, cov) or (mu, var)

        .. warning::
            When returning the emulator covariance matrix, this is a costly operation and will return a
            datastructure with (N_pix x N_pix) data points. For now, don't do it.
        """
        params = np.array(params)
        mu, cov = self(params)
        weights = self.draw_weights(mu, cov)
        if not full_cov:
            cov = np.diag(cov)
        C = self.eigenspectra.T @ cov @ self.eigenspectra
        return self.reconstruct(weights), C

    def determine_chunk_log(self, wavelength, buffer=50):
        """
        Possibly truncate the wavelength and eigenspectra in response to some new wavelengths

        Parameters
        ----------
        wavelength : array_like
            The new wavelengths to truncate to
        buffer : float, optional
            The wavelength buffer, in Angstrom. Default is 50

        See Also
        --------
        Starfish.grid_tools.utils.determine_chunk_log
        """
        if not isinstance(wavelength, np.ndarray):
            wavelength = np.array(wavelength)

        # determine the indices
        wl_min = wavelength.min()
        wl_max = wavelength.max()

        wl_min -= buffer
        wl_max += buffer

        ind = determine_chunk_log(self.wavelength, wl_min, wl_max)
        trunc_wavelength = self.wavelength[ind]

        assert (trunc_wavelength.min() <= wl_min) and (trunc_wavelength.max() >= wl_max), \
            "Emulator chunking ({:.2f}, {:.2f}) didn't encapsulate " \
            "full wl range ({:.2f}, {:.2f}).".format(trunc_wavelength.min(),
                                                     trunc_wavelength.max(),
                                                     wl_min, wl_max)

        self.wavelength = trunc_wavelength
        self.eigenspectra = self.eigenspectra[:, ind]

    def draw_weights(self, mu, cov):
        return np.random.multivariate_normal(mu, cov)

    def draw_many_weights(self, params):
        """
        :param params: multiple parameters to produce weight draws at.
        :type params: 2D np.array
        """
        # Local variables, different from instance attributes
        v12 = V12m(params, self.grid_points, self.h2params, self.ncomps)
        v22 = V22m(params, self.h2params, self.ncomps)

        mu = v12.T @ cho_solve(self.v11_cho, self.w_hat)
        sig = v22 - v12.T @ cho_solve(self.v11_cho, v12)

        weights = np.random.multivariate_normal(mu, sig)

        # Reshape these weights into a 2D matrix
        weights.shape = (len(params), self.ncomps)

        return weights