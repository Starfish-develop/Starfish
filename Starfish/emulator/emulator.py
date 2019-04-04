import logging
import math
import os
import warnings

import h5py
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.decomposition import PCA

from Starfish.grid_tools.utils import determine_chunk_log
from Starfish.utils import calculate_dv
from .covariance import batch_kernel
from ._utils import get_phi_squared, get_w_hat

log = logging.getLogger(__name__)


class Emulator:
    def __init__(self, grid_points, wavelength, weights, eigenspectra, w_hat, flux_mean, flux_std,
                 lambda_xi=1, variances=None, lengthscales=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.grid_points = grid_points
        self.wl = wavelength
        self.weights = weights
        self.eigenspectra = eigenspectra
        self.flux_mean = flux_mean
        self.flux_std = flux_std

        self.dv = calculate_dv(wavelength)
        self.ncomps = eigenspectra.shape[0]

        self.lambda_xi = lambda_xi
        self.variances = variances if variances is not None else 1e4 * \
            np.ones(self.ncomps)
        if lengthscales is None:
            unique = [sorted(np.unique(param_set))
                      for param_set in self.grid_points.T]
            sep = [5 * np.diff(param).max() for param in unique]
            lengthscales = np.tile(sep, (self.ncomps, 1))

        # self.lengthscales = lengthscales
        self.lengthscales = np.ones((len(self.variances), 3))

        # Determine the minimum and maximum bounds of the grid
        self.min_params = grid_points.min(axis=0)
        self.max_params = grid_points.max(axis=0)

        # TODO find better variable names for the following
        self.iPhiPhi = np.linalg.inv(get_phi_squared(
            self.eigenspectra, self.grid_points.shape[0]))
        self.v11 = self.iPhiPhi / self.lambda_xi + batch_kernel(
            self.grid_points, self.grid_points, self.variances, self.lengthscales)
        self.w_hat = w_hat

        self._trained = False

    @classmethod
    def load(cls, filename):
        """
        Create an Emulator object from an HDF5 file.
        """
        filename = os.path.expandvars(filename)
        with h5py.File(filename) as base:
            grid_points = base['grid_points'][:]
            wavelength = base['wavelength'][:]
            weights = base['weights'][:]
            eigenspectra = base['eigenspectra'][:]
            flux_mean = base['flux_mean'][:]
            flux_std = base['flux_std'][:]
            w_hat = base['w_hat'][:]
            lambda_xi = base['hyper_parameters']['lambda_xi'][()]
            variances = base['hyper_parameters']['variances'][:]
            lengthscales = base['hyper_parameters']['lengthscales'][:]
            trained = base.attrs['trained']

        emulator = cls(grid_points, wavelength, weights, eigenspectra, w_hat, flux_mean, flux_std, lambda_xi,
                       variances, lengthscales)
        emulator._trained = trained
        return emulator

    def save(self, filename):
        filename = os.path.expandvars(filename)
        with h5py.File(filename, 'w') as base:
            base.create_dataset(
                'grid_points', data=self.grid_points, compression=9)
            waves = base.create_dataset(
                'wavelength', data=self.wl, compression=9)
            waves.attrs['unit'] = 'Angstrom'
            base.create_dataset('weights', data=self.weights, compression=9)
            eigens = base.create_dataset(
                'eigenspectra', data=self.eigenspectra, compression=9)
            base.create_dataset(
                'flux_mean', data=self.flux_mean, compression=9)
            base.create_dataset('flux_std', data=self.flux_std, compression=9)
            eigens.attrs['unit'] = 'erg/cm^2/s/Angstrom'
            base.create_dataset('w_hat', data=self.w_hat, compression=9)
            base.attrs['trained'] = self._trained
            hp_group = base.create_group('hyper_parameters')
            hp_group.create_dataset('lambda_xi', data=self.lambda_xi)
            hp_group.create_dataset(
                'variances', data=self.variances, compression=9)
            hp_group.create_dataset(
                'lengthscales', data=self.lengthscales, compression=9)

        self.log.info('Saved file at {}'.format(filename))

    @classmethod
    def from_grid(cls, grid, **pca_kwargs):
        """
        Create an Emulator using PCA decomposition from a GridInterface.

        Parameters
        ----------
        grid : :class:`GridInterface`
            The grid interface to decompose
        pca_kwargs : dict, optional

        See Also
        --------
        sklearn.decomposition.PCA
        """
        fluxes = np.array(list(grid.fluxes))
        # Normalize to an average of 1 to remove uninteresting correlation
        fluxes /= fluxes.mean(1, keepdims=True)
        # Center and whiten
        flux_mean = fluxes.mean(0)
        fluxes -= flux_mean
        flux_std = fluxes.std(0)
        fluxes /= flux_std
        default_pca_kwargs = dict(n_components=0.99, svd_solver='full')
        default_pca_kwargs.update(pca_kwargs)
        pca = PCA(**default_pca_kwargs)
        weights = pca.fit_transform(fluxes)
        eigenspectra = pca.components_
        # This is basically the mean square error of the reconstruction
        log.info('PCA fit {:.2f}% of the variance with {} components.'.format(pca.explained_variance_ratio_.sum(),
                                                                              pca.n_components_))
        w_hat = get_w_hat(eigenspectra, fluxes, len(grid.grid_points))
        return cls(grid_points=grid.grid_points, wavelength=grid.wl, weights=weights, eigenspectra=eigenspectra,
                   w_hat=w_hat, flux_mean=flux_mean, flux_std=flux_std)

    def __call__(self, params, full_cov=True, reinterpret_batch=False):
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
        params = np.atleast_2d(params)

        if full_cov and reinterpret_batch:
            raise ValueError(
                'Cannot reshape the full_covariance matrix for many parameters.')

        if not self._trained:
            warnings.warn(
                'This emulator has not been trained and therefore is not reliable. call emulator.train() to train.')

        # If the pars is outside of the range of emulator values, raise a ModelError
        if np.any(params < self.min_params) or np.any(params > self.max_params):
            raise ValueError(
                'Querying emulator outside of original parameter range.')

        # Do this according to R&W eqn 2.18, 2.19
        # Recalculate V12, V21, and V22.
        v12 = batch_kernel(self.grid_points, params,
                           self.variances, self.lengthscales)
        v22 = batch_kernel(params, params, self.variances, self.lengthscales)
        v21 = v12.T

        # Recalculate the covariance
        mu = v21 @ np.linalg.solve(self.v11, self.w_hat)
        cov = v22 - v21 @ np.linalg.solve(self.v11, v12)
        if not full_cov:
            cov = np.diag(cov)
        if reinterpret_batch:
            mu = mu.reshape(-1, self.ncomps, order='F').squeeze()
            cov = cov.reshape(-1, self.ncomps, order='F').squeeze()
        return mu, cov

    @property
    def bulk_fluxes(self):
        return np.vstack([self.eigenspectra, self.flux_mean, self.flux_std])

    def load_flux(self, params):
        """
        Interpolate a model given any parameters within the grid's parameter range using eigenspectrum reconstruction
        by sampling from the weight distributions.

        :param params: The parameters to sample at. Should have same length as ``grid["parname"]`` in ``config.yaml``
        :type: array_like
        """
        mu, cov = self(params, reinterpret_batch=False)
        weights = np.random.multivariate_normal(
            mu, cov).reshape(-1, self.ncomps)
        return weights @ self.eigenspectra * self.flux_std + self.flux_mean

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

        ind = determine_chunk_log(self.wl, wl_min, wl_max)
        trunc_wavelength = self.wl[ind]

        assert (trunc_wavelength.min() <= wl_min) and (trunc_wavelength.max() >= wl_max), \
            "Emulator chunking ({:.2f}, {:.2f}) didn't encapsulate " \
            "full wl range ({:.2f}, {:.2f}).".format(trunc_wavelength.min(),
                                                     trunc_wavelength.max(),
                                                     wl_min, wl_max)

        self.wl = trunc_wavelength
        self.eigenspectra = self.eigenspectra[:, ind]

    def train(self, **opt_kwargs):
        """
        Trains the emulator's hyperparameters using gradient descent

        Parameters
        ----------
        lambda_xi : float, optional
            Starting guess for lambda_xi. If None defaults to the current value. Default is None.
        variances : numpy.ndarray, optional
            Starting guess for variances. If None defaults to the current value. Default is None.
        lengthscales : numpy.ndarray, optional
            Starting guess for lengthscales. If None defaults to the current value. Default is None.
        **opt_kwargs
            Any arguments to pass to the optimizer

        See Also
        --------
        scipy.optimize.minimize

        """
        P0 = self.get_param_vector()

        def nll(P):
            if np.any(P < 0):
                return np.inf
            self.set_param_vector(P)
            loss = -self.log_likelihood()
            self.log.debug('loss: {}'.format(loss))
            return loss

        default_kwargs = dict(method='Nelder-Mead')
        default_kwargs.update(opt_kwargs)
        soln = minimize(nll, P0, **default_kwargs)

        if not soln.success:
            self.log.warning('Optimization did not succeed.')
            self.log.info(soln.message)
            # self.set_param_vector(P0)
        else:
            self.set_param_vector(soln.x)
            self.log.info('Finished optimizing emulator hyperparameters')
            self.log.info('lambda_xi: {}'.format(self.lambda_xi))
            self.log.info('variances: {}'.format(self.variances))
            self.log.info('lengthscales: {}'.format(self.lengthscales))

            self._trained = True

    def get_index(self, params):
        """
        Given a list of stellar parameters (corresponding to a grid point),
        deliver the index that corresponds to the
        entry in the fluxes, grid_points, and weights.

        Parameters
        ----------
        params : array_like
            The stellar parameters

        Returns
        -------
        index : int

        """
        params = np.atleast_2d(params)
        marks = np.abs(self.grid_points -
                       np.expand_dims(params, 1)).sum(axis=-1)
        return marks.argmin(axis=1).squeeze()

    def get_param_vector(self):
        params = [self.lambda_xi] + self.variances.tolist()
        for l in self.lengthscales:
            params.extend(l)
        return np.array(params)

    def set_param_vector(self, params):
        self.lambda_xi = params[0]
        self.variances = params[1:(self.ncomps + 1)]
        self.lengthscales = params[(self.ncomps + 1):].reshape((self.ncomps, -1))
        self.v11 = self.iPhiPhi / self.lambda_xi + \
            batch_kernel(self.grid_points, self.grid_points,
                         self.variances, self.lengthscales)

    def log_likelihood(self):
        L, flag = cho_factor(self.v11)
        logdet = 2 * np.trace(np.log(L))
        central = self.w_hat.T @ cho_solve((L, flag), self.w_hat)
        return -0.5 * (logdet + central + self.weights.size * np.log(2 * np.pi))

    def grad_log_likelihood(self):
        raise NotImplementedError('Not implemented yet.')
