import logging
import os
import warnings

import h5py
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.decomposition import NMF

from Starfish.grid_tools.utils import determine_chunk_log
from Starfish.utils import calculate_dv
from ._covariance import block_sigma, V12, V22
from ._utils import skinny_kron, get_w_hat, inverse_block_diag

log = logging.getLogger(__name__)


class Emulator:
    def __init__(self, grid_points, wavelength, weights, eigenspectra, w_hat, lambda_xi=1, variances=None,
                 lengthscales=None, jitter=1e-8):
        self.log = logging.getLogger(self.__class__.__name__)
        self.grid_points = grid_points
        self.wl = wavelength
        self.weights = weights
        self.eigenspectra = eigenspectra

        self.dv = calculate_dv(wavelength)
        self.ncomps = eigenspectra.shape[0]

        self.lambda_xi = lambda_xi
        self.variances = variances if variances is not None else 1e4 * np.ones(self.ncomps)
        if lengthscales is None:
            unique = [sorted(np.unique(param_set)) for param_set in self.grid_points.T]
            sep = [np.median(5 * np.diff(param)) for param in unique]
            lengthscales = np.tile(sep, (self.ncomps, 1))

        self.lengthscales = lengthscales

        # Determine the minimum and maximum bounds of the grid
        self.min_params = grid_points.min(axis=0)
        self.max_params = grid_points.max(axis=0)

        self.jitter = jitter * np.eye(self.ncomps * self.grid_points.shape[0])

        # TODO find better variable names for the following
        self.PhiPhi = np.linalg.inv(skinny_kron(self.eigenspectra, self.grid_points.shape[0]))
        self.block_sigma = block_sigma(self.grid_points, self.variances, self.lengthscales) + self.jitter
        self.v11_cho = cho_factor(self.PhiPhi / self.lambda_xi + self.block_sigma)
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
            w_hat = base['w_hat'][:]
            lambda_xi = base['hyper_parameters']['lambda_xi'][()]
            variances = base['hyper_parameters']['variances'][:]
            lengthscales = base['hyper_parameters']['lengthscales'][:]
            trained = base.attrs['trained']
            jitter = base.attrs['jitter']

        emulator = cls(grid_points, wavelength, weights, eigenspectra, w_hat, lambda_xi, variances, lengthscales,
                       jitter)
        emulator._trained = trained
        return emulator

    def save(self, filename):
        filename = os.path.expandvars(filename)
        with h5py.File(filename, 'w') as base:
            base.create_dataset('grid_points', data=self.grid_points, compression=9)
            waves = base.create_dataset('wavelength', data=self.wl, compression=9)
            waves.attrs['unit'] = 'Angstrom'
            base.create_dataset('weights', data=self.weights, compression=9)
            eigens = base.create_dataset('eigenspectra', data=self.eigenspectra, compression=9)
            eigens.attrs['unit'] = 'erg/cm^2/s/Angstrom'
            base.create_dataset('w_hat', data=self.w_hat, compression=9)
            base.attrs['trained'] = self._trained
            base.attrs['jitter'] = self.jitter.max()
            hp_group = base.create_group('hyper_parameters')
            hp_group.create_dataset('lambda_xi', data=self.lambda_xi)
            hp_group.create_dataset('variances', data=self.variances, compression=9)
            hp_group.create_dataset('lengthscales', data=self.lengthscales, compression=9)

        self.log.info('Saved file at {}'.format(filename))

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
        fluxes /= fluxes.mean(1, keepdims=True)
        nmf = NMF(n_components=ncomps)
        weights = nmf.fit_transform(fluxes)
        eigenspectra = nmf.components_
        # This is basically the mean square error of the reconstruction
        log.info('NMF completed with reconstruction error {}'.format(nmf.reconstruction_err_))
        w_hat = get_w_hat(eigenspectra, fluxes, len(grid.grid_points))
        return cls(grid.grid_points, grid.wl, weights, eigenspectra, w_hat)

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
            raise ValueError('Cannot reshape the full_covariance matrix for many parameters.')

        if not self._trained:
            warnings.warn(
                'This emulator has not been trained and therefore is not reliable. call emulator.train() to train.')

        # If the pars is outside of the range of emulator values, raise a ModelError
        if np.any(params < self.min_params) or np.any(params > self.max_params):
            raise ValueError('Querying emulator outside of original parameter range.')

        # Do this according to R&W eqn 2.18, 2.19
        # Recalculate V12, V21, and V22.
        v12 = V12(self.grid_points, params, self.variances, self.lengthscales)
        v22 = V22(params, self.variances, self.lengthscales)

        # Recalculate the covariance
        mu = v12.T @ cho_solve(self.v11_cho, self.w_hat)
        cov = v22 - v12.T @ cho_solve(self.v11_cho, v12)
        if not full_cov:
            cov = np.diag(cov)
        if reinterpret_batch:
            mu = mu.reshape(-1, self.ncomps, order='F').squeeze()
            cov = cov.reshape(-1, self.ncomps, order='F').squeeze()
        return mu, cov

    def load_flux(self, params):
        """
        Interpolate a model given any parameters within the grid's parameter range using eigenspectrum reconstruction
        by sampling from the weight distributions.

        :param params: The parameters to sample at. Should have same length as ``grid["parname"]`` in ``config.yaml``
        :type: array_like
        """
        mu, cov = self(params, reinterpret_batch=False)
        weights = np.random.multivariate_normal(mu, cov).reshape(-1, self.ncomps)
        return weights @ self.eigenspectra

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
            self.log.debug('loss: {}\r'.format(loss))
            return loss

        soln = minimize(nll, P0, **opt_kwargs)

        # Extract hyper parameters
        self.set_param_vector(soln.x)
        self.log.debug(soln)

        self.log.info('Finished optimizing emulator hyperparameters')
        self.log.info('lambda_xi: {}'.format(self.lambda_xi))
        self.log.info('variances: {}'.format(self.variances))
        self.log.info('lengthscales: {}'.format(self.lengthscales))

        # Recalculate v11 given new parameters
        self.v11_cho = cho_factor(
            self.PhiPhi / self.lambda_xi + block_sigma(self.grid_points, self.variances, self.lengthscales))

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
        marks = np.abs(self.grid_points - np.expand_dims(params, 1)).sum(axis=-1)
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
        self.block_sigma = block_sigma(self.grid_points, self.variances, self.lengthscales) + self.jitter
        self.v11_cho = cho_factor(self.PhiPhi / self.lambda_xi + self.block_sigma)

    def log_likelihood(self):
        sigma = self.PhiPhi / self.lambda_xi + self.block_sigma
        L, flag = cho_factor(sigma)
        logdet = np.log(np.trace(L))
        central = self.w_hat.T @ cho_solve((L, flag), self.w_hat)
        return -0.5 * (logdet + central)

    def grad_log_likelihood(self):
        sigma = self.PhiPhi / self.lambda_xi + self.block_sigma
        sigma_adjugate = np.linalg.det(sigma) * np.linalg.inv(sigma)
        dsigma__dlambda_xi = -self.PhiPhi / self.lambda_xi ** 2

        d__lambda_xi = -0.5 * (np.trace(sigma_adjugate @ dsigma__dlambda_xi) +
                               self.w_hat.T @ np.linalg.solve(sigma, -dsigma__dlambda_xi) @
                               np.linalg.solve(sigma, self.w_hat))

        dkernel_dvariance = inverse_block_diag(block_sigma(self.grid_points, np.ones(self.ncomps),
                                                           self.lengthscales[n]))

