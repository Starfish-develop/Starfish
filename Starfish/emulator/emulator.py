import logging
import math
import os
import warnings
from collections import OrderedDict

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
    """
    A Bayesian spectral emulator.

    This emulator offers an interface to spectral libraries that offers interpolation while providing a variance-covariance matrix that can be forward-propagated in likelihood calculations. For more details, see the appendix from Czekala et al. (2015). 

    Parameters
    ----------
    grid_points : numpy.ndarray
        The parameter space from the library.
    param_names : array_like of str
        The names of each parameter from the grid
    wavelength : numpy.ndarray
        The wavelength of the library models
    weights : numpy.ndarray
        The PCA weights for the original grid points
    eigenspectra : numpy.ndarray
        The PCA components from the decomposition
    w_hat : numpy.ndarray
        The best-fit weights estimator
    flux_mean : numpy.ndarray
        The mean flux spectrum
    flux_std : numpy.ndarray
        The standard deviation flux spectrum
    lambda_xi : float, optional
        The scaling parameter for the augmented covariance calculations, default is 1
    variances : numpy.ndarray, optional
        The variance parameters for each of Gaussian process, default is array of 1s
    lengthscales : numpy.ndarray, optional
        The lengthscales for each Gaussian process, each row should have length equal to number of library parameters, default is arrays of 3 * the max grid separation for the grid_points
    name : str, optional
        If provided, will give a name to the emulator; useful for keeping track of filenames. Default is None.


    Attributes
    ----------
    bulk_fluxes : numpy.ndarray
        A vertically concatenated vector of the eigenspectra, flux_mean, and flux_std (in that order). Used for bulk processing with the emulator. 
    variances : numpy.ndarray
        The variances of each Gaussian process
    lengthscales : numpy.ndarray
        The lengthscales for each parameter for each Gaussian process
    params : OrderedDict
        The underlying hyperparameter dictionary
    """

    def __init__(self, grid_points, param_names, wavelength, weights, eigenspectra, w_hat, flux_mean, flux_std, lambda_xi=1, variances=None, lengthscales=None, name=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.grid_points = grid_points
        self.param_names = param_names
        self.wl = wavelength
        self.weights = weights
        self.eigenspectra = eigenspectra
        self.flux_mean = flux_mean
        self.flux_std = flux_std

        self.dv = calculate_dv(wavelength)
        self.ncomps = eigenspectra.shape[0]

        self.hyperparams = OrderedDict()
        self.name = name

        self.lambda_xi = lambda_xi

        self.variances = variances if variances is not None else 1e4 * \
            np.ones(self.ncomps)

        if lengthscales is None:
            unique = [sorted(np.unique(param_set))
                      for param_set in self.grid_points.T]
            sep = [3 * np.diff(param).max() for param in unique]
            lengthscales = np.tile(sep, (self.ncomps, 1))

        self.lengthscales = lengthscales

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

    @property
    def lambda_xi(self):
        return np.exp(self.hyperparams['log_lambda_xi'])

    @lambda_xi.setter
    def lambda_xi(self, value):
        self.hyperparams['log_lambda_xi'] = np.log(value)

    @property
    def variances(self):
        values = [val for key, val in self.hyperparams.items(
        ) if key.startswith('log_variance:')]
        return np.exp(values)

    @variances.setter
    def variances(self, values):
        for i, value in enumerate(values):
            self.hyperparams['log_variance:{}'.format(i)] = np.log(value)

    @property
    def lengthscales(self):
        values = [val for key, val in self.hyperparams.items(
        ) if key.startswith('log_lengthscale:')]
        return np.exp(values).reshape(self.ncomps, -1)

    @lengthscales.setter
    def lengthscales(self, values):
        for i, value in enumerate(values):
            for j, ls in enumerate(value):
                self.hyperparams['log_lengthscale:{}:{}'.format(i, j)] = np.log(ls)

    def __getitem__(self, key):
        return self.hyperparams[key]

    @classmethod
    def load(cls, filename):
        """
        Load an emulator from and HDF5 file

        Parameters
        ----------
        filename : str or path-like
        """
        filename = os.path.expandvars(filename)
        with h5py.File(filename) as base:
            grid_points = base['grid_points'][:]
            param_names = base['grid_points'].attrs['names']
            wavelength = base['wavelength'][:]
            weights = base['weights'][:]
            eigenspectra = base['eigenspectra'][:]
            flux_mean = base['flux_mean'][:]
            flux_std = base['flux_std'][:]
            w_hat = base['w_hat'][:]
            lambda_xi = base['hyperparameters']['lambda_xi'][()]
            variances = base['hyperparameters']['variances'][:]
            lengthscales = base['hyperparameters']['lengthscales'][:]
            trained = base.attrs['trained']
            if 'name' in base.attrs:
                name = base.attrs['name']
            else:
                name = '.'.join(filename.split('.')[:-1])

        emulator = cls(
            grid_points=grid_points,
            param_names=param_names,
            wavelength=wavelength,
            weights=weights,
            eigenspectra=eigenspectra,
            w_hat=w_hat,
            flux_mean=flux_mean,
            flux_std=flux_std,
            lambda_xi=lambda_xi,
            variances=variances,
            lengthscales=lengthscales,
            name=name
        )
        emulator._trained = trained
        return emulator

    def save(self, filename):
        """
        Save the emulator to an HDF5 file

        Parameters
        ----------
        filename : str or path-like
        """
        filename = os.path.expandvars(filename)
        with h5py.File(filename, 'w') as base:
            grid_points = base.create_dataset(
                'grid_points', data=self.grid_points, compression=9)
            grid_points.attrs['names'] = self.param_names
            waves = base.create_dataset(
                'wavelength', data=self.wl, compression=9)
            waves.attrs['units'] = 'Angstrom'
            base.create_dataset('weights', data=self.weights, compression=9)
            eigens = base.create_dataset(
                'eigenspectra', data=self.eigenspectra, compression=9)
            base.create_dataset(
                'flux_mean', data=self.flux_mean, compression=9)
            base.create_dataset('flux_std', data=self.flux_std, compression=9)
            eigens.attrs['units'] = 'erg/cm^2/s/Angstrom'
            base.create_dataset('w_hat', data=self.w_hat, compression=9)
            base.attrs['trained'] = self._trained
            if self.name is not None:
                base.attrs['name'] = self.name
            hp_group = base.create_group('hyperparameters')
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
            The keyword arguments to pass to PCA. By default, `n_components=0.99` and `svd_solver='full'`.

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

        # Perform PCA using sklearn
        default_pca_kwargs = dict(n_components=0.99, svd_solver='full')
        default_pca_kwargs.update(pca_kwargs)
        pca = PCA(**default_pca_kwargs)
        weights = pca.fit_transform(fluxes)
        eigenspectra = pca.components_

        exp_var = pca.explained_variance_ratio_.sum()
        # This is basically the mean square error of the reconstruction
        log.info(
            'PCA fit {:.2f}% of the variance with {:d} components.'.format(
                exp_var, pca.n_components_))
        w_hat = get_w_hat(eigenspectra, fluxes, len(grid.grid_points))

        emulator = cls(
            grid_points=grid.grid_points,
            param_names=grid.param_names,
            wavelength=grid.wl,
            weights=weights,
            eigenspectra=eigenspectra,
            w_hat=w_hat,
            flux_mean=flux_mean,
            flux_std=flux_std
        )
        return emulator

    def __call__(self, params, full_cov=True, reinterpret_batch=False):
        """
        Gets the mu and cov matrix for a given set of params

        Parameters
        ----------
        params : array_like
            The parameters to sample at. Should be consistent with the shapes of the original grid points.
        full_cov : bool, optional
            Return the full covariance or just the variance, default is True. This will have no effect of reinterpret_batch is true
        reinterpret_batch : bool, optional
            Will try and return a batch of output matrices if the input params are a list of params, default is False.

        Returns
        -------
        mu : numpy.ndarray (len(params),)
        cov : numpy.ndarray (len(params), len(params))

        Raises
        ------
        ValueError
            If full_cov and reinterpret_batch are True
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

        Parameters
        ----------
        params : array_like
            The parameters to sample at.

        Returns
        -------
        flux : numpy.ndarray
        """
        mu, cov = self(params, reinterpret_batch=False)
        weights = np.random.multivariate_normal(
            mu, cov).reshape(-1, self.ncomps)
        X = self.eigenspectra * self.flux_std
        return weights @ X + self.flux_mean

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
        **opt_kwargs
            Any arguments to pass to the optimizer. By default, `method='Nelder-Mead'` and `maxiter=10000`. 

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

        default_kwargs = {
            'method': 'Nelder-Mead',
            'options': {'maxiter': 10000}
        }
        default_kwargs.update(opt_kwargs)
        soln = minimize(nll, P0, **default_kwargs)

        if not soln.success:
            self.log.warning('Optimization did not succeed.')
            self.log.info(soln.message)
            # self.set_param_vector(P0)
        else:
            self.set_param_vector(soln.x)
            self._trained = True
            self.log.info('Finished optimizing emulator hyperparameters')
            self.log.info(self)

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

    def get_param_dict(self):
        """
        Gets the dictionary of parameters. This is the same as `Emulator.params`

        Returns
        -------
        dict
        """
        return self.hyperparams

    def set_param_dict(self, params):
        """
        Sets the parameters with a dictionary

        Parameters
        ----------
        params : dict
            The new parameters. If a key is present in ``self.frozen`` it will not be changed
        """
        for key, val in params.items():
            if key in self.hyperparams:
                self.hyperparams[key] = val

        self.v11 = self.iPhiPhi / self.lambda_xi + \
            batch_kernel(self.grid_points, self.grid_points,
                         self.variances, self.lengthscales)

    def get_param_vector(self):
        """
        Get a vector of the current trainable parameters of the emulator

        Returns
        -------
        numpy.ndarray
        """
        values = list(self.get_param_dict().values())
        return np.array(values)

    def set_param_vector(self, params):
        """
        Set the current trainable parameters given a vector. Must have the same form as :meth:`get_param_vector`

        Parameters
        ----------
        params : numpy.ndarray
        """
        parameters = self.get_param_dict()
        if len(params) != len(parameters):
            raise ValueError(
                'params must match length of parameters (get_param_vector())')
        for i, key in enumerate(parameters):
            self.hyperparams[key] = params[i]

        self.v11 = self.iPhiPhi / self.lambda_xi + \
            batch_kernel(self.grid_points, self.grid_points,
                         self.variances, self.lengthscales)

    def log_likelihood(self):
        """
        Get the log likelihood of the emulator in its current state as calculated in the appendix of Czekala et al. (2015)

        Returns
        -------
        float
        """
        L, flag = cho_factor(self.v11)
        logdet = 2 * np.sum(np.log(np.diag(L)))
        central = self.w_hat.T @ cho_solve((L, flag), self.w_hat)
        return -0.5 * (logdet + central)

    def grad_log_likelihood(self):
        raise NotImplementedError('Pull requests welcome!')

    def __repr__(self):
        output = 'Emulator\n'
        output += '-' * 8 + '\n'
        if self.name is not None:
            output += 'Name: {}\n'.format(self.name)
        output += 'Trained: {}\n'.format(self._trained)
        output += 'lambda_xi: {:.3f}\n'.format(
            np.exp(self.lambda_xi))
        output += 'Variances:\n'
        output += '\n'.join(['\t{:.2f}'.format(v) for v in self.variances])
        output += '\nLengthscales:\n'
        output += '\n'.join(
            ['\t[ ' + ' '.join(
                ['{:.2f} '.format(l) for l in ls]
            ) + ']' for ls in self.lengthscales]
        )
        output += '\nLog Likelihood: {:.2f}\n'.format(self.log_likelihood())
        return output
