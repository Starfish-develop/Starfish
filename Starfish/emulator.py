import math
import os
import multiprocessing as mp

import numpy as np
import h5py
from sklearn.decomposition import PCA
from scipy.optimize import fmin
import emcee

import Starfish
from Starfish.grid_tools import HDF5Interface, determine_chunk_log
from Starfish.covariance import Sigma, sigma, V12, V22, V12m, V22m
from Starfish import constants as C


def Phi(eigenspectra, M):
    '''
    Warning: for any spectra of real-world dimensions, this routine will
    likely over flow memory.

    :param eigenspectra:
    :type eigenspectra: 2D array
    :param M: number of spectra in the synthetic library
    :type M: int
    Calculate the matrix Phi using the kronecker products.
    '''

    return np.hstack([np.kron(np.eye(M), eigenspectrum[np.newaxis].T) for eigenspectrum in eigenspectra])


def get_w_hat(eigenspectra, fluxes, M):
    '''
    Since we will overflow memory if we actually calculate Phi, we have to
    determine w_hat in a memory-efficient manner.

    '''
    m = len(eigenspectra)
    out = np.empty((M * m,))
    for i in range(m):
        for j in range(M):
            out[i * M + j] = eigenspectra[i].T.dot(fluxes[j])

    PhiPhi = np.linalg.inv(skinny_kron(eigenspectra, M))

    return PhiPhi.dot(out)


def skinny_kron(eigenspectra, M):
    '''
    Compute Phi.T.dot(Phi) in a memory efficient manner.

    eigenspectra is a list of 1D numpy arrays.
    '''
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


def Gprior(x, s, r):
    return r ** s * x ** (s - 1) * np.exp(- x * r) / math.gamma(s)


def Glnprior(x, s, r):
    return s * np.log(r) + (s - 1.) * np.log(x) - r * x - math.lgamma(s)


class PCAGrid:
    '''
    Create and query eigenspectra.
    '''

    def __init__(self, wl, dv, flux_mean, flux_std, eigenspectra, w, w_hat, gparams):
        '''

        :param wl: wavelength array
        :type wl: 1D np.array
        :param dv: delta velocity
        :type dv: float
        :param flux_mean: mean flux spectrum
        :type flux_mean: 1D np.array
        :param flux_std: standard deviation flux spectrum
        :type flux_std: 1D np.array
        :param eigenspectra: the principal component eigenspectra
        :type eigenspectra: 2D np.array
        :param w: weights to reproduce any spectrum in the original grid
        :type w: 2D np.array (m, M)
        :param w_hat: maximum likelihood estimator of the grid weights
        :type w_hat: 2D np.array (m, M)
        :param gparams: The stellar parameters of the synthetic library
        :type gparams: 2D array of parameters (nspec, nparam)

        '''
        self.wl = wl
        self.dv = dv
        self.flux_mean = flux_mean
        self.flux_std = flux_std
        self.eigenspectra = eigenspectra
        self.m = len(self.eigenspectra)
        self.w = w
        self.w_hat = w_hat
        self.gparams = gparams

        self.npix = len(self.wl)
        self.M = self.w.shape[1]  # The number of spectra in the synthetic grid

    @classmethod
    def create(cls, interface, ncomp=None):
        '''
        Create a PCA grid object from a synthetic spectral library, with
        configuration options specified in a dictionary.

        :param interface: HDF5Interface containing the instrument-processed spectra.
        :type interface: HDF5Interface
        :param ncomp: number of eigenspectra to keep
        :type ncomp: int

        '''

        wl = interface.wl
        dv = interface.dv

        npix = len(wl)

        # number of spectra in the synthetic library
        M = len(interface.grid_points)

        fluxes = np.empty((M, npix))

        for i, spec in enumerate(interface.fluxes):
            fluxes[i] = spec

        # Normalize all of the fluxes to an average value of 1
        # In order to remove uninteresting correlations
        fluxes = fluxes / np.average(fluxes, axis=1)[np.newaxis].T

        # Subtract the mean from all of the fluxes.
        flux_mean = np.average(fluxes, axis=0)
        fluxes -= flux_mean

        # "Whiten" each spectrum such that the variance for each wavelength is 1
        flux_std = np.std(fluxes, axis=0)
        fluxes /= flux_std

        # Use the scikit-learn PCA module
        # Automatically select enough components to explain > threshold (say
        # 0.99, or 99%) of the variance.
        pca = PCA(n_components=Starfish.PCA["threshold"], svd_solver='full')
        pca.fit(fluxes)
        comp = pca.transform(fluxes)
        components = pca.components_
        mean = pca.mean_
        variance_ratio = np.sum(pca.explained_variance_ratio_)

        if ncomp is None:
            ncomp = len(components)

        print("found {} components explaining {}% of the" \
              " variance (threshold was {}%)".format(ncomp, 100 * variance_ratio, 100 * Starfish.PCA["threshold"]))

        print("Shape of PCA components {}".format(components.shape))

        if not np.allclose(mean, np.zeros_like(mean)):
            import sys
            sys.exit("PCA mean is more than just numerical noise. Something's wrong!")
            # Otherwise, the PCA mean is just numerical noise that we can ignore.

        print("Keeping only the first {} components".format(ncomp))
        eigenspectra = components[:ncomp]

        gparams = interface.grid_points

        # Create w, the weights corresponding to the synthetic grid
        w = np.empty((ncomp, M))
        for i, pcomp in enumerate(eigenspectra):
            for j, spec in enumerate(fluxes):
                w[i, j] = np.sum(pcomp * spec)

        # Calculate w_hat, Eqn 20 Habib
        w_hat = get_w_hat(eigenspectra, fluxes, M)

        return cls(wl, dv, flux_mean, flux_std, eigenspectra, w, w_hat, gparams)

    def write(self, filename=Starfish.PCA["path"]):
        '''
        Write the PCA decomposition to an HDF5 file.

        :param filename: name of the HDF5 to create.
        :type filename: str

        '''

        filename = os.path.expandvars(filename)
        hdf5 = h5py.File(filename, "w")

        hdf5.attrs["dv"] = self.dv

        # Store the eigenspectra plus the wavelength, mean, and std arrays.
        pdset = hdf5.create_dataset("eigenspectra", (self.m + 3, self.npix),
                                    compression='gzip', dtype="f8", compression_opts=9)
        pdset[0, :] = self.wl
        pdset[1, :] = self.flux_mean
        pdset[2, :] = self.flux_std
        pdset[3:, :] = self.eigenspectra

        wdset = hdf5.create_dataset("w", (self.m, self.M), compression='gzip',
                                    dtype="f8", compression_opts=9)
        wdset[:] = self.w

        w_hatdset = hdf5.create_dataset("w_hat", (self.m * self.M,), compression='gzip', dtype="f8",
                                        compression_opts=9)
        w_hatdset[:] = self.w_hat

        gdset = hdf5.create_dataset("gparams", (self.M, len(Starfish.parname)), compression='gzip', dtype="f8",
                                    compression_opts=9)
        gdset[:] = self.gparams

        hdf5.close()

    @classmethod
    def open(cls, filename=Starfish.PCA["path"]):
        '''
        Initialize an object using the PCA already stored to an HDF5 file.

        :param filename: filename of an HDF5 object containing the PCA components.
        :type filename: str
        '''

        filename = os.path.expandvars(filename)
        hdf5 = h5py.File(filename, "r")
        pdset = hdf5["eigenspectra"]

        dv = hdf5.attrs["dv"]

        wl = pdset[0, :]

        flux_mean = pdset[1, :]
        flux_std = pdset[2, :]
        eigenspectra = pdset[3:, :]

        wdset = hdf5["w"]
        w = wdset[:]

        w_hatdset = hdf5["w_hat"]
        w_hat = w_hatdset[:]

        gdset = hdf5["gparams"]
        gparams = gdset[:]

        pcagrid = cls(wl, dv, flux_mean, flux_std, eigenspectra, w, w_hat, gparams)
        hdf5.close()

        return pcagrid

    def determine_chunk_log(self, wl_data, buffer=Starfish.grid["buffer"]):
        '''
        Possibly truncate the wl, eigenspectra, and flux_mean and flux_std in
        response to some data.

        :param wl_data: The spectrum dataset you want to fit.
        :type wl_data: np.array

        '''

        # determine the indices
        wl_min, wl_max = np.min(wl_data), np.max(wl_data)

        wl_min -= buffer
        wl_max += buffer

        ind = determine_chunk_log(self.wl, wl_min, wl_max)

        assert (min(self.wl[ind]) <= wl_min) and (max(self.wl[ind]) >= wl_max), \
            "PCA/emulator chunking ({:.2f}, {:.2f}) didn't encapsulate " \
            "full wl range ({:.2f}, {:.2f}).".format(min(self.wl[ind]), \
                                                     max(self.wl[ind]), wl_min, wl_max)

        self.wl = self.wl[ind]
        self.npix = len(self.wl)
        self.eigenspectra = self.eigenspectra[:, ind]
        self.flux_mean = self.flux_mean[ind]
        self.flux_std = self.flux_std[ind]

    def get_index(self, params):
        '''
        Given a np.array of stellar params (corresponding to a grid point),
        deliver the index that corresponds to the
        entry in the fluxes, list_grid_points, and weights
        '''
        return np.sum(np.abs(self.gparams - params), axis=1).argmin()

    def get_weights(self, params):
        '''
        Given a np.array of parameters (corresponding to a grid point),
        deliver the weights that reconstruct this spectrum out of eigenspectra.

        '''

        ii = self.get_index(params)
        return self.w[:, ii]

    def reconstruct(self, weights):
        '''
        Reconstruct a spectrum given some weights.

        Also correct for original scaling.
        '''

        f = np.empty((self.m, self.npix))
        for i, (pcomp, weight) in enumerate(zip(self.eigenspectra, weights)):
            f[i, :] = pcomp * weight
        return np.sum(f, axis=0) * self.flux_std + self.flux_mean

    def reconstruct_all(self):
        '''
        Return a (M, npix) array with all of the spectra reconstructed.
        '''
        recon_fluxes = np.empty((self.M, self.npix))
        for i in range(self.M):
            f = np.empty((self.m, self.npix))
            for j, (pcomp, weight) in enumerate(zip(self.eigenspectra, self.w[:, i])):
                f[j, :] = pcomp * weight
            recon_fluxes[i, :] = np.sum(f, axis=0) * self.flux_std + self.flux_mean

        return recon_fluxes

    def optimize(self, method='fmin'):
        """
        Optimize the emulator and train the Gaussian Processes (GP) that will serve as interpolators.
        For more explanation about the choice of Gaussian Process covariance functions and the design of the emulator, see the appendix of our paper.

        :param method: The method to use for optimization; one of ['fmin', 'emcee'].
        :type method: string
        """
        self.PhiPhi = np.linalg.inv(skinny_kron(self.eigenspectra, self.M))

        if method == 'fmin':
            eparams = self._optimize_fmin()
        elif method == 'emcee':
            eparams = self._optimize_emcee()
        else:
            raise ValueError("Did not provide valid method, please choose from ['fmin', 'emceee']")

        filename = os.path.expandvars(Starfish.PCA["path"])
        hdf5 = h5py.File(filename, "r+")

        # check to see whether the dataset already exists
        if "eparams" in hdf5.keys():
            pdset = hdf5["eparams"]
        else:
            pdset = hdf5.create_dataset("eparams", eparams.shape, compression="gzip", compression_opts=9, dtype="f8")

        pdset[:] = eparams
        hdf5.close()

    def _lnprob(self, p, priors, fmin=False):
        '''
        The log-likelihood method for use in optimization
        :param p: Gaussian Processes hyper-parameters
        :type p: 1D np.array
        Calculate the lnprob using Habib's posterior formula for the emulator.
        '''

        # We don't allow negative parameters.
        if np.any(p < 0.):
            if fmin:
                return 1e99
            else:
                return -np.inf

        lambda_xi = p[0]
        hparams = p[1:].reshape((self.m, -1))

        # Calculate the prior for parname variables
        # We have two separate sums here, since hparams is a 2D array
        # hparams[:, 0] are the amplitudes, so we index i+1 here
        lnpriors = 0.0
        for i in range(len(Starfish.parname)):
            lnpriors += np.sum(Glnprior(hparams[:, i + 1], *priors[i]))

        h2params = hparams ** 2
        # Fold hparams into the new shape
        Sig_w = Sigma(self.gparams, h2params)

        C = (1. / lambda_xi) * self.PhiPhi + Sig_w

        sign, pref = np.linalg.slogdet(C)

        central = self.w_hat.T.dot(np.linalg.solve(C, self.w_hat))

        lnp = -0.5 * (pref + central + self.M * self.m * np.log(2. * np.pi)) + lnpriors

        # Negate this when using the fmin algorithm
        if fmin:
            lnp *= -1

        return lnp

    def _optimize_fmin(self):
        """
        Optimize the emulator using the downhill simplex algorithm from `numpy.optimize.fmin`
        """
        priors = Starfish.PCA["priors"]
        amp = 100.
        # Use the mean of the gamma distribution to start
        eigpars = np.array([amp] + [s / r for s, r in priors])
        p0 = np.hstack((np.array([1., ]),  # lambda_xi
                        np.hstack([eigpars for _ in range(self.m)])))


        func = lambda p: self._lnprob(p, priors, fmin=True)
        result = fmin(func, p0, maxiter=10000, maxfun=10000)
        return result

    def _optimize_emcee(self, nburn=100, nsamples=100):
        """
        Optimize the emulator using monte carlo sampling from `emcee`
        """
        priors = Starfish.PCA["priors"]
        ndim = 1 + (1 + len(Starfish.parname)) * self.m
        nwalkers = 4 * ndim  # about the minimum per dimension we can get by with

        # Assemble p0 based off either a guess or the previous state of walkers
        p0 = []
        # p0 is a (nwalkers, ndim) array
        amp = [10.0, 150]

        p0.append(np.random.uniform(0.01, 1.0, nwalkers))
        for i in range(self.m):
            p0 += [np.random.uniform(amp[0], amp[1], nwalkers)]
            for s, r in priors:
                # Draw randomly from the gamma priors
                p0 += [np.random.gamma(s, 1. / r, nwalkers)]

        p0 = np.array(p0).T

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob, threads=mp.cpu_count())

        # burn in
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        print("Burned in")

        # actual run
        pos, prob, state = sampler.run_mcmc(pos, nsamples)

        return np.median(sampler.flatchain, axis=0)


class Emulator:
    def __init__(self, pca, eparams):
        '''
        Provide the emulation products.

        :param pca: object storing the principal components, the eigenpsectra
        :type pca: PCAGrid
        :param eparams: Optimized GP hyperparameters.
        :type eparams: 1D np.array
        '''

        self.pca = pca
        self.lambda_xi = eparams[0]
        self.h2params = eparams[1:].reshape(self.pca.m, -1) ** 2

        # Determine the minimum and maximum bounds of the grid
        self.min_params = np.min(self.pca.gparams, axis=0)
        self.max_params = np.max(self.pca.gparams, axis=0)

        # self.eigenspectra = self.PCAGrid.eigenspectra
        self.dv = self.pca.dv
        self.wl = self.pca.wl

        self.iPhiPhi = (1. / self.lambda_xi) * np.linalg.inv(skinny_kron(self.pca.eigenspectra, self.pca.M))

        self.V11 = self.iPhiPhi + Sigma(self.pca.gparams, self.h2params)

        self._params = None  # Where we want to interpolate

        self.V12 = None
        self.V22 = None
        self.mu = None
        self.sig = None

    @classmethod
    def open(cls, filename=Starfish.PCA["path"]):
        '''
        Create an Emulator object from an HDF5 file.
        '''
        # Create the PCAGrid from this filename
        filename = os.path.expandvars(filename)
        pcagrid = PCAGrid.open(filename)
        hdf5 = h5py.File(filename, "r")

        eparams = hdf5["eparams"][:]
        hdf5.close()
        return cls(pcagrid, eparams)

    def determine_chunk_log(self, wl_data):
        '''
        Possibly truncate the wl grid in response to some data. Also truncate eigenspectra, and flux_mean and flux_std.
        '''
        self.pca.determine_chunk_log(wl_data)
        self.wl = self.pca.wl

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, pars):

        # If the pars is outside of the range of emulator values, raise a ModelError
        if np.any(pars < self.min_params) or np.any(pars > self.max_params):
            raise C.ModelError("Querying emulator outside of original PCA parameter range.")

        # Assumes pars is a single parameter combination, as a 1D np.array
        self._params = pars

        # Do this according to R&W eqn 2.18, 2.19

        # Recalculate V12, V21, and V22.
        self.V12 = V12(self._params, self.pca.gparams, self.h2params, self.pca.m)
        self.V22 = V22(self._params, self.h2params, self.pca.m)

        # Recalculate the covariance
        self.mu = self.V12.T.dot(np.linalg.solve(self.V11, self.pca.w_hat))
        self.mu.shape = (-1)
        self.sig = self.V22 - self.V12.T.dot(np.linalg.solve(self.V11, self.V12))

    @property
    def matrix(self):
        return (self.mu, self.sig)

    def draw_many_weights(self, params):
        '''
        :param params: multiple parameters to produce weight draws at.
        :type params: 2D np.array
        '''

        # Local variables, different from instance attributes
        v12 = V12m(params, self.pca.gparams, self.h2params, self.pca.m)
        v22 = V22m(params, self.h2params, self.pca.m)

        mu = v12.T.dot(np.linalg.solve(self.V11, self.pca.w_hat))
        sig = v22 - v12.T.dot(np.linalg.solve(self.V11, v12))

        weights = np.random.multivariate_normal(mu, sig)

        # Reshape these weights into a 2D matrix
        weights.shape = (len(params), self.pca.m)

        return weights

    def draw_weights(self):
        '''
        Using the current settings, draw a sample of PCA weights
        '''

        if self.V12 is None:
            print("No parameters are set, yet. Must set parameters first.")
            return

        return np.random.multivariate_normal(self.mu, self.sig)

    def reconstruct(self):
        '''
        Reconstructing a spectrum using a random draw of weights. In this case,
        we are making the assumption that we are always drawing a weight at a
        single stellar value.
        '''

        weights = self.draw_weights()
        return self.pca.reconstruct(weights)


