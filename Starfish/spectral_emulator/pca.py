import os
import multiprocessing as mp

from sklearn.decomposition import PCA
from scipy.optimize import minimize
import scipy.stats as stats
import numpy as np
import emcee
import h5py

import Starfish
from Starfish.grid_tools import HDF5Interface, determine_chunk_log
from Starfish.covariance import Sigma


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


def _lnprior(x, s, r):
    return stats.gamma.logpdf(x, s, scale=1 / r)


def _prior(x, s, r):
    return np.exp(_lnprior(x, s, r))


def _lnprob(p, PCA, minim=False, verbose=False):
    """
    Calculate the lnprob using Habib's posterior formula for the emulator.
    """
    # We don't allow negative parameters.
    if np.any(p < 0.):
        if minim:
            return 1e99
        else:
            return -np.inf

    lambda_xi = p[0]
    # Fold hparams into the new shape
    hparams = p[1:].reshape((PCA.m, -1))

    # Calculate the prior for parname variables
    # We have two separate sums here, since hparams is a 2D array
    # hparams[:, 0] are the amplitudes, so we index i+1 here
    lnpriors = 0.0
    for i in range(len(Starfish.parname)):
        a, r = PCA.priors[i]
        lnpriors += np.sum(_lnprior(hparams[:, i + 1], a, r))

    h2params = hparams ** 2
    Sig_w = Sigma(PCA.gparams, h2params)
    C = (1. / lambda_xi) * PCA.PhiPhi + Sig_w
    sign, pref = np.linalg.slogdet(C)
    central = PCA.w_hat.T.dot(np.linalg.solve(C, PCA.w_hat))
    lnp = -0.5 * (pref + central + PCA.M * PCA.m * np.log(2. * np.pi)) + lnpriors

    if verbose:
        print("lam_xi = {}".format(p[0]))
        print("Hyper Parameters:")
        [print("w{}: {}".format(i, ps)) for i, ps in enumerate(hparams[:, 1:])]
        print("logl = {}".format(lnp), end='\n\n')

    # Negate this when using the fmin algorithm
    if minim:
        lnp *= -1
    return lnp


class PCAGrid:
    """
    Create and query eigenspectra.
    """

    def __init__(self, wl, dv, flux_mean, flux_std, eigenspectra, w, w_hat, gparams, filename):
        """

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
        :param filename: The filename associated with this PCA Grid
        :type filename: str

        """
        self.wl = wl
        self.dv = dv
        self.flux_mean = flux_mean
        self.flux_std = flux_std
        self.eigenspectra = eigenspectra
        self.m = len(self.eigenspectra)
        self.w = w
        self.w_hat = w_hat
        self.gparams = gparams
        self.filename = filename

        self.npix = len(self.wl)
        self.M = self.w.shape[1]  # The number of spectra in the synthetic grid
        self.PhiPhi = np.linalg.inv(skinny_kron(self.eigenspectra, self.M))
        self.priors = Starfish.PCA["priors"]


    @classmethod
    def create(cls, interface, ncomp=None):
        """
        Create a PCA grid object from a synthetic spectral library, with
        configuration options specified in a dictionary.

        :param interface: HDF5Interface containing the instrument-processed spectra.
        :type interface: HDF5Interface
        :param ncomp: number of eigenspectra to keep. Default is to keep all.
        :type ncomp: int

        """

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
        components = pca.components_
        mean = pca.mean_
        variance_ratio = np.sum(pca.explained_variance_ratio_)

        if ncomp is None:
            ncomp = len(components)

        print("found {} components explaining {:.2f}% of the"
              " variance (threshold was {:.2f}%)".format(ncomp, 100 * variance_ratio, 100 * Starfish.PCA["threshold"]))

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

        return cls(wl, dv, flux_mean, flux_std, eigenspectra, w, w_hat, gparams, Starfish.PCA["path"])

    def write(self):
        """
        Write the PCA decomposition to an HDF5 file.

        :param filename: name of the HDF5 to create. Defaults to the ``PCA["path"]`` in ``config.yaml``.
        :type filename: str

        """

        hdf5 = h5py.File(self.filename, "w")

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
        """
        Initialize an object using the PCA already stored to an HDF5 file.

        :param filename: filename of an HDF5 object containing the PCA components. Defaults to the
            ``PCA["path"]`` in ``config.yaml``.
        :type filename: str
        """

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

        pcagrid = cls(wl, dv, flux_mean, flux_std, eigenspectra, w, w_hat, gparams, filename)
        hdf5.close()

        return pcagrid

    def determine_chunk_log(self, wl_data, buffer=Starfish.grid["buffer"]):
        """
        Possibly truncate the wl, eigenspectra, and flux_mean and flux_std in
        response to some data.

        :param wl_data: The spectrum dataset you want to fit.
        :type wl_data: np.array
        :param buffer: The length (in Angstrom) of the wavelength buffer at the edges of the spectrum.
            Defaults to the value specified in ``grid["buffer"]`` in ``config.yaml``.
        :type buffer: float
        """

        # determine the indices
        wl_min, wl_max = np.min(wl_data), np.max(wl_data)

        wl_min -= buffer
        wl_max += buffer

        ind = determine_chunk_log(self.wl, wl_min, wl_max)

        assert (min(self.wl[ind]) <= wl_min) and (max(self.wl[ind]) >= wl_max), \
            "PCA/emulator chunking ({:.2f}, {:.2f}) didn't encapsulate " \
            "full wl range ({:.2f}, {:.2f}).".format(min(self.wl[ind]),
                                                     max(self.wl[ind]), wl_min, wl_max)

        self.wl = self.wl[ind]
        self.npix = len(self.wl)
        self.eigenspectra = self.eigenspectra[:, ind]
        self.flux_mean = self.flux_mean[ind]
        self.flux_std = self.flux_std[ind]

    def get_index(self, params):
        """
        Given a list of stellar parameters (corresponding to a grid point),
        deliver the index that corresponds to the
        entry in the fluxes, list_grid_points, and weights.

        :param params: The parameters at a grid point. It should have the same length as the parameters in ``grid[
            "parnames"]`` in ``config.yaml``
        :type params: iterable
        """
        params = np.array(params)
        return np.sum(np.abs(self.gparams - params), axis=1).argmin()

    def get_weights(self, params):
        """
        Given a list of parameters (corresponding to a grid point),
        deliver the weights that reconstruct this spectrum out of eigenspectra.

        :param params: The parameters at a grid point. It should have the same length as the parameters in ``grid[
            "parnames"]`` in ``config.yaml``
        :type params: iterable
        """

        params = np.array(params)
        ii = self.get_index(params)
        return self.w[:, ii]

    def reconstruct(self, weights):
        """
        Reconstruct a spectrum given some weights.

        Also correct for original scaling.

        :param weights: THe weights for reconstructing a spectrum
        :type weights: NDarray
        :return: The reconstructed spectrum
        :rtype: NDarray
        """

        f = np.empty((self.m, self.npix))
        for i, (pcomp, weight) in enumerate(zip(self.eigenspectra, weights)):
            f[i, :] = pcomp * weight
        return np.sum(f, axis=0) * self.flux_std + self.flux_mean

    def reconstruct_all(self):
        """
        Return a (M, npix) array with all of the spectra reconstructed.
        """
        recon_fluxes = np.empty((self.M, self.npix))
        for i in range(self.M):
            f = np.empty((self.m, self.npix))
            for j, (pcomp, weight) in enumerate(zip(self.eigenspectra, self.w[:, i])):
                f[j, :] = pcomp * weight
            recon_fluxes[i, :] = np.sum(f, axis=0) * self.flux_std + self.flux_mean

        return recon_fluxes

    @property
    def eparams(self):
        with h5py.File(self.filename, "r+") as hdf5:
            try:
                return hdf5["eparams"][:]
            except:
                raise AttributeError("The grid has not been optimized. Please use `PCAGrid.optimize` before trying to "
                                     "access eparams.")

    @eparams.setter
    def eparams(self, params):
        with h5py.File(self.filename, "r+") as hdf5:
            if "eparams" in hdf5:
                hdf5["eparams"][:] = params
            else:
                hdf5.create_dataset("eparams", data=params, compression="gzip", compression_opts=9)

    @property
    def emcee_chain(self):
        with h5py.File(self.filename, "r+") as hdf5:
            try:
                return hdf5["emcee"]["chain"][:]
            except:
                raise AttributeError("There are no values stored for emcee. Make sure you have optimized using the "
                                     "emcee method before trying to access.")

    @emcee_chain.setter
    def emcee_chain(self, chain):
        with h5py.File(self.filename, "r+") as hdf5:
            if "emcee" in hdf5:
                emcee_group = hdf5["emcee"]
            else:
                emcee_group = hdf5.create_group("emcee")

            if "chain" in emcee_group:
                emcee_group["chain"][:] = chain
            else:
                emcee_group.create_dataset("chain", data=chain, compression="gzip", compression_opts=9)

    @property
    def emcee_walkers(self):
        with h5py.File(self.filename, "r+") as hdf5:
            try:
                return hdf5["emcee"]["walkers"][:]
            except:
                raise AttributeError("There are no values stored for emcee. Make sure you have optimized using the "
                                     "emcee method before trying to access.")

    @emcee_walkers.setter
    def emcee_walkers(self, pos):
        with h5py.File(self.filename, "r+") as hdf5:
            if "emcee" in hdf5:
                emcee_group = hdf5["emcee"]
            else:
                emcee_group = hdf5.create_group("emcee")

            if "walkers" in emcee_group:
                emcee_group["walkers"][:] = pos
            else:
                emcee_group.create_dataset("walkers", data=pos, compression="gzip", compression_opts=9)

    def optimize(self, method='min', **fit_kwargs):
        """
        Optimize the emulator and train the Gaussian Processes (GP) that will serve as interpolators.
        For more explanation about the choice of Gaussian Process covariance functions and the design of the emulator,
        see the appendix of our paper.

        :param method: The method to use for optimization; one of ['min', 'emcee'].
        :type method: string
        :param fit_kwargs: The keyword arguments to pass to the optimization methods.
        :type fit_kwargs: dict

        Example optimizing using minimization optimizer

        .. code-block:: python

            from Starfish.emulator import PCAGrid

            # Assuming you have already generated the initial PCA file
            pca = PCAGrid.open()
            pca.optimize()

        Example using the emcee optimizer

        .. code-block:: python

            from Starfish.emulator import PCAGrid

            # Assuming you have already generated the inital PCA file
            pca = PCAGrid.open()
            pca.optimize(method='emcee', nburn=100, nsamples=400)

        .. warning::
            This optimization may take a very long time to run (multiple hours). We recommend running the code on a
            server and running it in the background. For each PCAGrid you only have to optimize once, thankfully.
        """
        print('Starting optimization...')
        if method == 'min':
            self._optimize_min()
        elif method == 'emcee':
            self._optimize_emcee(**fit_kwargs)
        else:
            raise ValueError("Did not provide valid method, please choose from ['min', 'emceee']")
        print('Parameters Found')

    def _optimize_min(self):
        """
        Optimize the emulator using the downhill simplex algorithm from `numpy.optimize.fmin`
        """
        amp = 100.
        # Use the mean of the gamma distribution to start
        eigpars = np.array([amp] + [s / r for s, r in self.priors])
        p0 = np.hstack((np.array([1., ]),  # lambda_xi
                        np.hstack([eigpars for _ in range(self.m)])))

        result = minimize(_lnprob, p0, args=(self, True, True))

        self.eparams = result.x

    def _optimize_emcee(self, resume=False, max_samples=200):
        """
        Optimize the emulator using monte carlo sampling from `emcee`
        """
        ndim = 1 + (1 + len(Starfish.parname)) * self.m
        nwalkers = 4 * ndim  # about the minimum per dimension we can get by with

        # Assemble p0 based off either a guess or the previous state of walkers
        p0 = []
        # p0 is a (nwalkers, ndim) array
        amp = [10.0, 150]

        p0.append(np.random.uniform(0.01, 1.0, nwalkers))
        for i in range(self.m):
            p0 += [np.random.uniform(amp[0], amp[1], nwalkers)]
            for s, r in self.priors:
                # Draw randomly from the gamma priors
                p0 += [np.random.gamma(s, 1. / r, nwalkers)]

        p0 = np.array(p0).T
        filename = 'emcee_progress.hdf5'
        backend = emcee.backends.HDFBackend(filename)

        # If we want to start from scratch need to reset backend
        if resume:
            max_samples -= len(backend.get_chain(flat=True))
        else:
            backend.reset(nwalkers, ndim)

        p = mp.Pool()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _lnprob, args=(self, False, False), backend=backend, pool=p)

        # Set up loop for auto-checking the correlation
        old_tau = np.inf
        autocorr = np.empty(max_samples)

        # Using thin_by=5 to only calculate autocorr every 5 samples
        for sample in sampler.sample(p0, iterations=max_samples, thin_by=5, progress=True):
            # Get the autocorrelation time. tol=0 avoids getting an error on call
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[sampler.iteration] = np.mean(tau)
            # Check if chain is longer than 50 times the autocorr time
            converged = np.all(tau * 50 < sampler.iteration)
            # Check if tau has changed by more than 5% since last check
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            if converged:
                break
            old_tau = tau

        p.close()
        samples = sampler.get_chain(flat=True)

        self.eparams = np.median(samples, axis=0)
        self.emcee_chain = samples
