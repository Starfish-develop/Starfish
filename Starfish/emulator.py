import math
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from Starfish.grid_tools import HDF5Interface, determine_chunk_log
import Starfish.em_cov as em
import h5py
from Starfish import constants as C

class PCAGrid:

    def __init__(self, wl, min_v, flux_mean, flux_std, pcomps, w, gparams):
        self.wl = wl
        self.min_v = min_v
        self.flux_mean = flux_mean
        self.flux_std = flux_std
        self.pcomps = pcomps
        self.ncomp = len(self.pcomps)
        self.w = w
        self.gparams = gparams

        self.npix = len(self.wl)
        self.m = self.w.shape[1]

    @classmethod
    def open(cls, filename):
        '''
        :param filename: filename of an HDF5 object containing the PCA components.
        '''

        hdf5 = h5py.File(filename, "r")
        pdset = hdf5["pcomps"]

        min_v = hdf5.attrs["min_v"]

        wl = pdset[0,:]

        flux_mean = pdset[1,:]
        flux_std = pdset[2,:]
        pcomps = pdset[3:,:]

        wdset = hdf5["w"]
        w = wdset[:]

        gdset = hdf5["gparams"]
        gparams = gdset[:]

        pcagrid = cls(wl, min_v, flux_mean, flux_std, pcomps, w, gparams)
        hdf5.close()

        return pcagrid

    @classmethod
    def from_cfg(cls, cfg):
        '''
        :param cfg: dictionary containing the parameters.
        '''

        grid = HDF5Interface(cfg["grid"], ranges=cfg["ranges"])
        wl = grid.wl
        min_v = grid.wl_header["min_v"]

        if 'wl' in cfg:
            low, high = cfg['wl']
            ind = determine_chunk_log(wl, low, high) #Sets the wavelength vector using a power of 2
            wl = wl[ind]
        else:
            ind = np.ones_like(wl, dtype="bool")

        npix = len(wl)
        m = len(grid.list_grid_points)
        test_index = cfg['test_index']

        if test_index < m:
            #If the index actually corresponds to a spectrum in the grid, we're dropping it out. Otherwise,
            #leave it in by simply setting test_index to something larger than the number of spectra in the grid.
            m -= 1

        fluxes = np.empty((m, npix))

        z = 0
        for i, spec in enumerate(grid.fluxes):
            if i == test_index:
                test_spectrum = spec[ind]
                continue
            fluxes[z,:] = spec[ind]
            z += 1

        #Normalize all of the fluxes to an average value of 1
        #In order to remove interesting correlations
        fluxes = fluxes/np.average(fluxes, axis=1)[np.newaxis].T

        #Subtract the mean from all of the fluxes.
        flux_mean = np.average(fluxes, axis=0)
        fluxes -= flux_mean

        #"Whiten" each spectrum such that the variance for each wavelength is 1
        flux_std = np.std(fluxes, axis=0)
        fluxes /= flux_std

        pca = PCA()
        pca.fit(fluxes)
        comp = pca.transform(fluxes)
        components = pca.components_
        mean = pca.mean_
        print("Shape of PCA components {}".format(components.shape))

        if not np.allclose(mean, np.zeros_like(mean)):
            import sys
            sys.exit("PCA mean is more than just numerical noise. Something's wrong!")

            #Otherwise, the PCA mean is just numerical noise that we can ignore.

        ncomp = cfg['ncomp']
        print("Keeping only the first {} components".format(ncomp))
        pcomps = components[0:ncomp]

        gparams = np.empty((m, 3))
        z = 0
        for i, params in enumerate(grid.list_grid_points):
            if i == test_index:
                test_params = np.array([params["temp"], params["logg"], params["Z"]])
                continue
            gparams[z, :] = np.array([params["temp"], params["logg"], params["Z"]])
            z += 1

        #Create w

        w = np.empty((ncomp, m))
        for i,pcomp in enumerate(pcomps):
            for j,spec in enumerate(fluxes):
                w[i,j] = np.sum(pcomp * spec)

        pca = cls(wl, min_v, flux_mean, flux_std, pcomps, w, gparams)
        pca.ind = ind
        return pca

    def determine_chunk_log(self, wl_data):
        '''
        Possibly truncate the wl, pcomps, and flux_mean and flux_std in response to some data.
        '''

        #determine the indices
        wl_min, wl_max = np.min(wl_data), np.max(wl_data)
        #Length of the raw synthetic spectrum
        len_wl = len(self.wl)
        #Length of the data
        len_data = np.sum((self.wl > wl_min) & (self.wl < wl_max)) #what is the minimum amount of the
        # synthetic spectrum that we need?

        #Find the smallest length synthetic spectrum that is a power of 2 in length and larger than the data spectrum
        chunk = len_wl
        inds = (0, chunk) #Set to be the full spectrum

        while chunk > len_data:
            if chunk/2 > len_data:
                chunk = chunk//2
            else:
                break

        assert type(chunk) == np.int, "Chunk is no longer integer!. Chunk is {}".format(chunk)

        if chunk < len_wl:
            # Now that we have determined the length of the chunk of the synthetic spectrum, determine indices
            # that straddle the data spectrum.

            # What index corresponds to the wl at the center of the data spectrum?
            center_wl = np.median(wl_data)
            center_ind = (np.abs(self.wl - center_wl)).argmin()

            #Take the chunk that straddles either side.
            inds = (center_ind - chunk//2, center_ind + chunk//2)

            ind = (np.arange(len_wl) >= inds[0]) & (np.arange(len_wl) < inds[1])

        else:
            ind = np.ones_like(self.wl, dtype="bool")

        assert (min(self.wl[ind]) <= wl_min) and (max(self.wl[ind]) >= wl_max), "ModelInterpolator chunking ({:.2f}, " \
        "{:.2f}) didn't encapsulate full wl range ({:.2f}, {:.2f}).".format(min(self.wl[ind]), max(self.wl[ind]),
                                                                            wl_min, wl_max)

        self.wl = self.wl[ind]
        self.pcomps = self.pcomps[:, ind]
        self.flux_mean = self.flux_mean[ind]
        self.flux_std = self.flux_std[ind]

    def get_index(self, stellar_params):
        '''
        Given a np.array of stellar params (corresponding to a grid point), deliver the index that corresponds to the
        entry in the fluxes, list_grid_points, and weights
        '''
        return np.sum(np.abs(self.gparams - stellar_params), axis=1).argmin()

    def reconstruct(self, weights):
        '''
        Reconstruct a spectrum given some weights.

        Also correct for original scaling.
        '''

        f = np.empty((self.ncomp, self.npix))
        for i, (pcomp, weight) in enumerate(zip(self.pcomps, weights)):
            f[i, :] = pcomp * weight
        return np.sum(f, axis=0) * self.flux_std + self.flux_mean

    def reconstruct_all(self):
        '''
        Return a (m, npix) array with all of the spectra reconstructed.
        '''
        recon_fluxes = np.empty((self.m, self.npix))
        for i in range(self.m):
            f = np.empty((self.ncomp, self.npix))
            for j, (pcomp, weight) in enumerate(zip(self.pcomps, self.w[:,i])):
                f[j, :] = pcomp * weight
            recon_fluxes[i, :] = np.sum(f, axis=0) * self.flux_std + self.flux_mean

        return recon_fluxes

    def write(self, filename):
        '''
        Write the PCA decomposition to an HDF5 file.
        '''

        hdf5 = h5py.File(filename, "w")

        hdf5.attrs["min_v"] = self.min_v

        pdset = hdf5.create_dataset("pcomps", (self.ncomp + 3, self.npix), compression='gzip', dtype="f8",
                                    compression_opts=9)
        pdset[0,:] = self.wl
        pdset[1,:] = self.flux_mean
        pdset[2,:] = self.flux_std
        pdset[3:, :] = self.pcomps

        wdset = hdf5.create_dataset("w", (self.ncomp, self.m), compression='gzip', dtype="f8", compression_opts=9)
        wdset[:] = self.w

        gdset = hdf5.create_dataset("gparams", (self.m, 3), compression='gzip', dtype="f8", compression_opts=9)
        gdset[:] = self.gparams

        hdf5.close()

class WeightEmulator:
    def __init__(self, PCAGrid, emulator_params, w_index, samples=None):
        #Construct the emulator using the sampled parameters.

        self.PCAGrid = PCAGrid

        #vector of weights
        self.wvec = self.PCAGrid.w[w_index]

        #Optionally store the samples from an MCMC run.
        self.samples = samples
        if self.samples is not None:
            self.indexes = np.arange(len(self.samples))

        if emulator_params is None:
            if self.samples is None:
                raise ValueError("Must supply either emulator parameters or samples.")
            else:
                self.emulator_params = self.samples[np.random.choice(self.indexes)]
        else:
            loga, lt, ll, lz = emulator_params
            self.a2 = 10**(2 * loga)
            self.lt2 = lt**2
            self.ll2 = ll**2
            self.lz2 = lz**2

            C = em.sigma(self.PCAGrid.gparams, self.a2, self.lt2, self.ll2, self.lz2)

            self.V11 = C

        self._params = None #Where we want to interpolate

        self.V12 = None
        self.V22 = None
        self.mu = None
        self.sig = None


    @property
    def emulator_params(self):
        #return np.concatenate((np.log10(self.lambda_p), self.lambda_w, self.rho_w))
        pass

    @emulator_params.setter
    def emulator_params(self, emulator_params):
        loga, lt, ll, lz = emulator_params
        self.a2 = 10**(2 * loga)
        self.lt2 = lt**2
        self.ll2 = ll**2
        self.lz2 = lz**2

        C = em.sigma(self.PCAGrid.gparams, self.a2, self.lt2, self.ll2, self.lz2)

        self.V11 = C

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, pars):
        #Assumes pars are coming in as Temp, logg, Z.
        self._params = pars

        #Do this according to R&W eqn 2.18, 2.19

        #Recalculate V12, V21, and V22.
        self.V12 = em.V12_w(self._params, self.PCAGrid.gparams, self.a2, self.lt2, self.ll2, self.lz2)
        self.V22 = em.V22_w(self._params, self.a2, self.lt2, self.ll2, self.lz2)

        #Recalculate the covariance
        self.mu = self.V12.T.dot(np.linalg.solve(self.V11, self.wvec))
        self.mu.shape = (-1)
        self.sig = self.V22 - self.V12.T.dot(np.linalg.solve(self.V11, self.V12))

    def draw_weights(self):
        '''
        Using the current settings, draw a sample of PCA weights
        '''
        return np.random.multivariate_normal(self.mu, self.sig)

    def __call__(self, *args):
        '''
        If you call this with an arg that is an np.array, it will set the emulator to these parameters first and then
        draw weights.

        If no args are provided, the emulator uses the previous parameters but redraws the weights,
        for use in seeing the full scatter.

        If there are samples defined, it will also reseed the emulator parameters by randomly draw a parameter
        combination from MCMC samples.
        '''
        if self.samples is not None:
            self.emulator_params = self.samples[np.random.choice(self.indexes)]
            if not args:
                #Reset V12, V22, since we also changed emulator paramaters.
                self.params = self._params

        if args:
            params, *junk = args
            self.params = params

        if self.V12 is None:
            print("No parameters are set, yet. Must set parameters first.")
            return

        weights = self.draw_weights()
        return weights

class Emulator:
    '''
    Stores a Gaussian process for the weight of each principal component.
    '''
    def __init__(self, PCAGrid, samples_list):
        '''

        :param weights_list: [w0s, w1s, w2s, ...]
        :param samples_list:
        '''
        self.PCAGrid = PCAGrid

        #Determine the minimum and maximum bounds of the grid
        self.min_params = np.min(self.PCAGrid.gparams, axis=0)
        self.max_params = np.max(self.PCAGrid.gparams, axis=0)

        self.pcomps = self.PCAGrid.pcomps
        self.min_v = self.PCAGrid.min_v
        self.wl = self.PCAGrid.wl

        self.WEs = []
        for weight_index, samples in enumerate(samples_list):
            self.WEs.append(WeightEmulator(self.PCAGrid, None, weight_index, samples))

    @classmethod
    def open(cls, filename):
        #Create the PCAGrid from this filename
        pcagrid = PCAGrid.open(filename)
        hdf5 = h5py.File(filename, "r")
        samples = hdf5["samples"][:]
        hdf5.close()

        return cls(pcagrid, samples)

    def determine_chunk_log(self, wl_data):
        '''
        Possibly truncate the wl grid in response to some data. Also truncate pcomps, and flux_mean and flux_std.
        '''
        self.PCAGrid.determine_chunk_log(wl_data)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, pars):
        #Assumes pars are coming in as Temp, logg, Z.
        #If the parameters are out of bounds, raise an error
        if np.any(pars < self.min_params) or np.any(pars > self.max_params):
            raise C.ModelError("Emulating outside of the grid.")

        self._params = pars

    def draw_weights(self):
        '''
        Draw a weight from each WeightEmulator and return as an array.
        '''
        return np.array([WE(self._params) for WE in self.WEs])

    def __call__(self, *args):
        '''
        Same behavior as with WeightEmulator.
        '''
        if args:
            params, *junk = args
            self.params = params

        #In this case, we are making the assumption that we are always drawing a weight at a single stellar value.

        weights = self.draw_weights()
        recons = self.pcomps.T.dot(weights).reshape(-1)
        return recons * self.PCAGrid.flux_std + self.PCAGrid.flux_mean

def main():
    pass

if __name__=="__main__":
    main()

class HabibEmulator:
    '''
    Confusing implementation.
    '''
    def __init__(self, emulator_params):
        #Construct the emulator using the sampled parameters.

        #Emulator parameters include
        p = emulator_params
        self.lambda_p = 10**p[0]
        self.lambda_w = p[1:1+ncomp]
        self.rho_w = p[1+ncomp:]
        self.rho_w.shape = (ncomp, 3)

        inv = PP_inv/self.lambda_p
        bigsig = em.Sigma(sparams, self.lambda_w, self.rho_w)

        self.V11 = inv + bigsig

        self._params = None #Where we want to interpolate

        self.V12 = None
        self.V22 = None
        self.mu = None
        self.sig = None

    @property
    def emulator_params(self):
        return np.concatenate((np.log10(self.lambda_p), self.lambda_w, self.rho_w.flatten()))

    @emulator_params.setter
    def emulator_params(self, emulator_params):
        p = emulator_params
        self.lambda_p = 10**p[0]
        self.lambda_w = p[1:1+ncomp]
        self.rho_w = p[1+ncomp:]
        self.rho_w.shape = (ncomp, 3)

        inv = PP_inv/self.lambda_p
        bigsig = em.Sigma(sparams, self.lambda_w, self.rho_w)

        self.V11 = inv + bigsig

    @property
    def params(self):
        #Convert from [0, 1] back to Temp, logg, Z
        return self._params * deltas + mins

    @params.setter
    def params(self, pars):
        #Assumes pars are coming in as Temp, logg, Z; convert to [0, 1] interval
        self._params = (pars - mins)/deltas

        #Recalculate V12, V21, and V22.
        self.V12 = em.V12(self._params, sparams, self.rho_w)
        self.V22 = em.V22(self._params, self.rho_w)

        #Recalculate the covariance
        self.mu = self.V12.T.dot(sp.linalg.spsolve(self.V11, WHAT))
        self.sig = self.V22 - self.V12.T.dot(sp.linalg.spsolve(self.V11, self.V12))
        self.sig = self.sig.todense()

    def draw_weights(self):
        '''
        Using the current settings, draw a sample of PCA weights
        '''
        return np.random.multivariate_normal(self.mu, self.sig)

    def __call__(self, *args):
        '''
        If you call this with an arg, it will set the emulator to these parameters first and then
        draw weights.

        If no args are provided, the emulator uses the previous parameters but redraws the weights,
        for use in seeing the full scatter.
        '''

        if args:
            params, *junk = args
            self.params = params

        if self.V12 is None:
            print("No parameters are set, yet. Must set parameters first.")
            return

        weights = self.draw_weights()
        print("Weights are {}".format(weights))
        return reconstruct(weights)



# grid = HDF5Interface("../libraries/PHOENIX_SPEX_M.hdf5",
#                  ranges={"temp":(2800, 3400), "logg":(4.5, 6.0), "Z":(-1.0, 0.5), "alpha":(0.0, 0.0)})
#
# wl = grid.wl
# ind = (wl > 20000) * (wl < 24000)
# wl = wl[ind]
# npix = len(wl)
# m = len(grid.list_grid_points)# - 1
# print("Using {} spectra with Npix = {}".format(m, npix))

#test_index = 38
# test_index = 5000
# test_spectrum = None
#
# fluxes = np.empty((m,len(wl)))
# z = 0
# for i, spec in enumerate(grid.fluxes):
#     if i == test_index:
#         test_spectrum = spec[ind]
#         continue
#     fluxes[z,:] = spec[ind]
#     z += 1
#
# #Subtract the mean from all of the fluxes.
# flux_mean = np.average(fluxes, axis=0)
# fluxes -= flux_mean
#
# #"Whiten" each spectrum such that the variance for each wavelength is 1
# flux_std = np.std(fluxes, axis=0)
# fluxes /= flux_std
#
# pca = PCA()
# pca.fit(fluxes)
# comp = pca.transform(fluxes)
# components = pca.components_
# mean = pca.mean_
# print("Shape of PCA components {}".format(components.shape))
#
# if not np.allclose(mean, np.zeros_like(mean)):
#     import sys
#     sys.exit("PCA mean is more than just numerical noise. Something's wrong!")
#
#     #Otherwise, the PCA mean is just numerical noise that we can ignore.
#
# ncomp = 5
# print("Keeping only the first {} components".format(ncomp))
# pcomps = components[0:ncomp]

#Flatten into a large array for later use
#orign = np.ravel(fluxes)

#Load all the stellar parameters into an (m, 3) array
# gparams = np.empty((m, 3))
# z = 0
# test_params = None
# for i, params in enumerate(grid.list_grid_points):
#     if i == test_index:
#         test_params = np.array([params["temp"], params["logg"], params["Z"]])
#         continue
#     gparams[z, :] = np.array([params["temp"], params["logg"], params["Z"]])
#     z += 1
#
# print("Test spectrum is {}".format(test_params))

# #Then standardize onto the interval 0 - 1
# mins = np.min(gparams, axis=0)
# maxs = np.max(gparams, axis=0)
# deltas = maxs - mins
# sparams = (gparams - mins)/deltas
#
# def get_index(stellar_params):
#     '''
#     Given a np.array of stellar params (corresponding to a grid point), deliver the index that corresponds to the
#     entry in the fluxes, list_grid_points, and weights
#     '''
#     return np.sum(np.abs(gparams - stellar_params), axis=1).argmin()
#
# def Phi():
#     '''
#     Use the ncomponents to create the large Phi matrix
#     '''
#     out = []
#     #First, assemble the sub-\phis
#     for component in pcomps:
#         out.append(sp.kron(sp.eye(m, format="csc"), component[np.newaxis].T, format="csc"))
#     #Then hstack these together to form \Phi
#     return sp.hstack(out, format="csc")

# #Create the vector w
# def get_w():
#     out = []
#     print(fluxes.shape)
#     print(pcomps[0].shape)
#     for i in range(ncomp):
#         out.append(np.sum(fluxes * pcomps[i], axis=1))
#     w = np.hstack(out)[np.newaxis].T
#     return w

# def reconstruct_full(PHI, w):
#     '''
#     Using the large PHI parameter and the weights, reconstruct what the full
#     grid would look like.
#     '''
#     recons = PHI.dot(w)
#     recons = recons.reshape(m,-1)
#
#     recons = recons * flux_std + flux_mean
#
#     return np.ravel(recons)
#
# def get_what(PHI):
#     '''
#     Reconstruct what?!?
#     '''
#     what = sp.linalg.inv(PHI.T.dot(PHI).tocsc()).dot(PHI.T).dot(orign)
#     return what

# def reconstruct(weights):
#     '''
#     Reconstruct a spectrum given some weights.
#
#     Also correct for original scaling.
#     '''
#
#     f = np.empty((ncomp, len(wl)))
#     for i, (pcomp, weight) in enumerate(zip(pcomps, weights)):
#         f[i, :] = pcomp * weight
#     return np.sum(f, axis=0) * flux_std + flux_mean

# ws = get_w().reshape(ncomp, -1)
#
# PHI = Phi()
# PP = PHI.T.dot(PHI).tocsc()
# PP_inv = sp.linalg.inv(PP)
#
# WHAT = get_what(PHI)
#
# #The prior distributions that Habib et al. 2007 uses
# a_w = 5.
# b_w = 5.
# a_rho_w = 1.0
# b_rho_w = 0.1
#
# a_P = 1.0
# b_P = 0.0001
#
# a_Pprime = a_P + m * (npix - ncomp)/2.
#
# a1 = sp.linalg.inv(PP)
# a2 = PHI.T.dot(orign)
# a2.shape = (-1, 1)
# middle = np.dot(orign,orign) - a2.T.dot(a1.dot(a2))
#
# b_Pprime = b_P + 0.5 * middle[0,0]
#
# print("a_Pprime {}".format(a_Pprime))
# print("b_Pprime {}".format(b_Pprime))

# def lnprob(p):
#     '''
#     Calculate the LNPROB!!
#
#     p will contain many parameters
#
#     lambda_p : simulation precision (1,)
#     lambda_w : weight precision (ncomp,)
#     rho_w : (ncomp, 3)
#
#     Unpack these parameters from the list.
#
#     '''
#     lambda_p = 10**p[0]
#     lambda_w = p[1:1+ncomp]
#     rho_w = p[1+ncomp:]
#     rho_w.shape = (ncomp, 3)
#
#     if np.any((rho_w >= 1.0) | (rho_w <= 0.0)):
#         return -np.inf
#
#     if np.any((lambda_w < 0)):
#         return -np.inf
#
#     inv = PP_inv/lambda_p
#     bigsig = em.Sigma(sparams, lambda_w, rho_w)
#
#     #sparse matrix
#     comb = inv + bigsig
#
#     sign, pref = np.linalg.slogdet(comb.todense())
#     pref *= -0.5
#
#     central = -0.5 * (WHAT.T.dot(sp.linalg.spsolve(comb, WHAT)))
#
#     #prior = (a_Pprime - 1) * np.log(lambda_p) - \
#             # b_Pprime*lambda_p + np.sum((a_w - 1.)*lambda_w - b_w*lambda_w) + \
#             # np.sum((b_rho_w - 1.) * np.log(1. - rho_w))
#
#     return pref + central# + prior

# w_index = 0
# start = w_index * m
# end = (w_index + 1) * m
# w1 = get_w()[start:end]

# def test_lnprob():
#     pars = np.concatenate((np.array([-10.]), np.random.uniform(size=(ncomp,)), np.random.uniform(size=(ncomp*3,))))
#     lnprob(pars)
#
# def sample_lnprob():
#     import emcee
#
#     ndim = (1 + ncomp + ncomp * 3)
#     nwalkers = 4 * ndim
#     print("using {} walkers".format(nwalkers))
#
#     #Designed to be a list of walker positions
#     log_lambda_p = np.random.uniform(low=1, high=5, size=(1, nwalkers))
#     lambda_w = np.random.uniform(low=0.5, high=2, size=(ncomp, nwalkers))
#     rho_w = np.random.uniform(low=0.1, high=0.9, size=(ncomp*3, nwalkers))
#     p0 = np.vstack((log_lambda_p, lambda_w, rho_w)).T
#
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=54)
#
#     print("Running Sampler")
#     pos, prob, state = sampler.run_mcmc(p0, 5000)
#
#     print("Burn-in complete")
#     sampler.reset()
#     sampler.run_mcmc(pos, 5000)
#
#     samples = sampler.flatchain
#     np.save("samples_new.npy", samples)
#
#     import triangle
#     fig = triangle.corner(samples)
#     fig.savefig("triangle_new.png")