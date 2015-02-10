import numpy as np
import h5py
from sklearn.decomposition import PCA

from Starfish.grid_tools import HDF5Interface, determine_chunk_log
from Starfish.covariance import sigma, V12, V22
from Starfish import constants as C

def skinny_kron(eigenspectra):
    '''
    Compute Phi.T.dot(Phi) in a memory efficient manner.

    eigenspectra is a list of 1D numpy arrays.
    '''
    out = np.zeros((m * M, m * M))

    # Compute all of the dot products pairwise, beforehand
    dots = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            dots[i,j] = eigenspectra[i].T.dot(eigenspectra[j])

    for i in range(M * m):
        for jj in range(m):
            ii = i // M
            j = jj * M + (i % M)
            out[i, j] = dots[ii, jj]
    return out

class PCAGrid:
    '''
    Create and query eigenspectra.
    '''

    def __init__(self, wl, dv, flux_mean, flux_std, eigenspectra, w, gparams):
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
        :type w: 1D np.array
        :param gparams: The stellar parameters of the synthetic library
        :type gparams: 1D list of dictionaries


        '''
        self.wl = wl
        self.dv = dv
        self.flux_mean = flux_mean
        self.flux_std = flux_std
        self.eigenspectra = eigenspectra
        self.ncomp = len(self.eigenspectra)
        self.w = w
        self.gparams = gparams

        self.npix = len(self.wl)
        self.m = self.w.shape[1] # The number of spectra in the synthetic grid


    @classmethod
    def create(cls, interface, ncomp):
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
        m = len(interface.list_grid_points)

        fluxes = np.empty((m, npix))

        z = 0
        for i, spec in enumerate(interface.fluxes):
            fluxes[z,:] = spec
            z += 1

        # Normalize all of the fluxes to an average value of 1
        # In order to remove uninteresting correlations
        fluxes = fluxes/np.average(fluxes, axis=1)[np.newaxis].T

        # Subtract the mean from all of the fluxes.
        flux_mean = np.average(fluxes, axis=0)
        fluxes -= flux_mean

        # "Whiten" each spectrum such that the variance for each wavelength is 1
        flux_std = np.std(fluxes, axis=0)
        fluxes /= flux_std

        # Use the scikit-learn PCA module
        pca = PCA()
        pca.fit(fluxes)
        comp = pca.transform(fluxes)
        components = pca.components_
        mean = pca.mean_
        print("Shape of PCA components {}".format(components.shape))

        if not np.allclose(mean, np.zeros_like(mean)):
            import sys
            sys.exit("PCA mean is more than just numerical noise. Something's wrong!")
            # Otherwise, the PCA mean is just numerical noise that we can ignore.

        print("Keeping only the first {} components".format(ncomp))
        eigenspectra = components[0:ncomp]

        gparams = np.empty((m, 3))
        z = 0
        for i, params in enumerate(interface.list_grid_points):
            # if i == test_index:
                # test_params = np.array([params["temp"], params["logg"], params["Z"]])
                # continue
            gparams[z, :] = np.array([params["temp"], params["logg"], params["Z"]])
            z += 1

        #Create w, the weights corresponding to the synthetic grid

        w = np.empty((ncomp, m))
        for i,pcomp in enumerate(eigenspectra):
            for j,spec in enumerate(fluxes):
                w[i,j] = np.sum(pcomp * spec)

        return cls(wl, dv, flux_mean, flux_std, eigenspectra, w, gparams)

    def write(self, filename):
        '''
        Write the PCA decomposition to an HDF5 file.

        :param filename: name of the HDF5 to create.
        :type filename: str

        '''

        hdf5 = h5py.File(filename, "w")

        hdf5.attrs["dv"] = self.dv

        pdset = hdf5.create_dataset("eigenspectra", (self.ncomp + 3, self.npix),
            compression='gzip', dtype="f8", compression_opts=9)
        pdset[0,:] = self.wl
        pdset[1,:] = self.flux_mean
        pdset[2,:] = self.flux_std
        pdset[3:, :] = self.eigenspectra

        wdset = hdf5.create_dataset("w", (self.ncomp, self.m), compression='gzip',
            dtype="f8", compression_opts=9)
        wdset[:] = self.w

        gdset = hdf5.create_dataset("gparams", (self.m, 3), compression='gzip',
            dtype="f8", compression_opts=9)
        gdset[:] = self.gparams

        hdf5.close()

    @classmethod
    def open(cls, filename):
        '''
        Initialize an object using the PCA already stored to an HDF5 file.

        :param filename: filename of an HDF5 object containing the PCA components.
        :type filename: str
        '''

        hdf5 = h5py.File(filename, "r")
        pdset = hdf5["eigenspectra"]

        dv = hdf5.attrs["dv"]

        wl = pdset[0,:]

        flux_mean = pdset[1,:]
        flux_std = pdset[2,:]
        eigenspectra = pdset[3:,:]

        wdset = hdf5["w"]
        w = wdset[:]

        gdset = hdf5["gparams"]
        gparams = gdset[:]

        pcagrid = cls(wl, dv, flux_mean, flux_std, eigenspectra, w, gparams)
        hdf5.close()

        return pcagrid

    def determine_chunk_log(self, wl_data):
        '''
        Possibly truncate the wl, eigenspectra, and flux_mean and flux_std in
        response to some data.

        :param wl_data: The spectrum dataset you want to fit.
        :type wl_data: np.array


        '''

        # determine the indices
        wl_min, wl_max = np.min(wl_data), np.max(wl_data)
        ind = determine_chunk_log(self.wl, wl_min, wl_max)

        # # Length of the raw synthetic spectrum
        # len_wl = len(self.wl)
        # # Length of the data
        # len_data = np.sum((self.wl > wl_min) & (self.wl < wl_max)) #what is the minimum amount of the
        # # synthetic spectrum that we need?
        #
        # #Find the smallest length synthetic spectrum that is a power of 2 in length and larger than the data spectrum
        # chunk = len_wl
        # inds = (0, chunk) #Set to be the full spectrum
        #
        # while chunk > len_data:
        #     if chunk/2 > len_data:
        #         chunk = chunk//2
        #     else:
        #         break
        #
        # assert type(chunk) == np.int, "Chunk is no longer integer!. Chunk is {}".format(chunk)
        #
        # if chunk < len_wl:
        #     # Now that we have determined the length of the chunk of the synthetic spectrum, determine indices
        #     # that straddle the data spectrum.
        #
        #     # What index corresponds to the wl at the center of the data spectrum?
        #     center_wl = np.median(wl_data)
        #     center_ind = (np.abs(self.wl - center_wl)).argmin()
        #
        #     #Take the chunk that straddles either side.
        #     inds = (center_ind - chunk//2, center_ind + chunk//2)
        #
        #     ind = (np.arange(len_wl) >= inds[0]) & (np.arange(len_wl) < inds[1])
        #
        # else:
        #     ind = np.ones_like(self.wl, dtype="bool")

        assert (min(self.wl[ind]) <= wl_min) and (max(self.wl[ind]) >= wl_max),\
            "ModelInterpolator chunking ({:.2f}, {:.2f}) didn't encapsulate " \
            "full wl range ({:.2f}, {:.2f}).".format(min(self.wl[ind]),\
            max(self.wl[ind]), wl_min, wl_max)

        self.wl = self.wl[ind]
        self.eigenspectra = self.eigenspectra[:, ind]
        self.flux_mean = self.flux_mean[ind]
        self.flux_std = self.flux_std[ind]

    def get_index(self, stellar_params):
        '''
        Given a np.array of stellar params (corresponding to a grid point),
        deliver the index that corresponds to the
        entry in the fluxes, list_grid_points, and weights
        '''
        return np.sum(np.abs(self.gparams - stellar_params), axis=1).argmin()

    def reconstruct(self, weights):
        '''
        Reconstruct a spectrum given some weights.

        Also correct for original scaling.
        '''

        f = np.empty((self.ncomp, self.npix))
        for i, (pcomp, weight) in enumerate(zip(self.eigenspectra, weights)):
            f[i, :] = pcomp * weight
        return np.sum(f, axis=0) * self.flux_std + self.flux_mean

    def reconstruct_all(self):
        '''
        Return a (m, npix) array with all of the spectra reconstructed.
        '''
        recon_fluxes = np.empty((self.m, self.npix))
        for i in range(self.m):
            f = np.empty((self.ncomp, self.npix))
            for j, (pcomp, weight) in enumerate(zip(self.eigenspectra, self.w[:,i])):
                f[j, :] = pcomp * weight
            recon_fluxes[i, :] = np.sum(f, axis=0) * self.flux_std + self.flux_mean

        return recon_fluxes


class WeightEmulator:
    def __init__(self, PCAGrid, emulator_params, w_index):
        #Construct the emulator using the sampled parameters.

        self.PCAGrid = PCAGrid

        #vector of weights
        self.wvec = self.PCAGrid.w[w_index]

        loga, lt, ll, lz = emulator_params
        self.a2 = 10**(2 * loga)
        self.lt2 = lt**2
        self.ll2 = ll**2
        self.lz2 = lz**2

        C = sigma(self.PCAGrid.gparams, self.a2, self.lt2, self.ll2, self.lz2)

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

        C = sigma(self.PCAGrid.gparams, self.a2, self.lt2, self.ll2, self.lz2)

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
        self.V12 = V12(self._params, self.PCAGrid.gparams, self.a2, self.lt2, self.ll2, self.lz2)
        self.V22 = V22(self._params, self.a2, self.lt2, self.ll2, self.lz2)

        #Recalculate the covariance
        self.mu = self.V12.T.dot(np.linalg.solve(self.V11, self.wvec))
        self.mu.shape = (-1)
        self.sig = self.V22 - self.V12.T.dot(np.linalg.solve(self.V11, self.V12))

    def draw_weights(self, *args):
        '''
        Using the current settings, draw a sample of PCA weights

        If you call this with an arg that is an np.array, it will set the emulator to these parameters first and then
        draw weights.

        If no args are provided, the emulator uses the previous parameters but redraws the weights,
        for use in seeing the full scatter.

        If there are samples defined, it will also reseed the emulator parameters by randomly draw a parameter
        combination from MCMC samples.
        '''

        if args:
            params, *junk = args
            self.params = params

        if self.V12 is None:
            print("No parameters are set, yet. Must set parameters first.")
            return

        return np.random.multivariate_normal(self.mu, self.sig)

    def __call__(self, *args):
        '''
        If you call this with an arg that is an np.array, it will set the
        emulator to these parameters first and then draw weights.

        If no args are provided, then the emulator uses the previous parameters.

        If there are samples defined, it will also reseed the emulator
        parameters by randomly draw a parameter combination from MCMC samples.
        '''

        # Don't reset the parameters. We want to be using the optimized GP
        # parameters so that the likelihood call is deterministic

        if args:
            params, *junk = args
            self.params = params

        if self.V12 is None:
            print("No parameters are set, yet. Must set parameters first.")
            return

        return (self.mu, self.sig)

class Emulator:
    '''
    Stores a Gaussian process for the weight of each principal component.
    '''
    def __init__(self, PCAGrid, optimized_params):
        '''

        :param weights_list: [w0s, w1s, w2s, ...]
        :param samples_list:
        '''
        self.PCAGrid = PCAGrid

        #Determine the minimum and maximum bounds of the grid
        self.min_params = np.min(self.PCAGrid.gparams, axis=0)
        self.max_params = np.max(self.PCAGrid.gparams, axis=0)

        #self.eigenspectra = self.PCAGrid.eigenspectra
        self.dv = self.PCAGrid.dv
        self.wl = self.PCAGrid.wl

        self.WEs = []
        for weight_index, params in enumerate(optimized_params):
            self.WEs.append(WeightEmulator(self.PCAGrid, params, weight_index))

    @classmethod
    def open(cls, filename):
        '''
        Create an Emulator object from an HDF5 file.
        '''
        #Create the PCAGrid from this filename
        pcagrid = PCAGrid.open(filename)
        hdf5 = h5py.File(filename, "r")

        params = hdf5["params"][:]
        hdf5.close()
        return cls(pcagrid, params)

    def determine_chunk_log(self, wl_data):
        '''
        Possibly truncate the wl grid in response to some data. Also truncate eigenspectra, and flux_mean and flux_std.
        '''
        self.PCAGrid.determine_chunk_log(wl_data)
        self.wl = self.PCAGrid.wl

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

    def draw_weights(self, *args):
        '''
        Draw a weight from each WeightEmulator and return as an array.
        '''
        if args:
            params, *junk = args
            self.params = params
        return np.array([WE.draw_weights(self._params) for WE in self.WEs])

    def reconstruct_draw(self, *args):
        '''
        Reconstructing a spectrum. Taking the place of the old __call__ behavior.
        '''
        if args:
            params, *junk = args
            self.params = params

        #In this case, we are making the assumption that we are always drawing a weight at a single stellar value.

        weights = self.draw_weights()
        recons = self.PCAGrid.eigenspectra.T.dot(weights).reshape(-1)
        return recons * self.PCAGrid.flux_std + self.PCAGrid.flux_mean

    def reconstruct(self, weights):
        '''
        Reconstructing a spectrum. Taking the place of the old __call__ behavior.
        '''
        #In this case, we are making the assumption that we are always drawing a weight at a single stellar value.

        recons = self.PCAGrid.eigenspectra.T.dot(weights).reshape(-1)
        return recons * self.PCAGrid.flux_std + self.PCAGrid.flux_mean

    def __call__(self, *args):
        '''

        :param params: [temp, logg, Z] numpy array

        Same behavior as with WeightEmulator. Returns a two vectors of [mu_1, mu_2, ..., mu_m], [var_1, var_2, ...]
        '''
        if args:
            params, *junk = args
            self.params = params

        #In this case, we are making the assumption that we are always drawing a weight at a single stellar value.
        mu = np.empty((self.PCAGrid.ncomp,))
        sig = np.empty((self.PCAGrid.ncomp,))
        for i in range(self.PCAGrid.ncomp):
            mu[i], sig[i] = self.WEs[i](self._params)

        return mu, sig

def main():
    pass

if __name__=="__main__":
    main()
