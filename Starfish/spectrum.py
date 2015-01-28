import numpy as np
from numpy.polynomial import Chebyshev as Ch
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import j1
import scipy.sparse as sp
from astropy.io import ascii,fits
from scipy.sparse.linalg import spsolve
import gc
import warnings
import Starfish.constants as C
from Starfish.covariance import get_dense_C
from scipy.linalg import cho_factor, cho_solve
import copy

log_lam_kws = frozenset(("CDELT1", "CRVAL1", "NAXIS1"))
flux_units = frozenset(("f_lam", "f_nu"))

def calculate_dv(wl):
    '''
    Given a wavelength array, calculate the minimum ``dv`` of the array.

    :param wl: wavelength array
    :type wl: np.array

    :returns: (float) delta-v in units of km/s
    '''
    return C.c_kms * np.min(np.diff(wl)/wl[:-1])

def calculate_dv_dict(wl_dict):
    '''
    Given a ``wl_dict``, calculate the velocity spacing.

    :param wl_dict: wavelength dictionary
    :type wl_dict: dict

    :returns: (float) delta-v in units of km/s
    '''
    CDELT1 = wl_dict["CDELT1"]
    dv = C.c_kms * (10**CDELT1 - 1)
    return dv

def create_log_lam_grid(dv, wl_start=3000., wl_end=13000.):
    '''
    Create a log lambda spaced grid with ``N_points`` equal to a power of 2 for
    ease of FFT.

    :param wl_start: starting wavelength (inclusive)
    :type wl_start: float, AA
    :param wl_end: ending wavelength (inclusive)
    :type wl_end: float, AA
    :param dv: upper bound on the size of the velocity spacing (in km/s)
    :type dv: float

    :returns: a wavelength dictionary containing the specified properties. Note
        that the returned dv will be <= specified dv.
    :rtype: wl_dict

    '''
    assert wl_start < wl_end, "wl_start must be smaller than wl_end"

    CDELT_temp = np.log10(dv/C.c_kms + 1.)
    CRVAL1 = np.log10(wl_start)
    CRVALN = np.log10(wl_end)
    N = (CRVALN - CRVAL1) / CDELT_temp
    NAXIS1 = 2
    while NAXIS1 < N: #Make NAXIS1 an integer power of 2 for FFT purposes
        NAXIS1 *= 2

    CDELT1 = (CRVALN - CRVAL1) / (NAXIS1 - 1)

    p = np.arange(NAXIS1)
    wl = 10 ** (CRVAL1 + CDELT1 * p)
    return {"wl": wl, "CRVAL1": CRVAL1, "CDELT1": CDELT1, "NAXIS1": NAXIS1}


def rfftfreq(n, d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft).

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start). For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

    f = [0, 1, ..., n/2-1, n/2] / (d*n) if n is even
    f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n) if n is odd

    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
    the Nyquist frequency component is considered to be positive.

    :param n : Window length
    :type n: int
    :param d: Sample spacing (inverse of the sampling rate). Defaults to 1.
    ;type d: scalar, optional
    :returns: f, Array of length ``n//2 + 1`` containing the sample frequencies.
    :rtype: ndarray

    """
    if not isinstance(n, np.int):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = np.arange(0, N, dtype=np.int)
    return results * val

class DataSpectrum:
    '''
    Object to manipulate the data spectrum.

    :param wls: wavelength (in AA)
    :type wls: 1D or 2D np.array
    :param fls: flux (in f_lam)
    :type fls: 1D or 2D np.array
    :param sigmas: Poisson noise (in f_lam)
    :type sigmas: 1D or 2D np.array
    :param masks: Mask to blot out bad pixels or emission regions.
    :type masks: 1D or 2D np.array of boolean values

    If the wl, fl, are provided as 1D arrays (say for a single order), they will be converted to 2D arrays with length 1
    in the 0-axis.

    .. note::

       For now, the DataSpectrum wls, fls, sigmas, and masks must be a rectangular grid. No ragged Echelle orders allowed.

    '''
    def __init__(self, wls, fls, sigmas, masks=None, orders='all', name=None):
        self.wls = np.atleast_2d(wls)
        self.fls = np.atleast_2d(fls)
        self.sigmas = np.atleast_2d(sigmas)
        self.masks = np.atleast_2d(masks) if masks is not None else np.ones_like(self.wls, dtype='b')

        self.shape = self.wls.shape
        assert self.fls.shape == self.shape, "flux array incompatible shape."
        assert self.sigmas.shape == self.shape, "sigma array incompatible shape."
        assert self.masks.shape == self.shape, "mask array incompatible shape."

        if orders != 'all':
            #can either be a numpy array or a list
            orders = np.array(orders) #just to make sure
            self.wls = self.wls[orders]
            self.fls = self.fls[orders]
            self.sigmas = self.sigmas[orders]
            self.masks = self.masks[orders]
            self.shape = self.wls.shape
            self.orders = orders
        else:
            self.orders = np.arange(self.shape[0])

        self.name = name

    @classmethod
    def open(cls, file, orders='all'):
        '''
        Load a spectrum from a directory link pointing to HDF5 output from EchelleTools processing.

        :param base_file: HDF5 file containing files on disk.
        :type base_file: string
        :returns: DataSpectrum
        :param orders: Which orders should we be fitting?
        :type orders: np.array of indexes

        '''
        #Open the HDF5 file, try to load each of these values.
        import h5py
        with h5py.File(file, "r") as hdf5:
            wls = hdf5["wls"][:]
            fls = hdf5["fls"][:]
            sigmas = hdf5["sigmas"][:]

            try:
                #Try to see if masks is available, otherwise return an all-true mask.
                masks = np.array(hdf5["masks"][:], dtype="bool")
            except KeyError as e:
                masks = np.ones_like(wls, dtype="bool")

        #Although the actual fluxes and errors may be reasonably stored as float32, we need to do all of the calculations
        #in float64, and so we convert here.
        #The wls must be stored as float64, because of precise velocity issues.
        return cls(wls.astype(np.float64), fls.astype(np.float64), sigmas.astype(np.float64), masks, orders, name=file)

    @classmethod
    def open_npy(cls, base_file, orders='all'):
        '''
        Load a spectrum from a directory link pointing to .npy output from EchelleTools processing.

        :param base_file: base path name to be appended with ".wls.npy", ".fls.npy", ".sigmas.npy", and ".masks.npy" to load files from disk.
        :type base_file: string
        :returns: DataSpectrum
        :param orders: Which orders should we be fitting?
        :type orders: np.array of indexes

        '''
        wls = np.load(base_file + ".wls.npy")
        fls = np.load(base_file + ".fls.npy")
        sigmas = np.load(base_file + ".sigmas.npy")
        masks = np.load(base_file + ".masks.npy")
        return cls(wls, fls, sigmas, masks, orders)


    def add_mask(self, new_mask):
        '''
        Given a mask with the same self.shape, update self.masks to include the union with this new mask.
        '''
        assert new_mask.shape == self.shape, "new_mask shape ({}) must be the same shape as spectrum ({}).".format(
            new_mask.shape, self.shape)

        self.masks = self.masks & new_mask

    def __str__(self):
        return "DataSpectrum object {} with shape {}".format(self.name, self.shape)

class Mask:
    '''
    Mask to apply to DataSpectrum
    '''
    def __init__(self, masks, orders='all'):
        assert isinstance(masks, np.ndarray), "masks must be a numpy array"
        self.masks = np.atleast_2d(masks)

        if orders != 'all':
            #can either be a numpy array or a list
            orders = np.array(orders) #just to make sure
            self.masks = self.masks[orders]
            self.orders = orders
        else:
            self.orders = np.arange(self.masks.shape[0])


    @classmethod
    def open(cls, file, orders='all'):
        '''
        Load a Mask from a directory link pointing to HDF5 file output from EchelleTools or Generate_mask.ipynb
        processing.

        :param file: HDF5 file containing files on disk.
        :type file: string
        :returns: DataSpectrum
        :param orders: Which orders should we be fitting?
        :type orders: np.array of indexes

        '''
        import h5py
        with h5py.File(file, "r") as hdf5:
            masks = np.array(hdf5["masks"][:], dtype="bool")

        return cls(masks, orders)

class ModelSpectrum:
    '''
    A 1D synthetic spectrum.

    :param interpolator: object to query stellar parameters
    :type interpolator: :obj:`grid_tools.ModelInterpolator`
    :param instrument: Which instrument is this a model for?
    :type instrument: :obj:`grid_tools.Instrument` object describing wavelength range and instrumental profile

    We essentially want to preserve two capabilities.

    1. Sample in all "stellar parameters" at once
    2. Sample in only the easy "post-processing" parameters, like ff and v_z to speed the burn-in process.

    '''
    def __init__(self, Emulator, DataSpectrum, instrument):
        self.Emulator = Emulator

        #Set raw_wl from wl stored with emulator
        self.instrument = instrument
        self.DataSpectrum = DataSpectrum #so that we can downsample to the same wls

        self.wl_FFT = self.Emulator.wl

        self.min_v = self.Emulator.min_v
        self.ss = rfftfreq(len(self.wl_FFT), d=self.min_v)
        self.ss[0] = 0.01 #junk so we don't get a divide by zero error

        self.downsampled_fls = np.empty(self.DataSpectrum.shape)
        self.downsampled_fls_last = self.downsampled_fls

        self.grid_params = {}
        self.vz = 0
        self.Av = 0
        self.logOmega = 0.

        #Grab chunk from wl_FFT
        chunk = len(self.wl_FFT)
        assert chunk % 2 == 0, "Chunk is not a power of 2. FFT will be too slow."

    def __str__(self):
        return "Model Spectrum for Instrument {}".format(self.instrument.name)

    def update_logOmega(self, logOmega):
        '''
        Update the 'flux factor' parameter, :math:`\Omega = R^2/d^2`, which multiplies the model spectrum to be the same scale as the data.

        :param Omega: 'flux factor' parameter, :math:`\Omega = R^2/d^2`
        :type Omega: float

        '''
        #factor by which to correct from old Omega
        self.fl *= 10**(logOmega - self.logOmega)
        #self.errors *= 10**(logOmega - self.logOmega)
        self.logOmega = logOmega

    def update_vz(self, vz):
        '''
        Update the radial velocity parameter.

        :param vz: The radial velocity parameter. Positive means redshift and negative means blueshift.
        :type vz: float (km/s)

        '''
        #How far to shift based from old vz?
        vz_shift = vz - self.vz
        self.wl = self.wl_FFT * np.sqrt((C.c_kms + vz_shift) / (C.c_kms - vz_shift))
        self.vz = vz

    def update_Av(self, Av):
        '''
        Update the extinction parameter.

        :param Av: The optical extinction parameter.
        :type Av: float (magnitudes)

        '''
        pass


        #Av_shift = self.Av - Av
        #self.fl /= deredden(self.wl, Av, mags=False)
        #self.Av = Av

    #Make this private again after speed testing is done
    def update_grid_params(self, weights, grid_params):
        '''
        Private method to update just those stellar parameters. Designed to be used as part of update_all.
        ASSUMES that grid is log-linear spaced and already instrument-convolved

        :param grid_params: grid parameters, including alpha or not.
        :type grid_params: dict

        '''
        try:
            self.fl = self.Emulator.reconstruct(weights)

            self.grid_params.update(grid_params)

        except C.InterpolationError as e:
            raise C.ModelError("{} cannot be interpolated from the grid. C.InterpolationError: {}".format(grid_params, e))

    #Make this private again after speed testing is done
    def update_vsini(self, vsini):
        '''
        Private method to update just the vsini kernel. Designed to be used as part of update_all
        *after* the grid_params have been updated using _update_grid_params_approx.
        ASSUMES that the grid is log-linear spaced and already instrument-convolved

        :param vsini: projected stellar rotation velocity
        :type vsini: float (km/s)

        '''
        if vsini < 0.2:
            raise C.ModelError("vsini must be positive")

        self.vsini = vsini

        # self.influx[:] = self.fl
        # self.fft_object()
        FF = np.fft.rfft(self.fl)
        #FE = np.fft.rfft(self.errors, axis=1)

        #Determine the stellar broadening kernel
        ub = 2. * np.pi * self.vsini * self.ss

        # if np.allclose(self.vsini, 0):
        #     #If in fact, vsini=0, then just exit this subroutine and continue with the raw models.
        #     return None
        #     #sb = np.ones_like(ub)
        # else:
        sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)

        #set zeroth frequency to 1 separately (DC term)
        sb[0] = 1.

        #institute velocity and instrumental taper
        # self.FF *= sb
        FF_tap = FF * sb
        #FE_tap = FE * sb

        #do ifft
        # self.ifft_object()
        # self.fl[:] = self.outflux
        self.fl = np.fft.irfft(FF_tap, len(self.wl_FFT))
        #self.errors = np.fft.irfft(FE_tap, len(self.wl_FFT), axis=1)


    # def _update_vsini_and_instrument(self, vsini):
    #     '''
    #     Private method to update just the vsini and instrumental kernel. Designed to be used as part of update_all
    #     *after* the grid_params have been updated.
    #
    #     :param vsini: projected stellar rotation velocity
    #     :type vsini: float (km/s)
    #
    #     '''
    #     if vsini < 0:
    #         raise C.ModelError("vsini must be positive")
    #
    #     self.vsini = vsini
    #
    #     self.influx[:] = self.fl
    #     self.fft_object()
    #     # FF = np.fft.rfft(self.fl)
    #
    #     sigma = self.instrument.FWHM / 2.35 # in km/s
    #
    #     #Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space
    #     taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (self.ss ** 2))
    #
    #     #Determine the stellar broadening kernel
    #     ub = 2. * np.pi * self.vsini * self.ss
    #
    #     if np.allclose(self.vsini, 0):
    #         sb = np.ones_like(ub)
    #     else:
    #         sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
    #
    #     #set zeroth frequency to 1 separately (DC term)
    #     sb[0] = 1.
    #     taper[0] = 1.
    #
    #     #institute velocity and instrumental taper
    #     self.FF *= sb * taper
    #     # FF_tap = FF * sb * taper
    #
    #     #do ifft
    #     self.ifft_object()
    #     self.fl[:] = self.outflux
    #
    #     # self.fl = np.fft.irfft(FF_tap)


    def update_all(self, params):
        '''
        Update all of the stellar parameters

        Give parameters as a dict and choose the params to update.

        '''
        #Pull the weights from here
        weights = params["weights"]

        #First set stellar parameters using _update_grid_params
        grid_params = {key:params[key] for key in ["temp", "logg", "Z"]}

        self.update_grid_params(weights, grid_params)

        #Reset the relative variables
        self.vz = 0
        self.Av = 0
        self.logOmega = 0

        #Then vsini and instrument using _update_vsini
        self.update_vsini(params['vsini'])

        #Then vz, logOmega, and Av using public methods
        self.update_vz(params['vz'])
        self.update_logOmega(params['logOmega'])

        if 'Av' in params:
            self.update_Av(params['Av'])

        self.downsample()

    def downsample(self):
        '''
        Downsample the synthetic spectrum to the same wl pixels as the DataSpectrum.

        :returns fls: the downsampled fluxes that has the same shape as DataSpectrum.fls
        '''

        #print("inside downsample")
        #Check to make sure that the grid is within self.wl range, because a spline will not naturally raise an error!
        wls = self.DataSpectrum.wls.flatten()
        if min(wls) < min(self.wl) or max(wls) > max(self.wl):
            raise ValueError("Downsampled grid ({:.2f},{:.2f}) must fit within the range of self.wl ({:.2f},"
                             "{:.2f})".format(min(wls), max(wls), min(self.wl), max(self.wl)))

        interp = InterpolatedUnivariateSpline(self.wl, self.fl, k=5)

        self.downsampled_fls_last = self.downsampled_fls
        self.downsampled_fls = np.reshape(interp(wls), self.DataSpectrum.shape)
        del interp

        #if self.downsampled_errors is not None:
        #    self.downsampled_errors_last = np.copy(self.downsampled_errors)

        # if self.other_downsampled_errors is None:
        #     print("setting other downsampled errors")
        #     self.other_downsampled_errors = np.zeros((24,) + self.DataSpectrum.shape)
        #     for i, errspec in enumerate(self.errors):
        #         interp = InterpolatedUnivariateSpline(self.wl, errspec, k=5)
        #
        #         self.other_downsampled_errors[i, :, :] = np.reshape(interp(wls), self.DataSpectrum.shape)
        #         del interp

        #Interpolate each of the 24 error spectra to the grid points
        #if self.downsampled_errors is None:
        # self.downsampled_errors = np.zeros((24,) + self.DataSpectrum.shape)
        # for i, errspec in enumerate(self.errors):
        #     interp = InterpolatedUnivariateSpline(self.wl, errspec, k=5)
        #
        #     self.downsampled_errors[i, :, :] = np.reshape(interp(wls), self.DataSpectrum.shape)
        #     del interp

        #assert np.allclose(self.downsampled_errors, self.other_downsampled_errors), "No longer downsampling the same " \
                                                                                    #"thing"



        #self.downsampled_errors = 1e-15 * np.ones((24,) + self.DataSpectrum.shape)

        gc.collect()

    def revert_flux(self):
        '''
        If a MH proposal was rejected, revert the downsampled flux to it's last value.
        '''
        self.downsampled_fls = self.downsampled_fls_last
        #if self.downsampled_errors_last is not None:
        #    self.downsampled_errors = np.copy(self.downsampled_errors_last)

class ChebyshevSpectrum:
    '''
    A DataSpectrum-like object which multiplies downsampled fls to account for imperfect flux calibration issues.

    :param DataSpectrum: take shape from.
    :type DataSpectrum: :obj:`DataSpectrum` object

    If DataSpectrum.norders == 1, then only c1, c2, and c3 are required. Otherwise c0 is also reqired for each order.
    '''

    def __init__(self, DataSpectrum, index, npoly=4):
        self.wl = DataSpectrum.wls[index]
        len_wl = len(self.wl)

        self.fix_c0 = True if index == (len(DataSpectrum.wls) - 1) else False #Fix the last c0

        xs = np.arange(len_wl)

        #Create Ch1, etc... for each coefficient in npoly excepting logc0
        #Evaluate these and stuff them into self.T
        coeff = [1]
        T = []
        for i in range(1, npoly):
            # print("i = ", i)
            coeff = [0] + coeff
            Chtemp = Ch(coeff, domain=[0, len_wl - 1])
            Ttemp = Chtemp(xs)
            T += [Ttemp]

        # Ch1 = Ch([0, 1], domain=[0, len_wl - 1])
        # T1 = Ch1(xs)
        # Ch2 = Ch([0, 0, 1], domain=[0, len_wl - 1])
        # T2 = Ch2(xs)
        # Ch3 = Ch([0, 0, 0, 1], domain=[0, len_wl - 1])
        # T3 = Ch3(xs)

        # self.T = np.array([T1, T2, T3])
        self.T = np.array(T)
        self.npoly = npoly
        # assert self.npoly == 4, "Only handling order 4 Chebyshev for now."

        #Dummy holders
        self.k = np.ones(len_wl)
        self.k_last = self.k
        #self.c0s = np.ones(self.norders)
        #self.cns = np.zeros((self.norders, self.npoly - 1))
        #self.TT = np.einsum("in,jn->ijn", T, T)

        ##Priors
        ##    mu = np.array([0, 0, 0])
        ##    D = sigmac ** (-2) * np.eye(3)
        ##    Dmu = np.einsum("ij,j->j", D, mu)
        ##    muDmu = np.einsum("j,j->", mu, Dmu)

    def update(self, params):
        '''
        Given a dictionary of coefs, create a k array to multiply against model fls
        '''

        #Fix the last order c0 to 1.
        if self.fix_c0:
            logc0 = 0.0
        else:
            logc0 = params["logc0"]

        #Convert params dict to a 1d array of coefficients
        cns = np.array([params["c{}".format(i)] for i in range(1, self.npoly)])

        # cns = np.array([params["c1"], params["c2"], params["c3"]])

        #if c0 < 0:
        #    raise C.ModelError("Negative c0s are not allowed.")

        #now create polynomials for each order, and multiply through fls
        #print("T shape", self.T.shape)
        #print("cns shape", cns.shape)

        c0 = 10**logc0
        Tc = np.dot(self.T.T, cns) #self.T.T is the transpose of self.T
        #print("Tc shape", Tc.shape)
        k = c0 * (1 + Tc)
        self.k_last = self.k
        self.k = k

    def revert(self):
        self.k = self.k_last

class DataCovarianceMatrix:
    '''
    Non-trivial covariance matrices (one for each order) for correlated noise.

    Let's try doing everything dense.

    And, let's try doing everything all at once. Step in the Global Cov parameters and the Regions.

    How to efficiently fill the matrix?
    1. Define a k_func, based off of update
    2. Pass this to some optimized cython routine which only loops over the appropriate indices, evaluating k_func
    (this is get_dense_C.

    Then we will just have two methods

    CovarianceMatrix.update(params) #updates global and all regions
    CovarianceMatrix.evaluate() uses existing matrix to determine lnprob.

    Passes some partial function to the cython routine.

    initialized via CovarianceMatrix.__init__(self.DataSpectrum, self.index, max_v=max_v, debug=self.debug)

    CovarianceMatrix.update_global(params)

    CovarianceMatrix.revert_global()

    CovarianceMatrix.get_amp()

    CovarianceMatrix.get_region_coverage()

    CovarianceMatrix.evaluate(residuals)

    CovarianceMatrix.create_region(starting_param_dict, self.priors)

    CovMatrix.delete_region(self.region_index)

    CovMatrix.revert_region(self.region_index)

    CovMatrix.update_region(self.region_index, params)

'''

    def __init__(self, DataSpectrum, index):
        self.wl = DataSpectrum.wls[index]
        self.sigma = DataSpectrum.sigmas[index]

        self.npoints = len(self.wl)

        self.sigma_matrix = np.dot(self.sigma**2, np.eye(self.npoints))
        self.matrix = self.sigma_matrix

    #def update_logdet(self):
    #    #Calculate logdet
    #    self.logdet = np.sum(2 * np.log*np.diag(self.factor))

    def update(self, params):
        '''
        Update the covariance matrix using the parameters.

        :param params: parameters to update the covariance matrix
        :type params: dict
        '''

        #Currently expects {"sigAmp", "logAmp", "l"}
        amp = 10**params['logAmp']
        l = params['l']
        sigAmp = params['sigAmp']

        if (l <= 0) or (sigAmp < 0):
            raise C.ModelError("l {} and sigAmp {} must be positive.".format(l, sigAmp))

        mat = get_dense_C()

        #self.factor, self.flag = cho_factor(mat)
        #self.update_logdet()

    #def evaluate(self, residuals):
    #    '''
    #    Evaluate lnprob using the residuals and current covariance matrix.
    #    '''
    #
    #    lnp = -0.5 * (residuals.T.dot(cho_solve((self.factor, self.flag), residuals)) + self.logdet)
    #    return lnp
