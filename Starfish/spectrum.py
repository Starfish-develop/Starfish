import numpy as np
from astropy.io import ascii
from numpy.polynomial import Chebyshev as Ch

import Starfish.constants as C

log_lam_kws = frozenset(("CDELT1", "CRVAL1", "NAXIS1"))
flux_units = frozenset(("f_lam", "f_nu"))


def calculate_dv(wl):
    '''
    Given a wavelength array, calculate the minimum ``dv`` of the array.

    :param wl: wavelength array
    :type wl: np.array

    :returns: (float) delta-v in units of km/s
    '''
    return C.c_kms * np.min(np.diff(wl) / wl[:-1])


def calculate_dv_dict(wl_dict):
    '''
    Given a ``wl_dict``, calculate the velocity spacing.

    :param wl_dict: wavelength dictionary
    :type wl_dict: dict

    :returns: (float) delta-v in units of km/s
    '''
    CDELT1 = wl_dict["CDELT1"]
    dv = C.c_kms * (10 ** CDELT1 - 1)
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

    CDELT_temp = np.log10(dv / C.c_kms + 1.)
    CRVAL1 = np.log10(wl_start)
    CRVALN = np.log10(wl_end)
    N = (CRVALN - CRVAL1) / CDELT_temp
    NAXIS1 = 2
    while NAXIS1 < N:  # Make NAXIS1 an integer power of 2 for FFT purposes
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


def create_mask(wl, fname):
    '''
    Given a wavelength array (1D or 2D) and an ascii file containing the regions
    that one wishes to mask out, return a boolean array of indices for which
    wavelengths to KEEP in the calculation.

    :param wl: wavelength array (in AA)
    :param fname: filename of masking array

    :returns mask: boolean mask
    '''
    data = ascii.read(fname)

    ind = np.ones_like(wl, dtype="bool")

    for row in data:
        # starting and ending indices
        start, end = row
        print(start, end)

        # All region of wavelength that do not fall in this range
        ind = ind & ((wl < start) | (wl > end))

    return ind


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
            # can either be a numpy array or a list
            orders = np.array(orders)  # just to make sure
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
        # Open the HDF5 file, try to load each of these values.
        import h5py
        with h5py.File(file, "r") as hdf5:
            wls = hdf5["wls"][:]
            fls = hdf5["fls"][:]
            sigmas = hdf5["sigmas"][:]

            try:
                # Try to see if masks is available, otherwise return an all-true mask.
                masks = np.array(hdf5["masks"][:], dtype="bool")
            except KeyError as e:
                masks = np.ones_like(wls, dtype="bool")

        # Although the actual fluxes and errors may be reasonably stored as float32, we need to do all of the calculations
        # in float64, and so we convert here.
        # The wls must be stored as float64, because of precise velocity issues.
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
            # can either be a numpy array or a list
            orders = np.array(orders)  # just to make sure
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

        self.fix_c0 = True if index == (len(DataSpectrum.wls) - 1) else False  # Fix the last c0

        xs = np.arange(len_wl)

        # Create Ch1, etc... for each coefficient in npoly excepting logc0
        # Evaluate these and stuff them into self.T
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

        # Dummy holders for a flat spectrum
        self.k = np.ones(len_wl)
        self.k_last = self.k
        # self.c0s = np.ones(self.norders)
        # self.cns = np.zeros((self.norders, self.npoly - 1))
        # self.TT = np.einsum("in,jn->ijn", T, T)

        ##Priors
        ##    mu = np.array([0, 0, 0])
        ##    D = sigmac ** (-2) * np.eye(3)
        ##    Dmu = np.einsum("ij,j->j", D, mu)
        ##    muDmu = np.einsum("j,j->", mu, Dmu)

    def update(self, p):
        '''
        Given a dictionary of coefs, create a k array to multiply against model fls

        :param p: array of coefficients
        :type p: 1D np.array
        '''

        # Fix the last order c0 to 1.
        if self.fix_c0:
            c0 = 1.0
            cns = p
        else:
            c0 = 10 ** p[0]
            cns = p[1:]

        # now create polynomials for each order, and multiply through fls
        # print("T shape", self.T.shape)
        # print("cns shape", cns.shape)

        Tc = np.dot(self.T.T, cns)  # self.T.T is the transpose of self.T
        # print("Tc shape", Tc.shape)
        k = c0 * (1 + Tc)
        self.k_last = self.k
        self.k = k

    def revert(self):
        self.k = self.k_last
