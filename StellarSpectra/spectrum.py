import numpy as np
from numpy.polynomial import Chebyshev as Ch
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import j1
import scipy.sparse as sp
from astropy.io import ascii,fits
from scipy.sparse.linalg import spsolve
import gc
# import pyfftw
import warnings
import StellarSpectra.constants as C
import copy

log_lam_kws = frozenset(("CDELT1", "CRVAL1", "NAXIS1"))
flux_units = frozenset(("f_lam", "f_nu"))

class BaseSpectrum:
    '''
    The base spectrum object, designed to be inherited.

    :param wl: wavelength array
    :type wl: np.array
    :param fl: flux array, must be the same shape as :attr:`wl`
    :type fl: np.array
    :param unit: flux unit (``f_lam``, or ``f_nu``)
    :param air: is wavelength array measured in air?
    :type air: bool
    :param metadata: any extra metadata associated with the spectrum
    :type metadat: dict
    '''

    def __init__(self, wl, fl, unit="f_lam", air=True, metadata=None):
        #TODO: convert unit to use astropy units for later conversions
        assert wl.shape == fl.shape, "Spectrum wavelength and flux arrays must have the same shape."
        self.wl = wl
        self.fl = fl
        self.unit = unit
        self.air = air
        self.metadata = {} if metadata is None else metadata
        self.metadata.update({"air": self.air, "unit": self.unit})

    def convert_units(self, unit="f_nu"):
        '''
        Convert between f_lam and f_nu. If :attr:`unit` == :attr:`self.unit`, do nothing.

        :param unit: flux unit to convert to
        :type unit: string

        :raises AssertionError: if unit is not ``"f_lam"`` or ``"f_nu"``.
        '''
        assert unit in flux_units, "{} must be one of {} flux units".format(unit, flux_units)
        if unit == self.unit:
            print("Flux already in {}".format(self.unit))
            return
        elif unit == "f_lam" and self.unit == "f_nu":
            #Convert from f_nu to f_lam
            self.fl = self.fl * C.c_ang / self.wl ** 2
            self.unit = unit
            self.metadata.update({"unit": self.unit})

        elif unit == "f_nu" and self.unit == "f_lam":
            #Convert from f_lam to f_nu
            self.fl = self.fl * self.wl ** 2 / C.c_ang
            self.unit = unit
            self.metadata.update({"unit": self.unit})


    def save(self, name):
        '''
        Save the spectrum as a 2D numpy array. wl = arr[0,:], fl = arr[1,:]

        :param name: filename
        '''
        obj = np.array((self.wl, self.fl))
        np.save(name, obj)

    def __str__(self):
        '''
        Print metadata of spectrum
        '''
        return '''Spectrum object\n''' + "\n".join(
            ["{}:{}".format(key, self.metadata[key]) for key in sorted(self.metadata.keys())])

    def copy(self):
        '''
        return a copy of the spectrum
        '''
        return copy.deepcopy(self)

class Base1DSpectrum(BaseSpectrum):
    '''
    The base one-dimensional spectrum object, designed to be initialized with a generic spectrum.

    :param wl: wavelength array
    :type wl: np.array
    :param fl: flux array
    :type fl: np.array
    :param unit: flux unit (``f_lam``, or ``f_nu``)
    :param air: is wavelength array measured in air?
    :type air: bool
    :param metadata: any extra metadata associated with the spectrum
    :type metadat: dict

    Initialization sorts the wl array to make sure all points are sequential and unique.
    '''

    def __init__(self, wl, fl, unit="f_lam", air=True, metadata=None):
        assert len(wl.shape) == 1, "1D spectrum must be 1D"
        #must have this again in order to prevent an error for fl[ind]
        assert wl.shape == fl.shape, "Spectrum wavelength and flux arrays must have the same shape."
        #"Clean" the wl and flux points. Remove duplicates, sort in increasing wl
        wl_sorted, ind = np.unique(wl, return_index=True)
        fl_sorted = fl[ind]
        super().__init__(wl_sorted, fl_sorted, unit=unit, air=air, metadata=metadata)

    def calculate_log_lam_grid(self, min=True):
        '''
        Determine the minimum spacing and create a log lambda grid that satisfies the sampling requirements. This assumes
        that the input spectrum has already been properly oversampled.

        :returns: wl_dict containing a log lam wl and header keywords.
        '''
        dif = np.diff(self.wl)

        if min:
            #take the minimum wl spacing as the Nyquist sampling rate
            dif_wl = np.min(dif)
            wl = self.wl[np.argmin(dif)]
        else:
            #take the maximum wl spacing as the Nyquist sampling rate
            dif_wl = np.max(dif)
            wl = self.wl[np.argmax(dif)]

        print(dif_wl, wl)
        wl_dict = create_log_lam_grid(wl_start=self.wl[0], wl_end=self.wl[-1], min_wl=(dif_wl, wl))
        return wl_dict

    def resample_to_grid(self, grid, integrate=False):
        '''
        Resample the spectrum to a new grid. Update :attr:`wl` and :attr:`fl`.

        :param grid: the new wavelength grid to resample/rebin to
        :type grid: np.array
        :param integrate: rebin instead of resample?
        :type integrate: bool
        '''
        assert len(grid.shape) == 1, "grid must be 1D"

        #Check to make sure that the grid is within self.wl range, because a spline will not naturally raise an error!
        if min(grid) < min(self.wl) or max(grid) > max(self.wl):
            raise ValueError("grid must fit within the range of self.wl")

        if integrate:
            assert "rebin" not in self.metadata.keys(), "Spectrum has already been rebinned"
            assert self.unit == "f_lam", "Current Integration routine assumes f_lam"
            #Convert from f_lam to counts/ang via Bessel and Murphy 2012
            f = self.fl * self.wl / (C.h * C.c_ang)
            interp = InterpolatedUnivariateSpline(self.wl, f, k=5)

            #Assume that grid specifies the pixel centers. Now, need to calculate the edges.
            edges = np.empty((len(grid) + 1,))
            difs = np.diff(grid) / 2.
            edges[1:-1] = grid[:-1] + difs
            edges[0] = grid[0] - difs[0]
            edges[-1] = grid[-1] + difs[-1]
            starts = edges[:-1]
            ends = edges[1:]

            #do spline integration
            vint = np.vectorize(interp.integral)
            pix = vint(starts, ends)

            #Normalize the average counts/pixel to 100
            avgcounts = np.average(pix)
            pix = pix / avgcounts * 100
            self.fl = pix
            self.unit = 'counts'
            self.metadata.update({"rebin": True})

        else:
            interp = InterpolatedUnivariateSpline(self.wl, self.fl, k=5)
            self.fl = interp(grid)
            self.metadata.update({"resamp": True})

        del interp
        gc.collect()
        self.wl = grid


    def to_LogLambda(self, min_vc, instrument=None):
        '''
        Return a new spectrum that is :obj:`LogLambdaSpectrum`

        :param min_vc: The maximum spacing allowed
        :type min_vc: float
        :param instrument: :obj:`Instrument` object to transfer wl_ranges for truncation.



        '''
        if instrument is not None:
            low, high = instrument.wl_range
            #1) Truncate the wl, fl to the range of the instrument
            ind = (self.wl >= low) & (self.wl <= high)
            wl = self.wl[ind]
            fl = self.fl[ind]

        else:
            low, high = np.min(self.wl), np.max(self.wl)
            wl = self.wl
            fl = self.fl

        #2) create_log_lam_grid() with the instrument ranges and min_vc
        wl_dict = create_log_lam_grid(low, high, min_vc=min_vc)
        wl_log = wl_dict.pop("wl")

        #3) IUS the raw wl and fl
        interp = InterpolatedUnivariateSpline(wl, fl, k=5)

        #4) call IUS on log_lam_grid
        fl_log = interp(wl_log)

        #5) update metadata
        self.metadata.update(wl_dict)

        #6) create and return LogLambdaSpectrum object with wl_dict and header info
        return LogLambdaSpectrum(wl_log, fl_log, air=self.air, metadata=self.metadata)



def create_log_lam_grid(wl_start=3000., wl_end=13000., min_wl=None, min_vc=None):
    '''
    Create a log lambda spaced grid with ``N_points`` equal to a power of 2 for ease of FFT.

    :param wl_start: starting wavelength (inclusive)
    :type wl_start: float
    :param wl_end: ending wavelength (inclusive)
    :type wl_end: float
    :param min_wl: wavelength spacing at a specific wavelength
    :type min_wl: (delta_wl, wl)
    :param min_vc: tightest spacing
    :type min_vc: float
    :returns: a wavelength dictionary containing the specified properties.
    :rtype: wl_dict

    Takes the finer of min_WL or min_vc if both specified
    '''
    assert wl_start < wl_end, "wl_start must be smaller than wl_end"

    if (min_wl is None) and (min_vc is None):
        raise ValueError("You need to specify either min_wl or min_vc")
    if min_wl is not None:
        delta_wl, wl = min_wl #unpack
        Vwl = delta_wl / wl
        min_vc = Vwl
    if (min_wl is not None) and (min_vc is not None):
        min_vc = Vwl if Vwl < min_vc else min_vc

    CDELT_temp = np.log10(min_vc + 1)
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

def calculate_min_v(wl_dict):
    '''
    Given a ``wl_dict``, calculate the velocity spacing.

    :param wl_dict: wavelength dictionary
    :type wl_dict: dict
    '''
    CDELT1 = wl_dict["CDELT1"]
    min_v = C.c_kms * (10**CDELT1 - 1)
    return min_v

class LogLambdaSpectrum(Base1DSpectrum):
    '''
    A spectrum that has log lambda spaced wavelengths.

    :param wl: wavelength array
    :type wl: np.array
    :param fl: flux array
    :type fl: np.array
    :param unit: flux unit (``f_lam``, or ``f_nu``)
    :param air: is wavelength array measured in air?
    :type air: bool
    :param metadata: any extra metadata associated with the spectrum
    :type metadat: dict
    :param oversampling: how many samples fit across the :attr:`FWHM`
    :type oversampling: float

    All of the class methods modify :attr:`wl` and :attr:`fl` in place.

    .. note::

        Because of the difficulty of keeping track of convolution, you only get to do
        {:meth:`instrument_convolve`, :meth:`stellar_convolve`} or
        :meth:`instrument_and_stellar_convolve` once.
    '''

    def __init__(self, wl, fl, unit="f_lam", air=True, metadata=None, oversampling=3.5):
        super().__init__(wl, fl, unit, air=air, metadata=metadata)
        #Super class already checks that the wavelengths are np.unique
        #check that the vc spacing of each pixel is the same.
        vcs = np.diff(self.wl) / self.wl[:-1]
        self.min_vc = np.min(vcs)
        assert np.allclose(vcs, self.min_vc), "Array must be log-lambda spaced."

        #Calculate CDELT1, CRVAL1, and NAXIS1 (even if it's not power of 2)
        CDELT1 = np.log10(self.min_vc + 1)
        CRVAL1 = np.log10(self.wl[0])
        CRVALN = np.log10(self.wl[-1])
        NAXIS1 = int(np.ceil((CRVALN - CRVAL1) / CDELT1))

        wl_dict = {"CDELT1": CDELT1, "CRVAL1": CRVAL1, "NAXIS1": NAXIS1}

        if np.log(NAXIS1) / np.log(2) % 1 != 0:
            warnings.warn("Calculated NAXIS1={}, which is not a power of 2. FFT will be slow.".format(NAXIS1),
                          UserWarning)

        #If wl_dict keys are in the metadata, check to make sure that they are the same ones just calculated.
        if log_lam_kws <= set(self.metadata.keys()):
            assert np.allclose(np.array([self.metadata[key] for key in log_lam_kws]),
                               np.array([wl_dict[key] for key in log_lam_kws])), "Header keywords do not match wl file."
        else:
            self.metadata.update(wl_dict)

        self.oversampling = oversampling #taken to mean as how many points go across the FWHM of the Gaussian

    def resample_to_grid(self, wl_dict, integrate=False):
        '''
        Resample/interate the spectrum to a new log lambda grid.
        Updates the :attr:`wl`, :attr:`fl`, and log_lam_kws in the :attr:`metadata`. This method is slightly
        different than :meth:`Base1DSpectrum.resample_to_grid` since it uses :attr:`wl_dict` to ensure that the spectrum
        remains log lambda spaced and ``N_points`` is a power of 2.

        :param wl_dict: dictionary of log lam wavelength properties to resample to
        :type wl_dict: dict
        :param integrate: integrate flux to counts/pixel?
        :type integrate: bool

        .. note::

            Assumes that new wl grid does not violate any sampling rules and updates header values.
        '''

        wl = wl_dict['wl']

        hdr = {key: wl_dict[key] for key in log_lam_kws}
        self.metadata.update(hdr)

        #resamples the spectrum to these values and updates wl_grid using Base1DSpectrum's resample method.
        super().resample_to_grid(wl, integrate=integrate)

    def convolve_with_gaussian(self, FWHM):
        '''
        Convolve the spectrum with a Gaussian of FWHM

        :param FWHM: the FWHM of the Gaussian kernel
        :type FWHM: float (km/s)
        '''
        sigma = FWHM / 2.35 # in km/s

        chunk = len(self.fl)
        # influx = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
        # FF = pyfftw.n_byte_align_empty(chunk // 2 + 1, 16, 'complex128')
        # outflux = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
        # fft_object = pyfftw.FFTW(influx, FF, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'))
        # ifft_object = pyfftw.FFTW(FF, outflux, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'), direction='FFTW_BACKWARD')

        # influx[:] = self.fl
        # fft_object()

        FF = np.fft.rfft(self.fl)

        #The frequencies (cycles/km) corresponding to each point
        ss = rfftfreq(len(self.fl), d=self.min_vc * C.c_kms)

        #Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space
        taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
        # FF *= taper
        FF_tap = FF  * taper

        #Take the broadened spectrum back to wavelength space
        # ifft_object()
        # self.fl[:] = outflux
        self.fl = np.fft.irfft(FF_tap)

    def instrument_convolve(self, instrument, integrate=False):
        '''
        Convolve the spectrum with an instrumental profile.

        :param instrument: the :obj:`grid_tools.Instrument` object containing the instrumental profile
        :param integrate: integrate to counts/pixel? :attr:`downsample` must not be None

        '''
        assert "instcon" not in self.metadata.keys(), "Spectrum has already been instrument convolved"
        self.convolve_with_gaussian(instrument.FWHM)

        self.metadata.update({"instcon": True})

        #Update min_vc and oversampling, resample to grid
        assert instrument.FWHM >= self.min_vc, "Instrument spacing does not sufficiently oversample the spectrum."

        self.min_vc = instrument.FWHM / C.c_kms
        self.oversampling = instrument.oversampling
        self.resample_to_grid(instrument.wl_dict, integrate=integrate)


    def stellar_convolve(self, vsini):
        '''
        Broaden spectrum due to stellar rotation.

        :param vsini: projected rotational speed of star
        :type vsini: float (km/s)

        '''
        assert "vsini" not in self.metadata.keys(), "Spectrum has already been rotationally convolved"
        if vsini > 0:
            #Take FFT of f_grid
            chunk = len(self.fl)
            # influx = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
            # FF = pyfftw.n_byte_align_empty(chunk // 2 + 1, 16, 'complex128')
            # outflux = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
            # fft_object = pyfftw.FFTW(influx, FF, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'))
            # ifft_object = pyfftw.FFTW(FF, outflux, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'),
            #                           direction='FFTW_BACKWARD')

            # influx[:] = self.fl
            # fft_object()
            FF = np.fft.rfft(self.fl)

            ss = rfftfreq(len(self.wl), d=self.min_vc * C.c_kms_air)
            ss[0] = 0.01 #junk so we don't get a divide by zero error
            ub = 2. * np.pi * vsini * ss
            sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
            #set zeroth frequency to 1 separately (DC term)
            sb[0] = 1.

            #institute velocity taper
            # FF *= sb
            FF_tap = FF * sb

            #do ifft
            # ifft_object()
            # self.fl[:] = outflux
            self.fl = np.fft.irfft(FF_tap)

        else:
            warnings.warn("vsini={}. No stellar convolution performed.".format(vsini), UserWarning)
            vsini = 0.

        self.metadata.update({"vsini": vsini})


    def instrument_and_stellar_convolve(self, instrument, vsini, integrate=False):
        '''
        Perform both instrumental and stellar broadening on the spectrum.

        :param instrument: the :obj:`grid_tools.Instrument` object containing the instrumental profile
        :param vsini: projected rotational speed of star
        :type vsini: float (km/s)
        :param integrate: integrate to counts/pixel? :attr:`downsample` must not be None

        '''
        assert "instcon" not in self.metadata.keys(), "Spectrum has already been instrument convolved"
        assert "vsini" not in self.metadata.keys(), "Spectrum has already been stellar convolved"

        self.stellar_convolve(vsini)
        self.instrument_convolve(instrument, integrate)

    def write_to_FITS(self, out_unit, filename):
        '''
        Write a LogLambdaSpectrum to a FITS file.

        :param out_unit: `f_lam`, `f_nu`, `f_nu_log`, or `counts/pix`
        :type out_unit: string
        :param filename: output filename
        :type filename: string

        Depending on the `out_unit`, the spectrum may be converted.

        * `f_lam`: keep flux units the same
        * `f_nu`: convert flux to ergs/s/cm^2/Hz
        * `f_nu_log`: convert flux to ergs/s/cm^2/log(Hz)
        * `counts/pix`: integrate spectrum to counts/pixel
        '''

        #For now, unless we have integrated, we need to start with `f_lam`.
        if out_unit is not "counts/pix":
            assert self.unit == "f_lam", "FITS writer assumes f_lam input."
        else:
            raise NotImplementedError("counts/pix not yet implemented properly!")

        if out_unit is "f_nu_log":
            #Convert from f_lam to f_nu per log(AA)
            wl = self.wl
            fl = self.fl
            new = wl/(np.log10(C.c_ang) - np.log10(wl)) * fl
            self.fl = new
            out_unit = "ergs/s/cm^2/log(Hz)"
            self.metadata.update({"unit":"f_nu_log"})

        if out_unit is "f_nu":
            #Convert from f_lam to f_nu per log(AA)
            wl = self.wl
            fl = self.fl
            new = wl**2/C.c_ang * fl
            self.fl = new
            out_unit = "ergs/s/cm^2/Hz"
            self.metadata.update({"unit":"f_nu"})

        hdu = fits.PrimaryHDU(self.fl)
        head = hdu.header

        metadata = self.metadata.copy()

        head["DISPTYPE"] = 'log lambda'
        head["DISPUNIT"] = 'log angstroms'
        head["FLUXUNIT"] = out_unit
        head["CRPIX1"] = 1.
        head["DC-FLAG"] = 1
        for key in ['CRVAL1', 'CDELT1','temp','logg','Z','vsini']:
            head[key] = metadata.pop(key)

        #Alphebatize all other keywords, and add some comments
        comments = {"PHXTEFF": "[K] effective temperature",
                    "PHXLOGG": "[cm/s^2] log (surface gravity)",
                    "PHXM_H": "[M/H] metallicity (rel. sol. - Asplund &a 2009)",
                    "PHXALPHA": "[a/M] alpha element enhancement",
                    "PHXDUST": "Dust in atmosphere",
                    "PHXVER": "Phoenix version",
                    "PHXXI_L": "[km/s] microturbulence velocity for LTE lines",
                    "PHXXI_M": "[km/s] microturbulence velocity for molec lines",
                    "PHXXI_N": "[km/s] microturbulence velocity for NLTE lines",
                    "PHXMASS": "[g] Stellar mass",
                    "PHXREFF": "[cm] Effective stellar radius",
                    "PHXLUM": "[ergs] Stellar luminosity",
                    "PHXMXLEN": "Mixing length",
                    "air": "air wavelengths?"}

        for key in sorted(comments.keys()):
            try:
                head[key] = (metadata.pop(key), comments[key])
            except KeyError:
                continue

        extra = {"AUTHOR": "Ian Czekala", "COMMENT" : "Adapted from PHOENIX"}
        head.update(metadata)
        head.update(extra)

        hdu.writeto(filename, clobber=True)
        print("Wrote {} to FITS".format(filename))

    def get_min_v(self):
        CDELT1 = self.metadata["CDELT1"]
        min_v = C.c_kms * (10**CDELT1 - 1)
        return min_v


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

def plot_spectrum(spec, filename, wl_range=None):
    '''
    Plot a spectrum with `matplotlib` and save to a file.

    :param spec: spectrum to plot
    :type spec: LogLambdaSpectrum or Base1DSpectrum
    :param filename: path to save plot
    :type filename: string
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(spec.wl, spec.fl)
    ax.set_xlabel(r"$\lambda (\AA)$")
    if wl_range is not None:
        low, high = wl_range
        ax.set_xlim(low, high)
    fig.savefig(filename)
    plt.close('all')


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
    def __init__(self, wls, fls, sigmas, masks=None, orders='all'):
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
        return cls(wls.astype(np.float64), fls.astype(np.float64), sigmas.astype(np.float64), masks, orders)

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
        return "DataSpectrum object with shape {}".format(self.shape)

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
    def __init__(self, interpolator, instrument):
        self.interpolator = interpolator
        #Set raw_wl from wl stored with interpolator
        self.wl_raw = self.interpolator.wl #Has already been truncated thanks to initialization of Interpolator
        self.instrument = instrument
        self.DataSpectrum = self.interpolator.DataSpectrum #so that we can downsample to the same wls

        self.wl_FFT = self.interpolator.wl

        self.min_v = self.interpolator.interface.wl_header["min_v"]
        self.ss = rfftfreq(len(self.wl_FFT), d=self.min_v)
        self.ss[0] = 0.01 #junk so we don't get a divide by zero error

        self.downsampled_fls = np.empty(self.DataSpectrum.shape)
        self.grid_params = {}
        self.vz = 0
        self.Av = 0
        self.logOmega = 0.

        #Grab chunk from wl_FFT
        chunk = len(self.wl_FFT)
        assert chunk % 2 == 0, "Chunk is not a power of 2. FFT will be too slow."

        # self.influx = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
        # self.FF = pyfftw.n_byte_align_empty(chunk // 2 + 1, 16, 'complex128')
        # self.outflux = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
        # self.fft_object = pyfftw.FFTW(self.influx, self.FF, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'))
        # self.ifft_object = pyfftw.FFTW(self.FF, self.outflux, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'),
        #                           direction='FFTW_BACKWARD')

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
    def update_grid_params(self, grid_params):
        '''
        Private method to update just those stellar parameters. Designed to be used as part of update_all.
        ASSUMES that grid is log-linear spaced and already instrument-convolved

        :param grid_params: grid parameters, including alpha or not.
        :type grid_params: dict

        '''
        try:
            self.fl = self.interpolator(grid_params) #Query the interpolator with the new stellar combination
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
        if vsini < 0:
            raise C.ModelError("vsini must be positive")

        self.vsini = vsini

        # self.influx[:] = self.fl
        # self.fft_object()
        FF = np.fft.rfft(self.fl)

        #Determine the stellar broadening kernel
        ub = 2. * np.pi * self.vsini * self.ss

        if np.allclose(self.vsini, 0):
            sb = np.ones_like(ub)
        else:
            sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)

        #set zeroth frequency to 1 separately (DC term)
        sb[0] = 1.

        #institute velocity and instrumental taper
        # self.FF *= sb
        FF_tap = FF * sb

        #do ifft
        # self.ifft_object()
        # self.fl[:] = self.outflux
        self.fl = np.fft.irfft(FF_tap, len(self.wl_FFT))


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
        #First set stellar parameters using _update_grid_params
        # grid_params = {'temp':params['temp'], 'logg': 4.29, 'Z':params['Z']}
        grid_params = {key:params[key] for key in self.interpolator.parameters}

        #self._update_grid_params(grid_params)
        self.update_grid_params(grid_params)

        #Reset the relative variables
        self.vz = 0
        self.Av = 0
        self.logOmega = 0


        #Then vsini and instrument using _update_vsini
        #self._update_vsini_and_instrument(params['vsini'])
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

        #Check to make sure that the grid is within self.wl range, because a spline will not naturally raise an error!
        wls = self.DataSpectrum.wls.flatten()
        if min(wls) < min(self.wl) or max(wls) > max(self.wl):
            raise ValueError("Downsampled grid ({:.2f},{:.2f}) must fit within the range of self.wl ({:.2f},"
                             "{:.2f})".format(min(wls), max(wls), min(self.wl), max(self.wl)))

        interp = InterpolatedUnivariateSpline(self.wl, self.fl, k=5)

        self.downsampled_fls = np.reshape(interp(wls), self.DataSpectrum.shape)

        del interp
        gc.collect()

class ModelSpectrumHA:
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
    def __init__(self, interpolator, instrument):
        self.interpolator = interpolator
        #Set raw_wl from wl stored with interpolator
        self.wl_raw = self.interpolator.wl #Has already been truncated thanks to initialization of ModelInterpolator
        self.instrument = instrument
        self.DataSpectrum = self.interpolator.DataSpectrum #so that we can downsample to the same wls

        #Calculate a log_lam grid based upon the length of the DataSpectrum
        wl_min, wl_max = np.min(self.DataSpectrum.wls), np.max(self.DataSpectrum.wls)
        wl_dict = create_log_lam_grid(wl_min - 4., wl_max + 4., min_vc=0.08/C.c_kms)
        self.wl_FFT = wl_dict["wl"]
        print("Grid stretches from {} to {}".format(wl_min, wl_max))
        print("wl_FFT is {} km/s".format(calculate_min_v(wl_dict)))

        self.min_v = calculate_min_v(wl_dict)
        self.ss = rfftfreq(len(self.wl_FFT), d=self.min_v)
        self.ss[0] = 0.01 #junk so we don't get a divide by zero error

        self.downsampled_fls = np.empty(self.DataSpectrum.shape)
        self.grid_params = {}
        self.vz = 0
        self.Av = 0
        self.logOmega = 0.

        #Grab chunk from wl_FFT
        chunk = len(self.wl_FFT)
        assert chunk % 2 == 0, "Chunk is not a power of 2. FFT will be too slow."

        # self.influx = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
        # self.FF = pyfftw.n_byte_align_empty(chunk // 2 + 1, 16, 'complex128')
        # self.outflux = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
        # self.fft_object = pyfftw.FFTW(self.influx, self.FF, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'))
        # self.ifft_object = pyfftw.FFTW(self.FF, self.outflux, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'),
        #                           direction='FFTW_BACKWARD')

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

    def _update_grid_params(self, grid_params):
        '''
        Private method to update just those stellar parameters. Designed to be used as part of update_all.

        :param grid_params: grid parameters, including alpha or not.
        :type grid_params: dict

        '''
        try:
            fl = self.interpolator(grid_params) #Query the interpolator with the new stellar combination
            interp = InterpolatedUnivariateSpline(self.wl_raw, fl, k=5)
            self.fl = interp(self.wl_FFT)
            del interp
            gc.collect()
            self.grid_params.update(grid_params)

        except C.InterpolationError as e:
            raise C.ModelError("{} cannot be interpolated from the grid. C.InterpolationError: {}".format(grid_params, e))


    def _update_vsini_and_instrument(self, vsini):
        '''
        Private method to update just the vsini and instrumental kernel. Designed to be used as part of update_all
        *after* the grid_params have been updated.

        :param vsini: projected stellar rotation velocity
        :type vsini: float (km/s)

        '''
        if vsini < 0:
            raise C.ModelError("vsini must be positive")

        self.vsini = vsini

        # self.influx[:] = self.fl
        # self.fft_object()
        FF = np.fft.rfft(self.fl)

        sigma = self.instrument.FWHM / 2.35 # in km/s

        #Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space
        taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (self.ss ** 2))

        #Determine the stellar broadening kernel
        ub = 2. * np.pi * self.vsini * self.ss

        if np.allclose(self.vsini, 0):
            sb = np.ones_like(ub)
        else:
            sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)

        #set zeroth frequency to 1 separately (DC term)
        sb[0] = 1.
        taper[0] = 1.

        #institute velocity and instrumental taper
        # self.FF *= sb * taper
        FF_tap = FF * sb * taper

        #do ifft
        # self.ifft_object()
        # self.fl[:] = self.outflux

        self.fl = np.fft.irfft(FF_tap, len(self.wl_FFT))


    def update_all(self, params):
        '''
        Update all of the stellar parameters

        Give parameters as a dict and choose the params to update.

        '''
        #First set stellar parameters using _update_grid_params
        # grid_params = {'temp':params['temp'], 'logg': 4.29, 'Z':params['Z']}
        grid_params = {key:params[key] for key in self.interpolator.parameters}

        self._update_grid_params(grid_params)

        #Reset the relative variables
        self.vz = 0
        self.Av = 0
        self.logOmega = 0


        #Then vsini and instrument using _update_vsini
        self._update_vsini_and_instrument(params['vsini'])

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

        #Check to make sure that the grid is within self.wl range, because a spline will not naturally raise an error!
        wls = self.DataSpectrum.wls.flatten()
        if min(wls) < min(self.wl) or max(wls) > max(self.wl):
            raise ValueError("Downsampled grid ({:.2f},{:.2f}) must fit within the range of self.wl ({:.2f},"
                             "{:.2f})".format(min(wls), max(wls), min(self.wl), max(self.wl)))

        interp = InterpolatedUnivariateSpline(self.wl, self.fl, k=5)

        self.downsampled_fls = np.reshape(interp(wls), self.DataSpectrum.shape)

        del interp
        gc.collect()


class ModelSpectrumLog:
    '''
    A 1D synthetic spectrum drawing from the LogInterpolator

    :param interpolator: object to query stellar parameters
    :type interpolator: :obj:`grid_tools.ModelInterpolator`
    :param instrument: Which instrument is this a model for?
    :type instrument: :obj:`grid_tools.Instrument` object describing wavelength range and instrumental profile

    We essentially want to preserve two capabilities.

    1. Sample in all "stellar parameters" at once
    2. Sample in only the easy "post-processing" parameters, like ff and v_z to speed the burn-in process.

    '''
    def __init__(self, interpolator, instrument):
        self.interpolator = interpolator
        #Set raw_wl from wl stored with interpolator
        self.wl_raw = self.interpolator.wl #Has already been truncated thanks to initialization of ModelInterpolator
        CDELT1 = self.interpolator.wl_dict["CDELT1"]
        self.min_v = C.c_kms * (10**CDELT1 - 1)
        self.ss = rfftfreq(len(self.wl_raw), d=self.min_v)
        self.ss[0] = 0.01 #junk so we don't get a divide by zero error
        self.instrument = instrument
        self.DataSpectrum = self.interpolator.DataSpectrum #so that we can downsample to the same wls

        self.downsampled_fls = np.empty(self.DataSpectrum.shape)
        self.grid_params = {}
        self.vz = 0
        self.Av = 0
        self.logOmega = 0.

        #Grab chunk from the ModelInterpolator object
        chunk = len(self.wl_raw)
        assert chunk % 2 == 0, "Chunk is not a power of 2. FFT will be too slow."

        # self.influx = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
        # self.FF = pyfftw.n_byte_align_empty(chunk // 2 + 1, 16, 'complex128')
        # self.outflux = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
        # self.fft_object = pyfftw.FFTW(self.influx, self.FF, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'))
        # self.ifft_object = pyfftw.FFTW(self.FF, self.outflux, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'),
        #                                direction='FFTW_BACKWARD')

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
        self.logOmega = logOmega

    def update_vz(self, vz):
        '''
        Update the radial velocity parameter.

        :param vz: The radial velocity parameter. Positive means redshift and negative means blueshift.
        :type vz: float (km/s)

        '''
        #How far to shift based from old vz?
        vz_shift = vz - self.vz
        self.wl = self.wl_raw * np.sqrt((C.c_kms + vz_shift) / (C.c_kms - vz_shift))
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

    def _update_grid_params(self, grid_params):
        '''
        Private method to update just those stellar parameters. Designed to be used as part of update_all.

        :param grid_params: grid parameters, including alpha or not.
        :type grid_params: dict

        '''
        try:
            self.fl = self.interpolator(grid_params) #Query the interpolator with the new stellar combination
            self.grid_params.update(grid_params)

        except C.InterpolationError as e:
            raise C.ModelError("{} cannot be interpolated from the grid. C.InterpolationError: {}".format(grid_params, e))

    def _update_vsini_and_instrument(self, vsini):
        '''
        Private method to update just the vsini and instrumental kernel. Designed to be used as part of update_all
        *after* the grid_params have been updated.

        :param vsini: projected stellar rotation velocity
        :type vsini: float (km/s)

        '''
        if vsini < 0:
            raise C.ModelError("vsini must be positive")

        self.vsini = vsini

        # self.influx[:] = self.fl
        # self.fft_object()
        FF = np.fft.rfft(self.fl)

        sigma = self.instrument.FWHM / 2.35 # in km/s

        #Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space
        taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (self.ss ** 2))

        #Determine the stellar broadening kernel
        ub = 2. * np.pi * self.vsini * self.ss
        sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)

        #set zeroth frequency to 1 separately (DC term)
        sb[0] = 1.
        taper[0] = 1.

        #institute velocity and instrumental taper
        # self.FF *= (sb * taper)
        FF_tap = FF * sb * taper

        #do ifft
        # self.ifft_object()
        # self.fl[:] = self.outflux
        self.fl = np.fft.irfft(FF_tap)


    def update_all(self, params):
        '''
        Update all of the stellar parameters

        Give parameters as a dict and choose the params to update.

        '''
        #First set stellar parameters using _update_grid_params

        #self.interpolator.parameters automatically contains alpha or not

        #Kludgey code to account for Willie setting logg = 4.29
        # grid_params = {'temp':params['temp'], 'logg': 4.29, 'Z':params['Z']}
        grid_params = {key:params[key] for key in self.interpolator.parameters}


        self._update_grid_params(grid_params)

        #Reset the relative variables
        self.vz = 0
        self.Av = 0
        self.logOmega = 0

        #Then vsini and instrument using _update_vsini
        self._update_vsini_and_instrument(params['vsini'])

        #Then vz, ff, and Av using public methods
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

        #Check to make sure that the grid is within self.wl range, because a spline will not naturally raise an error!
        wls = self.DataSpectrum.wls.flatten()
        if min(wls) < min(self.wl) or max(wls) > max(self.wl):
            raise ValueError("New grid ({:.2f},{:.2f}) must fit within the range of self.wl ({:.2f},"
                             "{:.2f})".format(min(wls), max(wls), min(self.wl), max(self.wl)))

        interp = InterpolatedUnivariateSpline(self.wl, self.fl, k=5)

        self.downsampled_fls = np.reshape(interp(wls), self.DataSpectrum.shape)

        del interp
        gc.collect()


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
        print("params are ", params)

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
        self.k = k


class CovarianceMatrix:
    '''
    Non-trivial covariance matrices (one for each order) for correlated noise.

    '''

    def __init__(self, DataSpectrum, index):
        self.wl = DataSpectrum.wls[index]
        self.fl = DataSpectrum.fls[index]
        self.sigma = DataSpectrum.sigmas[index]

        self.npoints = len(self.wl)

        #Because sparse matrices only come in 2D, we have a list of sparse matrices.
        self.sigma_matrix = sp.diags([self.sigma**2], [0], dtype=np.float64, format="csc")
        self.matrix = sp.diags([self.sigma**2], [0], dtype=np.float64, format="csc")
        self.logdet = 0.0
        self.calculate_logdet()


    def sparse_k_3_2(self, xs, amp, l):
        N = len(xs)
        offset = 0
        r0 = 6 * l
        diags = []
        while offset < N:
            #Pairwise calculate rs
            if offset == 0:
                rs = np.zeros_like(xs)
            else:
                rs = np.abs(xs[offset:] - xs[:-offset])
                k = np.empty_like(rs)
            if np.min(rs) >= r0:
                break
            k = (0.5 + 0.5 * np.cos(np.pi * rs/r0)) * amp**2 * (1 + np.sqrt(3) * rs/l) * np.exp(-np.sqrt(3) * rs/l)
            k[rs >= r0] = 0
            diags.append(k)
            offset += 1

        #Mirror the diagonals
        front = diags[1:].copy()
        front.reverse()
        diags = front + diags
        offsets = [i for i in range(-offset + 1, offset)]
        return sp.diags(diags, offsets, format="csc")

    def calculate_logdet(self):
        #Calculate logdet

        sign, logdet = np.linalg.slogdet(self.matrix.todense())
        if sign <= 0:
            raise C.ModelError("The determinant is negative.")
        self.logdet = logdet


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

        self.matrix = self.sparse_k_3_2(self.wl, amp, l) + sigAmp**2 * self.sigma_matrix
        self.calculate_logdet()



    def gauss_func(x0i, x1i, x0v=None, x1v=None, amp=None, mu=None, sigma=None):
        x0 = x0v[x0i]
        x1 = x1v[x1i]
        return amp**2/(2 * np.pi * sigma**2) * np.exp(-((x0 - mu)**2 + (x1 - mu)**2)/(2 * sigma**2))

    def Cregion(xs, amp, mu, sigma, var=1):
        '''Create a sparse covariance matrix using identity and block_diagonal'''
        #In the region of the Gaussian, the matrix will be dense, so just create it as `fromfunction`
        #and then later turn it into a sparse matrix with size xs x xs

        #Given mu, and the extent of sigma, estimate the data points that are above, in Gaussian, and below
        n_above = np.sum(xs < (mu - 4 * sigma))
        n_below = np.sum(xs > (mu + 4 * sigma))

        #Create dense matrix and indexes, then convert to lists so that you can pack things in as:

        #csc_matrix((data, ij), [shape=(M, N)])
        #where data and ij satisfy the relationship a[ij[0, k], ij[1, k]] = data[k]

        len_x = len(xs)
        ind_in = (xs >= (mu - 4 * sigma)) & (xs <= (mu + 4 * sigma)) #indices to grab the x values
        len_in = np.sum(ind_in)
        #print(n_above, n_below, len_in)
        #that will be needed to evaluate the Gaussian


        #Create Gaussian matrix fromfunction
        x_gauss = xs[ind_in]
        gauss_mat = np.fromfunction(gauss_func, (len_in,len_in), x0v=x_gauss, x1v=x_gauss,
                                    amp=amp, mu=mu, sigma=sigma, dtype=np.int).flatten()

        #Create an index array that matches the Gaussian
        ij = np.indices((len_in, len_in)) + n_above
        ij.shape = (2, -1)

        return sp.csc_matrix((gauss_mat, ij), shape=(len_x,len_x))


    def evaluate(self, model_fl):
        '''
        Evaluate lnprob using the data flux, model flux and covariance matrix.
        '''

        a = self.fl - model_fl
        lnp = -0.5 * (a.T.dot(spsolve(self.matrix, a)) + self.logdet)
        return lnp

    #def evaluate_cho(self, model_fls):
    #    '''
    #    Evaluate lnprob but using the cho_factor for speedup
    #    '''
    #    A = self.DataSpectrum.fls - model_fls #2D array
    #
    #    #Go through each order
    #    lnp = np.empty((self.norders,))
    #    for i in range(self.norders):
    #        a = A[i]
    #        s = self.matrices[i].todense()
    #        factor, flag = cho_factor(s)
    #        logdet = np.sum(2 * np.log(np.diag(factor)))
    #        lnp[i] = -0.5 * (a.T.dot(cho_solve((factor, flag), a)) + logdet)
    #
    #    return np.sum(lnp)


class OrderSpectrum:
    '''
    Container class for ChebyshevSpectrum and CovarianceMatrix for each order.
    '''

    def __init__(self, DataSpectrum, index):
        self.index = index
        wl = DataSpectrum.wls[index]
        self.ChebyshevSpectrum = ChebyshevSpectrum(wl)
        self.CovarianceMatrix = CovarianceMatrix(DataSpectrum, index=index)
        self.regions = []

    def update_Cheb(self, coefs):
        self.ChebyshevSpectrum.update(coefs)

    def update_Cov(self, params):
        self.CovarianceMatrix.update(params)

    def evaluate(self, model_fl):
        return self.CovarianceMatrix.evaluate(model_fl)


#class VelocitySpectrum:
#    '''
#    Preserves the cool velocity shifting of BaseSpectrum, but we don't really need it for the general spectra.
#    '''
#    def __init__(self, wl, fl, unit="f_lam", air=True, vel=0.0, metadata=None):
#        #TODO: convert unit to use astropy units for later conversions
#        assert wl.shape == fl.shape, "Spectrum wavelength and flux arrays must have the same shape."
#        self.wl_raw = wl
#        self.fl = fl
#        self.unit = unit
#        self.air = air
#        self.velocity = vel #creates self.wl_vel
#        self.metadata = {} if metadata is None else metadata
#
#    def convert_units(self):
#        raise NotImplementedError
#
#    #Set air as a property which will update self.c it uses to calculate velocities
#    @property
#    def air(self):
#        return self._air
#
#    @air.setter
#    def air(self, air):
#        #TODO: rewrite this to be more specific about which c
#        assert type(air) == type(True)
#        self._air = air
#        if self.air:
#            self.c = C.c_kms_air
#        else:
#            self.c = C.c_kms
#
#    @property
#    def velocity(self):
#        return self._velocity
#
#    @velocity.setter
#    def velocity(self, vz):
#        '''Shift the wl_vel relative to wl_raw. Keeps track if in air. Positive vz is redshift.'''
#        self.wl_vel = self.wl_raw * np.sqrt((self.c + vz) / (self.c - vz))
#        self._velocity = vz
#
#    def add_metadata(self, keyVal):
#        key, val = keyVal
#        if key in self.metadata.keys():
#            self.metadata[key]+= val
#        else:
#            self.metadata[key] = val
#
#    def save(self,name):
#        obj = np.array((self.wl_vel, self.fl))
#        np.save(name, obj)
#
#
#    def __str__(self):
#        return '''Spectrum object\n''' + "\n".join(["{}:{}".format(key,self.metadata[key]) for key in sorted(self.metadata.keys())])
#
#    def copy(self):
#        return copy.copy(self)