import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import j1
import gc
import pyfftw
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

    def calculate_log_lam_grid(self):
        '''
        Determine the minimum spacing and create a log lambda grid that satisfies the sampling requirements.

        :returns: wl_dict containing a log lam wl and header keywords.
        '''
        dif = np.diff(self.wl)
        min_wl = np.min(dif)
        wl_at_min = self.wl[np.argmin(dif)]
        wl_dict = create_log_lam_grid(wl_start=self.wl[0], wl_end=self.wl[-1], min_wl=(min_wl, wl_at_min))
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
            interp = InterpolatedUnivariateSpline(self.wl, f)

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
            interp = InterpolatedUnivariateSpline(self.wl, self.fl)
            self.fl = interp(grid)
            self.metadata.update({"resamp": True})

        del interp
        gc.collect()
        self.wl = grid


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
        influx = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
        FF = pyfftw.n_byte_align_empty(chunk // 2 + 1, 16, 'complex128')
        outflux = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
        fft_object = pyfftw.FFTW(influx, FF, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'))
        ifft_object = pyfftw.FFTW(FF, outflux, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'), direction='FFTW_BACKWARD')

        influx[:] = self.fl
        fft_object()

        #The frequencies (cycles/km) corresponding to each point
        ss = rfftfreq(len(self.fl), d=self.min_vc * C.c_kms)

        #Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space
        taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
        FF *= taper

        #Take the broadened spectrum back to wavelength space
        ifft_object()
        self.fl[:] = outflux


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
            influx = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
            FF = pyfftw.n_byte_align_empty(chunk // 2 + 1, 16, 'complex128')
            outflux = pyfftw.n_byte_align_empty(chunk, 16, 'float64')
            fft_object = pyfftw.FFTW(influx, FF, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'))
            ifft_object = pyfftw.FFTW(FF, outflux, flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'),
                                      direction='FFTW_BACKWARD')

            influx[:] = self.fl
            fft_object()

            ss = rfftfreq(len(self.wl), d=self.min_vc * C.c_kms_air)
            ss[0] = 0.01 #junk so we don't get a divide by zero error
            ub = 2. * np.pi * vsini * ss
            sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
            #set zeroth frequency to 1 separately (DC term)
            sb[0] = 1.

            #institute velocity taper
            FF *= sb

            #do ifft
            ifft_object()
            self.fl[:] = outflux
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



class DataSpectrum(BaseSpectrum):
    def __init__(self, wl, fl, sigma, mask=None, unit="f_lam"):
        super().__init__(wl, fl, unit)
        self.sigma = sigma

        if mask is None:
            self.mask = np.ones_like(self.wl, dtype='bool') #create mask of all True
        else:
            self.mask = mask

        assert self.sigma.shape == self.shape, "sigma array incompatible shape."
        assert self.mask.shape == self.shape, "mask array incompatible shape."

        #
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