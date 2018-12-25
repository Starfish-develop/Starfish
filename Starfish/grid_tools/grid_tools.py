import gc
import os
import bz2
import h5py
import itertools
from collections import OrderedDict

import numpy as np
from numpy.fft import fft, ifft, fftfreq, rfftfreq
from astropy.io import ascii, fits
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.integrate import trapz
from scipy.special import j1
import multiprocessing as mp
from tqdm import tqdm

from Starfish import config
from Starfish.spectrum import create_log_lam_grid, calculate_dv, calculate_dv_dict
from Starfish import constants as C


def chunk_list(mylist, n=mp.cpu_count()):
    '''
    Divide a lengthy parameter list into chunks for parallel processing and
    backfill if necessary.

    :param mylist: a lengthy list of parameter combinations
    :type mylist: 1-D list

    :param n: number of chunks to divide list into. Default is ``mp.cpu_count()``
    :type n: integer

    :returns: **chunks** (*2-D list* of shape (n, -1)) a list of chunked parameter lists.

    '''
    length = len(mylist)
    size = int(length / n)
    chunks = [mylist[0 + size * i: size * (i + 1)] for i in range(n)]  # fill with evenly divisible
    leftover = length - size * n
    edge = size * n
    for i in range(leftover):  # backfill each with the last item
        chunks[i % n].append(mylist[edge + i])
    return chunks


def determine_chunk_log(wl, wl_min, wl_max):
    '''
    Take in a wavelength array and then, given two minimum bounds, determine
    the boolean indices that will allow us to truncate this grid to near the
    requested bounds while forcing the wl length to be a power of 2.

    :param wl: wavelength array
    :type wl: np.ndarray
    :param wl_min: minimum required wavelength
    :type wl_min: float
    :param wl_max: maximum required wavelength
    :type wl_max: float

    :returns: a np.ndarray boolean array used to index into the wl array.

    '''

    # wl_min and wl_max must of course be within the bounds of wl
    assert wl_min >= np.min(wl) and wl_max <= np.max(
        wl), "determine_chunk_log: wl_min {:.2f} and wl_max {:.2f} are not within the bounds of the grid {:.2f} to {:.2f}.".format(
        wl_min, wl_max, np.min(wl), np.max(wl))

    # Find the smallest length synthetic spectrum that is a power of 2 in length
    # and longer than the number of points contained between wl_min and wl_max
    len_wl = len(wl)
    npoints = np.sum((wl >= wl_min) & (wl <= wl_max))
    chunk = len_wl
    inds = (0, chunk)

    # This loop will exit with chunk being the smallest power of 2 that is
    # larger than npoints
    while chunk > npoints:
        if chunk / 2 > npoints:
            chunk = chunk // 2
        else:
            break

    assert type(chunk) == np.int, "Chunk is not an integer!. Chunk is {}".format(chunk)

    if chunk < len_wl:
        # Now that we have determined the length of the chunk of the synthetic
        # spectrum, determine indices that straddle the data spectrum.

        # Find the index that corresponds to the wl at the center of the data spectrum
        center_wl = (wl_min + wl_max) / 2.
        center_ind = (np.abs(wl - center_wl)).argmin()

        # Take a chunk that straddles either side.
        inds = (center_ind - chunk // 2, center_ind + chunk // 2)

        ind = (np.arange(len_wl) >= inds[0]) & (np.arange(len_wl) < inds[1])
    else:
        print("keeping grid as is")
        ind = np.ones_like(wl, dtype='bool')

    assert (min(wl[ind]) <= wl_min) and (max(wl[ind]) >= wl_max), "Model" \
                                                                  "Interpolator chunking ({:.2f}, {:.2f}) didn't encapsulate full" \
                                                                  " wl range ({:.2f}, {:.2f}).".format(min(wl[ind]),
                                                                                                       max(wl[ind]),
                                                                                                       wl_min, wl_max)

    return ind




class IndexInterpolator:
    '''
    Object to return fractional distance between grid points of a single grid variable.

    :param parameter_list: list of parameter values
    :type parameter_list: 1-D list
    '''

    def __init__(self, parameter_list):
        self.parameter_list = np.unique(parameter_list)
        self.index_interpolator = interp1d(self.parameter_list, np.arange(len(self.parameter_list)), kind='linear')
        pass

    def __call__(self, value):
        '''
        Evaluate the interpolator at a parameter.

        :param value:
        :type value: float
        :raises C.InterpolationError: if *value* is out of bounds.

        :returns: ((low_val, high_val), (frac_low, frac_high)), the lower and higher bounding points in the grid
        and the fractional distance (0 - 1) between them and the value.
        '''
        try:
            index = self.index_interpolator(value)
        except ValueError as e:
            raise C.InterpolationError("Requested value {} is out of bounds. {}".format(value, e))
        high = np.ceil(index)
        low = np.floor(index)
        frac_index = index - low
        return ((self.parameter_list[low], self.parameter_list[high]), ((1 - frac_index), frac_index))


class Interpolator:
    '''
    Quickly and efficiently interpolate a synthetic spectrum for use in an MCMC
    simulation. Caches spectra for easier memory load.

    :param interface: :obj:`HDF5Interface` (recommended) or :obj:`RawGridInterface` to load spectra
    :param wl: data wavelength of the region you are trying to fit. Used to truncate the grid for speed.
    :type DataSpectrum: np.array
    :param cache_max: maximum number of spectra to hold in cache
    :type cache_max: int
    :param cache_dump: how many spectra to purge from the cache once :attr:`cache_max` is reached
    :type cache_dump: int

    '''

    def __init__(self, wl, interface, cache_max=256, cache_dump=64):

        self.interface = interface

        self.wl = self.interface.wl
        self.dv = self.interface.dv
        self.npars = len(config.grid["parname"])
        self._determine_chunk_log(wl)

        self.setup_index_interpolators()
        self.cache = OrderedDict([])
        self.cache_max = cache_max
        self.cache_dump = cache_dump  # how many to clear once the maximum cache has been reached

    def _determine_chunk_log(self, wl):
        '''
        Using the DataSpectrum, determine the minimum chunksize that we can use and then
        truncate the synthetic wavelength grid and the returned spectra.

        Assumes HDF5Interface is LogLambda spaced, because otherwise you shouldn't need a grid
        with 2^n points, because you would need to interpolate in wl space after this anyway.
        '''

        wl_interface = self.interface.wl  # The grid we will be truncating.
        wl_min, wl_max = np.min(wl), np.max(wl)

        # Previously this routine retuned a tuple () which was the ranges to truncate to.
        # Now this routine returns a Boolean mask,
        # so we need to go and find the first and last true values
        ind = determine_chunk_log(wl_interface, wl_min, wl_max)
        self.wl = self.wl[ind]

        # Find the index of the first and last true values
        self.interface.ind = np.argwhere(ind)[0][0], np.argwhere(ind)[-1][0] + 1

    def _determine_chunk(self):
        '''
        Using the DataSpectrum, set the bounds of the interpolator to +/- 5 Ang
        '''

        wave_grid = self.interface.wl
        wl_min, wl_max = np.min(self.dataSpectrum.wls), np.max(self.dataSpectrum.wls)

        ind_low = (np.abs(wave_grid - (wl_min - 5.))).argmin()
        ind_high = (np.abs(wave_grid - (wl_max + 5.))).argmin()

        self.wl = self.wl[ind_low:ind_high]

        assert min(self.wl) < wl_min and max(
            self.wl) > wl_max, "ModelInterpolator chunking ({:.2f}, {:.2f}) didn't encapsulate full DataSpectrum range ({:.2f}, {:.2f}).".format(
            min(self.wl), max(self.wl), wl_min, wl_max)

        self.interface.ind = (ind_low, ind_high)
        print("Wl is {}".format(len(self.wl)))

    def __call__(self, parameters):
        '''
        Interpolate a spectrum

        :param parameters: stellar parameters
        :type parameters: dict

        Automatically pops :attr:`cache_dump` items from cache if full.
        '''
        if len(self.cache) > self.cache_max:
            [self.cache.popitem(False) for i in range(self.cache_dump)]
            self.cache_counter = 0
        return self.interpolate(parameters)

    def setup_index_interpolators(self):
        # create an interpolator between grid points indices.
        # Given a parameter value, produce fractional index between two points
        # Store the interpolators as a list
        self.index_interpolators = [IndexInterpolator(self.interface.points[i]) for i in range(self.npars)]

        lenF = self.interface.ind[1] - self.interface.ind[0]
        self.fluxes = np.empty((2 ** self.npars, lenF))  # 8 rows, for temp, logg, Z

    def interpolate(self, parameters):
        '''
        Interpolate a spectrum without clearing cache. Recommended to use :meth:`__call__` instead.

        :param parameters: grid parameters
        :type parameters: np.array
        :raises C.InterpolationError: if parameters are out of bounds.

        '''
        # Previously, parameters was a dictionary of the stellar parameters.
        # Now that we have moved over to arrays, it is a numpy array.
        try:
            edges = []
            for i in range(self.npars):
                edges.append(self.index_interpolators[i](parameters[i]))
        except C.InterpolationError as e:
            raise C.InterpolationError("Parameters {} are out of bounds. {}".format(parameters, e))

        # Edges is a list of [((6000, 6100), (0.2, 0.8)), ((), ()), ((), ())]

        params = [tup[0] for tup in edges]  # [(6000, 6100), (4.0, 4.5), ...]
        weights = [tup[1] for tup in edges]  # [(0.2, 0.8), (0.4, 0.6), ...]

        # Selects all the possible combinations of parameters
        param_combos = list(itertools.product(*params))
        # [(6000, 4.0, 0.0), (6100, 4.0, 0.0), (6000, 4.5, 0.0), ...]
        weight_combos = list(itertools.product(*weights))
        # [(0.2, 0.4, 1.0), (0.8, 0.4, 1.0), ...]

        # Assemble key list necessary for indexing cache
        key_list = [self.interface.key_name.format(*param) for param in param_combos]
        weight_list = np.array([np.prod(weight) for weight in weight_combos])

        assert np.allclose(np.sum(weight_list), np.array(1.0)), "Sum of weights must equal 1, {}".format(
            np.sum(weight_list))

        # Assemble flux vector from cache, or load into cache if not there
        for i, param in enumerate(param_combos):
            key = key_list[i]
            if key not in self.cache.keys():
                try:
                    # This method already allows loading only the relevant region from HDF5
                    fl = self.interface.load_flux(np.array(param))
                except KeyError as e:
                    raise C.InterpolationError("Parameters {} not in master HDF5 grid. {}".format(param, e))
                self.cache[key] = fl

            self.fluxes[i, :] = self.cache[key] * weight_list[i]

        # Do the averaging and then normalize the average flux to 1.0
        fl = np.sum(self.fluxes, axis=0)
        fl /= np.median(fl)
        return fl



def vacuum_to_air(wl):
    '''
    Converts vacuum wavelengths to air wavelengths using the Ciddor 1996 formula.

    :param wl: input vacuum wavelengths
    :type wl: np.array

    :returns: **wl_air** (*np.array*) - the wavelengths converted to air wavelengths

    .. note::

        CA Prieto recommends this as more accurate than the IAU standard.'''

    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    return wl / f


def calculate_n(wl):
    '''
    Calculate *n*, the refractive index of light at a given wavelength.

    :param wl: input wavelength (in vacuum)
    :type wl: np.array

    :return: **n_air** (*np.array*) - the refractive index in air at that wavelength
    '''
    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    new_wl = wl / f
    n = wl / new_wl
    print(n)


def vacuum_to_air_SLOAN(wl):
    '''
    Converts vacuum wavelengths to air wavelengths using the outdated SLOAN definition.

    :param wl:
        The input wavelengths to convert

    From the SLOAN website:

    AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)'''
    air = wl / (1.0 + 2.735182E-4 + 131.4182 / wl ** 2 + 2.76249E8 / wl ** 4)
    return air


def air_to_vacuum(wl):
    '''
    Convert air wavelengths to vacuum wavelengths.

    :param wl: input air wavelegths
    :type wl: np.array

    :return: **wl_vac** (*np.array*) - the wavelengths converted to vacuum.

    .. note::

        It is generally not recommended to do this, as the function is imprecise.
    '''
    sigma = 1e4 / wl
    vac = wl + wl * (6.4328e-5 + 2.94981e-2 / (146 - sigma ** 2) + 2.5540e-4 / (41 - sigma ** 2))
    return vac


def get_wl_kurucz(filename):
    '''The Kurucz grid is log-linear spaced.'''
    flux_file = fits.open(filename)
    hdr = flux_file[0].header
    num = len(flux_file[0].data)
    p = np.arange(num)
    w1 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wl = 10 ** (w1 + dw * p)
    return wl


@np.vectorize
def idl_float(idl_num):
    '''
    idl_float(idl_num)
    Convert an idl *string* number in scientific notation it to a float.

    :param idl_num:
        the idl number in sci_notation'''

    # replace 'D' with 'E', convert to float
    return np.float(idl_num.replace("D", "E"))


def load_BTSettl(temp, logg, Z, norm=False, trunc=False, air=False):
    rname = "BT-Settl/CIFIST2011/M{Z:}/lte{temp:0>3.0f}-{logg:.1f}{Z:}.BT-Settl.spec.7.bz2".format(temp=0.01 * temp,
                                                                                                   logg=logg, Z=Z)
    file = bz2.BZ2File(rname, 'r')

    lines = file.readlines()
    strlines = [line.decode('utf-8') for line in lines]
    file.close()

    data = ascii.read(strlines, col_starts=[0, 13], col_ends=[12, 25], Reader=ascii.FixedWidthNoHeader)
    wl = data['col1']
    fl_str = data['col2']

    fl = idl_float(fl_str)  # convert because of "D" exponent, unreadable in Python
    fl = 10 ** (fl - 8.)  # now in ergs/cm^2/s/A

    if norm:
        F_bol = trapz(fl, wl)
        fl = fl * (C.F_sun / F_bol)
        # this also means that the bolometric luminosity is always 1 L_sun

    if trunc:
        # truncate to only the wl of interest
        ind = (wl > 3000) & (wl < 13000)
        wl = wl[ind]
        fl = fl[ind]

    if air:
        wl = vacuum_to_air(wl)

    return [wl, fl]


def create_fits(filename, fl, CRVAL1, CDELT1, dict=None):
    '''Assumes that wl is already log lambda spaced'''

    hdu = fits.PrimaryHDU(fl)
    head = hdu.header
    head["DISPTYPE"] = 'log lambda'
    head["DISPUNIT"] = 'log angstroms'
    head["CRPIX1"] = 1.

    head["CRVAL1"] = CRVAL1
    head["CDELT1"] = CDELT1
    head["DC-FLAG"] = 1

    if dict is not None:
        for key, value in dict.items():
            head[key] = value

    hdu.writeto(filename)


class MasterToFITSIndividual:
    '''
    Object used to create one FITS file at a time.

    :param interpolator: an :obj:`Interpolator` object referenced to the master grid.
    :param instrument: an :obj:`Instrument` object containing the properties of the final spectra


    '''

    def __init__(self, interpolator, instrument):
        self.interpolator = interpolator
        self.instrument = instrument
        self.filename = "t{temp:0>5.0f}g{logg:0>2.0f}{Z_flag}{Z:0>2.0f}v{vsini:0>3.0f}.fits"

        # Create a master wl_dict which correctly oversamples the instrumental kernel
        self.wl_dict = self.instrument.wl_dict
        self.wl = self.wl_dict["wl"]

    def process_spectrum(self, parameters, out_unit, out_dir=""):
        '''
        Creates a FITS file with given parameters


        :param parameters: stellar parameters :attr:`temp`, :attr:`logg`, :attr:`Z`, :attr:`vsini`
        :type parameters: dict
        :param out_unit: output flux unit? Choices between `f_lam`, `f_nu`, `f_nu_log`, or `counts/pix`. `counts/pix` will do spline integration.
        :param out_dir: optional directory to prepend to output filename, which is chosen automatically for parameter values.

        Smoothly handles the *C.InterpolationError* if parameters cannot be interpolated from the grid and prints a message.
        '''

        # Preserve the "popping of parameters"
        parameters = parameters.copy()

        # Load the correct C.grid_set value from the interpolator into a LogLambdaSpectrum
        if parameters["Z"] < 0:
            zflag = "m"
        else:
            zflag = "p"

        filename = out_dir + self.filename.format(temp=parameters["temp"], logg=10 * parameters["logg"],
                                                  Z=np.abs(10 * parameters["Z"]), Z_flag=zflag,
                                                  vsini=parameters["vsini"])
        vsini = parameters.pop("vsini")
        try:
            spec = self.interpolator(parameters)
            # Using the ``out_unit``, determine if we should also integrate while doing the downsampling
            if out_unit == "counts/pix":
                integrate = True
            else:
                integrate = False
            # Downsample the spectrum to the instrumental resolution.
            spec.instrument_and_stellar_convolve(self.instrument, vsini, integrate)
            spec.write_to_FITS(out_unit, filename)
        except C.InterpolationError as e:
            print("{} cannot be interpolated from the grid.".format(parameters))

        print("Processed spectrum {}".format(parameters))


class MasterToFITSGridProcessor:
    '''
    Create one or many FITS files from a master HDF5 grid. Assume that we are not going to need to interpolate
    any values.

    :param interface: an :obj:`HDF5Interface` object referenced to the master grid.
    :param points: lists of output parameters (assumes regular grid)
    :type points: dict of lists
    :param flux_unit: format of output spectra {"f_lam", "f_nu", "ADU"}
    :type flux_unit: string
    :param outdir: output directory
    :param processes: how many processors to use in parallel

    Basically, this object is doing a one-to-one conversion of the PHOENIX spectra. No interpolation necessary,
    preserving all of the header keywords.

    '''

    def __init__(self, interface, instrument, points, flux_unit, outdir, alpha=False, integrate=False,
                 processes=mp.cpu_count()):
        self.interface = interface
        self.instrument = instrument
        self.points = points  # points is a dictionary with which values to spit out for each parameter
        self.filename = "t{temp:0>5.0f}g{logg:0>2.0f}{Z_flag}{Z:0>2.0f}v{vsini:0>3.0f}.fits"
        self.flux_unit = flux_unit
        self.integrate = integrate
        self.outdir = outdir
        self.processes = processes
        self.pids = []
        self.alpha = alpha

        self.vsini_points = self.points.pop("vsini")
        names = self.points.keys()

        # Creates a list of parameter dictionaries [{"temp":8500, "logg":3.5, "Z":0.0}, {"temp":8250, etc...}, etc...]
        # which does not contain vsini
        self.param_list = [dict(zip(names, params)) for params in itertools.product(*self.points.values())]

        # Create a master wl_dict which correctly oversamples the instrumental kernel
        self.wl_dict = self.instrument.wl_dict
        self.wl = self.wl_dict["wl"]

        # Check that temp, logg, Z are within the bounds of the interface
        for key, value in self.points.items():
            min_val, max_val = self.interface.bounds[key]
            assert np.min(self.points[key]) >= min_val, "Points below interface bound {}={}".format(key, min_val)
            assert np.max(self.points[key]) <= max_val, "Points above interface bound {}={}".format(key, max_val)

        # Create a temporary grid to resample to that matches the bounds of the instrument.
        low, high = self.instrument.wl_range
        self.temp_grid = create_log_lam_grid(wl_start=low, wl_end=high, min_vc=0.1)['wl']

    def process_spectrum_vsini(self, parameters):
        '''
        Create a set of FITS files with given stellar parameters temp, logg, Z and all combinations of `vsini`.

        :param parameters: stellar parameters
        :type parameters: dict

        Smoothly handles the *KeyError* if parameters cannot be drawn from the interface and prints a message.
        '''

        try:
            # Check to see if alpha, otherwise append alpha=0 to the parameter list.
            if not self.alpha:
                parameters.update({"alpha": 0.0})
            print(parameters)

            if parameters["Z"] < 0:
                zflag = "m"
            else:
                zflag = "p"

            # This is a Base1DSpectrum
            base_spec = self.interface.load_file(parameters)

            master_spec = base_spec.to_LogLambda(instrument=self.instrument,
                                                 min_vc=0.1 / C.c_kms)  # convert the Base1DSpectrum to a LogLamSpectrum

            # Now process the spectrum for all values of vsini

            for vsini in self.vsini_points:
                spec = master_spec.copy()
                # Downsample the spectrum to the instrumental resolution, integrate to give counts/pixel
                spec.instrument_and_stellar_convolve(self.instrument, vsini, integrate=self.integrate)

                # Update spectrum with vsini
                spec.metadata.update({"vsini": vsini})
                filename = self.outdir + self.filename.format(temp=parameters["temp"], logg=10 * parameters["logg"],
                                                              Z=np.abs(10 * parameters["Z"]), Z_flag=zflag,
                                                              vsini=vsini)

                spec.write_to_FITS(self.flux_unit, filename)

        except KeyError as e:
            print("{} cannot be loaded from the interface.".format(parameters))

    def process_chunk(self, chunk):
        '''
        Process a chunk of parameters to FITS

        :param chunk: stellar parameter dicts
        :type chunk: 1-D list
        '''
        print("Process {} processing chunk {}".format(os.getpid(), chunk))
        for param in chunk:
            self.process_spectrum_vsini(param)

    def process_all(self):
        '''
        Process all parameters in :attr:`points` to FITS by chopping them into chunks.
        '''
        print("Total of {} FITS files to create.".format(len(self.vsini_points) * len(self.param_list)))
        chunks = chunk_list(self.param_list, n=self.processes)
        for chunk in chunks:
            p = mp.Process(target=self.process_chunk, args=(chunk,))
            p.start()
            self.pids.append(p)

        for p in self.pids:
            # Make sure all threads have finished
            p.join()


def main():
    pass


if __name__ == "__main__":
    main()
