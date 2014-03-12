import numpy as np
from numpy.fft import fft, ifft, fftfreq
from astropy.io import ascii,fits
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.integrate import trapz
from scipy.special import j1
import multiprocessing as mp

import sys
import gc
import os
import bz2
import h5py
from functools import partial
import itertools
from collections import OrderedDict

from StellarSpectra.spectrum import Base1DSpectrum, LogLambdaSpectrum, create_log_lam_grid
import StellarSpectra.constants as C

def chunk_list(mylist, n=mp.cpu_count()):
    '''
    Divide a lengthy parameter list into chunks for parallel processing and backfill if necessary.

    :param mylist: a lengthy list of parameter combinations
    :type mylist: 1-D list

    :param n: number of chunks to divide list into. Default is ``mp.cpu_count()``
    :type n: integer

    :returns: **chunks** (*2-D list* of shape (n, -1)) a list of chunked parameter lists.

    '''
    length = len(mylist)
    size = int(length / n)
    chunks = [mylist[0+size*i : size*(i+1)] for i in range(n)] #fill with evenly divisible
    leftover = length - size*n
    edge = size*n
    for i in range(leftover): #backfill each with the last item
        chunks[i%n].append(mylist[edge+i])
    return chunks







class GridError(Exception):
    '''
    Raised when a spectrum cannot be found in the grid.
    '''
    def __init__(self, msg):
        self.msg = msg

class InterpolationError(Exception):
    '''
    Raised when the :obj:`Interpolator` or :obj:`IndexInterpolator` cannot properly interpolate a spectrum,
    usually grid bounds.
    '''
    def __init__(self, msg):
        self.msg = msg

class RawGridInterface:
    '''
    A base class to handle interfacing with synthetic spectral grids.

    :param name: name of the spectral library
    :param points: a dictionary of lists describing the grid points at which spectra exist (assumes grid is square, not ragged).
    :param air: Are the wavelengths in air?
    :type air: bool
    :param wl_range: the starting and ending wavelength ranges of the grid to truncate to.
    :type wl_range: list of len 2 [min, max]
    :param base: path to the root of the files on disk.
    :type base: string

    '''
    def __init__(self, name, points, air=True, wl_range=[3000,13000], base=None):
        self.name = name
        self.points = {}
        assert type(points) is dict, "points must be a dictionary."
        for key, value in points.items():
            if key in C.grid_parameters:
                self.points[key] = value
            else:
                raise KeyError("{0} is not an allowed parameter, skipping".format(key))

        self.air = air
        self.wl_range = wl_range
        self.base = base

    def check_params(self, parameters):
        '''
        Determine if a set of parameters is a subset of allowed parameters, and then determine if those parameters
        are allowed in the grid.

        :param parameters: parameter set to check
        :type parameters: dict

        :raises GridError: if parameters.keys() is not a subset of :data:`C.grid_parameters`
        :raises GridError: if the parameter values are outside of the grid bounds

        '''
        if not set(parameters.keys()) <= C.grid_parameters:
            raise GridError("{} not in allowable grid parameters {}".format(parameters.keys(), C.grid_parameters))

        for key,value in parameters.items():
            if value not in self.points[key]:
                raise GridError("{} not in the grid points {}".format(value, sorted(self.points[key])))

    def load_file(self, parameters, norm=True):
        '''
        Load a synthetic spectrum from disk and :meth:`check_params`

        :param parameters: stellar parameters describing a spectrum
        :type parameters: dict

         .. note::

            This method is designed to be extended by the inheriting class'''
        self.check_params(parameters)

class PHOENIXGridInterface(RawGridInterface):
    '''
    An Interface to the PHOENIX/Husser synthetic library.

    :param norm: normalize the spectrum to solar luminosity?
    :type norm: bool

    '''
    def __init__(self, air=True, norm=True, base="libraries/raw/PHOENIX/"):
        super().__init__(name="PHOENIX",
        points={"temp":
      np.array([2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 4000, 4100, 4200,
      4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100,
      6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000,
      9200, 9400, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 11200, 11400, 11600, 11800, 12000]),
        "logg":np.arange(0.0, 6.1, 0.5),
        "Z":np.arange(-1., 1.1, 0.5),
        "alpha":np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])},
        air=air, wl_range=[3000, 13000], base=base)

        self.norm = norm #Normalize to 1 solar luminosity?
        self.Z_dict = {-1: '-1.0', -0.5:'-0.5', 0.0: '-0.0', 0.5: '+0.5', 1: '+1.0'}
        self.alpha_dict = {-0.2:".Alpha=-0.20", 0.0: "", 0.2:".Alpha=+0.20", 0.4:".Alpha=+0.40", 0.6:".Alpha=+0.60",
                           0.8:".Alpha=+0.80"}

        #if air is true, convert the normally vacuum file to air wls.
        try:
            wl_file = fits.open(self.base + "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
        except OSError:
            raise GridError("Wavelength file improperly specified.")

        w_full = wl_file[0].data
        wl_file.close()
        if self.air:
            self.wl_full = vacuum_to_air(w_full)
        else:
            self.wl_full = w_full

        self.ind = (self.wl_full >= self.wl_range[0]) & (self.wl_full <= self.wl_range[1])
        self.wl = self.wl_full[self.ind]
        self.rname = self.base + "Z{Z:}{alpha:}/lte{temp:0>5.0f}-{logg:.2f}{Z:}{alpha:}" \
                     ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    def load_file(self, parameters):
        '''
        Load a file from the disk.

        :param parameters: stellar parameters
        :type parameters: dict

        :raises GridError: if the file cannot be found on disk.

        :returns: :obj:`model.Base1DSpectrum`
        '''
        super().load_file(parameters) #Check to make sure that the keys are allowed and that the values are in the grid

        str_parameters = parameters.copy()
        #Rewrite Z
        Z = parameters["Z"]
        str_parameters["Z"] = self.Z_dict[Z]

        #Rewrite alpha, allow alpha to be missing from parameters and set to 0
        try:
            alpha = parameters["alpha"]
        except KeyError:
            alpha = 0.0
            parameters["alpha"] = alpha
        str_parameters["alpha"] = self.alpha_dict[alpha]

        fname = self.rname.format(**str_parameters)

        #Still need to check that file is in the grid, otherwise raise a GridError
        #Read all metadata in from the FITS header, and append to spectrum
        try:
            flux_file = fits.open(fname)
            f = flux_file[0].data
            hdr = flux_file[0].header
            flux_file.close()
        except OSError:
            raise GridError("{} is not on disk.".format(fname))

        #If we want to normalize the spectra, we must do it now since later we won't have the full EM range
        if self.norm:
            f *= 1e-8 #convert from erg/cm^2/s/cm to erg/cm^2/s/A
            F_bol = trapz(f, self.wl_full)
            f = f * (C.F_sun / F_bol) #bolometric luminosity is always 1 L_sun

        #Add temp, logg, Z, alpha, norm to the metadata
        header = parameters
        header["norm"] = self.norm
        #Keep only the relevant PHOENIX keywords, which start with PHX
        for key, value in hdr.items():
            if key[:3] == "PHX":
                header[key] = value

        return Base1DSpectrum(self.wl, f[self.ind], metadata=header, air=self.air)

class KuruczGridInterface(RawGridInterface):
    '''Kurucz grid interface.'''
    def __init__(self):
        super().__init__("Kurucz", "Kurucz/",
        temp_points = np.arange(3500, 9751, 250),
        logg_points = np.arange(1.0, 5.1, 0.5),
        Z_points = np.arange(-0.5, 0.6, 0.5))

        self.Z_dict = {-0.5:"m05", 0.0:"p00", 0.5:"p05"}
        self.wl_full = np.load("wave_grids/kurucz_raw.npy")
        self.rname = None

    def load_file(self, temp, logg, Z):
        '''Includes an interface that can map a queried number to the actual string'''
        super().load_file(temp, logg, Z)

class BTSettlGridInterface(RawGridInterface):
    '''BTSettl grid interface.'''
    def __init__(self):
        pass

class HDF5GridCreator:
    '''Create a HDF5 grid to store all of the spectra from a RawGridInterface along with metadata.

    :param GridInterface: :obj:`RawGridInterface` object or subclass thereof to access raw spectra on disk.
    :param filename: where to create the HDF5 file. Suffix `*`.hdf5 recommended.
    :param wl_dict: a dictionary containing the wavelength and metadata describing the common wavelength.
    :type wl_dict: dict
    :param ranges: lower and upper limits for each stellar parameter, in order to truncate the number of spectra in the grid.
    :type ranges: dict of keywords mapped to 2-tuples
    :param nprocesses: if > 1, run in parallel using nprocesses
    :type nprocesses: int
    :param chunksize: chunksize to use for lazy mp.imap
    :type chunksize: int

    Once initialized, the HDF5Interface creates an HDF5 file and then closes it.

    '''
    def __init__(self, GridInterface, filename, wl_dict, ranges={"temp":(0,np.inf),
                 "logg":(-np.inf,np.inf), "Z":(-np.inf, np.inf), "alpha":(-np.inf, np.inf)},
                 nprocesses = 1, chunksize=1):
        self.GridInterface = GridInterface
        self.filename = filename #only store the name to the HDF5 file, because the object cannot be parallelized
        self.flux_name = "t{temp:.0f}g{logg:.1f}z{Z:.1f}a{alpha:.1f}"
        self.nprocesses = nprocesses
        self.chunksize = chunksize

        #Take only those points of the GridInterface that fall within the ranges specified
        self.points = {}
        for key, value in ranges.items():
            valid_points  = self.GridInterface.points[key]
            low,high = value
            ind = (valid_points >= low) & (valid_points <= high)
            self.points[key] = valid_points[ind]

        #wl_dict is the output from create_log_lam_grid, containing CRVAL1, CDELT1, etc...
        wl_dict = wl_dict.copy()
        self.wl = wl_dict.pop("wl")
        self.wl_params = wl_dict

        with h5py.File(self.filename, "w") as hdf5:
            hdf5.attrs["grid_name"] = GridInterface.name
            hdf5.flux_group = hdf5.create_group("flux")
            hdf5.flux_group.attrs["unit"] = "erg/cm^2/s/A"
            self.create_wl(hdf5)

        #The HDF5 master grid will always have alpha in the name, regardless of whether GridIterface uses it.

    def create_wl(self, hdf5):
        '''
        Creates the master wavelength as a dataset within an HDF5 file using the information in :attr:`wl_dict`

        :param hdf5: the hdf5 file which will store the wavelength
        :type hdf5: an h5py HDF5 file object

        '''
        #f8 is necessary since we will want to be converting to velocity space later, requiring great precision.
        wl_dset = hdf5.create_dataset("wl", (len(self.wl),), dtype="f8", compression='gzip', compression_opts=9)
        wl_dset[:] = self.wl
        for key, value in self.wl_params.items():
            wl_dset.attrs[key] = value
            wl_dset.attrs["air"] = self.GridInterface.air

    def process_flux(self, parameters):
        '''
        Processs a spectrum from the raw grid so that it is suitable to be inserted into the HDF5 file.

        :param parameters: the stellar parameters
        :type parameters: dict

        .. note::

           This function assumes that it's going to get parameters (temp, logg, Z, alpha), regardless of whether
           the :attr:`GridInterface` actually has alpha or not

        :raises AssertionError: if the `param parameters` dictionary is not length 4.
        '''
        assert len(parameters.keys()) == 4, "Must pass dictionary with keys (temp, logg, Z, alpha)"
        print("Processing", parameters)
        try:
            spec = self.GridInterface.load_file(parameters)
            spec.resample_to_grid(self.wl)
            sys.stdout.flush()
            return (parameters,spec)

        except GridError as e:
            print("No file with parameters {}. GridError: {}".format(parameters, e))
            sys.stdout.flush()
            return (None,None)

    def process_grid(self):
        '''
        Run :meth:`process_flux` for all of the spectra within the `ranges` and store the processed spectra in the
        HDF5 file.

        Executed in parallel if :attr:`nprocess` > 1.
        '''
        #Take all parameter permutations in self.points and create a list
        param_list = [] #list of parameter dictionaries
        keys,values = self.points.keys(),self.points.values()

        #use itertools.product to create permutations of all possible values
        for i in itertools.product(*values):
            param_list.append(dict(zip(keys,i)))

        if self.nprocesses > 1:
            pool = mp.Pool(self.nprocesses)
            M = lambda x,y : pool.imap_unordered(x, y, chunksize=self.chunksize)
        else:
            M = map

        for parameters, spec in M(self.process_flux, param_list): #lazy map
            if parameters is None:
                continue
            with h5py.File(self.filename, "r+") as hdf5:
                flux = hdf5["flux"].create_dataset(self.flux_name.format(**parameters), shape=(len(spec.fl),),
                                                      dtype="f", compression='gzip', compression_opts=9)
                flux[:] = spec.fl

                #Store header keywords as attributes in HDF5 file
                for key,value in spec.metadata.items():
                    flux.attrs[key] = value

class HDF5Interface:
    '''
    Connect to an HDF5 file that stores spectra.

    :param filename: the name of the HDF5 file
    :type param: string

    '''
    def __init__(self, filename):
        self.filename = filename
        self.flux_name = "t{temp:.0f}g{logg:.1f}z{Z:.1f}a{alpha:.1f}"

        with h5py.File(self.filename, "r") as hdf5:
            self.name = hdf5.attrs["grid_name"]
            self.wl = hdf5["wl"][:]
            self.wl_header = dict(hdf5["wl"].attrs.items())

            grid_points = []
            for key in hdf5["flux"].keys():
                #assemble all temp, logg, Z, alpha keywords into a giant list
                hdr = hdf5['flux'][key].attrs
                grid_points.append({k: hdr[k] for k in C.grid_parameters})
            self.list_grid_points = grid_points

        #determine the bounding regions of the grid by sorting the grid_points
        temp, logg, Z, alpha = [],[],[],[]
        for param in self.list_grid_points:
            temp.append(param['temp'])
            logg.append(param['logg'])
            Z.append(param['Z'])
            alpha.append(param['alpha'])

        self.bounds = {"temp": (min(temp),max(temp)), "logg": (min(logg), max(logg)), "Z": (min(Z), max(Z)),
        "alpha":(min(alpha),max(alpha))}
        self.points = {"temp": np.unique(temp), "logg": np.unique(logg), "Z": np.unique(Z), "alpha": np.unique(alpha)}
        self.ind = None #Overwritten by other methods using this as part of a ModelInterpolator

    def load_file(self, parameters):
        '''load a spectrum from the grid and return it as a :obj:`model.LogLambdaSpectrum`.

        :param parameters: the stellar parameters
        :type parameters: dict

        :raises KeyError: if spectrum is not found in the HDF5 file.

        .. note::

            Assumes that the wavelength file in the HDF5 file is in LogLambda format.'''

        key = self.flux_name.format(**parameters)
        with h5py.File(self.filename, "r") as hdf5:
            fl = hdf5['flux'][key][:]
            hdr = dict(hdf5['flux'][key].attrs)

        #Note: will raise a KeyError if the file is not found.

        hdr.update(self.wl_header) #add the flux metadata to the wl data

        return LogLambdaSpectrum(self.wl, fl, metadata=hdr)

    def load_flux(self, parameters):
        '''
        Load *just the flux* from the grid, with possibly a index truncation. This is a speed method designed for MCMC.

        :param parameters: the stellar parameters
        :type parameters: dict
        :param ind: slice to load
        :type ind: (int low, int high) 2-tuple

        :raises KeyError: if spectrum is not found in the HDF5 file.
        '''
        key = self.flux_name.format(**parameters)
        with h5py.File(self.filename, "r") as hdf5:
            if self.ind is not None:
                fl = hdf5['flux'][key][self.ind[0]:self.ind[1]]
            else:
                fl = hdf5['flux'][key][:]

        #Note: will raise a KeyError if the file is not found.

        return fl


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
        :raises InterpolationError: if *value* is out of bounds.

        :returns: ((low_val, high_val), (frac_low, frac_high)), the lower and higher bounding points in the grid
        and the fractional distance (0 - 1) between them and the value.
        '''
        try:
            index = self.index_interpolator(value)
        except ValueError as e:
            raise InterpolationError("Requested value {} is out of bounds. {}".format(value, e))
        high = np.ceil(index)
        low = np.floor(index)
        frac_index = index - low
        return ((self.parameter_list[low], self.parameter_list[high]), ((1 - frac_index), frac_index))

class Interpolator:
    '''
    An object which generates spectra interpolated from a grid.
    It is able to cache spectra for easier memory load.

    :param interface: :obj:`HDF5Interface` (recommended) or :obj:`RawGridInterface` to load spectra
    :param cache_max: maximum number of spectra to hold in cache
    :type cache_max: int
    :param cache_dump: how many spectra to purge from the cache once :attr:`cache_max` is reached
    :type cache_dump: int
    :param avg_hdr_keys: header keys whose properties can be averaged when combining spectra.
    :type avg_hdr_keys: 1-D list, tuple, or set.

    '''
    def __init__(self, interface, cache_max=256, cache_dump=64, avg_hdr_keys=None):
        self.interface = interface

        #If alpha only includes one value, then do trilinear interpolation
        (alow, ahigh) = self.interface.bounds['alpha']
        if alow == ahigh:
            self.parameters = C.grid_parameters - set(("alpha",))
        else:
            self.parameters = C.grid_parameters

        self.avg_hdr_keys = {} if avg_hdr_keys is None else avg_hdr_keys #These avg_hdr_keys specify the ones to average over

        self.setup_index_interpolators()
        self.hdr_cache = OrderedDict([])
        self.cache = OrderedDict([])
        self.cache_max = cache_max
        self.cache_dump = cache_dump #how many to clear once the maximum cache has been reached
        self.wl = self.interface.wl
        self.wl_dict = self.interface.wl_header

    def __call__(self, parameters):
        '''
        Interpolate a spectrum

        :param parameters: stellar parameters
        :type parameters: dict

        Automatically pops :attr:`cache_dump` items from cache if full.
        '''
        if len(self.cache) > self.cache_max:
            [(self.cache.popitem(False), self.hdr_cache.popitem(False)) for i in range(self.cache_dump)]
            self.cache_counter = 0
        return self.interpolate(parameters)


    def setup_index_interpolators(self):
        #create an interpolator between grid points indices. Given a temp, produce fractional index between two points
        self.index_interpolators = {key:IndexInterpolator(self.interface.points[key]) for key in self.parameters}

        lenF = len(self.interface.wl)
        self.fluxes = np.empty((2**len(self.parameters), lenF))

    def interpolate(self, parameters):
        '''
        Interpolate a spectrum without clearing cache. Recommended to use :meth:`__call__` instead.

        :param parameters: stellar parameters
        :type parameters: dict
        :raises InterpolationError: if parameters are out of bounds.
        '''

        try:
            edges = {key:self.index_interpolators[key](value) for key,value in parameters.items()}
        except InterpolationError as e:
            raise InterpolationError("Parameters {} are out of bounds. {}".format(parameters, e))

        #Edges is a dictionary of {"temp": ((6000, 6100), (0.2, 0.8)), "logg": (())..}
        names = [key for key in edges.keys()]
        params = [edges[key][0] for key in names]
        weights = [edges[key][1] for key in names]

        param_combos = itertools.product(*params)
        weight_combos = itertools.product(*weights)

        parameter_list = [dict(zip(names, param)) for param in param_combos]
        if "alpha" not in parameters.keys():
            [param.update({"alpha":C.var_default["alpha"]}) for param in parameter_list]
        key_list = [self.interface.flux_name.format(**param) for param in parameter_list]
        weight_list = np.array([np.prod(weight) for weight in weight_combos])
        #For each spectrum, want to extract a {"temp":5000, "logg":4.5, "Z":0.0, "alpha":0.0} and weight= 0.1 * 0.4 * .05 * 0.1

        assert np.allclose(np.sum(weight_list), np.array(1.0)), "Sum of weights must equal 1, {}".format(np.sum(weight_list))

        #Assemble flux vector from cache
        for i,param in enumerate(parameter_list):
            key = key_list[i]
            if key not in self.cache.keys():
                try:
                    spec = self.interface.load_file(param)
                except KeyError as e:
                    raise InterpolationError("Parameters {} not in master HDF5 grid. {}".format(param, e))
                self.cache[key] = spec.fl
                self.hdr_cache[key] = spec.metadata
                #Note: if we are dealing with a ragged grid, a GridError will be raised here because a Z=+1, alpha!=0 spectrum can't be found.
            self.fluxes[i,:] = self.cache[key]*weight_list[i]

        comb_metadata = self.wl_dict.copy()
        if "alpha" not in parameters.keys():
            parameters.update({"alpha":C.var_default["alpha"]})
        comb_metadata.update(parameters)



        for hdr_key in self.avg_hdr_keys:
            try:
                values = np.array([self.hdr_cache[key][hdr_key] for key in key_list])
                try:
                    value = np.average(values, weights=weight_list)
                except TypeError:
                    value = values[0]
            except KeyError:
                value = None
                continue

            comb_metadata[hdr_key] = value

        return LogLambdaSpectrum(self.wl, np.sum(self.fluxes, axis=0), metadata=comb_metadata)


class ModelInterpolator:
    '''
    Quickly and efficiently interpolate a synthetic spectrum for use in an MCMC simulation. Caches spectra for
    easier memory load.

    :param interface: :obj:`HDF5Interface` (recommended) or :obj:`RawGridInterface` to load spectra
    :param DataSpectrum: data spectrum that you are trying to fit. Used for truncating the synthetic spectra to the relevant region for speed.
    :type DataSpectrum: :obj:`spectrum.DataSpectrum`
    :param cache_max: maximum number of spectra to hold in cache
    :type cache_max: int
    :param cache_dump: how many spectra to purge from the cache once :attr:`cache_max` is reached
    :type cache_dump: int
    :param trilinear: Should this interpolate in temp, logg, and [Fe/H] AND [alpha/Fe], or just the first three parameters.
    :type trilinear: bool

    Setting :attr:`trilinear` to **True** is useful for when you want to do a run with [Fe/H] > 0.0

    '''

    def __init__(self, interface, DataSpectrum, cache_max=256, cache_dump=64, trilinear=False):
        self.interface = interface
        self.DataSpectrum = DataSpectrum

        #If alpha only includes one value, then do trilinear interpolation
        (alow, ahigh) = self.interface.bounds['alpha']
        if (alow == ahigh) or trilinear:
            self.parameters = C.grid_parameters - set(("alpha",))
        else:
            self.parameters = C.grid_parameters


        self.wl = self.interface.wl
        self.wl_dict = self.interface.wl_header
        self._determine_chunk()

        self.setup_index_interpolators()
        self.cache = OrderedDict([])
        self.cache_max = cache_max
        self.cache_dump = cache_dump #how many to clear once the maximum cache has been reached


    def _determine_chunk(self):
        '''
        Using the DataSpectrum, determine the minimum chunksize that we can use and then truncate the synthetic
        wavelength grid and the returned spectra.
        '''

        wave_grid = self.interface.wl
        wl_min, wl_max = np.min(self.DataSpectrum.wls), np.max(self.DataSpectrum.wls)
        #Length of the raw synthetic spectrum
        len_wg = len(wave_grid)
        #ind_wg = np.arange(len_wg) #Labels of pixels
        #Length of the data
        len_data = np.sum((self.wl > wl_min) & (self.wl < wl_max)) #How much of the synthetic spectrum do we need?

        #Find the smallest length synthetic spectrum that is a power of 2 in length and larger than the data spectrum
        chunk = len_wg
        assert chunk % 2 == 0, "spectrum is not a power of 2 to start!"
        while chunk > len_data:
            if chunk/2 > len_data:
                chunk = chunk//2
            else:
                break

        assert type(chunk) == np.int, "Chunk is no longer integer!. Chunk is {}".format(chunk)

        if chunk < len_wg:
            # Now that we have determined the length of the chunk of the synthetic spectrum, determine indices
            # that straddle the data spectrum.

            # What index corresponds to the wl at the center of the data spectrum?
            median_wl = np.median(self.DataSpectrum.wls)
            median_ind = (np.abs(wave_grid - median_wl)).argmin()

            #Take the chunk that straddles either side.
            ind = (median_ind - chunk//2, median_ind + chunk//2)

            self.wl = self.wl[ind[0]:ind[1]]
            assert min(self.wl) < wl_min and max(self.wl) > wl_max, "ModelInterpolator chunking ({:.2f}, {:.2f}) " \
                "didn't encapsulate full DataSpectrum range ({:.2f}, {:.2f}).".format(min(self.wl),
                                                                                  max(self.wl), wl_min, wl_max)

            self.interface.ind = ind
        else:
            self.interface.ind = None



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
        #create an interpolator between grid points indices. Given a temp, produce fractional index between two points
        self.index_interpolators = {key:IndexInterpolator(self.interface.points[key]) for key in self.parameters}

        lenF = self.interface.ind[1] - self.interface.ind[0]
        self.fluxes = np.empty((2**len(self.parameters), lenF))

    def interpolate(self, parameters):
        '''
        Interpolate a spectrum without clearing cache. Recommended to use :meth:`__call__` instead.

        :param parameters: stellar parameters
        :type parameters: dict
        :raises InterpolationError: if parameters are out of bounds.
        '''

        try:
            edges = {key:self.index_interpolators[key](value) for key,value in parameters.items()}
        except InterpolationError as e:
            raise InterpolationError("Parameters {} are out of bounds. {}".format(parameters, e))

        #Edges is a dictionary of {"temp": ((6000, 6100), (0.2, 0.8)), "logg": (())..}
        names = [key for key in edges.keys()]
        params = [edges[key][0] for key in names]
        weights = [edges[key][1] for key in names]

        param_combos = itertools.product(*params)
        weight_combos = itertools.product(*weights)

        parameter_list = [dict(zip(names, param)) for param in param_combos]
        if "alpha" not in parameters.keys():
            [param.update({"alpha":C.var_default["alpha"]}) for param in parameter_list]
        key_list = [self.interface.flux_name.format(**param) for param in parameter_list]
        weight_list = np.array([np.prod(weight) for weight in weight_combos])
        #For each spectrum, want to extract a {"temp":5000, "logg":4.5, "Z":0.0, "alpha":0.0} and weight= 0.1 * 0.4 * .05 * 0.1

        assert np.allclose(np.sum(weight_list), np.array(1.0)), "Sum of weights must equal 1, {}".format(np.sum(weight_list))

        #Assemble flux vector from cache
        for i,param in enumerate(parameter_list):
            key = key_list[i]
            if key not in self.cache.keys():
                try:
                    fl = self.interface.load_flux(param) #This method allows loading only the relevant region from HDF5
                except KeyError as e:
                    raise InterpolationError("Parameters {} not in master HDF5 grid. {}".format(param, e))
                self.cache[key] = fl
                #Note: if we are dealing with a ragged grid, a GridError will be raised here because a Z=+1, alpha!=0 spectrum can't be found.
            self.fluxes[i,:] = self.cache[key]*weight_list[i]

        return np.sum(self.fluxes, axis=0)



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

        #Create a master wl_dict which correctly oversamples the instrumental kernel
        self.wl_dict = self.instrument.wl_dict
        self.wl = self.wl_dict["wl"]



    def process_spectrum(self, parameters, out_unit, out_dir=""):
        '''
        Creates a FITS file with given parameters

        :param parameters: stellar parameters :attr:`temp`, :attr:`logg`, :attr:`Z`, :attr:`vsini`
        :type parameters: dict
        :param out_unit: output flux unit? Choices between `f_lam`, `f_nu`, `f_nu_log`, or `counts/pix`. `counts/pix` will do spline integration.
        :param out_dir: optional directory to prepend to output filename, which is chosen automatically for parameter values.

        Smoothly handles the *InterpolationError* if parameters cannot be interpolated from the grid and prints a message.
        '''

        #Preserve the "popping of parameters"
        parameters = parameters.copy()

        #Load the correct C.grid_parameters value from the interpolator into a LogLambdaSpectrum
        if parameters["Z"] < 0:
            zflag = "m"
        else:
            zflag = "p"

        filename = out_dir + self.filename.format(temp=parameters["temp"], logg=10*parameters["logg"],
                                    Z=np.abs(10*parameters["Z"]), Z_flag=zflag, vsini=parameters["vsini"])
        vsini = parameters.pop("vsini")
        try:
            spec = self.interpolator(parameters)
            # Using the ``out_unit``, determine if we should also integrate while doing the downsampling
            if out_unit=="counts/pix":
                integrate=True
            else:
                integrate=False
            # Downsample the spectrum to the instrumental resolution.
            spec.instrument_and_stellar_convolve(self.instrument, vsini, integrate)
            spec.write_to_FITS(out_unit, filename)
        except InterpolationError as e:
            print("{} cannot be interpolated from the grid.".format(parameters))

        print("Processed spectrum {}".format(parameters))

class MasterToFITSGridProcessor:
    '''
    Create one or many FITS files from a master HDF5 grid.

    :param interpolator: an :obj:`Interpolator` object referenced to the master grid.
    :param instrument: an :obj:`Instrument` object containing the properties of the final spectra
    :param points: lists of output parameters (assumes regular grid)
    :type points: dict of lists
    :param flux_unit: format of output spectra {"f_lam", "f_nu", "ADU"}
    :type flux_unit: string
    :param outdir: output directory
    :param processes: how many processors to use in parallel

    '''
    def __init__(self, interpolator, instrument, points, flux_unit, outdir, integrate=False, processes=mp.cpu_count()):
        self.interpolator = interpolator
        self.instrument = instrument
        self.points = points #points is a dictionary with which values to spit out
        self.filename = "t{temp:0>5.0f}g{logg:0>2.0f}{Z_flag}{Z:0>2.0f}v{vsini:0>3.0f}.fits"
        self.flux_unit = flux_unit
        self.integrate = integrate
        self.outdir = outdir
        self.processes = processes
        self.pids = []

        self.vsini_points = self.points.pop("vsini")
        names = self.points.keys()

        #Creates a list of parameter dictionaries [{"temp":8500, "logg":3.5, "Z":0.0}, {"temp":8250, etc...}, etc...]
        #Does not contain vsini
        self.param_list = [dict(zip(names,params)) for params in itertools.product(*self.points.values())]

        #Create a master wl_dict which correctly oversamples the instrumental kernel
        self.wl_dict = self.instrument.wl_dict
        self.wl = self.wl_dict["wl"]

        #Check that temp, logg, Z are within bounds
        for key,value in self.points.items():
            min_val, max_val = self.interpolator.interface.bounds[key]
            assert np.min(self.points[key]) >= min_val,"Points below interpolator bound {}={}".format(key, min_val)
            assert np.max(self.points[key]) <= max_val,"Points above interpolator bound {}={}".format(key, max_val)


    def process_spectrum_vsini(self, parameters):
        '''
        Creates a FITS file with given parameters (not including *vsini*).

        :param parameters: stellar parameters
        :type parameters: dict

        Smoothly handles the *InterpolationError* if parameters cannot be interpolated from the grid and prints a message.
        '''

        #Load the correct C.grid_parameters value from the interpolator into a LogLambdaSpectrum
        try:
            master_spec = self.interpolator(parameters)
            #Now process the spectrum for all values of vsini
            for vsini in self.vsini_points:
                spec = master_spec.copy()
                #Downsample the spectrum to the instrumental resolution, integrate to give counts/pixel
                spec.instrument_and_stellar_convolve(self.instrument, vsini, integrate=self.integrate)
                self.write_to_FITS(spec)
        except InterpolationError as e:
            print("{} cannot be interpolated from the grid.".format(parameters))

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
        chunks = chunk_list(self.param_list, n=self.processes)
        for chunk in chunks:
            p = mp.Process(target=self.process_chunk, args=(chunk,))
            p.start()
            self.pids.append(p)

        for p in self.pids:
            #Make sure all threads have finished
            p.join()


class Instrument:
    '''
    Object describing an instrument. This will be used by other methods for processing raw synthetic spectra.

    :param name: name of the instrument
    :type name: string
    :param FWHM: the FWHM of the instrumental profile in km/s
    :type FWHM: float
    :param wl_range: wavelength range of instrument
    :type wl_range: 2-tuple (low, high)
    :param oversampling: how many samples fit across the :attr:`FWHM`
    :type oversampling: float

    upon initialization, calculates a ``wl_dict`` with the properties of the instrument.
    '''
    def __init__(self, name, FWHM, wl_range, oversampling=3.5):
        self.name = name
        self.FWHM = FWHM #km/s
        self.oversampling = oversampling
        self.wl_range = wl_range

        self.wl_dict = create_log_lam_grid(*self.wl_range, min_vc=self.FWHM/(self.oversampling * C.c_kms))
        #Take the starting and ending wavelength ranges, the FWHM,
        # and oversampling value and generate an outwl grid  that can be resampled to.

    def __str__(self):
        '''
        Prints the relevant properties of the instrument.
        '''
        return "Instrument Name: {}, FWHM: {:.1f}, oversampling: {}, wl_range: {}".format(self.name, self.FWHM,
                                                                              self.oversampling, self.wl_range)

class TRES(Instrument):
    '''TRES instrument'''
    def __init__(self, name="TRES", FWHM=6.8, wl_range=(3500, 9500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)
        #sets the FWHM and wl_range

class TRESPhotometry(Instrument):
    '''This one has a wider wl range to allow for synthetic photometry comparisons.'''
    def __init__(self, name="TRES", FWHM=6.8, wl_range=(3000, 13000)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)
        #sets the FWHM and wl_range

class Reticon(Instrument):
    '''Reticon Instrument'''
    def __init__(self, name="Reticon", FWHM=8.5, wl_range=(5150,5250)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class KPNO(Instrument):
    '''KNPO Instrument'''
    def __init__(self, name="KPNO", FWHM=14.4, wl_range=(6200,6700)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)



#wl_file = fits.open("raw_grids/PHOENIX/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
#w_full = wl_file[0].data
#wl_file.close()
#ind = (w_full > 3000.) & (w_full < 13000.) #this corresponds to some extra space around the
# shortest U and longest z band

#global w
#w = w_full[ind]
#len_p = len(w)

#wave_grid_raw_PHOENIX = np.load("wave_grids/PHOENIX_raw_trim_air.npy")
#wave_grid_fine = np.load('wave_grids/PHOENIX_0.35kms_air.npy')
#wave_grid_coarse = np.load('wave_grids/PHOENIX_2kms_air.npy')
#wave_grid_kurucz_raw = np.load("wave_grids/kurucz_raw.npy")
#wave_grid_2kms_kurucz = np.load("wave_grids/kurucz_2kms_air.npy") #same wl as PHOENIX_2kms_air, but trimmed

grids = {"kurucz": {'T_points': np.arange(3500, 9751, 250),
                    'logg_points': np.arange(1.0, 5.1, 0.5), 'Z_points': ["m05", "p00", "p05"]},
         'BTSettl': {'T_points': np.arange(3000, 7001, 100), 'logg_points': np.arange(2.5, 5.6, 0.5),
                     'Z_points': ['-0.5a+0.2', '-0.0a+0.0', '+0.5a+0.0']}}


def create_wave_grid(v=1., start=3700., end=10000):
    '''Returns a grid evenly spaced in velocity'''
    size = 9000000 #this number just has to be bigger than the final array
    lam_grid = np.zeros((size,))
    i = 0
    lam_grid[i] = start
    vel = np.sqrt((C.c_kms + v) / (C.c_kms - v))
    while (lam_grid[i] < end) and (i < size - 1):
        lam_new = lam_grid[i] * vel
        i += 1
        lam_grid[i] = lam_new
    return lam_grid[np.nonzero(lam_grid)][:-1]


def create_fine_and_coarse_wave_grid():
    wave_grid_2kms_PHOENIX = create_wave_grid(2., start=3050., end=11322.2) #chosen for 3 * 2**16 = 196608
    wave_grid_fine = create_wave_grid(0.35, start=3050., end=12089.65) # chosen for 9 * 2 **17 = 1179648

    np.save('wave_grid_2kms.npy', wave_grid_2kms_PHOENIX)
    np.save('wave_grid_0.35kms.npy', wave_grid_fine)
    print(len(wave_grid_2kms_PHOENIX))
    print(len(wave_grid_fine))


def create_coarse_wave_grid_kurucz():
    start = 5050.00679905
    end = 5359.99761468
    wave_grid_2kms_kurucz = create_wave_grid(2.0, start + 1, 5333.70 + 1)
    #8192 = 2**13
    print(len(wave_grid_2kms_kurucz))
    np.save('wave_grid_2kms_kurucz.npy', wave_grid_2kms_kurucz)


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
    Calculate *n*, the refractive index of light at a given wavelength.T

    :param wl: input wavelength (in vacuum)
    :type wl: np.array

    :return: **n_air** (*np.array*) - the refractive index in air at that wavelength
    '''
    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    new_wl = wl / f
    n = wl/new_wl
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


def get_wl_kurucz():
    '''The Kurucz grid is already convolved with a FWHM=6.8km/s Gaussian. WL is log-linear spaced.'''
    sample_file = "Kurucz/t06000g45m05v000.fits"
    flux_file = fits.open(sample_file)
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

    #replace 'D' with 'E', convert to float
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

    fl = idl_float(fl_str) #convert because of "D" exponent, unreadable in Python
    fl = 10 ** (fl - 8.) #now in ergs/cm^2/s/A

    if norm:
        F_bol = trapz(fl, wl)
        fl = fl * (C.F_sun / F_bol)
        #this also means that the bolometric luminosity is always 1 L_sun

    if trunc:
        #truncate to only the wl of interest
        ind = (wl > 3000) & (wl < 13000)
        wl = wl[ind]
        fl = fl[ind]

    if air:
        wl = vacuum_to_air(wl)

    return [wl, fl]


def load_flux_full(temp, logg, Z, alpha=None, norm=False, vsini=0, grid="PHOENIX"):
    '''Load a raw PHOENIX or kurucz spectrum based upon temp, logg, and Z. Normalize to C.F_sun if desired.'''

    if grid == "PHOENIX":
        if alpha is not None:
            rname = "raw_grids/PHOENIX/Z{Z:}{alpha:}/lte{temp:0>5.0f}-{logg:.2f}{Z:}{alpha:}" \
                ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(Z=Z, temp=temp, logg=logg, alpha=alpha)
        else:
            rname = "raw_grids/PHOENIX/Z{Z:}/lte{temp:0>5.0f}-{logg:.2f}{Z:}" \
                    ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(Z=Z, temp=temp, logg=logg)
    elif grid == "kurucz":
        rname = "raw_grids/Kurucz/TRES/t{temp:0>5.0f}g{logg:.0f}{Z:}v{vsini:0>3.0f}.fits".format(temp=temp,
                                                                                       logg=10 * logg, Z=Z, vsini=vsini)
    else:
        print("No grid %s" % (grid))
        return 1

    flux_file = fits.open(rname)
    f = flux_file[0].data

    if norm:
        f *= 1e-8 #convert from erg/cm^2/s/cm to erg/cm^2/s/A
        F_bol = trapz(f, w_full)
        f = f * (C.F_sun / F_bol)
        #this also means that the bolometric luminosity is always 1 L_sun
    if grid == "kurucz":
        f *= C.c_ang / wave_grid_kurucz_raw ** 2 #Convert from f_nu to f_lambda

    flux_file.close()
    #print("Loaded " + rname)
    return f


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

def main():
    pass


if __name__ == "__main__":
    main()
