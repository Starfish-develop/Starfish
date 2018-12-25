import itertools
from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d

import Starfish.constants as C
from Starfish import config
from .utils import determine_chunk_log


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
