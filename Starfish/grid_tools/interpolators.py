import itertools
from collections import OrderedDict
import logging

import numpy as np
from scipy.interpolate import interp1d

from .utils import determine_chunk_log

BUFFER = 50
class IndexInterpolator:
    """
    Object to return fractional distance between grid points of a single grid variable.

    :param parameter_list: list of parameter values
    :type parameter_list: iterable
    """

    def __init__(self, parameter_list):
        if not isinstance(parameter_list, np.ndarray):
            parameter_list = np.array(parameter_list)
        self.npars = parameter_list.shape[-1]
        self.parameter_list = np.unique(parameter_list)
        self.index_interpolator = interp1d(self.parameter_list, np.arange(len(self.parameter_list)), kind='linear')

    def __call__(self, param):
        """
        Evaluate the interpolator at a parameter.

        :param param:
        :type param: list
        :raises ValueError: if *value* is out of bounds.

        :returns: ((low_val, high_val), (frac_low, frac_high)), the lower and higher bounding points in the grid
        and the fractional distance (0 - 1) between them and the value.
        """
        if len(param) != self.npars:
            raise ValueError('Incorrect number of parameters. Expected {} but got {}'.format(self.npars, len(param)))
        try:
            index = self.index_interpolator(param)
        except ValueError:
            raise ValueError("Requested param {} is out of bounds.".format(param))
        high = np.ceil(index).astype(int)
        low = np.floor(index).astype(int)
        frac_index = index - low
        return ((self.parameter_list[low], self.parameter_list[high]), ((1 - frac_index), frac_index))


class Interpolator:
    """
    Quickly and efficiently interpolate a synthetic spectrum for use in an MCMC
    simulation. Caches spectra for easier memory load.

    :param interface: The interface to the spectra
    :type interface: :obj:`HDF5Interface` (recommended) or :obj:`RawGridInterface`
    :param wl_range: If provided, the data wavelength range of the region you are trying to fit. Used to truncate the
        grid for speed. Default is (0, np.inf)
    :type wl_range: tuple (min, max)
    :param cache_max: maximum number of spectra to hold in cache
    :type cache_max: int
    :param cache_dump: how many spectra to purge from the cache once :attr:`cache_max` is reached
    :type cache_dump: int

    .. warning:: Interpolation causes degradation of information of the model spectra without properly forward
        propagating the errors from interpolation. We highly recommend using the :ref:`Spectral Emulator <Spectral
        Emulator>`
    """

    def __init__(self, interface, wl_range=(0, np.inf), cache_max=256, cache_dump=64):

        self.interface = interface

        self.wl = self.interface.wl
        mask = (self.wl < wl_range[-1]) & (self.wl < wl_range[1])
        self.wl = self.wl[mask]
        self.dv = self.interface.dv
        self.npars = len(interface.param_names)
        self._determine_chunk_log()

        self._setup_index_interpolators()
        self.cache = OrderedDict([])
        self.cache_max = cache_max
        self.cache_dump = cache_dump  # how many to clear once the maximum cache has been reached
        self.log = logging.getLogger(self.__class__.__name__)

    def _determine_chunk_log(self):
        """
        Determine the minimum chunk size that we can use and then
        truncate the synthetic wavelength grid and the returned spectra.

        Assumes HDF5Interface is LogLambda spaced, because otherwise you shouldn't need a grid
        with 2^n points, because you would need to interpolate in wl space after this anyway.
        """

        wl_interface = self.interface.wl  # The grid we will be truncating.
        wl_min, wl_max = np.min(self.wl), np.max(self.wl)

        # Previously this routine retuned a tuple () which was the ranges to truncate to.
        # Now this routine returns a Boolean mask,
        # so we need to go and find the first and last true values
        ind = determine_chunk_log(wl_interface, wl_min, wl_max)
        self.wl = self.wl[ind]

        # Find the index of the first and last true values
        self.interface.ind = np.argwhere(ind)[0][0], np.argwhere(ind)[-1][0] + 1

    def __call__(self, parameters):
        """
        Interpolate a spectrum

        :param parameters: stellar parameters
        :type parameters: numpy.ndarray or list

        .. note:: Automatically pops :attr:`cache_dump` items from cache if full.
        """
        if not isinstance(parameters, np.ndarray):
            parameters = np.array(parameters)

        if len(self.cache) > self.cache_max:
            [self.cache.popitem(False) for i in range(self.cache_dump)]
            self.cache_counter = 0

        return self.interpolate(parameters)

    def _setup_index_interpolators(self):
        # create an interpolator between grid points indices.
        # Given a parameter value, produce fractional index between two points
        # Store the interpolators as a list
        self.index_interpolator = IndexInterpolator(self.interface.points)

        lenF = self.interface.ind[1] - self.interface.ind[0]
        self.fluxes = np.empty((2 ** self.npars, lenF))  # 8 rows, for temp, logg, Z

    def interpolate(self, parameters):
        """
        Interpolate a spectrum without clearing cache. Recommended to use :meth:`__call__` instead to
        take advantage of caching.

        :param parameters: grid parameters
        :type parameters: numpy.ndarray or list

        :raises ValueError: if parameters are out of bounds.

        """
        if not isinstance(parameters, np.ndarray):
            parameters = np.array(parameters)

        # Previously, parameters was a dictionary of the stellar parameters.
        # Now that we have moved over to arrays, it is a numpy array.

        params, weights = self.index_interpolator(parameters)

        # Selects all the possible combinations of parameters and weights
        param_combos = list(itertools.product(*np.array(params).T))
        weight_combos = list(itertools.product(*np.array(weights).T))

        # Assemble key list necessary for indexing cache
        key_list = [self.interface.key_name.format(*param) for param in param_combos]
        weight_list = np.array([np.prod(weight) for weight in weight_combos])

        assert np.allclose(np.sum(weight_list), np.array(1.0)), "Sum of weights must equal 1, {}".format(
            np.sum(weight_list))

        # Assemble flux vector from cache, or load into cache if not there
        for i, param in enumerate(param_combos):
            key = key_list[i]
            if key not in self.cache.keys():
                # This method already allows loading only the relevant region from HDF5
                fl = self.interface.load_flux(param, header=False)
                self.cache[key] = fl

            self.fluxes[i] = self.cache[key] * weight_list[i]

        return np.sum(self.fluxes, axis=0)
