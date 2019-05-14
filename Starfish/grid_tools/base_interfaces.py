import itertools
import logging
import multiprocessing as mp
import os

import h5py
import numpy as np
from astropy.io import fits
from tqdm import tqdm

import Starfish.constants as C
from Starfish.models.transforms import instrumental_broaden, resample
from Starfish.utils import calculate_dv, calculate_dv_dict, create_log_lam_grid
from .utils import chunk_list

log = logging.getLogger(__name__)


class GridInterface:
    """
    A base class to handle interfacing with synthetic spectral libraries.

    Parameters
    ----------
    base : str or path-like
        path to the root of the files on disk.
    param_names : list of str
        The names of the parameters (dimensions) of the grid
    points :  array_like
        the grid points at which
        spectra exist (assumes grid is square, not ragged, meaning that every combination
        of parameters specified exists in the grid).
    wave_units : str
        The units of the wavelengths. Preferably equivalent to an astropy unit string.
    flux_units : str
        The units of the model fluxes. Preferable equivalent to an astropy unit string.
    wl_range :  list [min, max], optional
        the starting and ending wavelength ranges of the grid to
        truncate to. If None, will use whole available grid. Default is None.
    air :  bool, optional
        Are the wavelengths measured in air? Default is True
    name : str, optional
        name of the spectral library, Default is None
    """

    def __init__(self, path, param_names, points, wave_units, flux_units, wl_range=None, air=True, name=None):
        self.path = path
        self.param_names = param_names
        self.points = points
        self.air = air
        self.wl_range = wl_range
        self.wave_units = wave_units
        self.flux_units = flux_units
        if name is None:
            name = 'Grid Interface'
        self.name = name

    def check_params(self, parameters):
        """
        Determine if the specified parameters are allowed in the grid.

        Parameters
        ----------
        parameters : array_like
            parameter set to check

        Raises
        -------
        ValueError
            if the parameter values are outside of the grid bounds

        Returns
        -------
        bool
            True if found in grid
        """
        if not isinstance(parameters, np.ndarray):
            parameters = np.array(parameters)

        if len(parameters) != len(self.param_names):
            raise ValueError('Length of given parameters ({}) does not match length of grid parameters ({})'.format(
                len(parameters), len(self.param_names)))

        for param, params in zip(parameters, self.points):
            if param not in params:
                raise ValueError(
                    '{} not in the grid points {}'.format(param, params))
        return True

    def load_flux(self, parameters, header=False, norm=True):
        """
        Load the flux and header information.

        Parameters
        ----------
        parameters : array_like
            stellar parameters
        header : bool, optional
            If True, will return the header alongside the flux. Default is False.
        norm : bool, optional
            If True, will normalize the flux to solar luminosity. Default is True.

        Raises
        ------
        ValueError
            if the file cannot be found on disk.

        Returns
        -------
        numpy.ndarray if header is False, tuple of (numpy.ndarray, dict) if header is True
        """
        raise NotImplementedError(
            '`load_flux` is abstract and must be implemented by subclasses')

    def __repr__(self):
        output = '{}\n'.format(self.name)
        output += '-' * len(self.name) + '\n'
        output += 'Base: {}\n'.format(self.path)
        for par, point in zip(self.param_names, self.points):
            output += '{}: {}\n'.format(par, point)
        return output


class HDF5Interface:
    """
    Connect to an HDF5 file that stores spectra.

    Parameters
    ----------
    filename : str or path-like
        The path of the saved HDF5 file
    """

    def __init__(self, filename):
        self.filename = os.path.expandvars(filename)

        # In order to properly interface with the HDF5 file, we need to learn
        # a few things about it

        # 1.) Which parameter combinations exist in the file (self.grid_points)
        # 2.) What are the minimum and maximum values for each parameter (self.bounds)
        # 3.) Which values exist for each parameter (self.points)

        with h5py.File(self.filename, 'r') as base:
            self.wl = base['wl'][:]
            self.key_name = base['flux'].attrs['key_name']
            self.wl_header = dict(base['wl'].attrs.items())
            self.dv = self.wl_header['dv']
            self.grid_points = base['grid_points'][:]
            self.param_names = base['grid_points'].attrs['names']
            self.wave_units = base['wl'].attrs['units']
            self.flux_units = base['flux'].attrs['units']

        # determine the bounding regions of the grid by sorting the grid_points
        low = np.min(self.grid_points, axis=0)
        high = np.max(self.grid_points, axis=0)
        self.bounds = np.vstack((low, high)).T
        self.points = [np.unique(self.grid_points[:, i])
                       for i in range(self.grid_points.shape[1])]

        self.ind = None  # Overwritten by other methods using this as part of a ModelInterpolator

        # Test if key-name is specified correctly
        try:
            self.load_flux(self.grid_points[0])
        except (IndexError, KeyError):
            raise ValueError('key_name is ill-specified.')

    def load_flux(self, parameters, header=False):
        """
        Load just the flux from the grid, with possibly an index truncation.

        parameters : array_like
            the stellar parameters
        header : bool, optional
            If True, will return the header as well as the flux. Default is False


        Returns
        -------
        numpy.ndarray if header is False, otherwise (numpy.ndarray, dict)
        """
        if not isinstance(parameters, np.ndarray):
            parameters = np.array(parameters)

        key = self.key_name.format(*parameters)
        with h5py.File(self.filename, 'r') as hdf5:
            hdr = dict(hdf5['flux'][key].attrs)
            if self.ind is not None:
                fl = hdf5['flux'][key][self.ind[0]:self.ind[1]]
            else:
                fl = hdf5['flux'][key][:]

        # Note: will raise a KeyError if the file is not found.
        if header:
            return fl, hdr
        else:
            return fl

    @property
    def fluxes(self):
        """
        Iterator to loop over all of the spectra stored in the grid, for PCA.
        Loops over parameters in the order specified by grid_points.

        Returns
        -------
        Generator of numpy.ndarrays
        """

        for grid_point in self.grid_points:
            yield self.load_flux(grid_point, header=False)


class HDF5Creator:
    """
    Create a HDF5 grid to store all of the spectra from a RawGridInterface,
    along with metadata.

    Parameters
    ----------
    GridInterface : :class:`GridInterface`
        The raw grid interface to process while creating the HDF5 file
    filename : str or path-like
        Where to save the HDF5 file
    instrument : :class:`Instrument`, optional
        If provided, the instrument to convolve/truncate the grid. If None, will
        maintain the grid's original wavelengths and resolution. Default is None
    wl_range : list [min, max], optional
        The wavelength range to truncate the grid to. Will be truncated to match grid wavelengths and instrument wavelengths if over or under specified. If set to None, will not truncate grid. Default is NOne
    ranges : array_like, optional
        lower and upper limits for each stellar parameter,
        in order to truncate the number of spectra in the grid. If None, will not restrict the range of the parameters. Default is None.
    key_name : format str
        formatting string that has keys for each of the parameter names to translate into a hash-able string. If set to None, will create a name by taking each parameter name followed by value with underscores delimiting parameters. Default is None.

    Raises
    ------ 
    ValueError 
        if the wl_range is ill-specified or if the parameter range are completely disjoint from the grid points.
    """

    def __init__(self, GridInterface, filename, instrument=None, wl_range=None, ranges=None, key_name=None):

        self.log = logging.getLogger(self.__class__.__name__)

        self.GridInterface = GridInterface
        self.filename = os.path.expandvars(filename)
        self.instrument = instrument

        # The flux formatting key will always have alpha in the name, regardless
        # of whether or not the library uses it as a parameter.
        if key_name is None:
            key_name = self.GridInterface.rname.replace('/', '__').replace('.fits', '').replace('.FITS', '')

        self.key_name = key_name

        if ranges is None:
            self.points = self.GridInterface.points
        else:
            # Take only those points of the GridInterface that fall within the ranges specified
            self.points = []
            # We know which subset we want, so use these.
            for i, (low, high) in enumerate(ranges):
                valid_points = self.GridInterface.points[i]
                ind = (valid_points >= low) & (valid_points <= high)
                self.points.append(valid_points[ind])
                # Note that at this point, this is just the grid points that fall within the rectangular
                # bounds set by ranges. If the raw library is actually irregular (e.g. CIFIST),
                # then self.points will contain points that don't actually exist in the raw library.

        # the raw wl from the spectral library
        self.wl_native = self.GridInterface.wl  # raw grid
        self.dv_native = calculate_dv(self.wl_native)

        self.hdf5 = h5py.File(self.filename, 'w')
        self.hdf5.attrs['grid_name'] = GridInterface.name
        self.flux_group = self.hdf5.create_group('flux')
        self.flux_group.attrs['units'] = GridInterface.flux_units
        self.flux_group.attrs['key_name'] = self.key_name

        # We'll need a few wavelength grids
        # 1. The original synthetic grid: ``self.wl_native``
        # 2. A finely spaced log-lambda grid respecting the ``dv`` of
        #   ``self.wl_native``, onto which we can interpolate the flux values
        #   in preperation of the FFT: ``self.wl_loglam``
        # [ DO FFT ]
        # 3. A log-lambda spaced grid onto which we can downsample the result
        #   of the FFT, spaced with a ``dv`` such that we respect the remaining
        #   Fourier modes: ``self.wl_final``

        # There are three ranges to consider when wanting to make a grid:
        # 1. The full range of the synthetic library
        # 2. The full range of the instrument/dataset
        # 3. The range specified by the user in config.yaml
        # For speed reasons, we will always truncate to to wl_range. If either
        # the synthetic library or the instrument library is smaller than this range,
        # raise an error.
        if wl_range is None:
            wl_min, wl_max = 0, np.inf
        else:
            wl_min, wl_max = wl_range
        buffer = 50  # [AA]
        wl_min -= buffer
        wl_max += buffer

        # If the raw synthetic grid doesn't span the full range of the user
        # specified grid, raise an error.
        # Instead, let's choose the maximum limit of the synthetic grid?
        if self.instrument is not None:
            inst_min, inst_max = self.instrument.wl_range
        else:
            inst_min, inst_max = 0, np.inf
        imposed_min = np.max([self.wl_native[0], inst_min])
        imposed_max = np.min([self.wl_native[-1], inst_max])
        if wl_min < imposed_min:
            self.log.info('Given minimum wavelength ({}) is less than instrument or grid minimum. Truncating to {}'
                          .format(wl_min, imposed_min))
            wl_min = imposed_min
        if wl_max > imposed_max:
            self.log.info('Given maximum wavelength ({}) is greater than instrument or grid maximum. Truncating to {}'
                          .format(wl_max, imposed_max))
            wl_max = imposed_max

        if wl_max < wl_min:
            raise ValueError(
                'Minimum wavelength must be less than maximum wavelength')

        # Calculate wl_loglam
        # use the dv that preserves the native quality of the raw PHOENIX grid
        wl_dict = create_log_lam_grid(self.dv_native, wl_min, wl_max)
        wl_loglam = wl_dict['wl']
        dv_loglam = calculate_dv_dict(wl_dict)

        self.log.info('FFT grid stretches from {} to {}'.format(
            wl_loglam[0], wl_loglam[-1]))
        self.log.info('wl_loglam dv is {} km/s'.format(dv_loglam))

        if self.instrument is None:
            mask = (self.wl_native > wl_min) & (self.wl_native < wl_max)
            self.wl_final = self.wl_native[mask]
            self.dv_final = self.dv_native
            def inst_broaden(w, f): 
                return (w, f)
        else:
            def inst_broaden(w, f): 
                return (
                w, instrumental_broaden(w, f, self.instrument.FWHM))
            # The final wavelength grid, onto which we will interpolate the
            # Fourier filtered wavelengths, is part of the instrument object
            dv_temp = self.instrument.FWHM / self.instrument.oversampling
            wl_dict = create_log_lam_grid(dv_temp, wl_min, wl_max)
            self.wl_final = wl_dict['wl']
            self.dv_final = calculate_dv_dict(wl_dict)

        def resample_loglam(w, f): return (
            wl_loglam, resample(w, f, wl_loglam))
        def resample_final(w, f): return (
            self.wl_final, resample(w, f, self.wl_final))
        self.transform = lambda flux: resample_final(
            *inst_broaden(*resample_loglam(self.wl_native, flux)))

        # Create the wl dataset separately using float64 due to rounding errors w/ interpolation.
        wl_dset = self.hdf5.create_dataset(
            'wl', data=self.wl_final, compression=9)
        wl_dset.attrs['air'] = self.GridInterface.air
        wl_dset.attrs['dv'] = self.dv_final
        wl_dset.attrs['units'] = self.GridInterface.wave_units

    def process_grid(self):
        """
        Run :meth:`process_flux` for all of the spectra within the `ranges`
        and store the processed spectra in the HDF5 file.
        """

        # points is now a list of numpy arrays of the values in the grid
        # Take all parameter permutations in self.points and create a list
        # param_list will be a list of numpy arrays, specifying the parameters
        param_list = []

        # use itertools.product to create permutations of all possible values
        for i in itertools.product(*self.points):
            param_list.append(np.array(i))

        all_params = np.array(param_list)

        invalid_params = []

        self.log.debug('Total of {} files to process.'.format(len(param_list)))

        pbar = tqdm(all_params)
        for i, param in enumerate(pbar):
            pbar.set_description('Processing {}'.format(param))
            # Load and process the flux
            try:
                flux, header = self.GridInterface.load_flux(param, header=True)
            except ValueError:
                self.log.warning(
                    'Deleting {} from all params, does not exist.'.format(param))
                invalid_params.append(i)
                continue

            _, fl_final = self.transform(flux)

            flux = self.flux_group.create_dataset(self.key_name.format(*param),
                                                    data=fl_final, compression=9)
            # Store header keywords as attributes in HDF5 file
            for key, value in header.items():
                if key != '' and value != '':  # check for empty FITS kws
                    flux.attrs[key] = value

        # Remove parameters that do no exist
        all_params = np.delete(all_params, invalid_params, axis=0)

        gp = self.hdf5.create_dataset(
            'grid_points', data=all_params, compression=9)
        gp.attrs['names'] = self.GridInterface.param_names

        self.hdf5.close()
