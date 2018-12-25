import os
import gc
import itertools
import bz2

import numpy as np
from numpy.fft import fft, ifft, fftfreq, rfftfreq
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import j1
import h5py
import tqdm

from Starfish import config
import Starfish.constants as C


class RawGridInterface:
    '''
    A base class to handle interfacing with synthetic spectral libraries.

    :param name: name of the spectral library
    :type name: string
    :param param_names: the names of the parameters (dimensions) of the grid
    :type param_names: list
    :param points: the grid points at which
        spectra exist (assumes grid is square, not ragged, meaning that every combination
        of parameters specified exists in the grid).
    :type points: list of numpy arrays
    :param air: Are the wavelengths measured in air?
    :type air: bool
    :param wl_range: the starting and ending wavelength ranges of the grid to
        truncate to.
    :type wl_range: list of len 2 [min, max]
    :param base: path to the root of the files on disk.
    :type base: string

    '''

    def __init__(self, name, param_names, points, air=True, wl_range=[3000, 13000], base=None):
        self.name = name

        self.param_names = param_names
        self.points = points

        self.air = air
        self.wl_range = wl_range
        self.base = base

    def check_params(self, parameters):
        '''
        Determine if the specified parameters are allowed in the grid.

        :param parameters: parameter set to check
        :type parameters: np.array

        :raises C.GridError: if the parameter values are outside of the grid bounds

        '''
        if len(parameters) != len(self.param_names):
            raise ValueError("Length of given parameters ({}) does not match length of grid parameters ({})".format(
                len(parameters), len(self.param_names)))

        for param, ppoints in zip(parameters, self.points):
            if param not in ppoints:
                raise ValueError("{} not in the grid points {}".format(param, ppoints))

    def load_flux(self, parameters, norm=True):
        '''
        Load the synthetic flux from the disk and  :meth:`check_params`

        :param parameters: stellar parameters describing a spectrum
        :type parameters: np.array
        :param norm: normalize the spectrum to solar luminosity?
        :type norm: bool

         .. note::

            This method is designed to be extended by the inheriting class
        '''
        raise NotImplementedError("`load_flux` is abstract and must be implemented by subclasses")




class HDF5Creator:
    '''
    Create a HDF5 grid to store all of the spectra from a RawGridInterface,
    along with metadata.

    '''

    def __init__(self, GridInterface, filename, Instrument, ranges=None,
                 key_name=config.grid["key_name"], vsinis=None):
        '''
        :param GridInterface: :obj:`RawGridInterface` object or subclass thereof
            to access raw spectra on disk.
        :param filename: where to create the HDF5 file. Suffix ``*.hdf5`` recommended.
        :param Instrument: the instrument to convolve/truncate the grid. If you
            want a high-res grid, use the NullInstrument.
        :param ranges: lower and upper limits for each stellar parameter,
            in order to truncate the number of spectra in the grid.
        :type ranges: dict of keywords mapped to 2-tuples
        :param key_name: formatting string that has keys for each of the parameter
            names to translate into a hash-able string.
        :type key_name: string

        This object is designed to be run in serial.
        '''

        if ranges is None:
            # Programatically define each range to be (-np.inf, np.inf)
            ranges = []
            for par in config.grid["parname"]:
                ranges.append([-np.inf, np.inf])

        self.GridInterface = GridInterface
        self.filename = os.path.expandvars(filename)  # only store the name to the HDF5 file, because
        # otherwise the object cannot be parallelized
        self.Instrument = Instrument

        # The flux formatting key will always have alpha in the name, regardless
        # of whether or not the library uses it as a parameter.
        self.key_name = key_name

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

        self.hdf5 = h5py.File(self.filename, "w")
        self.hdf5.attrs["grid_name"] = GridInterface.name
        self.hdf5.flux_group = self.hdf5.create_group("flux")
        self.hdf5.flux_group.attrs["unit"] = "erg/cm^2/s/A"

        # We'll need a few wavelength grids
        # 1. The original synthetic grid: ``self.wl_native``
        # 2. A finely spaced log-lambda grid respecting the ``dv`` of
        #   ``self.wl_native``, onto which we can interpolate the flux values
        #   in preperation of the FFT: ``self.wl_FFT``
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

        # inst_min, inst_max = self.Instrument.wl_range
        wl_min, wl_max = config.grid["wl_range"]
        buffer = config.grid["buffer"]  # [AA]
        wl_min -= buffer
        wl_max += buffer

        # If the raw synthetic grid doesn't span the full range of the user
        # specified grid, raise an error.
        # Instead, let's choose the maximum limit of the synthetic grid?
        if (self.wl_native[0] > wl_min) or (self.wl_native[-1] < wl_max):
            print(
                "Synthetic grid does not encapsulate chosen wl_range in config.yaml, truncating new grid to extent of synthetic grid, {}, {}".format(
                    self.wl_native[0], self.wl_native[-1]))
            wl_min, wl_max = self.wl_native[0], self.wl_native[-1]

        # Calculate wl_FFT
        # use the dv that preserves the native quality of the raw PHOENIX grid
        wl_dict = create_log_lam_grid(self.dv_native, wl_min, wl_max)
        self.wl_FFT = wl_dict["wl"]
        self.dv_FFT = calculate_dv_dict(wl_dict)

        print("FFT grid stretches from {} to {}".format(self.wl_FFT[0], self.wl_FFT[-1]))
        print("wl_FFT dv is {} km/s".format(self.dv_FFT))

        # The Fourier coordinate
        self.ss = rfftfreq(len(self.wl_FFT), d=self.dv_FFT)

        # The instrumental taper
        sigma = self.Instrument.FWHM / 2.35  # in km/s
        # Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space
        self.taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (self.ss ** 2))

        self.ss[0] = 0.01  # junk so we don't get a divide by zero error

        # The final wavelength grid, onto which we will interpolate the
        # Fourier filtered wavelengths, is part of the Instrument object
        dv_temp = self.Instrument.FWHM / self.Instrument.oversampling
        wl_dict = create_log_lam_grid(dv_temp, wl_min, wl_max)
        self.wl_final = wl_dict["wl"]
        self.dv_final = calculate_dv_dict(wl_dict)

        # Create the wl dataset separately using float64 due to rounding errors w/ interpolation.
        wl_dset = self.hdf5.create_dataset("wl", (len(self.wl_final),), dtype="f8", compression='gzip',
                                           compression_opts=9)
        wl_dset[:] = self.wl_final
        wl_dset.attrs["air"] = self.GridInterface.air
        wl_dset.attrs["dv"] = self.dv_final

    def process_flux(self, parameters):
        '''
        Take a flux file from the raw grid, process it according to the
        instrument, and insert it into the HDF5 file.

        :param parameters: the model parameters.
        :type parameters: 1D np.array

        :raises AssertionError: if the `parameters` vector is not
            the same length as that of the raw grid.

        :returns: a tuple of (parameters, flux, header). If the flux could
            not be loaded, returns (None, None, None).

        '''
        # assert len(parameters) == len(config.grid["parname"]), "Must pass numpy array {}".format(config.grid["parname"])

        # If the parameter length is one more than the grid pars,
        # assume this is for vsini convolution
        if len(parameters) == (len(config.grid["parname"]) + 1):
            vsini = parameters[-1]
            parameters = parameters[:-1]
        else:
            vsini = 0.0

        try:
            flux, header = self.GridInterface.load_flux(parameters)

            # Interpolate the native spectrum to a log-lam FFT grid
            interp = InterpolatedUnivariateSpline(self.wl_native, flux, k=5)
            fl = interp(self.wl_FFT)
            del interp
            gc.collect()

            # Do the FFT
            FF = np.fft.rfft(fl)

            if vsini > 0.0:
                # Calculate the stellar broadening kernel
                ub = 2. * np.pi * vsini * self.ss
                sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
                # set zeroth frequency to 1 separately (DC term)
                sb[0] = 1.

                # institute vsini and instrumental taper
                FF_tap = FF * sb * self.taper

            else:
                # apply just instrumental taper
                FF_tap = FF * self.taper

            # do IFFT
            fl_tapered = np.fft.irfft(FF_tap)

            # downsample to the final grid
            interp = InterpolatedUnivariateSpline(self.wl_FFT, fl_tapered, k=5)
            fl_final = interp(self.wl_final)
            del interp
            gc.collect()

            return (fl_final, header)

        except C.GridError as e:
            print("No file with parameters {}. C.GridError: {}".format(parameters, e))
            return (None, None)

    def process_grid(self):
        '''
        Run :meth:`process_flux` for all of the spectra within the `ranges`
        and store the processed spectra in the HDF5 file.

        Only executed in serial for now.
        '''

        # points is now a list of numpy arrays of the values in the grid
        # Take all parameter permutations in self.points and create a list
        # param_list will be a list of numpy arrays, specifying the parameters
        param_list = []

        # use itertools.product to create permutations of all possible values
        for i in itertools.product(*self.points):
            param_list.append(np.array(i))

        all_params = np.array(param_list)

        invalid_params = []

        print("Total of {} files to process.".format(len(param_list)))

        pbar = tqdm(all_params)
        for i, param in enumerate(pbar):
            pbar.set_description("Processing {}".format(param))
            fl, header = self.process_flux(param)
            if fl is None:
                print("Deleting {} from all params, does not exist.".format(param))
                invalid_params.append(i)
                continue

            # The PHOENIX spectra are stored as float32, and so we do the same here.
            flux = self.hdf5["flux"].create_dataset(self.key_name.format(*param),
                                                    shape=(len(fl),), dtype="f", compression='gzip',
                                                    compression_opts=9)
            flux[:] = fl

            # Store header keywords as attributes in HDF5 file
            for key, value in header.items():
                if key != "" and value != "":  # check for empty FITS kws
                    flux.attrs[key] = value

        # Remove parameters that do no exist
        all_params = np.delete(all_params, invalid_params, axis=0)

        par_dset = self.hdf5.create_dataset("pars", all_params.shape, dtype="f8", compression='gzip',
                                            compression_opts=9)
        par_dset[:] = all_params

        self.hdf5.close()


class HDF5Interface:
    '''
    Connect to an HDF5 file that stores spectra.
    '''

    def __init__(self, filename=config.grid["hdf5_path"], key_name=config.grid["key_name"]):
        '''
        :param filename: the name of the HDF5 file
        :type param: string
        :param ranges: optionally select a smaller part of the grid to use.
        :type ranges: dict
        '''
        self.filename = os.path.expandvars(filename)
        self.key_name = key_name

        # In order to properly interface with the HDF5 file, we need to learn
        # a few things about it

        # 1.) Which parameter combinations exist in the file (self.grid_points)
        # 2.) What are the minimum and maximum values for each parameter (self.bounds)
        # 3.) Which values exist for each parameter (self.points)

        with h5py.File(self.filename, "r") as hdf5:
            self.wl = hdf5["wl"][:]
            self.wl_header = dict(hdf5["wl"].attrs.items())
            self.dv = self.wl_header["dv"]
            self.grid_points = hdf5["pars"][:]

        # determine the bounding regions of the grid by sorting the grid_points
        low = np.min(self.grid_points, axis=0)
        high = np.max(self.grid_points, axis=0)
        self.bounds = np.vstack((low, high)).T
        self.points = [np.unique(self.grid_points[:, i]) for i in range(self.grid_points.shape[1])]

        self.ind = None  # Overwritten by other methods using this as part of a ModelInterpolator

    def load_flux(self, parameters):
        '''
        Load just the flux from the grid, with possibly an index truncation.

        :param parameters: the stellar parameters
        :type parameters: np.array

        :raises KeyError: if spectrum is not found in the HDF5 file.

        :returns: flux array
        '''

        key = self.key_name.format(*parameters)
        with h5py.File(self.filename, "r") as hdf5:
            try:
                if self.ind is not None:
                    fl = hdf5['flux'][key][self.ind[0]:self.ind[1]]
                else:
                    fl = hdf5['flux'][key][:]
            except KeyError as e:
                raise C.GridError(e)

        # Note: will raise a KeyError if the file is not found.

        return fl

    @property
    def fluxes(self):
        '''
        Iterator to loop over all of the spectra stored in the grid, for PCA.

        Loops over parameters in the order specified by grid_points.
        '''

        for grid_point in self.grid_points:
            yield self.load_flux(grid_point)

    def load_flux_hdr(self, parameters):
        '''
        Just like load_flux, but also returns the header
        '''
        key = self.key_name.format(*parameters)
        with h5py.File(self.filename, "r") as hdf5:
            try:
                hdr = dict(hdf5['flux'][key].attrs)
                if self.ind is not None:
                    fl = hdf5['flux'][key][self.ind[0]:self.ind[1]]
                else:
                    fl = hdf5['flux'][key][:]
            except KeyError as e:
                raise C.GridError(e)

        # Note: will raise a KeyError if the file is not found.

        return (fl, hdr)
