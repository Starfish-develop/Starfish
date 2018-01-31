import numpy as np
from numpy.fft import fft, ifft, fftfreq, rfftfreq
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

import Starfish
from .spectrum import create_log_lam_grid, calculate_dv, calculate_dv_dict
from . import constants as C

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
    chunks = [mylist[0+size*i : size*(i+1)] for i in range(n)] #fill with evenly divisible
    leftover = length - size*n
    edge = size*n
    for i in range(leftover): #backfill each with the last item
        chunks[i%n].append(mylist[edge+i])
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
    assert wl_min >= np.min(wl) and wl_max <= np.max(wl), "determine_chunk_log: wl_min {:.2f} and wl_max {:.2f} are not within the bounds of the grid {:.2f} to {:.2f}.".format(wl_min, wl_max, np.min(wl), np.max(wl))

    # Find the smallest length synthetic spectrum that is a power of 2 in length
    # and longer than the number of points contained between wl_min and wl_max
    len_wl = len(wl)
    npoints = np.sum((wl >= wl_min) & (wl <= wl_max))
    chunk = len_wl
    inds = (0, chunk)

    # This loop will exit with chunk being the smallest power of 2 that is
    # larger than npoints
    while chunk > npoints:
        if chunk/2 > npoints:
            chunk = chunk//2
        else:
            break


    assert type(chunk) == np.int, "Chunk is not an integer!. Chunk is {}".format(chunk)

    if chunk < len_wl:
        # Now that we have determined the length of the chunk of the synthetic
        # spectrum, determine indices that straddle the data spectrum.

        # Find the index that corresponds to the wl at the center of the data spectrum
        center_wl = (wl_min + wl_max)/2.
        center_ind = (np.abs(wl - center_wl)).argmin()

        #Take a chunk that straddles either side.
        inds = (center_ind - chunk//2, center_ind + chunk//2)

        ind = (np.arange(len_wl) >= inds[0]) & (np.arange(len_wl) < inds[1])
    else:
        print("keeping grid as is")
        ind = np.ones_like(wl, dtype='bool')

    assert (min(wl[ind]) <= wl_min) and (max(wl[ind]) >= wl_max), "Model"\
        "Interpolator chunking ({:.2f}, {:.2f}) didn't encapsulate full"\
        " wl range ({:.2f}, {:.2f}).".format(min(wl[ind]), max(wl[ind]), wl_min, wl_max)

    return ind


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
    def __init__(self, name, param_names, points, air=True, wl_range=[3000,13000], base=None):
        self.name = name

        self.param_names = param_names
        self.points = points

        self.air = air
        self.wl_range = wl_range
        self.base = os.path.expandvars(base)

    def check_params(self, parameters):
        '''
        Determine if the specified parameters are allowed in the grid.

        :param parameters: parameter set to check
        :type parameters: np.array

        :raises C.GridError: if the parameter values are outside of the grid bounds

        '''
        assert len(parameters) == len(self.param_names)

        for param, ppoints in zip(parameters, self.points):
            if param not in ppoints:
                raise C.GridError("{} not in the grid points {}".format(param, ppoints))

    def load_flux(self, parameters, norm=True):
        '''
        Load the synthetic flux from the disk and  :meth:`check_params`

        :param parameters: stellar parameters describing a spectrum
        :type parameters: np.array

         .. note::

            This method is designed to be extended by the inheriting class
        '''
        pass

class PHOENIXGridInterface(RawGridInterface):
    '''
    An Interface to the PHOENIX/Husser synthetic library.

    :param norm: normalize the spectrum to solar luminosity?
    :type norm: bool

    '''
    def __init__(self, air=True, norm=True, wl_range=[3000, 54000],
        base=Starfish.grid["raw_path"]):

        super().__init__(name="PHOENIX",
            param_names = ["temp", "logg", "Z", "alpha"],
            points=[
          np.array([2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200,
          3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400,
          4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600,
          5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800,
          6900, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000,
          9200, 9400, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 11200,
          11400, 11600, 11800, 12000]),
            np.arange(0.0, 6.1, 0.5),
            np.arange(-2., 1.1, 0.5),
            np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])],
            air=air, wl_range=wl_range, base=base) #wl_range used to be [2999, 13001]

        self.norm = norm #Normalize to 1 solar luminosity?
        self.par_dicts = [None,
                        None,
                        {-2:"-2.0", -1.5:"-1.5", -1:'-1.0', -0.5:'-0.5',
                            0.0: '-0.0', 0.5: '+0.5', 1: '+1.0'},
                        {-0.4:".Alpha=-0.40", -0.2:".Alpha=-0.20",
                            0.0: "", 0.2:".Alpha=+0.20", 0.4:".Alpha=+0.40",
                            0.6:".Alpha=+0.60", 0.8:".Alpha=+0.80"}]

        # if air is true, convert the normally vacuum file to air wls.
        try:
            base = os.path.expandvars(self.base)
            wl_file = fits.open(base + "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
        except OSError:
            raise C.GridError("Wavelength file improperly specified.")

        w_full = wl_file[0].data
        wl_file.close()
        if self.air:
            self.wl_full = vacuum_to_air(w_full)
        else:
            self.wl_full = w_full

        self.ind = (self.wl_full >= self.wl_range[0]) & (self.wl_full <= self.wl_range[1])
        self.wl = self.wl_full[self.ind]
        self.rname = self.base + "Z{2:}{3:}/lte{0:0>5.0f}-{1:.2f}{2:}{3:}" \
                     ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    def load_flux(self, parameters, norm=True):
        '''
       Load just the flux and header information.

       :param parameters: stellar parameters
       :type parameters: np.array

       :raises C.GridError: if the file cannot be found on disk.

       :returns: tuple (flux_array, header_dict)

       '''
        self.check_params(parameters) # Check to make sure that the keys are
        # allowed and that the values are in the grid

        # Create a list of the parameters to be fed to the format string
        # optionally replacing arguments using the dictionaries, if the formatting
        # of a certain parameter is tricky
        str_parameters = []
        for param, par_dict in zip(parameters, self.par_dicts):
            if par_dict is None:
                str_parameters.append(param)
            else:
                str_parameters.append(par_dict[param])

        fname = self.rname.format(*str_parameters)

        #Still need to check that file is in the grid, otherwise raise a C.GridError
        #Read all metadata in from the FITS header, and append to spectrum
        try:
            flux_file = fits.open(fname)
            f = flux_file[0].data
            hdr = flux_file[0].header
            flux_file.close()
        except OSError:
            raise C.GridError("{} is not on disk.".format(fname))

        #If we want to normalize the spectra, we must do it now since later we won't have the full EM range
        if self.norm:
            f *= 1e-8 #convert from erg/cm^2/s/cm to erg/cm^2/s/A
            F_bol = trapz(f, self.wl_full)
            f = f * (C.F_sun / F_bol) #bolometric luminosity is always 1 L_sun

        #Add temp, logg, Z, alpha, norm to the metadata
        header = {}
        header["norm"] = self.norm
        header["air"] = self.air
        #Keep only the relevant PHOENIX keywords, which start with PHX
        for key, value in hdr.items():
            if key[:3] == "PHX":
                header[key] = value

        return (f[self.ind], header)

class PHOENIXGridInterfaceNoAlpha(PHOENIXGridInterface):
        '''
        An Interface to the PHOENIX/Husser synthetic library.

        :param norm: normalize the spectrum to solar luminosity?
        :type norm: bool

        '''
        def __init__(self, air=True, norm=True, wl_range=[3000, 54000],
            base=Starfish.grid["raw_path"]):

            # Initialize according to the regular PHOENIX values
            super().__init__(air=air, norm=norm, wl_range=wl_range, base=base)

            # Now override parameters to exclude alpha
            self.param_names = ["temp", "logg", "Z"]
            self.points=[
          np.array([2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200,
          3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400,
          4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600,
          5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800,
          6900, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000,
          9200, 9400, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 11200,
          11400, 11600, 11800, 12000]),
            np.arange(0.0, 6.1, 0.5),
            np.arange(-2., 1.1, 0.5)]

            self.par_dicts = [None,
                            None,
                            {-2:"-2.0", -1.5:"-1.5", -1:'-1.0', -0.5:'-0.5',
                                0.0: '-0.0', 0.5: '+0.5', 1: '+1.0'}]

            base = os.path.expandvars(self.base)
            self.rname = base + "Z{2:}/lte{0:0>5.0f}-{1:.2f}{2:}" \
                         ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"


class KuruczGridInterface(RawGridInterface):
    '''Kurucz grid interface.

    Spectra are stored in ``f_nu`` in a filename like
    ``t03500g00m25ap00k2v070z1i00.fits``, ``ap00`` means zero alpha enhancement,
    and ``k2`` is the microturbulence, while ``z1`` is the macroturbulence.
    These particular values are roughly the ones appropriate for the Sun.
    '''
    def __init__(self, air=True, norm=True, wl_range=[5000, 5400], base=Starfish.grid["raw_path"]):
        super().__init__(name="Kurucz",
            param_names = ["temp", "logg", "Z"],
            points=[np.arange(3500, 9751, 250),
                    np.arange(0.0, 5.1, 0.5),
                    np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5])],
                     air=air, wl_range=wl_range, base=base)

        self.par_dicts = [None, None, {-2.5:"m25", -2.0:"m20", -1.5:"m15", -1.0:"m10", -0.5:"m05", 0.0:"p00", 0.5:"p05"}]

        self.norm = norm #Convert to f_lam and average to 1, or leave in f_nu?
        self.rname = base + "t{0:0>5.0f}/g{1:0>2.0f}/t{0:0>5.0f}g{1:0>2.0f}{2}ap00k2v000z1i00.fits"
        self.wl_full = np.load(base + "kurucz_raw_wl.npy")
        self.ind = (self.wl_full >= self.wl_range[0]) & (self.wl_full <= self.wl_range[1])
        self.wl = self.wl_full[self.ind]

    def load_flux(self, parameters, norm=True):
        '''
        Load a the flux and header information.

        :param parameters: stellar parameters
        :type parameters: dict

        :raises C.GridError: if the file cannot be found on disk.

        :returns: tuple (flux_array, header_dict)

        '''
        self.check_params(parameters)

        str_parameters = []
        for param, par_dict in zip(parameters, self.par_dicts):
            if par_dict is None:
                str_parameters.append(param)
            else:
                str_parameters.append(par_dict[param])

        #Multiply logg by 10
        str_parameters[1] *= 10

        fname = self.rname.format(*str_parameters)

        #Still need to check that file is in the grid, otherwise raise a C.GridError
        #Read all metadata in from the FITS header, and append to spectrum
        try:
            flux_file = fits.open(fname)
            f = flux_file[0].data
            hdr = flux_file[0].header
            flux_file.close()
        except OSError:
            raise C.GridError("{} is not on disk.".format(fname))

        #We cannot normalize the spectra, since we don't have a full wl range, so instead we set the average
        #flux to be 1

        #Also, we should convert from f_nu to f_lam
        if self.norm:
            f *= C.c_ang / self.wl**2 #Convert from f_nu to f_lambda
            f /= np.average(f) #divide by the mean flux, so avg(f) = 1

        #Add temp, logg, Z, norm to the metadata
        header = {}
        header["norm"] = self.norm
        header["air"] = self.air
        #Keep the relevant keywords
        for key, value in hdr.items():
            header[key] = value

        return (f[self.ind], header)

class BTSettlGridInterface(RawGridInterface):
    '''BTSettl grid interface. Unlike the PHOENIX and Kurucz grids, the
    individual files of the BTSettl grid do not always have the same wavelength
    sampling. Therefore, each call of :meth:`load_flux` will interpolate the
    flux onto a LogLambda spaced grid that ranges between `wl_range` and has a
    velocity spacing of 0.08 km/s or better.

    If you have a choice, it's probably easier to use the Husser PHOENIX grid.
    '''
    def __init__(self, air=True, norm=True, wl_range=[2999, 13000], base="libraries/raw/BTSettl/"):
        super().__init__(name="BTSettl",
        points={"temp":np.arange(3000, 7001, 100),
                "logg":np.arange(2.5, 5.6, 0.5),
                "Z":np.arange(-0.5, 0.6, 0.5),
                "alpha": np.array([0.0])},
        air=air, wl_range=wl_range, base=base)

        self.norm = norm #Normalize to 1 solar luminosity?
        self.rname = self.base + "CIFIST2011/M{Z:}/lte{temp:0>3.0f}-{logg:.1f}{Z:}.BT-Settl.spec.7.bz2"
        # self.Z_dict = {-2:"-2.0", -1.5:"-1.5", -1:'-1.0', -0.5:'-0.5', 0.0: '-0.0', 0.5: '+0.5', 1: '+1.0'}
        self.Z_dict = {-0.5:'-0.5a+0.2', 0.0: '-0.0a+0.0', 0.5: '+0.5a0.0'}

        wl_dict = create_log_lam_grid(wl_start=self.wl_range[0], wl_end=self.wl_range[1], min_vc=0.08/C.c_kms)
        self.wl = wl_dict['wl']


    def load_flux(self, parameters):
        '''
        Because of the crazy format of the BTSettl, we need to sort the wl to make sure
        everything is unique, and we're not screwing ourselves with the spline.
        '''

        super().load_file(parameters) #Check to make sure that the keys are allowed and that the values are in the grid

        str_parameters = parameters.copy()

        #Rewrite Z
        Z = parameters["Z"]
        str_parameters["Z"] = self.Z_dict[Z]

        #Multiply temp by 0.01
        str_parameters["temp"] = 0.01 * parameters['temp']

        fname = self.rname.format(**str_parameters)
        file = bz2.BZ2File(fname, 'r')

        lines = file.readlines()
        strlines = [line.decode('utf-8') for line in lines]
        file.close()

        data = ascii.read(strlines, col_starts=[0, 13], col_ends=[12, 25], Reader=ascii.FixedWidthNoHeader)
        wl = data['col1']
        fl_str = data['col2']

        fl = idl_float(fl_str) #convert because of "D" exponent, unreadable in Python
        fl = 10 ** (fl - 8.) #now in ergs/cm^2/s/A

        #"Clean" the wl and flux points. Remove duplicates, sort in increasing wl
        wl, ind = np.unique(wl, return_index=True)
        fl = fl[ind]

        if self.norm:
            F_bol = trapz(fl, wl)
            fl = fl * (C.F_sun / F_bol)
            # the bolometric luminosity is always 1 L_sun

        # truncate the spectrum to the wl range of interest
        # at this step, make the range a little more so that the next stage of
        # spline interpolation is properly in bounds
        ind = (wl >= (self.wl_range[0] - 50.)) & (wl <= (self.wl_range[1] + 50.))
        wl = wl[ind]
        fl = fl[ind]

        if self.air:
            #Shift the wl that correspond to the raw spectrum
            wl = vacuum_to_air(wl)

        #Now interpolate wl, fl onto self.wl
        interp = InterpolatedUnivariateSpline(wl, fl, k=5)
        fl_interp = interp(self.wl)

        return fl_interp

class CIFISTGridInterface(RawGridInterface):
    '''CIFIST grid interface, grid available here: https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/FITS/.
    Unlike the PHOENIX and Kurucz grids, the
    individual files of the BTSettl grid do not always have the same wavelength
    sampling. Therefore, each call of :meth:`load_flux` will interpolate the
    flux onto a LogLambda spaced grid that ranges between `wl_range` and has a
    velocity spacing of 0.08 km/s or better.

    If you have a choice, it's probably easier to use the Husser PHOENIX grid.
    '''
    def __init__(self, air=True, norm=True, wl_range=[3000, 13000], base=Starfish.grid["raw_path"]):
        super().__init__(name="CIFIST",
        points=[np.concatenate((np.arange(1200, 2351, 50), np.arange(2400, 7001, 100)), axis=0),
                np.arange(2.5, 5.6, 0.5)],
                param_names = ["temp", "logg"],
                air=air, wl_range=wl_range, base=base)

        self.par_dicts = [None, None]
        self.norm = norm #Normalize to 1 solar luminosity?
        self.rname = self.base + "lte{0:0>5.1f}-{1:.1f}-0.0a+0.0.BT-Settl.spec.fits.gz"

        wl_dict = create_log_lam_grid(dv=0.08, wl_start=self.wl_range[0], wl_end=self.wl_range[1])
        self.wl = wl_dict['wl']

        print(self.wl)


    def load_flux(self, parameters):
        '''
        Because of the crazy format of the BTSettl, we need to sort the wl to make sure
        everything is unique, and we're not screwing ourselves with the spline.
        '''

        self.check_params(parameters)

        str_parameters = []
        for param, par_dict in zip(parameters, self.par_dicts):
            if par_dict is None:
                str_parameters.append(param)
            else:
                str_parameters.append(par_dict[param])


        #Multiply temp by 0.01
        str_parameters[0] = 0.01 * parameters[0]

        fname = self.rname.format(*str_parameters)

        #Still need to check that file is in the grid, otherwise raise a C.GridError
        #Read all metadata in from the FITS header, and append to spectrum
        try:
            flux_file = fits.open(fname)
            data = flux_file[1].data
            hdr = flux_file[1].header

            wl = data["Wavelength"] * 1e4 # [Convert to angstroms]
            fl = data["Flux"]

            flux_file.close()
        except OSError:
            raise C.GridError("{} is not on disk.".format(fname))

        #"Clean" the wl and flux points. Remove duplicates, sort in increasing wl
        wl, ind = np.unique(wl, return_index=True)
        fl = fl[ind]

        if self.norm:
            F_bol = trapz(fl, wl)
            fl = fl * (C.F_sun / F_bol)
            # the bolometric luminosity is always 1 L_sun

        # truncate the spectrum to the wl range of interest
        # at this step, make the range a little more so that the next stage of
        # spline interpolation is properly in bounds
        ind = (wl >= (self.wl_range[0] - 50.)) & (wl <= (self.wl_range[1] + 50.))
        wl = wl[ind]
        fl = fl[ind]

        if self.air:
            #Shift the wl that correspond to the raw spectrum
            wl = vacuum_to_air(wl)

        #Now interpolate wl, fl onto self.wl
        interp = InterpolatedUnivariateSpline(wl, fl, k=5)
        fl_interp = interp(self.wl)

        #Add temp, logg, Z, norm to the metadata
        header = {}
        header["norm"] = self.norm
        header["air"] = self.air
        #Keep the relevant keywords
        for key, value in hdr.items():
            header[key] = value

        return (fl_interp, header)


class HDF5Creator:
    '''
    Create a HDF5 grid to store all of the spectra from a RawGridInterface,
    along with metadata.

    '''
    def __init__(self, GridInterface, filename, Instrument, ranges=None,
        key_name=Starfish.grid["key_name"], vsinis=None):
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
            for par in Starfish.parname:
                ranges.append([-np.inf,np.inf])

        self.GridInterface = GridInterface
        self.filename = os.path.expandvars(filename) #only store the name to the HDF5 file, because
        # otherwise the object cannot be parallelized
        self.Instrument = Instrument

        # The flux formatting key will always have alpha in the name, regardless
        # of whether or not the library uses it as a parameter.
        self.key_name = key_name

        # Take only those points of the GridInterface that fall within the ranges specified
        self.points = []

        # We know which subset we want, so use these.
        for i,(low, high) in enumerate(ranges):
            valid_points  = self.GridInterface.points[i]
            ind = (valid_points >= low) & (valid_points <= high)
            self.points.append(valid_points[ind])
            # Note that at this point, this is just the grid points that fall within the rectangular
            # bounds set by ranges. If the raw library is actually irregular (e.g. CIFIST),
            # then self.points will contain points that don't actually exist in the raw library.

        # the raw wl from the spectral library
        self.wl_native = self.GridInterface.wl #raw grid
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

        #inst_min, inst_max = self.Instrument.wl_range
        wl_min, wl_max = Starfish.grid["wl_range"]
        buffer = Starfish.grid["buffer"] # [AA]
        wl_min -= buffer
        wl_max += buffer

        # If the raw synthetic grid doesn't span the full range of the user
        # specified grid, raise an error.
        # Instead, let's choose the maximum limit of the synthetic grid?
        if (self.wl_native[0] > wl_min) or (self.wl_native[-1] < wl_max):
            print("Synthetic grid does not encapsulate chosen wl_range in config.yaml, truncating new grid to extent of synthetic grid, {}, {}".format(self.wl_native[0], self.wl_native[-1]))
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
        sigma = self.Instrument.FWHM / 2.35 # in km/s
        # Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space
        self.taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (self.ss ** 2))

        self.ss[0] = 0.01 # junk so we don't get a divide by zero error

        # The final wavelength grid, onto which we will interpolate the
        # Fourier filtered wavelengths, is part of the Instrument object
        dv_temp = self.Instrument.FWHM/self.Instrument.oversampling
        wl_dict = create_log_lam_grid(dv_temp, wl_min, wl_max)
        self.wl_final = wl_dict["wl"]
        self.dv_final = calculate_dv_dict(wl_dict)

        #Create the wl dataset separately using float64 due to rounding errors w/ interpolation.
        wl_dset = self.hdf5.create_dataset("wl", (len(self.wl_final),), dtype="f8", compression='gzip', compression_opts=9)
        wl_dset[:] = self.wl_final
        wl_dset.attrs["air"] = self.GridInterface.air
        wl_dset.attrs["dv"] = self.dv_final


    def process_flux(self, parameters):
        '''
        Take a flux file from the raw grid, process it according to the
        instrument, and insert it into the HDF5 file.

        :param parameters: the model parameters.
        :type parameters: 1D np.array

        .. note::

        :raises AssertionError: if the `parameters` vector is not
            the same length as that of the raw grid.

        :returns: a tuple of (parameters, flux, header). If the flux could
            not be loaded, returns (None, None, None).

        '''
        # assert len(parameters) == len(Starfish.parname), "Must pass numpy array {}".format(Starfish.parname)
        print("Processing", parameters)

        # If the parameter length is one more than the grid pars,
        # assume this is for vsini convolution
        if len(parameters) == (len(Starfish.parname) + 1):
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

        for i,param in enumerate(all_params):
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
            for key,value in header.items():
                if key != "" and value != "": #check for empty FITS kws
                    flux.attrs[key] = value

        # Remove parameters that do no exist
        all_params = np.delete(all_params, invalid_params, axis=0)

        par_dset = self.hdf5.create_dataset("pars", all_params.shape, dtype="f8", compression='gzip', compression_opts=9)
        par_dset[:] = all_params

        self.hdf5.close()


class HDF5Interface:
    '''
    Connect to an HDF5 file that stores spectra.
    '''
    def __init__(self, filename=Starfish.grid["hdf5_path"], key_name=Starfish.grid["key_name"]):
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

        #determine the bounding regions of the grid by sorting the grid_points
        low = np.min(self.grid_points, axis=0)
        high = np.max(self.grid_points, axis=0)
        self.bounds = np.vstack((low, high)).T
        self.points = [np.unique(self.grid_points[:, i]) for i in range(self.grid_points.shape[1])]

        self.ind = None #Overwritten by other methods using this as part of a ModelInterpolator

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

        #Note: will raise a KeyError if the file is not found.

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

        #Note: will raise a KeyError if the file is not found.

        return (fl, hdr)

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
        self.npars = len(Starfish.grid["parname"])
        self._determine_chunk_log(wl)

        self.setup_index_interpolators()
        self.cache = OrderedDict([])
        self.cache_max = cache_max
        self.cache_dump = cache_dump #how many to clear once the maximum cache has been reached

    def _determine_chunk_log(self, wl):
        '''
        Using the DataSpectrum, determine the minimum chunksize that we can use and then
        truncate the synthetic wavelength grid and the returned spectra.

        Assumes HDF5Interface is LogLambda spaced, because otherwise you shouldn't need a grid
        with 2^n points, because you would need to interpolate in wl space after this anyway.
        '''

        wl_interface = self.interface.wl # The grid we will be truncating.
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

        assert min(self.wl) < wl_min and max(self.wl) > wl_max, "ModelInterpolator chunking ({:.2f}, {:.2f}) didn't encapsulate full DataSpectrum range ({:.2f}, {:.2f}).".format(min(self.wl),  max(self.wl), wl_min, wl_max)


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
        self.fluxes = np.empty((2**self.npars, lenF)) #8 rows, for temp, logg, Z

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

        #Edges is a list of [((6000, 6100), (0.2, 0.8)), ((), ()), ((), ())]

        params = [tup[0] for tup in edges] #[(6000, 6100), (4.0, 4.5), ...]
        weights = [tup[1] for tup in edges] #[(0.2, 0.8), (0.4, 0.6), ...]

        #Selects all the possible combinations of parameters
        param_combos = list(itertools.product(*params))
        #[(6000, 4.0, 0.0), (6100, 4.0, 0.0), (6000, 4.5, 0.0), ...]
        weight_combos = list(itertools.product(*weights))
        #[(0.2, 0.4, 1.0), (0.8, 0.4, 1.0), ...]

        # Assemble key list necessary for indexing cache
        key_list = [self.interface.key_name.format(*param) for param in param_combos]
        weight_list = np.array([np.prod(weight) for weight in weight_combos])

        assert np.allclose(np.sum(weight_list), np.array(1.0)), "Sum of weights must equal 1, {}".format(np.sum(weight_list))

        #Assemble flux vector from cache, or load into cache if not there
        for i,param in enumerate(param_combos):
            key = key_list[i]
            if key not in self.cache.keys():
                try:
                    #This method already allows loading only the relevant region from HDF5
                    fl = self.interface.load_flux(np.array(param))
                except KeyError as e:
                    raise C.InterpolationError("Parameters {} not in master HDF5 grid. {}".format(param, e))
                self.cache[key] = fl

            self.fluxes[i,:] = self.cache[key]*weight_list[i]

        # Do the averaging and then normalize the average flux to 1.0
        fl = np.sum(self.fluxes, axis=0)
        fl /= np.median(fl)
        return fl


#Convert R to FWHM in km/s by \Delta v = c/R
class Instrument:
    '''
    Object describing an instrument. This will be used by other methods for
    processing raw synthetic spectra.

    :param name: name of the instrument
    :type name: string
    :param FWHM: the FWHM of the instrumental profile in km/s
    :type FWHM: float
    :param wl_range: wavelength range of instrument
    :type wl_range: 2-tuple (low, high)
    :param oversampling: how many samples fit across the :attr:`FWHM`
    :type oversampling: float

    Upon initialization, calculates a ``wl_dict`` with the properties of the
    instrument.
    '''
    def __init__(self, name, FWHM, wl_range, oversampling=4.):
        self.name = name
        self.FWHM = FWHM #km/s
        self.oversampling = oversampling
        self.wl_range = wl_range

    def __str__(self):
        '''
        Prints the relevant properties of the instrument.
        '''
        return "Instrument Name: {}, FWHM: {:.1f}, oversampling: {}, " \
            "wl_range: {}".format(self.name, self.FWHM, self.oversampling, self.wl_range)


class TRES(Instrument):
    '''TRES instrument'''
    def __init__(self, name="TRES", FWHM=6.8, wl_range=(3500, 9500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)
        #sets the FWHM and wl_range

class Reticon(Instrument):
    '''Reticon Instrument'''
    def __init__(self, name="Reticon", FWHM=8.5, wl_range=(5145,5250)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class KPNO(Instrument):
    '''KNPO Instrument'''
    def __init__(self, name="KPNO", FWHM=14.4, wl_range=(6250,6650)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class SPEX(Instrument):
    '''SPEX Instrument'''
    def __init__(self, name="SPEX", FWHM=150., wl_range=(7500, 54000)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class SPEX_SXD(Instrument):
    '''SPEX Instrument short mode'''
    def __init__(self, name="SPEX", FWHM=150., wl_range=(7500, 26000)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class IGRINS_H(Instrument):
    '''IGRINS H band instrument'''
    def __init__(self, name="IGRINS_H", FWHM=7.5, wl_range=(14250, 18400)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)
        self.air = False

class IGRINS_K(Instrument):
    '''IGRINS K band instrument'''
    def __init__(self, name="IGRINS_K", FWHM=7.5, wl_range=(18500, 25200)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)
        self.air = False

class ESPaDOnS(Instrument):
    '''ESPaDOnS Instrument'''
    def __init__(self, name="ESPaDOnS", FWHM=4.4, wl_range=(3700, 10500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class DCT_DeVeny(Instrument):
    '''DCT DeVeny spectrograph Instrument.'''
    def __init__(self, name="DCT_DeVeny", FWHM=105.2, wl_range=(6000, 10000)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class WIYN_Hydra(Instrument):
    '''WIYN Hydra spectrograph Instrument.'''
    def __init__(self, name="WIYN_Hydra", FWHM=300., wl_range=(5500, 10500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

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

    #replace 'D' with 'E', convert to float
    return np.float(idl_num.replace("D", "E"))


def load_BTSettl(temp, logg, Z, norm=False, trunc=False, air=False):
    rname = "BT-Settl/CIFIST2011/M{Z:}/lte{temp:0>3.0f}-{logg:.1f}{Z:}.BT-Settl.spec.7.bz2".format(temp=0.01 * temp, logg=logg, Z=Z)
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

        Smoothly handles the *C.InterpolationError* if parameters cannot be interpolated from the grid and prints a message.
        '''

        #Preserve the "popping of parameters"
        parameters = parameters.copy()

        #Load the correct C.grid_set value from the interpolator into a LogLambdaSpectrum
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

    def __init__(self, interface, instrument, points, flux_unit, outdir, alpha=False, integrate=False, processes=mp.cpu_count()):
        self.interface = interface
        self.instrument = instrument
        self.points = points #points is a dictionary with which values to spit out for each parameter
        self.filename = "t{temp:0>5.0f}g{logg:0>2.0f}{Z_flag}{Z:0>2.0f}v{vsini:0>3.0f}.fits"
        self.flux_unit = flux_unit
        self.integrate = integrate
        self.outdir = outdir
        self.processes = processes
        self.pids = []
        self.alpha = alpha

        self.vsini_points = self.points.pop("vsini")
        names = self.points.keys()

        #Creates a list of parameter dictionaries [{"temp":8500, "logg":3.5, "Z":0.0}, {"temp":8250, etc...}, etc...]
        #which does not contain vsini
        self.param_list = [dict(zip(names,params)) for params in itertools.product(*self.points.values())]

        #Create a master wl_dict which correctly oversamples the instrumental kernel
        self.wl_dict = self.instrument.wl_dict
        self.wl = self.wl_dict["wl"]

        #Check that temp, logg, Z are within the bounds of the interface
        for key,value in self.points.items():
            min_val, max_val = self.interface.bounds[key]
            assert np.min(self.points[key]) >= min_val,"Points below interface bound {}={}".format(key, min_val)
            assert np.max(self.points[key]) <= max_val,"Points above interface bound {}={}".format(key, max_val)

        #Create a temporary grid to resample to that matches the bounds of the instrument.
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
            #Check to see if alpha, otherwise append alpha=0 to the parameter list.
            if not self.alpha:
                parameters.update({"alpha": 0.0})
            print(parameters)

            if parameters["Z"] < 0:
                zflag = "m"
            else:
                zflag = "p"

            #This is a Base1DSpectrum
            base_spec = self.interface.load_file(parameters)

            master_spec = base_spec.to_LogLambda(instrument=self.instrument, min_vc=0.1/C.c_kms) #convert the Base1DSpectrum to a LogLamSpectrum

            #Now process the spectrum for all values of vsini

            for vsini in self.vsini_points:
                spec = master_spec.copy()
                #Downsample the spectrum to the instrumental resolution, integrate to give counts/pixel
                spec.instrument_and_stellar_convolve(self.instrument, vsini, integrate=self.integrate)

                #Update spectrum with vsini
                spec.metadata.update({"vsini":vsini})
                filename = self.outdir + self.filename.format(temp=parameters["temp"], logg=10*parameters["logg"],
                                                              Z=np.abs(10*parameters["Z"]), Z_flag=zflag, vsini=vsini)

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
            #Make sure all threads have finished
            p.join()

def main():
    pass


if __name__ == "__main__":
    main()
