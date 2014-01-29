import numpy as np
from numpy.fft import fft, ifft, fftfreq
import astropy.io.fits as pf
from astropy.io import ascii,fits
import warnings
import pytest

import multiprocessing as mp

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d, UnivariateSpline
from scipy.integrate import trapz
from scipy.special import j1

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter as FSF

import gc
import bz2
import h5py
from functools import partial
import itertools

from model import Base1DSpectrum, LogLambdaSpectrum

c_kms = 2.99792458e5 #km s^-1
c_ang = 2.99792458e18 #A s^-1
L_sun = 3.839e33 #erg/s, PHOENIX header says W, but is really erg/s
R_sun = 6.955e10 #cm
F_sun = L_sun / (4 * np.pi * R_sun ** 2) #bolometric flux of the Sun measured at the surface

#Allowed grid parameters
grid_parameters = frozenset(("temp", "logg", "Z", "alpha"))

#Allowed "post processing parameters"
pp_parameters = frozenset(("vsini", "FWHM", "vz", "Av", "Omega"))

#All allowed parameters, the union of grid_parameters and pp_parameters
all_parameters = grid_parameters | pp_parameters

#Dictionary of allowed variables with default values
var_default = {"temp":5800, "logg":4.5, "Z":0.0, "alpha":0.0, "vsini":0.0, "FWHM": 0.0, "vz":0.0, "Av":0.0, "Omega":1.0}

class GridError(Exception):
    def __init__(self, msg):
        self.msg = msg

class Base:
    def __init__(self, parameters):
        '''Parameters are given as a dictionary, which is cycled through and instantiates self.parameters.
         If the parameter is not found allowed parameters, raise a warning and do not instantiate it. Further down the
         line, if a method needs a value for a parameter, it can query the default value, otherwise it can act as
         a short circuit if this parameter is not in the dictionary.'''
        self.parameters = set([])
        assert type(parameters) is dict, "Parameters must be a dictionary"
        for key, value in parameters.items():
            if key in var_default.keys():
                self.parameters[key] = value
            else:
                warnings.warn("{0} is not an allowed parameter, skipping".format(key), UserWarning)

    def __str__(self):
        prtstr = "".join(["{0} : {1:.2f} \n".format(key, value) for key,value in self.parameters.items()])
        return prtstr


class RawGridInterface:
    '''Takes in points, which is a dictionary with key values as parameters and values the sets of grid points.'''
    def __init__(self, name, points, air=True, wl_range=[3000,13000]):
        self.name = name
        self.points = {}
        assert type(points) is dict, "points must be a dictionary."
        for key, value in points.items():
            if key in grid_parameters:
                self.points[key] = value
            else:
                warnings.warn("{0} is not an allowed parameter, skipping".format(key), UserWarning)

        self.air = air #read files in air wavelengths?
        self.wl_range = wl_range #values to truncate grid

    def check_params(self, parameters):
        '''Checks to see if parameter dict is a subset of allowed parameters, otherwise raises an AssertionError,
        which can be chosen to be handled later..'''
        if not set(parameters.keys()) <= grid_parameters:
            raise GridError("{} not in allowable grid parameters {}".format(parameters.keys(), grid_parameters))

        for key,value in parameters.items():
            if value not in self.points[key]:
                raise GridError("{} not in the grid points {}".format(value, sorted(self.points[key])))

    def load_file(self, parameters, norm=True):
        '''Designed to be overwritten by an extended class'''
        self.check_params(parameters)
        #loads file
        #truncates to wl_range
        #returns a spectrum object defined by subclass
        #also includes metadata from the FITS header

class PHOENIXGridInterface(RawGridInterface):
    def __init__(self, air=True, norm=True):
        super().__init__(name="PHOENIX",
        points={"temp":
      np.array([2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 4000, 4100, 4200,
      4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100,
      6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000,
      9200, 9400, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 11200, 11400, 11600, 11800, 12000]),
        "logg":np.arange(0.0, 6.1, 0.5),
        "Z":np.arange(-1., 1.1, 0.5),
        "alpha":np.array([0.0, 0.2, 0.4, 0.6, 0.8])},
        air=air, wl_range=[3000, 13000])

        self.norm = norm #Normalize to 1 solar luminosity?
        self.rname = "raw_grids/PHOENIX/Z{Z:}/lte{temp:0>5.0f}-{logg:.2f}{Z:}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
        self.Z_dict = {-1: '-1.0', -0.5:'-0.5', 0.0: '-0.0', 0.5: '+0.5', 1: '+1.0'}
        self.alpha_dict = {-0.2:".Alpha=-0.20", 0.0: "", 0.2:".Alpha=+0.20", 0.4:".Alpha=+0.40", 0.6:".Alpha=+0.60",
                           0.8:".Alpha=+0.80"}

        #if air is true, convert the normally vacuum file to air wls.
        wl_file = pf.open("raw_grids/PHOENIX/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
        w_full = wl_file[0].data
        wl_file.close()
        if self.air:
            self.wl_full = vacuum_to_air(w_full)
        else:
            self.wl_full = w_full

        self.ind = (self.wl_full >= self.wl_range[0]) & (self.wl_full <= self.wl_range[1])
        self.wl = self.wl_full[self.ind]
        self.rname = "raw_grids/PHOENIX/Z{Z:}{alpha:}/lte{temp:0>5.0f}-{logg:.2f}{Z:}{alpha:}" \
                     ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    def load_file(self, parameters):
        super().load_file(parameters) #Check to make sure that the keys are allowed and that the values are in the grid

        str_parameters = parameters.copy()
        #Rewrite Z
        Z = parameters["Z"]
        str_parameters["Z"] = self.Z_dict[Z]

        #Rewrite alpha, allow alpha to be missing from parameters
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
            flux_file = pf.open(fname)
            f = flux_file[0].data
            hdr = flux_file[0].header
            flux_file.close()
        except OSError:
            raise GridError("{} is not on disk.".format(fname))

        #If we want to normalize the spectra, we must do it now since later we won't have the full EM range
        if self.norm:
            f *= 1e-8 #convert from erg/cm^2/s/cm to erg/cm^2/s/A
            F_bol = trapz(f, self.wl_full)
            f = f * (F_sun / F_bol) #bolometric luminosity is always 1 L_sun

        #Add temp, logg, Z, alpha, norm to the metadata
        header = parameters
        header["norm"] = self.norm
        #Keep only the relevant PHOENIX keywords, which start with PHX
        for key, value in hdr.items():
            if key[:3] == "PHX":
                header[key] = value

        return Base1DSpectrum(self.wl, f[self.ind], metadata=header, air=self.air)


class KuruczGridInterface:
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

class BTSettlGridInterface:
    def __init__(self):
        pass

def process_flux(tup):
    HDF5, parameters = tup
    print(parameters)
    #HDF5Obj.process_flux(parameters)

class HDF5Holder:
    def __init__(self, fname):
        self.file = h5py.File(fname, "w")

class HDF5GridCreator:
    '''Take a GridInterface, load all spectra in specified ranges (default all), and then stuff them into an HDF5
    file with the proper attributes.

    HDF5 file has the following levels
    /

    wl
       - CRVAL1
       - CRDELT1
       - NAXIS1
       - vres

    flux/
        t6000g3.5z0.0a0.0
           - temp
           - logg
           - Z
           - alpha
           - PHX params
           - anything else

        t6100g3.5z0.0a0.0
        etc...
       '''
    def __init__(self, GridInterface, ranges, filename, wldict, chunksize):
        self.GridInterface = GridInterface
        self.filename = filename #only store the name to the HDF5 file, because the object cannot be parallelized
        self.flux_name = "t{temp:.0f}g{logg:.1f}z{Z:.1f}a{alpha:.1f}"
        self.chunksize = chunksize

        #Discern between HDF5GridCreator points and GridInterface points using ranges
        self.points = {}
        for key, value in ranges.items():
            valid_points  = self.GridInterface.points[key]
            low,high = value
            ind = (valid_points >= low) & (valid_points <= high)
            self.points[key] = valid_points[ind]

        #wldict is the output from create_log_lam_grid, containing CRVAL1, CDELT1, etc...
        self.wl = wldict.pop("wl")
        self.wl_params = wldict

        with h5py.File(self.filename, "w") as hdf5:
            hdf5.attrs["grid_name"] = GridInterface.name
            hdf5.flux_group = hdf5.create_group("flux")
            hdf5.flux_group.attrs["unit"] = "erg/cm^2/s/A"
            self.create_wl(hdf5)

        #The HDF5 master grid will always have alpha in the name, regardless of whether GridIterface uses it.

        #TODO: check alpha behavior
        #TODO: store points as attr of fl folder, eh, too much to store though

    def create_wl(self, hdf5):
        wl_dset = hdf5.create_dataset("wl", (len(self.wl),), dtype="f", compression='gzip', compression_opts=9)
        wl_dset[:] = self.wl
        for key, value in self.wl_params.items():
            wl_dset.attrs[key] = value
            wl_dset.attrs["air"] = self.GridInterface.air

    def process_flux(self, parameters):
        #'''Assumes that it's going to get parameters (temp, logg, Z, alpha), regardless of whether
        #the GridInterface actually has alpha or not.'''
        try:
            spec = self.GridInterface.load_file(parameters) #this
            spec.resample_to_grid(self.wl) #and this are the two slow methods
            return (parameters,spec)

        except GridError:
            print("Not able to process file with parameters {}".format(parameters))

    def process_grid(self):
        #Take all parameter permutations in self.points and create a list
        param_list = [] #list of parameter dictionaries
        keys,values = self.points.keys(),self.points.values()

        #use itertools.product to create permutations of all possible values
        for i in itertools.product(*values):
            param_list.append(dict(zip(keys,i)))

        pool = mp.Pool(4)
        with h5py.File(self.filename, "r+") as hdf5:
            for parameters, spec in pool.imap(self.process_flux, param_list): #python 3 is lazy map
                flux = hdf5["flux"].create_dataset(self.flux_name.format(**parameters), shape=(len(spec.fl),),
                                                      dtype="f", compression='gzip', compression_opts=9)
                flux[:] = spec.fl

                #Store header keywords as attributes in HDF5 file
                for key,value in spec.metadata.items():
                    flux.attrs[key] = value


class HDF5Interface:
    '''Connect to an HDF5 file that stores spectra. Properly close the file when done.
    Useful for creating grid when using a GridProcessor, or when reading from the
    HDF5 grid in model or general use.'''
    def __init__(self, name, filename, mode, wave_grid, temp_points, logg_points, Z_points, alpha_points=None):
        super().__init__(name, temp_points, logg_points, Z_points, alpha_points)
        self.wave_grid = wave_grid
        lenT, lenG, lenZ, lenA, lenwl = (len(self.temp_points), len(self.logg_points),
            len(self.Z_points), len(self.alpha_points), len(self.wave_grid))
        if lenA == 1:
            self.shape = (lenT, lenG, lenZ, lenwl)
        else:
            self.shape = (lenT, lenG, lenZ, lenA, lenwl)

        #Create index lookup
        self.temp_index = dict(zip(self.temp_points, np.arange(lenT)))
        self.logg_index = dict(zip(self.logg_points, np.arange(lenG)))
        self.Z_index = dict(zip(self.Z_points, np.arange(lenZ)))
        self.alpha_index = dict(zip(self.alpha_points, np.arange(lenA)))

        if mode == "w": #write
            self.new(filename)
        elif mode == "r": #read
            self.open(filename)

    def __del__(self):
        self.close()

    def open(self, filename):
        self.HDF5_file = h5py.File(filename, 'r')
        self.dset = self.HDF5_file['LIB']

    def new(self, filename):
        self.HDF5_file = h5py.File(filename, "w")
        self.dset = self.HDF5_file.create_dataset("LIB", self.shape, dtype="f", compression='gzip', compression_opts=9)

    def close(self):
        print("closing file")
        self.HDF5_file.close()

    def load_flux(self, temp, logg, Z, alpha=0.0):
        super().load_file(temp, logg, Z, alpha)
        return self.dset[self.temp_index[temp], self.logg_index[logg], self.Z_index[Z], self.alpha_index[alpha], :]

    def load_spectrum(self, temp, logg, Z, alpha=0.0):
        super().load_file(temp, logg, Z, alpha)
        flux = self.load_flux(temp, logg, Z, alpha)
        return Base1DSpectrum(self.wave_grid, flux, air=self.air)

    def write_flux(self, flux, temp, logg, Z, alpha):
        assert len(flux) == len(self.wave_grid)
        self.dset[self.temp_index[temp], self.logg_index[logg], self.Z_index[Z], self.alpha_index[alpha], :] = flux

class GridProcessor:
    def __init__(self, grid=None, temp_range=[0, np.inf], logg_range=[0, np.inf], Z_range=[0, np.inf],
                 alpha_range=[0, np.inf], chunksize=20):
        self.grid = grid

        #Create interface object using range, and parameters of host grid
        self.out_grid = HDF5Interface("PHOENIX", "PHOENIX.h5py", "w",
        temp_points=grid.temp_points[(grid.temp_points >= temp_range[0]) & (grid.temp_points <= temp_range[1])],
        logg_points=grid.logg_points[(grid.logg_points >= logg_range[0]) & (grid.logg_points <= logg_range[1])],
        Z_points=grid.Z_points[(grid.Z_points >= Z_range[0]) & (grid.Z_points <= Z_range[1])],
        alpha_points=grid.alpha_points[(grid.alpha_points >= alpha_range[0]) & (grid.alpha_points <= alpha_range[1])])

        #Uses grid.wave_grid as in_grid

        #If grid.wave_grid is not log-lambda spaced, it converts it

        #Uses interface wave_grid as out_grid


        self.pool = mp.Pool(mp.cpu_count())
        self.chunksize = chunksize

    def process_all(self):
        self.index_combos = []
        self.var_combos = []
        for t, temp in enumerate(self.temp_points):
            for l, logg in enumerate(self.logg_points):
                for z, Z in enumerate(self.Z_points):
                    for a, A in enumerate(self.alpha_points):
                        self.index_combos.append([t, l, z, a])
                        self.var_combos.append([temp, logg, Z, A])

        spec_gen = self.pool.imap(self.process_spectrum, self.var_combos, chunksize=20)

        for i, spec in enumerate(spec_gen):
            #t, l, z, a = index_combos[i]
            self.save(self.index_combos[i], spec)

    def process_spectrum(self, var_combos):
        '''This depends on what you want to do (resample, convolve, etc) and must be defined in a subclass.'''
        spec = self.grid.load_file(var_combos)

        #Convert to a LogLamb grid

        raise NotImplementedError

    def resample_and_convolve(f, wg_raw, wg_fine, wg_coarse, wg_fine_d=0.35, sigma=2.89):
        '''Take a full-resolution PHOENIX model spectrum `f`, with raw spacing wg_raw, resample it to wg_fine
        (done because the original grid is not log-linear spaced), instrumentally broaden it in the Fourier domain,
        then resample it to wg_coarse. sigma in km/s.'''

        #resample PHOENIX to 0.35km/s spaced grid using InterpolatedUnivariateSpline. First check to make sure there
        #are no duplicates and the wavelength is increasing, otherwise the spline will fail and return NaN.
        wl_sorted, ind = np.unique(wg_raw, return_index=True)
        fl_sorted = f[ind]
        interp_fine = InterpolatedUnivariateSpline(wl_sorted, fl_sorted)
        f_grid = interp_fine(wg_fine)

        #Fourier Transform
        out = fft(f_grid)
        #The frequencies (cycles/km) corresponding to each point
        freqs = fftfreq(len(f_grid), d=wg_fine_d)

        #Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space (corresponding to FWHM 6.8km/s)
        taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (freqs ** 2))
        tout = out * taper

        #Take the broadened spectrum back to wavelength space
        f_grid6 = ifft(tout)
        #print("Total of imaginary components", np.sum(np.abs(np.imag(f_grid6))))

        #Resample the broadened spectrum to a uniform coarse grid
        interp_coarse = InterpolatedUnivariateSpline(wg_fine, np.abs(f_grid6))
        f_coarse = interp_coarse(wg_coarse)

        del interp_fine
        del interp_coarse
        gc.collect() #necessary to prevent memory leak!

        return f_coarse

    def save(self, i, spec):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

class TRES_PHOENIX_HDF5_Processor(GridProcessor):
    def __init__(self, temp_range=[0, np.inf], logg_range=[0, np.inf], Z_range=[0, np.inf],
                 alpha_range=[0, np.inf]):
        super().__init__(temp_range=[0, np.inf], logg_range=[0, np.inf], Z_range=[0, np.inf],
                         alpha_range=[0, np.inf])

        #Set the interface as a HDF5 grid
        #Set the grid as a PHOENIX grid
        #Set the instrument as TRES


        #Gets the output grid parameters from the instrument

class Instrument:
    def __init__(self):
        self.response_kernel
        self.min_resolution
        pass





wl_file = pf.open("raw_grids/PHOENIX/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
w_full = wl_file[0].data
wl_file.close()
ind = (w_full > 3000.) & (w_full < 13000.) #this corresponds to some extra space around the
# shortest U and longest z band

global w
w = w_full[ind]
len_p = len(w)

wave_grid_raw_PHOENIX = np.load("wave_grids/PHOENIX_raw_trim_air.npy")
wave_grid_fine = np.load('wave_grids/PHOENIX_0.35kms_air.npy')
wave_grid_coarse = np.load('wave_grids/PHOENIX_2kms_air.npy')
wave_grid_kurucz_raw = np.load("wave_grids/kurucz_raw.npy")
wave_grid_2kms_kurucz = np.load("wave_grids/kurucz_2kms_air.npy") #same wl as PHOENIX_2kms_air, but trimmed

L_sun = 3.839e33 #erg/s, PHOENIX header says W, but is really erg/s
R_sun = 6.955e10 #cm

F_sun = L_sun / (4 * np.pi * R_sun ** 2) #bolometric flux of the Sun measured at the surface

grids = {"PHOENIX": {'T_points': np.array(
    [2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 4000, 4100, 4200,
     4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100,
     6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000,
     9200, 9400, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 11200, 11400, 11600, 11800, 12000]),
                     'logg_points': np.arange(0.0, 6.1, 0.5), 'Z_points': ['-1.0', '-0.5', '-0.0', '+0.5', '+1.0'],
                     'alpha_points': [".Alpha=-0.20", "", ".Alpha=+0.20", ".Alpha=+0.40", ".Alpha=+0.60",
                                      ".Alpha=+0.80"]},
         "kurucz": {'T_points': np.arange(3500, 9751, 250),
                    'logg_points': np.arange(1.0, 5.1, 0.5), 'Z_points': ["m05", "p00", "p05"]},
         'BTSettl': {'T_points': np.arange(3000, 7001, 100), 'logg_points': np.arange(2.5, 5.6, 0.5),
                     'Z_points': ['-0.5a+0.2', '-0.0a+0.0', '+0.5a+0.0']}}


def create_wave_grid(v=1., start=3700., end=10000):
    '''Returns a grid evenly spaced in velocity'''
    size = 9000000 #this number just has to be bigger than the final array
    lam_grid = np.zeros((size,))
    i = 0
    lam_grid[i] = start
    vel = np.sqrt((c_kms + v) / (c_kms - v))
    while (lam_grid[i] < end) and (i < size - 1):
        lam_new = lam_grid[i] * vel
        i += 1
        lam_grid[i] = lam_new
    return lam_grid[np.nonzero(lam_grid)][:-1]

def create_log_lam_grid(wl_start=3000., wl_end=13000., min_WL=None, min_V=None):
    '''min_WL = (delta_WL, WL). Takes the finer of the two specified.'''
    if (min_WL is None) and (min_V is None):
        raise ValueError("You need to specify either min_WL or min_V")
    if min_WL is not None:
        delta_wl, wl = min_WL #unpack
        Vwl = delta_wl/wl
        min_vc = Vwl
    if min_V is not None:
        Vv = min_V/c_kms
        min_vc = Vv
    if (min_WL is not None) and (min_V is not None):
        min_vc = Vwl if Vwl < Vv else Vv

    CDELT_temp = np.log10(min_vc +1)
    CRVAL1 = np.log10(wl_start)
    CRVALN = np.log10(wl_end)
    N = (CRVALN - CRVAL1)/CDELT_temp
    NAXIS1 = 2
    while NAXIS1 < N: #Make NAXIS1 an integer power of 2 for FFT purposes
        NAXIS1 *= 2

    CDELT1 = (CRVALN - CRVAL1)/(NAXIS1 - 1)

    p = np.arange(NAXIS1)
    wl = 10 ** (CRVAL1 + CDELT1 * p)
    return {"wl":wl, "CRVAL1":CRVAL1, "CDELT1":CDELT1, "NAXIS1":NAXIS1}


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


def grid_loader():
    pass


@np.vectorize
def vacuum_to_air(wl):
    '''CA Prieto recommends this as more accurate than the IAU standard. Ciddor 1996.'''
    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    return wl / f

def calculate_n(wl):
    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    new_wl = wl / f
    n = wl/new_wl
    print(n)


@np.vectorize
def vacuum_to_air_SLOAN(wl):
    '''Takes wavelength in angstroms and maps to wl in air.
    from SLOAN website
     AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)'''
    air = wl / (1.0 + 2.735182E-4 + 131.4182 / wl ** 2 + 2.76249E8 / wl ** 4)
    return air


@np.vectorize
def air_to_vacuum(wl):
    sigma = 1e4 / wl
    vac = wl + wl * (6.4328e-5 + 2.94981e-2 / (146 - sigma ** 2) + 2.5540e-4 / (41 - sigma ** 2))
    return vac


def rewrite_wl():
    np.save("ind.npy", ind)
    np.save("wave_trim.npy", w)


def get_wl_kurucz():
    '''The Kurucz grid is already convolved with a FWHM=6.8km/s Gaussian. WL is log-linear spaced.'''
    sample_file = "Kurucz/t06000g45m05v000.fits"
    flux_file = pf.open(sample_file)
    hdr = flux_file[0].header
    num = len(flux_file[0].data)
    p = np.arange(num)
    w1 = hdr['CRVAL1']
    dw = hdr['CDELT1']
    wl = 10 ** (w1 + dw * p)
    return wl


def get_wl_BTSettl():
    pass


@np.vectorize
def idl_float(idl):
    '''Take an idl number and convert it to scientific notation.'''
    #replace 'D' with 'E', convert to float
    return np.float(idl.replace("D", "E"))


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
        fl = fl * (F_sun / F_bol)
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
    '''Load a raw PHOENIX or kurucz spectrum based upon temp, logg, and Z. Normalize to F_sun if desired.'''

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

    flux_file = pf.open(rname)
    f = flux_file[0].data

    if norm:
        f *= 1e-8 #convert from erg/cm^2/s/cm to erg/cm^2/s/A
        F_bol = trapz(f, w_full)
        f = f * (F_sun / F_bol)
        #this also means that the bolometric luminosity is always 1 L_sun
    if grid == "kurucz":
        f *= c_ang / wave_grid_kurucz_raw ** 2 #Convert from f_nu to f_lambda

    flux_file.close()
    #print("Loaded " + rname)
    return f


@np.vectorize
def gauss_taper(s, sigma=2.89):
    '''This is the FT of a gaussian w/ this sigma. Sigma in km/s'''
    return np.exp(-2 * np.pi ** 2 * sigma ** 2 * s ** 2)


def resample_and_convolve(f, wg_raw, wg_fine, wg_coarse, wg_fine_d=0.35, sigma=2.89):
    '''Take a full-resolution PHOENIX model spectrum `f`, with raw spacing wg_raw, resample it to wg_fine
    (done because the original grid is not log-linear spaced), instrumentally broaden it in the Fourier domain,
    then resample it to wg_coarse. sigma in km/s.'''

    #resample PHOENIX to 0.35km/s spaced grid using InterpolatedUnivariateSpline. First check to make sure there
    #are no duplicates and the wavelength is increasing, otherwise the spline will fail and return NaN.
    wl_sorted, ind = np.unique(wg_raw, return_index=True)
    fl_sorted = f[ind]
    interp_fine = InterpolatedUnivariateSpline(wl_sorted, fl_sorted)
    f_grid = interp_fine(wg_fine)

    #Fourier Transform
    out = fft(f_grid)
    #The frequencies (cycles/km) corresponding to each point
    freqs = fftfreq(len(f_grid), d=wg_fine_d)

    #Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space (corresponding to FWHM 6.8km/s)
    taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (freqs ** 2))
    tout = out * taper

    #Take the broadened spectrum back to wavelength space
    f_grid6 = ifft(tout)
    #print("Total of imaginary components", np.sum(np.abs(np.imag(f_grid6))))

    #Resample the broadened spectrum to a uniform coarse grid
    interp_coarse = InterpolatedUnivariateSpline(wg_fine, np.abs(f_grid6))
    f_coarse = interp_coarse(wg_coarse)

    del interp_fine
    del interp_coarse
    gc.collect() #necessary to prevent memory leak!

    return f_coarse


def resample(f, wg_input, wg_output):
    '''Take a TRES spectrum and resample it to 2km/s binning. For the kurucz grid.'''

    # check to make sure there are no duplicates and the wavelength is increasing,
    # otherwise the spline will fail and return NaN.
    wl_sorted, ind = np.unique(wg_input, return_index=True)
    fl_sorted = f[ind]

    interp = InterpolatedUnivariateSpline(wl_sorted, fl_sorted)
    f_output = interp(wg_output)
    del interp
    gc.collect()
    return f_output


def process_spectrum_PHOENIX(pars, convolve=True):
    temp, logg, Z, alpha = pars
    try:
        f = load_flux_full(temp, logg, Z, alpha, norm=True, grid="PHOENIX")[ind]
        if convolve:
            flux = resample_and_convolve(f, wave_grid_raw_PHOENIX, wave_grid_fine, wave_grid_coarse)
        else:
            flux = resample(f, wave_grid_raw_PHOENIX, wave_grid_fine)
        print("PROCESSED: %s, %s, %s %s" % (temp, logg, Z, alpha))
    except OSError:
        print("FAILED: %s, %s, %s, %s" % (temp, logg, Z, alpha))
        flux = np.nan
    return flux


def process_spectrum_kurucz(pars):
    temp, logg, Z = pars
    try:
        f = load_flux_full(temp, logg, Z, norm=False, grid="kurucz")
        flux = resample(f, wave_grid_kurucz_raw, wave_grid_2kms_kurucz)
    except OSError:
        print("%s, %s, %s does not exist!" % (temp, logg, Z))
        flux = np.nan
    return flux


def process_spectrum_BTSettl(pars, convolve=True):
    temp, logg, Z = pars
    try:
        wl, f = load_BTSettl(temp, logg, Z, norm=True, trunc=True, air=True)
        if convolve:
            flux = resample_and_convolve(f, wl, wave_grid_fine, wave_grid_coarse)
        else:
            flux = resample(f, wl, wave_grid_fine)
        print("PROCESSED: %s, %s, %s" % (temp, logg, Z))
    except FileNotFoundError: #on Python2 gives IOError, Python3 use FileNotFoundError
        print("FAILED: %s, %s, %s" % (temp, logg, Z))
        flux = np.nan
    return flux


process_routines = {"PHOENIX": process_spectrum_PHOENIX, "kurucz": process_spectrum_kurucz,
                    "BTSettl": process_spectrum_BTSettl}


def create_grid_parallel(ncores, hdf5_filename, grid_name, convolve=True):
    '''create an hdf5 file of the stellar grid. Go through each T point, if the corresponding logg exists,
    write it. If not, write nan. Each spectrum is normalized to the bolometric flux at the surface of the Sun.'''
    f = h5py.File(hdf5_filename, "w")

    #Grid parameters
    grid = grids[grid_name]
    T_points = grid['T_points']
    logg_points = grid['logg_points']
    Z_points = grid['Z_points']
    alpha_points = grid['alpha_points']

    if grid_name == 'kurucz':
        process_spectrum = process_spectrum_kurucz
        wave_grid_out = wave_grid_2kms_kurucz
    elif (grid_name == 'PHOENIX') or (grid_name == "BTSettl"):
        process_spectrum = {"PHOENIX": partial(process_spectrum_PHOENIX, convolve=convolve),
                            "BTSettl": partial(process_spectrum_BTSettl, convolve=convolve)}[grid_name]
        if convolve:
            wave_grid_out = np.load("wave_grids/PHOENIX_2kms_air.npy")
        else:
            wave_grid_out = np.load("wave_grids/PHOENIX_0.35kms_air.npy")
    else:
        print("No grid %s" % grid_name)
        return 1

    shape = (len(T_points), len(logg_points), len(Z_points), len(alpha_points), len(wave_grid_out))
    dset = f.create_dataset("LIB", shape, dtype="f", compression='gzip', compression_opts=9)

    # A thread pool of P processes
    pool = mp.Pool(ncores)

    index_combos = []
    var_combos = []
    for t, temp in enumerate(T_points):
        for l, logg in enumerate(logg_points):
            for z, Z in enumerate(Z_points):
                for a, A in enumerate(alpha_points):
                    index_combos.append([t, l, z, a])
                    var_combos.append([temp, logg, Z, A])

    spec_gen = pool.imap(process_spectrum, var_combos, chunksize=20)

    for i, spec in enumerate(spec_gen):
        t, l, z, a = index_combos[i]
        dset[t, l, z, a, :] = spec
        print("Writing ", var_combos[i], "to HDF5")

    f.close()


def interpolate_raw_test_temp():
    base = 'data/LkCa15//LkCa15_2013-10-13_09h37m31s_cb.flux.spec.'
    wls = np.load(base + "wls.npy")
    fls = np.load(base + "fls.npy")
    wl = wls[22]
    ind2 = (m.w_full > wl[0]) & (m.w_full < wl[-1])
    w = m.w_full[ind2]
    f58 = load_flux_npy(5800, 3.5)[ind2]
    f59 = load_flux_npy(5900, 3.5)[ind2]
    f60 = load_flux_npy(6000, 3.5)[ind2]

    bit = np.array([5800, 6000])
    f = np.array([f58, f60]).T
    func = interp1d(bit, f)
    f59i = func(5900)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.axhline(0, color="k")
    ax.plot(w, (f59 - f59i) * 100 / f59)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/5800_5900_6000_logg3.5.png")


def interpolate_raw_test_logg():
    base = 'data/LkCa15//LkCa15_2013-10-13_09h37m31s_cb.flux.spec.'
    wls = np.load(base + "wls.npy")
    fls = np.load(base + "fls.npy")

    wl = wls[22]
    ind2 = (m.w_full > wl[0]) & (m.w_full < wl[-1])
    w = m.w_full[ind2]
    f3 = load_flux_npy(5900, 3.0)[ind2]
    f3_5 = load_flux_npy(5900, 3.5)[ind2]
    f4 = load_flux_npy(5900, 4.0)[ind2]

    bit = np.array([3.0, 4.0])
    f = np.array([f3, f4]).T
    func = interp1d(bit, f)
    f3_5i = func(3.5)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.axhline(0, color="k")
    ax.plot(w, (f3_5 - f3_5i) * 100 / f3_5)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/5900_logg3_3.5_4.png")


def interpolate_test_temp():
    base = 'data/LkCa15//LkCa15_2013-10-13_09h37m31s_cb.flux.spec.'
    wls = np.load(base + "wls.npy")
    fls = np.load(base + "fls.npy")

    f58 = load_flux_npy(2400, 3.5)
    f59 = load_flux_npy(2500, 3.5)
    f60 = load_flux_npy(2600, 3.5)
    bit = np.array([2400, 2600])
    f = np.array([f58, f60]).T
    func = interp1d(bit, f)
    f59i = func(2500)

    d59 = m.degrade_flux(wl, m.w_full, f59)
    d59i = m.degrade_flux(wl, m.w_full, f59i)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.axhline(0, color="k")
    ax.plot(wl, (d59 - d59i) * 100 / d59)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/2400_2500_2600_logg3.5_degrade.png")


def interpolate_test_logg():
    base = 'data/LkCa15//LkCa15_2013-10-13_09h37m31s_cb.flux.spec.'
    wls = np.load(base + "wls.npy")
    fls = np.load(base + "fls.npy")

    wl = wls[22]

    f3 = load_flux_npy(2400, 3.0)
    f3_5 = load_flux_npy(2500, 3.5)
    f4 = load_flux_npy(2600, 4.0)

    bit = np.array([3.0, 4.0])
    f = np.array([f3, f4]).T
    func = interp1d(bit, f)
    f3_5i = func(3.5)

    d3_5 = m.degrade_flux(wl, m.w_full, f3_5)
    d3_5i = m.degrade_flux(wl, m.w_full, f3_5i)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.axhline(0, color="k")
    ax.plot(wl, (d3_5 - d3_5i) * 100 / d3_5)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/2500logg3_3.5_4_degrade.png")


def compare_PHOENIX_TRES_spacing():
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    wave_TRES = trunc_tres()
    #index as percentage of full grid
    #ind = (wave_grid > wave_TRES[0]) & (wave_grid < wave_TRES[-1])
    #w_grid = wave_grid[ind]
    #w_pixels = np.arange(0,len(w_grid),1)/len(w_grid)
    #ax.plot(w_grid[:-1], v(w_grid[:-1],w_grid[1:]),label="Constant V")

    #t_pixels = np.arange(0,len(wave_TRES),1)/len(wave_TRES)
    #linear = np.linspace(wave_TRES[0],wave_TRES[-1])
    #l_pixels = np.arange(0,len(linear),1)/len(linear)

    ax.plot(wave_TRES[:-1], v(wave_TRES[:-1], wave_TRES[1:]), "g", label="TRES")
    ax.axhline(2.5)
    #ax.plot(linear[:-1], v(linear[:-1],linear[1:]),label="Linear")

    ax.set_xlabel(r"$\lambda$ [\AA]")
    ax.set_ylabel(r"$\Delta v$ [km/s]")
    ax.legend(loc='best')
    ax.set_ylim(2.2, 2.8)
    #plt.show()
    fig.savefig("plots/pixel_spacing_v.png")


@np.vectorize
def v(ls, lo):
    return c_kms * (lo ** 2 - ls ** 2) / (ls ** 2 + lo ** 2)

def create_FITS_wavegrid(wl_start, wl_end, vel_spacing):
    '''Taking the desired wavelengths, output CRVAL1, CDELT1, NAXIS1 and the actual wavelength array.
    vel_spacing in km/s, wavelengths in angstroms.'''
    CRVAL1 = np.log10(wl_start)
    CDELT1 = np.log10(vel_spacing/c_kms + 1)
    NAXIS1 = int(np.ceil((np.log10(wl_end) - CRVAL1)/CDELT1)) + 1
    p = np.arange(NAXIS1)
    wl = 10 ** (CRVAL1 + CDELT1 * p)
    return [wl, CRVAL1, CDELT1, NAXIS1]


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

def process_PHOENIX_to_grid(temp, logg, Z, alpha, vsini, instFWHM, air=True):
    #Create the wave_grid
    out_grid, CRVAL1, CDELT1, NAXIS = create_FITS_wavegrid(6200, 6700, 2.)

    #Load the raw file
    flux = load_flux_full(temp, logg, Z, alpha, norm=True, grid="PHOENIX")[ind]

    global w
    if air:
        w_new = vacuum_to_air(w)

    #resample to equally spaced v grid, convolve w/ instrumental profile,
    f_coarse = resample_and_convolve(flux, w_new, wave_grid_fine, out_grid, wg_fine_d=0.35, sigma=instFWHM/2.35)

    ss = np.fft.fftfreq(len(out_grid), d=2.) #2km/s spacing for wave_grid
    ss[0] = 0.01 #junk so we don't get a divide by zero error
    ub = 2. * np.pi * vsini * ss
    sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
    #set zeroth frequency to 1 separately (DC term)
    sb[0] = 1.
    FF = fft(f_coarse)
    FF *= sb

    #do ifft
    f_lam = np.abs(ifft(FF))

    #convert to f_nu
    f_nu = out_grid**2/c_ang * f_lam

    filename = "t{temp:0>5.0f}g{logg:.0f}p00v{vsini:0>3.0f}.fits".format(temp=temp,
    logg=10 * logg, vsini=vsini)

    create_fits(filename, f_nu, CRVAL1, CDELT1, {"BUNIT": ('erg/s/cm^2/Hz', 'Unit of flux'),
                                                                "AUTHOR": "Ian Czekala",
                                                                "COMMENT" : "Adapted from PHOENIX"})



    pass

def main():
    ncores = mp.cpu_count()
    wldict = create_log_lam_grid(min_V=2.)
    rawgrid = PHOENIXGridInterface(air=True, norm=True)
    HDF5Creator = HDF5GridCreator(rawgrid, ranges={"temp":(5000, 6000), "logg":(3.5, 4.5), "Z":(0.0,0.0),
                                                         "alpha":(0.0, 0.0)}, outfile="test.hdf5", wldict=wldict)
    HDF5Creator.process_grid()
    #create_fine_and_coarse_wave_grid()
    #create_coarse_wave_grid_kurucz()

    #create_grid_parallel(ncores, "LIB_kurucz_2kms_air.hdf5", grid_name="kurucz")
    #create_grid_parallel(ncores, "LIB_PHOENIX_2kms_air.hdf5", grid_name="PHOENIX", convolve=True)
    #create_grid_parallel(ncores, "LIB_PHOENIX_0.35kms_air.hdf5", grid_name="PHOENIX", convolve=False)
    #load_flux_full(5900, 7.0, "-0.0", norm=False, vsini=0, grid="PHOENIX")

    #create_grid_parallel(ncores, "LIB_BTSettl_2kms_air.hdf5", grid_name="BTSettl", convolve=True)
    #create_grid_parallel(ncores, "LIB_BTSettl_0.35kms_air.hdf5", grid_name="BTSettl", convolve=False)
    #n = np.linspace(6200, 6700, num=11627)
    #create_fits("test.fits", n, 3.7923916895, 2.89729125382e-06, dict={"Author":"Ian Czekala"})
    #wl, CRVAL1, CDELT1, NAXIS = create_FITS_wavegrid(6200, 6700, 2.)
    #np.save("wave_grids/willie_custom_2kms.npy", wl)
    #print(wl)
    #print(CRVAL1)
    #print(CDELT1)
    #print(NAXIS)
    #out_grid = np.load("wave_grids/willie_custom_2kms.npy")
    #process_PHOENIX_to_grid(6000, 4.5, "-0.0", None, 8, 14.4)
    #process_PHOENIX_to_grid(4000, 4.5, "-0.0", None, 4, 14.4)
    #def test():
    #    HDF5 = HDF5Interface("PHOENIX", "PHOENIX.h5py", "w", np.linspace(5000, 6000), np.array([6000, 6100]), np.array([3.5, 4.0]),
    #                     np.array([-0.5, 0.0]), np.array([0.0, 0.2]))
    #test()
    #processor = TRES_PHOENIX_HDF5_Processor()


if __name__ == "__main__":
    main()
