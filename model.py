import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator, InterpolatedUnivariateSpline
from scipy.ndimage.filters import convolve
from scipy.special import j1
from numpy.polynomial import Chebyshev as Ch
from functools import partial
import h5py
import yaml
import gc
import sys
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import pyfftw
import emcee
import os
import StellarSpectra.constants as C


def load_config():
    config = None
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if ".yaml" in arg:
                confname = arg
                f = open(confname)
                config = yaml.load(f)
                f.close()
    if config is None:
        import StellarSpectra #this triggers the __init__.py code
        config = StellarSpectra.default_config

class BaseSpectrum:
    def __init__(self, wl, fl, fl_type="flam", air=True, vel=0.0, metadata=None):
        #TODO: convert fl_type to use astropy units for later conversions
        assert wl.shape == fl.shape, "Spectrum wavelength and flux arrays must have the same shape."
        self.wl_raw = wl
        self.fl = fl
        self.fl_type = fl_type
        self.air = air
        self.velocity = vel #creates self.wl_vel
        self.metadata = {} if metadata is None else metadata

    def convert_units(self):
        raise NotImplementedError

    #Set air as a property which will update self.c it uses to calculate velocities
    @property
    def air(self):
        return self._air

    @air.setter
    def air(self, air):
        #TODO: rewrite this to be more specific about which c
        assert type(air) == type(True)
        self._air = air
        if self.air:
            self.c = C.c_kms_air
        else:
            self.c = C.c_kms

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, vz):
        '''Shift the wl_vel relative to wl_raw. Keeps track if in air. Positive vz is redshift.'''
        self.wl_vel = self.wl_raw * np.sqrt((self.c + vz) / (self.c - vz))
        self._velocity = vz

    def add_metadata(self, keyVal):
        key, val = keyVal
        if key in self.metadata.keys():
            self.metadata[key]+= val
        else:
            self.metadata[key] = val

    def save(self,name):
        obj = np.array((self.wl_vel, self.fl))
        np.save(name, obj)


    def __str__(self):
        return "Spectrum object."


class DataSpectrum(BaseSpectrum):
    def __init__(self, wl, fl, sigma, mask=None, fl_type="flam"):
        super().__init__(wl, fl, fl_type)
        self.sigma = sigma

        if mask is None:
            self.mask = np.ones_like(self.wl, dtype='bool') #create mask of all True
        else:
            self.mask = mask

        assert self.sigma.shape == self.shape, "sigma array incompatible shape."
        assert self.mask.shape == self.shape, "mask array incompatible shape."





class Base1DSpectrum(BaseSpectrum):
    def __init__(self, wl, fl, fl_type="flam", air=True, metadata=None):
        assert len(wl.shape) == 1, "1D spectrum must be 1D"
        #"Clean" the wl and flux points. Remove duplicates, sort in increasing wl
        wl_sorted, ind = np.unique(wl, return_index=True)
        fl_sorted = fl[ind]
        super().__init__(wl_sorted, fl_sorted, fl_type=fl_type, air=air, metadata=metadata)

    def calculate_log_lam_grid(self):
        dif = np.diff(self.wl_vel)
        min_wl = np.min(dif)
        wl_at_min = self.wl_vel[np.argmin(dif)]
        wl_dict = create_log_lam_grid(wl_start=self.wl_vel[0], wl_end=self.wl_vel[-1], min_wl=(min_wl, wl_at_min))
        #(log_lam_grid, CRVAL1, CDELT1, NAXIS1)
        return wl_dict

    def resample_to_grid(self, grid):
        '''Resamples to a new grid. For other methods, Grid does not necessarily have to be 1D.'''
        #TODO: how to properly set the velocity of the new grid when resampling?
        assert len(grid.shape) == 1, "grid must be 1D"
        interp = InterpolatedUnivariateSpline(self.wl_vel, self.fl)
        self.fl = interp(grid)
        del interp
        gc.collect()
        self.wl_vel = grid

def create_log_lam_grid(wl_start=3000., wl_end=13000., min_wl=None, min_vc=None):
    '''min_WL = (delta_WL, WL). Takes the finer of the two specified.'''
    if (min_wl is None) and (min_vc is None):
        raise ValueError("You need to specify either min_wl or min_vc")
    if min_wl is not None:
        delta_wl, wl = min_wl #unpack
        Vwl = delta_wl/wl
        min_vc = Vwl
    if (min_wl is not None) and (min_vc is not None):
        min_vc = Vwl if Vwl < min_vc else min_vc

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


class LogLambdaSpectrum(Base1DSpectrum):
    def __init__(self, wl, fl, fl_type="flam", air=True, metadata=None, oversampling=3.5):
        super().__init__(wl, fl, fl_type, air=air, metadata=metadata)
        #Super class already checks that the wavelengths are np.unique
        #Need to check that the vc spacing of each pixel is the same.
        vcs = np.diff(wl)/wl[:-1] * C.c_kms_air
        print(vcs)
        assert np.allclose(vcs, vcs[0]), "Array must be log-lambda spaced."

        #Check to see if CDELT1 is defined, verify to make sure that these are the same.



        self.min_vc = 10**self.metadata["CDELT1"] - 1
        self.oversampling = oversampling #taken to mean as how many points go across the FWHM of the Gaussian

    def downsample(self):
        #Takes the new min_vc and oversampling factor
        min_vc = self.min_vc/self.oversampling
        wldict = create_log_lam_grid(self.wl[0], self.wl[-1], min_vc=min_vc)

        #creates new wl grid and updates header values
        wl = wldict.pop("wl")
        self.metadata.update(wldict)

        #resamples the spectrum to these values and updates wl_grid
        self.resample_to_grid(wl)

    def instrument_convolve(self, instrument, downsample=True):
        sigma = instrument.FWHM/2.35 # in km/s

        FF = fft(self.fl)
        #The frequencies (cycles/km) corresponding to each point
        ss = fftfreq(len(self.fl), d=self.min_vc * C.c_kms_air)

        #Instrumentally broaden the spectrum by multiplying with a Gaussian in Fourier space
        taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
        FF *= taper

        #Take the broadened spectrum back to wavelength space
        self.fl = ifft(FF)

        #Update min_vc and oversampling, possibly downsample
        if instrument.FWHM > self.min_vc:
            self.min_vc = instrument.FWHM
            self.oversampling = instrument.oversampling

            if downsample:
                #downsample the broadened spectrum to a coarser grid
                self.downsample()

        #TODO: this logic needs to be firmed up


    def stellar_convolve(self, vsini, downsample=True):
        #Take FFT of f_grid
        FF = fft(self.fl)

        ss = fftfreq(len(self.wl), d=self.min_vc * C.c_kms_air)
        ss[0] = 0.01 #junk so we don't get a divide by zero error
        ub = 2. * np.pi * vsini * ss
        sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
        #set zeroth frequency to 1 separately (DC term)
        sb[0] = 1.

        #institute velocity taper
        FF *= sb

        #do ifft
        self.fl = np.abs(ifft(FF))

        #Update min_vc and oversampling, possibly downsample
        if vsini > self.min_vc:
            self.min_vc = vsini

            if downsample:
                #downsample the broadened spectrum to a coarser grid
                self.downsample()


    def instrument_and_stellar_convolve(self, instrument, vsini, downsample=True):
        '''Does both instrument and stellar convolution in one step, in the Fourier domain.'''
        ss = fftfreq(len(self.wl), d=self.min_vc * C.c_kms_air)
        ss[0] = 0.01 #junk so we don't get a divide by zero error
        ub = 2. * np.pi * vsini * ss
        sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
        #set zeroth frequency to 1 separately (DC term)
        sb[0] = 1.

        sigma = instrument.FWHM/2.35 # in km/s
        taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))

        FF = fft(self.fl)
        FF *= (taper * sb)

        #Take the broadened spectrum back to wavelength space
        self.fl = ifft(FF)

        #Update min_vc and oversampling, possibly downsample
        if instrument.FWHM > self.min_vc:
            self.min_vc = instrument.FWHM
            self.oversampling = instrument.oversampling

            if downsample:
                #downsample the broadened spectrum to a coarser grid
                self.downsample()


class ModelSpectrum(LogLambdaSpectrum):
    '''Specifically designed to match the shape of the data.'''

    wl = None #This wl is a "static" variable. Or, does it make sense because there will only be one model per
    #process, to just keep updating the flux and wl? Rather than forcing one.

    def __init__(self, wl, fl, dataSpectrum, fl_type="flam"):
        super().__init__(wl, fl, fl_type)
        self.dataSpectrum = dataSpectrum

#grids = {"PHOENIX": {'T_points': np.array(
#    [2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 4000, 4100, 4200,
#     4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100,
#     6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000,
#     9200, 9400, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 11200, 11400, 11600, 11800, 12000]),
#                     'logg_points': np.arange(0.0, 6.1, 0.5), 'Z_points': np.array([-1., -0.5, 0.0, 0.5, 1.0])},
#                     #'alpha_points': np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])},
#         "kurucz": {'T_points': np.arange(3500, 9751, 250),
#                    'logg_points': np.arange(1.0, 5.1, 0.5), 'Z_points': np.array([-0.5, 0.0, 0.5])},
#         "BTSettl": {'T_points': np.arange(3000, 7001, 100), 'logg_points': np.arange(2.5, 5.6, 0.5),
#                     'Z_points': np.array([-0.5, 0.0, 0.5])}}
#
#base = os.path.dirname(__file__) + "/"
#
#
##if config['grid'] == 'PHOENIX':
##    wave_grid = np.load(base + "wave_grids/PHOENIX_2kms_air.npy")
##    LIB_filename = "libraries/LIB_PHOENIX_2kms_air.hdf5"
##elif config['grid'] == "kurucz":
##    wave_grid = np.load(base + "wave_grids/kurucz_2kms_air.npy")
##    LIB_filename = "libraries/LIB_kurucz_2kms_air.hdf5"
##elif config['grid'] == 'BTSettl':
##    wave_grid = np.load(base + "wave_grids/PHOENIX_2kms_air.npy")
##    LIB_filename = "libraries/LIB_BTSettl_2kms_air.hdf5"
#
#grid = grids[config['grid']]
#
#T_points = grid['T_points']
#logg_points = grid['logg_points']
#Z_points = grid['Z_points']
##alpha_points = grid['alpha_points']
#
##Limit grid size to relevant region
#grid_params = config['grid_params']
#
#T_low, T_high = grid_params['temp_range']
#T_ind = (T_points >= T_low) & (T_points <= T_high)
#T_points = T_points[T_ind]
#T_arg = np.where(T_ind)[0]
#
#g_low, g_high = grid_params['logg_range']
#logg_ind = (logg_points >= g_low) & (logg_points <= g_high)
#logg_points = logg_points[logg_ind]
#logg_arg = np.where(logg_ind)[0]
#
#Z_low, Z_high = grid_params['Z_range']
#Z_ind = (Z_points >= Z_low) & (Z_points <= Z_high)
#Z_points = Z_points[Z_ind]
#Z_arg = np.where(Z_ind)[0]
#
##A_low, A_high = grid_params['alpha_range']
##A_ind = (alpha_points >= A_low) & (alpha_points <= A_high)
##A_points = alpha_points[A_ind]
##A_arg = np.where(A_ind)[0]
#
##Will want to use pgutil.get_data http://docs.python.org/2/library/pkgutil.html
##http://stackoverflow.com/questions/10935127/way-to-access-resource-files-in-python
#
##Load the data to fit
##database = base + 'data/' + config['dataset']
##wls = np.load(database + ".wls.npy")
##fls = np.load(database + ".fls.npy")
##sigmas = np.load(database + ".sigma.npy")
##masks = np.load(database + ".mask.npy")
#
##orders = np.array(config['orders'])
##norder = len(orders)
#
##Truncate the data to include only those orders you wish to fit
##wls = wls[orders]
##fls = fls[orders]
##sigmas = sigmas[orders]
##masks = masks[orders]
#
##sigma for Gaussian priors on nuisance coefficients
##sigmac = config['sigmac']
##sigmac0 = config['sigmac0']
##
##wr = config['walker_ranges']
##
##len_wl = len(wls[0])
##
##wl_buffer = 5.0 #Angstroms on either side, to account for velocity shifts
##wl_min = wls[0, 0] - wl_buffer
##wl_max = wls[-1, -1] + wl_buffer
#
#
## Truncate wave_grid and red_grid to include only the regions necessary for fitting the specified orders.
## but do it in such a way that it is a power of 2 to speed up the FFT
#
##len_wg = len(wave_grid)
##
##len_data = np.sum((wave_grid > wl_min) & (wave_grid < wl_max))
##
##if len_data < (len_wg / 16):
##    chunk = int(len_wg / 16)
##elif len_data < (len_wg / 8):
##    chunk = int(len_wg / 8)
##elif len_data < (len_wg / 4):
##    chunk = int(len_wg / 4)
##elif len_data < (len_wg / 2):
##    chunk = int(len_wg / 2)
##else:
##    use the  full spectrum
#    #chunk = len_wg
#    #ind = np.ones_like(wave_grid, dtype='bool')
##
##if chunk < len_wg:
##    ind_wg = np.arange(len_wg)
##    Determine if the data region is closer to the start or end of the wave_grid
#    #if (wl_min - wave_grid[0]) < (wave_grid[-1] - wl_max):
#    #    the data region is closer to the start
#    #    find starting index
#    #    start at index corresponding to wl_min and go chunk forward
#        #start_ind = np.argwhere(wave_grid > wl_min)[0][0]
#        #end_ind = start_ind + chunk
#        #ind = (ind_wg >= start_ind) & (ind_wg < end_ind)
#    #
#    #else:
#    #    the data region is closer to the finish
#        # start at index corresponding to wl_max and go chunk backward
#        #end_ind = np.argwhere(wave_grid < wl_max)[-1][0]
#        #start_ind = end_ind - chunk
#        #ind = (ind_wg > start_ind) & (ind_wg <= end_ind)
#
##wave_grid = wave_grid[ind]
##red_grid = np.load(base + 'red_grid.npy')[ind]
#
#
#def load_hdf5_spectrum(temp, logg, Z, grid_name, LIB_filename):
#    '''Load a spectrum (nearest in grid point) from the specified HDF5 library and return it without interpolation.
#    User should check that the loading message is the same as the one they specified.'''
#    grid = grids[grid_name]
#    T_points = grid['T_points']
#    lenT = len(T_points)
#
#    logg_points = grid['logg_points']
#    lenG = len(logg_points)
#
#    Z_points = grid['Z_points']
#    lenZ = len(Z_points)
#
#    #Create index interpolators
#    T_intp = interp1d(T_points, np.arange(lenT), kind='nearest')
#    logg_intp = interp1d(logg_points, np.arange(lenG), kind='nearest')
#    Z_intp = interp1d(Z_points, np.arange(lenZ), kind='nearest')
#
#    fhdf5 = h5py.File(LIB_filename, 'r')
#    LIB = fhdf5['LIB']
#
#    T = int(T_intp(temp))
#    G = int(logg_intp(logg))
#    Z = int(Z_intp(Z))
#    print("Loading", T_points[T], logg_points[G], Z_points[Z], grid_name)
#    f = LIB[T, G, Z]
#    return f
#
#def quadlinear_interpolator():
#    '''Return a function that will take temp, logg, Z as arguments and do trilinear interpolation on it.'''
#    fhdf5 = h5py.File(LIB_filename, 'r')
#    LIB = fhdf5['LIB']
#
#    #Load only those indexes we want into a grid in memory
#    grid = LIB[T_arg[0]:T_arg[-1] + 1, logg_arg[0]:logg_arg[-1] + 1, Z_arg[0]:Z_arg[-1] + 1, A_arg[0]:A_arg[-1] + 1, ind] #weird syntax because
#    #sequence indexing is not supported for more than one axis in h5py
#    lenT, lenG, lenZ, lenA, lenF = grid.shape
#
#    #Create index interpolators
#    T_intp = interp1d(T_points, np.arange(lenT), kind='linear')
#    logg_intp = interp1d(logg_points, np.arange(lenG), kind='linear')
#    Z_intp = interp1d(Z_points, np.arange(lenZ), kind='linear')
#    A_intp = interp1d(alpha_points, np.arange(lenA), kind='linear')
#
#    fluxes = np.empty((16, lenF))
#    zeros = np.zeros(lenF)
#
#    def intp_func(temp, logg, Z, alpha):
#        if (logg < g_low) or (logg > g_high) or (temp < T_low) or (temp > T_high) or (Z < Z_low) or (Z > Z_high)\
#            or (alpha < A_low) or (alpha > A_high):
#            return zeros
#        else:
#            '''Following trilinear interpolation scheme from http://paulbourke.net/miscellaneous/interpolation/'''
#            indexes = np.array((T_intp(temp), logg_intp(logg), Z_intp(Z), A_intp(alpha)))
#            ui = np.ceil(indexes) #upper cube vertices
#            li = np.floor(indexes) #lower cube vertices
#            #print(li,ui)
#            w, x, y, z = (indexes - li) #range between 0 - 1
#            wu, xu, yu, zu = ui
#            wl, xl, yl, zl = li
#            fluxes[:] = np.array([
#                grid[wl, xl, yl, zl],
#                grid[wu, xl, yl, zl],
#                grid[wl, xu, yl, zl],
#                grid[wl, xl, yu, zl],
#                grid[wu, xl, yu, zl],
#                grid[wl, xu, yu, zl],
#                grid[wu, xu, yl, zl],
#                grid[wu, xu, yu, zl],
#                grid[wl, xl, yl, zu],
#                grid[wu, xl, yl, zu],
#                grid[wl, xu, yl, zu],
#                grid[wl, xl, yu, zu],
#                grid[wu, xl, yu, zu],
#                grid[wl, xu, yu, zu],
#                grid[wu, xu, yl, zu],
#                grid[wu, xu, yu, zu],
#                ])
#
#            weights = np.array([
#                (1 - w) * (1 - x) * (1 - y) * (1 - z),
#                w * (1 - x) * (1 - y) * (1 - z),
#                (1 - w) * x * (1 - y) * (1 - z),
#                (1 - w) * (1 - x) * y * (1 - z),
#                w * (1 - x) * y * (1 - z),
#                (1 - w) * x * y * (1 - z),
#                w * x * (1 - y) * (1 - z),
#                w * x * y * (1 - z),
#                (1 - w) * (1 - x) * (1 - y) * z,
#                w * (1 - x) * (1 - y) * z,
#                (1 - w) * x * (1 - y) * z,
#                (1 - w) * (1 - x) * y * z,
#                w * (1 - x) * y * z,
#                (1 - w) * x * y * z,
#                w * x * (1 - y) * z,
#                w * x * y * z])
#
#            #print(weights)
#            #print(np.sum(weights))
#
#            return np.average(fluxes, axis=0, weights=weights)
#
#    return intp_func
#
#def trilinear_interpolator():
#    '''Return a function that will take temp, logg, Z as arguments and do trilinear interpolation on it.'''
#    fhdf5 = h5py.File(LIB_filename, 'r')
#    LIB = fhdf5['LIB']
#
#    #Load only those indexes we want into a grid in memory
#    grid = LIB[T_arg[0]:T_arg[-1] + 1, logg_arg[0]:logg_arg[-1] + 1, Z_arg[0]:Z_arg[-1] + 1, ind] #weird syntax because
#    #sequence indexing is not supported for more than one axis in h5py
#    lenT, lenG, lenZ, lenF = grid.shape
#
#    #Create index interpolators
#    T_intp = interp1d(T_points, np.arange(lenT), kind='linear')
#    logg_intp = interp1d(logg_points, np.arange(lenG), kind='linear')
#    Z_intp = interp1d(Z_points, np.arange(lenZ), kind='linear')
#
#    fluxes = np.empty((8, lenF))
#    zeros = np.zeros(lenF)
#
#    def intp_func(temp, logg, Z):
#        if (logg < g_low) or (logg > g_high) or (temp < T_low) or (temp > T_high) or (Z < Z_low) or (Z > Z_high):
#            return zeros
#        else:
#            '''Following trilinear interpolation scheme from http://paulbourke.net/miscellaneous/interpolation/'''
#            indexes = np.array((T_intp(temp), logg_intp(logg), Z_intp(Z)))
#            ui = np.ceil(indexes) #upper cube vertices
#            li = np.floor(indexes) #lower cube vertices
#            #print(li,ui)
#            x, y, z = (indexes - li) #range between 0 - 1
#            xu, yu, zu = ui
#            xl, yl, zl = li
#            fluxes[:] = np.array([
#                grid[xl, yl, zl],
#                grid[xu, yl, zl],
#                grid[xl, yu, zl],
#                grid[xl, yl, zu],
#                grid[xu, yl, zu],
#                grid[xl, yu, zu],
#                grid[xu, yu, zl],
#                grid[xu, yu, zu]])
#
#            weights = np.array([
#                (1 - x) * (1 - y) * (1 - z),
#                x * (1 - y) * (1 - z),
#                (1 - x) * y * (1 - z),
#                (1 - x) * (1 - y) * z,
#                x * (1 - y) * z,
#                (1 - x) * y * z,
#                x * y * (1 - z),
#                x * y * z])
#
#            #print(weights)
#            #print(np.sum(weights))
#
#            return np.average(fluxes, axis=0, weights=weights)
#
#    return intp_func
#
#
##flux = trilinear_interpolator()
#
###################################################
##Stellar Broadening
###################################################
#
#def karray(center, width, res):
#    '''Creates a kernel array with an odd number of elements, the central element centered at `center` and spanning
#    out to +/- width in steps of resolution. Works similar to arange in that it may or may not get all the way to the
#    edge.'''
#    neg = np.arange(center - res, center - width, -res)[::-1]
#    pos = np.arange(center, center + width, res)
#    kar = np.concatenate([neg, pos])
#    return kar
#
#
#@np.vectorize
#def vsini_ang(lam0, vsini, dlam=0.01, epsilon=0.6):
#    '''vsini in km/s. Epsilon is the limb-darkening coefficient, typically 0.6. Formulation uses Eqn 18.14 from Gray,
#    The Observation and Analysis of Stellar Photospheres, 3rd Edition.'''
#    lamL = vsini * 1e13 * lam0 / C.c_ang
#    lam = karray(0, lamL, dlam)
#    c1 = 2. * (1 - epsilon) / (np.pi * lamL * (1 - epsilon / 3.))
#    c2 = epsilon / (2. * lamL * (1 - epsilon / 3.))
#    series = c1 * np.sqrt(1. - (lam / lamL) ** 2) + c2 * (1. - (lam / lamL) ** 2) ** 2
#    return series / np.sum(series)
#
#
#@np.vectorize
#def G(s, vL):
#    '''vL in km/s. Gray pg 475'''
#    if s != 0:
#        ub = 2. * np.pi * vL * s
#        return j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
#    else:
#        return 1.
#
###################################################
##Radial Velocity Shift
###################################################
#@np.vectorize
#def shift_vz(lam_source, vz):
#    '''Given the source wavelength, lam_sounce, return the observed wavelength based upon a radial velocity vz in
#    km/s. Negative velocities are towards the observer (blueshift).'''
#    lam_observe = lam_source * np.sqrt((C.c_kms + vz) / (C.c_kms - vz))
#    #TODO: when applied to full spectrum, this sqrt is repeated
#    return lam_observe
#
#
##def shift_TRES(vz, wls=wls):
##    wlsz = shift_vz(wls, vz)
##    return wlsz
#
###################################################
##TRES Instrument Broadening
###################################################
#@np.vectorize
#def gauss_kernel(dlam, lam0, V=6.8):
#    '''V is the FWHM in km/s. lam0 is the central wavelength in A'''
#    sigma = V / 2.355 * 1e13 #A/s
#    return np.exp(- (C.c_ang * dlam / lam0) ** 2 / (2. * sigma ** 2))
#
#
#def gauss_series(dlam, lam0, V=6.8):
#    '''sampled from +/- 3sigma at dlam. V is the FWHM in km/s'''
#    sigma_l = V / (2.355 * C.c_kms) * lam0 # sigma in AA (lambda)
#    wl = karray(0., 6 * sigma_l, dlam) # Gaussian kernel stretching +/- 6 sigma in lambda (AA)
#    gk = gauss_kernel(wl, lam0, V)
#    return gk / np.sum(gk)
#
###################################################
##Downsample to TRES bins
###################################################
#
#ones = np.ones((10,))
#
#
#def downsample(w_m, f_m, w_TRES):
#    out_flux = np.zeros_like(w_TRES)
#    len_mod = len(w_m)
#
#    #Determine the TRES bin edges
#    len_TRES = len(w_TRES)
#    edges = np.empty((len_TRES + 1,))
#    difs = np.diff(w_TRES) / 2.
#    edges[1:-1] = w_TRES[:-1] + difs
#    edges[0] = w_TRES[0] - difs[0]
#    edges[-1] = w_TRES[-1] + difs[-1]
#
#    #Determine PHOENIX bin edges
#    Pedges = np.empty((len_mod + 1,))
#    Pdifs = np.diff(w_m) / 2.
#    Pedges[1:-1] = w_m[:-1] + Pdifs
#    Pedges[0] = w_m[0] - Pdifs[0]
#    Pedges[-1] = w_m[-1] + Pdifs[-1]
#
#    i_start = np.argwhere((edges[0] < Pedges))[0][
#                  0] - 1 #return the first starting index for the model wavelength edges array (Pedges)
#
#    edges_i = 1
#    left_weight = (Pedges[i_start + 1] - edges[0]) / (Pedges[i_start + 1] - Pedges[i_start])
#
#    for i in range(len_mod + 1):
#
#        if Pedges[i] > edges[edges_i]:
#            right_weight = (edges[edges_i] - Pedges[i - 1]) / (Pedges[i] - Pedges[i - 1])
#            weights = ones[:(i - i_start)].copy()
#            weights[0] = left_weight
#            weights[-1] = right_weight
#
#            out_flux[edges_i - 1] = np.average(f_m[i_start:i], weights=weights)
#
#            edges_i += 1
#            i_start = i - 1
#            left_weight = 1. - right_weight
#            if edges_i > len_TRES:
#                break
#    return out_flux
#
###################################################
## Models
###################################################
#
##def old_model(wlsz, temp, logg, vsini, flux_factor):
##    '''Does the vsini and TRES broadening using convolution rather than Fourier tricks
##    Given parameters, return the model, exactly sliced to match the format of the echelle spectra in `efile`.
##    `temp` is effective temperature of photosphere. vsini in km/s. vz is radial velocity, negative values imply
##    blueshift. Assumes M, R are in solar units, and that d is in parsecs'''
##    #wlsz has length norders
##
##    #M = M * M_sun #g
##    #R = R * R_sun #cm
##    #d = d * pc #cm
##
##    #logg = np.log10(G * M / R**2)
##    #flux_factor = R**2/d**2 #prefactor by which to multiply model flux (at surface of star) to get recieved TRES flux
##
##    #Loads the ENTIRE spectrum, not limited to a specific order
##    f_full = flux_factor * flux(temp, logg)
##
##    model_flux = np.zeros_like(wlsz)
##    #Cycle through all the orders in the echelle spectrum
##    #might be able to np.vectorize this
##    for i, wlz in enumerate(wlsz):
##        #print("Processing order %s" % (orders[i]+1,))
##
##        #Limit huge file to the necessary order. Even at 4000 ang, 1 angstrom corresponds to 75 km/s. Add in an extra
##        # 5 angstroms to be sure.
##        ind = (w_full > (wlz[0] - 5.)) & (w_full < (wlz[-1] + 5.))
##        w = w_full[ind]
##        f = f_full[ind]
##
##        #convolve with stellar broadening (sb)
##        k = vsini_ang(np.mean(wlz), vsini) # stellar rotation kernel centered at order
##        f_sb = convolve(f, k)
##
##        dlam = w[1] - w[0] # spacing of model points for TRES resolution kernel
##
##        #convolve with filter to resolution of TRES
##        filt = gauss_series(dlam, lam0=np.mean(wlz))
##        f_TRES = convolve(f_sb, filt)
##
##        #downsample to TRES bins
##        dsamp = downsample(w, f_TRES, wlz)
##        #red = dsamp/deredden(wlz,Av,mags=False)
##
##        #If the redenning interpolation is taking a while here, we could save the points for a given redenning and
##        # simply multiply each again
##
##        model_flux[i] = dsamp
##
##    #Only returns the fluxes, because the wlz is actually the TRES wavelength vector
##    return model_flux
#
##Constant for all models
##ss = np.fft.fftfreq(len(wave_grid), d=2.) #2km/s spacing for wave_grid
##
##f_full = pyfftw.n_byte_align_empty(chunk, 16, 'complex128')
##FF = pyfftw.n_byte_align_empty(chunk, 16, 'complex128')
##blended = pyfftw.n_byte_align_empty(chunk, 16, 'complex128')
##blended_real = pyfftw.n_byte_align_empty(chunk, 16, "float64")
##fft_object = pyfftw.FFTW(f_full, FF)
##ifft_object = pyfftw.FFTW(FF, blended, direction='FFTW_BACKWARD')
#
#def model(wlsz, temp, logg, Z, vsini, Av, flux_factor):
#    '''Given parameters, return the model, exactly sliced to match the format of the echelle spectra in `efile`.
#    `temp` is effective temperature of photosphere. vsini in km/s. vz is radial velocity, negative values imply
#    blueshift. Assumes M, R are in solar units, and that d is in parsecs'''
#    #wlsz has length norders
#
#    #M = M * M_sun #g
#    #R = R * R_sun #cm
#    #d = d * pc #cm
#
#    #logg = np.log10(G * M / R**2)
#    #flux_factor = R**2/d**2 #prefactor by which to multiply model flux (at surface of star) to get recieved TRES flux
#
#    #Loads the ENTIRE spectrum, not limited to a specific order
#    f_full[:] = flux_factor * flux(temp, logg, Z)
#    #f_full = flux_factor * flux(temp, logg, Z)
#
#
#    #Take FFT of f_grid
#    #FF = fft(f_full)
#    fft_object()
#
#    ss[0] = 0.01 #junk so we don't get a divide by zero error
#    ub = 2. * np.pi * vsini * ss
#    sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
#    #set zeroth frequency to 1 separately (DC term)
#    sb[0] = 1.
#
#    FF[:] *= sb #institute velocity taper
#    #FF *= sb
#
#    #do ifft
#    ifft_object()
#    #blended_real = np.abs(ifft(FF))
#
#    blended_real[:] = np.abs(blended) #remove tiny complex component
#
#    #redden spectrum
#    red = blended_real / 10 ** (0.4 * Av * red_grid)
#    #red = blended_real
#
#    #do synthetic photometry to compare to points
#
#    f = InterpolatedUnivariateSpline(wave_grid, red)
#    fresult = f(wlsz.flatten()) #do spline interpolation to TRES pixels
#    result = np.reshape(fresult, (norder, -1))
#    del f
#    gc.collect() #necessary to prevent memory leak!
#    return result
#
#def model_alpha(wlsz, temp, logg, Z, alpha, vsini, Av, flux_factor):
#    '''Given parameters, return the model, exactly sliced to match the format of the echelle spectra in `efile`.
#    `temp` is effective temperature of photosphere. vsini in km/s. vz is radial velocity, negative values imply
#    blueshift. Assumes M, R are in solar units, and that d is in parsecs'''
#    #wlsz has length norders
#
#    #M = M * M_sun #g
#    #R = R * R_sun #cm
#    #d = d * pc #cm
#
#    #logg = np.log10(G * M / R**2)
#    #flux_factor = R**2/d**2 #prefactor by which to multiply model flux (at surface of star) to get recieved TRES flux
#
#    #Loads the ENTIRE spectrum, not limited to a specific order
#    f_full[:] = flux_factor * flux(temp, logg, Z, alpha)
#    #f_full = flux_factor * flux(temp, logg, Z)
#
#
#    #Take FFT of f_grid
#    #FF = fft(f_full)
#    fft_object()
#
#    ss[0] = 0.01 #junk so we don't get a divide by zero error
#    ub = 2. * np.pi * vsini * ss
#    sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
#    #set zeroth frequency to 1 separately (DC term)
#    sb[0] = 1.
#
#    FF[:] *= sb #institute velocity taper
#    #FF *= sb
#
#    #do ifft
#    ifft_object()
#    #blended_real = np.abs(ifft(FF))
#
#    blended_real[:] = np.abs(blended) #remove tiny complex component
#
#    #redden spectrum
#    red = blended_real / 10 ** (0.4 * Av * red_grid)
#    #red = blended_real
#
#    #do synthetic photometry to compare to points
#
#    f = InterpolatedUnivariateSpline(wave_grid, red)
#    fresult = f(wlsz.flatten()) #do spline interpolation to TRES pixels
#    result = np.reshape(fresult, (norder, -1))
#    del f
#    gc.collect() #necessary to prevent memory leak!
#    return result
#
#def model_partI():
#    '''Take care of temp, logg, Z, vsini'''
#    pass
#
#def model_partII():
#    '''Take care of vz, Av, flux_factor'''
#    pass
#
#def draw_cheb_vectors(p):
#    '''This function is only worthwhile in the lnprob_XXX_marg cases, and is used to generate samples of the nuisance
#    parameters that have already been marginalized over analytically. This means we wish to draw samples from the
#    un-marginalized probability function. Doing that analytically is tough, but there is no reason emcee can't help us
#     out. Intake a set of stellar parameters, run emcee to determine the nuisance parameters. Returns the many
#     samples from the posterior.'''
#
#    if (config['lnprob'] == "lnprob_lognormal") or (config['lnprob'] == "lnprob_gaussian"):
#        print("Mini chain not designed to work on un-marginalized function")
#        return 0
#    if (config['lnprob'] == 'lnprob_gaussian_marg') or (config['lnprob'] == 'lnprob_lognormal_marg'):
#        #Appropriately translate the parameter chain into something that can be split up for each order.
#        if (config['lnprob'] == 'lnprob_gaussian_marg'):
#            print("Not implemented yet")
#            return 0
#        if (config['lnprob'] == 'lnprob_lognormal_marg'):
#            lnprob_mini = lnprob_lognormal_nuis_func(p)
#
#            #sample cns with emcee
#            ndim = (config['ncoeff'] - 1) * norder
#            nwalkers = 4 * ndim
#            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_mini)
#            p0 = np.random.uniform(low=wr['cs'][0], high=wr['cs'][1], size=(nwalkers, ndim))
#            pos, prob, state = sampler.run_mcmc(p0, 1000)
#            sampler.reset()
#            print("Burned in cheb mini-chain")
#            sampler.run_mcmc(pos, 500, rstate0=state)
#            flatchain = sampler.flatchain
#            return flatchain
#
#
#def model_p(p):
#    '''Post processing routine that can take all parameter values and return the model.
#    Actual sampling does not require the use of this method since it is slow. Returns flatchain.'''
#    temp, logg, Z, vsini, vz, Av, flux_factor = p[:config['nparams']]
#
#    wlsz = wls * np.sqrt((C.c_kms - vz) / (C.c_kms + vz))
#    fmods = model(wlsz, temp, logg, Z, vsini, Av, flux_factor)
#
#    coefs = p[config['nparams']:]
#
#    if (config['lnprob'] == "lnprob_lognormal") or (config['lnprob'] == "lnprob_gaussian") \
#        or (config['lnprob'] == 'lnprob_mixed'):
#        # reshape to (norders, 4)
#        coefs_arr = coefs.reshape(len(orders), -1)
#        c0s = coefs_arr[:, 0] #length norders
#        cns = coefs_arr[:, 1:] #shape (norders, 3)
#
#        Tc = np.einsum("jk,ij->ik", T, cns)
#        #print("Tc.shape",Tc.shape)
#        k = np.einsum("i,ij->ij", c0s, 1 + Tc)
#        #print("k.shape",k.shape)
#        #print("fmods.shape",fmods.shape)
#        refluxed = k * fmods
#        return [wlsz, refluxed, k, None]
#
#    if config['lnprob'] == 'lnprob_lognormal_marg':
#        c0s = p[config['nparams']:]
#
#        #get flatchain
#        flatchain = draw_cheb_vectors(p)
#
#        #get random k vector
#        ind = np.random.choice(np.arange(len(flatchain)))
#        cns = flatchain[ind]
#        cns.shape = ((norder, -1))
#
#        Tc = np.einsum("jk,ij->ik", T, cns)
#        k = np.einsum("i,ij->ij", c0s, 1 + Tc)
#        refluxed = k * fmods
#
#        return [wlsz, refluxed, k, flatchain]
#
#
##xs = np.arange(len_wl)
##T0 = np.ones_like(xs)
##Ch1 = Ch([0, 1], domain=[0, len_wl - 1])
##T1 = Ch1(xs)
##Ch2 = Ch([0, 0, 1], domain=[0, len_wl - 1])
##T2 = Ch2(xs)
##Ch3 = Ch([0, 0, 0, 1], domain=[0, len_wl - 1])
##T3 = Ch3(xs)
##
##if (config['lnprob'] == "lnprob_gaussian") or (config['lnprob'] == 'lnprob_gaussian_marg'):
##    T = np.array([T0, T1, T2, T3])
##    TT = np.einsum("in,jn->ijn", T, T)
##    mu = np.array([1, 0, 0, 0])
##    D = sigmac ** (-2) * np.eye(4)
##    Dmu = np.einsum("ij,j->j", D, mu)
##    muDmu = np.einsum("j,j->", mu, Dmu)
##
##if (config['lnprob'] == "lnprob_lognormal") or (config['lnprob'] == 'lnprob_lognormal_marg') \
##    or (config['lnprob'] == 'lnprob_mixed'):
##    T = np.array([T1, T2, T3])
##    TT = np.einsum("in,jn->ijn", T, T)
##    mu = np.array([0, 0, 0])
##    D = sigmac ** (-2) * np.eye(3)
##    Dmu = np.einsum("ij,j->j", D, mu)
##    muDmu = np.einsum("j,j->", mu, Dmu)
#
#############################################################
## Various lnprob functions
#############################################################
#
#def lnprob_gaussian_marg(p):
#    '''New lnprob, no nuisance coeffs'''
#    temp, logg, Z, vsini, vz, Av, flux_factor = p
#    if (logg < g_low) or (logg > g_high) or (vsini < 0) or (temp < T_low) or \
#            (temp > T_high) or (Z < Z_low) or (Z > Z_high) or (Av < 0):
#        return -np.inf
#    else:
#        #shift TRES wavelengths to output spectra to.
#        wlsz = wls * np.sqrt((C.c_kms - vz) / (C.c_kms + vz))
#        fmods = model(wlsz, temp, logg, Z, vsini, Av, flux_factor) * masks #mask all the bad model points
#
#        a = fmods ** 2 / sigmas ** 2
#        A = np.einsum("in,jkn->ijk", a, TT)
#        Ap = A + D
#        detA = np.array(list(map(np.linalg.det, Ap)))
#        invA = np.array(list(map(np.linalg.inv, Ap)))
#
#        b = fmods * fls / sigmas ** 2
#        B = np.einsum("in,jn->ij", b, T)
#        Bp = B + Dmu
#
#        g = -0.5 * fls ** 2 / sigmas ** 2 * masks
#        G = np.einsum("ij->i", g)
#        Gp = G - 0.5 * muDmu
#
#        invAB = np.einsum("ijk,ik->ij", invA, Bp)
#        BAB = np.einsum("ij,ij->i", Bp, invAB)
#
#        lnp = np.sum(0.5 * np.log((2. * np.pi) ** norder / detA) + 0.5 * BAB + Gp)
#
#        return lnp
#
#
#def lnprob_lognormal(p):
#    temp, logg, Z, vsini, vz, Av, flux_factor = p[:config['nparams']]
#    if (logg < g_low) or (logg > g_high) or (vsini < 0) or (temp < T_low) or \
#            (temp > T_high) or (Z < Z_low) or (Z > Z_high) or (Av < 0):
#        #if the call is outside of the loaded grid.
#        return -np.inf
#    else:
#        #shift TRES wavelengths
#        wlsz = wls * np.sqrt((C.c_kms - vz) / (C.c_kms + vz))
#        fmods = model(wlsz, temp, logg, Z, vsini, Av, flux_factor)
#
#        coefs = p[config['nparams']:]
#        # reshape to (norders, 4)
#        coefs_arr = coefs.reshape(len(orders), -1)
#        c0s = coefs_arr[:, 0] #length norders
#        cns = coefs_arr[:, 1:] #shape (norders, 3)
#        #This does correctly unpack the coefficients into c0s, cns by order 11/17/13
#
#        #If any c0s are less than 0, return -np.inf
#        if np.any((c0s < 0)):
#            return -np.inf
#
#        fdfmc0 = np.einsum('i,ij->ij', c0s, fmods * fls)
#        fm2c2 = np.einsum("i,ij->ij", c0s ** 2, fmods ** 2)
#
#        a = fm2c2 / sigmas ** 2
#        A = np.einsum("in,jkn->ijk", a, TT)
#        Ap = A + D
#
#        b = (-fm2c2 + fdfmc0) / sigmas ** 2
#        B = np.einsum("in,jn->ij", b, T)
#        Bp = B + Dmu
#
#        g = -0.5 / sigmas ** 2 * (fm2c2 - 2 * fdfmc0 + fls ** 2)
#        G = np.einsum("ij->i", g)
#        Gp = G - 0.5 * muDmu
#
#        Ac = np.einsum("ijk,ik->ij", Ap, cns)
#        cAc = np.einsum("ij,ij->i", cns, Ac)
#        Bc = np.einsum("ij,ij->i", Bp, cns)
#
#        lnp = np.sum(-0.5 * cAc + Bc + Gp) + np.sum(
#            np.log(1 / (c0s * sigmac0 * np.sqrt(2. * np.pi))) - np.log(c0s) ** 2 / (2 * sigmac0 ** 2))
#
#        return lnp
#
#
#def lnprob_lognormal_nuis_func(p):
#    '''Used for sampling the lnprob_lognormal at a fixed p for the cns.'''
#    temp, logg, Z, vsini, vz, Av, flux_factor = p[:config['nparams']]
#
#    if (logg < g_low) or (logg > g_high) or (vsini < 0) or (temp < T_low) or \
#            (temp > T_high) or (Z < Z_low) or (Z > Z_high) or (Av < 0):
#        #if the call is outside of the loaded grid.
#        return -np.inf
#    else:
#        #shift TRES wavelengths
#        wlsz = wls * np.sqrt((C.c_kms - vz) / (C.c_kms + vz))
#        fmods = model(wlsz, temp, logg, Z, vsini, Av, flux_factor)
#
#        c0s = p[config['nparams']:]
#        #If any c0s are less than 0, return -np.inf
#        if np.any((c0s < 0)):
#            return -np.inf
#
#        fdfmc0 = np.einsum('i,ij->ij', c0s, fmods * fls)
#        fm2c2 = np.einsum("i,ij->ij", c0s ** 2, fmods ** 2)
#
#        a = fm2c2 / sigmas ** 2
#        A = np.einsum("in,jkn->ijk", a, TT)
#        Ap = A + D
#
#        b = (-fm2c2 + fdfmc0) / sigmas ** 2
#        B = np.einsum("in,jn->ij", b, T)
#        Bp = B + Dmu
#
#        g = -0.5 / sigmas ** 2 * (fm2c2 - 2 * fdfmc0 + fls ** 2)
#        G = np.einsum("ij->i", g)
#        Gp = G - 0.5 * muDmu
#
#    def nuis_func(cns):
#        '''input as flat array'''
#        cns.shape = (norder, -1)
#        Ac = np.einsum("ijk,ik->ij", Ap, cns)
#        cAc = np.einsum("ij,ij->i", cns, Ac)
#        Bc = np.einsum("ij,ij->i", Bp, cns)
#        lnp = np.sum(-0.5 * cAc + Bc + Gp) + np.sum(
#            np.log(1 / (c0s * sigmac0 * np.sqrt(2. * np.pi))) - np.log(c0s) ** 2 / (2 * sigmac0 ** 2))
#        return lnp
#
#    return nuis_func
#
#
#mu_temp = 6462
#sigma_temp = 400
#mu_logg = 4.29
#sigma_logg = 0.0001
#mu_Z = -0.13
#sigma_Z = 0.7
#mu_vsini = 3.5
#sigma_vsini = 0.9
#mu_Av = 0.0
#sigma_Av = 0.01
#
#
#def lnprob_lognormal_marg(p):
#    '''Sample only in c0's  '''
#    temp, logg, Z, vsini, vz, Av, flux_factor = p[:config['nparams']]
#
#    if (logg < g_low) or (logg > g_high) or (vsini < 0) or (temp < T_low) or (temp > T_high) \
#        or (Z < Z_low) or (Z > Z_high) or (flux_factor <= 0) or (Av < 0):
#        #if the call is outside of the loaded grid.
#        return -np.inf
#    else:
#        #shift TRES wavelengths
#        wlsz = wls * np.sqrt((C.c_kms - vz) / (C.c_kms + vz))
#        fmods = model(wlsz, temp, logg, Z, vsini, Av, flux_factor) * masks
#
#        c0s = p[config['nparams']:]
#        #If any c0s are less than 0, return -np.inf
#        if np.any((c0s < 0)):
#            return -np.inf
#
#        fdfmc0 = np.einsum('i,ij->ij', c0s, fmods * fls)
#        fm2c2 = np.einsum("i,ij->ij", c0s ** 2, fmods ** 2)
#
#        a = fm2c2 / sigmas ** 2
#        A = np.einsum("in,jkn->ijk", a, TT)
#        Ap = A + D
#        detA = np.array(list(map(np.linalg.det, Ap)))
#        invA = np.array(list(map(np.linalg.inv, Ap)))
#
#        b = (-fm2c2 + fdfmc0) / sigmas ** 2
#        B = np.einsum("in,jn->ij", b, T)
#        Bp = B + Dmu
#
#        g = -0.5 / sigmas ** 2 * (fm2c2 - 2 * fdfmc0 + masks * fls ** 2)
#        G = np.einsum("ij->i", g)
#        Gp = G - 0.5 * muDmu
#
#        invAB = np.einsum("ijk,ik->ij", invA, Bp)
#        BAB = np.einsum("ij,ij->i", Bp, invAB)
#
#        lnp = np.sum(0.5 * np.log((2. * np.pi) ** norder / detA) + 0.5 * BAB + Gp) \
#              + np.sum(np.log(1 / (c0s * sigmac0 * np.sqrt(2. * np.pi))) - 0.5 * np.log(c0s) ** 2 / sigmac0 ** 2) \
#              - 0.5 * (temp - mu_temp) ** 2 / sigma_temp ** 2 - 0.5 * (logg - mu_logg) ** 2 / sigma_logg ** 2 \
#              - 0.5 * (Z - mu_Z) ** 2 / sigma_Z ** 2 - 0.5 * (vsini - mu_vsini) ** 2 / sigma_vsini \
#              - 0.5 * (Av - mu_Av) ** 2 / sigma_Av
#        return lnp
#
#
##A = 0.4
##var_G = (1.5 * sigmas) ** 2
##sigma_E = 3.0 * sigmas
##
#
#def lnprob_mixed_exp(p):
#    temp, logg, Z, vsini, vz, Av, flux_factor = p[:config['nparams']]
#
#    if (logg < g_low) or (logg > g_high) or (vsini < 0) or (temp < T_low) or (temp > T_high) \
#        or (Z < Z_low) or (Z > Z_high) or (flux_factor <= 0) or (Av < 0):
#        #if the call is outside of the loaded grid.
#        return -np.inf
#    else:
#        #shift TRES wavelengths
#        wlsz = wls * np.sqrt((C.c_kms - vz) / (C.c_kms + vz))
#        fmods = model(wlsz, temp, logg, Z, vsini, Av, flux_factor)
#
#        coefs = p[config['nparams']:]
#        # reshape to (norders, 4)
#        coefs_arr = coefs.reshape(len(orders), -1)
#        c0s = coefs_arr[:, 0] #length norders
#        cns = coefs_arr[:, 1:] #shape (norders, 3)
#        #print("c0s.shape", c0s.shape)
#        #print("cns.shape", cns.shape)
#
#        #If any c0s are less than 0, return -np.inf
#        if np.any((c0s < 0)):
#            return -np.inf
#
#        #now create polynomials for each order, and multiply through fls
#        #print("T.shape", T.shape)
#        Tc = np.einsum("jk,ij->ik", T, cns)
#        k = np.einsum("i,ij->ij", c0s, 1 + Tc)
#        #print("k.shape", k.shape)
#        kf = k * fmods
#
#        lnp = np.sum(np.log(np.exp(-0.5 * (fls - kf) ** 2 / var_G) + A * np.exp(- np.abs(fls - kf) / sigma_E))) \
#              + np.sum(np.log(1 / (c0s * sigmac0 * np.sqrt(2. * np.pi))) - 0.5 * np.log(c0s) ** 2 / sigmac0 ** 2) \
#              - 0.5 * np.sum(cns ** 2 / sigmac ** 2) \
#              - 0.5 * (Av - mu_Av) ** 2 / sigma_Av
#        #- 0.5 * (temp - mu_temp)**2/sigma_temp**2 - 0.5 * (logg - mu_logg)**2/sigma_logg**2 \
#        #- 0.5 * (Z - mu_Z)**2/sigma_Z**2 - 0.5 * (vsini - mu_vsini)**2/sigma_vsini
#        return lnp
#
#def lnprob_mixed(p):
#    temp, logg, Z, vsini, vz, Av, flux_factor = p[:config['nparams']]
#
#    if (logg < g_low) or (logg > g_high) or (vsini < 0) or (temp < T_low) or (temp > T_high) \
#        or (Z < Z_low) or (Z > Z_high) or (flux_factor <= 0) or (Av < 0):
#        #if the call is outside of the loaded grid.
#        return -np.inf
#    else:
#        #shift TRES wavelengths
#        wlsz = wls * np.sqrt((C.c_kms - vz) / (C.c_kms + vz))
#        fmods = model(wlsz, temp, logg, Z, vsini, Av, flux_factor)
#
#        coefs = p[config['nparams']:]
#        # reshape to (norders, 4)
#        coefs_arr = coefs.reshape(len(orders), -1)
#        c0s = coefs_arr[:, 0] #length norders
#        cns = coefs_arr[:, 1:] #shape (norders, 3)
#        #print("c0s.shape", c0s.shape)
#        #print("cns.shape", cns.shape)
#
#        #If any c0s are less than 0, return -np.inf
#        if np.any((c0s < 0)):
#            return -np.inf
#
#        #now create polynomials for each order, and multiply through fls
#        #print("T.shape", T.shape)
#        Tc = np.einsum("jk,ij->ik", T, cns)
#        k = np.einsum("i,ij->ij", c0s, 1 + Tc)
#        #print("k.shape", k.shape)
#        kf = k * fmods
#
#        R = (fls - kf)/sigmas
#
#        lnp = np.sum(np.log((1 - np.exp(-0.5 * R**2))/R**2)) \
#              + np.sum(np.log(1 / (c0s * sigmac0 * np.sqrt(2. * np.pi))) - 0.5 * np.log(c0s) ** 2 / sigmac0 ** 2) \
#              - 0.5 * np.sum(cns ** 2 / sigmac ** 2) \
#              - 0.5 * (Av - mu_Av) ** 2 / sigma_Av
#        #- 0.5 * (temp - mu_temp)**2/sigma_temp**2 - 0.5 * (logg - mu_logg)**2/sigma_logg**2 \
#        #- 0.5 * (Z - mu_Z)**2/sigma_Z**2 - 0.5 * (vsini - mu_vsini)**2/sigma_vsini
#        if np.isnan(lnp):
#            return -np.inf
#        else:
#            return lnp
#
#def wrap_lnprob(lnprob, temp, logg, z, vsini):
#    '''Return a lnprob function that keeps these parameters fixed'''
#    def func_lnprob(p):
#        '''This lnprob only takes vz, Av, fluxfactor, + nuisance coeffs'''
#
#        #Ideally, this does all the FFT transforming.
#
#        pnew = np.hstack((np.array([temp, logg, z, vsini]), p))
#        return lnprob(pnew)
#    return func_lnprob
#
#def lnprob_classic(p):
#    '''p is the parameter vector, contains both theta_s and theta_n'''
#    #print(p)
#    temp, logg, Z, vsini, vz, Av, flux_factor = p[:config['nparams']]
#    if (logg < g_low) or (logg > g_high) or (vsini < 0) or (temp < T_low) or \
#            (temp > T_high) or (np.abs(Z) >= 0.5) or (Av < 0):
#        return -np.inf
#    else:
#        coefs = p[config['nparams']:]
#        #print(coefs)
#        coefs_arr = coefs.reshape(len(orders), -1)
#        print(coefs_arr)
#
#        #shift TRES wavelengths
#        wlsz = wls * np.sqrt((C.c_kms + vz) / (C.c_kms - vz))
#
#        flsc = data(coefs_arr, wlsz, fls)
#
#        fs = model(wlsz, temp, logg, Z, vsini, Av, flux_factor)
#
#        chi2 = np.sum(((flsc - fs) / sigmas) ** 2)
#        L = -0.5 * chi2
#        #prior = - np.sum((coefs_arr[:,2])**2/0.1) - np.sum((coefs_arr[:,[1,3,4]]**2/0.01))
#        prior = 0
#        return L + prior
#
#
#def degrade_flux(wl, w, f_full):
#    vsini = 40.
#    #Limit huge file to the necessary order. Even at 4000 ang, 1 angstrom corresponds to 75 km/s. Add in an extra 5
#    # angstroms to be sure.
#    ind = (w_full > (wl[0] - 5.)) & (w_full < (wl[-1] + 5.))
#    w = w_full[ind]
#    f = f_full[ind]
#    #convolve with stellar broadening (sb)
#    k = vsini_ang(np.mean(wl), vsini) #stellar rotation kernel centered at order
#    f_sb = convolve(f, k)
#
#    dlam = w[1] - w[0] #spacing of model points for TRES resolution kernel
#
#    #convolve with filter to resolution of TRES
#    filt = gauss_series(dlam, lam0=np.mean(wl))
#    f_TRES = convolve(f_sb, filt)
#
#    #downsample to TRES bins
#    dsamp = downsample(w, f_TRES, wl)
#
#    return dsamp
#
#
#def data(coefs_arr, wls, fls):
#    '''coeff is a (norders, npoly) shape array'''
#    flsc = np.zeros_like(fls)
#    for i, coefs in enumerate(coefs_arr):
#        #do this to keep constant fixed at 1
#        flsc[i] = Ch(np.append([1], coefs), domain=[wls[i][0], wls[i][-1]])(wls[i]) * fls[i]
#        #do this to allow tweaks to each order
#        #flsc[i] = Ch(coefs, domain=[wls[i][0], wls[i][-1]])(wls[i]) * fls[i]
#    return flsc
#
#
#def generate_fake_data(SNR, temp, logg, Z, vsini, vz, Av, flux_factor):
#    import os
#
#    '''Generate an echelle-like spectrum to test method. SNR is quoted per-resolution element,
#    and so is converted to per-pixel via the formula on 10/31/13. The sigma is created at the Poisson level only.'''
#    SNR_pix = SNR / 1.65 #convert to per-pixel for TRES
#
#    #use LkCa15 wl grid, shifted
#    LkCa15_wls = np.load('data/LkCa15/LkCa15_2013-10-13_09h37m31s_cb.flux.spec.wls.npy')
#
#    #When running this, also need to set config['orders'] = all
#    wlsz = shift_TRES(vz, wls=LkCa15_wls)
#    fls_fake = model(wlsz, temp, logg, Z, vsini, Av, flux_factor) #create flux on a shifted grid
#    sigmas = fls_fake / SNR_pix
#
#    print("Generated data with SNR:{SNR:}, temp:{temp:}, logg:{logg:}, Z:{Z:}, "
#          "vsini:{vsini:}, vz: {vz:}, Av:{Av:}, flux-factor:{ff:}".format(SNR=SNR, temp=temp,
#                                                                          logg=logg, Z=Z, vsini=vsini, vz=vz, Av=Av,
#                                                                          ff=flux_factor))
#
#    #func = lambda x: np.random.normal(loc=0,scale=x)
#    #noise = np.array(list(map(func,sigmas)))
#    noise = np.random.normal(loc=0, scale=sigmas, size=fls_fake.shape)
#    fls_noise = fls_fake + noise
#    mask = np.ones_like(fls_noise, dtype='bool')
#
#    basedir = 'data/Fake/%.0f/' % SNR #create in a subfolder that has the SNR labelled
#    #Create necessary output directories using os.mkdir, if it does not exist
#    if not os.path.exists(basedir):
#        os.mkdir(basedir)
#        print("Created output directory", basedir)
#    else:
#        print(basedir, "already exists, overwriting.")
#    base = basedir + 'Fake'
#    np.save(base + '.wls.npy', LkCa15_wls) #write original, unshifted grid
#    np.save(base + '.fls.npy', fls_noise)
#    np.save(base + '.true.fls.npy', fls_fake)
#    np.save(base + '.sigma.npy', noise)
#    np.save(base + '.mask.npy', mask)

#@profile
def main():
    print("Starting main of model")

    pass


if __name__ == "__main__":
    main()
