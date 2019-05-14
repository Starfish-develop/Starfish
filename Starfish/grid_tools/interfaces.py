import bz2
import logging
import os

import numpy as np
from astropy.io import fits, ascii
from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm

import Starfish.constants as C
from Starfish.utils import create_log_lam_grid
from .base_interfaces import GridInterface
from .utils import vacuum_to_air, idl_float

log = logging.getLogger(__name__)


class PHOENIXGridInterface(GridInterface):
    """
    An Interface to the PHOENIX/Husser synthetic library.

    Note that the wavelengths in the spectra are in Angstrom and the flux are in :math:`F_\\lambda` as
    :math:`erg/s/cm^2/cm`
    """

    def __init__(self, path, air=True, wl_range=(3000, 54000)):
        super().__init__(name='PHOENIX',
                         param_names=['T', 'logg', 'Z', 'alpha'],
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
                         wave_units='AA', flux_units='erg/s/cm^2/cm',
                         air=air, wl_range=wl_range, path=path)  # wl_range used to be (2999, 13001)

        self.par_dicts = [None,
                          None,
                          {-2: '-2.0', -1.5: '-1.5', -1: '-1.0', -0.5: '-0.5',
                           0.0: '-0.0', 0.5: '+0.5', 1: '+1.0'},
                          {-0.4: '.Alpha=-0.40', -0.2: '.Alpha=-0.20',
                           0.0: '', 0.2: '.Alpha=+0.20', 0.4: '.Alpha=+0.40',
                           0.6: '.Alpha=+0.60', 0.8: '.Alpha=+0.80'}]

        # if air is true, convert the normally vacuum file to air wls.
        try:
            wl_filename = os.path.join(
                self.path, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
            w_full = fits.getdata(wl_filename)
        except:
            raise ValueError('Wavelength file improperly specified.')

        if self.air:
            self.wl_full = vacuum_to_air(w_full)
        else:
            self.wl_full = w_full

        self.ind = (self.wl_full >= self.wl_range[0]) & (
            self.wl_full <= self.wl_range[1])
        self.wl = self.wl_full[self.ind]
        self.rname = 'Z{2:}{3:}/lte{0:0>5.0f}-{1:.2f}{2:}{3:}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        self.full_rname = os.path.join(self.path, self.rname)

    def load_flux(self, parameters, header=False, norm=True):
        self.check_params(parameters)  # Check to make sure that the keys are
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

        fname = self.full_rname.format(*str_parameters)

        # Still need to check that file is in the grid, otherwise raise a C.GridError
        # Read all metadata in from the FITS header, and append to spectrum
        if not os.path.exists(fname):
            raise ValueError('{} is not on disk.'.format(fname))

        hdu_list = fits.open(fname)
        flux = hdu_list[0].data
        hdr = dict(hdu_list[0].header)
        hdu_list.close()

        # If we want to normalize the spectra, we must do it now since later we won't have the full EM range
        if norm:
            flux *= 1e-8  # convert from erg/cm^2/s/cm to erg/cm^2/s/A
            F_bol = trapz(flux, self.wl_full)
            # bolometric luminosity is always 1 L_sun
            flux *= (C.F_sun / F_bol)

        # Add temp, logg, Z, alpha, norm to the metadata
        hdr['norm'] = norm
        hdr['air'] = self.air

        if header:
            return (flux[self.ind], hdr)
        else:
            return flux[self.ind]


class PHOENIXGridInterfaceNoAlpha(PHOENIXGridInterface):
    """
    An Interface to the PHOENIX/Husser synthetic library without any Alpha concentration doping.
    """

    def __init__(self, path, air=True, wl_range=(3000, 54000)):
        # Initialize according to the regular PHOENIX values
        super().__init__(air=air, wl_range=wl_range, path=path)

        # Now override parameters to exclude alpha
        self.param_names = ['T', 'logg', 'Z']
        self.points = [
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
                          {-2: '-2.0', -1.5: '-1.5', -1: '-1.0', -0.5: '-0.5',
                           0.0: '-0.0', 0.5: '+0.5', 1: '+1.0'}]
        self.rname = 'Z{2:}/lte{0:0>5.0f}-{1:.2f}{2:}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        self.full_rname = os.path.join(self.path, self.rname)


class KuruczGridInterface(GridInterface):
    """
    Kurucz grid interface.

    Spectra are stored in ``f_nu`` in a filename like
    ``t03500g00m25ap00k2v070z1i00.fits``, ``ap00`` means zero alpha enhancement,
    and ``k2`` is the microturbulence, while ``z1`` is the macroturbulence.
    These particular values are roughly the ones appropriate for the Sun.
    """

    def __init__(self, path, air=True, wl_range=(5000, 5400)):
        super().__init__(name='Kurucz',
                         param_names=['T', 'logg', 'Z'],
                         points=[np.arange(3500, 9751, 250),
                                 np.arange(0.0, 5.1, 0.5),
                                 np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5])],
                         wave_units='AA', flux_units='',
                         air=air, wl_range=wl_range, path=path)

        self.par_dicts = [None, None,
                          {-2.5: 'm25', -2.0: 'm20', -1.5: 'm15', -1.0: 'm10', -0.5: 'm05', 0.0: 'p00', 0.5: 'p05'}]

        # Convert to f_lam and average to 1, or leave in f_nu?
        self.rname = 't{0:0>5.0f}/g{1:0>2.0f}/t{0:0>5.0f}g{1:0>2.0f}{2}ap00k2v000z1i00.fits'
        self.full_rname = os.path.join(self.path, self.rname)
        self.wl_full = np.load(os.path.join(path, 'kurucz_raw_wl.npy'))
        self.ind = (self.wl_full >= self.wl_range[0]) & (
            self.wl_full <= self.wl_range[1])
        self.wl = self.wl_full[self.ind]

    def load_flux(self, parameters, header=False, norm=True):
        self.check_params(parameters)

        str_parameters = []
        for param, par_dict in zip(parameters, self.par_dicts):
            if par_dict is None:
                str_parameters.append(param)
            else:
                str_parameters.append(par_dict[param])

        # Multiply logg by 10
        str_parameters[1] *= 10

        fname = self.full_rname.format(*str_parameters)

        # Still need to check that file is in the grid, otherwise raise a C.GridError
        # Read all metadata in from the FITS header, and append to spectrum
        try:
            flux_file = fits.open(fname)
            f = flux_file[0].data
            hdr = dict(flux_file[0].header)
            flux_file.close()
        except:
            raise ValueError('{} is not on disk.'.format(fname))

        # We cannot normalize the spectra, since we don't have a full wl range, so instead we set the average
        # flux to be 1

        # Also, we should convert from f_nu to f_lam
        if norm:
            f *= C.c_ang / self.wl ** 2  # Convert from f_nu to f_lambda
            f /= np.average(f)  # divide by the mean flux, so avg(f) = 1

        # Add temp, logg, Z, norm to the metadata
        hdr['norm'] = norm
        hdr['air'] = self.air

        if header:
            return (f[self.ind], hdr)
        else:
            return f[self.ind]

    @staticmethod
    def get_wl_kurucz(filename):
        """The Kurucz grid is log-linear spaced."""
        flux_file = fits.open(filename)
        hdr = flux_file[0].header
        num = len(flux_file[0].data)
        p = np.arange(num)
        w1 = hdr['CRVAL1']
        dw = hdr['CDELT1']
        wl = 10 ** (w1 + dw * p)
        return wl


class BTSettlGridInterface(GridInterface):
    """
    BTSettl grid interface. Unlike the PHOENIX and Kurucz grids, the
    individual files of the BTSettl grid do not always have the same wavelength
    sampling. Therefore, each call of :meth:`load_flux` will interpolate the
    flux onto a LogLambda spaced grid that ranges between `wl_range` and has a
    velocity spacing of 0.08 km/s or better.

    If you have a choice, it's probably easier to use the Husser PHOENIX grid.
    """

    def __init__(self, path, air=True, wl_range=(2999, 13000)):
        super().__init__(name='BTSettl',
                         points={'T': np.arange(3000, 7001, 100),
                                 'logg': np.arange(2.5, 5.6, 0.5),
                                 'Z': np.arange(-0.5, 0.6, 0.5),
                                 'alpha': np.array([0.0])},
                         wave_units='AA', flux_units='',
                         air=air, wl_range=wl_range, path=path)

        # Normalize to 1 solar luminosity?
        self.rname = 'CIFIST2011/M{Z:}/lte{T:0>3.0f}-{logg:.1f}{Z:}.BT-Settl.spec.7.bz2'
        self.full_rname = os.path.join(self.path, self.rname)
        # self.Z_dict = {-2:'-2.0', -1.5:'-1.5', -1:'-1.0', -0.5:'-0.5', 0.0: '-0.0', 0.5: '+0.5', 1: '+1.0'}
        self.Z_dict = {-0.5: '-0.5a+0.2', 0.0: '-0.0a+0.0', 0.5: '+0.5a0.0'}

        wl_dict = create_log_lam_grid(
            0.08 / C.c_kms, wl_start=self.wl_range[0], wl_end=self.wl_range[1])
        self.wl = wl_dict['wl']

    def load_flux(self, parameters, norm=True):
        """
        Because of the crazy format of the BTSettl, we need to sort the wl to make sure
        everything is unique, and we're not screwing ourselves with the spline.

        :param parameters: stellar parameters
        :type parameters: dict
        :param norm: If True, will normalize the spectrum to solar luminosity. Default is True
        :type norm: bool
        """
        # Check to make sure that the keys are allowed and that the values are in the grid
        self.check_params(parameters)

        str_parameters = parameters.copy()

        # Rewrite Z
        Z = parameters['Z']
        str_parameters['Z'] = self.Z_dict[Z]

        # Multiply temp by 0.01
        str_parameters['T'] = 0.01 * parameters['T']

        fname = self.full_rname.format(**str_parameters)
        file = bz2.BZ2File(fname, 'r')

        lines = file.readlines()
        strlines = [line.decode('utf-8') for line in lines]
        file.close()

        data = ascii.read(strlines, col_starts=(0, 13), col_ends=(
            12, 25), Reader=ascii.FixedWidthNoHeader)
        wl = data['col1']
        fl_str = data['col2']

        # convert because of 'D' exponent, unreadable in Python
        fl = idl_float(fl_str)
        fl = 10 ** (fl - 8.)  # now in ergs/cm^2/s/A

        # 'Clean' the wl and flux points. Remove duplicates, sort in increasing wl
        wl, ind = np.unique(wl, return_index=True)
        fl = fl[ind]

        if norm:
            F_bol = trapz(fl, wl)
            fl = fl * (C.F_sun / F_bol)
            # the bolometric luminosity is always 1 L_sun

        # truncate the spectrum to the wl range of interest
        # at this step, make the range a little more so that the next stage of
        # spline interpolation is properly in bounds
        ind = (wl >= (self.wl_range[0] - 50.)
               ) & (wl <= (self.wl_range[1] + 50.))
        wl = wl[ind]
        fl = fl[ind]

        if self.air:
            # Shift the wl that correspond to the raw spectrum
            wl = vacuum_to_air(wl)

        # Now interpolate wl, fl onto self.wl
        interp = InterpolatedUnivariateSpline(wl, fl, k=5)
        fl_interp = interp(self.wl)

        return fl_interp


class CIFISTGridInterface(GridInterface):
    """
    CIFIST grid interface, grid available here: https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/FITS/.
    Unlike the PHOENIX and Kurucz grids, the
    individual files of the BTSettl grid do not always have the same wavelength
    sampling. Therefore, each call of :meth:`load_flux` will interpolate the
    flux onto a LogLambda spaced grid that ranges between `wl_range` and has a
    velocity spacing of 0.08 km/s or better.

    If you have a choice, it's probably easier to use the Husser PHOENIX grid.
    """

    def __init__(self, path, air=True, wl_range=(3000, 13000)):
        super().__init__(name='CIFIST',
                         points=[np.concatenate((np.arange(1200, 2351, 50), np.arange(2400, 7001, 100)), axis=0),
                                 np.arange(2.5, 5.6, 0.5)],
                         param_names=['T', 'logg'],
                         wave_units='AA', flux_units='',
                         air=air, wl_range=wl_range, path=path)

        self.par_dicts = [None, None]
        self.rname = 'lte{0:0>5.1f}-{1:.1f}-0.0a+0.0.BT-Settl.spec.fits.gz'
        self.full_rname = os.path.join(self.path, self.rname)

        wl_dict = create_log_lam_grid(
            dv=0.08, wl_start=self.wl_range[0], wl_end=self.wl_range[1])
        self.wl = wl_dict['wl']

    def load_flux(self, parameters, header=False, norm=True):
        self.check_params(parameters)

        str_parameters = []
        for param, par_dict in zip(parameters, self.par_dicts):
            if par_dict is None:
                str_parameters.append(param)
            else:
                str_parameters.append(par_dict[param])

        # Multiply temp by 0.01
        str_parameters[0] = 0.01 * parameters[0]

        fname = self.full_rname.format(*str_parameters)

        # Still need to check that file is in the grid, otherwise raise a C.GridError
        # Read all metadata in from the FITS header, and append to spectrum
        try:
            flux_file = fits.open(fname)
            data = flux_file[1].data
            hdr = dict(flux_file[1].header)

            wl = data['Wavelength'] * 1e4  # [Convert to angstroms]
            fl = data['Flux']

            flux_file.close()
        except OSError:
            raise C.GridError('{} is not on disk.'.format(fname))

        # "Clean" the wl and flux points. Remove duplicates, sort in increasing wl
        wl, ind = np.unique(wl, return_index=True)
        fl = fl[ind]

        if norm:
            F_bol = trapz(fl, wl)
            fl = fl * (C.F_sun / F_bol)
            # the bolometric luminosity is always 1 L_sun

        # truncate the spectrum to the wl range of interest
        # at this step, make the range a little more so that the next stage of
        # spline interpolation is properly in bounds
        ind = (wl >= (self.wl_range[0] - 50.)
               ) & (wl <= (self.wl_range[1] + 50.))
        wl = wl[ind]
        fl = fl[ind]

        if self.air:
            # Shift the wl that correspond to the raw spectrum
            wl = vacuum_to_air(wl)

        # Now interpolate wl, fl onto self.wl
        interp = InterpolatedUnivariateSpline(wl, fl, k=5)
        fl_interp = interp(self.wl)

        # Add temp, logg, Z, norm to the metadata

        hdr['norm'] = norm
        hdr['air'] = self.air

        if header:
            return (fl_interp, hdr)
        else:
            return fl_interp


def load_BTSettl(T, logg, Z, norm=False, trunc=False, air=False):
    rname = 'BT-Settl/CIFIST2011/M{Z:}/lte{T:0>3.0f}-{logg:.1f}{Z:}.BT-Settl.spec.7.bz2'.format(T=0.01 * T,
                                                                                                logg=logg, Z=Z)
    file = bz2.BZ2File(rname, 'r')

    lines = file.readlines()
    strlines = [line.decode('utf-8') for line in lines]
    file.close()

    data = ascii.read(strlines, col_starts=[0, 13], col_ends=[
                      12, 25], Reader=ascii.FixedWidthNoHeader)
    wl = data['col1']
    fl_str = data['col2']

    # convert because of "D" exponent, unreadable in Python
    fl = idl_float(fl_str)
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
