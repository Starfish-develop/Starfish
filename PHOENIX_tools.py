import numpy as np
from numpy.fft import fft, ifft, fftfreq
import astropy.io.fits as pf
from astropy.io import ascii

import multiprocessing as mp

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d, UnivariateSpline
from scipy.integrate import trapz

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter as FSF

import gc
import bz2
import h5py

c_kms = 2.99792458e5 #km s^-1
wl_file = pf.open("PHOENIX/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
w_full = wl_file[0].data
wl_file.close()
ind = (w_full > 3000.) & (w_full < 13000.) #this corresponds to some extra space around the
# shortest U and longest z band

global w
w = w_full[ind]
len_p = len(w)

wave_grid_raw = np.load("wave_grids/PHOENIX_raw_trim_air.npy")
wave_grid_fine = np.load('wave_grids/PHOENIX_0.35kms_air.npy')
wave_grid_coarse = np.load('wave_grids/PHOENIX_2kms_air.npy')
#wave_grid_kurucz_raw = np.load("wave_grid_kurucz_raw.npy")
#wave_grid_2kms_kurucz = np.load("wave_grid_2kms_kurucz.npy")

L_sun = 3.839e33 #erg/s, PHOENIX header says W, but is really erg/s
R_sun = 6.955e10 #cm

F_sun = L_sun/(4 * np.pi * R_sun**2) #bolometric flux of the Sun measured at the surface

grid_PHOENIX = {'T_points': np.array(
    [2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 4000, 4100, 4200,
     4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100,
     6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000,
     9200, 9400, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 11200, 11400, 11600, 11800, 12000]),
    'logg_points': np.arange(0.0, 6.1, 0.5), 'Z_points': ['-1.0', '-0.5', '-0.0', '+0.5', '+1.0']}

#Kurucz parameters
grid_kurucz = {'T_points': np.arange(3500, 9751, 250),
               'logg_points': np.arange(1.0, 5.1, 0.5), 'Z_points': ["m05", "p00", "p05"]}

grid_BTSettl = {'T_points': np.arange(3000, 7001, 100), 'logg_points': np.arange(2.5, 5.6, 0.5),
                'Z_points': ['-0.5a+0.2', '-0.0a+0.0', '+0.5a+0.0']}

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

def create_fine_and_coarse_wave_grid():
    wave_grid_2kms_PHOENIX = create_wave_grid(2., start=3050., end=11322.2) #chosen for 3 * 2**16 = 196608
    wave_grid_fine = create_wave_grid(0.35, start=3050., end=12089.65) # chosen for 9 * 2 **17 = 1179648

    np.save('wave_grid_2kms.npy',wave_grid_2kms_PHOENIX)
    np.save('wave_grid_0.35kms.npy',wave_grid_fine)
    print(len(wave_grid_2kms_PHOENIX))
    print(len(wave_grid_fine))

def create_coarse_wave_grid_kurucz():
    start = 5050.00679905
    end = 5359.99761468
    wave_grid_2kms_kurucz = create_wave_grid(2.0, start+1,  5333.70+1)
    #8192 = 2**13
    print(len(wave_grid_2kms_kurucz))
    np.save('wave_grid_2kms_kurucz.npy', wave_grid_2kms_kurucz)


@np.vectorize
def vacuum_to_air(wl):
    '''CA Prieto recommends this as more accurate than the IAU standard. Ciddor 1996.'''
    sigma = (1e4/wl)**2
    f = 1.0 + 0.05792105/(238.0185 - sigma) + 0.00167917/(57.362 - sigma)
    return wl/f

@np.vectorize
def vacuum_to_air_SLOAN(wl):
    '''Takes wavelength in angstroms and maps to wl in air.
    from SLOAN website
     AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)'''
    air = wl / (1.0 + 2.735182E-4 + 131.4182 / wl**2 + 2.76249E8 / wl**4)
    return air

@np.vectorize
def air_to_vacuum(wl):
    sigma = 1e4/wl
    vac = wl + wl * (6.4328e-5 + 2.94981e-2/(146 - sigma**2) + 2.5540e-4/(41 - sigma**2))
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
    wl = 10**(w1 + dw * p)
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
    fl = 10**(fl - 8.) #now in ergs/cm^2/s/A

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

def load_flux_full(temp, logg, Z, norm=False, vsini=0, grid="PHOENIX"):
    '''Load a raw PHOENIX or kurucz spectrum based upon temp, logg, and Z. Normalize to F_sun if desired.'''

    if grid == "PHOENIX":
        rname = "HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{Z:}/lte{temp:0>5.0f}-{logg:.2f}{Z:}" \
            ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(Z=Z, temp=temp, logg=logg)
    elif grid == "kurucz":
        rname = "Kurucz/t{temp:0>5.0f}g{logg:.0f}{Z:}v{vsini:0>3.0f}.fits".format(temp=temp,
                                                                                  logg=10*logg, Z=Z, vsini=vsini)
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

def resample(f_input, wave_grid_input, wave_grid_output):
    '''Take a TRES spectrum and resample it to 2km/s binning. For the kurucz grid.'''

    interp = InterpolatedUnivariateSpline(wave_grid_input, f_input)
    f_output = interp(wave_grid_output)
    return f_output

def process_spectrum_PHOENIX(pars):
    temp, logg, Z = pars
    try:
        f = load_flux_full(temp, logg, Z, norm=True, grid="PHOENIX")[ind]
        flux = resample_and_convolve(f, wave_grid_raw, wave_grid_fine, wave_grid_coarse)
        print("processing %s, %s, %s" % (temp, logg, Z))
    except OSError:
        print("%s, %s, %s does not exist!" % (temp, logg, Z))
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

def process_spectrum_BTSettl(pars):
    temp, logg, Z = pars
    try:
        wl, f = load_BTSettl(temp, logg, Z, norm=True, trunc=True, air=True)
        flux = resample_and_convolve(f, wl, wave_grid_fine, wave_grid_coarse)
        print("PROCESSED: %s, %s, %s" % (temp, logg, Z))
    except (FileNotFoundError, OSError): #on Python2 gives OS, Python3 gives FileNotFound
        print("FAILED: %s, %s, %s" % (temp, logg, Z))
        flux = np.nan
    return flux

def create_grid_parallel(ncores, hdf5_filename, grid_name="PHOENIX"):
    '''create an hdf5 file of the stellar grid. Go through each T point, if the corresponding logg exists,
    write it. If not, write nan. Each spectrum is normalized to the bolometric flux at the surface of the Sun.'''
    f = h5py.File(hdf5_filename, "w")

    if grid_name == "PHOENIX":
        grid = grid_PHOENIX
        process_spectrum = process_spectrum_PHOENIX
        wave_grid_2kms = wave_grid_coarse
    elif grid_name == "kurucz":
        grid = grid_kurucz
        process_spectrum = process_spectrum_kurucz
        wave_grid_2kms = wave_grid_2kms_kurucz
    elif grid_name == 'BTSettl':
        grid = grid_BTSettl
        process_spectrum = process_spectrum_BTSettl
        wave_grid_2kms = wave_grid_coarse
    else:
        print("No grid %s" % (grid_name))
        return 1

    T_points = grid['T_points']
    logg_points = grid['logg_points']
    Z_points = grid['Z_points']

    shape = (len(T_points), len(logg_points), len(Z_points), len(wave_grid_2kms))
    dset = f.create_dataset("LIB", shape, dtype="f", compression='gzip', compression_opts=9)

    # A thread pool of P processes
    pool = mp.Pool(ncores)

    index_combos = []
    var_combos = []
    for t, temp in enumerate(T_points):
        for l, logg in enumerate(logg_points):
            for z, Z in enumerate(Z_points):
                index_combos.append([t, l, z])
                var_combos.append([temp, logg, Z])

    spec_gen = pool.imap(process_spectrum, var_combos, chunksize=20)

    for i, spec in enumerate(spec_gen):
        t, l, z = index_combos[i]
        dset[t, l, z, :] = spec
        print("Writing ", var_combos[i], "to HDF5")

    f.close()

# Interpolation routines
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
def v(ls,lo):
    return c_kms * (lo ** 2 - ls ** 2) / (ls ** 2 + lo ** 2)




def main():
    ncores = mp.cpu_count()
    #create_fine_and_coarse_wave_grid()
    #create_coarse_wave_grid_kurucz()

    #create_grid_parallel(ncores, "LIB_kurucz_2kms.hdf5", grid_name="kurucz")
    #create_grid_parallel(ncores, "LIB_PHOENIX_2kms_air.hdf5", grid_name="PHOENIX")

    create_grid_parallel(ncores, "LIB_BTSettl_2kms_air.hdf5", grid_name="BTSettl")




if __name__ == "__main__":
    main()
