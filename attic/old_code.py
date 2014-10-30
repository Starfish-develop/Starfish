
print("Hello")


def downsample(w_m, f_m, w_TRES):
    '''Given a model wavelength and flux (w_m, f_m) and the instrument wavelength (w_TRES), downsample the model to
    exactly match the TRES wavelength bins. '''
    spec_interp = interp1d(w_m, f_m, kind="linear")

    @np.vectorize
    def avg_bin(bin0, bin1):
        mdl_ind = (w_m > bin0) & (w_m < bin1)
        wave = np.empty((np.sum(mdl_ind) + 2,))
        flux = np.empty((np.sum(mdl_ind) + 2,))
        wave[0] = bin0
        wave[-1] = bin1
        flux[0] = spec_interp(bin0)
        flux[-1] = spec_interp(bin1)
        wave[1:-1] = w_m[mdl_ind]
        flux[1:-1] = f_m[mdl_ind]
        return trapz(flux, wave) / (bin1 - bin0)

    #Determine the bin edges
    edges = np.empty((len(w_TRES) + 1,))
    difs = np.diff(w_TRES) / 2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]
    b0s = edges[:-1]
    b1s = edges[1:]

    samp = avg_bin(b0s, b1s)
    return (samp)


def downsample2(w_m, f_m, w_TRES):
    '''Given a model wavelength and flux (w_m, f_m) and the instrument wavelength (w_TRES), downsample the model to
    exactly match the TRES wavelength bins. Try this without calling the interpolation routine.'''

    @np.vectorize
    def avg_bin(bin0, bin1):
        mdl_ind = (w_m > bin0) & (w_m < bin1)
        length = np.sum(mdl_ind) + 2
        wave = np.empty((length,))
        flux = np.empty((length,))
        wave[0] = bin0
        wave[-1] = bin1
        wave[1:-1] = w_m[mdl_ind]
        flux[1:-1] = f_m[mdl_ind]
        flux[0] = flux[1]
        flux[-1] = flux[-2]
        return trapz(flux, wave) / (bin1 - bin0)

    #Determine the bin edges
    edges = np.empty((len(w_TRES) + 1,))
    difs = np.diff(w_TRES) / 2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]
    b0s = edges[:-1]
    b1s = edges[1:]

    return avg_bin(b0s, b1s)


def downsample3(w_m, f_m, w_TRES):
    '''Given a model wavelength and flux (w_m, f_m) and the instrument wavelength (w_TRES), downsample the model to
    exactly match the TRES wavelength bins. Try this only by averaging.'''

    #More time could be saved by splitting up the original array into averageable chunks.

    @np.vectorize
    def avg_bin(bin0, bin1):
        return np.average(f_m[(w_m > bin0) & (w_m < bin1)])

    #Determine the bin edges
    edges = np.empty((len(w_TRES) + 1,))
    difs = np.diff(w_TRES) / 2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]
    b0s = edges[:-1]
    b1s = edges[1:]

    return avg_bin(b0s, b1s)


def downsample4(w_m, f_m, w_TRES):
    out_flux = np.zeros_like(w_TRES)
    len_mod = len(w_m)

    #Determine the bin edges
    len_TRES = len(w_TRES)
    edges = np.empty((len_TRES + 1,))
    difs = np.diff(w_TRES) / 2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]

    i_start = np.argwhere((w_m > edges[0]))[0][0] #return the first starting index for the model wavelength array

    edges_i = 1
    for i in range(len(w_m)):
        if w_m[i] > edges[edges_i]:
            i_finish = i - 1
            out_flux[edges_i - 1] = np.mean(f_m[i_start:i_finish])
            edges_i += 1
            i_start = i_finish
            if edges_i > len_TRES:
                break
    return out_flux


#Keep out here so memory keeps getting overwritten
fluxes = np.empty((4, len(wave_grid)))


def flux_interpolator_mini(temp, logg):
    '''Load flux in a memory-nice manner. lnprob will already check that we are within temp = 2300 - 12000 and logg =
    0.0 - 6.0, so we do not need to check that here.'''
    #Determine T plus and minus
    #If the previous check by lnprob was correct, these should always have elements
    #Determine logg plus and minus
    i_Tm = np.argwhere(temp >= T_points)[-1][0]
    Tm = T_points[i_Tm]
    i_Tp = np.argwhere(temp < T_points)[0][0]
    Tp = T_points[i_Tp]
    i_lm = np.argwhere(logg >= logg_points)[-1][0]
    lm = logg_points[i_lm]
    i_lp = np.argwhere(logg < logg_points)[0][0]
    lp = logg_points[i_lp]

    indexes = [(i_Tm, i_lm), (i_Tm, i_lp), (i_Tp, i_lm), (i_Tp, i_lp)]
    points = np.array([(Tm, lm), (Tm, lp), (Tp, lm), (Tp, lp)])
    for i in range(4):
    #Load spectra for these points
        #print(indexes[i])
        fluxes[i] = LIB[indexes[i]]
    if np.isnan(fluxes).any():
    #If outside the defined grid (demarcated in the hdf5 object by nan's) just return 0s
        return zero_flux

    #Interpolate spectra with LinearNDInterpolator
    flux_intp = LinearNDInterpolator(points, fluxes)
    new_flux = flux_intp(temp, logg)
    return new_flux


def flux_interpolator():
    #points = np.loadtxt("param_grid_GWOri.txt")
    points = np.loadtxt("param_grid_interp_test.txt")
    #TODO: make this dynamic, specify param_grid dynamically too
    len_w = 716665
    fluxes = np.empty((len(points), len_w))
    for i in range(len(points)):
        fluxes[i] = load_flux(points[i][0], points[i][1])
        #flux_intp = NearestNDInterpolator(points, fluxes)
    flux_intp = LinearNDInterpolator(points, fluxes, fill_value=1.)
    del fluxes
    print("Loaded flux_interpolator")
    return flux_intp


#Originally from PHOENIX_tools

def create_grid_parallel_Z0(ncores):
    '''create an hdf5 file of the PHOENIX grid. Go through each T point, if the corresponding logg exists,
    write it. If not, write nan.'''
    f = h5py.File("LIB_2kms.hdf5", "w")
    shape = (len(T_points), len(logg_points), len(wave_grid_coarse))
    dset = f.create_dataset("LIB", shape, dtype="f")

    # A thread pool of P processes
    pool = mp.Pool(ncores)

    param_combos = []
    var_combos = []
    for t, temp in enumerate(T_points):
        for l, logg in enumerate(logg_points):
            param_combos.append([t, l])
            var_combos.append([temp, logg])

    spec_gen = list(pool.map(process_spectrum_Z0, var_combos))
    for i in range(len(param_combos)):
        t, l = param_combos[i]
        dset[t, l, :] = spec_gen[i]

    f.close()

def process_spectrum_Z0(pars):
    temp, logg = pars
    try:
        f = load_flux_full(temp, logg, True)[ind]
        flux = resample_and_convolve(f,wave_grid_fine,wave_grid_coarse)
        print("Finished %s, %s" % (temp, logg))
    except OSError:
        print("%s, %s does not exist!" % (temp, logg))
        flux = np.nan
    return flux

def load_flux_full_Z0(temp, logg, norm=False):
    rname = "HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0" \
            ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(
        temp=temp, logg=logg)
    flux_file = pf.open(rname)
    f = flux_file[0].data
    L = flux_file[0].header['PHXLUM'] #W
    if norm:
        f = f * (L_sun / L)
        print("Normalized luminosity to 1 L_sun")
    flux_file.close()
    print("Loaded " + rname)
    return f

def flux_interpolator():
    points = ascii.read("param_grid.txt")
    T_list = points["T"].data
    logg_list = points["logg"].data
    fluxes = np.empty((len(T_list), len(w)))
    for i in range(len(T_list)):
        fluxes[i] = load_flux_npy(T_list[i], logg_list[i])
    flux_intp = NearestNDInterpolator(np.array([T_list, logg_list]).T, fluxes)
    return flux_intp


def flux_interpolator_np():
    points = np.loadtxt("param_grid.txt")
    print(points)
    #T_list = points["T"].data
    #logg_list = points["logg"].data
    len_w = 716665
    fluxes = np.empty((len(points), len_w))
    for i in range(len(points)):
        fluxes[i] = load_flux_npy(points[i][0], points[i][1])
    flux_intp = NearestNDInterpolator(points, fluxes)
    return flux_intp

def flux_interpolator_hdf5():
    #load hdf5 file of PHOENIX grid
    fhdf5 = h5py.File(LIB, 'r')
    LIB = fhdf5['LIB']
    index_combos = []
    var_combos = []
    for ti in range(len(T_points)):
        for li in range(len(logg_points)):
            for zi in range(len(Z_points)):
                index_combos.append([T_arg[ti], logg_arg[li], Z_arg[zi]])
                var_combos.append([T_points[ti], logg_points[li], Z_points[zi]])
        #print(param_combos)
    num_spec = len(index_combos)
    points = np.array(var_combos)

    fluxes = np.empty((num_spec, len(wave_grid)))
    for i in range(num_spec):
        t, l, z = index_combos[i]
        fluxes[i] = LIB[t, l, z][ind]
    flux_intp = LinearNDInterpolator(points, fluxes, fill_value=1.)
    fhdf5.close()
    del fluxes
    gc.collect()
    return flux_intp


import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, griddata
import matplotlib.pyplot as plt
import model as m
from scipy.special import hyp0f1, struve, j1
import PHOENIX_tools as pt

c_kms = 2.99792458e5 #km s^-1

f_full = pt.load_flux_full(5900, 3.5, True)
w_full = pt.w_full
#Truncate to Dave's order
#ind = (w_full > 5122) & (w_full < 5218)
ind = (w_full > 3000) & (w_full < 12000.6)
w_full = w_full[ind]
f_full = f_full[ind]


def calc_lam_grid(v=1., start=3700., end=10000):
    '''Returns a grid evenly spaced in velocity'''
    size = 600000 #this number just has to be bigger than the final array
    lam_grid = np.zeros((size,))
    i = 0
    lam_grid[i] = start
    vel = np.sqrt((c_kms + v) / (c_kms - v))
    while (lam_grid[i] < end) and (i < size - 1):
        lam_new = lam_grid[i] * vel
        i += 1
        lam_grid[i] = lam_new
    return lam_grid[np.nonzero(lam_grid)][:-1]

#grid = calc_lam_grid(2.,start=3050.,end=11232.) #chosen to correspond to min U filter and max z filter
#wave_grid = calc_lam_grid(0.35, start=3050., end=11232.) #this spacing encapsulates the maximal velocity resolution
# of the PHOENIX grid, and corresponds to Delta lambda = 0.006 Ang at 5000 Ang.

#np.save('wave_grid_0.35kms.npy',wave_grid)
#Truncate wave_grid to Dave's order
wave_grid = np.load('wave_grid_0.35kms.npy')[:-1]
wave_grid = wave_grid[(wave_grid > 5165) & (wave_grid < 5190)]
np.save('wave_grid_trunc.npy', wave_grid)


@np.vectorize
def gauss_taper(s, sigma=2.89):
    '''This is the FT of a gaussian w/ this sigma. Sigma in km/s'''
    return np.exp(-2 * np.pi ** 2 * sigma * 2 * s ** 2)


def convolve_gauss(wl, fl, sigma=2.89, spacing=2.):
    ##Take FFT of f_grid
    out = np.fft.fft(np.fft.fftshift(fl))
    N = len(fl)
    freqs = np.fft.fftfreq(N, d=spacing)
    taper = gauss_taper(freqs, sigma)
    tout = out * taper
    blended = np.fft.fftshift(np.fft.ifft(tout))
    return np.absolute(blended) #remove tiny complex component


def IUS(w, f, wl):
    f = InterpolatedUnivariateSpline(w, f)
    return f(wl)


def plot_interpolated():
    f_grid = IUS(w_full, f_full, wave_grid)
    np.save('f_grid.npy', f_grid)
    print("Calculated flux_grid")
    print("Length flux grid", len(f_grid))
    f_grid6 = convolve_gauss(wave_grid, f_grid, spacing=0.35)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wave_grid, f_grid)
    #ax.plot(m.wls[0], IUS(wave_grid, f_grid6, m.wls[0]),"o")
    plt.show()


@np.vectorize
def lanczos_kernel(x, a=2):
    if np.abs(x) < a:
        return np.sinc(np.pi * x) * np.sinc(np.pi * x / a)
    else:
        return 0.


def grid_interp():
    return griddata(grid, blended, m.wls, method='linear')


def G(s, vL):
    '''vL in km/s. Gray pg 475'''
    ub = 2. * np.pi * vL * s
    return j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)


def plot_gray():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ss = np.linspace(0.001, 2, num=200)
    Gs1 = G(ss, 1.)
    Gs2 = G(ss, 2.)
    ax.plot(ss, Gs1)
    ax.plot(ss, Gs2)
    plt.show()


def main():
    plot_interpolated()
    pass


if __name__ == "__main__":
    main()


#Old sinc interpolation routines that didn't work out
#Test sinc interpolation
#def func(x):
#    return (x - 3)**2 + 2 * x
#
#xs = np.arange(-299,301,1)
#ys = xs 
#
#def sinc_interpolate(x):
#    ind = np.argwhere(x > xs )[-1][0]
#    ind2 = ind + 1
#    print("ind",ind)
#    print(xs[ind])
#    print(xs[ind2])
#    frac = x - xs[ind]
#    print(frac)
#    spacing = 1 
#    pts_grid = np.arange(-299.5,300,1)
#    sinc_pts = np.sinc(pts_grid)
#    print(pts_grid,sinc_pts,trapz(sinc_pts))
#    flux_pts = ys
#    print("Interpolated value",np.sum(sinc_pts * flux_pts))
#    print("Neighboring value", ys[ind], ys[ind2])
#    return(sinc_pts,flux_pts)

#Now, do since interpolation to the TRES pixels on the blended spectrum

##Take TRES pixel, call that the center of sinc, then sample it at +/- the other pixels in the grid
#def sinc_interpolate(wl_TRES):
#    ind = np.argwhere(wl_TRES > grid)[-1][0]
#    ind2 = ind + 1
#    print(grid[ind])
#    print(grid[ind2])
#    frac = (wl_TRES - grid[ind])/(grid[ind2] - grid[ind])
#    print(frac)
#    spacing = 2 #km/s
#    veloc_grid = np.arange(-48.,51,spacing) - frac * spacing
#    print(veloc_grid)
#    #convert wl spacing to velocity spacing
#    sinc_pts = 0.5 * np.sinc(0.5 * veloc_grid)
#    print(sinc_pts,trapz(sinc_pts,veloc_grid))
#    print("Interpolated flux",np.sum(sinc_pts * f_grid[ind - 25: ind + 25]))
#    print("Neighboring flux", f_grid[ind], f_grid[ind2])

#sinc_interpolate(6610.02)
