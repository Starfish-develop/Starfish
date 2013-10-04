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
