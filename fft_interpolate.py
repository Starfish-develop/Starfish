import numpy as np
from scipy.interpolate import interp1d,UnivariateSpline,griddata
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import model as m
from scipy.special import hyp0f1,struve,j1

c_kms = 2.99792458e5 #km s^-1

#f_lam = pt.load_flux_full(5900,3.5)
# pt.w_full wavelength array
#ind = (pt.w_full > 2990.) & (pt.w_full < 10010.)
#w = pt.w_full[ind]
#f_lam = f_lam[ind]

#intp = interp1d(pt.w, f_lam, kind="cubic")

def calc_lam_grid(v=1.,start=3700.,end=10000):
    '''Returns a grid evenly spaced in velocity'''
    size = 600000 #this number just has to be bigger than the final array
    lam_grid = np.zeros((size,))
    lam = start
    i = 0
    lam_grid[i] = lam
    while (lam < end) and (i < size - 1):
        lam_new = lam * np.sqrt((c_kms + v)/(c_kms - v))
        i += 1
        lam_grid[i] = lam_new
        lam = lam_new
    return lam_grid[np.nonzero(lam_grid)][:-1]

#grid = calc_lam_grid(2.,start=3050.,end=11232.) #chosen to correspond to min U filter and max z filter
#np.save('wave_grid_2kms.npy',grid)

ones = np.ones((10,))
def downsample(w_m,f_m,w_TRES):
    out_flux = np.zeros_like(w_TRES)
    len_mod = len(w_m)

    #Determine the TRES bin edges
    len_TRES = len(w_TRES)
    edges = np.empty((len_TRES+1,))
    difs = np.diff(w_TRES)/2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]

    #Determine PHOENIX bin edges
    Pedges = np.empty((len_mod+1,))
    Pdifs = np.diff(w_m)/2.
    Pedges[1:-1] = w_m[:-1] + Pdifs
    Pedges[0] = w_m[0] - Pdifs[0]
    Pedges[-1] = w_m[-1] + Pdifs[-1]

    i_start = np.argwhere((edges[0] < Pedges))[0][0] - 1 #return the first starting index for the model wavelength edges array (Pedges)

    edges_i = 1
    left_weight = (Pedges[i_start + 1] - edges[0])/(Pedges[i_start + 1] - Pedges[i_start])

    for i in range(len_mod+1):

        if Pedges[i] > edges[edges_i]:
            right_weight = (edges[edges_i] - Pedges[i - 1])/(Pedges[i] - Pedges[i - 1])
            weights = ones[:(i - i_start)].copy()
            weights[0] = left_weight
            weights[-1] = right_weight

            out_flux[edges_i - 1] = np.average(f_m[i_start:i],weights=weights)

            edges_i += 1
            i_start = i - 1
            left_weight = 1. - right_weight
            if edges_i > len_TRES:
                break
    return out_flux

#f_grid = downsample(w, f_lam, grid)

#truncate to 5165 to 5190
#ind = (w > 5165) & (w < 5190)
#ind2 = (grid > 5165) & (grid < 5190)
#plt.plot(w[ind],f_lam[ind])
#plt.plot(grid[ind2], f_grid[ind2])
#plt.show()

grid = np.load("grid.npy")
f_grid = np.load("f_grid.npy")
#
####Take FFT of f_grid
#out = np.fft.fft(np.fft.fftshift(f_grid))
#N = len(f_grid)
#freqs = np.fft.fftfreq(N,d=2)
#print(freqs)
#
#plt.plot(np.real(freqs),out)
#plt.plot(freqs,np.imag(out))
#plt.show()

@np.vectorize
def gauss_taper(s,sigma=2.89):
    '''This is the FT of a gaussian w/ this sigma. Sigma in km/s'''
    return np.exp(-2 * np.pi**2 * sigma*2 * s**2)
#
#taper = gauss_taper(freqs)
#tout = out * taper
#blended = np.fft.fftshift(np.fft.ifft(tout))

def convolve_gauss(wl,fl,sigma=2.89,spacing=2.):
    ##Take FFT of f_grid
    out = np.fft.fft(np.fft.fftshift(fl))
    N = len(fl)
    freqs = np.fft.fftfreq(N,d=spacing) #2km/s spacing
    taper = gauss_taper(freqs,sigma)
    tout = out * taper
    blended = np.fft.fftshift(np.fft.ifft(tout))
    return np.absolute(blended) #remove tiny complex component

#ind = (grid > 5165) & (grid < 5190)
#
#plt.plot(grid[ind], f_grid[ind])
#plt.plot(grid[ind], np.abs(blended[ind]))
#plt.show()

#Now, do since interpolation to the TRES pixels on the blended spectrum

#Take TRES pixel, call that the center of sinc, then sample it at +/- the other pixels in the grid
def sinc_interpolate(wl_TRES):
    ind = np.argwhere(wl_TRES > grid)[-1][0]
    ind2 = ind + 1
    print(grid[ind])
    print(grid[ind2])
    frac = (wl_TRES - grid[ind])/(grid[ind2] - grid[ind])
    print(frac)
    spacing = 2 #km/s
    veloc_grid = np.arange(-48.,51,spacing) - frac * spacing
    print(veloc_grid)
    #convert wl spacing to velocity spacing
    sinc_pts = 0.5 * np.sinc(0.5 * veloc_grid)
    print(sinc_pts,trapz(sinc_pts,veloc_grid))
    print("Interpolated flux",np.sum(sinc_pts * f_grid[ind - 25: ind + 25]))
    print("Neighboring flux", f_grid[ind], f_grid[ind2])

#sinc_interpolate(6610.02)




def linear_interpolate():
    f = interp1d(grid,blended,kind='linear')
    return f(m.wls)

def grid_interp():
    return griddata(grid,blended,m.wls,method='linear')


#def G(s,vL,eps=0.6):
#    k = 2. * np.pi * s * vL
#    k2 = k**2
#    return 1./(vL * (1 - eps/3.)) * ( 2 * (1 - eps)/np.pi * (1.5708 * vL * hyp0f1(2, -0.25 * k2) + np.exp(-0.5j * np.pi)/s * (struve(1, -k) - struve(1,k))) + eps/(np.pi * s * k2) * (np.sin(k) - k * (np.cos(k) + 0.5 * k * np.sin(k))) + eps/2.)

def G(s,vL):
    '''vL in km/s. Gray pg 475'''
    ub = 2. * np.pi * vL * s
    return j1(ub)/ub - 3 * np.cos(ub)/(2 * ub**2) + 3. * np.sin(ub)/(2* ub**3)

def plot_gray():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ss = np.linspace(0.001,2,num=200)
    Gs1 = G(ss, 1.)
    Gs2 = G(ss, 2.)
    ax.plot(ss, Gs1)
    ax.plot(ss, Gs2)
    plt.show()

def main():
    #plt.plot(m.wls[0],np.load("mod_flux.npy")[0])
    #plt.plot(m.wls[0],linear_interpolate())
    #plt.show()
    #linear_interpolate() # linear interpolate is faster, 0.041
    #grid_interp() # 0.046
    plot_gray()


if __name__=="__main__":
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
