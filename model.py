import numpy as np
import matplotlib.pyplot as plt
from echelle_io import rechelletxt,load_masks
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.ndimage.filters import convolve
from scipy.optimize import leastsq
import asciitable

c_ang = 2.99792458e18 #A s^-1
c_kms = 2.99792458e5 #km s^-1

Mg = np.array([5168.7605, 5174.1251, 5185.0479])

#Load normalized order spectrum
efile_n = rechelletxt("GWOri_cn") #has structure len = 51, for each order: [wl,fl]
sigmas = np.load("sigmas.npy") #has shape (51, 2299), a sigma array for each order

#Load masking region
masks = load_masks()

#Load 3700 to 10000 ang wavelength vector
w_full = np.load("wave_trim.npy")

norder = len(efile_n)

def load_flux(temp,logg):
    fname="HiResNpy/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.npy".format(temp=temp,logg=logg)
    print("Loaded " + fname)
    f = np.load(fname)
    return f


##################################################
#Data processing steps
##################################################


##################################################
#Stellar Broadening
##################################################

def karray(center, width, res):
    '''Creates a kernel array with an odd number of elements, the central element centered at `center` and spanning out to +/- width in steps of resolution. Works similar to arange in that it may or may not get all the way to the edge.'''
    neg = np.arange(center - res, center-width, -res)[::-1]
    pos = np.arange(center, center+width, res)
    kar = np.concatenate([neg,pos])
    return kar

@np.vectorize
def vsini_ang(lam0,vsini,dlam=0.01,epsilon=0.6):
    '''vsini in km/s. Epsilon is the limb-darkening coefficient, typically 0.6. Formulation uses Eqn 18.14 from Gray, The Observation and Analysis of Stellar Photospheres, 3rd Edition.'''
    lamL = vsini * 1e13 * lam0/c_ang
    lam = karray(0,lamL, dlam)
    c1 = 2.*(1-epsilon)/(np.pi * lamL * (1 - epsilon/3.))
    c2 = epsilon/(2. * lamL * (1 - epsilon/3.))
    series = c1 * np.sqrt(1. - (lam/lamL)**2) + c2 * (1. - (lam/lamL)**2)**2
    return series/np.sum(series)


##################################################
#Radial Velocity Shift
##################################################
@np.vectorize
def shift_vz(lam_source, vz):
    '''Given the source wavelength, lam_sounce, return the observed wavelength based upon a radial velocity vz in km/s. Negative velocities are towards the observer (blueshift).'''
    lam_observe = lam_source * np.sqrt((c_kms + vz)/(c_kms - vz))
    return lam_observe


##################################################
#TRES Instrument Broadening
##################################################
@np.vectorize
def gauss_kernel(dlam,V=6.8,lam0=6500.):
    '''V is the FWHM in km/s. lam0 is the central wavelength in A'''
    sigma = V/2.355 * 1e13 #A/s
    return c_ang/lam0 * 1/(sigma * np.sqrt(2*np.pi)) * np.exp( - (c_ang*dlam/lam0)**2/(2. * sigma**2))

def gauss_series(dlam,V=6.5,lam0=6500.):
    '''sampled from +/- 3sigma at dlam. V is the FWHM in km/s'''
    sigma_l = V/(2.355 * 3e5) * 6500. #A
    wl = karray(0., 4*sigma_l,dlam)
    gk = gauss_kernel(wl)
    return gk/np.sum(gk)


##################################################
#Downsample to TRES bins 
##################################################


def downsample(w_m,f_m,w_TRES):
    '''Given a model wavelength and flux (w_m, f_m) and the instrument wavelength (w_TRES), downsample the model to exactly match the TRES wavelength bins. '''
    spec_interp = interp1d(w_m,f_m,kind="linear")

    @np.vectorize
    def avg_bin(bin0,bin1):
        mdl_ind = (w_m > bin0) & (w_m < bin1)
        wave = np.empty((np.sum(mdl_ind)+2,))
        flux = np.empty((np.sum(mdl_ind)+2,))
        wave[0] = bin0
        wave[-1] = bin1
        flux[0] = spec_interp(bin0)
        flux[-1] = spec_interp(bin1)
        wave[1:-1] = w_m[mdl_ind]
        flux[1:-1] = f_m[mdl_ind]
        return trapz(flux,wave)/(bin1-bin0)

    #Determine the bin edges
    edges = np.empty((len(w_TRES)+1,))
    difs = np.diff(w_TRES)/2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]
    b0s = edges[:-1]
    b1s = edges[1:]

    samp = avg_bin(b0s,b1s)
    return(samp)

def downsample2(w_m,f_m,w_TRES):
    '''Given a model wavelength and flux (w_m, f_m) and the instrument wavelength (w_TRES), downsample the model to exactly match the TRES wavelength bins. Try this without calling the interpolation routine.'''

    @np.vectorize
    def avg_bin(bin0,bin1):
        mdl_ind = (w_m > bin0) & (w_m < bin1)
        length = np.sum(mdl_ind)+2
        wave = np.empty((length,))
        flux = np.empty((length,))
        wave[0] = bin0
        wave[-1] = bin1
        wave[1:-1] = w_m[mdl_ind]
        flux[1:-1] = f_m[mdl_ind]
        flux[0] = flux[1]
        flux[-1] = flux[-2]
        return trapz(flux,wave)/(bin1-bin0)

    #Determine the bin edges
    edges = np.empty((len(w_TRES)+1,))
    difs = np.diff(w_TRES)/2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]
    b0s = edges[:-1]
    b1s = edges[1:]

    return avg_bin(b0s,b1s)

def downsample3(w_m,f_m,w_TRES):
    '''Given a model wavelength and flux (w_m, f_m) and the instrument wavelength (w_TRES), downsample the model to exactly match the TRES wavelength bins. Try this only by averaging.'''

    #More time could be saved by splitting up the original array into averageable chunks.

    @np.vectorize
    def avg_bin(bin0,bin1):
        return np.average(f_m[(w_m > bin0) & (w_m < bin1)])

    #Determine the bin edges
    edges = np.empty((len(w_TRES)+1,))
    difs = np.diff(w_TRES)/2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]
    b0s = edges[:-1]
    b1s = edges[1:]

    return avg_bin(b0s,b1s)

def downsample4(w_m,f_m,w_TRES):

    out_flux = np.zeros_like(w_TRES)
    len_mod = len(w_m)

    #Determine the bin edges
    len_TRES = len(w_TRES)
    edges = np.empty((len_TRES+1,))
    difs = np.diff(w_TRES)/2.
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

def test_downsample():
    wl,fl = np.loadtxt("GWOri_cn/23.txt",unpack=True)
    f_full = load_flux(5900,3.5)

    #Limit huge file to the necessary order. Even at 4000 ang, 1 angstrom corresponds to 75 km/s. Add in an extra 5 angstroms to be sure.
    ind = (w_full > (wl[0] - 5.)) & (w_full < (wl[-1] + 5.))
    w = w_full[ind]
    f = f_full[ind]

    downsample4(w,f,wl)

##################################################
#Plotting Checks
##################################################

def plot_check():
    ys = np.ones((11,))
    plt.plot(wl[-10:],ys[-10:],"o")
    plt.plot(edges[-11:],ys,"o")
    plt.show()

def model(temp, vz=-30, vsini=40., logg=4.5):
    '''Given parameters, return the model, exactly sliced to match the format of the echelle spectra in `efile`. `temp` is effective temperature of photosphere. vsini in km/s. vz is radial velocity, negative values imply blueshift.'''
    #Loads the ENTIRE spectrum, not limited to a specific order
    f_full = load_flux(temp,logg)

    model_flux = []
    #Cycle through all the orders in the echelle spectrum
    for i in range(norder):
        print("Processing order %s" % (i+1,))
        wl,fl = efile_n[i]

        #Limit huge file to the necessary order. Even at 4000 ang, 1 angstrom corresponds to 75 km/s. Add in an extra 5 angstroms to be sure.
        ind = (w_full > (wl[0] - 5.)) & (w_full < (wl[-1] + 5.))
        w = w_full[ind]
        f = f_full[ind]

        #convolve with stellar broadening (sb)
        k = vsini_ang(np.mean(wl),vsini) #stellar rotation kernel centered at order
        f_sb = convolve(f, k)

        #wvz = shift_vz(w,vz) #shifted wavelengths due to radial velocity

        #dlam = wvz[1] - wvz[0] #spacing of shifted wavelengths necessary for TRES resolution kernel
        dlam = w[1] - w[0]

        #convolve with filter to resolution of TRES
        #filt = gauss_series(dlam,lam0=np.mean(wvz))
        filt = gauss_series(dlam,lam0=np.mean(w))
        f_TRES = convolve(f_sb,filt)

        #downsample to TRES bins
        #dsamp = downsample4(wvz, f_TRES, wl)
        dsamp = downsample4(w, f_TRES, wl)

        model_flux.append(dsamp)

    #Only returns the fluxes, because the wl is actually the TRES wavelength vector
    return model_flux

def find_pref(flux):
    func = lambda x : chi(x*flux)
    return leastsq(func, 1e-15)[0][0]

@np.vectorize
def calc_chi2(temp,logg):
    raw_flux = model(temp,logg=logg)
    pref = find_pref(raw_flux)
    return chi2(pref * raw_flux)

    
def model2(scale):
    return scale*gmod 

def chi(flux):
    #f = model(*p)
    val = np.sum((fl - flux)/sigma)
    return val

def chi2(flux):
    val = np.sum((fl - flux)**2/sigma**2)
    return val

def r(p_cur, p_old):
    return np.exp(-(chi2(p_cur) - chi2(p_old))/2.)

#def posterior(p):
#    return likelihood(p)*prior(p)
#
#def r(p_cur,p_old):
#    return posterior(p_cur)/posterior(p_old)

#global p_old
#p_old = np.array([8.04e-16,0.335])

def run_chain():
    sequences = []
    for j in range(10000):
        print(j)
        global p_old
        accept = False
        p_jump = np.array([np.random.normal(scale=2e-18), 
            np.random.normal(scale=2e-3)])

        p_new = p_old + p_jump
        ratio = r(p_new, p_old)
        print("Ratio ",ratio)
        if ratio >= 1.0:
            p_old = p_new
            accept = True
        else:
            u = np.random.uniform()
            if ratio >= u:
                p_old = p_new
                accept = True
        scale, pedestal = p_old
        sequences.append([j,ratio,accept,scale,pedestal])

    asciitable.write(sequences,"run_0.dat",names=["j","ratio","accept","scale","pedestal"])

def main():
    mod = model(5900)
    #test_downsample()
    #compare_sample()
    #run_chain()
    #print(calc_chi2(5900,3.5))
    #print(calc_chi2(5900,4.0))

    pass

if __name__=="__main__":
    main()
