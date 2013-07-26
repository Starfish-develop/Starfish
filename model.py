import numpy as np
import matplotlib.pyplot as plt
from echelle_io import rechelletxt
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.ndimage.filters import convolve
from scipy.optimize import leastsq
import asciitable

c_ang = 2.99792458e18 #A s^-1
c_kms = 2.99792458e5 #km s^-1

Mg = np.array([5168.7605, 5174.1251, 5185.0479])

#Load real data
#wl,fl_r = np.loadtxt("GWOri_c/23.txt",unpack=True)
#Load normalized spectrum
wl,fl = np.loadtxt("GWOri_cn/23.txt",unpack=True)
#wl,fl = np.loadtxt("GWOri_cn/25.txt",unpack=True)
#Create continuum
#cont = fl_r/fl

efile = rechelletxt("GWOri_c")
efile_n = rechelletxt("GWOri_cn")
order,sigma_norm = np.loadtxt("sigmas.dat",unpack=True)

#use order 36 for all testing
#wl,fl = efile[22]

w = np.load("wave_trim.npy")

#Limit huge file to necessary range
ind = (w > (wl[0] - 10.)) & (w < (wl[-1] + 10))
w = w[ind]

def load_flux(temp,logg):
    fname="HiResNpy/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.npy".format(temp=temp,logg=logg)
    print("Loaded " + fname)
    f = np.load(fname)
    return f[ind]

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

##################################################
#Plotting Checks
##################################################

def plot_check():
    ys = np.ones((11,))
    plt.plot(wl[-10:],ys[-10:],"o")
    plt.plot(edges[-11:],ys,"o")
    plt.show()


def model(temp, vz=-30, scale=1.0, vsini=40., logg=4.5):
    '''Given parameters, return the model. `temp` is effective temperature of photosphere. vsini in km/s. vz is radial velocity, negative values imply blueshift.'''
    f = load_flux(temp,logg)

    #convolve with stellar broadening (sb)
    k = vsini_ang(np.mean(wl),vsini)
    f_sb = convolve(f, k)

    wvz = shift_vz(w,vz) #shifted wavelengths

    dlam = wvz[1] - wvz[0]
    #convolve with filter to resolution of TRES
    filt = gauss_series(dlam,lam0=np.mean(wvz))
    f_TRES = convolve(f_sb,filt)

    #downsample to TRES bins,multiply by prefactor
    dsamp = downsample(wvz, f_TRES, wl)
    return scale*dsamp 

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

global sigma
sigma = 0.04 /0.06 * np.load("sigma.npy")

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
    #compare_sample()
    #run_chain()
    print(calc_chi2(5900,3.5))
    print(calc_chi2(5900,4.0))
    pass

if __name__=="__main__":
    main()
