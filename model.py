import numpy as np
from echelle_io import rechellenpflat,load_masks
from scipy.interpolate import interp1d,NearestNDInterpolator
from scipy.integrate import trapz
from scipy.ndimage.filters import convolve
from scipy.optimize import leastsq,fmin
#from deredden import deredden
from numpy.polynomial import Chebyshev as Ch

'''
Coding convention:
    wl: refers to an individual 1D TRES wavelength array, shape = (2304,)
    fl: refers to an individual 1D TRES flux array, shape = (2304,)

    wls: referes to 2D TRES wavelength array, shape = (51, 2304)
    fls: referes to 2D TRES flux array, shape = (51, 2304)

    wlsz: refers to 2D TRES wavelength array, shifted in velocity, shape = (51, 2304)

    w: refers to individual 1D PHOENIX wavelength array, spacing 0.01A, shape = (large number,)
    f: refers to individual 1D PHOENIX flux array, shape = (large number,)

    fls: refers to 2D model flux array, after being downsampled to TRES resolution, shape = (51, 2304)

'''

##################################################
# Constants
##################################################
c_ang = 2.99792458e18 #A s^-1
c_kms = 2.99792458e5 #km s^-1
G = 6.67259e-8 #cm3 g-1 s-2
M_sun = 1.99e33 #g
R_sun = 6.955e10 #cm
pc = 3.0856776e18 #cm
AU = 1.4959787066e13 #cm

Mg = np.array([5168.7605, 5174.1251, 5185.0479])

#Load normalized order spectrum
wls, fls = rechellenpflat("GWOri_cf")

sigmas = np.load("sigmas.npy") #has shape (51, 2299), a sigma array for each order

masks = np.load("masks_array.npy")

#Load 3700 to 10000 ang wavelength vector
w_full = np.load("wave_trim.npy")

norder = len(wls)

#Numpy array of orders I want to use, indexed to 1
#good_orders = [i for i in range(5,18)] + [i for i in range(20,30)] + [i for i in range(31,37)] + [43,46]
#orders = np.array(good_orders) - 1 #index to 0
#orders = np.array([21,22,23])
orders = np.array([22])

#Truncate TRES to include only those orders
wls = wls[orders]
fls = fls[orders]
sigmas = sigmas[orders]
masks = masks[orders]


def load_flux(temp,logg):
    fname="HiResNpy/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.npy".format(temp=temp,logg=logg)
    print("Loaded " + fname)
    f = np.load(fname)
    return f

def flux_interpolator():
    points = np.loadtxt("param_grid_GWOri.txt")
    len_w = 716665
    fluxes = np.empty((len(points),len_w)) 
    for i in range(len(points)):
        fluxes[i] = load_flux(points[i][0],points[i][1])
    flux_intp = NearestNDInterpolator(points, fluxes)
    return flux_intp

flux = flux_interpolator()

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


def shift_TRES(vz):
    wlsz = shift_vz(wls,vz)
    return wlsz


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

ones = np.ones((10,))

def downsample5(w_m,f_m,w_TRES):
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

def test_downsample():
    wl,fl = np.loadtxt("GWOri_cn/23.txt",unpack=True)
    f_full = load_flux(5900,3.5)

    #Limit huge file to the necessary order. Even at 4000 ang, 1 angstrom corresponds to 75 km/s. Add in an extra 5 angstroms to be sure.
    ind = (w_full > (wl[0] - 5.)) & (w_full < (wl[-1] + 5.))
    w = w_full[ind]
    f = f_full[ind]

    downsample4(w,f,wl)

##################################################
# Model 
##################################################


def model(wlsz, temp, logg, vsini):
    '''Given parameters, return the model, exactly sliced to match the format of the echelle spectra in `efile`. `temp` is effective temperature of photosphere. vsini in km/s. vz is radial velocity, negative values imply blueshift. Assumes M, R are in solar units, and that d is in parsecs'''
    #M = M * M_sun #g
    #R = R * R_sun #cm
    #d = d * pc #cm

    #logg = np.log10(G * M / R**2)
    #flux_factor = R**2/d**2 #prefactor by which to multiply model flux (at surface of star) to get recieved TRES flux

    #Loads the ENTIRE spectrum, not limited to a specific order
    f_full = flux(temp, logg)

    model_flux = np.zeros_like(wlsz)
    #Cycle through all the orders in the echelle spectrum
    for i,wlz in enumerate(wlsz):
        #print("Processing order %s" % (orders[i]+1,))

        #Limit huge file to the necessary order. Even at 4000 ang, 1 angstrom corresponds to 75 km/s. Add in an extra 5 angstroms to be sure.
        ind = (w_full > (wlz[0] - 5.)) & (w_full < (wlz[-1] + 5.))
        w = w_full[ind]
        f = f_full[ind]

        #convolve with stellar broadening (sb)
        k = vsini_ang(np.mean(wlz),vsini) #stellar rotation kernel centered at order
        f_sb = convolve(f, k)

        dlam = w[1] - w[0] #spacing of model points for TRES resolution kernel

        #convolve with filter to resolution of TRES
        filt = gauss_series(dlam,lam0=np.mean(wlz))
        f_TRES = convolve(f_sb,filt)

        #downsample to TRES bins
        dsamp = downsample5(w, f_TRES, wlz)

        #redden spectrum
        #red = dsamp/deredden(wlz,Av,mags=False)

        model_flux[i] = dsamp

    #Only returns the fluxes, because the wlz is actually the TRES wavelength vector
    return model_flux

def data(coefs_arr, wls, fls):
    '''coeff is a (norders, npoly) shape array'''
    flsc = np.zeros_like(fls)
    for i,coefs in enumerate(coefs_arr):
        flsc[i] = Ch(coefs,domain=[wls[i][0],wls[i][-1]])(wls[i]) * fls[i]
    return flsc

def lnprob(p):
    '''p is the parameter vector, contains both theta_s and theta_n'''
    #print(p)
    temp, logg, vsini, vz = p[:4]
    if (logg < 0) or (logg > 6.0) or (vsini < 0) or (temp < 4000) or (temp > 6900):
        return -np.inf
    else:
        coefs = p[4:]
        #print(coefs)
        coefs_arr = coefs.reshape(len(orders),-1)

        wlsz = shift_TRES(vz) 

        flsc = data(coefs_arr, wlsz, fls)

        fs = model(wlsz, temp, logg, vsini)

        chi2 = np.sum((flsc - fs)**2/sigmas**2)
        return -0.5 * chi2 

def model_and_data(p):
    '''p is the parameter vector, contains both theta_s and theta_n'''
    #print(p)
    temp, logg, vsini, vz = p[:4]
    coefs = p[4:]
    #print(coefs)
    coefs_arr = coefs.reshape(len(orders),-1)

    wlsz = shift_TRES(vz) 

    flsc = data(coefs_arr, wlsz, fls)

    fs = model(wlsz, temp, logg, vsini)
    return [wlsz,flsc,fs]

def find_chebyshev(wl, f, fl, sigma):
    func = lambda p : chi(f*Ch(p,domain=[wl[0],wl[-1]])(wl),fl,sigma)
    ans = leastsq(func, np.zeros((150,)))[0]
    #print(ans)
    return ans
    
def global_chi2(model_flux):
    '''Given a model flux, do a global chi2 comparison to the TRES data.'''
    norders = len(model_flux)
    const_coeff = np.zeros((norders,150))
    chi2_list = np.zeros((norders,))
    chiR_list = np.zeros((norders,))
    for i in range(norders):
        print("Order: ", orders[i] + 1)
        f = model_flux[i]
        wl,fl = efile_z[orders[i]]
        sigma = sigmas[orders[i]]
        #mask all these values
        z = masks[orders[i]]
        wl,f,fl,sigma = wl[z],f[z],fl[z],sigma[z]
        #p = find_const(wl,f,fl,sigma)
        p = find_chebyshev(wl,f,fl,sigma)
        const_coeff[i] = p
        #chi2_list[i] = chi2(f/p,fl,sigma)
        chi2_list[i] = chi2(f*Ch(p,domain=[wl[0],wl[-1]])(wl),fl,sigma)
        print(chi2_list[i])
        chiR_list[i] = chi2_list[i]/len(wl)
        print(len(wl),chiR_list[i])
        #print()
    np.save("const_coeff.npy",const_coeff)
    np.save("chi2_list.npy", chi2_list)
    np.save("chiR_list.npy", chiR_list)
    print("Global chi^2", np.sum(chi2_list))

def main():
    print(lnprob(np.array([6000, 4.0, 40, 30, 1e25,0,0,0,0])))

    pass

if __name__=="__main__":
    main()
