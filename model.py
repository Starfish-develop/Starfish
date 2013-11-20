import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator,InterpolatedUnivariateSpline
from scipy.ndimage.filters import convolve
from scipy.special import j1
from numpy.polynomial import Chebyshev as Ch
import h5py
import yaml
import gc
import sys
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import pyfftw

if len(sys.argv) > 1:
    confname= sys.argv[1]
else:
    confname = 'config.yaml'
f = open(confname)
config = yaml.load(f)
f.close()

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

T_points = np.array(
    [2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 4000, 4100, 4200,
     4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100,
     6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000,
     9200, 9400, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 11200, 11400, 11600, 11800, 12000])
logg_points = np.arange(0.0, 6.1, 0.5)
Z_points = np.array([-0.5, 0.0, 0.5])

#Limit grid size to relevant region
grid_params = config['grid_params']

T_low, T_high = grid_params['temp_range']
T_ind = (T_points > T_low) & (T_points < T_high)
T_points = T_points[T_ind]
T_arg = np.where(T_ind)[0]

g_low, g_high = grid_params['logg_range']
logg_ind = (logg_points > g_low) & (logg_points < g_high)
logg_points = logg_points[logg_ind]
logg_arg = np.where(logg_ind)[0]

Z_low, Z_high = grid_params['Z_range']
#Z_points = Z_points[(Z_points > Z_low) & (Z_points < Z_high)]
#print("Limiting PHOENIX grid to temp: ", T_points, " logg: ", logg_points, " Z: ", Z_points)

base = 'data/' + config['dataset']
wls = np.load(base + ".wls.npy")
fls = np.load(base + ".fls.npy")
#fls = np.load("fls_fake.npy")
sigmas = np.load(base + ".sigma.npy") #3.8 gives chi^2_red = 1
#sigmas = np.load('sigmas_fake.npy')
masks = np.load(base + ".mask.npy")


orders = np.array(config['orders'])
norder = len(orders)

#Truncate TRES to include only those orders
wls = wls[orders]
fls = fls[orders]
sigmas = sigmas[orders]
masks = masks[orders]

sigmac = config['sigmac']
sigmac0 = config['sigmac0']

len_wl = len(wls[0])

wave_grid = np.load("wave_grid_2kms.npy")

wl_buffer = 5.0 #Angstroms on either side, to account for velocity shifts
wl_min = wls[0,0] - wl_buffer
wl_max = wls[-1,-1] + wl_buffer

#####
# Truncate wave_grid and red_grid to include only the regions necessary for fitting orders.
# But do this so that it is a power of 2
#####

len_wg = len(wave_grid)

len_data = np.sum((wave_grid > wl_min) & (wave_grid < wl_max))

if len_data < (len_wg/16):
    chunk = int(len_wg/16)
elif len_data < (len_wg/8):
    chunk = int(len_wg/8)
elif len_data < (len_wg/4):
    chunk = int(len_wg/4)
elif len_data < (len_wg/2):
    chunk = int(len_wg/2)
else:
    #use the  full spectrum
    chunk = len_wg
    ind = np.ones_like(wave_grid, dtype='bool')

if chunk < len_wg:
    ind_wg = np.arange(len_wg)
    #Determine if the data region is closer to the start or end of the wave_grid
    if (wl_min - wave_grid[0]) < (wave_grid[-1] - wl_max):
        #the data region is closer to the start
        #find starting index
        #start at index corresponding to wl_min and go chunk forward
        start_ind = np.argwhere(wave_grid > wl_min)[0][0]
        end_ind = start_ind + chunk
        ind = (ind_wg >= start_ind) & (ind_wg < end_ind)

    else:
        # the data region is closer to the finish
        # start at index corresponding to wl_max and go chunk backward
        end_ind = np.argwhere(wave_grid < wl_max)[-1][0]
        start_ind = end_ind - chunk
        ind = (ind_wg > start_ind) & (ind_wg <= end_ind)

#ind = (wave_grid > wl_min) & (wave_grid < wl_max)

wave_grid = wave_grid[ind]
red_grid = np.load('red_grid.npy')[ind]

def flux_interpolator_hdf5():
    #load hdf5 file of PHOENIX grid 
    fhdf5 = h5py.File('LIB_2kms.hdf5', 'r')
    LIB = fhdf5['LIB']
    param_combos = []
    var_combos = []
    for ti in range(len(T_points)):
        for li in range(len(logg_points)):
            for z, Z in enumerate(Z_points):
                param_combos.append([T_arg[ti], logg_arg[li], z])
                var_combos.append([T_points[ti], logg_points[li], Z])
    #print(param_combos)
    num_spec = len(param_combos)
    points = np.array(var_combos)
    #print(points)
    fluxes = np.empty((num_spec, len(wave_grid)))
    for i in range(num_spec):
        t, l, z = param_combos[i]
        fluxes[i] = LIB[t, l, z][ind]
    flux_intp = LinearNDInterpolator(points, fluxes, fill_value=1.)
    #print("Loaded HDF5 interpolator")
    fhdf5.close()
    del fluxes
    gc.collect()
    return flux_intp

flux = flux_interpolator_hdf5()

##################################################
#Data processing steps
##################################################


##################################################
#Stellar Broadening
##################################################

def karray(center, width, res):
    '''Creates a kernel array with an odd number of elements, the central element centered at `center` and spanning
    out to +/- width in steps of resolution. Works similar to arange in that it may or may not get all the way to the
    edge.'''
    neg = np.arange(center - res, center - width, -res)[::-1]
    pos = np.arange(center, center + width, res)
    kar = np.concatenate([neg, pos])
    return kar


@np.vectorize
def vsini_ang(lam0, vsini, dlam=0.01, epsilon=0.6):
    '''vsini in km/s. Epsilon is the limb-darkening coefficient, typically 0.6. Formulation uses Eqn 18.14 from Gray,
    The Observation and Analysis of Stellar Photospheres, 3rd Edition.'''
    lamL = vsini * 1e13 * lam0 / c_ang
    lam = karray(0, lamL, dlam)
    c1 = 2. * (1 - epsilon) / (np.pi * lamL * (1 - epsilon / 3.))
    c2 = epsilon / (2. * lamL * (1 - epsilon / 3.))
    series = c1 * np.sqrt(1. - (lam / lamL) ** 2) + c2 * (1. - (lam / lamL) ** 2) ** 2
    return series / np.sum(series)


@np.vectorize
def G(s, vL):
    '''vL in km/s. Gray pg 475'''
    if s != 0:
        ub = 2. * np.pi * vL * s
        return j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
    else:
        return 1.


##################################################
#Radial Velocity Shift
##################################################
@np.vectorize
def shift_vz(lam_source, vz):
    '''Given the source wavelength, lam_sounce, return the observed wavelength based upon a radial velocity vz in
    km/s. Negative velocities are towards the observer (blueshift).'''
    lam_observe = lam_source * np.sqrt((c_kms + vz) / (c_kms - vz))
    #TODO: when applied to full spectrum, this sqrt is repeated
    return lam_observe


def shift_TRES(vz, wls=wls):
    wlsz = shift_vz(wls, vz)
    return wlsz


##################################################
#TRES Instrument Broadening
##################################################
@np.vectorize
def gauss_kernel(dlam, V=6.8, lam0=6500.):
    '''V is the FWHM in km/s. lam0 is the central wavelength in A'''
    sigma = V / 2.355 * 1e13 #A/s
    return c_ang / lam0 * 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (c_ang * dlam / lam0) ** 2 / (2. * sigma ** 2))


def gauss_series(dlam, V=6.8, lam0=6500.):
    '''sampled from +/- 3sigma at dlam. V is the FWHM in km/s'''
    sigma_l = V / (2.355 * c_kms) * lam0 #A
    wl = karray(0., 6 * sigma_l, dlam)
    gk = gauss_kernel(wl)
    return gk / np.sum(gk)


##################################################
#Downsample to TRES bins 
##################################################

ones = np.ones((10,))


def downsample(w_m, f_m, w_TRES):
    out_flux = np.zeros_like(w_TRES)
    len_mod = len(w_m)

    #Determine the TRES bin edges
    len_TRES = len(w_TRES)
    edges = np.empty((len_TRES + 1,))
    difs = np.diff(w_TRES) / 2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]

    #Determine PHOENIX bin edges
    Pedges = np.empty((len_mod + 1,))
    Pdifs = np.diff(w_m) / 2.
    Pedges[1:-1] = w_m[:-1] + Pdifs
    Pedges[0] = w_m[0] - Pdifs[0]
    Pedges[-1] = w_m[-1] + Pdifs[-1]

    i_start = np.argwhere((edges[0] < Pedges))[0][
                  0] - 1 #return the first starting index for the model wavelength edges array (Pedges)

    edges_i = 1
    left_weight = (Pedges[i_start + 1] - edges[0]) / (Pedges[i_start + 1] - Pedges[i_start])

    for i in range(len_mod + 1):

        if Pedges[i] > edges[edges_i]:
            right_weight = (edges[edges_i] - Pedges[i - 1]) / (Pedges[i] - Pedges[i - 1])
            weights = ones[:(i - i_start)].copy()
            weights[0] = left_weight
            weights[-1] = right_weight

            out_flux[edges_i - 1] = np.average(f_m[i_start:i], weights=weights)

            edges_i += 1
            i_start = i - 1
            left_weight = 1. - right_weight
            if edges_i > len_TRES:
                break
    return out_flux


##################################################
# Model 
##################################################

def old_model(wlsz, temp, logg, vsini, flux_factor):
    '''Given parameters, return the model, exactly sliced to match the format of the echelle spectra in `efile`.
    `temp` is effective temperature of photosphere. vsini in km/s. vz is radial velocity, negative values imply
    blueshift. Assumes M, R are in solar units, and that d is in parsecs'''
    #wlsz has length norders

    #M = M * M_sun #g
    #R = R * R_sun #cm
    #d = d * pc #cm

    #logg = np.log10(G * M / R**2)
    #flux_factor = R**2/d**2 #prefactor by which to multiply model flux (at surface of star) to get recieved TRES flux

    #Loads the ENTIRE spectrum, not limited to a specific order
    f_full = flux_factor * flux(temp, logg)

    model_flux = np.zeros_like(wlsz)
    #Cycle through all the orders in the echelle spectrum
    #might be able to np.vectorize this
    for i, wlz in enumerate(wlsz):
        #print("Processing order %s" % (orders[i]+1,))

        #Limit huge file to the necessary order. Even at 4000 ang, 1 angstrom corresponds to 75 km/s. Add in an extra
        # 5 angstroms to be sure.
        ind = (w_full > (wlz[0] - 5.)) & (w_full < (wlz[-1] + 5.))
        w = w_full[ind]
        f = f_full[ind]

        #convolve with stellar broadening (sb)
        k = vsini_ang(np.mean(wlz), vsini) #stellar rotation kernel centered at order
        f_sb = convolve(f, k)

        dlam = w[1] - w[0] #spacing of model points for TRES resolution kernel

        #convolve with filter to resolution of TRES
        filt = gauss_series(dlam, lam0=np.mean(wlz))
        f_TRES = convolve(f_sb, filt)

        #downsample to TRES bins
        dsamp = downsample(w, f_TRES, wlz)
        #red = dsamp/deredden(wlz,Av,mags=False)

        #If the redenning interpolation is taking a while here, we could save the points for a given redenning and
        # simply multiply each again

        model_flux[i] = dsamp

    #Only returns the fluxes, because the wlz is actually the TRES wavelength vector
    return model_flux

#Constant for all models
ss = np.fft.fftfreq(len(wave_grid), d=2.) #2km/s spacing for wave_grid

f_full = pyfftw.n_byte_align_empty(chunk, 16, 'complex128')
FF = pyfftw.n_byte_align_empty(chunk, 16, 'complex128')
blended = pyfftw.n_byte_align_empty(chunk, 16, 'complex128')
blended_real = pyfftw.n_byte_align_empty(chunk, 16, "float64")
fft_object = pyfftw.FFTW(f_full, FF)
ifft_object = pyfftw.FFTW(FF, blended, direction='FFTW_BACKWARD')


def model(wlsz, temp, logg, Z, vsini, Av, flux_factor):
    '''Given parameters, return the model, exactly sliced to match the format of the echelle spectra in `efile`.
    `temp` is effective temperature of photosphere. vsini in km/s. vz is radial velocity, negative values imply
    blueshift. Assumes M, R are in solar units, and that d is in parsecs'''
    #wlsz has length norders

    #M = M * M_sun #g
    #R = R * R_sun #cm
    #d = d * pc #cm

    #logg = np.log10(G * M / R**2)
    #flux_factor = R**2/d**2 #prefactor by which to multiply model flux (at surface of star) to get recieved TRES flux

    #Loads the ENTIRE spectrum, not limited to a specific order
    f_full[:] = flux_factor * flux(temp, logg, Z)
    #f_full = flux_factor * flux(temp, logg, Z)

    #Take FFT of f_grid
    #FF = fft(f_full)
    fft_object()

    ss[0] = 0.01 #junk so we don't get a divide by zero error
    ub = 2. * np.pi * vsini * ss
    sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
    #set zeroth frequency to 1 separately (DC term)
    sb[0] = 1.

    FF[:] *= sb #institute velocity taper
    #FF *= sb

    #do ifft
    ifft_object()
    #blended_real = np.abs(ifft(FF))

    blended_real[:] = np.abs(blended) #remove tiny complex component

    #redden spectrum
    red = blended_real / 10**(0.4 * Av * red_grid)
    #red = blended_real

    #do synthetic photometry to compare to points

    f = InterpolatedUnivariateSpline(wave_grid, red)
    fresult = f(wlsz.flatten()) #do spline interpolation to TRES pixels
    result = np.reshape(fresult,(norder,-1))
    del f
    gc.collect() #necessary to prevent memory leak!
    return result

def model_p(p):
    '''Post processing routine that can take all parameter values and return the model.
    Actual sampling does not require the use of this method since it is slow.'''
    temp, logg, Z, vsini, vz, Av, flux_factor = p[:config['nparams']]

    wlsz = wls * np.sqrt((c_kms + vz) / (c_kms - vz))
    fmods = model(wlsz, temp, logg, Z, vsini, Av, flux_factor)

    coefs = p[config['nparams']:]
    # reshape to (norders, 4)
    coefs_arr = coefs.reshape(len(orders), -1)
    c0s = coefs_arr[:,0] #length norders
    cns = coefs_arr[:,1:] #shape (norders, 3)

    #now create polynomials for each order, and multiply through fls
    #print("c0s.shape", c0s.shape)
    #print("cns.shape", cns.shape)
    #print("T.shape", T.shape)

    Tc = np.einsum("jk,ij->ik", T,cns)
    #print("Tc.shape",Tc.shape)
    k = np.einsum("i,ij->ij",c0s, 1 + Tc)
    #print("k.shape",k.shape)
    #print("fmods.shape",fmods.shape)
    refluxed = k * fmods
    return [k, refluxed]

def degrade_flux(wl, w, f_full):
    vsini = 40.
    #Limit huge file to the necessary order. Even at 4000 ang, 1 angstrom corresponds to 75 km/s. Add in an extra 5
    # angstroms to be sure.
    ind = (w_full > (wl[0] - 5.)) & (w_full < (wl[-1] + 5.))
    w = w_full[ind]
    f = f_full[ind]
    #convolve with stellar broadening (sb)
    k = vsini_ang(np.mean(wl), vsini) #stellar rotation kernel centered at order
    f_sb = convolve(f, k)

    dlam = w[1] - w[0] #spacing of model points for TRES resolution kernel

    #convolve with filter to resolution of TRES
    filt = gauss_series(dlam, lam0=np.mean(wl))
    f_TRES = convolve(f_sb, filt)

    #downsample to TRES bins
    dsamp = downsample(w, f_TRES, wl)

    return dsamp


def data(coefs_arr, wls, fls):
    '''coeff is a (norders, npoly) shape array'''
    flsc = np.zeros_like(fls)
    for i, coefs in enumerate(coefs_arr):
        #do this to keep constant fixed at 1
        flsc[i] = Ch(np.append([1],coefs),domain=[wls[i][0],wls[i][-1]])(wls[i]) * fls[i]
        #do this to allow tweaks to each order
        #flsc[i] = Ch(coefs, domain=[wls[i][0], wls[i][-1]])(wls[i]) * fls[i]
    return flsc

xs = np.arange(len_wl)
T0 = np.ones_like(xs)
Ch1 = Ch([0,1], domain=[0,len_wl-1])
T1 = Ch1(xs)
Ch2 = Ch([0,0,1],domain=[0,len_wl-1])
T2 = Ch2(xs)
Ch3 = Ch([0,0,0,1],domain=[0,len_wl-1])
T3 = Ch3(xs)

if (config['lnprob'] == "lnprob_gaussian") or (config['lnprob'] == 'lnprob_gaussian_marg'):
    T = np.array([T0,T1,T2,T3])
    TT = np.einsum("in,jn->ijn",T,T)
    mu = np.array([1,0,0,0])
    D = sigmac**(-2) * np.eye(4)
    Dmu = np.einsum("ij,j->j",D,mu)
    muDmu = np.einsum("j,j->",mu,Dmu)

if (config['lnprob'] == "lnprob_lognormal") or (config['lnprob'] == 'lnprob_lognormal_marg'):
    T = np.array([T1,T2,T3])
    TT = np.einsum("in,jn->ijn",T,T)
    mu = np.array([0,0,0])
    D = sigmac**(-2) * np.eye(3)
    Dmu = np.einsum("ij,j->j",D,mu)
    muDmu = np.einsum("j,j->",mu,Dmu)

############################################################
# Various lnprob functions
############################################################

def lnprob_gaussian_marg(p):
    '''New lnprob, no nuisance coeffs'''
    temp, logg, Z, vsini, vz, Av, flux_factor = p
    if (logg < g_low) or (logg > g_high) or (vsini < 0) or (temp < T_low) or \
            (temp > T_high) or (np.abs(Z) >= 0.5) or (Av < 0):
        return -np.inf
    else:
        #shift TRES wavelengths
        wlsz = wls * np.sqrt((c_kms + vz) / (c_kms - vz))
        fmods = model(wlsz, temp, logg, Z, vsini, Av, flux_factor)

        a= fmods**2/sigmas**2
        A = np.einsum("in,jkn->ijk",a,TT)
        Ap = A + D
        detA = np.array(list(map(np.linalg.det, Ap)))
        invA = np.array(list(map(np.linalg.inv, Ap)))

        b = fmods * fls / sigmas**2
        B = np.einsum("in,jn->ij",b,T)
        Bp = B + Dmu

        g = -0.5 * fls**2/sigmas**2
        G = np.einsum("ij->i",g)
        Gp = G - 0.5 * muDmu

        invAB = np.einsum("ijk,ik->ij",invA,Bp)
        BAB = np.einsum("ij,ij->i",Bp,invAB)

        lnp = np.sum(0.5 * np.log((2. * np.pi)**norder/detA) + 0.5 * BAB + Gp)

        return lnp

def lnprob_lognormal(p):
    '''Sample only in c0's  '''
    temp, logg, Z, vsini, vz, Av, flux_factor = p[:config['nparams']]
    if (logg < g_low) or (logg > g_high) or (vsini < 0) or (temp < T_low) or \
            (temp > T_high) or (np.abs(Z) >= 0.5) or (Av < 0):
        #if the call is outside of the loaded grid.
        return -np.inf
    else:
        #shift TRES wavelengths
        wlsz = wls * np.sqrt((c_kms + vz) / (c_kms - vz))
        fmods = model(wlsz, temp, logg, Z, vsini, Av, flux_factor)

        coefs = p[config['nparams']:]
        # reshape to (norders, 4)
        coefs_arr = coefs.reshape(len(orders), -1)
        c0s = coefs_arr[:,0] #length norders
        cns = coefs_arr[:,1:] #shape (norders, 3)
        #This does correctly unpack the coefficients into c0s, cns by order 11/17/13
        #print("c0s.shape", c0s.shape)
        #print("cns.shape", cns.shape)

        fdfmc0 = np.einsum('i,ij->ij', c0s, fmods * fls)
        fm2c2 = np.einsum("i,ij->ij", c0s**2,fmods**2)

        a= fm2c2/sigmas**2
        A = np.einsum("in,jkn->ijk",a,TT)
        #print("A.shape: ", A.shape)
        Ap = A + D

        b = (-fm2c2 + fdfmc0) / sigmas**2
        B = np.einsum("in,jn->ij",b,T)
        #print("B.shape", B.shape)
        Bp = B + Dmu

        g = -0.5/sigmas**2 * (fm2c2 - 2 * fdfmc0 + fls**2)
        G = np.einsum("ij->i",g)
        #print("G.shape", G.shape)
        Gp = G - 0.5 * muDmu

        Ac = np.einsum("ijk,ik->ij",Ap,cns)
        cAc = np.einsum("ij,ij->i",cns,Ac)
        Bc = np.einsum("ij,ij->i",Bp,cns)
        #print()
        #print("Ac.shape", Ac.shape)
        #print("cAc.shape", cAc.shape)
        #print("Bc.shape", Bc.shape)

        #we should have a cAc and a Bc for each order

        #to obtain the original, unnormalized, unmarginalized P given c
        #addition of lognormal prior

        #print()
        #print("-0.5 cAc: ",-0.5 * cAc)
        #print("Bc: ", Bc)
        #print("-0.5 cAc + Bc", -0.5 * cAc + Bc)
        #print("Gp: ", Gp)

        #Each variable should have two orders up to this point
        lnp = np.sum(-0.5 * cAc + Bc + Gp) + np.sum(np.log(1/(c0s * sigmac0 * np.sqrt(2. * np.pi))) - np.log(c0s)**2/(2 * sigmac0**2))

        return lnp

def lnprob_lognormal_marg(p):
    '''Sample only in c0's  '''
    temp, logg, Z, vsini, vz, Av, flux_factor = p[:config['nparams']]
    if (logg < g_low) or (logg > g_high) or (vsini < 0) or (temp < T_low) or (temp > T_high) \
        or (np.abs(Z) >= 0.5) or (Av < 0):
        #if the call is outside of the loaded grid.
        return -np.inf
    else:
        #shift TRES wavelengths
        wlsz = wls * np.sqrt((c_kms + vz) / (c_kms - vz))
        fmods = model(wlsz, temp, logg, Z, vsini, Av, flux_factor)

        c0s = p[config['nparams']:]

        fdfmc0 = np.einsum('i,ij->ij', c0s, fmods * fls)
        fm2c2 = np.einsum("i,ij->ij", c0s**2,fmods**2)

        a= fm2c2/sigmas**2
        A = np.einsum("in,jkn->ijk",a,TT)
        Ap = A + D
        detA = np.array(list(map(np.linalg.det, Ap)))
        invA = np.array(list(map(np.linalg.inv, Ap)))

        b = (-fm2c2 + fdfmc0) / sigmas**2
        B = np.einsum("in,jn->ij",b,T)
        Bp = B + Dmu

        g = -0.5/sigmas**2 * (fm2c2 - 2 * fdfmc0 + fls**2)
        G = np.einsum("ij->i",g)
        Gp = G - 0.5 * muDmu

        invAB = np.einsum("ijk,ik->ij",invA,Bp)
        BAB = np.einsum("ij,ij->i",Bp,invAB)

        #addition of lognormal prior
        lnp = np.sum(0.5 * np.log((2. * np.pi)**norder/detA) + 0.5 * BAB + Gp) + np.sum(np.log(1/(c0s * sigmac0 * np.sqrt(2. * np.pi))) - np.log(c0s)**2/(2 * sigmac0**2))
        return lnp

def lnprob_classic(p):
    '''p is the parameter vector, contains both theta_s and theta_n'''
    #print(p)
    temp, logg, Z, vsini, vz, Av, flux_factor = p[:config['nparams']]
    if (logg < g_low) or (logg > g_high) or (vsini < 0) or (temp < T_low) or \
            (temp > T_high) or (np.abs(Z) >= 0.5) or (Av < 0):
        return -np.inf
    else:
        coefs = p[config['nparams']:]
        #print(coefs)
        coefs_arr = coefs.reshape(len(orders), -1)
        print(coefs_arr)

        #shift TRES wavelengths
        wlsz = wls * np.sqrt((c_kms + vz) / (c_kms - vz))

        flsc = data(coefs_arr, wlsz, fls)

        fs = model(wlsz, temp, logg, Z, vsini, Av, flux_factor)

        chi2 = np.sum(((flsc - fs) / sigmas) ** 2)
        L = -0.5 * chi2
        #prior = - np.sum((coefs_arr[:,2])**2/0.1) - np.sum((coefs_arr[:,[1,3,4]]**2/0.01))
        prior = 0
        return L + prior

def generate_fake_data(SNR, temp, logg, Z, vsini, vz, Av, flux_factor):
    import os
    '''Generate an echelle-like spectrum to test method. SNR is quoted per-resolution element,
    and so is converted to per-pixel via the formula on 10/31/13. The sigma is created at the Poisson level only.'''
    SNR_pix = SNR/1.65 #convert to per-pixel for TRES

    #use LkCa15 wl grid, shifted
    LkCa15_wls = np.load('data/LkCa15/LkCa15_2013-10-13_09h37m31s_cb.flux.spec.wls.npy')

    #When running this, also need to set config['orders'] = all
    wlsz = shift_TRES(vz, wls=LkCa15_wls)
    fls_fake = model(wlsz, temp, logg, Z, vsini, Av, flux_factor) #create flux on a shifted grid
    sigmas = fls_fake/SNR_pix

    print("Generated data with SNR:{SNR:}, temp:{temp:}, logg:{logg:}, Z:{Z:}, "
        "vsini:{vsini:}, vz: {vz:}, Av:{Av:}, flux-factor:{ff:}".format(SNR=SNR, temp=temp,
        logg=logg, Z=Z, vsini=vsini, vz = vz, Av=Av, ff=flux_factor))

    #func = lambda x: np.random.normal(loc=0,scale=x)
    #noise = np.array(list(map(func,sigmas)))
    noise = np.random.normal(loc=0, scale=sigmas, size=fls_fake.shape)
    fls_noise = fls_fake + noise
    mask = np.ones_like(fls_noise, dtype='bool')

    basedir = 'data/Fake/%.0f/' % SNR #create in a subfolder that has the SNR labelled
    #Create necessary output directories using os.mkdir, if it does not exist
    if not os.path.exists(basedir):
        os.mkdir(basedir)
        print("Created output directory", basedir)
    else:
        print(basedir, "already exists, overwriting.")
    base = basedir + 'Fake'
    np.save(base + '.wls.npy',LkCa15_wls) #write original, unshifted grid
    np.save(base + '.fls.npy', fls_noise)
    np.save(base + '.true.fls.npy',fls_fake)
    np.save(base + '.sigma.npy',noise)
    np.save(base + '.mask.npy', mask)

def main():

    #fake_params = (5900., 3.5, 0.0, 5., 2.0, 0.0, 1e-10)
    #generate_fake_data(30., *fake_params)
    #generate_fake_data(50., *fake_params)
    #generate_fake_data(70., *fake_params)
    #generate_fake_data(100., *fake_params)

    #print(lnprob_lognormal_marg(np.array([5900., 3.5, 0.0, 5., 2.0, 0.0, 1e-10, 1.0])))
    #print(lnprob_lognormal(np.array([5900., 3.5, 0.0, 5., 2.0, 0.0, 1e-10, 1.0, 0.00, 0.00, 0.00, 1, 0, 0, 0, 1.0, 0, 0, 0])))
    print(lnprob_lognormal(np.array([5900., 3.5, 0, 5., 2.0, 0.0, 1e-10, 1.0, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1,1, 0.1, 0.1, 0.1])))
    #print(lnprob_lognormal(np.array([5900., 3.5, 0.0, 5., 2.1, 0.0, 1e-10, 1.0, 0.0, 0.0, 0.0, 1, 0, 0, 0])))
    #print(lnprob_lognormal(np.array([5900., 3.5, 0.0, 5., 3, 0.0, 1e-10, 1.0, 0.0, 0.0, 0.0, 1, 0, 0, 0])))
    #print(lnprob_lognormal(np.array([5900., 3.5, 0.0, 5., 50, 0.0, 1e-10, 1.0, 0.0, 0.0, 0.0, 1, 0, 0, 0])))

    #available in sample05
    #print(lnprob_lognormal(np.array([ 5.83723742e+03 , 3.42143796e+00  ,-3.40050296e-01  , 9.74222590e+00,
    #                                  6.67791281e+00 ,  4.39022745e-01 , 8.92392783e-11 ,  7.67364875e-01,
    #                                 -3.59142121e-01 ,-1.02655686e+00  , 9.32378921e-03 ,  1.82976554e+00,
    #                                 -2.01284175e-01 , -3.97143675e-02 , -2.53125531e-01])))

    #tweaking to see where the breaking point is
    #print(lnprob_lognormal(np.array([ 5900 , 3.5  , 0.0 , 5, 2 ,  4.39022745e-01 , 8.92392783e-11 ,  7.67364875e-01,
    #                                  -3.59142121e-01 ,-1.02655686e+00  , 9.32378921e-03 ,  1.82976554e+00,
    #                                  -2.01284175e-01 , -3.97143675e-02 , -2.53125531e-01])))
    #print(lnprob_lognormal(np.array([ 5900 , 3.5  , 0.0 , 5, 6 ,  4.39022745e-01 , 8.92392783e-11 ,  7.67364875e-01,
    #                                  -3.59142121e-01 ,-1.02655686e+00  , 9.32378921e-03 ,  1.82976554e+00,
    #                                  -2.01284175e-01 , -3.97143675e-02 , -2.53125531e-01, 1.82976554e+00,
    #                                  -2.01284175e-01 , -3.97143675e-02 , -2.53125531e-01, 1.82976554e+00,
    #                                  -2.01284175e-01 , -3.97143675e-02 , -2.53125531e-01, 1.82976554e+00,
    #                                  -2.01284175e-01 , -3.97143675e-02 , -2.53125531e-01])))
    #model_p(np.array([5900., 3.5, -0.45, 5., 2.0, 0.0, 1e-10, 1.0, 0.0, 0.0, 0.0, 1, 0, 0, 0]))

    pass


if __name__ == "__main__":
    main()
