import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Starfish.grid_tools import HDF5Interface
from Starfish.em_cov import Sigma

grid = HDF5Interface("../libraries/PHOENIX_SPEX_M.hdf5",
                 ranges={"temp":(2800, 3400), "logg":(4.5, 6.0), "Z":(-0.5, 1.0), "alpha":(0.0, 0.0)})

wl = grid.wl
ind = (wl > 20000) * (wl < 24000)
wl = wl[ind]
npix = len(wl)
m = len(grid.list_grid_points)
print("Using {} spectra with Npix = {}".format(m, npix))

fluxes = np.empty((m,len(wl)))
for i, spec in enumerate(grid.fluxes):
    fluxes[i,:] = spec[ind]

pca = PCA()
pca.fit(fluxes)
comp = pca.transform(fluxes)
components = pca.components_
mean = pca.mean_
print("Shape of PCA components {}".format(components.shape))

ncomp = 5
print("Keeping only the first {} components".format(ncomp))
pcomps = components[0:ncomp]

#Subtract the mean from all the fluxes
nfluxes = fluxes - mean

#Flatten into a large array for later use
orign = np.ravel(fluxes - mean)

#Load all the stellar parameters into an (m, 3) array
gparams = np.empty((m, 3))
for i, params in enumerate(grid.list_grid_points):
    gparams[i, :] = np.array([params["temp"], params["logg"], params["Z"]])

#Then standardize onto the interval 0 - 1
mins = np.min(gparams, axis=0)
maxs = np.max(gparams, axis=0)
gparams = (gparams - mins)/(maxs - mins)

def Phi():
    '''
    Use the ncomponents to create the large Phi matrix
    '''
    out = []
    #First, assemble the sub-\phis
    for component in pcomps:
        out.append(sp.kron(sp.eye(m, format="csc"), component[np.newaxis].T, format="csc"))
    #Then hstack these together to form \Phi
    return sp.hstack(out, format="csc")


#Create w
def get_w():
    out = []
    for i in range(ncomp):
        temp = np.sum(nfluxes * pcomps[i], axis=1)
        out.append(temp)
    w = np.hstack(out)[np.newaxis].T
    return w

def reconstruct_full(PHI, w):
    '''
    Using the large PHI parameter and the weights, reconstruct what the full
    grid would look like.
    '''
    recons = PHI.dot(w)
    recons = recons.reshape(m,-1) + mean

    recons = np.ravel(recons)
    return recons

def get_what(PHI):
    '''
    Reconstruct what?!?
    '''
    what = sp.linalg.inv(PHI.T.dot(PHI).tocsc()).dot(PHI.T).dot(orign)
    return what

def reconstruct(i, weights):
    '''
    Reconstruct a spectrum given some weights.
    '''
    #TODO

    f = np.empty((ncomp, len(wl)))
    for k in range(ncomp):
        f[k, :] = components[k] * weights[i,k]
    return mean + np.sum(f, axis=0)


def R(p0, p1, irhos):
    '''
    Autocorrelation function.

    p0, p1 : the two sets of parameters, each shape (nparams,)

    irhos : shape (nparams,)
    '''
    return np.prod(irhos**(4 * (p0 - p1)**2 ))

def Sigma_i(iprecision, irhos):
    '''
    Function to create the dense matrix Sigma, for a specific eigenspectra

    iprecision : scalar
    irhos : shape (nparams,)

    returns a matrix with shape (m, m)
    '''
    mat = sp.dok_matrix((m,m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            mat[i,j] = iprecision * R(gparams[i], gparams[j], irhos)
    return mat

PHI = Phi()
PP = PHI.T.dot(PHI).tocsc()

WHAT = get_what(PHI)

#The prior distributions that Habib et al. 2007 uses
a_w = 5.
b_w = 5.
a_rho_w = 1.0
b_rho_w = 0.1

a_P = 1.0
b_P = 0.0001

a_Pprime = a_P + m * (npix - ncomp)/2.

a1 = sp.linalg.inv(PP)
a2 = PHI.T.dot(orign)
a2.shape = (-1, 1)
middle = np.dot(orign,orign) - a2.T.dot(a1.dot(a2))

b_Pprime = b_P + 0.5 * middle

def lnprob(p):
    '''
    Calculate the LNPROB!!

    p will contain many parameters

    lambda_p : simulation precision (1,)
    lambda_w : weight precision (ncomp,)
    rho_w : (ncomp, 3)

    Unpack these parameters from the list.

    '''
    lambda_p = 10**p[0]
    lambda_w = 10**p[1:1+ncomp]
    rho_w = p[1+ncomp:]
    rho_w.shape = (ncomp, 3)

    # print("lambda_p : {}".format(lambda_p))
    # print("lambda_w : {}".format(lambda_w))
    # print("rho_w : {}".format(rho_w))

    if np.any((rho_w > 1.0) | (rho_w < 0.0)):
        return -np.inf

    inv = sp.linalg.inv(lambda_p * PP)
    bigsig = Sigma(gparams, lambda_w, rho_w)

    #sparse matrix
    comb = inv + bigsig

    sign, pref = np.linalg.slogdet(comb.todense())
    pref *= -0.5

    central = -0.5 * (WHAT.T.dot(sp.linalg.spsolve(inv, WHAT)))

    prior = (a_Pprime - 1) * np.log(lambda_p) - \
            b_Pprime*lambda_p + np.sum((a_P - 1.)*lambda_w - b_P*lambda_w) + \
            np.sum((b_rho_w - 1.) * np.log(1 - rho_w))

    return pref + central + prior


def sample_lnprob():
    import emcee

    ndim = (1 + ncomp + ncomp * 3)
    nwalkers = 2 * ndim

    #Designed to be a list of walker positions
    log_lambda_p = np.random.uniform(low=-10.2, high=-10.08, size=(1, nwalkers))
    log_lambda_w = np.random.uniform(low=-0.8, high=0.8, size=(ncomp, nwalkers))
    rho_w = np.random.uniform(low=0.001, high=0.99, size=(ncomp*3, nwalkers))
    p0 = np.vstack((log_lambda_p, log_lambda_w, rho_w)).T

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4)

    print("Running Sampler")
    pos, prob, state = sampler.run_mcmc(p0, 80)
    print("Burn-in complete")
    sampler.reset()
    sampler.run_mcmc(pos, 240)

    samples = sampler.flatchain
    np.save("samples.npy", samples)

    import triangle
    fig = triangle.corner(samples)
    fig.savefig("triangle.png")


class Emulator:
    '''
    Emulate spectra like a boss.
    '''
    def __init__(self, emulator_params):
        #Construct the emulator using the sampled parameters.

        #Emulator parameters include
        p = emulator_params
        self.lambda_p = 10**p[0]
        self.lambda_w = p[1:1+ncomp]
        self.rho_w = p[1+ncomp:]
        self.rho_w.shape = (ncomp, 3)


        pass

    def V12(self, params):
        '''
        Construct V12 using the anticipated set of parameters.
        '''
        pass