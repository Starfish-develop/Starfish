import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Starfish.grid_tools import HDF5Interface
import Starfish.em_cov as em

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

#Subtract the mean from all of the fluxes.
flux_mean = np.average(fluxes, axis=0)
fluxes = fluxes - flux_mean

#Scale each spectrum by the variance.
std_flux = np.std(fluxes)
fluxes /= std_flux

pca = PCA()
pca.fit(fluxes)
comp = pca.transform(fluxes)
components = pca.components_
mean = pca.mean_
print("Shape of PCA components {}".format(components.shape))

# import matplotlib.pyplot as plt
# plt.plot(wl, flux_mean)
# plt.show()
#
# plt.plot(wl, mean)
# plt.show()
# import sys
# sys.exit()

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
sparams = (gparams - mins)/(maxs - mins)

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


PHI = Phi()
PP = PHI.T.dot(PHI).tocsc()
PP_inv = sp.linalg.inv(PP)

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

    if np.any((rho_w > 1.0) | (rho_w < 0.0)):
        return -np.inf

    inv = PP_inv/lambda_p
    bigsig = em.Sigma(sparams, lambda_w, rho_w)

    #sparse matrix
    comb = inv + bigsig

    sign, pref = np.linalg.slogdet(comb.todense())
    pref *= -0.5

    central = -0.5 * (WHAT.T.dot(sp.linalg.spsolve(comb, WHAT)))

    prior = (a_Pprime - 1) * np.log(lambda_p) - \
            b_Pprime*lambda_p + np.sum((a_P - 1.)*lambda_w - b_P*lambda_w) + \
            np.sum((b_rho_w - 1.) * np.log(1 - rho_w))

    return pref + central #+ prior


def test_lnprob():
    pars = np.concatenate((np.array([-10.]), np.random.uniform(size=(ncomp,)), np.random.uniform(size=(ncomp*3,))))
    lnprob(pars)

def sample_lnprob():
    import emcee

    ndim = (1 + ncomp + ncomp * 3)
    nwalkers = 4 * ndim
    print("using {} walkers".format(nwalkers))

    #Designed to be a list of walker positions
    log_lambda_p = np.random.uniform(low=-3.9, high=-3.5, size=(1, nwalkers))
    log_lambda_w = np.random.uniform(low=-1, high=1, size=(ncomp, nwalkers))
    rho_w = np.random.uniform(low=0.4, high=0.99, size=(ncomp*3, nwalkers))
    p0 = np.vstack((log_lambda_p, log_lambda_w, rho_w)).T

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=54)

    print("Running Sampler")
    pos, prob, state = sampler.run_mcmc(p0, 3000)

    print("Burn-in complete")
    np.save("after_burn_in.npy", np.array(pos))
    sampler.reset()
    sampler.run_mcmc(pos, 3000)

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

        inv = sp.linalg.inv(self.lambda_p * PP)
        bigsig = em.Sigma(sparams, self.lambda_w, self.rho_w)

        self.V11 = inv + bigsig

        self._params = None #Where we want to interpolate

        self.V12 = None
        self.V22 = None
        self.mu = None
        self.sig = None

    @property
    def params(self):
        return self._params
        pass

    @params.setter
    def params(self, pars):
        self._params = pars

        #Recalculate V12, V21, and V22.
        self.V12 = em.V12(self._params, sparams, self.rho_w)
        self.V22 = em.V22(self._params, self.rho_w)

        #Recalculate the covariance
        self.mu = self.V12.T.dot(sp.linalg.spsolve(self.V11, WHAT))
        self.sig = self.V22 - self.V12.T.dot(sp.linalg.spsolve(self.V11, self.V12))

    def __call__(self, *args):
        '''
        If you call this with an arg, it will set the emulator to these parameters first. If not,
        uses the previous result.
        '''
        if args:
            params, *junk = args
            self.params = params

        if self.V12 is None:
            print("No parameters are set, yet. Must set parameters first.")
            return

        else:
            return np.random.multivariate_normal(self.mu, self.sig)


def main():
    sample_lnprob()

if __name__=="__main__":
    main()