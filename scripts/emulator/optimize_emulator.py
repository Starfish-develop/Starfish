import argparse
parser = argparse.ArgumentParser(description="Optimize the emulator.")
parser.add_argument("-m", "--min", action="store_true", help="Use fmin instead of emcee.")
parser.add_argument("-f", "--fresh", action="store_true", help="Start from a good guess, rather than the previously stored parameters.")
args = parser.parse_args()


import Starfish
from Starfish import emulator
from Starfish.covariance import Sigma
import numpy as np
import math
import multiprocessing as mp

# Load the PCAGrid from the location specified in the config file
pca = emulator.PCAGrid.open()
PhiPhi = np.linalg.inv(emulator.skinny_kron(pca.eigenspectra, pca.M))

def Glnprior(x, s, r):
    return s * np.log(r) + (s - 1.) * np.log(x) - r*x - math.lgamma(s)

def lnprob(p, fmin=False):
    '''

    :param p: Gaussian Processes hyper-parameters
    :type p: 1D np.array

    Calculate the lnprob using Habib's posterior formula for the emulator.

    '''

    # We don't allow negative parameters.
    if np.any(p < 0.):
        if fmin:
            return 1e99
        else:
            return -np.inf

    lambda_xi = p[0]
    hparams = p[1:].reshape((pca.m, -1))

    # Calculate the prior for temp, logg, and Z
    priors = np.sum(Glnprior(hparams[:, 1], 2., 0.0075)) + np.sum(Glnprior(hparams[:, 2], 2., 0.75)) + np.sum(Glnprior(hparams[:, 3], 2., 0.75))

    h2params = hparams**2
    #Fold hparams into the new shape
    Sig_w = Sigma(pca.gparams, h2params)

    C = (1./lambda_xi) * PhiPhi + Sig_w

    sign, pref = np.linalg.slogdet(C)

    central = pca.w_hat.T.dot(np.linalg.solve(C, pca.w_hat))

    lnp = -0.5 * (pref + central + pca.M * pca.m * np.log(2. * np.pi)) + priors
    # print(lnp)

    # Negate this when using the fmin algorithm
    if fmin:
        print("lambda_xi", lambda_xi)
        for row in hparams:
            print(row)
        print()
        print(lnp)
        
        return -lnp
    else:
        return lnp

def minimize():

    if args.fresh():

        amp = 50.0
        lt = 200.
        ll = 1.25
        lZ = 1.25
        #
        p0 = np.hstack((np.array([1., ]),
        np.hstack([np.array([amp, lt, ll, lZ]) for i in range(pca.m)]) ))

        # # Set lambda_xi separately
        p0[0] = 0.3

    else:
        p0 = np.load("eparams.npy")

    from scipy.optimize import fmin
    func = lambda p : lnprob(p, fmin=True)
    result = fmin(func, p0, maxiter=10000, maxfun=10000)
    print(result)
    np.save("eparams.npy", result)

def sample():
    import emcee

    ndim = 1 + (1 + len(Starfish.parname)) * pca.m
    nwalkers = 4 * ndim # about the minimum per dimension we can get by with

    # Assemble p0 based off either a guess or the previous state of walkers
    if args.fresh:
        p0 = []
        # p0 is a (nwalkers, ndim) array
        amp = [40.0, 150]
        lt = [150., 300]
        ll = [0.8, 1.65]
        lZ = [0.8, 1.65]

        p0.append(np.random.uniform(0.1, 1.0, nwalkers))
        for i in range(pca.m):
            p0 +=   [np.random.uniform(amp[0], amp[1], nwalkers),
                    np.random.uniform(lt[0], lt[1], nwalkers),
                    np.random.uniform(ll[0], ll[1], nwalkers),
                    np.random.uniform(lZ[0], lZ[1], nwalkers)]

        p0 = np.array(p0).T

    else:
        p0 = np.load("walkers_start.npy")

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=mp.cpu_count())

    # burn in
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    print("Burned in")

    # actual run
    pos, prob, state = sampler.run_mcmc(pos, 100)

    # Save the last position of the walkers
    np.save("walkers_start.npy", pos)
    np.save("eparams_walkers.npy", sampler.flatchain)

def main():

    # Set up starting parameters and see if lnprob evaluates.
    # p will have a length of 1 + (pca.m * (1 + len(Starfish.parname)))

    if args.min:
        minimize()
    else:
        sample()


if __name__=="__main__":
    main()
