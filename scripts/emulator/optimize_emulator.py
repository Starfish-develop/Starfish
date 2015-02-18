import Starfish
from Starfish import emulator
from Starfish.covariance import Sigma
import numpy as np
import math

# Load the PCAGrid from the location specified in the config file
pca = emulator.PCAGrid.open()
PhiPhi = np.linalg.inv(emulator.skinny_kron(pca.eigenspectra, pca.M))

def Glnprior(x, s, r):
    return s * np.log(r) + (s - 1.) * np.log(x) - r*x - math.lgamma(s)

def lnprob(p):
    '''

    :param p: Gaussian Processes hyper-parameters
    :type p: 1D np.array

    Calculate the lnprob using Habib's posterior formula for the emulator.

    '''

    # We don't allow negative parameters.
    if np.any(p < 0.):
        return 1e99

    lambda_xi = p[0]
    hparams = p[1:].reshape((pca.m, -1))

    print("lambda_xi", lambda_xi)
    for row in hparams:
        print(row)
    print()

    # Calculate the prior for temp, logg, and Z
    priors = np.sum(Glnprior(hparams[:, 1], 2., 0.0075)) + np.sum(Glnprior(hparams[:, 2], 2., 0.75)) + np.sum(Glnprior(hparams[:, 3], 2., 0.75))

    h2params = hparams**2
    #Fold hparams into the new shape
    Sig_w = Sigma(pca.gparams, h2params)

    C = (1./lambda_xi) * PhiPhi + Sig_w

    sign, pref = np.linalg.slogdet(C)

    central = pca.w_hat.T.dot(np.linalg.solve(C, pca.w_hat))

    lnp = -0.5 * (pref + central + pca.M * pca.m * np.log(2. * np.pi)) + priors
    print(lnp)

    # Negate this since we are using the fmin algorithm
    return -lnp


def main():

    # Set up starting parameters and see if lnprob evaluates.
    # p will have a length of 1 + (pca.m * (1 + len(Starfish.parname)))
    amp = 50.0
    lt = 200.
    ll = 1.25
    lZ = 1.25
    #
    p0 = np.hstack((np.array([1., ]),
    np.hstack([np.array([amp, lt, ll, lZ]) for i in range(pca.m)]) ))

    # # Set lambda_xi separately
    p0[0] = 0.3

    # p0 = np.load("eparams.npy")

    print(lnprob(p0))

    from scipy.optimize import fmin
    result = fmin(lnprob, p0)
    print(result)
    np.save("eparams.npy", result)


if __name__=="__main__":
    main()
