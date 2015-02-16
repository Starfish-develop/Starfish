import Starfish
from Starfish import emulator
from Starfish.covariance import Sigma
import numpy as np

# Load the PCAGrid from the location specified in the config file
pca = emulator.PCAGrid.open()
PhiPhi = np.linalg.inv(emulator.skinny_kron(pca.eigenspectra, pca.M))

def lnprob(p):
    '''

    :param p: Gaussian Processes hyper-parameters
    :type p: 1D np.array

    Calculate the lnprob using Habib's posterior formula for the emulator.

    '''

    print(p)
    if np.any(p < 0.):
        return 1e99

    lambda_xi = p[0]
    h2params = p[1:].reshape((pca.m, -1))
    #Fold hparams into the new shape
    Sig_w = Sigma(pca.gparams, h2params)

    C = (1./lambda_xi) * PhiPhi + Sig_w

    sign, pref = np.linalg.slogdet(C)

    central = pca.w_hat.T.dot(np.linalg.solve(C, pca.w_hat))

    # s = 5.
    # r = 5.
    # prior_l = s * np.log(r) + (s - 1.) * np.log(ll) - r*ll - math.lgamma(s)
    #
    # s = 5.
    # r = 5.
    # prior_z = s * np.log(r) + (s - 1.) * np.log(lz) - r*lz - math.lgamma(s)

    lnp = 0.5 * (pref + central + pca.M * pca.m * np.log(2. * np.pi)) #+ prior_l + prior_z
    print(lnp)
    return lnp


def main():

    # Set up starting parameters and see if lnprob evaluates.
    # p will have a length of 1 + (pca.m * (1 + len(Starfish.parname)))
    amp = 1.0
    lt = 200.
    ll = 0.5
    lZ = 0.5

    p0 = np.hstack((np.array([10., ]),
    np.hstack([np.array([amp, lt, ll, lZ]) for i in range(pca.m)]) ))

    # Deal only with squared parameters for the GP.
    p0 = p0**2

    print(lnprob(p0))

    from scipy.optimize import fmin
    result = fmin(lnprob, p0)
    print(result)


if __name__=="__main__":
    main()
