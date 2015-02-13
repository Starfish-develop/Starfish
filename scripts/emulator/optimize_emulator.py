import Starfish
from Starfish import emulator
from Starfish.covariance import Sigma
import numpy as np
import math

# Load the PCAGrid from the location specified in the config file
pca = emulator.PCAGrid.open()
M = len(pca.gparams)
PhiPhi = np.linalg.inv(emulator.skinny_kron(pca.eigenspectra))


def lnprob(p):
    '''

    :param p: Gaussian Processes hyper-parameters
    :type p: 1D np.array

    Calculate the lnprob using Habib's posterior formula for the emulator.

    '''
    lambda_xi = p[0]
    hparams = p[1:].reshape((M, -1))
    #Fold hparams into the new shape
    Sig_w = Sigma(pca.gparams, hparams)

    C = 1/lambda_xi * PhiPhi + Sig_w

    sign, pref = np.linalg.slogdet(C)

    central = pca.w_hat.T.dot(np.linalg.solve(C, pca.w_hat))

    # s = 5.
    # r = 5.
    # prior_l = s * np.log(r) + (s - 1.) * np.log(ll) - r*ll - math.lgamma(s)
    #
    # s = 5.
    # r = 5.
    # prior_z = s * np.log(r) + (s - 1.) * np.log(lz) - r*lz - math.lgamma(s)

    return -0.5 * (pref + central + pca.M * pca.m * np.log(2. * np.pi)) #+ prior_l + prior_z


def fmin_lnprob(weight_index):
    # from scipy.optimize import fmin
    # #from scipy.optimize import minimize
    # p0 = np.array([1., 200., 1.0, 1.0])
    # func = lambda x: -lnprob(x, weight_index)
    # result = fmin(func, p0)
    # #result = minimize(func, p0, bounds=[(-3, 3),(40, 400),(0.1, 2.0),(0.1, 2.0)])
    # print(weight_index, result)
    # return result
    pass


def main():

    # Set up starting parameters and see if lnprob evaluates.
    # p will have a length of 1 + (pca.m * (1 + len(Starfish.parname)))
    amp = 1.0
    lt = 200.
    ll = 0.5
    lZ = 0.5

    p = np.hstack((np.array([0.01, ]), np.hstack([np.array([amp, lt, ll, lZ]) for i in range(pca.m)]))

    print(lnprob(p))


if __name__=="__main__":
    main()
