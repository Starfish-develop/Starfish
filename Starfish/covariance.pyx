# encoding: utf-8
# cython: profile=True
# filename: covariance.pyx

import numpy as np
from scipy.linalg import block_diag
cimport numpy as np
cimport cython
import Starfish.constants as C
import math

# Routines for emulator setup

# @cython.boundscheck(False)
# cdef k(np.ndarray[np.double_t, ndim=1] p0, np.ndarray[np.double_t, ndim=1] p1,
#       double a2, double lt2, double ll2, double lz2):
    #'''
    #Assumes that kernel params are already squared : a**2, l_temp**2
    #'''
    #return a2 * math.exp(-0.5 * ((p0[0] - p1[0])**2/lt2 + (p0[1] - p1[1])**2/ll2 + (p0[2] - p1[2])**2/lz2))

@cython.boundscheck(False)
cdef k(np.ndarray[np.double_t, ndim=1] p0, np.ndarray[np.double_t, ndim=1] p1, np.ndarray[np.double_t, ndim=1] h2params):
    '''
    Covariance function for the emulator. Defines the amount of covariance
    between two sets of input parameters.

    :param p0: first set of input parameters
    :type p0: np.array
    :param p1: second set of input parameters
    :type p1: np.array
    :param h2params: the set of Gaussian Process hyperparameters that set the
      degree of covariance. [amplitude, l0, l1, l2, ..., l(len(parname) - 1)].
      To save computation, these are already input squared.
    :type h2params: np.array

    :returns: (double) value of covariance
    '''
    dp = (p1 - p0)**2 # The covariance only depends on the distance squared
    cdef double a2 = h2params[0]
    l2 = h2params[1:]
    return a2 * math.exp(-0.5 * np.sum(dp/l2))

#@cython.boundscheck(False)
#def sigma(np.ndarray[np.double_t, ndim=2] gparams, double a2, double lt2, double ll2, double lz2):
#    '''
    #Assumes gparams have real units.

    #Assumes kernel parameters are coming in squared.
    #'''

    #cdef int m = len(gparams)
    #cdef int i = 0
    #cdef int j = 0
    #cdef double cov = 0.0

    #cdef np.ndarray[np.double_t, ndim=2] mat = np.empty((m,m), dtype=np.float64)

    #for i in range(m):
        #for j in range(i+1):
        #    cov = k(gparams[i], gparams[j], a2, lt2, ll2, lz2)
            #mat[i,j] = cov
            #mat[j,i] = cov

    #return mat

@cython.boundscheck(False)
def sigma(np.ndarray[np.double_t, ndim=2] gparams, np.ndarray[np.double_t, ndim=1] h2params):
    '''
    Compute the Lambda block of covariance for a single eigenspectrum weight.

    :param gparams: parameters at which the synthetic grid provides spectra
    :type gparams: 2D np.array
    :param hparams: the Gaussian Process hyperparameters defining amplitude and
      correlation length
    :type hparams: 1D np.array

    :returns: (2D np.array) Lambda block of covariance.
    '''

    cdef int m = len(gparams)
    cdef int i = 0
    cdef int j = 0
    cdef double cov = 0.0

    cdef np.ndarray[np.double_t, ndim=2] mat = np.empty((m,m), dtype=np.float64)

    for i in range(m):
        for j in range(i+1):
            cov = k(gparams[i], gparams[j], h2params)
            mat[i,j] = cov
            mat[j,i] = cov

    return mat

def Sigma(np.ndarray[np.double_t, ndim=2] gparams, np.ndarray[np.double_t, ndim=2] h2params):
    '''
    Fill in the large Sigma matrix using blocks of smaller sigma matrices.

    :param gparams: parameters at which the synthetic grid provides spectra
    :type gparams: 2D numpy array
    :param hparams: parameters for all of the different eigenspectra weights.
    :type hparams: 2D numpy array
    '''
    sig_list = []
    m = len(h2params)

    for h2param in h2params:
        sig_list.append(sigma(gparams, h2param))

    return block_diag(*sig_list)


def V12(params, np.ndarray[np.double_t, ndim=2] gparams, np.ndarray[np.double_t, ndim=1] hparams):
    '''
    Create V12, but just for a single weight.

    Assumes kernel params coming in squared
    '''
    cdef int m = len(gparams)
    cdef int i = 0
    cdef int j = 0

    #In the case that we might actually be predicting weights at more than one location.
    params.shape = (-1, 3)
    npoints = len(params)

    cdef np.ndarray[np.double_t, ndim=1] h2params = hparams**2

    mat = np.empty((m, npoints), dtype=np.float64)
    for i in range(m):
        for j in range(npoints):
            mat[i,j] = k(gparams[i], params[j], h2params)
    return mat

def V22(params, np.ndarray[np.double_t, ndim=1] hparams):
    '''
    Create V22, but just for a single weight.

    Assumes kernel parameters are coming in squared.
    '''
    cdef int i = 0
    cdef int j = 0

    #In the case that we might actually be predicting weights at more than one location.
    params.shape = (-1, 3)
    npoints = len(params)

    cdef np.ndarray[np.double_t, ndim=1] h2params = hparams**2

    mat = np.empty((npoints, npoints))
    for i in range(npoints):
        for j in range(npoints):
            mat[i,j] = k(params[i], params[j], h2params)
    return mat


# Routines for data covariance matrix generation

#New covariance filler routines
@cython.boundscheck(False)
def get_dense_C(np.ndarray[np.double_t, ndim=1] wl, k_func, double max_r):
    '''
    Fill out the covariance matrix.

    :param wl: numpy wavelength vector

    :param k_func: partial function to fill in matrix

    :param max_r: (km/s) max velocity to fill out to
    '''

    cdef int N = len(wl)
    cdef int i = 0
    cdef int j = 0
    cdef double cov = 0.0

    #Find all the indices that are less than the radius
    rr = np.abs(wl[:, np.newaxis] - wl[np.newaxis, :]) * C.c_kms/wl #Velocity space
    flag = (rr < max_r)
    indices = np.argwhere(flag)

    #The matrix that we want to fill
    mat = np.zeros((N,N))

    #Loop over all the indices
    for index in indices:
        i,j = index
        if j > i:
            continue
        else:
            #Initilize [i,j] and [j,i]
            cov = k_func(wl[i], wl[j])
            mat[i,j] = cov
            mat[j,i] = cov

    return mat

def make_k_func(params):
    cdef double amp = 10**params["cov"]["logAmp"]
    cdef double l = params["cov"]["l"] #Given in Km/s
    cdef double r0 = 6.0 * l #Km/s
    cdef double taper
    regions = params["regions"] #could be an empty dictionary {}

    cdef double a, mu, sigma, rx0, rx1, r_tap, r0_r

    def k_func(wl0, wl1):
        cdef double cov = 0.0

        #Initialize the global covariance
        cdef double r = C.c_kms/wl0 * math.fabs(wl0 - wl1) # Km/s
        if r < r0:
            taper = (0.5 + 0.5 * math.cos(np.pi * r/r0))
            cov = taper * amp*amp * (1 + math.sqrt(3) * r/l) * math.exp(-math.sqrt(3.) * r/l)

        #If covered by a region, instantiate
        for rparams in regions.values():
            a = 10**rparams["logAmp"]
            mu = rparams["mu"]
            sigma = rparams["sigma"]

            rx0 = C.c_kms / mu * math.fabs(wl0 - mu)
            rx1 = C.c_kms / mu * math.fabs(wl1 - mu)
            r_tap = rx0 if rx0 > rx1 else rx1 # choose the larger distance
            r0_r = 4.0 * sigma # where the kernel goes to 0

            if r_tap < r0_r:
                taper = (0.5 + 0.5 * math.cos(np.pi * r_tap/r0_r))
                cov += taper * a*a * math.exp(-0.5 * (C.c_kms * C.c_kms) / (mu * mu) * ((wl0 - mu)*(wl0 - mu) +
                                                             (wl1 - mu)*(wl1 - mu))/(sigma * sigma))
        return cov

    return k_func
