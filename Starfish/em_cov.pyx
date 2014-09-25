import scipy.sparse as sp
import numpy as np
import math
cimport numpy as np
cimport cython

@cython.boundscheck(False)
cdef R(np.ndarray[np.double_t, ndim=1] p0, np.ndarray[np.double_t, ndim=1] p1, np.ndarray[np.double_t, ndim=1] irhos):
    '''
    Autocorrelation function.

    p0, p1 : the two sets of parameters, each shape (nparams,)

    irhos : shape (nparams,)
    '''
    cdef double sum = 0.
    cdef unsigned int i = 0
    for i in range(3):
        sum += 4. * (p0[i] - p1[i])*(p0[i] - p1[i]) * irhos[i]
    return math.exp(sum)

@cython.boundscheck(False)
cdef csigma(int m, iprecision, irhos, gparams):
    '''
    Create the dense matrix using cython
    '''
    cdef int i = 0
    cdef int j = 0
    cdef double cov = 0.0
    lnrhos = np.log(irhos)
    cdef np.ndarray[np.double_t, ndim=2] mat = np.empty((m,m), dtype=np.float64)
    for i in range(m):
        for j in range(i+1):
                cov = R(gparams[i], gparams[j], lnrhos)
                mat[i,j] = cov
                mat[j,i] = cov
    return mat/iprecision

def Sigma(gparams, precisions, rhos):
    '''
    Create all Sigmas
    '''
    m = len(gparams)

    sigmas = []

    for iprecision, irhos in zip(precisions, rhos):
        sigmas.append(csigma(m, iprecision, irhos, gparams))

    return sp.block_diag(sigmas)


def V12(params, gparams, rhos):
    '''
    Given the new parameters and the set parameters, setup the covariance matrix.
    '''
    m = len(gparams)
    cdef int ncomp = len(rhos)
    nparams = len(params)

    #Do hstack on these
    out = []

    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    lnrhos = np.log(rhos)

    for ilnrhos in lnrhos:
        mat = np.empty((m , ncomp), dtype=np.float64)
        for i in range(m):
            for j in range(ncomp):
                mat[i,j] = R(gparams[i], params, ilnrhos)
        out.append(mat)

    return np.vstack(out)


def V22(params, rhos):
    lnrhos = np.log(rhos)
    out = []
    for ilnrhos in lnrhos:
        out.append(R(params, params, ilnrhos))

    return np.eye(len(rhos)).dot(np.array(out))





