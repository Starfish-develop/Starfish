import scipy.sparse as sp
import numpy as np
import math
cimport numpy as np
cimport cython

#@cython.boundscheck(False)
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

#@cython.boundscheck(False)
cdef csigma(int m, iprecision, irhos, sparams):
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
                cov = R(sparams[i], sparams[j], lnrhos)
                mat[i,j] = cov
                mat[j,i] = cov
    return mat/iprecision

def Sigma(sparams, precisions, rhos):
    '''
    Create all Sigmas
    '''
    m = len(sparams)

    sigmas = []

    for iprecision, irhos in zip(precisions, rhos):
        sigmas.append(csigma(m, iprecision, irhos, sparams))

    return sp.block_diag(sigmas)


def V12(params, sparams, rhos):
    '''
    Given the new parameters and the set parameters, setup the covariance matrix.
    '''
    m = len(sparams)
    cdef int ncomp = len(rhos)
    nparams = len(params)

    #Create all of the small matrices and then do an hstack on them
    out = []

    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    lnrhos = np.log(rhos)

    #For each eigenspectra component
    for j, ilnrhos in enumerate(lnrhos):
        mat = sp.dok_matrix((m , ncomp), dtype=np.float64)
        for i in range(m):
            mat[i,j] = R(sparams[i], params, ilnrhos)
        out.append(mat)

    return sp.vstack(out)

def V22(params, rhos):
    lnrhos = np.log(rhos)
    cdef int ncomp = len(rhos)
    cdef int i = 0

    mat = sp.dok_matrix((ncomp, ncomp))
    for i, ilnrhos in enumerate(lnrhos):
        mat[i,i] = R(params, params, ilnrhos)

    return mat





