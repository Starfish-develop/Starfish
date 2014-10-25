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

@cython.boundscheck(False)
cdef k(np.ndarray[np.double_t, ndim=1] p0, np.ndarray[np.double_t, ndim=1] p1,
    double a2, double lt2, double ll2, double lz2):
    '''
    Assumes that kernel params are already squared : a**2, l_temp**2
    '''
    return a2 * math.exp(-0.5 * ((p0[0] - p1[0])**2/lt2 + (p0[1] - p1[1])**2/ll2 + (p0[2] - p1[2])**2/lz2))

@cython.boundscheck(False)
def sigma(np.ndarray[np.double_t, ndim=2] gparams, double a2, double lt2, double ll2, double lz2):
    '''
    Assumes gparams have real units: [temp, logg, Z]

    Assumes kernel parameters are coming in squared.
    '''

    cdef int m = len(gparams)
    cdef int i = 0
    cdef int j = 0
    cdef double cov = 0.0

    cdef np.ndarray[np.double_t, ndim=2] mat = np.empty((m,m), dtype=np.float64)

    for i in range(m):
        for j in range(i+1):
            cov = k(gparams[i], gparams[j], a2, lt2, ll2, lz2)
            mat[i,j] = cov
            mat[j,i] = cov

    return mat

def V12_w(params, np.ndarray[np.double_t, ndim=2] gparams, double a2, double lt2, double ll2, double lz2):
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

    mat = np.empty((m, npoints), dtype=np.float64)
    for i in range(m):
        for j in range(npoints):
            mat[i,j] = k(gparams[i], params[j], a2, lt2, ll2, lz2)
    return mat

def V22_w(params, double a2, double lt2, double ll2, double lz2):
    '''
    Create V22, but just for a single weight.

    Assumes kernel parameters are coming in squared.
    '''
    cdef int i = 0
    cdef int j = 0

    #In the case that we might actually be predicting weights at more than one location.
    params.shape = (-1, 3)
    npoints = len(params)

    mat = np.empty((npoints, npoints))
    for i in range(npoints):
        for j in range(npoints):
            mat[i,j] = k(params[i], params[j], a2, lt2, ll2, lz2)
    return mat

