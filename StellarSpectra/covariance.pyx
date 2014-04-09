# encoding: utf-8
# cython: profile=True
# filename: covariance.pyx

#Most of this file is designed after scikits-sparse
#https://github.com/njsmith/scikits-sparse/blob/master/scikits/sparse/cholmod.pyx

import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import StellarSpectra.constants as C

cdef extern from "cholmod.h":

    cdef enum:
        CHOLMOD_INT
        CHOLMOD_PATTERN, CHOLMOD_REAL, CHOLMOD_COMPLEX
        CHOLMOD_DOUBLE
        CHOLMOD_AUTO, CHOLMOD_SIMPLICIAL, CHOLMOD_SUPERNODAL
        CHOLMOD_OK, CHOLMOD_NOT_POSDEF
        CHOLMOD_A, CHOLMOD_LDLt, CHOLMOD_LD, CHOLMOD_DLt, CHOLMOD_L
        CHOLMOD_Lt, CHOLMOD_D, CHOLMOD_P, CHOLMOD_Pt

    ctypedef struct cholmod_common:
        int supernodal
        int status

    int cholmod_start(cholmod_common *) except? 0
    int cholmod_finish(cholmod_common *) except? 0

    ctypedef struct cholmod_sparse:
        size_t nrow, ncol, nzmax
        void * p # column pointers
        void * i # row indices
        void * x
        int stype # 0 = regular, >0 = upper triangular, <0 = lower triangular
        int itype # type of p, i, nz
        int xtype
        int dtype
        int sorted
        int packed

    int cholmod_free_sparse(cholmod_sparse **, cholmod_common *) except? 0
    cholmod_sparse *cholmod_add(cholmod_sparse *A, cholmod_sparse *B,
            double alpha[2], double beta[2], int values, int sorted, 
            cholmod_common *c) except? NULL
    int cholmod_print_sparse(cholmod_sparse *, const char *, cholmod_common *) except? 0

    ctypedef struct cholmod_dense:
        size_t nrow, ncol, nzmax
        size_t d
        void * x
        int xtype, dtype

    cholmod_dense *cholmod_allocate_dense(size_t nrow, size_t ncol, size_t d, int xtype, cholmod_common *) except? NULL
    int cholmod_free_dense(cholmod_dense **, cholmod_common *) except? 0
    int cholmod_print_dense(cholmod_dense *, const char *, cholmod_common *) except? 0

    ctypedef struct cholmod_factor:
        size_t n
        void * Perm
        int itype
        int xtype
        int is_ll, is_super, is_monotonic
        size_t xsize, nzmax, nsuper
        void * x
        void * p
        void * super_ "super"
        void * pi
        void * px
    int cholmod_free_factor(cholmod_factor **, cholmod_common *) except? 0
    cholmod_factor * cholmod_analyze(cholmod_sparse *, cholmod_common *) except? NULL
    int cholmod_factorize(cholmod_sparse *, cholmod_factor *, cholmod_common *) except? 0
    cholmod_sparse * cholmod_spsolve(int, cholmod_factor *, cholmod_sparse *, cholmod_common *) except? NULL

#These are the functions that I have written myself
cdef extern from "../extern/cov.h":

    void linspace(double *wl, int N, double start, double end)
    double get_min_sep (double *wl, int N)

    cholmod_sparse *create_sigma(double *sigma, int N, cholmod_common *c)
    cholmod_sparse *create_sparse(double *wl, int N, double max_sep, double a, 
        double l, cholmod_common *c)
    cholmod_sparse *create_sparse_region(double *wl, int N, double h, double a, 
        double mu, double sigma, cholmod_common *c)

    double get_logdet(cholmod_factor *L)
    double chi2 (cholmod_dense *r, cholmod_factor *L, cholmod_common *c)

def initialize_sigma(mysigma):
    '''
    Debugging routine.
    '''

    cdef np.ndarray[np.double_t, ndim=1] sigmanp = mysigma

    npoints = 2298
    cdef cholmod_common c
    cholmod_start(&c)
    cdef double *sigma_C = <double*> PyMem_Malloc(npoints * sizeof(double))
    for i in range(npoints):
        sigma_C[i] = sigmanp[i]
        print(sigma_C[i])
    #for i in range(npoints):
    #    sigma_C[i] = 1.0

    cdef cholmod_sparse *sigma = create_sigma(sigma_C, npoints, &c)

    PyMem_Free(sigma_C)
    cholmod_print_sparse(sigma, "sigma", &c)
    cholmod_free_sparse(&sigma, &c)
    cholmod_finish(&c)

cdef class Cov:

    cdef cholmod_sparse *sigma
    cdef cholmod_sparse *A
    cdef cholmod_factor *L
    cdef cholmod_common c
    cdef double *wl
    cdef double min_sep
    cdef double logdet
    cdef fl
    cdef npoints

    def __init__(self, DataSpectrum, index):
        #convert wl into an array

        cdef np.ndarray[np.double_t, ndim=1] wl = DataSpectrum.wls[index]
        self.npoints = len(wl)
    
        #Dynamically allocate wl
        self.wl = <double*> PyMem_Malloc(self.npoints * sizeof(double))
        
        for i in range(self.npoints):
            self.wl[i] = wl[i]

        self.fl = DataSpectrum.wls[index]
        self.min_sep = get_min_sep(self.wl, self.npoints)
        #wl, fl, sigma, and min_sep do not change with update, since the data is fixed
        self.logdet = 0.0

        cholmod_start(&self.c)

        self._initialize_sigma(DataSpectrum.sigmas[index])


    def __dealloc__(self):
        PyMem_Free(self.wl)
        cholmod_free_sparse(&self.sigma, &self.c)
        cholmod_free_sparse(&self.A, &self.c)
        cholmod_free_factor(&self.L, &self.c)
        cholmod_finish(&self.c)

    def _initialize_sigma(self, mysigma):
        '''
        Take in a numpy array from a DataSpectrum containing the sigma values, then
        initialize the sparse array containing the sigma values
        '''

        cdef np.ndarray[np.double_t, ndim=1] sigmanp = mysigma

        #Dynamically allocate sigma
        cdef double *sigma_C = <double*> PyMem_Malloc(self.npoints * sizeof(double))

        for i in range(self.npoints):
            sigma_C[i] = sigmanp[i]

        self.sigma = create_sigma(sigma_C, self.npoints, &self.c)
        PyMem_Free(sigma_C) #since we do not need sigma_C for anything else, free it now

    def update(self, params):
        '''
        On update, calculate the logdet and the new cholmod_factorization.

        Parameters is a dictionary.
        '''
        amp = 10**params['logAmp']
        l = params['l']
        sigAmp = params['sigAmp']

        if (l <= 0) or (sigAmp < 0):
            raise C.ModelError("l {} and sigAmp {} must be positive.".format(l, sigAmp))

        cdef double *alpha =  [1, 0]
        cdef double *beta = [sigAmp**2, 0]
        
        self.A = cholmod_add(create_sparse(self.wl, self.npoints, self.min_sep, amp, l, &self.c), self.sigma, alpha, beta, True, True, &self.c)

        self.L = cholmod_analyze(self.A, &self.c) #do Cholesky factorization 
        cholmod_factorize(self.A, self.L, &self.c)

        self.logdet = get_logdet(self.L)

    def evaluate(self, fl):
        '''
        Use the existing covariance matrix to evaluate the residuals.

        Input is model flux, calculate the residuals by subtracting from the dataflux, then convert to a dense_matrix.
        '''
        residuals = self.fl - fl

        #convert the residuals to a cholmod_dense matrix
        cdef np.ndarray[np.double_t, ndim=1] rr = residuals
        cdef cholmod_dense *r = cholmod_allocate_dense(self.npoints, 1, self.npoints, CHOLMOD_REAL, &self.c)
        cdef double *x = <double*>r.x #pointer to the data in cholmod_dense struct
        for i in range(self.npoints):
            x[i] = rr[i]

        #logdet does not depend on the residuals, so it is pre-computed
        #evaluate lnprob with logdet and chi2
        cdef double lnprob = -0.5 * (chi2(r, self.L, &self.c) + self.logdet)

        cholmod_free_dense(&r, &self.c)

        return lnprob
 
#Output, plot the sigma_matrix as a 1D slice through row=constant

#Run test suites using py.test

#Run some profiling code to see what our bottlenecks might be.
#Update takes 0.138 seconds, evaluate takes 0.001s

#Figure out how to update individual regions

