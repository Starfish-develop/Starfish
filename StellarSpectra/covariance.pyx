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
    cholmod_sparse *cholmod_copy_sparse(cholmod_sparse *A, cholmod_common *c) except? NULL
    int cholmod_print_sparse(cholmod_sparse *, const char *, cholmod_common *c) except? 0

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
    int cholmod_free_factor(cholmod_factor **, cholmod_common *c) except? 0
    cholmod_factor * cholmod_analyze(cholmod_sparse *, cholmod_common *c) except? NULL
    int cholmod_factorize(cholmod_sparse *, cholmod_factor *, cholmod_common *c) except? 0
    cholmod_sparse * cholmod_spsolve(int, cholmod_factor *, cholmod_sparse *, cholmod_common *c) except? NULL

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



cdef class CovarianceMatrix:

    cdef cholmod_sparse *A
    cdef cholmod_factor *L
    cdef cholmod_common c
    cdef double logdet
    cdef fl
    cdef npoints
    cdef order_index
    cdef GlobalCovarianceMatrix
    cdef RegionList
    cdef current_region

    #Both sum_regions and partial_sum_regions are initially NULL pointers, and will be until
    #any elements in the RegionList have been declared.
    cdef cholmod_sparse *sum_regions #Holds a sparse matrix that is the sum of all regions
    cdef cholmod_sparse *partial_sum_regions #Holds a sparse matrix that is the sum of all regions but the one currently sampling (N -1)

    def __init__(self, DataSpectrum, order_index):

        #Pass the DataSpectrum to initialize the GlobalCovarianceMatrix
        self.DataSpectrum = DataSpectrum
        self.order_index = order_index
        self.fl = DataSpectrum.fls[self.order_index]
        self.logdet = 0.0

        cholmod_start(&self.c)

        self.GlobalCovarianceMatrix = GlobalCovarianceMatrix(self.DataSpectrum, self.order_index, self.c)
        self.RegionList = []
        self.current_region = -1 #non physical value to force initialization

    def __dealloc__(self):
        #The GlobalCovarianceMatrix and all matrices in the RegionList will be cleared automatically?
        cholmod_free_sparse(&self.sum_regions, &self.c)
        cholmod_free_sparse(&self.partial_sum_regions, &self.c)
        cholmod_free_sparse(&self.A, &self.c)
        cholmod_free_factor(&self.L, &self.c)
        cholmod_finish(&self.c)

    def update_global(self, params):
        '''
        Update the global covariance matrix by passing the parameters to the 
        GlobalCovarianceMatrix. Then update the factorization.
        '''
        self.GlobalCovarianceMatrix.update(params)
        self.update_factorization()

    def create_region(self, params):
        '''
        Called to instantiate a new region and add it to self.RegionList
        '''
        self.RegionList.append(RegionCovarianceMatrix(self.DataSpectrum, self.order_index, params))

    def delete_region(self, region_index):
        self.RegionList.pop(region_index)

    def update_region(self, region_index, params):
        '''
        Key into the region using the index and update it's parameters. Then, update sum_regions
        and update the factorization.
        '''

        if region_index != self.current_region:
            #We have switched sampling regions, time to update the partial sum
            if len(self.RegionList) > 1:
                self.update_partial_sum_regions()
                #if len == 1, then we are the first region initialized and there is no partial sum.
            self.current_region = region_index

        region = self.RegionList[region_index]
        region.update(params)

        cdef double *alpha =  [1, 0]
        cdef double *beta = [1, 0]

        if self.partial_sum_regions != NULL:
            self.sum_regions = cholmod_add(region.A, self.partial_sum_regions, alpha, 
                    beta, True, True, &self.c)
        else:
            assert len(self.RegionList) == 1,"partial_sum_regions is NULL but RegionList contains more than one region."
            self.sum_regions = region.A
        self.update_factorization()

    def update_sum_regions(self):
        '''
        Update the sparse matrix that is the sum of all region matrices. Used before starting global
        covariance sampling.
        '''
        cdef cholmod_sparse *R = self.RegionList[0].A
        cdef double *alpha =  [1, 0]
        cdef double *beta = [1, 0]

        for region in self.RegionList[1:]:
            R = cholmod_add(R, region.A, alpha, beta, True, True, &self.c) 

        self.sum_regions = R 

    def update_partial_sum_regions(self, region_index):
        '''
        Update the partial sparse matrix that is the partial sum of all region matrices except the one
        specified by index. Used before starting the sampling for an individual region.
        '''
        partialRegionList = self.RegionList[:region_index] + self.RegionList[region_index + 1:]
        cdef cholmod_sparse *R = partialRegionList[0]
        cdef double *alpha =  [1, 0]
        cdef double *beta = [1, 0]

        for region in partialRegionList[1:]:
            R = cholmod_add(R, region, alpha, beta, True, True, &self.c) 

        self.partial_sum_regions = R 

    def update_factorization(self):
        '''
        Sum together the global covariance matrix and the sum_regions, calculate 
        the logdet and the new cholmod_factorization.

        '''
        if self.sum_regions != NULL:
            self.A = cholmod_add(self.GlobalCovarianceMatrix.A, self.sum_regions, alpha, beta, True, True, &self.c)
        else:
            assert len(self.RegionList) == 0, "RegionList is not empty but pointer is NULL"
            #there are no regions declared
            self.A = self.GlobalCovarianceMatrix.A 

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

cdef class GlobalCovarianceMatrix:

    cdef cholmod_sparse *sigma
    cdef cholmod_sparse *A
    cdef cholmod_common c
    cdef double *wl
    cdef double min_sep
    cdef npoints

    def __init__(self, DataSpectrum, order_index, c):
        self.c = c #cholmod_start(&self.c) has already been run by CovarianceMatrix

        #convert wl into an array
        cdef np.ndarray[np.double_t, ndim=1] wl = DataSpectrum.wls[order_index]
        self.npoints = len(wl)
    
        #Dynamically allocate wl
        self.wl = <double*> PyMem_Malloc(self.npoints * sizeof(double))
        
        for i in range(self.npoints):
            self.wl[i] = wl[i]

        self.min_sep = get_min_sep(self.wl, self.npoints)
        #wl, fl, sigma, and min_sep do not change with update, since the data is fixed
        self.logdet = 0.0


        self._initialize_sigma(DataSpectrum.sigmas[order_index])


    def __dealloc__(self):
        PyMem_Free(self.wl)
        cholmod_free_sparse(&self.sigma, &self.c)
        cholmod_free_sparse(&self.A, &self.c)

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


cdef class RegionCovarianceMatrix:

    cdef cholmod_sparse *A
    cdef cholmod_common c
    cdef double *wl
    cdef npoints
    cdef mu
    cdef sigma0

    def __init__(self, DataSpectrum, order_index, params, c):
        self.c = c #cholmod_start(&self.c) has already been run by CovarianceMatrix

        #convert wl into an array
        cdef np.ndarray[np.double_t, ndim=1] wl = DataSpectrum.wls[order_index]
        self.npoints = len(wl)
    
        #Dynamically allocate wl
        self.wl = <double*> PyMem_Malloc(self.npoints * sizeof(double))
        
        for i in range(self.npoints):
            self.wl[i] = wl[i]

        self.mu = params["mu"] #take the anchor point for reference?
        self.sigma0 = 3.
        self.update(params) #do the first initialization


    def __dealloc__(self):
        PyMem_Free(self.wl)
        cholmod_free_sparse(&self.A, &self.c)

    def update(self, params):
        '''
        On update, calculate the logdet and the new cholmod_factorization.

        Parameters is a dictionary of {h, a, mu, sigma}.
        '''
        h = params['h']
        a = params['a']
        mu = params['mu']
        sigma = params['sigma']

        if (h <= 0) or (sigma <=0) or (a < 0):
            raise C.ModelError("h {}, sigma {}, and a {} must be positive.".format(h, sigma, a))

        if np.abs((mu - self.mu)) > self.sigma0:
            raise C.ModelError("mu {} has strayed too far from the \
                    original specification {}".format(mu, self.mu))
        
        self.A = create_sparse_region(self.wl, self.npoints, h, a, mu, sigma, &self.c)
 
#Run some profiling code to see what our bottlenecks might be.
#Update takes 0.138 seconds, evaluate takes 0.001s

#When doing just the region sampling, the sampler keys into the update_region method attached to the CovarianceMatrix

#How long does the add operation actually take? About 0.02s to add together two large covariance regions. I think it might be OK to leave in the extra add operation at this point.
