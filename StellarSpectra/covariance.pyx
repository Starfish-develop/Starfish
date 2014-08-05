# encoding: utf-8
# cython: profile=True
# filename: covariance.pyx

#Most of this file is designed after scikits-sparse
#https://github.com/njsmith/scikits-sparse/blob/master/scikits/sparse/cholmod.pyx

import numpy as np
cimport numpy as np
import scipy
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

    int cholmod_start(cholmod_common *c) except? 0
    int cholmod_finish(cholmod_common *c) except? 0

    ctypedef struct cholmod_triplet:
        size_t nrow
        size_t ncol
        size_t nzmax
        size_t nnz
        void *i
        void *j
        void *x
        void *z
        int stype
        int itype
        int dtype

    int cholmod_free_triplet(cholmod_triplet **T, cholmod_common *c) except? 0

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

    int cholmod_free_sparse(cholmod_sparse **, cholmod_common *c) except? 0
    cholmod_sparse *cholmod_add(cholmod_sparse *A, cholmod_sparse *B,
            double alpha[2], double beta[2], int values, int sorted, 
            cholmod_common *c) except? NULL
    cholmod_sparse *cholmod_copy_sparse(cholmod_sparse *A, cholmod_common *c) except? NULL
    double cholmod_norm_sparse(cholmod_sparse *A, int norm, cholmod_common *c) except? 0.0
    int cholmod_print_sparse(cholmod_sparse *A, const char *, cholmod_common *c) except? 0
    cholmod_triplet *cholmod_sparse_to_triplet(cholmod_sparse *A, cholmod_common *c) except? NULL

    ctypedef struct cholmod_dense:
        size_t nrow, ncol, nzmax
        size_t d
        void * x
        int xtype, dtype

    cholmod_dense *cholmod_allocate_dense(size_t nrow, size_t ncol, size_t d, int xtype, cholmod_common *c) except? NULL
    int cholmod_free_dense(cholmod_dense **, cholmod_common *c) except? 0
    int cholmod_print_dense(cholmod_dense *, const char *, cholmod_common *c) except? 0

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
    double cholmod_rcond(cholmod_factor *L, cholmod_common*c) except? 0.0

#These are the functions that I have written myself
cdef extern from "../extern/cov.h":

    void linspace(double *wl, int N, double start, double end)
    double get_min_sep (double *wl, int N)

    cholmod_sparse *create_sigma(double *sigma, int N, cholmod_common *c)
    cholmod_sparse *create_sparse(double *wl, int N, double min_sep, double a,
        double l, cholmod_common *c)
    cholmod_sparse *create_sparse_region(double *wl, int N, double a,
        double mu, double sigma, cholmod_common *c)

    double get_logdet(cholmod_factor *L)
    double chi2 (cholmod_dense *r, cholmod_factor *L, cholmod_common *c)

cdef class Common:

    cdef cholmod_common c

    def __init__(self):
        cholmod_start(&self.c)

    def __dealloc__(self):
        print("Deallocating Common")
        cholmod_finish(&self.c)

cdef cholmod_sparse *get_A(RegionCovarianceMatrix region):
    return region.A

cdef class CovarianceMatrix:

    cdef cholmod_sparse *A
    cdef cholmod_factor *L
    cdef cholmod_factor *L_last
    cdef Common common
    cdef cholmod_common *c
    cdef double *one
    cdef double logdet
    cdef double logdet_last
    cdef wl
    cdef fl
    cdef npoints
    cdef order_index
    cdef DataSpectrum
    cdef GlobalCovarianceMatrix GCM
    cdef RegionList
    cdef current_region
    cdef logPrior
    cdef logPrior_last
    cdef debug

    # initially a NULL pointer, and will be until any elements in the RegionList have been declared.
    cdef cholmod_sparse *sum_regions #Holds a sparse matrix that is the sum of all regions
    cdef cholmod_sparse *sum_regions_last

    def __init__(self, DataSpectrum, order_index, debug=False):
        self.common = Common()
        self.c = <cholmod_common *>&self.common.c

        self.one = <double*> PyMem_Malloc(2 * sizeof(double))
        self.one[0] = 1
        self.one[1] = 0

        #Pass the DataSpectrum to initialize the GlobalCovarianceMatrix
        #Here we need to pass only the masked pixels
        self.DataSpectrum = DataSpectrum
        self.order_index = order_index
        mask = self.DataSpectrum.masks[self.order_index]
        self.wl = DataSpectrum.wls[self.order_index][mask]
        self.fl = DataSpectrum.fls[self.order_index][mask]
        self.npoints = len(self.fl)
        self.logdet = 0.0
        self.logdet_last = self.logdet
        self.logPrior = 0.0 #neutral prior, reflects the sum of GlobalCovariance.logPrior and all region.logPrior's
        self.logPrior_last = self.logPrior
        self.debug = debug

        self.GCM = GlobalCovarianceMatrix(self.DataSpectrum, self.order_index,self.common, debug=self.debug)
        self.RegionList = []
        self.current_region = -1 #non physical value to force initialization

        self.update_factorization()

    def __dealloc__(self):
        #The GlobalCovarianceMatrix and all matrices in the RegionList will be cleared automatically?
        print("Deallocating Covariance Matrix")

        if self.sum_regions != self.sum_regions_last:
            if self.sum_regions != NULL:
                cholmod_free_sparse(&self.sum_regions, self.c)
            if self.sum_regions_last != NULL:
                cholmod_free_sparse(&self.sum_regions_last, self.c)
        else:
            if self.sum_regions != NULL:
                cholmod_free_sparse(&self.sum_regions, self.c)

        if self.A != NULL:
            cholmod_free_sparse(&self.A, self.c)

        # Due to revert functionality, there is a chance that __dealloc__ may be called in a state when self.L and
        # self.L_last both point to the same piece of memory
        if self.L != self.L_last:
            if self.L != NULL:
                cholmod_free_factor(&self.L, self.c)
            if self.L_last != NULL:
                cholmod_free_factor(&self.L_last, self.c)
        else:
            if self.L != NULL:
                cholmod_free_factor(&self.L, self.c)



    def update_global(self, params):
        '''
        Update the global covariance matrix by passing the parameters to the 
        GlobalCovarianceMatrix. Then update the factorization.
        '''
        if self.debug:
            print("updating global")
        self.GCM.update(params)
        self.update_factorization()
        self.update_logPrior()

    def create_region(self, params):
        '''
        Called to instantiate a new region and add it to self.RegionList
        '''
        #Check to make sure that mu is within bounds, if not, raise a C.RegionError
        mu = params["mu"]
        if (mu > np.min(self.wl)) and (mu < np.max(self.wl)):
            self.RegionList.append(RegionCovarianceMatrix(self.DataSpectrum, self.order_index, params, self.common,
            debug=self.debug))
            print("Added region to self.RegionList")
            ##Do update on partial_sum_regions and sum_regions
            self.update_region(region_index=(len(self.RegionList)-1), params=params)
        else:
            raise C.RegionError("chosen mu {:.4f} for region is outside bounds ({:.4f}, {:.4f})".format(mu, np.min(self.wl), np.max(self.wl)))


    def delete_region(self, region_index):
        self.RegionList.pop(region_index)

    def update_region(self, region_index, params):
        '''
        Key into the region using the index and update it's parameters. Then, update sum_regions
        and update the factorization.
        '''

        region = self.RegionList[region_index]
        if self.debug:
            print("updating region {} with params {}".format(region_index, params))
        region.update(params)

        self.update_sum_regions()
        self.update_factorization()
        self.update_logPrior()

    def print_all_regions(self):
        string = ""
        for i in range(len(self.RegionList)):
            string += "Region {}, ".format(i) + self.RegionList[i].__str__() + "\n"

        return string

    def get_regions_dict(self):
        '''
        Return a JSON dictionary with all of the region parameters.
        '''
        mydict = {}
        for i in range(len(self.RegionList)):
            mydict[str(i)] = self.RegionList[i].get_params()
        return mydict

    def update_sum_regions(self):
        '''
        Update the sparse matrix that is the sum of all region matrices. Used before starting global
        covariance sampling.
        '''
        #Move self.sum_regions_last to point to self.sum_regions
        if self.sum_regions_last != NULL and (self.sum_regions_last != self.sum_regions):
          #free the old memory before having it to something new
          #as long as they don't point to the same piece of memory
          if self.debug:
              print("freeing self.sum_regions_last inside of update_factorization")
          cholmod_free_sparse(&self.sum_regions_last, self.c)

        #Shift the self.sum_regions_last pointer to self.sum_regions
        if self.debug:
          print("shifting self.sum_regions_last to point to self.sum_regions")
        self.sum_regions_last = self.sum_regions

        cdef cholmod_sparse *R = cholmod_copy_sparse(get_A(self.RegionList[0]), self.c)
        cdef cholmod_sparse *temp

        for region in self.RegionList[1:]:
            temp = R
            R = cholmod_add(temp, get_A(region), self.one, self.one, True, True, self.c) 
            cholmod_free_sparse(&temp, self.c)

        self.sum_regions = R 

    def get_region_coverage(self):
        '''
        Go through all of the bounds in the region list and return a Boolean array that 
        specifies which pixels are already covered by a region.

        '''
        coverage = np.zeros_like(self.wl, dtype='bool') #An array of [False False ... False]

        if len(self.RegionList) == 0:
            return coverage

        else:
            #go through all of the bounds contained in the region list and update the values
            for region in self.RegionList:
                min, max = region.get_bounds()
                temp_ind = (self.wl >= min) & (self.wl <= max)
                #combine temp_ind with coverage
                coverage = temp_ind | coverage

            return coverage

    def update_logPrior(self):
        '''
        Go through all of the regions in RegionList and sum together their logPrior along 
        with the GlobalCovariance logPrior
        '''
        self.logPrior_last = self.logPrior
        self.logPrior = self.GCM.logPrior + np.sum([region.get_prior() for region in self.RegionList])

    def update_factorization(self):
        '''
        Sum together the global covariance matrix and the sum_regions, calculate 
        the logdet and the new cholmod_factorization.

        I think here we must free the memory previously afforded to A.

        '''
        if self.debug:
            print("updating factorization")

        if self.A != NULL:
            cholmod_free_sparse(&self.A, self.c)

        if self.sum_regions != NULL:
            self.A = cholmod_add(self.GCM.A, self.sum_regions, self.one, self.one, True, True, self.c)
        else:
            assert len(self.RegionList) == 0, "RegionList is not empty but pointer is NULL"
            #there are no regions declared, so we should COPY self.GCM.A
            self.A = cholmod_copy_sparse(self.GCM.A, self.c)

        #Copy the old self.L to self.L_last
        if self.L_last != NULL and (self.L_last != self.L):
            #free the old memory before having it to something new
            #as long as they don't point to the same piece of memory
            if self.debug:
                print("freeing self.L_last inside of update_factorization")
            cholmod_free_factor(&self.L_last, self.c)

        #Shift the self.L_last pointer to self.L
        if self.debug:
            print("shifting self.L_last to point to self.L")
        self.L_last = self.L

        self.L = cholmod_analyze(self.A, self.c) #do Cholesky factorization
        cholmod_factorize(self.A, self.L, self.c)
        #print("rcond is ", cholmod_rcond(self.L, self.c))

        if self.debug:
            print("updating logdet")
        self.logdet_last = self.logdet
        self.logdet = get_logdet(self.L)

    def revert(self):
        if self.debug:
            print("reverting covariance matrix")
        self.logdet = self.logdet_last
        self.logPrior = self.logPrior_last

        #As long as L and L_last don't point to the same thing, clear L
        if self.L != NULL and (self.L != self.L_last):
            if self.debug:
                print("freeing self.L inside of revert")
            cholmod_free_factor(&self.L, self.c)

        #Move self.L to point to self.L_last
        if self.debug:
            print("shifting self.L to point to self.L_last inside of revert")
        self.L = self.L_last

    def revert_global(self):
        self.GCM.revert()
        self.revert()

    def revert_region(self, region_index):
        region = self.RegionList[region_index]
        if self.debug:
            print("reverting region {}".format(region_index))
        region.revert()

        #move sum_regions to point to the sum_regions_last
        if self.debug:
            print("shifting self.sum_regions to point to self.sum_regions_last")
        #as long as sum_regions and sum_regions_last don't point to the same thing, clear sum_regions
        if self.sum_regions != NULL and (self.sum_regions_last != self.sum_regions):
            if self.debug:
                print("freeing self.sum_regions inside of revert_region")
            cholmod_free_sparse(&self.sum_regions, self.c)

        self.sum_regions = self.sum_regions_last

        self.revert()

    def get_amp(self):
        return self.GCM.amp

    def return_logdet(self):
        return self.logdet

    def evaluate(self, fl):
        '''
        Use the existing covariance matrix to evaluate the residuals.

        Input is model flux, calculate the residuals by subtracting from the dataflux, then convert to a dense_matrix.
        '''
        if self.debug:
            print("evaluating covariance matrix")

        residuals = self.fl - fl

        #convert the residuals to a cholmod_dense matrix
        cdef np.ndarray[np.double_t, ndim=1] rr = residuals
        cdef cholmod_dense *r = cholmod_allocate_dense(self.npoints, 1, self.npoints, CHOLMOD_REAL, self.c)
        cdef double *x = <double*>r.x #pointer to the data in cholmod_dense struct
        for i in range(self.npoints):
            x[i] = rr[i]

        #logdet does not depend on the residuals, so it is pre-computed
        #evaluate lnprob with logdet and chi2
        if self.debug:
            print("evaluating chi2")
        cdef double lnprob = -0.5 * (chi2(r, self.L, self.c) + self.logdet) + self.logPrior

        cholmod_free_dense(&r, self.c)

        return lnprob


    def cholmod_to_scipy_sparse(self):
        '''
        Instantiate and return a scipy.sparse matrix from cholmod_sparse matrix self.A
        '''
        #First, convert the cholmod_sparse to cholmod_triplet form so it's easier
        #to read off
        cdef cholmod_triplet *T = cholmod_sparse_to_triplet(self.A, self.c)

        #initialize a scipy.sparse.dok_matrix
        S = scipy.sparse.dok_matrix((self.npoints, self.npoints), dtype=np.float64)

        #Iterate through the T to pull out row, column, and value into three separate lists. 
        cdef int *Ti = <int*>T.i
        cdef int *Tj = <int*>T.j
        cdef double *Tx = <double*>T.x
        cdef int stype = <int>T.stype

        if stype == 0:
            #Matrix is "unsymmetric, and all values are stored
            for k in range(T.nnz):
                row = Ti[k]
                column = Tj[k]
                value = Tx[k]

                S[row, column] = value 

        else:
            #Matrix is symmetric and either only lower or only upper values are stored
            #So therefore also store the transpose (but not diagonal)
            for k in range(T.nnz):
                row = Ti[k]
                column = Tj[k]
                value = Tx[k]

                S[row, column] = value
                if row != column:
                    S[column, row] = value

        cholmod_free_triplet(&T, self.c)

        return S

    def test_common_equal(self):
        '''
        Figure out if all the cholmod_common's are actually the same.
        '''
        print(self.c == self.GCM.c)
        print(self.common == self.GCM.common)


cdef class GlobalCovarianceMatrix:

    cdef cholmod_sparse *sigma
    cdef cholmod_sparse *A
    cdef cholmod_sparse *A_last
    cdef Common common
    cdef cholmod_common *c
    cdef double *wl
    cdef double min_sep
    cdef npoints
    cdef amp
    cdef logPrior
    cdef debug

    def __init__(self, DataSpectrum, order_index, Common common, debug=False):
        self.common = common
        self.c = <cholmod_common *>&self.common.c
        self.debug = debug
        self.A = NULL
        self.A_last = self.A
        self.amp = 1.0
        self.logPrior = 0.0

        #mask wl
        mask = DataSpectrum.masks[order_index]
        #convert wl into an array
        cdef np.ndarray[np.double_t, ndim=1] wl = DataSpectrum.wls[order_index][mask]
        self.npoints = len(wl)
    
        #Dynamically allocate wl
        self.wl = <double*> PyMem_Malloc(self.npoints * sizeof(double))
        
        for i in range(self.npoints):
            self.wl[i] = wl[i]

        self.min_sep = get_min_sep(self.wl, self.npoints)
        #wl, fl, sigma, and min_sep do not change with update, since the data is fixed
        self._initialize_sigma(DataSpectrum.sigmas[order_index])

    def __dealloc__(self):
        print("Deallocating GlobalCovarianceMatrix")
        PyMem_Free(self.wl)
        if self.sigma != NULL:
            cholmod_free_sparse(&self.sigma, self.c)
        if self.A != self.A_last:
            if self.A != NULL:
                cholmod_free_sparse(&self.A, self.c)
            if self.A_last != NULL:
                cholmod_free_sparse(&self.A_last, self.c)
        else:
            if self.A != NULL:
                cholmod_free_sparse(&self.A, self.c)

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

        self.sigma = create_sigma(sigma_C, self.npoints, self.c)
        PyMem_Free(sigma_C) #since we do not need sigma_C for anything else, free it now
        #also, set self.A = self.sigma, because we haven't yet created any global
        #covariance structure
        self.A = cholmod_copy_sparse(self.sigma, self.c)

    def update(self, params):
        '''
        On update, calculate the logdet and the new cholmod_factorization.

        Parameters is a dictionary.
        '''
        amp = 10**params['logAmp']
        self.amp = amp

        l = params['l']
        sigAmp = params['sigAmp']

        if (l <= 0) or (sigAmp < 0):
            raise C.ModelError("l {} and sigAmp {} must be positive.".format(l, sigAmp))

        cdef double *alpha =  [1, 0]
        cdef double *beta = [sigAmp**2, 0]

        #Copy the old self.A to self.A_last
        if self.A_last != NULL and (self.A_last != self.A):
            #Free the old memory before having it point to something new
            #as long as they don't point to the same piece of memory
            if self.debug:
                print("freeing self.A inside of GCM.update")
            cholmod_free_sparse(&self.A_last, self.c)

        #Shift the self.A_last pointer to self.A
        if self.debug:
            print("shifting self.A_last to point to self.A")
        self.A_last = self.A

        cdef cholmod_sparse *temp = create_sparse(self.wl, self.npoints, self.min_sep, amp, l, self.c)
        
        self.A = cholmod_add(temp, self.sigma, alpha, beta, True, True, self.c)
        cholmod_free_sparse(&temp, self.c)

    def revert(self):
        #move A to point to A_last
        if self.debug:
            print("reverting global covariance matrix")
        #as long as A and A_last don't point to the same thing, clear A
        if self.A != NULL and (self.A_last != self.A):
            if self.debug:
                print("freeing self.A inside of revert")
            cholmod_free_sparse(&self.A, self.c)

        #move self.A to point to self.A_last
        if self.debug:
            print("shifting self.A to point to self.A_last")
        self.A = self.A_last


cdef class RegionCovarianceMatrix:

    cdef cholmod_sparse *A
    cdef cholmod_sparse *A_last
    cdef Common common
    cdef cholmod_common *c
    cdef double *wl
    cdef npoints
    cdef mu
    cdef sigma0
    cdef params
    cdef logPrior #This is carried around with each CovarianceMatrix
    cdef logPrior_last
    cdef debug

    def __init__(self, DataSpectrum, order_index, params, Common common, debug=False):
        self.common = common 
        self.c = <cholmod_common *>&self.common.c

        #convert wl into an array
        cdef np.ndarray[np.double_t, ndim=1] wl = DataSpectrum.wls[order_index]
        self.npoints = len(wl)
    
        #Dynamically allocate wl
        self.wl = <double*> PyMem_Malloc(self.npoints * sizeof(double))
        
        for i in range(self.npoints):
            self.wl[i] = wl[i]

        self.mu = params["mu"] #take the anchor point for reference?
        self.sigma0 = 0.5 #how far can the region stray?
        self.logPrior = 0.0 #neutral prior
        self.logPrior_last = self.logPrior
        print("Created Region and logPrior is ", self.logPrior)
        self.update(params) #do the first initialization
        self.params = None
        self.debug = debug

    def __dealloc__(self):
        print("Deallocating RegionCovarianceMatrix")
        PyMem_Free(self.wl)

        if self.A != self.A_last:
            if self.A != NULL:
                cholmod_free_sparse(&self.A, self.c)
            if self.A_last != NULL:
                cholmod_free_sparse(&self.A_last, self.c)
        else:
            if self.A != NULL:
                cholmod_free_sparse(&self.A, self.c)
    
    def __str__(self):
        return "mu = {}, sigma0 = {}".format(self.mu, self.sigma0)

    def get_bounds(self):
        return (self.mu - self.sigma0, self.mu + self.sigma0)

    def get_params(self):
        return self.params

    def eval_prior(self, params):
        '''
        Define and evaluate the prior for a given set of parameters.
        '''
        #Use a ln(logistic) function on sigma, that is flat before 10km/s and dies off for anything greater
        sigma = params['sigma']

        lnLogistic = np.log(-1./(1. + np.exp(10. - sigma)) + 1.)

        #Use a Gaussian prior on mu, that it keeps the region within the original setting.
        # 1/(sqrt(2pi) * sigma) exp(-0.5 (mu-x)^2/sigma^2)
        #-ln(sigma * sqrt(2 pi)) - 0.5 (mu - x)^2 / sigma^2
        width = 0.05
        mu = params['mu']
        lnGauss = -0.5 * np.abs(mu - self.mu)**2/width**2 - np.log(width * np.sqrt(2. * np.pi))

        self.logPrior_last = self.logPrior
        self.logPrior = lnLogistic + lnGauss

    def get_prior(self):
        return self.logPrior


    def update(self, params):
        '''
        Parameters is a dictionary of {a, mu, sigma}.
        Back in CovarianceMatrix, calculate the logdet and the new cholmod_factorization.
        '''
        a = 10**params['loga']
        mu = params['mu']
        sigma = params['sigma']

        if (sigma <=0) or (a < 0):
            raise C.ModelError("sigma {}, and a {} must be positive.".format(sigma, a))

        if np.abs((mu - self.mu)) > self.sigma0:
            raise C.RegionError("mu {} has strayed too far from the \
                    original specification {}".format(mu, self.mu))

        cdef cholmod_sparse *temp = create_sparse_region(self.wl, self.npoints, a, mu, sigma, self.c)

        if temp == NULL:
            raise C.RegionError("region is too small to contain any nonzero elements.")

        #Copy the old self.A to self.A_last
        if self.A_last != NULL and (self.A_last != self.A):
            #as long as A_last and A don't point to the same piece of memory,
            #free the old memory before having it point to something new

            if self.debug:
                print("freeing self.A inside of RegionCovarianceMatrix.update")
            cholmod_free_sparse(&self.A_last, self.c)

        #Shift the self.A_last pointer to self.A
        if self.debug:
            print("shifting self.A_last to point to self.A")
        self.A_last = self.A

        self.A = temp
        self.params = params
        self.eval_prior(params)

    def revert(self):
        self.logPrior = self.logPrior_last

        if self.debug:
            print("reverting region covariance matrix")

        #as long as A and A_last don't point to the same thing, clear A
        if self.A != NULL and (self.A_last != self.A):
            if self.debug:
                print("freeing RegionMatrix.A inside of revert")
            cholmod_free_sparse(&self.A, self.c)

        #move self.A to point to self.A_last
        if self.debug:
            print("shifting self.A to point to self.A_last")
        self.A = self.A_last


#How long does the add operation actually take? About 0.02s to add together two large covariance regions. I think it might be OK to leave in the extra add operation at this point.
