import numpy as np
cimport cython
cimport numpy as np
import math
import Starfish.constants as C

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
    rr = np.abs(wl[:, np.newaxis] - wl[np.newaxis, :]) * C.c_kms / wl  #Velocity space
    flag = (rr < max_r)
    indices = np.argwhere(flag)

    #The matrix that we want to fill
    mat = np.zeros((N, N))

    #Loop over all the indices
    for index in indices:
        i, j = index
        if j > i:
            continue
        else:
            #Initilize [i,j] and [j,i]
            cov = k_func(wl[i], wl[j])
            mat[i, j] = cov
            mat[j, i] = cov

    return mat

def make_k_func(par):
    cdef double amp = 10 ** par.logAmp
    cdef double l = par.l  #Given in Km/s
    cdef double r0 = 6.0 * l  #Km/s
    cdef double taper
    regions = par.regions  #could be None or a 2D array

    cdef double a, mu, sigma, rx0, rx1, r_tap, r0_r

    if regions is None:
        # make a k_func that excludes regions and is faster
        def k_func(wl0, wl1):
            cdef double cov = 0.0

            #Initialize the global covariance
            cdef double r = C.c_kms / wl0 * math.fabs(wl0 - wl1)  # Km/s
            if r < r0:
                taper = (0.5 + 0.5 * math.cos(np.pi * r / r0))
                cov = taper * amp * amp * (1 + math.sqrt(3) * r / l) * math.exp(-math.sqrt(3.) * r / l)

            return cov
    else:
        # make a k_func which includes regions
        def k_func(wl0, wl1):
            cdef double cov = 0.0

            #Initialize the global covariance
            cdef double r = C.c_kms / wl0 * math.fabs(wl0 - wl1)  # Km/s
            if r < r0:
                taper = (0.5 + 0.5 * math.cos(np.pi * r / r0))
                cov = taper * amp * amp * (1 + math.sqrt(3) * r / l) * math.exp(-math.sqrt(3.) * r / l)

            #If covered by a region, instantiate
            for row in regions:
                a = 10 ** row[0]
                mu = row[1]
                sigma = row[2]

                rx0 = C.c_kms / mu * math.fabs(wl0 - mu)
                rx1 = C.c_kms / mu * math.fabs(wl1 - mu)
                r_tap = rx0 if rx0 > rx1 else rx1  # choose the larger distance
                r0_r = 4.0 * sigma  # where the kernel goes to 0

                if r_tap < r0_r:
                    taper = (0.5 + 0.5 * math.cos(np.pi * r_tap / r0_r))
                    cov += taper * a * a * math.exp(
                        -0.5 * (C.c_kms * C.c_kms) / (mu * mu) * ((wl0 - mu) * (wl0 - mu) + (wl1 - mu) * (wl1 - mu)) / (
                                sigma * sigma))
            return cov

    return k_func

# Make just the matrix for the regions
def make_k_func_region(phi):
    cdef double taper
    regions = phi.regions  #could be None or a 2D array

    cdef double a, mu, sigma, rx0, rx1, r_tap, r0_r

    cdef double cov = 0.0
    # make a k_func which includes regions
    # this is a closure so it keeps the defined regions
    def k_func(wl0, wl1):

        cov = 0.0
        #print("cov begin", cov)

        #Initialize the global covariance
        cdef double r = C.c_kms / wl0 * math.fabs(wl0 - wl1)  # Km/s

        #If covered by a region, instantiate
        for row in regions:
            # print("logAmp", row[0])
            a = 10 ** row[0]
            # print("a", a)
            mu = row[1]
            sigma = row[2]

            rx0 = C.c_kms / mu * math.fabs(wl0 - mu)
            rx1 = C.c_kms / mu * math.fabs(wl1 - mu)
            r_tap = rx0 if rx0 > rx1 else rx1  # choose the larger distance
            r0_r = 4.0 * sigma  # where the kernel goes to 0

            if r_tap < r0_r:
                taper = (0.5 + 0.5 * math.cos(np.pi * r_tap / r0_r))
                #print("exp", math.exp(-0.5 * (C.c_kms * C.c_kms) / (mu * mu) * ((wl0 - mu)*(wl0 - mu) + (wl1 - mu)*(wl1 - mu))/(sigma * sigma)))
                #print("taper", taper)
                #print("amp", a*a)
                #print("additional", taper * a*a * math.exp(-0.5 * (C.c_kms * C.c_kms) / (mu * mu) * ((wl0 - mu)*(wl0 - mu) + (wl1 - mu)*(wl1 - mu))/(sigma * sigma)))
                cov += taper * a * a * math.exp(
                    -0.5 * (C.c_kms * C.c_kms) / (mu * mu) * ((wl0 - mu) * (wl0 - mu) + (wl1 - mu) * (wl1 - mu)) / (
                            sigma * sigma))

        #print("cov end", cov)
        return cov

    return k_func
