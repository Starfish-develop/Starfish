import numpy as np
import multiprocessing as mp
import StellarSpectra.constants as C

def random_draws(cov, num, nprocesses=mp.cpu_count()):
    '''
    Return random multivariate Gaussian draws from the covariance matrix.

    :param cov: covariance matrix
    :param num: number of draws

    :returns: array of random draws
    '''

    N = cov.shape[0]
    pool = mp.Pool(nprocesses)

    result = pool.starmap_async(np.random.multivariate_normal, zip([np.zeros((N,))] * num, [cov]*num))
    return np.array(result.get())



#Set of kernels *exactly* as defined in extern/cov.h
@np.vectorize
def k_gloabl(r, a, l):
    r0=6.*l
    taper = (0.5 + 0.5 * np.cos(np.pi * r/r0))
    if r >= taper:
        return 0.
    else:
        return taper * a**2 * (1 + np.sqrt(3) * r/l) * np.exp(-np.sqrt(3) * r/l)

@np.vectorize
def k_local(x0, x1, a, mu, sigma):
    r0 = 4.0 * sigma #spot where kernel goes to 0
    rx0 = C.c_kms / mu * np.abs(x0 - mu)
    rx1 = C.c_kms / mu * np.abs(x1 - mu)
    r_tap = rx0 if rx0 > rx1 else rx1 #choose the larger distance

    if r_tap >= r0:
        return 0.
    else:
        taper = (0.5 + 0.5 * np.cos(np.pi * r_tap/r0))
        return taper * a**2 * np.exp(-0.5 * C.c_kms**2/mu**2 * ((x0 - mu)**2 + (x1 - mu)**2)/sigma**2)


def k_global_func(x0i, x1i, x0v=None, x1v=None, a=None, l=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    r = np.abs(x1 - x0)
    pass

def k_local_func(x0i, x1i, x0v=None, x1v=None, a=None, mu=None, sigma=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    pass



@np.vectorize
def hann(r, r0):
    if np.abs(r) < r0:
        return 0.5 + 0.5 * np.cos(np.pi * r/r0)
    else:
        return 0

def k_3_2_hann(r, l):
    return k_3_2(r, l) * hann(r, 6 * l)


def gauss_func(x0i, x1i, x0v=None, x1v=None, amp=None, mu=None, sigma=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    return amp**2/(2 * np.pi * sigma**2) * np.exp(-((x0 - mu)**2 + (x1 - mu)**2)/(2 * sigma**2))

def k_3_2_func(x0i, x1i, x0v=None, x1v=None, amp=None, l=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    r = np.abs(x1 - x0)
    return amp**2 * k_3_2(r, l)

def k_3_2_hann_func(x0i, x1i, x0v=None, x1v=None, amp=None, l=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    r = np.abs(x1 - x0)
    return amp**2 * k_3_2_hann(r, l)


#mat_3_2 = np.fromfunction(k_3_2_hann_func, (N,N), x0v=wl23, x1v=wl23, amp=0.028, l=0.14, dtype=np.int) #matrix from Matern kernel
#mat = mat_3_2 + sigma23n**2 * np.eye(N) #add in the per-pixel noise

def k_3_2(r, l):
    return (1 + np.sqrt(3) * r/l) * np.exp(-np.sqrt(3) * r/l)

@np.vectorize
def hann(r, r0):
    if np.abs(r) < r0:
        return 0.5 + 0.5 * np.cos(np.pi * r/r0)
    else:
        return 0

def k_3_2_hann(r, l):
    return k_3_2(r, l) * hann(r, 6 * l)


def gauss_func(x0i, x1i, x0v=None, x1v=None, amp=None, mu=None, sigma=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    return amp**2/(2 * np.pi * sigma**2) * np.exp(-((x0 - mu)**2 + (x1 - mu)**2)/(2 * sigma**2))

def k_3_2_func(x0i, x1i, x0v=None, x1v=None, amp=None, l=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    r = np.abs(x1 - x0)
    return amp**2 * k_3_2(r, l)

def k_3_2_hann_func(x0i, x1i, x0v=None, x1v=None, amp=None, l=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    r = np.abs(x1 - x0)
    return amp**2 * k_3_2_hann(r, l)

def gauss_hann_func(x0i, x1i, x0v=None, x1v=None, amp=None, mu=None, sigma=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    r = np.abs(x1 - x0)
    return hann(r, 4 * sigma) * amp**2/(2 * np.pi * sigma**2) * np.exp(-((x0 - mu)**2 + (x1 - mu)**2)/(2 * sigma**2))


# How to generate a matrix from one of these functions
#mat_3_2 = np.fromfunction(k_3_2_hann_func, (N,N), x0v=wl23, x1v=wl23, amp=0.028, l=0.14, dtype=np.int) #matrix from
# Matern kernel
#mat = mat_3_2 + sigma23n**2 * np.eye(N) #add in the per-pixel noise


#All of these return *dense* covariance matrices as defined in the paper
def Poisson_matrix(wl, sigma):
    raise NotImplementedError
    return matrix

def k_global_matrix(wl, amp, l):
    raise NotImplementedError
    return matrix

def k_local_matrix(wl, a, mu, sigma):
    raise NotImplementedError
    return matrix





def main():
    cov = np.eye(20)

    draws = random_draws(cov, 5)
    print(draws)

if __name__=='__main__':
    main()

