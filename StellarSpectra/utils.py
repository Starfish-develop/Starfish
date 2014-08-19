import numpy as np
import multiprocessing as mp

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


class FlatchainTree:
    '''
    Object defined to wrap a Flatchain structure in order to facilitate combining, burning, etc.

    The Tree will always follow the same structure.

    flatchains.hdf5:

    stellar samples:    stellar

    folder for model:   0

        folder for order: 22

                        cheb
                        cov
                        cov_region00
                        cov_region01
                        cov_region02
                        ....


        folder for order: 23

                        cheb
                        cov
                        cov_region00
                        cov_region01
                        cov_region02
                        ....

    folder for model:   1


    '''
    pass


class OldFlatchainTree:
    '''
    The old structure which assumed only 1 DataSpectrum. For legacy's sake.
    '''
    pass



def main():
    cov = np.eye(20)

    draws = random_draws(cov, 5)
    print(draws)

if __name__=='__main__':
    main()

