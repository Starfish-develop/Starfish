import numpy as np
import multiprocessing as mp
import StellarSpectra.constants as C

def multivariate_normal(cov):
    N = cov.shape[0]
    mu = np.zeros((N,))
    return np.random.multivariate_normal(mu, cov)

def random_draws(cov, num, nprocesses=mp.cpu_count()):
    '''
    Return random multivariate Gaussian draws from the covariance matrix.

    :param cov: covariance matrix
    :param num: number of draws

    :returns: array of random draws
    '''

    N = cov.shape[0]
    pool = mp.Pool(nprocesses)

    result = pool.map(multivariate_normal, [cov]*num)
    return np.array(result)

def envelope(spectra):
    '''
    Given a 2D array of spectra, shape (Nspectra, Npix), return the minimum/maximum envelope of these as two spectra.
    '''
    return np.min(spectra, axis=0), np.max(spectra, axis=0)


def saveall(fig, fname, formats=[".png", ".pdf", ".svg"]):
    '''
    Save a matplotlib figure instance to many different formats
    '''
    for format in formats:
        fig.savefig(fname + format)



#Set of kernels *exactly* as defined in extern/cov.h
@np.vectorize
def k_global(r, a, l):
    r0=6.*l
    taper = (0.5 + 0.5 * np.cos(np.pi * r/r0))
    if r >= r0:
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
    r = np.abs(x1 - x0) * C.c_kms/x0
    return k_global(r=r, a=a, l=l)

def k_local_func(x0i, x1i, x0v=None, x1v=None, a=None, mu=None, sigma=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    return k_local(x0=x0, x1=x1, a=a, mu=mu, sigma=sigma)

#All of these return *dense* covariance matrices as defined in the paper
def Poisson_matrix(wl, sigma):
    '''
    Sigma can be an array or a single float.
    '''
    N = len(wl)
    matrix = sigma**2 * np.eye(N)
    return matrix

def k_global_matrix(wl, a, l):
    N = len(wl)
    matrix = np.fromfunction(k_global_func, (N,N), x0v=wl, x1v=wl, a=a, l=l, dtype=np.int)
    return matrix

def k_local_matrix(wl, a, mu, sigma):
    N = len(wl)
    matrix = np.fromfunction(k_local_func, (N, N), x0v=wl, x1v=wl, a=a, mu=mu, sigma=sigma, dtype=np.int)
    return matrix



def main():
    cov = np.eye(20)

    draws = random_draws(cov, 5)
    print(draws)

if __name__=='__main__':
    main()

