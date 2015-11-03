import numpy as np
import multiprocessing as mp
import Starfish.constants as C
import csv
import h5py
from astropy.table import Table
from astropy.io import ascii

def multivariate_normal(cov):
    np.random.seed()
    N = cov.shape[0]
    mu = np.zeros((N,))
    result = np.random.multivariate_normal(mu, cov)
    print("Generated residual")
    return result

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


def std_envelope(spectra):
    '''
    Given a 2D array of spectra, shape (Nspectra, Npix), return the std envelope of these as two spectra.
    '''
    std = np.std(spectra, axis=0)
    return -std, std

def visualize_draws(spectra, num=20):
    '''
    Given a 2D array of spectra, shape (Nspectra, Npix), visualize them to choose the most illustrative "random"
    samples.
    '''
    import matplotlib.pyplot as plt
    offset = 6 * np.std(spectra[0], axis=0)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    for i, (spectrum, off) in enumerate(zip(spectra[:num], offset * np.arange(0, num))):
        ax.axhline(off, ls=":", color="0.5")
        ax.plot(spectrum + off, "k")
        ax.annotate(i, (1, off))

    plt.show()


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


# Tools to examine Markov Chain Runs
def h5read(fname, burn=0, thin=1):
    '''
    Read the flatchain from the HDF5 file and return it.
    '''
    fid = h5py.File(fname, "r")
    assert burn < fid["samples"].shape[0]
    print("{} burning by {} and thinning by {}".format(fname, burn, thin))
    flatchain = fid["samples"][burn::thin]

    fid.close()

    return flatchain

def csvread(fname, burn=0, thin=1):
    '''
    Read the flatchain from a CSV file and return it.
    '''
    flatchain = np.genfromtxt(fname, skip_header=1, dtype=float, delimiter=",")[burn::thin]

    return flatchain

def gelman_rubin(samplelist):
    '''
    Given a list of flatchains from separate runs (that already have burn in cut
    and have been trimmed, if desired), compute the Gelman-Rubin statistics in
    Bayesian Data Analysis 3, pg 284. If you want to compute this for fewer
    parameters, then slice the list before feeding it in.
    '''

    full_iterations = len(samplelist[0])
    assert full_iterations % 2 == 0, "Number of iterations must be even. Try cutting off a different number of burn in samples."
    shape = samplelist[0].shape
    #make sure all the chains have the same number of iterations
    for flatchain in samplelist:
        assert len(flatchain) == full_iterations, "Not all chains have the same number of iterations!"
        assert flatchain.shape == shape, "Not all flatchains have the same shape!"

    #make sure all chains have the same number of parameters.

    #Following Gelman,
    # n = length of split chains
    # i = index of iteration in chain
    # m = number of split chains
    # j = index of which chain
    n = full_iterations//2
    m = 2 * len(samplelist)
    nparams = samplelist[0].shape[-1] #the trailing dimension of a flatchain

    #Block the chains up into a 3D array
    chains = np.empty((n, m, nparams))
    for k, flatchain in enumerate(samplelist):
        chains[:,2*k,:] = flatchain[:n]  #first half of chain
        chains[:,2*k + 1,:] = flatchain[n:] #second half of chain

    #Now compute statistics
    #average value of each chain
    avg_phi_j = np.mean(chains, axis=0, dtype="f8") #average over iterations, now a (m, nparams) array
    #average value of all chains
    avg_phi = np.mean(chains, axis=(0,1), dtype="f8") #average over iterations and chains, now a (nparams,) array

    B = n/(m - 1.0) * np.sum((avg_phi_j - avg_phi)**2, axis=0, dtype="f8") #now a (nparams,) array

    s2j = 1./(n - 1.) * np.sum((chains - avg_phi_j)**2, axis=0, dtype="f8") #now a (m, nparams) array

    W = 1./m * np.sum(s2j, axis=0, dtype="f8") #now a (nparams,) arary

    var_hat = (n - 1.)/n * W + B/n #still a (nparams,) array
    std_hat = np.sqrt(var_hat)

    R_hat = np.sqrt(var_hat/W) #still a (nparams,) array


    data = Table({   "Value": avg_phi,
                     "Uncertainty": std_hat},
                 names=["Value", "Uncertainty"])

    print(data)

    ascii.write(data, sys.stdout, Writer = ascii.Latex, formats={"Value":"%0.3f", "Uncertainty":"%0.3f"}) #

    #print("Average parameter value: {}".format(avg_phi))
    #print("std_hat: {}".format(np.sqrt(var_hat)))
    print("R_hat: {}".format(R_hat))

    if np.any(R_hat >= 1.1):
        print("You might consider running the chain for longer. Not all R_hats are less than 1.1.")


def plot(flatchain, base, format=".png"):
    '''
    Make a triangle plot
    '''

    import triangle

    labels = [r"$T_\mathrm{eff}$ [K]", r"$\log g$ [dex]", r"$Z$ [dex]",
    r"$v_z$ [km/s]", r"$v \sin i$ [km/s]", r"$\log_{10} \Omega$"]
    figure = triangle.corner(flatchain, quantiles=[0.16, 0.5, 0.84],
        plot_contours=True, plot_datapoints=False, labels=labels, show_titles=True)
    figure.savefig(base + "triangle" + format)

def paper_plot(flatchain, base, format=".pdf"):
    '''
    Make a triangle plot of just M vs i
    '''

    import matplotlib
    matplotlib.rc("font", size=8)
    matplotlib.rc("lines", linewidth=0.5)
    matplotlib.rc("axes", linewidth=0.8)
    matplotlib.rc("patch", linewidth=0.7)
    import matplotlib.pyplot as plt
    #matplotlib.rc("axes", labelpad=10)
    from matplotlib.ticker import FormatStrFormatter as FSF
    from matplotlib.ticker import MaxNLocator
    import triangle

    labels = [r"$M_\ast\enskip [M_\odot]$", r"$i_d \enskip [{}^\circ]$"]
    #r"$r_c$ [AU]", r"$T_{10}$ [K]", r"$q$", r"$\log M_\textrm{CO} \enskip [\log M_\oplus]$",
    #r"$\xi$ [km/s]"]
    inds = np.array([0, 6, ]) #1, 2, 3, 4, 5])

    K = len(labels)
    fig, axes = plt.subplots(K, K, figsize=(3., 2.5))

    figure = triangle.corner(flatchain[:, inds], plot_contours=True,
    plot_datapoints=False, labels=labels, show_titles=False,
        fig=fig)

    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.4, 0.5)
    for ax in axes[-1, :]:
        ax.xaxis.set_label_coords(0.5, -0.4)

    figure.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.2)

    figure.savefig(base + "ptriangle" + format)


def plot_walkers(flatchain, base, start=0, end=-1, labels=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    # majorLocator = MaxNLocator(nbins=4)
    ndim = len(flatchain[0, :])
    sample_num = np.arange(len(flatchain[:,0]))
    sample_num = sample_num[start:end]
    samples = flatchain[start:end]
    plt.rc("ytick", labelsize="x-small")

    fig, ax = plt.subplots(nrows=ndim, sharex=True)
    for i in range(0, ndim):
        ax[i].plot(sample_num, samples[:,i])
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        if labels is not None:
            ax[i].set_ylabel(labels[i])

    ax[-1].set_xlabel("Sample number")
    fig.subplots_adjust(hspace=0)
    fig.savefig(base + "walkers.png")
    plt.close(fig)

def estimate_covariance(flatchain, base):

    if args.ndim:
        d = args.ndim
    else:
        d = flatchain.shape[1]

    import matplotlib.pyplot as plt

    #print("Parameters {}".format(flatchain.param_tuple))
    #samples = flatchain.samples
    cov = np.cov(flatchain, rowvar=0)

    #Now try correlation coefficient
    cor = np.corrcoef(flatchain, rowvar=0)
    print("Correlation coefficient")
    print(cor)

    # Make a plot of correlation coefficient.

    fig, ax = plt.subplots(figsize=(0.5 * d, 0.5 * d), nrows=1, ncols=1)
    ext = (0.5, d + 0.5, 0.5, d + 0.5)
    ax.imshow(cor, origin="upper", vmin=-1, vmax=1, cmap="bwr", interpolation="none", extent=ext)
    fig.savefig("cor_coefficient.png")

    print("'Optimal' jumps with covariance (units squared)")

    opt_jump = 2.38**2/d * cov
    # opt_jump = 1.7**2/d * cov # gives about ??
    print(opt_jump)

    print("Standard deviation")
    std_dev = np.sqrt(np.diag(cov))
    print(std_dev)

    print("'Optimal' jumps")
    if args.ndim:
        d = args.ndim
    else:
        d = flatchain.shape[1]
    print(2.38/np.sqrt(d) * std_dev)

    np.save(base + "opt_jump.npy", opt_jump)


def cat_list(file, flatchainList):
    '''
    Given a list of flatchains, concatenate all of these and write them to a
    single HDF5 file.
    '''
    #Write this out to the new file
    print("Opening", file)
    hdf5 = h5py.File(file, "w")

    cat = np.concatenate(flatchainList, axis=0)

    dset = hdf5.create_dataset("samples", cat.shape, compression='gzip',
        compression_opts=9)
    dset[:] = cat
    # dset.attrs["parameters"] = "{}".format(param_tuple)

    hdf5.close()



def main():
    cov = np.eye(20)

    draws = random_draws(cov, 5)
    print(draws)

if __name__=='__main__':
    main()
