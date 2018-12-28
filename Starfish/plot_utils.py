import h5py
import numpy as np


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


def plot(flatchain, base, format=".png"):
    '''
    Make a triangle plot
    '''

    import corner

    labels = [r"$T_\mathrm{eff}$ [K]", r"$\log g$ [dex]", r"$Z$ [dex]",
              r"$v_z$ [km/s]", r"$v \sin i$ [km/s]", r"$\log_{10} \Omega$"]
    figure = corner.corner(flatchain, quantiles=[0.16, 0.5, 0.84],
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
    # matplotlib.rc("axes", labelpad=10)
    import corner

    labels = [r"$M_\ast\enskip [M_\odot]$", r"$i_d \enskip [{}^\circ]$"]
    # r"$r_c$ [AU]", r"$T_{10}$ [K]", r"$q$", r"$\log M_\textrm{CO} \enskip [\log M_\oplus]$",
    # r"$\xi$ [km/s]"]
    inds = np.array([0, 6, ])  # 1, 2, 3, 4, 5])

    K = len(labels)
    fig, axes = plt.subplots(K, K, figsize=(3., 2.5))

    figure = corner.corner(flatchain[:, inds], plot_contours=True,
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
    sample_num = np.arange(len(flatchain[:, 0]))
    sample_num = sample_num[start:end]
    samples = flatchain[start:end]
    plt.rc("ytick", labelsize="x-small")

    fig, ax = plt.subplots(nrows=ndim, sharex=True)
    for i in range(0, ndim):
        ax[i].plot(sample_num, samples[:, i])
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        if labels is not None:
            ax[i].set_ylabel(labels[i])

    ax[-1].set_xlabel("Sample number")
    fig.subplots_adjust(hspace=0)
    fig.savefig(base + "walkers.png")
    plt.close(fig)