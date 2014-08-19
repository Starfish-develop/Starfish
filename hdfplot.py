#!/usr/bin/env python
import os
import numpy as np


'''
Script designed to plot the HDF5 output from MCMC runs.
'''

#Plot kw
label_dict = {"temp":r"$T_{\rm eff}$", "logg":r"$\log_{10} g$", "Z":r"$[{\rm Fe}/{\rm H}]$", "alpha":r"$[\alpha/{\rm Fe}]$",
    "vsini":r"$v \sin i$", "vz":r"$v_z$",  "vz1":r"$v_z1$", "logOmega1":r"$\log{10} \Omega 1$", "logOmega":r"$\log_{10} \Omega$", "logc0":r"$\log_{10} c_0$",
    "c1":r"$c_0$", "c2":r"$c_1$", "c3":r"$c_3$",
    "sigAmp":r"$b$", "logAmp":r"$\log_{10} a_{\rm g}$", "l":r"$l$",
    "h":r"$h$", "loga":r"$\log_{10} a$", "mu":r"$\mu$", "sigma":r"$\sigma$"}


import argparse
parser = argparse.ArgumentParser(description="Plot the MCMC output. Default is to plot everything possible.")
parser.add_argument("HDF5file", help="The HDF5 file containing the MCMC samples.")
parser.add_argument("--lnprob", help="The HDF5 file containing the lnprobability chains.")
parser.add_argument("-o", "--outdir", default="mcmcplot", help="Output directory to contain all plots.")
parser.add_argument("--clobber", action="store_true", help="Overwrite existing output directory?")

parser.add_argument("-t", "--triangle", action="store_true", help="Make a triangle (staircase) plot of the parameters.")
parser.add_argument("--chain", action="store_true", help="Make a plot of the position of the chains.")
parser.add_argument("--range", nargs=2, help="start and end ranges for chain plot, separated by WHITESPACE")
parser.add_argument("--acor", action="store_true", help="Calculate the autocorrelation of the chain")
parser.add_argument("--acor-window", type=int, default=50, help="window to compute acor with")

parser.add_argument("--cov", action="store_true", help="Estimate the covariance between two parameters.")

parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain "
                                                        "for burn in.")
parser.add_argument("--thin", type=int, default=1, help="Thin the chain by this factor. E.g., --thin 100 will take "
                                                        "every 100th sample.")
parser.add_argument("--stellar_params", nargs="*", default="all", help="A list of which stellar parameters to plot, "
                                                                    "separated by WHITESPACE. Default is to plot all.")
args = parser.parse_args()

#Check to see if outdir exists. If --clobber, overwrite, otherwise exit.
if os.path.exists(args.outdir):
    if not args.clobber:
        import sys
        sys.exit("Error: --outdir already exists and --clobber is not set. Exiting.")
else:
    os.makedirs(args.outdir)

args.outdir += "/"


#Load the HDF5 file
import h5py

hdf5 = h5py.File(args.HDF5file, "r")
stellar = hdf5.get("stellar")
if stellar is None:
    sys.exit("HDF5 file contains no stellar samples.")

if args.lnprob:
    hdf5lnprob = h5py.File(args.lnprob, "r")
    stellarLnprob = hdf5lnprob.get("stellar")
    stellarLnprob = stellarLnprob[args.burn::args.thin]

#Determine which parameters we want to plot from --stellar-params
stellar_tuple = stellar.attrs["parameters"]
stellar_tuple = tuple([param.strip("'() ") for param in stellar_tuple.split(",")])

print("Thinning by ", args.thin)
print("Burning out first {} samples".format(args.burn))
stellar = stellar[args.burn::args.thin]

if args.stellar_params == "all":
    stellar_params = stellar_tuple
else:
    stellar_params = args.stellar_params
    #Figure out which rows we need to select
    index_arr = []
    for param in stellar_params:
        #What index is this param in stellar_tuple?
        index_arr += [stellar_tuple.index(param)]
    index_arr = np.array(index_arr)
    stellar = stellar[:, index_arr]

stellar_labels = [label_dict[key] for key in stellar_params]

#Was good
#
# def find_cov(name):
#     if "cov" in name:
#         return True
#     return None
#
# def find_region(name):
#     if "cov_region" in name:
#         return True
#     return None
#
# #Determine how many orders, if there is global covariance, or regions
# #choose the first chain
# orders = [int(key) for key in hdf5.keys() if key != "stellar"]
# orders.sort()
#
# yes_cov = hdf5.visit(find_cov)
# yes_region = hdf5.visit(find_region)
# #Order list will always be a 2D list, with the items being flatchains
# cheb_parameters = hdf5.get("{}/cheb".format(orders[0])).attrs["parameters"]
# cheb_tuple = tuple([param.strip("'() ") for param in cheb_parameters.split(",")])
# cheb_labels = [label_dict[key] for key in cheb_tuple]
# if yes_cov:
#     cov_parameters = hdf5.get("{}/cov".format(orders[0])).attrs["parameters"]
#     cov_tuple = tuple([param.strip("'() ") for param in cov_parameters.split(",")])
#     cov_labels = [label_dict[key] for key in cov_tuple]
#
# #                            order22,     order23
# ordersList = [] #2D list of [[cheb, cov], [cheb, cov],     []]
# #                                   order22                              order23
# ordersListRegions = [] #2D list of [[region00, region01, region02,...], [region00,] ]
# for order in orders:
#
#     temp = [hdf5.get("{}/cheb".format(order))]
#     if yes_cov:
#         temp += [hdf5.get("{}/cov".format(order))]
#
#     #accumulate all of the orders
#     ordersList += [temp]
#
#     if yes_region:
#         #Determine how many regions we have, if any
#         temp = []
#         #figure out list of which regions are in this order
#         regionKeys = [key for key in hdf5["{}".format(order)].keys() if "cov_region" in key]
#
#         for key in regionKeys:
#             temp += [hdf5.get("{}/{}".format(order, key))[:]]
#         ordersListRegions += [temp]
#
# ordersList = [[flatchain[args.burn::args.thin] for flatchain in subList] for subList in ordersList]
#
# if args.lnprob:
#     ordersListLnprob = []
#     ordersListRegionsLnprob = []
#     for order in orders:
#         temp = [hdf5lnprob.get("{}/cheb".format(order))]
#         if yes_cov:
#             temp += [hdf5lnprob.get("{}/cov".format(order))]
#
#         #accumulate all of the orders
#         ordersListLnprob += [temp]
#
#         if yes_region:
#             #Determine how many regions we have, if any
#             temp = []
#             #figure out list of which regions are in this order
#             regionKeys = [key for key in hdf5lnprob["{}".format(order)].keys() if "cov_region" in key]
#
#             for key in regionKeys:
#                 temp += [hdf5lnprob.get("{}/{}".format(order, key))[:]]
#             ordersListRegionsLnprob += [temp]
#
#     ordersListLnprob = [[lnchain[args.burn::args.thin] for lnchain in subList] for subList in ordersListLnprob]

if args.triangle:
    import matplotlib
    matplotlib.rc("axes", labelsize="large")
    import triangle
    figure = triangle.corner(stellar, labels=stellar_labels, quantiles=[0.16, 0.5, 0.84],
                             show_titles=True, title_args={"fontsize": 12})
    figure.savefig(args.outdir + "stellar_triangle.png")

    #Now plot all the other parameters
    # for i, order in enumerate(orders):
    #     orderList = ordersList[i]
    #     cheb = orderList[0]
    #     figure = triangle.corner(cheb, labels=cheb_labels, quantiles=[0.16, 0.5, 0.84],
    #                              show_titles=True, title_args={"fontsize": 12})
    #     figure.savefig(args.outdir + "{}_cheb_triangle.png".format(order))
    #
    #     if yes_cov:
    #         cov = orderList[1]
    #         figure = triangle.corner(cov, labels=cov_labels, quantiles=[0.16, 0.5, 0.84],
    #                                  show_titles=True, title_args={"fontsize": 12})
    #         figure.savefig(args.outdir + "{}_cov_triangle.png".format(order))

def plot_walkers(filename, samples, lnprobs=None, start=0, end=-1, labels=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    # majorLocator = MaxNLocator(nbins=4)
    ndim = len(samples[0, :])
    sample_num = np.arange(len(samples[:,0]))
    sample_num = sample_num[start:end]
    samples = samples[start:end]
    plt.rc("ytick", labelsize="x-small")

    if lnprobs is not None:
        fig, ax = plt.subplots(nrows=ndim + 1, sharex=True)
        ax[0].plot(sample_num, lnprobs[start:end])
        ax[0].set_ylabel("lnprob")
        for i in range(0, ndim):
            ax[i+1].plot(sample_num, samples[:,i])
            ax[i+1].yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
            if labels is not None:
                ax[i+1].set_ylabel(labels[i])

    else:
        fig, ax = plt.subplots(nrows=ndim, sharex=True)
        for i in range(0, ndim):
            ax[i].plot(sample_num, samples[:,i])
            ax[i].yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
            if labels is not None:
                ax[i].set_ylabel(labels[i])

    ax[-1].set_xlabel("Sample number")
    fig.subplots_adjust(hspace=0)
    fig.savefig(filename)
    plt.close(fig)

if args.chain:
    if args.range is not None:
        start, end = [int(el) for el in args.range]
    else:
        start=0
        end=-1

    if args.lnprob:
        plot_walkers(args.outdir + "stellar_chain_pos.png", stellar, stellarLnprob, start=start, end=end)
    else:
        plot_walkers(args.outdir + "stellar_chain_pos.png", stellar, start=start, end=end)

    #Now plot the orders
    for i, order in enumerate(orders):
        orderList = ordersList[i]
        cheb = orderList[0]

        if args.lnprob:
            orderListLnprob = ordersListLnprob[i]
            chebLnprob = orderListLnprob[0]
            plot_walkers(args.outdir + "{}_cheb_chain_pos.png".format(order), cheb, chebLnprob, start=start, end=end)

        else:
            plot_walkers(args.outdir + "{}_cheb_chain_pos.png".format(order), cheb, start=start, end=end)

        if yes_cov:
            cov = orderList[1]
            plot_walkers(args.outdir + "{}_cov_chain_pos.png".format(order), cov, start=start, end=end)

            if args.lnprob:
                covLnprob = orderListLnprob[1]
                plot_walkers(args.outdir + "{}_cov_chain_pos.png".format(order), cov, covLnprob, start=start,
                             end=end)
            else:
                plot_walkers(args.outdir + "{}_cov_chain_pos.png".format(order), cov, start=start, end=end)

    if yes_region:
        #Now plot the regions
        for i, order in enumerate(orders):
            orderList = ordersListRegions[i]

            if args.lnprob:
                orderListLnprob = ordersListRegionsLnprob[i]

                #Now cycle through all regions in this list?
                for j, (samples, lnprob) in enumerate(zip(orderList, orderListLnprob)):
                    plot_walkers(args.outdir + "{}_cov_region{:0>2}_chain_pos.png".format(order, j), samples,
                                 lnprobs=lnprob, start=start, end=end)

            else:
                #Now cycle through all regions in this list?
                for j, samples in enumerate(orderList):
                    plot_walkers(args.outdir + "{}_cov_region{:0>2}_chain_pos.png".format(order, j), samples,
                                 start=start, end=end)

if args.cov:
    '''
    Estimate the covariance of the remaining samples.
    '''
    print("Estimating covariances")
    print(stellar_params)

    cov = np.cov(stellar, rowvar=0)

    print("Standard deviation")
    print(np.sqrt(np.diag(cov)))

    print("Covariance")
    print(cov)

    #Now try correlation coefficient
    cor = np.corrcoef(stellar, rowvar=0)
    print("Correlation coefficient")
    print(cor)

# plot_walkers(self.outdir + self.fname + "_chain_pos.png", samples, labels=self.param_tuple)
# plt.close(figure)

if args.acor:
    from emcee import autocorr
    print("Stellar Autocorrelation")
    acors = autocorr.integrated_time(stellar, axis=0, window=args.acor_window)
    for param,acor in zip(stellar_tuple, acors):
        print("{} : {}".format(param, acor))

hdf5.close()
if args.lnprob:
    hdf5lnprob.close()