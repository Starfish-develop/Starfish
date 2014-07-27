#!/usr/bin/env python
import os
import numpy as np


'''
Script designed to concatenate multiple HDF5 files from MCMC runs into one.
'''

import argparse
parser = argparse.ArgumentParser(description="Concatenate multiple HDF5 files into one.")
parser.add_argument("--dir", action="store_true", help="Concatenate all of the flatchains stored within run* "
                              "folders in the current directory. Designed to collate runs from a JobArray.")
parser.add_argument("--files", nargs="+", help="The HDF5 files containing the MCMC samples, separated by whitespace.")
parser.add_argument("-o", "--output", default="combined.hdf5", help="Output HDF5 file.")
parser.add_argument("--clobber", action="store_true", help="Overwrite existing file?")
parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain "
                                                        "for burn in.")
parser.add_argument("--thin", type=int, default=1, help="Thin the chain by this factor. E.g., --thin 100 will take "
                                                        "every 100th sample.")
args = parser.parse_args()

#Check to see if output exists. If --clobber, overwrite, otherwise exit.
if os.path.exists(args.output):
    if not args.clobber:
        import sys
        sys.exit("Error: --output already exists and --clobber is not set. Exiting.")

if args.dir:
    #assemble all of the flatchains.hdf5 files from the run* subdirectories.
    import glob
    folders = glob.glob("run*")
    files = [folder + "/flatchains.hdf5" for folder in folders]

elif args.files:
    assert len(args.files) >= 2, "Must provide 2 or more HDF5 files to combine."
    files = args.files
else:
    import sys
    sys.exit("Must specify either --dir or --files")

import h5py

hdf5list = [h5py.File(file, "r") for file in files]
stellarlist = [hdf5.get("stellar") for hdf5 in hdf5list]

for stellar in stellarlist:
    assert stellar.attrs["parameters"] == stellarlist[0].attrs["parameters"], "Parameter lists do not match."
    assert stellar.shape[1] == stellarlist[0].shape[1], "Different number of parameters."

stellar_parameters = stellarlist[0].attrs["parameters"]

def find_cov(name):
    if name == "cov":
        return True
    return None

def find_region(name):
    if "cov_region" in name:
        return True
    return None

#Determine how many orders, if there is global covariance, or regions
#choose the first chain
hdf5 = hdf5list[0]
orders = [int(key) for key in hdf5.keys() if key != "stellar"]
orders.sort()

yes_cov = hdf5.visit(find_cov)
yes_region = hdf5.visit(find_region)

# give this a key relative from the top, and it will return a list of all flatchains
def get_flatchains(key):
    return [hdf5.get(key)[:] for hdf5 in hdf5list]

cheb_parameters = hdf5list[0].get("{}/cheb".format(orders[0])).attrs["parameters"]
if yes_cov:
    cov_parameters = hdf5list[0].get("{}/cov".format(orders[0])).attrs["parameters"]

#Order list will always be a 2D list, with the items being flatchains
ordersList = []
for order in orders:

    temp = [get_flatchains("{}/cheb".format(order))]
    if yes_cov:
        temp += [get_flatchains("{}/cov".format(order))]

    #TODO: do something about regions here

    #accumulate all of the orders
    ordersList += [temp]

print("Thinning by ", args.thin)
print("Burning out first {} samples".format(args.burn))
stellarlist = [stellar[args.burn::args.thin] for stellar in stellarlist]
#a triple list comprehension is bad for readability, but I can't think of something better
ordersList = [[[flatchain[args.burn::args.thin] for flatchain in subList] for subList in orderList] for orderList in
              ordersList]

#Concatenate all of the stellar samples
cat_stellar = np.concatenate(stellarlist, axis=0)

#Concatenate all of the order samples
cat_orders = [[np.concatenate(subList, axis=0) for subList in orderList] for orderList in ordersList]

#Write this out to the new file
hdf5 = h5py.File(args.output, "w")
dset = hdf5.create_dataset("stellar", cat_stellar.shape, compression='gzip', compression_opts=9)
dset[:] = cat_stellar
dset.attrs["parameters"] = stellar_parameters
for i, order in enumerate(orders):
    orderList = cat_orders[i]
    dset = hdf5.create_dataset("{}/cheb".format(order), orderList[0].shape, compression='gzip', compression_opts=9)
    dset[:] = orderList[0]
    dset.attrs["parameters"] = cheb_parameters
    if yes_cov:
        dset = hdf5.create_dataset("{}/cov".format(order), orderList[1].shape, compression='gzip', compression_opts=9)
        dset[:] = orderList[1]
        dset.attrs["parameters"] = cov_parameters

#close the new file
hdf5.close()
#close all the old files
[hdf5.close() for hdf5 in hdf5list]