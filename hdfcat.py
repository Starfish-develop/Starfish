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

#Concatenate all of the stellar samples
cat_stellar = np.concatenate(stellarlist, axis=0)

#Write this out to the new file
hdf5 = h5py.File(args.output, "w")
dset = hdf5.create_dataset("stellar", cat_stellar.shape, compression='gzip', compression_opts=9)
dset[:] = cat_stellar
dset.attrs["parameters"] = stellarlist[0].attrs["parameters"]
hdf5.close()


#Later, do this for each of the same orders.

[hdf5.close for hdf5 in hdf5list]