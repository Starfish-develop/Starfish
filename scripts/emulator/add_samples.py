'''
Take the MCMC samples from optimizing the emulator, concatenate them and add them to the HDF5 file.
'''

import argparse
parser = argparse.ArgumentParser(prog="add_samples.py",
                description="Decompose the spectra into eigenspectra.")
parser.add_argument("input", help="*.yaml file specifying parameters.")
parser.add_argument("--params", action="store_true", help="Store optimized parameters instead"
                    "of samples parameters.")
args = parser.parse_args()

import yaml
import h5py
import numpy as np

f = open(args.input)
cfg = yaml.load(f)
f.close()

filename = cfg["PCA_grid"]
ncomp = cfg["ncomp"]

#Load individual samples and then concatenate them
base = cfg["outdir"]

hdf5 = h5py.File(filename, "r+")

if args.params:
    params = np.load(base + "params.npy")
    pdset = hdf5.create_dataset("params", params.shape, compression="gzip", compression_opts=9)
    pdset[:] = params
else:
    samples = np.array([np.load(base + "samples_w{}.npy".format(i)) for i in range(ncomp)])
    sdset = hdf5.create_dataset("samples", samples.shape, compression='gzip', compression_opts=9)
    sdset[:] = samples

hdf5.close()
