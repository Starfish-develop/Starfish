'''
Take the MCMC samples from optimizing the emulator, concatenate them and add them to the HDF5 file.
'''

import argparse
parser = argparse.ArgumentParser(prog="add_samples.py",
                description="Decompose the spectra into eigenspectra.")
parser.add_argument("input", help="*.yaml file specifying parameters.")
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
samples = np.array([np.load(base + "samples_w{}.npy".format(i)) for i in range(ncomp)])

hdf5 = h5py.File(filename, "r+")
sdset = hdf5.create_dataset("samples", samples.shape, compression='gzip', compression_opts=9)
sdset[:] = samples
hdf5.close()
