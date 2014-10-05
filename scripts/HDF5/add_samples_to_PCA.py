import h5py
import numpy as np

filename = "libraries/PHOENIX_SPEX_M_PCA.hdf5"
ncomp = 5

#Load individual samples and then concatenate them
base = "output/emulator/Gl51/"
samples = np.array([np.load(base + "samples_w{}.npy".format(i)) for i in range(ncomp)])

hdf5 = h5py.File(filename, "r+")
sdset = hdf5.create_dataset("samples", samples.shape, compression='gzip', compression_opts=9)
sdset[:] = samples
hdf5.close()
