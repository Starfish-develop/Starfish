#Test to see if the covariance matrix actually works for interpolation errors

import matplotlib.pyplot as plt
import numpy as np
from StellarSpectra.grid_tools import HDF5Interface, Interpolator
import StellarSpectra.constants as C
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.covariance import CovarianceMatrix
import logging

#Set up the logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", filename="log.log",
                    level=logging.DEBUG, filemode="w", datefmt='%m/%d/%Y %I:%M:%S %p')

#interface = HDF5Interface("../libraries/PHOENIX_F.hdf5")
interface = HDF5Interface("../libraries/PHOENIX_TRES_F.hdf5")
dataspec = DataSpectrum.open("../data/WASP14/WASP14-2009-06-14.hdf5", orders=np.array([22]))

interpolator = Interpolator(interface, dataspec, trilinear=True)

params = {"temp":6010, "logg":4.1, "Z":-0.3}
fl, errspec = interpolator(params)

#A good test here would be to create errspec as an arange() so that inside of cov.h we know how we are indexing this
N = np.prod(errspec.shape)
testspec = np.arange(N, dtype="f8")
print(len(fl))
testspec.shape = errspec.shape
print(testspec)

#Create a CovarianceMatrix object
cov = CovarianceMatrix(dataspec, 0, 20, debug=True)


cov.update_interp_errs(testspec)

