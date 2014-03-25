from StellarSpectra.grid_tools import *
import StellarSpectra.constants as C
import numpy as np

library_path = "/n/holyscratch/panstarrs/iczekala/master_grids/PHOENIX_master.hdf5"
myHDF5Interface = HDF5Interface(library_path)
myInstrument = Reticon()

vsini1 = np.arange(0.0, 16.1, 1.)
vsini2 = np.arange(20, 71, 5.)
vsini3 = np.arange(80, 121, 10.)
vsini4 = np.arange(140, 201, 20.)
vsini = np.hstack((vsini1, vsini2, vsini3, vsini4))

outdir = "/n/holyscratch/panstarrs/iczekala/willie/Reticon/"
mycreator = MasterToFITSGridProcessor(interface=myHDF5Interface, instrument=myInstrument,
                                      points={"temp":np.arange(2500, 12000, 100), "logg":np.arange(0.0, 5.6, 0.5), "Z":np.arange(-2., 1.1, 0.5),
                                              "vsini":vsini}, flux_unit="f_nu", outdir=outdir, processes=128)

mycreator.process_all()
