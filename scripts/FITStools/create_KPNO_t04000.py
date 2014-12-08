from StellarSpectra.grid_tools import *
import StellarSpectra.constants as C

library_path = "/n/holyscratch/panstarrs/iczekala/master_grids/PHOENIX_master.hdf5"
myHDF5Interface = HDF5Interface(library_path)
myInterpolator = Interpolator(myHDF5Interface, avg_hdr_keys=["air", "PHXLUM", "PHXMXLEN",
                    "PHXLOGG", "PHXDUST", "PHXM_H", "PHXREFF", "PHXXI_L", "PHXXI_M", "PHXXI_N", "PHXALPHA", "PHXMASS",
                    "norm", "PHXVER", "PHXTEFF"])
myInstrument = KPNO()

vsini1 = np.arange(0.0, 16.1, 1.)
vsini2 = np.arange(20, 71, 5.)
vsini3 = np.arange(80, 121, 10.)
vsini4 = np.arange(140, 201, 20.)
vsini = np.hstack((vsini1, vsini2, vsini3, vsini4))

outdir = "/n/holyscratch/panstarrs/iczekala/willie/KPNO/"
mycreator = MasterToFITSProcessor(interpolator=myInterpolator, instrument=myInstrument, outdir=outdir,
    points={"temp":np.array([4000, 4250]), "logg":np.array([4.0,4.5]), "Z":np.array([-0.5,0.0]), "vsini":vsini}, processes=32)

mycreator.process_all()