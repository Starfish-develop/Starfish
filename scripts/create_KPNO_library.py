from StellarSpectra.grid_tools import *
import StellarSpectra.constants as C

myHDF5Interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
myInterpolator = Interpolator(myHDF5Interface, avg_hdr_keys=["air", "PHXLUM", "PHXMXLEN",
                    "PHXLOGG", "PHXDUST", "PHXM_H", "PHXREFF", "PHXXI_L", "PHXXI_M", "PHXXI_N", "PHXALPHA", "PHXMASS",
                    "norm", "PHXVER", "PHXTEFF"])
myInstrument = KPNO()

mycreator = MasterToFITSProcessor(interpolator=myInterpolator, instrument=myInstrument, outdir="willie/KPNO/", points={"temp":np.arange(3500, 9751, 250), "logg":np.arange(1, 5.1, 0.5), "Z":np.arange(-0.5, 0.6, 0.5)})
