from StellarSpectra import grid_tools
import StellarSpectra.constants as C

#Odyssey

raw_library_path = "/n/holyscratch/panstarrs/iczekala/raw_libraries/PHOENIX/"
mygrid = grid_tools.PHOENIXGridInterface(base=raw_library_path)

#spec = mygrid.load_file({"temp":5000, "logg":3.5, "Z":0.0,"alpha":0.0})
#wl_dict = spec.calculate_log_lam_grid()

wl_dict = grid_tools.create_log_lam_grid(wl_start=3000, wl_end=13000, min_vc=0.06/C.c_kms)

out_path = "/scratch/" + "PHOENIX_master.hdf5"
HDF5Creator = grid_tools.HDF5GridCreator(mygrid, filename=out_path, wl_dict=wl_dict, nprocesses=10, chunksize=1)

HDF5Creator.process_grid()

#This requires at least 250MB per process. Spectrum objects (combination of wl, fl) can be very large. Also references
#to interpolator objects are created (but handled properly).


