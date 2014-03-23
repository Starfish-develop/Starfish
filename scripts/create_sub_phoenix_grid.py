from StellarSpectra import grid_tools
import StellarSpectra.constants as C

#Odyssey

raw_library_path = "/n/holyscratch/panstarrs/iczekala/raw_libraries/PHOENIX/"
mygrid = grid_tools.PHOENIXGridInterface(base=raw_library_path)

out_path = "/scratch/" + "PHOENIX_submaster.hdf5"
HDF5Stuffer = grid_tools.HDF5GridStuffer(mygrid, filename=out_path,
                        ranges={"temp":(5000, 7000), "logg":(3.5,5.5), "Z":(-1.0,0.0), "alpha":(0.0,0.4)})

HDF5Stuffer.process_grid()

#This requires at least 250MB per process. Spectrum objects (combination of wl, fl) can be very large. Also references
#to interpolator objects are created (but handled properly).