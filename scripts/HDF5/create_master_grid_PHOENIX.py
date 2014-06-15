from StellarSpectra import grid_tools
import StellarSpectra.constants as C

#Odyssey

raw_library_path = "/n/holyscratch/panstarrs/iczekala/raw_libraries/PHOENIX/"
mygrid = grid_tools.PHOENIXGridInterface(base=raw_library_path)

out_path = "/scratch/" + "PHOENIX_master.hdf5"
HDF5Stuffer = grid_tools.HDF5GridStuffer(mygrid, filename=out_path)

HDF5Stuffer.process_grid()



