from StellarSpectra import grid_tools
import StellarSpectra.constants as C

#Odyssey
# raw_library_path = "/n/holyscratch/panstarrs/iczekala/raw_libraries/PHOENIX/"
#scout
raw_library_path = "libraries/raw/PHOENIX/"

mygrid = grid_tools.PHOENIXGridInterface(base=raw_library_path)

# out_path = "/scratch/" + "PHOENIX_LkCa15.hdf5"
# out_path = "libraries/" + "PHOENIX_LkCa15.hdf5"
out_path = "libraries/" + "PHOENIX_M.hdf5"

HDF5Stuffer = grid_tools.HDF5GridStuffer(mygrid, filename=out_path,
                        ranges={"temp":(2300, 5000), "logg":(2.5,5.5), "Z":(-1.0,0.5), "alpha":(0.0,0.0)})

HDF5Stuffer.process_grid()