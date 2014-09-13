from StellarSpectra import grid_tools
import StellarSpectra.constants as C

raw_library_path = "libraries/raw/PHOENIX/"

mygrid = grid_tools.PHOENIXGridInterface(base=raw_library_path)

#out_path = "libraries/" + "PHOENIX_M.hdf5"
out_path = "libraries/" + "PHOENIX_F.hdf5"


HDF5Stuffer = grid_tools.HDF5GridStuffer(mygrid, filename=out_path,
                        ranges={"temp":(5300, 6600), "logg":(2.5,5.0), "Z":(-1.5,0.5), "alpha":(0.0,0.0)})

HDF5Stuffer.process_grid()