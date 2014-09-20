from StellarSpectra import grid_tools
import StellarSpectra.constants as C

raw_library_path = "libraries/raw/PHOENIX/"

mygrid = grid_tools.PHOENIXGridInterface(base=raw_library_path, wl_range=[19500, 25000])

out_path = "libraries/" + "PHOENIX_M_julia_hires.hdf5"
#out_path = "libraries/" + "PHOENIX_F.hdf5"


HDF5Stuffer = grid_tools.HDF5GridStuffer(mygrid, filename=out_path,
                        ranges={"temp":(2500, 3800), "logg":(3.5,6.0), "Z":(-1.5,1.0), "alpha":(0.0,0.0)})

HDF5Stuffer.process_grid()