from StellarSpectra import grid_tools
import StellarSpectra.constants as C

#Odyssey

raw_library_path = "/pool/scout0/libraries/raw/PHOENIX/"

mygrid = grid_tools.PHOENIXGridInterface(base=raw_library_path, wl_range=(4900, 7000))

out_path = "libraries/PHOENIX_F_julia_hires.hdf5"
HDF5Stuffer = grid_tools.HDF5GridStuffer(mygrid, filename=out_path, ranges={"temp":(5600, 6600), "logg":(2.5, 6.0),
                                                                            "Z":(-1.5, 1.0), "alpha":(0.0, 0.0)})

HDF5Stuffer.process_grid()



