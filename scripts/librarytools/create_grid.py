from Starfish.grid_tools import PHOENIXGridInterface, HDF5Creator, TRES

raw_library_path = "../../libraries/raw/PHOENIX/"
mygrid = PHOENIXGridInterface(base=raw_library_path, wl_range=[4700, 5500])

out_path = "../../libraries/" + "PHOENIX_F.hdf5"

instrument = TRES()

# Limit the range of stellar parameters correspondi to an F star
creator = HDF5Creator(mygrid, out_path, instrument,
                        ranges={"temp":(5800, 6500), "logg":(3.5,6.0),
                        "Z":(-1.5,1.0), "alpha":(0.0,0.0)})

creator.process_grid()
