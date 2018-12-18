from Starfish import config
from Starfish.grid_tools import PHOENIXGridInterface, HDF5Creator, TRES

mygrid = PHOENIXGridInterface(base=config.grid["raw_path"], wl_range=config.grid["wl_range"])

instrument = TRES()

creator = HDF5Creator(mygrid, config.grid["hdf5_path"], instrument,
    ranges=config.grid["parrange"])

creator.process_grid()
