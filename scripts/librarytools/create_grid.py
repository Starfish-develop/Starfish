import Starfish
from Starfish.config.grid_tools import PHOENIXGridInterface, HDF5Creator, TRES

mygrid = PHOENIXGridInterface(base=Starfish.config.grid["raw_path"], wl_range=Starfish.config.grid["wl_range"])

instrument = TRES()

creator = HDF5Creator(mygrid, Starfish.config.grid["hdf5_path"], instrument,
    ranges=Starfish.config.grid["parrange"])

creator.process_grid()
