import Starfish
from Starfish.grid_tools import PHOENIXGridInterface, HDF5Creator, TRES

mygrid = PHOENIXGridInterface(base=Starfish.grid["raw_path"], wl_range=Starfish.grid["wl_range"])

instrument = TRES()

creator = HDF5Creator(mygrid, Starfish.grid["hdf5_path"], instrument,
    ranges=Starfish.grid["parrange"])

creator.process_grid()
