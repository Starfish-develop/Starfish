from StellarSpectra import grid_tools
import StellarSpectra.constants as C

#Odyssey
# raw_library_path = "/n/holyscratch/panstarrs/iczekala/raw_libraries/PHOENIX/"
#scout
# raw_library_path = "libraries/raw/PHOENIX/"

myHDF5Interface = grid_tools.HDF5Interface("libraries/PHOENIX_M.hdf5")
myInstrument = grid_tools.SPEX()

# out_path = "/scratch/" + "PHOENIX_LkCa15.hdf5"
# out_path = "libraries/" + "PHOENIX_LkCa15.hdf5"
out_path = "libraries/" + "PHOENIX_SPEX_M.hdf5"

HDF5ObjGridCreator = grid_tools.HDF5ObjGridCreator(myHDF5Interface, filename=out_path, Instrument=myInstrument,
                        ranges={"temp":(2300, 2400), "logg":(4.5,5.0), "Z":(-0.5,0.0), "alpha":(0.0,0.0)})

HDF5ObjGridCreator.process_grid()
