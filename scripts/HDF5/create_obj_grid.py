from StellarSpectra import grid_tools
import StellarSpectra.constants as C

#Odyssey
# raw_library_path = "/n/holyscratch/panstarrs/iczekala/raw_libraries/PHOENIX/"
#scout
# raw_library_path = "libraries/raw/PHOENIX/"

myHDF5Interface = grid_tools.HDF5Interface("libraries/PHOENIX_submaster.hdf5")
myInstrument = grid_tools.TRES()

# out_path = "/scratch/" + "PHOENIX_LkCa15.hdf5"
# out_path = "libraries/" + "PHOENIX_LkCa15.hdf5"
out_path = "libraries/" + "PHOENIX_objgrid.hdf5"

HDF5ObjGridCreator = grid_tools.HDF5ObjGridCreator(myHDF5Interface, filename=out_path, Instrument=myInstrument,
                        ranges={"temp":(5500, 6500), "logg":(4.0,5.0), "Z":(-0.5,0.0), "alpha":(0.0,0.0)})

# print(HDF5ObjGridCreator.process_flux({"temp":5100, "logg":4.5, "Z":0.0, "alpha":0.0}))
print(HDF5ObjGridCreator.points)

HDF5ObjGridCreator.process_grid()
