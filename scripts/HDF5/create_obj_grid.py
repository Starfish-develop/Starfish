from StellarSpectra import grid_tools

#Odyssey
# raw_library_path = "/n/holyscratch/panstarrs/iczekala/raw_libraries/PHOENIX/"
# out_path = "/scratch/" + "PHOENIX_LkCa15.hdf5"

# myHDF5Interface = grid_tools.HDF5Interface("libraries/PHOENIX_submaster.hdf5")
# myHDF5Interface = grid_tools.HDF5Interface("libraries/PHOENIX_LkCa15.hdf5")
myHDF5Interface = grid_tools.HDF5Interface("libraries/PHOENIX_submaster_M.hdf5")

print("Library only spans from", myHDF5Interface.bounds)

# myInstrument = grid_tools.TRES()
myInstrument = grid_tools.SPEX_SXD()
# myInstrument = grid_tools.SPEXMgb()

# out_path = "libraries/" + "PHOENIX_TRES_6000.hdf5"
# out_path = "libraries/" + "PHOENIX_TRES_4000.hdf5"
# out_path = "libraries/" + "PHOENIX_TRES_2300.hdf5"

out_path = "libraries/" + "PHOENIX_SPEX_2300.hdf5"

HDF5InstGridCreator = grid_tools.HDF5InstGridCreator(myHDF5Interface, filename=out_path, Instrument=myInstrument,
                        ranges={"temp":(2300, 4000), "logg":(3.0,5.0), "Z":(-1.,0.5), "alpha":(0.0,0.0)})

HDF5InstGridCreator.process_grid()
