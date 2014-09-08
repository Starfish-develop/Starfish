from StellarSpectra import grid_tools

#Odyssey
# raw_library_path = "/n/holyscratch/panstarrs/iczekala/raw_libraries/PHOENIX/"
# out_path = "/scratch/" + "PHOENIX_LkCa15.hdf5"

myHDF5Interface = grid_tools.HDF5Interface("libraries/Kurucz_master.hdf5")
#myHDF5Interface = grid_tools.HDF5Interface("libraries/PHOENIX_F_julia.hdf5")
#myHDF5Interface = grid_tools.HDF5Interface("libraries/PHOENIX_F.hdf5")
# myHDF5Interface = grid_tools.HDF5Interface("libraries/PHOENIX_LkCa15.hdf5")
# myHDF5Interface = grid_tools.HDF5Interface("libraries/PHOENIX_M.hdf5")

print("Library only spans from", myHDF5Interface.bounds)

myInstrument = grid_tools.TRES()
# myInstrument = grid_tools.SPEX_SXD()
# myInstrument = grid_tools.SPEXMgb()

#out_path = "libraries/" + "PHOENIX_TRES_F.hdf5"
# out_path = "libraries/" + "PHOENIX_TRES_4000.hdf5"
# out_path = "libraries/" + "PHOENIX_TRES_2300.hdf5"

out_path = "libraries/" + "Kurucz_TRES.hdf5"

#out_path = "libraries/" + "PHOENIX_TRES_F_julia.hdf5"

HDF5InstGridCreator = grid_tools.HDF5InstGridCreator(myHDF5Interface, filename=out_path, Instrument=myInstrument)#,
                        #ranges={"temp":(5400, 6700), "logg":(2.5,5.5), "Z":(-1.5,1.), "alpha":(0.0,0.0)})

HDF5InstGridCreator.process_grid()
