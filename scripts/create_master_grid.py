from StellarSpectra import grid_tools


#Odyssey

raw_library_path = "/n/holyscratch/panstarrs/iczekala/raw_libraries/PHOENIX/"

mygrid = grid_tools.PHOENIXGridInterface(air=True, norm=True, base=raw_library_path)
print("creating mygrid")

spec = mygrid.load_file({"temp":5000, "logg":3.5, "Z":0.0,"alpha":0.0})
wldict = spec.calculate_log_lam_grid()
print("calculated spec")

out_path = "/n/holyscratch/panstarrs/iczekala/master_grids/" + "PHOENIX_master.hdf5"

HDF5Creator = grid_tools.HDF5GridCreator(mygrid, filename=out_path,
                                    ranges={"temp":(6000, 7000), "logg":(2.0,6.0), "Z":(0.0,0.0), "alpha":(0.0,0.0)},
                                         wldict=wldict, nprocesses=4, chunksize=1)
print("created HDF5creator")
HDF5Creator.process_grid()
print("processed grid")


#Local
#
#mygrid = grid_tools.PHOENIXGridInterface(air=True, norm=True)
#print("creating mygrid")
#
#spec = mygrid.load_file({"temp":5000, "logg":3.5, "Z":0.0,"alpha":0.0})
#wldict = spec.calculate_log_lam_grid()
#print("calculated spec")
#
#out_path = "PHOENIX_master.hdf5"
#
#HDF5Creator = grid_tools.HDF5GridCreator(mygrid, filename=out_path, wldict=wldict, nprocesses=4, chunksize=1,
#                            ranges={"temp":(2300, 10000), "logg":(2.0,6.0), "Z":(0.0,0.0), "alpha":(0.0,0.0)})
#print("created HDF5creator")
#HDF5Creator.process_grid()
#print("processed grid")

