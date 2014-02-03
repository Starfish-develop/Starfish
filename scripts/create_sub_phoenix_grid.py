from StellarSpectra import grid_tools


#Odyssey

raw_library_path = "/n/holyscratch/panstarrs/iczekala/raw_libraries/PHOENIX/"
mygrid = grid_tools.PHOENIXGridInterface(air=True, norm=True, base=raw_library_path)

spec = mygrid.load_file({"temp":5000, "logg":3.5, "Z":0.0,"alpha":0.0})
wldict = spec.calculate_log_lam_grid()

out_path = "/n/holyscratch/panstarrs/iczekala/master_grids/" + "PHOENIX_submaster.hdf5"
HDF5Creator = grid_tools.HDF5GridCreator(mygrid, filename=out_path, wldict=wldict,
        ranges={"temp":(6000, 7000), "logg":(3.5,5.5), "Z":(-1.0,0.0), "alpha":(0.0,0.4)},nprocesses=6, chunksize=1)

HDF5Creator.process_grid()

#This requires at least 250MB per process. Spectrum objects (combination of wl, fl) can be very large. Also references
#to interpolator objects are created (but handled properly).


