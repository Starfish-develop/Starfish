from StellarSpectra import grid_tools

mygrid = grid_tools.KuruczGridInterface()

out_path = "libraries/Kurucz_master.hdf5"
HDF5Stuffer = grid_tools.HDF5GridStuffer(mygrid, filename=out_path)
HDF5Stuffer.process_grid()


