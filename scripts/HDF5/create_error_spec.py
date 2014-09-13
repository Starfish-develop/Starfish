from StellarSpectra import grid_tools

#interface = grid_tools.HDF5Interface("libraries/PHOENIX_TRES_F.hdf5")
#interface = grid_tools.HDF5Interface("libraries/Kurucz_TRES.hdf5")
interface = grid_tools.HDF5Interface("libraries/PHOENIX_SPEX_M.hdf5")

machine = grid_tools.ErrorSpectrumCreator(interface)
machine.process_grid()
