'''
Take an HDF5 file and downsize it to a PCA grid, write out to HDF5.
'''

from Starfish.emulator import PCAGrid

cfg = {"grid": "libraries/PHOENIX_SPEX_M.hdf5",
       "ranges": {
           "temp" : [2800, 3400],
           "logg": [4.5, 6.0],
           "Z": [-0.5, 1.0],
           "alpha": [0.0, 0.0],
       },
       "wl" : [20100, 23900],
       "test_index" : 500,
       "ncomp" : 5}

pca = PCAGrid.from_cfg(cfg)
pca.write("libraries/PHOENIX_SPEX_M_PCA.hdf5")
