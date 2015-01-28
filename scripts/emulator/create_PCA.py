'''
Take an HDF5 file and downsize it to a PCA grid, then write out to HDF5.
'''

from Starfish.grid_tools import HDF5Interface
from Starfish.emulator import PCAGrid

# Load the HDF5 interface

myHDF5 = HDF5Interface("../../libraries/PHOENIX_TRES_F.hdf5")
ncomp = 40

pca = PCAGrid.create(myHDF5, ncomp)
pca.write("../../libraries/PHOENIX_TRES_F_PCA.hdf5")
