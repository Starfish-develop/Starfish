from mpi4py import MPI
import h5py

rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)


class Parallel:
    def __init__(self):
        self.file = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
        self.dset = self.file.create_dataset('test', (4,), dtype='i')
        pass

    def process(self, parameters):
        return parameters + 2

    def do_parallel(self):
        self.dset[rank] = self.process(rank)
        self.dset.attrs["monkey"] = "chimp" #OK because it is done by all processes

    def __del__(self):
        self.file.close()

par = Parallel()
par.do_parallel()

#In the paradigm of the HDF5 grid creator, we'll have to create all the folders and attributes collectively
#However, the actual processing of the data will have to be done in parallel
