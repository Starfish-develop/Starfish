from mpi4py import MPI
import numpy as np
from itertools import zip_longest

#How to determine total number of processes?

#self.comm = MPI.COMM_WORLD if comm is None else comm
#self.rank = self.comm.Get_rank()
#self.size = self.comm.Get_size() - 1

indexes = [i for i in range(233)]

rank = MPI.COMM_WORLD.rank  # The process ID
nprocesses = MPI.COMM_WORLD.size

#Divide up indexes into chunks based on size
#length = len(indexes)

#print(length//size)

#I have a list of "length", and "nprocesses" to deal with it. I want a new list of length nprocesses.
#print(np.ceil(length/size))


def chunk(mylist, n=nprocesses):
    length = len(mylist)
    size = int(length / n)
    chunks = [mylist[0+size*i : size*(i+1)] for i in range(n)] #fill with evenly divisible
    leftover = length - size*n
    edge = size*n
    for i in range(leftover): #backfill each with the last item
        chunks[i%n].append(mylist[edge+i])
    return chunks

chunks = chunk(indexes)

print("Process #{} has".format(rank), chunks[rank])