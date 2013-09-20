#import sharedmem as shm
import multiprocessing as mp
from multiprocessing.sharedctypes import Value, Array
import ctypes
import numpy as np

#Create a giant fake block of data

shape = (180,500,1000)

#@profile
def create_shared_data():
    #sdata = shm.empty(shape)
    #sdata[:] = np.random.normal(size=shape)[:]
    # 100 = 1527 Mb
    # 180 = 2747 Mb

    print("Created data")
    return shared_array

#@profile
def create_data(): 
    data = np.random.normal(size=shape)
    print("Created data")
    return data

data = create_shared_data()
#data = create_data()
N = 1028

def process_function(prefactor):#, data = data):
    print("Processing %s" % prefactor)
    return np.sum(prefactor * data)

def process_serial():
    print("Processing serial")
    results = list(map(process_function, np.arange(N)))
    return results

def process_parallel(pool):
    print("Processing parallel")
    results = list(pool.map(process_function, np.arange(N)))
    return results


def main():
    #create_data()
    #process_serial()
    #process_parallel()
    number = Value('i', 7, lock=False)
    # A thread pool of P processes
    pool = mp.Pool(4)
    process_parallel(pool)
    pass

if __name__=="__main__":
    main()

