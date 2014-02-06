import multiprocessing as mp
import numpy as np
import os
#
#def info(title):
#    print(title)
#    print( 'module name:', __name__)
#    if hasattr(os, 'getppid'):  # only available on Unix
#        print( 'parent process:', os.getppid())
#    print( 'process id:', os.getpid())
#
#def f(name):
#    info('function f')
#    print('hello', name)
#
#if __name__ == '__main__':
#    info('main line')
#    p = mp.Process(target=f, args=('bob',))
#    p.start()
#    p.join()


#from multiprocessing import Process, Queue
#
#def f(q):
#    q.put([42, None, 'hello'])
#
#if __name__ == '__main__':
#    q = Queue()
#    p = Process(target=f, args=(q,))
#    p.start()
#    print(q.get())    # prints "[42, None, 'hello']"
#    p.join()
#class Parallel:
#    def __init__(self):
#        #self.pool = mp.Pool(4)
#        #self.process = process
#        pass
#
#    def process(self, parameters):
#        print(parameters)
#
#    def do_parallel(self):
#        pass
#
#def f(args):
#    print(args)

#p = mp.Process(target=f, args=('bob',))
#p.start()
#p.join()


import time
import multiprocessing as mp

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AsyncPlotter():

    def __init__(self, processes=mp.cpu_count()):
        self.processes = processes
        self.pids = []
        self.pid = os.getpid()
        self.sqrt = np.sqrt

        #Stuff queue


    def async_plotter(self, fig, filename):
        fig.savefig(filename)
        plt.close(fig)


    def join(self):
        for i in range(10):
            # Generate random points
            x = np.random.random(10000)
            y = np.random.random(10000)

            # Generate figure
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.scatter(x, y)

            # Add figure to queue
            p = mp.Process(target=self.async_plotter, args=(fig, '%04i.png' % i))
            p.start()
            self.pids.append(p)

        for p in self.pids:
            p.join()

# Create instance of Asynchronous plotter
a = AsyncPlotter()
a.join()


#Class parallel methods will work
#1) If the function is defined at the top level of the module, or the class does not include Pool
#2) The pool object is not defined as a class object but instead as a local variable (it means NO pool object may be
# referenced in the class
#3) They will also work if you define the self.process method as a reference to a TOP level module which is
#outside the class
#4) If the pool object is defined as part of the class, you can still do map as long as process is defined (or referenced
# to outside the scope).

#If the function is bound to the class, it is a METHOD __class__
#If the function is referenced to something outside the class, it is a FUNCTION __class__

#EMCEE works because lnprob is defined somewhere else

#Basically, we need to separate the HDF5 file from the actual processing of the data. It sounds like if you really
#want to use parallel functions with Python, then it's better to use MPI. Will we have to use MPI for our
#cached interpolator? Probably.

#They will not work if you attempt to do both at the same time, because when it passes self.function, self also includes
#the pool object.
#Things can also be complicated by HDF5 files, which probably do not like to be pickled.

#def process(parameters):
#    print(parameters)
#
#class _function_wrapper(object):
#    """
#    This is a hack to make the likelihood function pickleable when ``args``
#    are also included.
#
#    """
#    def __init__(self, f, args):
#        self.f = f
#        self.args = args
#
#    def __call__(self, x):
#        try:
#            return self.f(x, *self.args)
#        except:
#            import traceback
#            print("emcee: Exception while calling your likelihood function:")
#            print("  params:", x)
#            print("  args:", self.args)
#            print("  exception:")
#            traceback.print_exc()
#            raise


#if __name__=="__main__":
#par = Parallel()
#par.do_parallel()

