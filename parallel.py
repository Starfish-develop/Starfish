#simple parallel implementatino

from multiprocessing import Process, Pipe, cpu_count
import os
import numpy as np

def info(title):
    print(title)
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid())


#Eventually we will want a separate OrderModel started on each process. The only thing we send back and forth is
# parameter information.

#after the models have been created, now fork
#
# class Model:
#     def __init__(self):
#         self.param = 0
#
#     def square_param(self):
#         return self.param**2
#
#     def set_param(self, param):
#         self.param = param
#
#     def get_communication(self, conn):
#         info("Model")
#         print("Communication was", conn.recv())
#         conn.send("Hello back")

class Model:
    def __init__(self, DataSpectrum, npoly=4, debug=False):
        self.DataSpectrum = DataSpectrum
        self.lnprob = -np.inf

        self.func_dict = {"INIT": self.initialize,
                          "UPDATE": self.update_stellar,
                          "REVERT": self.revert_stellar,
                          "IND" : self.independent_sample,
                          }

    def initialize(self, order_index=0):
        '''
        Designed to be called after the other processes have been created.
        '''
        print("Initializing model on order {}.".format(order_index))

        #Load the spectrum interpolator


        pass

    def update_stellar(self, params):
        '''
        Update the stellar flux. Evaluate the likelihood, return.
        '''
        print("Updating stellar parameters", params)
        lnprob = 1
        return lnprob

    def revert_stellar(self, *args):
        '''
        Revert the downsampled stellar flux.
        '''
        print("Reverting stellar parameters")
        pass

    def independent_sample(self, *args):
        '''
        Now, do all the sampling that is specific to this order.
        '''

        #Take the lnprob from the previous iteration


        print("Doing indepedent sampling.")

        #Don't return anything to the main process.

    def brain(self, conn):
        self.conn = conn
        alive = True
        while alive:
            #Keep listening for messages put on the Pipe
            alive = self.interpret()
        self.conn.send("DEAD")

    def interpret(self):
        #Interpret the messages being put into the Pipe, and do something with them.
        #Right now we only expect one function and one argument but we could generalize this to **args
        #info("brain")

        fname, arg = self.conn.recv()
        print("{} received message {}".format(os.getpid(), (fname, arg)))

        func = self.func_dict.get(fname, False)
        if func:
            response = func(arg)
        else:
            print("Given an unknown function {}, assuming kill signal.".format(fname))
            return False

        #Functions only return a response other than None when they want them communicated back to the master process.
        if response:
            print("{} sending back {}".format(os.getpid(), response))
            self.conn.send(response)
        return True

#Message sent will alwas be a 2-arg tuple ()

#Just create one order model. Then when the process forks, each process has an order model. Initialization will have
# to be done through the pipe entirely.
model = Model(None)

orders = [22, 23, 24]

#Set up an subprocess for each order
pconns = {}
cconns = {}
ps = {}
for order in orders:
    pconn, cconn = Pipe()
    pconns[order], cconns[order] = pconn, cconn
    p = Process(target=model.brain, args=(cconn,))
    p.start()
    ps[order] = p

#Initialize all of the orders
for order, pconn in pconns.items():
    pconn.send(("INIT", order))

#Update all of the stellar parameters and calculate the lnprob
#This function should act to synchronize all of the processes, since it will need to wait until the lnprob message
# has been put on the queue.
params = {"temp":5700, "logg":3.5}
lnprob = 0.
for pconn in pconns.values():
    pconn.send(("UPDATE", params))
    lnprob += pconn.recv()

print("Combined lnprob is {}".format(lnprob))

#Decide we don't want these stellar params, now revert.
for pconn in pconns.values():
    pconn.send(("REVERT", None))

#Go ahead and do all of the independent sampling
for pconn in pconns.values():
    pconn.send(("IND", None))

params = {"temp":5700, "logg":3.6}
lnprob = 0.
for pconn in pconns.values():
    pconn.send(("UPDATE", params))
    lnprob += pconn.recv()

print("Combined lnprob is {}".format(lnprob))

#Kill all of the orders
for pconn in pconns.values():
    pconn.send(("DIE", None))

#Join on everything and terminate
for p in ps.values():
    p.join()
    p.terminate()


print("This should only be printed once")
import sys;sys.exit()


#Also, all subprocesses will inherit pipe file descriptors created in the master process.
# http://www.pushingbits.net/posts/python-multiprocessing-with-pipes/
# thus, to really close a pipe, you need to close it in every subprocess.