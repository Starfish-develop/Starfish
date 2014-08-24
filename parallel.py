#parallel implementation outline.

#This is potentially a very powerful implementation, because it means that we can use as many data spectra or orders
# as we want, and there will only be slight overheads. Basically, scaling is flat, as long as we keep adding more
# processors. This might be tricky on Odyssey going across nodes, but it's still pretty excellent.

# Pie-in-the-sky dream about how to hierarchically fit all stars.
# Thinking about the alternative problem, where we actually link region parameters across different stars might be a
# little bit more tricky. The implemenatation might be similar as to what's here, only now we'd also have a
# synchronization step where regions are proposed (from a common distribution) for each star and each order. This
# step could be selected upon all stars, specific order. This could get a little complicated, but then again we knew
# it would be.

# All orders are being selected up and updated based upon which star it is.
# All regions are being updated selected upon all stars, but a specific order.

from multiprocessing import Process, Pipe, cpu_count
import os
import numpy as np

def info(title):
    print(title)
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid())


class Model:
    def __init__(self, npoly=4, debug=False):
        '''
        This is designed to be called within the main processes and then distributed to the other processes
        just so we have something to send stuff to. We don't yet want to load any of the items that would be
        specific to that specific order, which will be customized by the subprocess in `self.initialize()`.
        '''
        self.lnprob = -np.inf

        self.func_dict = {"INIT": self.initialize,
                          "UPDATE": self.update_stellar,
                          "REVERT": self.revert_stellar,
                          "IND": self.independent_sample,
                          }

    def initialize(self, key):
        '''
        Designed to be called after the other processes have been created.

        For multiple DataSpectra, need to specify key.
        '''

        self.id = key
        model_index, order_index = self.id

        print("Initializing model on order {}, DataSpectrum {}.".format(order_index, model_index))

        #Load the appropriate DataSpectrum from DataSpectra list

        #Load the appropriate Instrument from the Instrument list

        #Load the spectrum interpolator

        #Initialiaze the covariance matrix
        self.CovarianceMatrix = None #Cython object


    def evaluate(self):
        '''
        Using the attached covariance matrix (cython object), evaluate the lnposterior.
        '''
        lnp = self.CovarianceMatrix.evaluate()
        return lnp

    def update_stellar(self, params):
        '''
        Update the stellar flux. Evaluate the likelihood, return.
        '''

        #When hierarchically fitting many spectra, we can either send the full stellar parameter list and then parse
        #what we want from it here, OR, we can do the parsing in the master process and always assume we are getting a
        #complete stellar param list here. I think that makes more sense.

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

        #Take a Chebyshev sample, decide if we should keep or revert this

        #Take a Covariance sample, decide if we should keep or revert factorization


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
DataSpectra = [] #list of all DataSpectra we want to use
Instruments = []

model = Model()

spectra = [0]
orders = [22, 23, 24]

#Set up an subprocess for each order. Key is (spectra, order)
pconns = {}
cconns = {}
ps = {}
for spectrum in spectra:
    for order in orders:
        pconn, cconn = Pipe()
        key = (spectrum, order)
        pconns[key], cconns[key] = pconn, cconn
        p = Process(target=model.brain, args=(cconn,))
        p.start()
        ps[key] = p

#Initialize all of the orders based upon which DataSpectrum and which order it is
for key, pconn in pconns.items():
    pconn.send(("INIT", key))

#Update all of the stellar parameters and calculate the lnprob
#This function should act to synchronize all of the processes, since it will need to wait until the lnprob message
# has been put on the queue.

#Parameters are proposed by the MCMC algorithm
params = {"temp":5700, "logg":3.5}
lnprob = 0.
for pconn in pconns.values():
    #Parse the parameters into what needs to be sent to each Model here.
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