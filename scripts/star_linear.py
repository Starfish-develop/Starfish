#!/usr/bin/env python

# All of the argument parsing is done in the `parallel.py` module.

import numpy as np
import Starfish
from Starfish import parallel_linear
from Starfish.parallel_linear import args
from Starfish.model import ThetaParam, PhiParam

if args.generate:
    model = parallel_linear.OptimizeTheta(debug=True)

    # Now that the different processes have been forked, initialize them
    pconns, cconns, ps = parallel_linear.initialize(model)

    pars = ThetaParam.from_dict(Starfish.config["Theta"])

    for ((spectrum_id, order_id), pconn) in pconns.items():
        #Parse the parameters into what needs to be sent to each Model here.
        pconn.send(("LNPROB", pars))
        pconn.recv() # Receive and discard the answer so we can send the save
        pconn.send(("SAVE", None))

    # Kill all of the orders
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))

    # Join on everything and terminate
    for p in ps.values():
        p.join()
        p.terminate()

    import sys;sys.exit()


if args.optimize == "Theta":

    # Check to see if the order JSONs exist, if so, then recreate the noise structure according to these.

    # Otherwise assume white noise.
    model = parallel_linear.OptimizeTheta(debug=True)

    # Now that the different processes have been forked, initialize them
    pconns, cconns, ps = parallel_linear.initialize(model)

    def fprob(p):

        # Assume p is [temp, logg, Z, vz, vsini, logOmega]

        pars = ThetaParam(grid=p[0:3], vz=p[3], vsini=p[4], logOmega=p[5])

        #Distribute the calculation to each process
        for ((spectrum_id, order_id), pconn) in pconns.items():
            #Parse the parameters into what needs to be sent to each Model here.
            pconn.send(("LNPROB", pars))

        #Collect the answer from each process
        lnps = np.empty((len(Starfish.data["orders"]),))
        for i, pconn in enumerate(pconns.values()):
            lnps[i] = pconn.recv()

        s = np.sum(lnps)

        print(pars, "lnp:", s)

        if s == -np.inf:
            return 1e99
        else:
            return -s

    start = Starfish.config["Theta"]
    p0 = np.array(start["grid"] + [start["vz"], start["vsini"], start["logOmega"]])

    from scipy.optimize import fmin
    p = fmin(fprob, p0, maxiter=10000, maxfun=10000)
    print(p)
    pars = ThetaParam(grid=p[0:3], vz=p[3], vsini=p[4], logOmega=p[5])
    pars.save()

    # Kill all of the orders
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))

    # Join on everything and terminate
    for p in ps.values():
        p.join()
        p.terminate()

    import sys;sys.exit()

if args.initPhi:
    # Figure out how many models and orders we have
    i_last = len(Starfish.data["orders"]) - 1

    for spec_id in range(len(Starfish.data["files"])):
        for i, order in enumerate(Starfish.data["orders"]):
            fix_c0 = True if i==i_last else False
            if fix_c0:
                cheb = np.zeros((Starfish.config["cheb_degree"] - 1,))
            else:
                cheb = np.zeros((Starfish.config["cheb_degree"],))

            # For each order, create a Phi with these values
            # Automatically reads all of the Phi parameters from config.yaml
            phi = PhiParam(spectrum_id=spec_id, order=int(order), fix_c0=fix_c0, cheb=cheb)
            # Write to CWD using predetermined format string
            phi.save()

if args.optimize == "Cheb":

    model = parallel_linear.OptimizeTheta(debug=True)

    # Now that the different processes have been forked, initialize them
    pconns, cconns, ps = parallel_linear.initialize(model)

    # Initialize to the basics
    pars = ThetaParam.from_dict(Starfish.config["Theta"])

    #Distribute the calculation to each process
    for ((spectrum_id, order_id), pconn) in pconns.items():
        #Parse the parameters into what needs to be sent to each Model here.
        pconn.send(("LNPROB", pars))
        pconn.recv() # Receive and discard the answer so we can send the optimize
        pconn.send(("OPTIMIZE_CHEB", None))

    # Kill all of the orders
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))

    # Join on everything and terminate
    for p in ps.values():
        p.join()
        p.terminate()

    import sys;sys.exit()

if args.sample == "ThetaCheb" or args.sample == "ThetaPhi" or args.sample == "ThetaPhiLines":

    if args.sample == "ThetaCheb":
        model = parallel_linear.SampleThetaCheb(debug=True)
    if args.sample == "ThetaPhi":
        model = parallel_linear.SampleThetaPhi(debug=True)
    if args.sample == "ThetaPhiLines":
        model = parallel_linear.SampleThetaPhiLines(debug=True)


    pconns, cconns, ps = parallel_linear.initialize(model)

    # These functions store the variables pconns, cconns, ps.
    def lnprob(p):
        pars = ThetaParam(grid=p[0:3], vz=p[3], vsini=p[4], logOmega=p[5])
        #Distribute the calculation to each process
        for ((spectrum_id, order_id), pconn) in pconns.items():
            pconn.send(("LNPROB", pars))

        #Collect the answer from each process
        lnps = np.empty((len(Starfish.data["orders"]),))
        for i, pconn in enumerate(pconns.values()):
            lnps[i] = pconn.recv()

        result = np.sum(lnps) # + lnprior
        print("proposed:", p, result)
        return result

    def query_lnprob():
        for ((spectrum_id, order_id), pconn) in pconns.items():
            pconn.send(("GET_LNPROB", None))

        #Collect the answer from each process
        lnps = np.empty((len(Starfish.data["orders"]),))
        for i, pconn in enumerate(pconns.values()):
            lnps[i] = pconn.recv()

        result = np.sum(lnps) # + lnprior
        print("queried:", result)
        return result

    def acceptfn():
        print("Calling acceptfn")
        for ((spectrum_id, order_id), pconn) in pconns.items():
            pconn.send(("DECIDE", True))

    def rejectfn():
        print("Calling rejectfn")
        for ((spectrum_id, order_id), pconn) in pconns.items():
            pconn.send(("DECIDE", False))

    from Starfish.samplers import StateSampler

    start = Starfish.config["Theta"]
    p0 = np.array(start["grid"] + [start["vz"], start["vsini"], start["logOmega"]])

    jump = Starfish.config["Theta_jump"]
    cov = np.diag(np.array(jump["grid"] + [jump["vz"], jump["vsini"], jump["logOmega"]])**2)

    sampler = StateSampler(lnprob, p0, cov, query_lnprob=query_lnprob, acceptfn=acceptfn, rejectfn=rejectfn, debug=True, outdir=Starfish.routdir)

    p, lnprob, state = sampler.run_mcmc(p0, N=args.samples)
    print("Final", p)

    sampler.write()

    # Kill all of the orders
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))

    # Join on everything and terminate
    for p in ps.values():
        p.join()
        p.terminate()

    import sys;sys.exit()
