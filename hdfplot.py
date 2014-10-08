#!/usr/bin/env python

#Defines a bunch of global variables and also parses args.
import hdfutils


'''
Script designed to plot the HDF5 output from MCMC runs. It will only plot the first chain, to prevent overwriting.
'''

if hdfutils.args.cov:
    hdfutils.estimate_covariance(hdfutils.flatchainTreeList[0])

if hdfutils.args.paper:
    hdfutils.plot_paper(hdfutils.flatchainTreeList[0], clip_stellar=hdfutils.args.stellar_params)
else:
    hdfutils.plot(hdfutils.flatchainTreeList[0], clip_stellar=hdfutils.args.stellar_params)


