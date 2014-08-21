#!/usr/bin/env python

#Defines a bunch of global variables and also parses args.
import hdfutils



'''
Script designed to plot the HDF5 output from MCMC runs.
'''

for ftree in hdfutils.flatchainTreeList:
    hdfutils.plot(ftree, clip_stellar=hdfutils.args.stellar_params)

