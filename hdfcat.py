#!/usr/bin/env python

'''
Script designed to concatenate multiple HDF5 files from MCMC runs into one.
'''

import hdfutils

hdfutils.cat_list(hdfutils.args.output, hdfutils.flatchainTreeList)