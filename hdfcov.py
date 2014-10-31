#!/usr/bin/env python

#Defines a bunch of global variables and also parses args.
import hdfutils

if hdfutils.args.cov:
    hdfutils.estimate_covariance(hdfutils.flatchainList[0])
