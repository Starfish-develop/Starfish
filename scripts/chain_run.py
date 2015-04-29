#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Measure statistics from MCMC runs, either for a single chain or across multiple chains.")
parser.add_argument("--chain", action="store_true", help="Make a plot of the position of the chains.")
parser.add_argument("-t", "--triangle", action="store_true", help="Make a triangle (staircase) plot of the parameters.")
parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain for burn in.")

args = parser.parse_args()


# Apply the chain diagnostics to everything in the current run directory.
import os
import sys
import subprocess
import contextlib

from Starfish import utils
from glob import glob

# from http://www.astropython.org/snippet/2009/10/chdir-context-manager
@contextlib.contextmanager
def chdir(dirname=None):
  curdir = os.getcwd()
  try:
    if dirname is not None:
      os.chdir(dirname)
    yield
  finally:
    os.chdir(curdir)


paths = glob("s*o*/mc.hdf5")

if args.triangle:
    flatchain = utils.h5read("mc.hdf5", args.burn)
    utils.plot(flatchain, base="")

    cmd = ["chain.py",  "--files", "mc.hdf5", "-t", "--burn", "{}".format(args.burn)]

    for path in paths:
        dirname, fname = path.split("/")

        # CD to this directory using context managers
        with chdir(dirname):
            subprocess.call(cmd)



if args.chain:
    flatchain = utils.h5read("mc.hdf5", args.burn)
    utils.plot_walkers(flatchain, base="")
    cmd = ["chain.py", "--files", "mc.hdf5", "--chain", "--burn", "{}".format(args.burn)]
    print(cmd)

    for path in paths:
        dirname, fname = path.split("/")

        # CD to this directory using context managers
        with chdir(dirname):
            subprocess.call("ls")
            subprocess.call(cmd)
