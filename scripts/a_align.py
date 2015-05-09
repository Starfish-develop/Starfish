#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(prog="single.py", description="Run Starfish fitting model in parallel.")
# Even though these arguments aren't being used, we need to add them.
parser.add_argument("--generate", action="store_true", help="Write out the data, mean model, and residuals for current parameter settings.")
# parser.add_argument("--initPhi", action="store_true", help="Create *phi.json files for each order using values in config.yaml")
parser.add_argument("--optimize", action="store_true", help="Optimize the parameters.")
args = parser.parse_args()


from Starfish import astroseismic_align as AA

if args.optimize:
    AA.optimize()

if args.generate:
    # Save the residuals as a large JSON file, using the known parameters
    AA.generate()
