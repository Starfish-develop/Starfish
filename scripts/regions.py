#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(prog="regions.py", description="Identify region zones.")
parser.add_argument("file", help="JSON file containing the data, model, and residual.")
parser.add_argument("--sigma", type=float, default=4.0, help="Sigma clipping threshold.")
parser.add_argument("--sigma0", type=float, default=2.0, help="How close (in AA) regions are allowed to be next to each other.")
args = parser.parse_args()

from Starfish import config
import json
import numpy as np
from astropy.stats import sigma_clip
from operator import itemgetter

f = open(args.file, "r")
read = json.load(f) # read is a dictionary
f.close()

wl = np.array(read["wl"])
# data = np.array(read["data"])
# model = np.array(read["model"])
residuals = np.array(read["resid"])
spectrum_id = read["spectrum_id"]
order = read["order"]

# array that specifies if a pixel is already covered.
# to start, it should be all False
covered = np.zeros((len(wl),), dtype='bool')

# #average all of the spectra in the deque together
# residual_array = np.array(self.resid_deque)
# if len(self.resid_deque) == 0:
#     raise RuntimeError("No residual spectra stored yet.")
# else:
#     residuals = np.average(residual_array, axis=0)

# run the sigma_clip algorithm until converged, and we've identified the outliers
filtered_data = sigma_clip(residuals, sig=args.sigma, iters=None)
mask = filtered_data.mask

# sigma0 = config['region_priors']['sigma0']
# logAmp = config["region_params"]["logAmp"]
# sigma = config["region_params"]["sigma"]

# Sort in decreasing strength of residual
nregions = 0
mus = []

for w, resid in sorted(zip(wl[mask], np.abs(residuals[mask])), key=itemgetter(1), reverse=True):
    if w in wl[covered]:
        continue
    else:
        # check to make sure region is not *right* at the edge of the echelle order
        if w <= np.min(wl) or w >= np.max(wl):
            continue
        else:
            # instantiate region and update coverage

            # Default amp and sigma values
            mus.append(w) # for evaluating the mu prior
            nregions += 1

            # determine the stretch of wl covered by this new region
            ind = (wl >= (w - args.sigma0)) & (wl <= (w + args.sigma0))
            # update the covered regions
            covered = covered | ind

# Save the mu's to file.
my_dict = {"mus":sorted(mus), "spectrum_id":spectrum_id, "order":order}
fname = Starfish.specfmt.format(spectrum_id, order) + "regions.json"
f = open(fname, 'w')
json.dump(my_dict, f, indent=2, sort_keys=True)
f.close()
