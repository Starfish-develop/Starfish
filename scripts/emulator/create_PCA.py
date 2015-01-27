'''
Take an HDF5 file and downsize it to a PCA grid, write out to HDF5.
'''

import argparse
parser = argparse.ArgumentParser(prog="create_PCA.py",
                description="Decompose the spectra into eigenspectra.")
parser.add_argument("input", help="*.yaml file specifying parameters.")
args = parser.parse_args()

import yaml
from Starfish.emulator import PCAGrid

f = open(args.input)
cfg = yaml.load(f)
f.close()

pca = PCAGrid.from_cfg(**cfg)
pca.write(cfg["PCA_grid"])
