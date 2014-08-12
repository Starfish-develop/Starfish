from StellarSpectra.model import Model, StellarSampler, ChebSampler, MegaSampler, CovGlobalSampler, RegionsSampler
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
import StellarSpectra.constants as C
import numpy as np
import yaml
import os
import shutil


import argparse
parser = argparse.ArgumentParser(prog="base_lnprob.py", description="Run StellarSpectra fitting model.")
parser.add_argument("-i", "--input", help="*.yaml file specifying parameters.")
parser.add_argument("-r", "--run_index", help="Which run (of those running concurrently) is this? All data will "
                                        "be written into this directory, overwriting any that exists.")
parser.add_argument("-p", "--perturb", type=float, help="Randomly perturb the starting position of the "
                                                                 "chain, as a multiple of the jump parameters.")
args = parser.parse_args()

if args.input: #
    #assert that we actually specified a *.yaml file
    if ".yaml" in args.input:
        yaml_file = args.input
        f = open(args.input)
        config = yaml.load(f)
        f.close()

    else:
        import sys
        sys.exit("Must specify a *.yaml file.")

else:
    #load the default config file
    yaml_file = "scripts/stars/input.yaml"
    f = open(yaml_file)
    config = yaml.load(f)
    f.close()

def perturb(startingDict, jumpDict, factor=3.):
    '''
    Given a starting parameter dictionary loaded from a config file, perturb the values as a multiple of the jump
    distribution. This is designed so that chains do not all start in the same place, when run in parallel.

    Modifies the startingDict
    '''
    for key in startingDict.keys():
        startingDict[key] += factor * np.random.normal(loc=0, scale=jumpDict[key])


myDataSpectrum = DataSpectrum.open(config['data'], orders=config['orders'])
#Load mask and add it to DataSpectrum
# mask = np.load("data/WASP14/WASP14_23.mask.npy")
# myDataSpectrum.add_mask(np.atleast_2d(mask))
myInstrument = TRES()
myHDF5Interface = HDF5Interface(config['HDF5_path'])

stellar_Starting = config['stellar_params']
stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)
#go for each item in stellar_tuple, and assign the appropriate covariance to it
stellar_MH_cov = np.array([float(config["stellar_jump"][key]) for key in stellar_tuple])**2 \
                 * np.identity(len(stellar_Starting))

try:
    fix_logg = config['fix_logg']
except KeyError:
    fix_logg = None

#Updating specific correlations to speed mixing
if config["use_cov"]:
    stellar_cov = config["stellar_cov"]
    factor = stellar_cov["factor"]

    stellar_MH_cov[0, 1] = stellar_MH_cov[1, 0] = stellar_cov['temp_logg']  * factor
    stellar_MH_cov[0, 2] = stellar_MH_cov[2, 0] = stellar_cov['temp_Z'] * factor
    stellar_MH_cov[1, 2] = stellar_MH_cov[2, 1] = stellar_cov['logg_Z'] * factor
    if fix_logg is None:
        stellar_MH_cov[0, 5] = stellar_MH_cov[5, 0] = stellar_cov['temp_logOmega'] * factor
    else:
        stellar_MH_cov[0, 4] = stellar_MH_cov[4, 0] = stellar_cov['temp_logOmega'] * factor


cheb_degree = config['cheb_degree']
cheb_MH_cov = float(config["cheb_jump"])**2 * np.identity(cheb_degree)
cheb_tuple = ("logc0",)
#add in new coefficients
for i in range(1, cheb_degree):
    cheb_tuple += ("c{}".format(i),)
#set starting position to 0
cheb_Starting = {k:0.0 for k in cheb_tuple}


cov_Starting = config['cov_params']
cov_tuple = C.dictkeys_to_cov_global_tuple(cov_Starting)
cov_MH_cov = np.array([float(config["cov_jump"][key]) for key in cov_tuple])**2 \
                 * np.identity(len(cov_Starting))

if args.perturb:
    perturb(stellar_Starting, config["stellar_jump"], factor=args.perturb)
    #cheb_jump = {key: config["cheb_jump"] for key in cheb_tuple}
    #perturb(cheb_Starting, cheb_jump, factor=args.perturb)
    perturb(cov_Starting, config['cov_jump'], factor=args.perturb)


region_Starting = config['region_params']
region_tuple = ("loga", "mu", "sigma")
region_MH_cov = np.array([float(config["region_jump"][key]) for key in region_tuple])**2 * np.identity(len(region_tuple))


outdir = config['outdir']
name = config['name']
base = outdir + name + "run{:0>2}/"

#This code is necessary for multiple simultaneous runs on odyssey, so that different runs do not write into the same
#output directory
if args.run_index == None:
    run_index = 0
    while os.path.exists(base.format(run_index)) and (run_index < 20):
        print(base.format(run_index), "exists")
        run_index += 1
    outdir = base.format(run_index)

else:
    run_index = args.run_index
    outdir = base.format(run_index)
    #Delete this outdir, if it exists
    if os.path.exists(outdir):
        print("Deleting", outdir)
        shutil.rmtree(outdir)

print("Creating ", outdir)
os.makedirs(outdir)

for order in config['orders']:
    order_dir = "{}{}".format(outdir, order)
    print("Creating ", order_dir)
    os.makedirs(order_dir)

#Copy yaml file to outdir
shutil.copy(yaml_file, outdir + "/input.yaml")

myModel = Model(myDataSpectrum, myInstrument, myHDF5Interface, stellar_tuple=stellar_tuple, cheb_tuple=cheb_tuple,
                cov_tuple=cov_tuple, region_tuple=region_tuple, outdir=outdir, debug=False)



myStellarSampler = StellarSampler(model=myModel, cov=stellar_MH_cov, starting_param_dict=stellar_Starting,
                                  fix_logg=fix_logg, outdir=outdir, debug=False)

#Create the three subsamplers for each order
samplerList = []

for i in range(len(config['orders'])):
    samplerList.append(ChebSampler(model=myModel, cov=cheb_MH_cov, starting_param_dict=cheb_Starting, order_index=i,
                                   outdir=outdir, debug=False))
    if not config['no_cov']:
        samplerList.append(CovGlobalSampler(model=myModel, cov=cov_MH_cov, starting_param_dict=cov_Starting, order_index=i,
                                       outdir=outdir, debug=False))
        samplerList.append(RegionsSampler(model=myModel, cov=region_MH_cov, max_regions=config['max_regions'],
                        default_param_dict=region_Starting, order_index=i, outdir=outdir, debug=False))

mySampler = MegaSampler(samplers=[myStellarSampler] + samplerList, debug=True)

def main():
    mySampler.run(config["burn_in"])
    mySampler.reset()
    mySampler.trim_regions()

    mySampler.run(config["burn_in"])
    mySampler.reset()
    mySampler.trim_regions()

    mySampler.run(config["samples"])
    print(mySampler.acceptance_fraction)
    print(mySampler.acor)
    myModel.to_json("model_final.json")
    mySampler.write()
    mySampler.plot()


if __name__=="__main__":
    main()