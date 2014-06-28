from StellarSpectra.model import Model, StellarSampler, ChebSampler, CovGlobalSampler, RegionsSampler, MegaSampler
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
import StellarSpectra.constants as C
import numpy as np
import yaml
import os
import shutil


import argparse
parser = argparse.ArgumentParser(prog="base_lnprob.py", description="Run StellarSpectra fitting model.")
parser.add_argument("-p", "--params", help="*.yaml file specifying parameters.")
args = parser.parse_args()

if args.params: #
    #assert that we actually specified a *.yaml file
    if ".yaml" in args.params:
        yaml_file = args.params
        f = open(args.params)
        config = yaml.load(f)
        f.close()

    else:
        import sys
        sys.exit("Must specify a *.yaml file.")
        yaml_file = args.params
else:
    #load the default config file
    yaml_file = "scripts/stars/input.yaml"
    f = open(yaml_file)
    config = yaml.load(f)
    f.close()

myDataSpectrum = DataSpectrum.open(config['data'], orders=config['orders'])
myInstrument = TRES()
myHDF5Interface = HDF5Interface(config['HDF5_path'])

stellar_Starting = config['stellar_params']
stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)
#go for each item in stellar_tuple, and assign the appropriate covariance to it
stellar_MH_cov = np.array([float(config["stellar_jump"][key]) for key in stellar_tuple])**2 \
                 * np.identity(len(stellar_Starting))

#Tuning the correlations should depend on how the models were normalized, right?

#Attempt at updating specific correlations
# #Temp/Logg correlation
# temp_logg = 0.2 * np.sqrt(0.01 * 0.001)
# stellar_MH_cov[0, 1] = temp_logg
# stellar_MH_cov[1, 0] = temp_logg
#
# #Temp/logOmega correlation
# temp_logOmega = - 0.9 * np.sqrt(stellar_MH_cov[0,0] * stellar_MH_cov[5,5])
# stellar_MH_cov[0, 5] = temp_logOmega
# stellar_MH_cov[5, 0] = temp_logOmega

#We could make a function which takes the two positions of the parameters (0, 5) and then updates the covariance
#based upon a rho we feed it.

#We could test to see if these jumps are being executed in the right direction by checking to see what the 2D pairwise
# chain positions look like

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


region_tuple = ("loga", "mu", "sigma")
region_MH_cov = np.array([0.04, 0.02, 0.1])**2 * np.identity(len(region_tuple))


outdir = config['outdir']
name = config['name']
base = outdir + name + "run{:0>2}/"
run_num = 0
while os.path.exists(base.format(run_num)) and (run_num < 20):
    print(base.format(run_num), "exists")
    run_num += 1

outdir = base.format(run_num)
print("Creating ", outdir)
os.makedirs(outdir)

for order in config['orders']:
    order_dir = "{}{}".format(outdir, order)
    print("Creating ", order_dir)
    os.makedirs(order_dir)

#Copy yaml file to outdir
shutil.copy(yaml_file, outdir)

myModel = Model(myDataSpectrum, myInstrument, myHDF5Interface, stellar_tuple=stellar_tuple, cheb_tuple=cheb_tuple,
                cov_tuple=cov_tuple, region_tuple=region_tuple, outdir=outdir)

try:
    fixlogg = config['fixlogg']
except KeyError:
    fixlogg = None

myStellarSampler = StellarSampler(myModel, stellar_MH_cov, stellar_Starting, fix_logg=config['fix_logg'],
                                  outdir=outdir)

#Create the three subsamplers for each order
samplerList = []
cadenceList = []
for i in range(len(config['orders'])):
    samplerList.append(ChebSampler(myModel, cheb_MH_cov, cheb_Starting, order_index=i, outdir=outdir))
    samplerList.append(CovGlobalSampler(myModel, cov_MH_cov, cov_Starting, order_index=i, outdir=outdir))
    samplerList.append(RegionsSampler(myModel, region_MH_cov, max_regions=config['max_regions'], order_index=i, outdir=outdir))
    cadenceList += [6, 6, 2]
    # cadenceList += [6, 2]

mySampler = MegaSampler(myModel, samplers=[myStellarSampler] + samplerList,
                        burnInCadence=[10] + cadenceList, cadence=[10] + cadenceList)

def main():
    mySampler.burn_in(config["burn_in"])
    mySampler.reset()

    mySampler.run(config["samples"])
    mySampler.acceptance_fraction()
    myModel.to_json("model_final.json")
    mySampler.write()
    mySampler.plot()


if __name__=="__main__":
    main()