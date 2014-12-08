#!/bin/bash

#SBATCH -o slurm/a.out #standard output

#SBATCH -e slurm/a.err #standard error

#SBATCH -p general #partition

#SBATCH -t 8:00:00 #running time

#SBATCH --mail-type=BEGIN

#SBATCH --mail-type=END

#SBATCH --mail-user=iancze@gmail.com

#SBATCH --mem-per-cpu 1000 #memory request

#SBATCH -n 1

python scripts/stars/base_lnprob.py -p scripts/stars/input.yaml
