#!/bin/bash

#SBATCH -J racer #Single job name for the entire JobArray

#SBATCH -o slurm/racer_%A_%a.out #standard output

#SBATCH -e slurm/racer_%A_%a.err #standard error

#SBATCH -p general #partition

#SBATCH -t 8:30:00 #running time

#SBATCH --mail-type=BEGIN

#SBATCH --mail-type=END

#SBATCH --mail-user=iancze@gmail.com

#SBATCH --mem-per-cpu 1000 #memory request

#SBATCH -n 1

python scripts/stars/base_lnprob.py -p scripts/stars/input.yaml
