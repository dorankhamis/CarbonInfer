#!/bin/bash 
#SBATCH --partition=par-single
#SBATCH --job-name=shrubinfer
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=48:00:00
#SBATCH -n 1

python ./inference.py 1 1000 1 3
