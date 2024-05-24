#!/bin/bash 
#SBATCH --partition=short-serial
####SBATCH --account=short4hr
#SBATCH --job-name=longspin
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=24:00:00

python ./run_long_spin.py
