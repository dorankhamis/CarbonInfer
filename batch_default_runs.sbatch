#!/bin/bash 
#SBATCH --partition=par-single
#SBATCH --job-name=defaultruns
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=1:00:00
#SBATCH -n 1
#SBATCH --array=0-18 ## this is the number of sites

python ./run_with_default_params.py ${SLURM_ARRAY_TASK_ID}
