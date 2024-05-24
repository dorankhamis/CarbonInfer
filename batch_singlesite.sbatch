#!/bin/bash 
#SBATCH --partition=par-single
#SBATCH --job-name=infersinglesites
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=48:00:00
#SBATCH -n 8
#SBATCH --array=0-16 ## this is the number of sites

python ./single_site_runs.py 8 2000 8 ${SLURM_ARRAY_TASK_ID} 1
