#!/bin/bash 
#SBATCH --partition=par-single
#SBATCH --job-name=param0singlesites
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=48:00:00
#SBATCH -n 8
#SBATCH --array=0-16

python ./single_param_site_runs.py 8 300 8 ${SLURM_ARRAY_TASK_ID} 0
