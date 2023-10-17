#!/bin/bash
#SBATCH --partition=short-serial-4hr
#SBATCH --account=short4hr
#SBATCH --job-name=train_cpredict
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=2:00:00
source /home/users/doran/software/envs/pytorch/bin/activate
python ./train_carbon_multiscale.py

