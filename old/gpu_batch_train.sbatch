#!/bin/bash 
#SBATCH --partition=lotus_gpu 
#SBATCH --account=lotus_gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_mb_tair_deep
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=2:00:00
source /home/users/doran/software/envs/pytorch/bin/activate
python ./train_mud_breath.py

