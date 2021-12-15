#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1

module load gcc/8.3.0
module load python/3.7.6
module load cuda/10.1.243
module load cudnn/8.0.2-10.1
module load pgi-nvhpc

python3 MRI_model.py
