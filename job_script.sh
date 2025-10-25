#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=DGX   # Replace with the name of your GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=12:00:00

# Load any necessary modules
# module load your_module

# Run your script on the GPU node
srun -n 1 -N 1 -l ./run_training.sh
