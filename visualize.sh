#!/bin/bash
#SBATCH --job-name=job_visualize
#SBATCH --output=output_visualize.log
#SBATCH --error=output_visualize.log
source /home/aryan/miniconda3/etc/profile.d/conda.sh
conda activate facekd
cd /home/aryan/FSCIL/FaceKD/
python train_test.py --mode visualize --train_dataset umdface --test_dataset umdface --resume_visualize_model '/home/aryan/FSCIL/FaceKD/results/2024-05-10-22-05-56/models/0'