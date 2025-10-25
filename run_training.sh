#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --output=test.log
#SBATCH --error=test.log
set -e
ulimit -n 32000
ulimit -a >> test.log
python train_test.py --train_dataset umdface arcface --test_dataset umdface arcface --test_frequency 0 --total_continual_train_epochs 2 --total_train_epochs 2