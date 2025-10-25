#!/bin/bash
timestamp=$(date +%Y-%m-%d-%H-%M-%S)
name="${timestamp}_lr_change_t5_k4"
log_dir="/home/aryan/FSCIL/FaceKD/output_logs/"
#SBATCH --job-name=${name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${log_dir}/output_${name}.log
#SBATCH --error=${log_dir}/output_${name}.log
source /home/aryan/miniconda3/etc/profile.d/conda.sh
conda activate facekd_new
cd /home/aryan/FSCIL/FaceKD/
python train_test.py --train_dataset umdface arcface vggface retinaface casiaface --test_dataset umdface arcface vggface retinaface casiaface --total_train_epochs 30 --total_continual_train_epochs 30 --weight_a 0.01