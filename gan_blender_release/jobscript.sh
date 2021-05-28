#!/bin/bash
#SBATCH --partition=gpu_shared_course
#SBATCH --time 02:00:00#
#SBATCH --gres=gpu:1
#SBATCH --job-name TestJob
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=cmiveenker@gmail.com
#SBATCH --output=slurm_output%A.out
module purge
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load CUDA

cd $HOME/CV2_Assignment3_git/gan_blender_release
srun python3 train_blender.py
