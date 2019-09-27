#!/bin/sh
#SBATCH -p dgx
#SBATCH --gres=gpu:1
python train.py