#!/bin/bash
#SBATCH -J train_job
#SBATCH -o log.o%j
#SBATCH -t 50:00:00
#SBATCH -N 1 -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=64GB
#SBATCH -A tsekos

module add python/3.8

#pip install opencv-python


python train.py
