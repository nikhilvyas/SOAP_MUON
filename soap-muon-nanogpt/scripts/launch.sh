#!/bin/bash
#SBATCH --job-name=modded-nanogpt
#SBATCH --account=kempner_sham_lab
#SBATCH --output=/n/netscratch/kempner_sham_lab/Lab/soap-muon/%A_%a.log
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=24
#SBATCH --time=6:00:00
#SBATCH --mem=250GB		
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100

# Custom environment
source ~/.bashrc
conda deactivate
conda activate modded-nanogpt

module load cuda
module load cudnn

torchrun --standalone --nproc_per_node=1 train_gpt2.py
