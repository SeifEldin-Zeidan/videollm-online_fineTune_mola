#!/bin/bash

#SBATCH --job-name=t_24
#SBATCH --partition=H200-AV,H200-PCI,H200,H100,H100-PCI
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G 

#SBATCH --time=0-04:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=seifeldin.zeidan@dfki.de

#SBATCH --output=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTrain_logs/t_24/t_24_%x_%j.out
#SBATCH --error=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTrain_logs/t_24/t_24_%x_%j.err


mkdir -p /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTrain_logs/t_24

source /netscratch/zeidan/miniconda3/bin/activate 

conda activate videoLLM-online-Gradio-clone-ft


cd /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola



bash scripts/ego4d/mola_train_sh_files/live1+_mola_t24_rand_vShiftPerc_FIXED_learnBoundaryFalse.sh


