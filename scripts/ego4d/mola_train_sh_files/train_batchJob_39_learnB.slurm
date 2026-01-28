#!/bin/bash

#SBATCH --job-name=t_callE_39_learnB
#SBATCH --partition=H200-AV,H200-PCI,H200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G 

#SBATCH --time=0-03:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=seifeldin.zeidan@dfki.de

#SBATCH --output=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTrain_logs/t_39_learnB/t_39_learnB_%x_%j.out
#SBATCH --error=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTrain_logs/t_39_learnB/t_39_learnB_%x_%j.err


mkdir -p /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTrain_logs/t_39_learnB

source /netscratch/zeidan/miniconda3/bin/activate 

conda activate videoLLM-online-Gradio-clone-ft


cd /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola



bash scripts/ego4d/mola_train_sh_files/live1+_mola_t39_learnB_fullOptions.sh && \

sbatch /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/scripts/ego4d/mola_train_sh_files/eval_batchJob_1_t_39_learnB.slurm