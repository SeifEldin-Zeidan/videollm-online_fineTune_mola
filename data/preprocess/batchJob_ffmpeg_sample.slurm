#!/bin/bash

#SBATCH --job-name=combineMola

#SBATCH --cpus-per-task=16

#SBATCH --partition=batch,V100-32GB,RTXA6000

#SBATCH --mem=8G
#SBATCH --time=0-10:00

#SBATCH --output=/netscratch/zeidan/mola_segments_combined_ftVLLMOnline/logs/logs-%j.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=seifeldin.zeidan@dfki.de

mkdir -p /netscratch/zeidan/mola_segments_combined_ftVLLMOnline/logs

cd /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/data/preprocess

python ffmpeg_mine_parallel.py