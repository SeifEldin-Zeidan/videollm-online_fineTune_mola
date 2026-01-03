#!/bin/bash

#SBATCH --job-name=combineMola

#SBATCH --cpus-per-task=6

#SBATCH --partition=batch

#SBATCH --mem=8G
#SBATCH --time=0-10:00

#SBATCH --output=/netscratch/zeidan/mola_segments_combined_ftVLLMOnline/logs/logs-%j.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=seifeldin.zeidan@dfki.de

mkdir -p /netscratch/zeidan/mola_segments_combined_ftVLLMOnline/logs

source /netscratch/zeidan/miniconda3/bin/activate

conda activate yolo_1

cd /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/annotating_mola

python combineMola.py --type train & python combineMola.py --type val & python combineMola.py --type test & wait