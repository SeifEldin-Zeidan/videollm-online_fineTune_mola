#!/bin/bash

#SBATCH --job-name=t_e_36
#SBATCH --partition=H200-AV,H200-PCI,H200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G 

#SBATCH --time=1-00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=seifeldin.zeidan@dfki.de

#SBATCH --output=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTrain_eval_logs/t_e_36/t_e_36_%x_%j.out
#SBATCH --error=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTrain_eval_logs/t_e_36/t_e_36_%x_%j.err


mkdir -p /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTrain_eval_logs/t_e_36

source /netscratch/zeidan/miniconda3/bin/activate 

conda activate videoLLM-online-Gradio-clone-ft


cd /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola



bash scripts/ego4d/mola_train_sh_files/live1+_mola_t36_fullOptions_augPerEpoch.sh && \

python -m demo.eval_mola_stream_chkPnts_auto \
    --resume_from_checkpoint /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/outputs/mola_live1+/train_36 \
    --pt_root_path "/netscratch/zeidan/mola_segments_combined_ftVLLMOnline/videos_sampled_4fps_1+3x3_google--siglip-large-patch16-384" \
    --frame_fps 4 \
    --min_chkPnt 40