#!/bin/bash

#SBATCH --job-name=eval_t_30
#SBATCH --partition=H200-AV,H200,H200-PCI,H100,A100-PCI,A100-80GB,A100-40GB,RTXA6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#SBATCH --time=0-15:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=seifeldin.zeidan@dfki.de

#SBATCH --output=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchEval_logs/t_30/t_30_%x_%j.out
#SBATCH --error=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchEval_logs/t_30/t_30_%x_%j.err


mkdir -p /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchEval_logs/t_30

source /netscratch/zeidan/miniconda3/bin/activate 

conda activate videoLLM-online-Gradio-clone-ft


cd /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola



# bash scripts/ego4d/mola_train_sh_files/live1+_mola_t27_rand_vShiftPerc_summary.sh
python -m demo.eval_mola_stream_chkPnts_auto \
    --resume_from_checkpoint /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/outputs/mola_live1+/train_30 \
    --pt_root_path "/netscratch/zeidan/mola_segments_combined_ftVLLMOnline/videos_sampled_4fps_1+3x3_google--siglip-large-patch16-384" \
    --frame_fps 4 

