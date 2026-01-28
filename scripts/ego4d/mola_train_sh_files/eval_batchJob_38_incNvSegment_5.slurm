#!/bin/bash

#SBATCH --job-name=e_38_incNvSeg_5
#SBATCH --partition=RTXA6000-AV,RTXA6000,A100-PCI,A100-80GB,A100-40GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G 

#SBATCH --time=0-02:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=seifeldin.zeidan@dfki.de

#SBATCH --output=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchEval_logs/e_38_incNvSeg/e_38_incNvSeg_%x_%j.out
#SBATCH --error=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchEval_logs/e_38_incNvSeg/e_38_incNvSeg_%x_%j.err


mkdir -p /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchEval_logs/e_38_incNvSeg

source /netscratch/zeidan/miniconda3/bin/activate 

conda activate videoLLM-online-Gradio-clone-ft


cd /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola



# bash scripts/ego4d/mola_train_sh_files/live1+_mola_t36_fullOptions.sh && \

python -m demo.eval_mola_stream_chkPnts_auto \
    --resume_from_checkpoint /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/outputs/mola_live1+/train_38_incNvSegment \
    --pt_root_path "/netscratch/zeidan/mola_segments_combined_ftVLLMOnline/videos_sampled_4fps_1+3x3_google--siglip-large-patch16-384" \
    --frame_fps 4 \
    --min_chkPnt 55 \
    --max_chkPnt 55 

