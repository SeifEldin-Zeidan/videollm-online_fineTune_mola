#!/bin/bash

#SBATCH --job-name=test_incNvSeg
#SBATCH --partition=RTXA6000-AV,H100,A100-PCI,A100-80GB,A100-40GB,RTXA6000,H100,H100-PCI,H200,H200-AV,H200-PCI
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G 

#SBATCH --time=0-01:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=seifeldin.zeidan@dfki.de

#SBATCH --output=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTest_logs/t_incNvSeg_probAnalyze/t_incNvSeg_probAnalyze_%x_%j.out
#SBATCH --error=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTest_logs/t_incNvSeg_probAnalyze/t_incNvSeg_probAnalyze_%x_%j.err


mkdir -p /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/batchTest_logs/t_incNvSeg_probAnalyze

source /netscratch/zeidan/miniconda3/bin/activate 

conda activate videoLLM-online-Gradio-clone-ft


cd /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola



# bash scripts/ego4d/mola_train_sh_files/live1+_mola_t36_fullOptions.sh && \

# python -m demo.eval_mola_stream_chkPnts_auto \
#     --resume_from_checkpoint /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/outputs/mola_live1+/train_38_incNvSegment \
#     --pt_root_path "/netscratch/zeidan/mola_segments_combined_ftVLLMOnline/videos_sampled_${FPS}fps_1+3x3_google--siglip-large-patch16-384" \
#     --frame_fps $FPS \
#     --anno_path /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withSampledNumFrames_segDescSummary/vLLMonline_mola_${TYPESET}.json \
#     --min_chkPnt $CKPNT \
#     --max_chkPnt $CKPNT \
#     --results_fileName "${TYPESET}_results_${FPS}fps.json"


resultsRootPath="/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/demo/eval_mola_stream_results/probAnalyze/${FPS}fps"
mkdir -p $resultsRootPath




python -m demo.probAnalyze_eval_mola_stream_chkPnts_auto \
   --resume_from_checkpoint /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/outputs/mola_live1+/train_38_incNvSegment \
   --pt_root_path "/netscratch/zeidan/mola_segments_combined_ftVLLMOnline/videos_sampled_${FPS}fps_1+3x3_google--siglip-large-patch16-384" \
   --frame_fps $FPS \
   --anno_path "/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withSampledNumFrames_segDescSummary/vLLMonline_mola_${TYPESET}.json" \
   --min_chkPnt "$CKPNT" \
   --max_chkPnt "$CKPNT" \
   --results_fileName "${TYPESET}_results_${FPS}fps_thresh_0_analyzeProb.json" \
   --skip_flip \
   --thresh 0 \
   --results_root_path "$resultsRootPath"
   