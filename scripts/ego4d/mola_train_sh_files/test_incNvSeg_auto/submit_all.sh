#!/bin/bash
set -euo pipefail

CKPT_ROOT=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/outputs/mola_live1+/train_38_incNvSegment

LOWER_THRESH=0
UPPER_THRESH=2000

SBATCH_FILE=/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/scripts/ego4d/mola_train_sh_files/test_incNvSeg_auto/eval_one_checkpoint.slurm

find "$CKPT_ROOT" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | while read -r ckpt; do
  ckpt_name=$(basename "$ckpt")        # checkpoint-283
  ckpt_id=${ckpt_name#checkpoint-}     # 283
  
  fps=8
  echo "Using FPS: $fps"
  typeSet="test"
  echo "Running $typeSet set"

  # Sanity check: ensure it's numeric
  [[ "$ckpt_id" =~ ^[0-9]+$ ]] || continue

  if (( ckpt_id >= LOWER_THRESH && ckpt_id < UPPER_THRESH )); then
    echo "Submitting checkpoint $ckpt_id (range: $LOWER_THRESH–$((UPPER_THRESH-1)))"
    # sbatch --export=ALL,CKPT_PATH="$ckpt",FPS="$fps" "$SBATCH_FILE"
    sbatch \
    --job-name="incNvSeg_fps${fps}_ckpt${ckpt_id}_probAnalyze" \
    --export=ALL,CKPNT="$ckpt_id",FPS="$fps",TYPESET="$typeSet" \
    "$SBATCH_FILE"

  else
    echo "Skipping checkpoint $ckpt_id (outside range)"
  fi
done
