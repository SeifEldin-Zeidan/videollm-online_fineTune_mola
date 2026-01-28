#!/bin/bash

#SBATCH --job-name=eval_iv2Chat_genBinary_chk_eval
#SBATCH --partition=RTXA6000,RTXA6000-AV,A100-40GB,A100-PCI
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G 

#SBATCH --time=0-04:00 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=seifeldin.zeidan@dfki.de

#SBATCH --output=/netscratch/zeidan/finetune_iv2Chat_2/finetune_newRepo_iv2Chat/evaluating_iv2Chat_files_both/evalMola_fineTunedChkpnt_NLA/updated_iv2_chat_HD_MOLA_NLA/window/logs/gen_binary/%x_%j.out
#SBATCH --error=/netscratch/zeidan/finetune_iv2Chat_2/finetune_newRepo_iv2Chat/evaluating_iv2Chat_files_both/evalMola_fineTunedChkpnt_NLA/updated_iv2_chat_HD_MOLA_NLA/window/logs/gen_binary/%x_%j.err

mkdir -p /netscratch/zeidan/finetune_iv2Chat_2/finetune_newRepo_iv2Chat/evaluating_iv2Chat_files_both/evalMola_fineTunedChkpnt_NLA/updated_iv2_chat_HD_MOLA_NLA/window/logs/gen_binary


# ---- critical: CKPT_PATH must be provided at submission time ----
if [[ -z "${CKPT_PATH:-}" ]]; then
  echo "ERROR: CKPT_PATH is not set. Submit like:"
  echo '  sbatch --export=ALL,CKPT_PATH="/path/to/checkpoint-283" eval_one_checkpoint.sbatch'
  exit 1
fi

source /netscratch/zeidan/miniconda3/bin/activate 

conda activate internvideo2_chat_8B_HD_clone

cd /netscratch/zeidan/finetune_iv2Chat_2/finetune_newRepo_iv2Chat/evaluating_iv2Chat_files_both/evalMola_fineTunedChkpnt_NLA/updated_iv2_chat_HD_MOLA_NLA/window

echo "Running for checkpoint ${CKPT_PATH}"

# python iv2_chat_mola_eval_window.py --run_type eval --view_answers --chkPnt_path "/netscratch/zeidan/finetune_iv2Chat_2/Combined_gen_molaWindow_3/checkpoint-94"
python iv2_chat_mola_eval_window.py --run_type eval --view_answers --chkPnt_path "$CKPT_PATH"
