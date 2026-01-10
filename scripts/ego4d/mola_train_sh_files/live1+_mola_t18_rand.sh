#!/bin/bash

VIDEOS_PT_ROOT="/netscratch/zeidan/mola_segments_combined_ftVLLMOnline/videos_sampled_randFps_1+3x3_google--siglip-large-patch16-384"
ANNOS_ROOT="/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withRandomFps_withSampledNumFrames"
SYSTEM_PROMPT="You are a vision-language model expert in analyzing surveillance videos. You'll be given video footage stream of people seated in the backseat of a vehicle."


deepspeed train.py --deepspeed configs/deepspeed/zero2.json \
    --live_version live1+ \
    --train_datasets ego4d_nlq_stream_mola_train \
    --videos_pt_root_dir "${VIDEOS_PT_ROOT}" \
    --annotations_root_dir "${ANNOS_ROOT}" \
    --system_prompt "${SYSTEM_PROMPT}" \
    --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus \
    --num_train_epochs 15 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --eval_strategy no \
    --prediction_loss_only False \
    --save_strategy epoch \
    --save_only_model True \
    --learning_rate 0.0002 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --dataloader_num_workers 6 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/mola_live1+/train_18
