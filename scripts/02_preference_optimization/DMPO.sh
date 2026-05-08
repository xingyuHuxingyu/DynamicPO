#!/bin/bash

# DMPO baseline experiment.
# This script explicitly triggers the baseline DMPO branch implemented in
# `trainer/dynamicpo_trainer.py` via `filter_mode="DMPO"`.

set -e

GPUS="0,1,2,3"
NPROC_PER_NODE=4
MASTER_PORT=$((25000 + RANDOM % 2000))

MODEL_NAME="your_base_llm_model"
SFT_CHECKPOINT="your_sft_checkpoint"
DATASET="lastfm"
PROMPT_PATH="./prompt/music.txt"

BETA="1.0"
NEG_NUM="15"
BATCH_SIZE="4"
GRAD_ACC="8"
NUM_EPOCHS="3"
LR="1e-5"
EVAL_STEP="0.033"

FILTER_MODE="DMPO"
LOSS_TYPE="w_ref"
CUSTOM_NOTE="DMPO"
INFO_NOTE=""

OUTPUT_DIR="./DMPO/dmpo_neg_${NEG_NUM}_beta_${BETA}"
WANDB_NAME="dmpo_neg_${NEG_NUM}_beta_${BETA}_bs_${BATCH_SIZE}_ga_${GRAD_ACC}"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node ${NPROC_PER_NODE} --master_port=${MASTER_PORT} DynamicPO.py \
    --model_name "${MODEL_NAME}" \
    --resume_from_checkpoint "${SFT_CHECKPOINT}" \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --dataset "${DATASET}" \
    --prompt_path "${PROMPT_PATH}" \
    --learning_rate ${LR} \
    --eval_step ${EVAL_STEP} \
    --beta ${BETA} \
    --filter_mode "${FILTER_MODE}" \
    --custom_note "${CUSTOM_NOTE}" \
    --info_note "${INFO_NOTE}" \
    --loss_type "${LOSS_TYPE}" \
    --neg_num ${NEG_NUM} \
    --num_train_epochs ${NUM_EPOCHS} \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_name "${WANDB_NAME}" > "${OUTPUT_DIR}/train.log" 2>&1
