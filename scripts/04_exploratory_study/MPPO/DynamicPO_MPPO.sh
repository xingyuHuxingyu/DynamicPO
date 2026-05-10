#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

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
CUTOFF_LEN="512"
EVAL_STEP="0.033"

OUTPUT_DIR="./outputs/exploratory_study/dynamicpo_mppo"
WANDB_NAME="exploratory_dynamicpo_mppo"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node ${NPROC_PER_NODE} --master_port=${MASTER_PORT} exploratory_study.py \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --prompt_path "${PROMPT_PATH}" \
  --dataset "${DATASET}" \
  --resume_from_checkpoint "${SFT_CHECKPOINT}" \
  --wandb_name "${WANDB_NAME}" \
  --custom_note "exploratory study" \
  --info_note "DynamicPO-MPPO" \
  --beta ${BETA} \
  --filter_mode "DynamicPO_MPPO" \
  --loss_type "wo_ref" \
  --neg_num ${NEG_NUM} \
  --batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --num_train_epochs ${NUM_EPOCHS} \
  --learning_rate ${LR} \
  --cutoff_len ${CUTOFF_LEN} \
  --eval_step ${EVAL_STEP} \
  > "${OUTPUT_DIR}/train.log" 2>&1
