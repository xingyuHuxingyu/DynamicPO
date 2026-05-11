#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# Position the number of processes specified after the --nproc_per_node flag
OUTPUT_DIR="./lastfm-sft"
MODEL_NAME="your_base_llm_model"

model_name_to_slug() {
    basename "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
}

MODEL_SLUG="$(model_name_to_slug "${MODEL_NAME}")"
OUTPUT_DIR="${OUTPUT_DIR}_${MODEL_SLUG}"
WANDB_NAME="wandb_run_name_${MODEL_SLUG}"

mkdir -p "${OUTPUT_DIR}"


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25664 sft.py \
        --model_name "${MODEL_NAME}"  \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --dataset lastfm \
        --prompt_path "./prompt/music.txt" \
        --logging_dir "./" \
        --output_dir "${OUTPUT_DIR}" \
        --wandb_project dpo-rec-nf4 \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --eval_step 0.05 \
        --wandb_name "${WANDB_NAME}" \
> "${OUTPUT_DIR}/lastfm_sft.log" 2>&1
