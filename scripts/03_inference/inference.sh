#!/bin/bash

set -e

CHECKPOINT_DIR="your_target_dir_such_as_./DynamicPO_DMPO/checkpoint-1122"
CUDA_DEVICE=0
BASE_MODEL="your_base_llm_model"
DATASET="lastfm"
PROMPT_PATH="./prompt/music.txt"
BATCH_SIZE=32

used_ports=()

generate_unique_port() {
    while true; do
        local port=$((24000 + RANDOM % 3000))
        if [[ ! " ${used_ports[*]} " =~ " $port " ]]; then
            used_ports+=($port)
            echo $port
            return
        fi
    done
}

LOG_FILE="${CHECKPOINT_DIR}/eval.log"
MASTER_PORT=$(generate_unique_port)

echo "Using CUDA device: ${CUDA_DEVICE}, Master Port: ${MASTER_PORT}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" torchrun --nproc_per_node 1 --master_port="${MASTER_PORT}" \
    inference.py \
    --dataset "${DATASET}" \
    --external_prompt_path "${PROMPT_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --base_model "${BASE_MODEL}" \
    --resume_from_checkpoint "${CHECKPOINT_DIR}" > "${LOG_FILE}" 2>&1
