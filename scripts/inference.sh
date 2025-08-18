#!/bin/bash

target_dir="your_target_dir,such as ./lastfm-sft/checkpoint-1122"
cuda_device=1
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

log_file="$checkpoint/eval.log"

master_port=$(generate_unique_port)

echo "Using CUDA device: $cuda_device, Master Port: $master_port"

CUDA_VISIBLE_DEVICES="$cuda_device" torchrun --nproc_per_node 1 --master_port="$master_port" \
    inference.py \
    --dataset lastfm \
    --external_prompt_path "./prompt/music.txt" \
    --batch_size 32 \
    --base_model "your_base_llm_model" \
    --resume_from_checkpoint "$checkpoint" > "$log_file" 2>&1
