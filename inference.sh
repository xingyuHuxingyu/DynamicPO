#!/bin/bash

# 定义要评测的目录路径
target_dir="/home/sankuai/dolphinfs_huxingyu05/DPO_series/beta-DPO-final-all/lastfm/novel_exp/4_10_v1_dynamicPO_负样本筛选_多负样本loss_负样本beta调整_neg15_beta_1_bs_4_ga_8_mw_0.0_fm_simpo_Dynamic_select_alpha_0.0_bd_False_al_instance_level_beta_1.0/checkpoint-1070"



# 定义可用的CUDA设备
cuda_device=1  # 只使用一个CUDA设备

# 用于存储已使用的端口号
used_ports=()

# 生成一个未使用的随机端口号
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

# 定义一个函数来执行评测任务
evaluate_checkpoint() {
    checkpoint=$1
    cuda_device=$2

    echo "正在评测，检查点目录：$checkpoint"

    # 检查目录是否存在
    if [[ ! -d "$checkpoint" ]]; then
        echo "目录不存在：$checkpoint，跳过此检查点。"
        return
    fi

    # 定义日志文件路径
    log_file="$checkpoint/eval.log"

    # 获取一个未使用的随机端口号
    master_port=$(generate_unique_port)

    echo "使用CUDA设备：$cuda_device，Master Port：$master_port"

    # 执行评测任务并重定向输出
    CUDA_VISIBLE_DEVICES="$cuda_device" torchrun --nproc_per_node 1 --master_port="$master_port" \
        inference.py \
        --dataset lastfm \
        --external_prompt_path "./prompt/music.txt" \
        --batch_size 32 \
        --base_model "/home/sankuai/dolphinfs_huxingyu05/DPO_series/Llama-2-7b-hf" \
        --resume_from_checkpoint "$checkpoint" > "$log_file" 2>&1

    # 检查命令是否成功执行
    if [[ $? -ne 0 ]]; then
        echo "评测失败，检查点目录：$checkpoint，日志文件：$log_file"
    else
        echo "评测完毕，检查点目录：$checkpoint，日志文件：$log_file"
    fi
    echo "--------------------------------------------"
}

# 检查目标目录是否存在
if [[ -d "$target_dir" ]]; then
    # 对目标目录进行评测
    evaluate_checkpoint "$target_dir" "$cuda_device"
else
    echo "目标目录不存在：$target_dir，跳过此目录。"
fi

echo "评测任务已顺利执行完毕。"
