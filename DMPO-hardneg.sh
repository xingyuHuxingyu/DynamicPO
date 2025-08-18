#!/bin/bash


# 基准目录
base_dir="/home/hadoop-mining/dolphinfs_hdd_hadoop-mining/huxingyu05/DynamicPO/betadpo4rec/exps-7.11/DMPO"
# 定义输出目录
loss_type="dpo"
master_port=$((25000 + RANDOM % 2000))

# 定义要循环的参数数组
batch_sizes=(4)
gradient_accumulation_steps_list=(8)
neg_nums=(5)
filter_modes=("DMPO_hard_negative")
adjust_levels=("instance_level")
beta_list=(1.0)

# 自定义备注内容
custom_note="DMPO"
info_note=""


check_gpus() {
    for gpu_id in 4 5 6 7; do
        utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
        if (( utilization > 0 )); then
            echo "GPU $gpu_id 正在使用中，等待空闲..."
            return 1
        fi
    done
    return 0
}

while ! check_gpus; do
    sleep 200
done
# 循环遍历所有参数组合
for batch_size in "${batch_sizes[@]}"; do
  for gradient_accumulation_steps in "${gradient_accumulation_steps_list[@]}"; do
    for neg_num in "${neg_nums[@]}"; do
      for filter_mode in "${filter_modes[@]}"; do
        for adjust_level in "${adjust_levels[@]}"; do
          for beta in "${beta_list[@]}"; do
            
            # 生成目标目录，包含自定义备注
            target_dir="${base_dir}/${custom_note}_neg_${neg_num}_fm_${filter_mode}_beta_${beta}"
            
            # 确保生成的目录存在
            mkdir -p "$target_dir"

            echo "正在执行任务，输出目录：$target_dir，batch_size=$batch_size，gradient_accumulation_steps=$gradient_accumulation_steps，filter_mode=$filter_mode，adjust_level=$adjust_level，beta=$beta"

            # 定义日志文件路径
            log_file="$target_dir/train.log"

            # 执行任务并重定向输出
            CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port="$master_port" softmax_dpo.py \
                --model_name "/home/hadoop-mining/dolphinfs_hdd_hadoop-mining/huxingyu05/Llama-2-7b-hf"  \
                --resume_from_checkpoint "./SFT" \
                --batch_size "$batch_size" \
                --gradient_accumulation_steps "$gradient_accumulation_steps" \
                --dataset lastfm \
                --prompt_path "./prompt/music.txt" \
                --learning_rate 1e-5 \
                --eval_step 0.033 \
                --beta "$beta" \
                --filter_mode "$filter_mode" \
                --custom_note "$custom_note"\
                --info_note "$info_note" \
                --loss_type "$loss_type"\
                --neg_num "$neg_num" \
                --num_train_epochs 3 \
                --output_dir "$target_dir" \
                --wandb_name "wandb_run_name_${target_dir##*/}_bs_${batch_size}_ga_${gradient_accumulation_steps}_fm_${filter_mode}_beta_${beta}" > "$log_file" 2>&1

            if [[ $? -ne 0 ]]; then
                echo "任务失败，输出目录：$target_dir，batch_size=$batch_size，gradient_accumulation_steps=$gradient_accumulation_steps，filter_mode=$filter_mode，adjust_level=$adjust_level，beta=$beta，停止执行后续任务。"
                exit 1
            fi

            echo "任务执行完毕，输出目录：$target_dir，batch_size=$batch_size，gradient_accumulation_steps=$gradient_accumulation_steps，filter_mode=$filter_mode，adjust_level=$adjust_level，beta=$beta，日志文件：$log_file"
          done
        done
      done
    done
  done
done

echo "所有任务已顺利执行完毕。"