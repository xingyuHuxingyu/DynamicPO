# Position the number of processes specified after the --nproc_per_node flag
OUTPUT_DIR="./lastfm-sft"
mkdir -p $OUTPUT_DIR


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25664 sft.py \
        --model_name "your_base_llm_model"  \
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
        --wandb_name wandb_run_name \
> "${OUTPUT_DIR}/lastfm_sft.log" 2>&1