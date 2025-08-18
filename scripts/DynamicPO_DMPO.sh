#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=$((25000 + RANDOM % 2000)) dynamicpo_dmpo.py \
    --model_name "your_base_llm_model"  \
    --resume_from_checkpoint "your_sft_checkpoint" \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --dataset lastfm \
    --prompt_path "./prompt/music.txt" \
    --learning_rate 1e-5 \
    --eval_step 0.033 \
    --beta 1.0 \
    --filter_mode "DMPO_hard_negative" \
    --custom_note "DMPO"\
    --info_note "" \
    --loss_type "w_ref"\
    --neg_num 5 \
    --num_train_epochs 3 \
    --output_dir "./DynamicPO_DMPO/DMPO_neg_5_fm_DMPO_hard_negative_beta_1.0" \
    --wandb_name "wandb_run_name_DMPO_neg_5_fm_DMPO_hard_negative_beta_1.0_bs_4_ga_8_fm_DMPO_hard_negative_beta_1.0" > "./DynamicPO_DMPO/DMPO_neg_5_fm_DMPO_hard_negative_beta_1.0/train.log" 2>&1
