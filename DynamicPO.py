import os
import torch
import re
import random
import numpy as np
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trainer.dynamicpo_trainer import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from trainer.utils import Prompt
import json
import torch
import bitsandbytes as bnb
from accelerate import Accelerator
import fire

import matplotlib.pyplot as plt
import os 
os.environ["WANDB_MODE"] = 'disabled'
random.seed(1958)


def train(
    #train
    output_dir="./",
    model_name ="",
    prompt_path = "./prompt/music.txt",
    dataset="lastfm",
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter
    wandb_name: str = "",   # the name of the wandb run
    custom_note:str="",
    # training hyperparameters
    beta: float = 1.0,
    
    filter_mode:str='',
    info_note:str='',
    loss_type:str="w_ref",
    neg_num: int = 15,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    eval_step = 4
):
    
    print(f"custom_note:{custom_note}")
    print(f"info_note:{info_note}")
    print(f"beta: {beta}")
    print(f"filter_mode: {filter_mode}")
    print(f"loss_type:{loss_type}")
    print(f"neg_num: {neg_num}")
    print(f"batch_size: {batch_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    
    if dataset=="lastfm":
        data_files = {
            "train":"./data/lastfm-sft-cans20/lastfm-train.json",
            "validation": "./data/lastfm-sft-cans20/lastfm-val.json",
        }
    elif dataset=="goodreads":
        data_files = {
            "train":"./data/goodreads-sft-cans20/goodreads-train.json",
            "validation": "./data/goodreads-sft-cans20/goodreads-val.json",
        }
    elif dataset=="steam":
        data_files = {
            "train":"./data/steam-sft-cans20/steam-train.json",
            "validation": "./data/steam-sft-cans20/steam-val.json",
        }






    def convert_dict_to_prompt(d:dict):
        t = Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t

    def process_data(examples):
        dic = {"prompt":[], "chosen":[]}
        for i in range(1, neg_num+1):
            dic[f"rejected{i}"] = []
        columns = list(examples.keys())
        for i in range(len(examples[columns[0]])):
            data_point = {}
            data_point["trueSelection"] = examples["trueSelection"][i]
            data_point["itemList"] = examples["itemList"][i]
            data_point["historyList"] = examples["historyList"][i]  
            t = convert_dict_to_prompt(data_point)
            prompt = str(t)
            chosen = data_point["trueSelection"]
            negative_items = [item for item in data_point["itemList"] if item != data_point["trueSelection"]]
            sample_negs = random.sample(negative_items, neg_num)
            dic["prompt"].append(prompt)
            dic["chosen"].append(chosen)
            cnt = 0  
            for rejected in sample_negs:
                cnt += 1
                dic[f"rejected{cnt}"].append(rejected)
        return dic


    data = load_dataset("json", data_files=data_files)

    columns = data["train"].column_names
    train_data = data["train"].map(process_data, remove_columns=columns, \
                                    num_proc=8, batched=True).shuffle(seed=42)#.select(range(256))
    for i in range(0,4):
        print(train_data[i])

    # random 2000 samples for validation
    val_data = data["validation"].map(process_data, remove_columns=columns, \
                                        num_proc=8, batched=True).shuffle(seed=42)
    if val_data.num_rows > 2000:
        val_data = val_data.select(range(2000))
    
   # print(val_data)


    device_index = Accelerator().process_index
    device_map = {"": device_index}
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


    base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                device_map=device_map, 
                                                quantization_config=bnb_config)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint,#)#, 
                                        is_trainable=True)
    base_model.print_trainable_parameters()

    ref_enable=loss_type not in ["wo_ref"]
    if ref_enable:
        model_ref = AutoModelForCausalLM.from_pretrained(model_name,
                                                    device_map=device_map, 
                                                  
                                                    quantization_config=bnb_config)
        reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
        reference_model.print_trainable_parameters()



    if 'Llama-3' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=num_train_epochs, 
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=eval_step,
        save_total_limit=100,
        evaluation_strategy="no",
        eval_steps=eval_step,
        load_best_model_at_end=False,
        logging_steps=1,
        output_dir=output_dir,
        report_to = "wandb",
        run_name = wandb_name,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        ddp_find_unused_parameters=False,
    )

    dpo_trainer = DPOTrainer(
        base_model,
        ref_model=reference_model if ref_enable else None,
        args=training_args,
        beta=beta,
        filter_mode=filter_mode,
        loss_type=loss_type,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len
    )



    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)