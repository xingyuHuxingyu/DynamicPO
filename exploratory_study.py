import os
import random
import re

import fire
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from trainer.exploratory_study_trainer import DPOTrainer
from Prompt import Prompt

os.environ["WANDB_MODE"] = "disabled"
random.seed(1958)


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _model_name_to_slug(model_name: str) -> str:
    base_name = os.path.basename(model_name.rstrip("/")) or "model"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base_name).strip("-").lower()
    return slug or "model"


def _append_model_suffix(value: str, model_name: str) -> str:
    if not value:
        return value
    slug = _model_name_to_slug(model_name)
    if slug in value.lower():
        return value
    return f"{value.rstrip('/')}_{slug}"


def _note_to_slug(note: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", note).strip("-").lower()
    return slug


def _append_note_suffix(value: str, note: str) -> str:
    if not value or not note:
        return value
    slug = _note_to_slug(note)
    if not slug or slug in value.lower():
        return value
    return f"{value.rstrip('/')}_{slug}"


def train(
    output_dir="./outputs/exploratory_study",
    model_name="",
    prompt_path="./prompt/music.txt",
    dataset="lastfm",
    resume_from_checkpoint="",
    wandb_name="",
    custom_note="",
    beta: float = 1.0,
    filter_mode: str = "DynamicPO_MPPO",
    info_note: str = "",
    loss_type: str = "w_ref",
    neg_num: int = 15,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    eval_step: int = 4,
):
    output_dir = _append_note_suffix(output_dir, custom_note)
    wandb_name = _append_note_suffix(wandb_name, custom_note)
    output_dir = _append_model_suffix(output_dir, model_name)
    wandb_name = _append_model_suffix(wandb_name, model_name)

    if _is_main_process():
        print(f"custom_note: {custom_note}")
        print(f"info_note: {info_note}")
        print(f"beta: {beta}")
        print(f"filter_mode: {filter_mode}")
        print(f"loss_type: {loss_type}")
        print(f"neg_num: {neg_num}")
        print(f"batch_size: {batch_size}")
        print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
        print(f"output_dir: {output_dir}")
        print(f"wandb_name: {wandb_name}")

    if dataset == "lastfm":
        data_files = {
            "train": "./data/lastfm-sft-cans20/lastfm-train.json",
            "validation": "./data/lastfm-sft-cans20/lastfm-val.json",
        }
    elif dataset == "goodreads":
        data_files = {
            "train": "./data/goodreads-sft-cans20/goodreads-train.json",
            "validation": "./data/goodreads-sft-cans20/goodreads-val.json",
        }
    elif dataset == "steam":
        data_files = {
            "train": "./data/steam-sft-cans20/steam-train.json",
            "validation": "./data/steam-sft-cans20/steam-val.json",
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    def convert_dict_to_prompt(d: dict):
        prompt_builder = Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        prompt_builder.historyList = d["historyList"]
        prompt_builder.itemList = d["itemList"]
        prompt_builder.trueSelection = d["trueSelection"]
        return prompt_builder

    def process_data(examples):
        processed = {"prompt": [], "chosen": []}
        for i in range(1, neg_num + 1):
            processed[f"rejected{i}"] = []

        columns = list(examples.keys())
        for i in range(len(examples[columns[0]])):
            data_point = {
                "trueSelection": examples["trueSelection"][i],
                "itemList": examples["itemList"][i],
                "historyList": examples["historyList"][i],
            }
            prompt = str(convert_dict_to_prompt(data_point))
            chosen = data_point["trueSelection"]
            negative_items = [item for item in data_point["itemList"] if item != chosen]
            sampled_negs = random.sample(negative_items, neg_num)

            processed["prompt"].append(prompt)
            processed["chosen"].append(chosen)
            for idx, rejected in enumerate(sampled_negs, start=1):
                processed[f"rejected{idx}"].append(rejected)
        return processed

    data = load_dataset("json", data_files=data_files)
    columns = data["train"].column_names

    train_data = data["train"].map(
        process_data,
        remove_columns=columns,
        num_proc=8,
        batched=True,
    ).shuffle(seed=42)
    val_data = data["validation"].map(
        process_data,
        remove_columns=columns,
        num_proc=8,
        batched=True,
    ).shuffle(seed=42)
    if val_data.num_rows > 2000:
        val_data = val_data.select(range(2000))

    device_index = Accelerator().process_index
    device_map = {"": device_index}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if "Llama-3" in model_name:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=bnb_config,
        )
    elif "Qwen" in model_name:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=bnb_config,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=bnb_config,
        )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    base_model = PeftModel.from_pretrained(
        base_model,
        resume_from_checkpoint,
        is_trainable=True,
    )
    if _is_main_process():
        base_model.print_trainable_parameters()

    ref_enable = loss_type != "wo_ref" and filter_mode not in {
        "MPPO",
        "DynamicPO_MPPO",
        "mppo_multineg_loss",
        "mppo_hard_negative_dynamic_beta_fixed_margin",
    }
    reference_model = None
    if ref_enable:
        if "Llama-3" in model_name:
            model_ref = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                quantization_config=bnb_config,
            )
        elif "Qwen" in model_name:
            model_ref = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                quantization_config=bnb_config,
            )
        else:
            model_ref = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                quantization_config=bnb_config,
            )
        reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
        if _is_main_process():
            reference_model.print_trainable_parameters()

    if "Llama-3" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
    elif "Qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
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
        report_to="wandb",
        run_name=wandb_name,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        ddp_find_unused_parameters=False,
    )

    dpo_trainer = DPOTrainer(
        base_model,
        ref_model=reference_model,
        args=training_args,
        beta=beta,
        filter_mode=filter_mode,
        loss_type=loss_type,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)


if __name__ == "__main__":
    fire.Fire(train)
