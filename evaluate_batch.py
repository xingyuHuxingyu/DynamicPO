import torch

import transformers
from typing import List
from datasets import load_dataset
import json
from transformers import LlamaForCausalLM, LlamaTokenizer,GenerationConfig
from peft import PeftModel
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from fire import Fire
from tqdm import tqdm

device_map = "auto"
def evaluate(
    model,
    tokenizer,
    val_data,
    batch_size: int = 32
):
    model_name = getattr(tokenizer, "name_or_path", "") or getattr(model.config, "_name_or_path", "")
    use_token_level_slice = "llama-3" in model_name.lower()
    
    def output_generate(
        prompts,
        temperature = 0,
    ):
        # print([len(prompt) for prompt in prompts])
        inputs = tokenizer(prompts,return_tensors="pt",truncation=True,padding=True,max_length=1024).to(model.device)
        generation_config = GenerationConfig(
            # temperature = temperature,
            do_sample = False,
        )
        generation_output = model.generate(
            **inputs,
            pad_token_id = tokenizer.pad_token_id,
            generation_config = generation_config,
            return_dict_in_generate = True,
            output_scores = True,
            max_new_tokens = 20
        )
        if use_token_level_slice:
            prompt_token_count = inputs["input_ids"].shape[1]
            generated_tokens = generation_output.sequences[:, prompt_token_count:]
            output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        else:
            output = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        output = [_.strip() for _ in output]
        return output
    
    targets = []
    inputs = []
    cans = []
    for elm in val_data:
        prompt = elm["prompt"]
        target = elm["trueSelection"]
        targets.append(target)
        inputs.append(prompt)
        cans.append(elm["itemList"])

    batch_num = (len(inputs)-1)// batch_size + 1
    score = 0
    valid = 0
    for i in tqdm(range(batch_num), desc="Testing..."):
        start = i*batch_size
        end = min(len(inputs), start+batch_size)
        batch_inputs = inputs[start:end]
        outputs = output_generate(batch_inputs)
        batch_targets = targets[start:end]
        batch_cans = cans[start:end]
        for input_text, output, target, candidates in zip(batch_inputs, outputs, batch_targets, batch_cans):
            if use_token_level_slice:
                selection = output
            else:
                selection = output[len(input_text):]
            num_cans = sum([1 for can in candidates if can in selection])  
            print(input_text)
            print(candidates)
            print(selection)
            print([target])
            if num_cans == 1:
                valid += 1
                if target in selection:
                    score += 1
                    print(f"score increased to {score}")
            print("\n")

            
    return score/len(inputs), valid/len(inputs)
