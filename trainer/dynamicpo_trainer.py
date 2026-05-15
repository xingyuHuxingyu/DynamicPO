# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/lifcenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import importlib
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
import torch.distributed as dist
from .utils import DPODataCollatorWithPadding, pad_to_length
from sklearn.cluster import KMeans
from omegaconf import DictConfig,OmegaConf

import torch.distributed as dist
import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

FIXED_MARGIN = 6.0

def is_peft_available():
    return importlib.util.find_spec("peft") is not None

if is_peft_available():
    from peft import get_peft_model, prepare_model_for_kbit_training


def _group_batch_values(values_dict: Dict[str, torch.FloatTensor]) -> Dict[int, List[torch.Tensor]]:
    grouped_values: Dict[int, List[torch.Tensor]] = {}
    first_key = next(iter(values_dict))
    batch_size = len(values_dict[first_key])
    for _, value in values_dict.items():
        for i in range(batch_size):
            grouped_values.setdefault(i, []).append(value[i])
    return grouped_values


def _to_python_nested_list(values: Dict[int, List[torch.Tensor]]) -> List[List[float]]:
    nested_values: List[List[float]] = []
    for i in range(len(values)):
        nested_values.append([float(tensor.detach().cpu().item()) for tensor in values[i]])
    return nested_values


def _cluster_high_reward_indices(rewards: List[torch.Tensor], num_clusters: int = 3) -> List[int]:
    reward_array = np.array([float(tensor.detach().cpu().item()) for tensor in rewards])
    cluster_num = min(num_clusters, len(reward_array))
    if cluster_num <= 1:
        return list(range(len(reward_array)))

    kmeans = KMeans(n_clusters=cluster_num, random_state=42).fit(reward_array.reshape(-1, 1))
    labels = kmeans.labels_
    cluster_means = [np.mean(reward_array[labels == idx]) for idx in range(cluster_num)]
    return np.where(labels == int(np.argmax(cluster_means)))[0].tolist()


def _select_boundary_critical_samples(
    target_values: Dict[str, torch.FloatTensor],
    threshold_values: torch.FloatTensor,
    policy_rejected_values: Dict[str, torch.FloatTensor],
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], List[torch.Tensor], List[List[float]]]:
    batch_target = _group_batch_values(target_values)
    batch_policy_rejected = _group_batch_values(policy_rejected_values)
    batch_policy_rejected_list = _to_python_nested_list(batch_policy_rejected)

    boundary_critical_sample_list: List[List[torch.Tensor]] = []
    policy_boundary_critical_list: List[List[torch.Tensor]] = []
    policy_model_discriminative_mean_list: List[torch.Tensor] = []

    for sample_idx in range(len(batch_policy_rejected)):
        sample_target_values = batch_target[sample_idx]
        sample_policy_values = batch_policy_rejected[sample_idx]

        boundary_critical_indices = [
            idx for idx, value in enumerate(sample_policy_values) if value >= threshold_values[sample_idx]
        ]
        if len(boundary_critical_indices) < 1:
            boundary_critical_indices = _cluster_high_reward_indices(sample_policy_values)

        model_discriminative_indices = [
            idx for idx in range(len(sample_policy_values)) if idx not in boundary_critical_indices
        ]
        if len(model_discriminative_indices) < 1:
            model_discriminative_indices = boundary_critical_indices

        boundary_critical_sample_list.append([sample_target_values[idx] for idx in boundary_critical_indices])
        policy_boundary_critical_list.append([sample_policy_values[idx] for idx in boundary_critical_indices])
        policy_model_discriminative_mean_list.append(
            sum(sample_policy_values[idx] for idx in model_discriminative_indices) / len(model_discriminative_indices)
        )

    return (
        boundary_critical_sample_list,
        policy_boundary_critical_list,
        policy_model_discriminative_mean_list,
        batch_policy_rejected_list,
    )


def _build_dynamic_beta_records(
    policy_chosen_logps: torch.FloatTensor,
    policy_boundary_critical_list: List[List[torch.Tensor]],
    policy_model_discriminative_mean_list: List[torch.Tensor],
    beta: float,
    margin: float = FIXED_MARGIN,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    sample_level_beta_record: List[List[float]] = []
    delta_pos2boundary_record: List[List[float]] = []
    delta_boundary2discriminative_record: List[List[float]] = []

    for i in range(policy_chosen_logps.size(0)):
        beta_record = []
        pos2boundary_record = []
        boundary2discriminative_record = []

        for boundary_critical_logp in policy_boundary_critical_list[i]:
            beta_value = beta
            delta_pos2boundary = policy_chosen_logps[i] - boundary_critical_logp
            delta_boundary2discriminative = (
                boundary_critical_logp - policy_model_discriminative_mean_list[i]
            )

            pos2boundary_record.append(float(delta_pos2boundary.detach().cpu().item()))
            boundary2discriminative_record.append(float(delta_boundary2discriminative.detach().cpu().item()))

            delta_diff = delta_pos2boundary - delta_boundary2discriminative - margin
            delta_beta = delta_diff / (abs(delta_boundary2discriminative) + abs(delta_pos2boundary))
            delta_beta = torch.tanh(delta_beta) * 0.5

            beta_value = beta_value * (1 + delta_beta)
            beta_record.append(float(beta_value.detach().cpu().item()))

        sample_level_beta_record.append(beta_record)
        delta_pos2boundary_record.append(pos2boundary_record)
        delta_boundary2discriminative_record.append(boundary2discriminative_record)

    return sample_level_beta_record, delta_pos2boundary_record, delta_boundary2discriminative_record



def preference_loss(
                    ref_model_enabled:bool,
                    policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    filter_mode: str,
                    beta: float) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """


    chosen_logratios = policy_chosen_logps - reference_chosen_logps if ref_model_enabled else policy_chosen_logps
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach() if ref_model_enabled else beta*policy_chosen_logps.detach()
   
    rejected_logratios = {}
    for key in policy_rejected_logps:
        rejected_logratios[key] = policy_rejected_logps[key] - reference_rejected_logps[key] if ref_model_enabled else policy_rejected_logps[key]

    rejected_rewards = {}
    for key in policy_rejected_logps:
        rejected_rewards[key] = beta * (policy_rejected_logps[key] - reference_rejected_logps[key]).detach() if ref_model_enabled else beta*policy_rejected_logps[key].detach()



   
   

    if filter_mode== 'DMPO':
        K=len(rejected_logratios.keys())
        losses=-F.logsigmoid(beta*chosen_logratios-sum(beta*rejected_logratios[key] for key in rejected_logratios)/K)
        return (losses,None),None,None,chosen_rewards,rejected_rewards,(beta,beta,beta)
    
    
  
  
    elif filter_mode == "DynamicPO_DMPO":
        (
            boundary_critical_sample_list,
            policy_boundary_critical_list,
            policy_model_discriminative_mean_list,
            batch_policy_rejected_list,
        ) = _select_boundary_critical_samples(
            target_values=rejected_logratios,
            threshold_values=policy_chosen_logps,
            policy_rejected_values=policy_rejected_logps,
        )

        (
            sample_level_beta_record,
            delta_pos2boundary_record,
            delta_boundary2discriminative_record,
        ) = _build_dynamic_beta_records(
            policy_chosen_logps=policy_chosen_logps,
            policy_boundary_critical_list=policy_boundary_critical_list,
            policy_model_discriminative_mean_list=policy_model_discriminative_mean_list,
            beta=beta,
        )

        batch_losses_list = []
        for i in range(chosen_logratios.size(0)):
            k = len(boundary_critical_sample_list[i])
            dmpo_loss_item = -F.logsigmoid(
                sum(
                    sample_level_beta_record[i][idx] * chosen_logratios[i] / k
                    - sample_level_beta_record[i][idx] * neg_sample_logratio / k
                    for idx, neg_sample_logratio in enumerate(boundary_critical_sample_list[i])
                )
            )
            batch_losses_list.append(dmpo_loss_item)

        losses = torch.stack(batch_losses_list)

        return (
            (losses, None),
            policy_boundary_critical_list,
            batch_policy_rejected_list,
            chosen_rewards,
            rejected_rewards,
            (sample_level_beta_record, delta_pos2boundary_record, delta_boundary2discriminative_record),
        )


   
    

class DPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        beta: float = 1.0,
        filter_mode:str='',
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            data_collator = DPODataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        self.beta = beta
        self.ref_model = ref_model
        self.policy=model
 
        self.filter_mode=filter_mode


    
        self.delta_boundary2discriminative_mean = torch.zeros(1, device='cuda')
        self.delta_boundary2discriminative_std = torch.zeros(1, device='cuda')

        self.delta_pos2boundary_mean = torch.zeros(1, device='cuda')
        self.delta_pos2boundary_std = torch.zeros(1, device='cuda')

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            None,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        # Since we inherit from trainer we always have access to an accelerator
        if hasattr(self, "accelerator"):
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True) if self.ref_model is not None else None
        else:
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        
        rejected_max_len = max([batch[key].shape[1] for key in batch if key.startswith("rejected") and key.endswith("_input_ids")])
        max_length = max(batch["chosen_input_ids"].shape[1], rejected_max_len)
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                # concatenated_key = k.replace("rejected", "concatenated")
                prefix = k.split("_")[0]
                concatenated_key = "concatenated" + k[len(prefix):]
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)
        return concatenated_batch


    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor], torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False#
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        step = batch["chosen_input_ids"].shape[0]
        rejected_logps = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logps[f"rejected{cnt}"] = all_logps[step*cnt : step*(cnt+1)]

        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logits[f"rejected{cnt}"] = all_logits[step*cnt : step*(cnt+1)]
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
    


 
    def _get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)


        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, batch) if self.ref_model is not None else (None, None, None, None)
        

        chosen_logratios = policy_chosen_logps - reference_chosen_logps if self.ref_model is not None else None
        rejected_logratios = {}
        for key in policy_rejected_logps:
            rejected_logratios[key] = policy_rejected_logps[key] - reference_rejected_logps[key] if self.ref_model is not None else None


        (
            losses,
            margin_record,
        ), boundary_critical_reward, all_negative_reward, chosen_rewards, rejected_rewards, (
            beta_used,
            delta_pos2boundary_record,
            delta_boundary2discriminative_record,
        ) = preference_loss(
            ref_model_enabled=self.ref_model is not None,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            filter_mode=self.filter_mode,
            beta=self.beta
        
        )



        
        reward_accuracies = None
        for key in rejected_rewards:
            if reward_accuracies is None:
                reward_accuracies = (chosen_rewards > rejected_rewards[key]).float()
            else:
                reward_accuracies *= (chosen_rewards > rejected_rewards[key]).float()

        prefix = "eval_" if train_eval == "eval" else "train_"
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/{key}"] = rejected_rewards[key].cpu().numpy().mean()

        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().tolist() #.mean()
           

        for key in rejected_rewards:
            metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        for key in policy_rejected_logps:
            metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()



        def convert_tensors_to_list(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            elif isinstance(obj, list):
                return [convert_tensors_to_list(item) for item in obj]
            else:
                return obj  

 
        metrics["boundary_critical_reward"] = convert_tensors_to_list(boundary_critical_reward)
        metrics["all_negative_reward"] = convert_tensors_to_list(all_negative_reward)
        metrics[f"margin_record"] = convert_tensors_to_list(margin_record)


       
        
        

        if isinstance(delta_boundary2discriminative_record, list) and len(delta_boundary2discriminative_record) > 0:
            flattened_deltas = []
            for sample_deltas in delta_boundary2discriminative_record:
                if isinstance(sample_deltas, list):
                    flattened_deltas.extend(sample_deltas)
                else:
                    flattened_deltas.append(sample_deltas)

        if isinstance(delta_pos2boundary_record, list) and len(delta_pos2boundary_record) > 0:
            
            flattened_pos2boundary = []
            
            for sample_deltas in delta_pos2boundary_record:
                if isinstance(sample_deltas, list):
                    flattened_pos2boundary.extend(sample_deltas)
                else:
                    flattened_pos2boundary.append(sample_deltas)
            




        
        metrics[f"{prefix}/delta_boundary2discriminative_mean"] = (
            self.delta_boundary2discriminative_mean.cpu().numpy().tolist()
        )
        metrics[f"{prefix}/delta_boundary2discriminative_std"] = (
            self.delta_boundary2discriminative_std.cpu().numpy().tolist()
        )
        metrics[f"{prefix}/delta_pos2boundary_mean"] = self.delta_pos2boundary_mean.cpu().numpy().tolist()
        metrics[f"{prefix}/delta_pos2boundary_std"] = self.delta_pos2boundary_std.cpu().numpy().tolist()
        if isinstance(beta_used, float):
            beta_used_list_or_float = beta_used
            
        elif isinstance(beta_used, list):
            beta_used_list_or_float = beta_used
        else:
            beta_used_list_or_float = beta_used.cpu().numpy().tolist()
            delta_pos2boundary_record = delta_pos2boundary_record.cpu().numpy().tolist()
            delta_boundary2discriminative_record = delta_boundary2discriminative_record.cpu().numpy().tolist()
        if isinstance(beta_used_list_or_float, list):
            metrics[f"{prefix}/beta_used"] = beta_used_list_or_float
            metrics[f"{prefix}/delta_pos2boundary_record"] = delta_pos2boundary_record
            metrics[f"{prefix}/delta_boundary2discriminative_record"] = delta_boundary2discriminative_record
        elif isinstance(beta_used_list_or_float, float):
            metrics[f"{prefix}/beta_used"] = [beta_used_list_or_float]
            metrics[f"{prefix}/delta_pos2boundary_record"] = delta_pos2boundary_record
            metrics[f"{prefix}/delta_boundary2discriminative_record"] = delta_boundary2discriminative_record

        
        return losses.mean(), metrics
    

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics = self._get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.config.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        reference_output = self.ref_model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.config.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        ) if self.ref_model is not None else None

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id) if self.ref_model is not None else None
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True) if self.ref_model is not None else None

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self._get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "logits_test/chosen": metrics["logits_test/chosen"],
            # "logits_test/rejected": metrics["logits_test/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)



    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            if key in ['filtered_losses', 'margin_record','losses_list','boundary_critical_reward',
                       'actual_losses_list','all_negative_reward','local_mask','eval_/beta_used','train_/beta_used','train_/delta_boundary2discriminative_record','train_/delta_pos2boundary_record','train_rewards/accuracies']:
                logs[key] = metrics  
            else:
                logs[key] = torch.tensor(metrics).float().mean().item()



        del self._stored_metrics[train_eval]
        return super().log(logs)
