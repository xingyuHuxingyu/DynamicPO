import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Literal, Tuple, Union

from .dynamicpo_trainer import (
    DPOTrainer as BaseDPOTrainer,
    _build_dynamic_beta_records,
    _group_batch_values,
    _select_boundary_critical_samples,
)


def _compute_dmpo_losses(
    chosen_logratios: torch.FloatTensor,
    boundary_critical_targets: List[List[torch.Tensor]],
    beta_values: Union[float, List[List[float]]],
) -> torch.FloatTensor:
    batch_losses = []
    for i, sample_negatives in enumerate(boundary_critical_targets):
        if isinstance(beta_values, list):
            k = len(sample_negatives)
            loss_value = -F.logsigmoid(
                sum(
                    beta_values[i][idx] * chosen_logratios[i] / k
                    - beta_values[i][idx] * neg_sample / k
                    for idx, neg_sample in enumerate(sample_negatives)
                )
            )
        else:
            k = len(sample_negatives)
            loss_value = -F.logsigmoid(
                beta_values * chosen_logratios[i]
                - sum(beta_values * neg_sample for neg_sample in sample_negatives) / k
            )
        batch_losses.append(loss_value)
    return torch.stack(batch_losses)


def _compute_mppo_losses(
    policy_chosen_logps: torch.FloatTensor,
    boundary_critical_targets: List[List[torch.Tensor]],
    beta_values: Union[float, List[List[float]]],
) -> torch.FloatTensor:
    batch_losses = []
    for i, sample_negatives in enumerate(boundary_critical_targets):
        if isinstance(beta_values, list):
            loss_value = -F.logsigmoid(
                sum(
                    beta_values[i][idx] * torch.exp(policy_chosen_logps[i])
                    - beta_values[i][idx] * torch.exp(neg_sample)
                    for idx, neg_sample in enumerate(sample_negatives)
                )
            )
        else:
            k = len(sample_negatives)
            loss_value = -F.logsigmoid(
                beta_values * k * torch.exp(policy_chosen_logps[i])
                - sum(beta_values * torch.exp(neg_sample) for neg_sample in sample_negatives)
            )
        batch_losses.append(loss_value)
    return torch.stack(batch_losses)


def _compute_sdpo_losses(
    chosen_logratios: torch.FloatTensor,
    boundary_critical_targets: List[List[torch.Tensor]],
    beta_values: Union[float, List[List[float]]],
) -> torch.FloatTensor:
    batch_losses = []
    for i, sample_negatives in enumerate(boundary_critical_targets):
        if isinstance(beta_values, list):
            temp = sum(
                torch.exp(beta_values[i][idx] * (neg_sample - chosen_logratios[i]))
                for idx, neg_sample in enumerate(sample_negatives)
            )
        else:
            temp = sum(torch.exp(beta_values * (neg_sample - chosen_logratios[i])) for neg_sample in sample_negatives)
        loss_value = -F.logsigmoid(-torch.log(temp + 1e-8))
        batch_losses.append(loss_value)
    return torch.stack(batch_losses)


def preference_loss(
    ref_model_enabled: bool,
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: Dict[str, torch.FloatTensor],
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: Dict[str, torch.FloatTensor],
    filter_mode: str,
    beta: float,
):
    chosen_logratios = policy_chosen_logps - reference_chosen_logps if ref_model_enabled else policy_chosen_logps
    chosen_rewards = (
        beta * (policy_chosen_logps - reference_chosen_logps).detach()
        if ref_model_enabled
        else beta * policy_chosen_logps.detach()
    )

    rejected_logratios = {}
    for key in policy_rejected_logps:
        rejected_logratios[key] = (
            policy_rejected_logps[key] - reference_rejected_logps[key]
            if ref_model_enabled
            else policy_rejected_logps[key]
        )

    rejected_rewards = {}
    for key in policy_rejected_logps:
        rejected_rewards[key] = (
            beta * (policy_rejected_logps[key] - reference_rejected_logps[key]).detach()
            if ref_model_enabled
            else beta * policy_rejected_logps[key].detach()
        )

    batch_rejected_logratios = _group_batch_values(rejected_logratios)
    all_rejected_logratios = [batch_rejected_logratios[i] for i in range(len(batch_rejected_logratios))]
    all_policy_rejected = _group_batch_values(policy_rejected_logps)
    all_policy_rejected_list = [all_policy_rejected[i] for i in range(len(all_policy_rejected))]

    if filter_mode == "MPPO":
        losses = _compute_mppo_losses(policy_chosen_logps, all_policy_rejected_list, beta)
        return (losses, None), None, None, chosen_rewards, rejected_rewards, (beta, beta, beta)

    if filter_mode == "DynamicPO_MPPO":
        boundary_critical_targets, boundary_critical_policy_values, model_discriminative_policy_means, batch_policy_rejected_list = (
            _select_boundary_critical_samples(
                target_values=policy_rejected_logps,
                threshold_values=chosen_logratios,
                policy_rejected_values=policy_rejected_logps,
            )
        )
        beta_records, pos2boundary_records, boundary2discriminative_records = _build_dynamic_beta_records(
            policy_chosen_logps,
            boundary_critical_policy_values,
            model_discriminative_policy_means,
            beta,
        )
        losses = _compute_mppo_losses(policy_chosen_logps, boundary_critical_targets, beta_records)
        return (
            (losses, None),
            boundary_critical_policy_values,
            batch_policy_rejected_list,
            chosen_rewards,
            rejected_rewards,
            (beta_records, pos2boundary_records, boundary2discriminative_records),
        )

    if filter_mode in {"SDPO", "S-DPO"}:
        losses = _compute_sdpo_losses(chosen_logratios, all_rejected_logratios, beta)
        return (losses, None), None, None, chosen_rewards, rejected_rewards, (beta, beta, beta)

    if filter_mode in {"DynamicPO_SDPO", "SDPO_hard_negative_dynamic_beta_fixed_margin"}:
        boundary_critical_targets, boundary_critical_policy_values, model_discriminative_policy_means, batch_policy_rejected_list = (
            _select_boundary_critical_samples(
                target_values=rejected_logratios,
                threshold_values=policy_chosen_logps,
                policy_rejected_values=policy_rejected_logps,
            )
        )
        beta_records, pos2boundary_records, boundary2discriminative_records = _build_dynamic_beta_records(
            policy_chosen_logps,
            boundary_critical_policy_values,
            model_discriminative_policy_means,
            beta,
        )
        losses = _compute_sdpo_losses(chosen_logratios, boundary_critical_targets, beta_records)
        return (
            (losses, None),
            boundary_critical_policy_values,
            batch_policy_rejected_list,
            chosen_rewards,
            rejected_rewards,
            (beta_records, pos2boundary_records, boundary2discriminative_records),
        )

    raise ValueError(f"Unsupported filter_mode for exploratory study trainer: {filter_mode}")


class DPOTrainer(BaseDPOTrainer):
    def __init__(self, *args, loss_type: str = "", **kwargs):
        self.loss_type = loss_type
        super().__init__(*args, **kwargs)

    def _get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
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

        (losses, margin_record), boundary_critical_reward, all_negative_reward, chosen_rewards, rejected_rewards, (
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
            beta=self.beta,
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

        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().tolist()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        for key in policy_rejected_logps:
            metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()

        def convert_tensors_to_list(obj: Any):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            if isinstance(obj, list):
                return [convert_tensors_to_list(item) for item in obj]
            return obj

        metrics["boundary_critical_reward"] = convert_tensors_to_list(boundary_critical_reward)
        metrics["all_negative_reward"] = convert_tensors_to_list(all_negative_reward)
        metrics["margin_record"] = convert_tensors_to_list(margin_record)
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

        if isinstance(beta_used_list_or_float, list):
            metrics[f"{prefix}/beta_used"] = beta_used_list_or_float
        elif isinstance(beta_used_list_or_float, float):
            metrics[f"{prefix}/beta_used"] = [beta_used_list_or_float]

        metrics[f"{prefix}/delta_pos2boundary_record"] = delta_pos2boundary_record
        metrics[f"{prefix}/delta_boundary2discriminative_record"] = delta_boundary2discriminative_record

        return losses.mean(), metrics
