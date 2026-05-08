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

def is_peft_available():
    return importlib.util.find_spec("peft") is not None

if is_peft_available():
    from peft import get_peft_model, prepare_model_for_kbit_training



def preference_loss(
                    ref_model_enabled:bool,
                    policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    filter_mode: str,
                    adjust_level: str,
                    beta_clamp:bool,
                    beta_clamp_min:float,
                    beta_clamp_max:float,
                    beta: float,
                    mode_weight: float,
                    alpha_a: float,
                    epoch: float,
                    betadpo_delay:bool,
                    gap_mean=None, gap_std=None,delta_hard_neg2easy_neg_mean=None,delta_pos2hardneg_mean=None) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
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

    ## logratios 是 policy logps - reference logps的 差异
    #rewards在前面乘以beta
    ### 可以尝试simpo
    ## s-dpo multi neg sample调整
    chosen_logratios = policy_chosen_logps - reference_chosen_logps if ref_model_enabled else policy_chosen_logps
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach() if ref_model_enabled else beta*policy_chosen_logps.detach()
   
    rejected_logratios = {}
    for key in policy_rejected_logps:
        rejected_logratios[key] = policy_rejected_logps[key] - reference_rejected_logps[key] if ref_model_enabled else policy_rejected_logps[key]

    rejected_rewards = {}
    for key in policy_rejected_logps:
        rejected_rewards[key] = beta * (policy_rejected_logps[key] - reference_rejected_logps[key]).detach() if ref_model_enabled else beta*policy_rejected_logps[key].detach()

    ## 涉及到多个负样本，需要求正负样本gap的均值，有多对pairwise:chosen,rejected
    #### 1. select data
    ##   DPO loss中的 winner,loser样本的 log-ratios 的差值，A_gap,A

    A_gap_sum = sum((chosen_logratios - rejected_logratios[key]).detach() for key in rejected_logratios)
    A_gap_mean = A_gap_sum / len(rejected_logratios)
    
    A=A_gap_mean
    #A:正负样本logratio差值
    mean = gap_mean
    std = gap_std

    
    weight_sample = torch.exp(-0.5 * ((A - mean) / std).pow(2))
    weight_sample_softmax = torch.nn.functional.softmax(weight_sample, dim=0)
    
  
    if filter_mode == 'beta_DPO_DMPO_hard_negative_cluster_3':
        sample_num = int( weight_sample.numel() * (1 - mode_weight) ) #采样比例
        sample_index = torch.multinomial(weight_sample_softmax, sample_num, replacement=False)
        one_hot_like = torch.zeros_like(weight_sample)
        one_hot_like[sample_index] = 1
        global_mask = one_hot_like.detach()

        # batch内的平均 margin,dpo loss    
        A_used = torch.mean(A[sample_index]) # if adjust_level == 'batch_level'
        ##### else A
        alpha_a=0.6
        beta_used = beta * (1 + alpha_a * (A_used - gap_mean))
        beta_used=beta_used.detach()
        beta_used = beta_used.clamp(min=1e-3)

        beta_used = beta_used.clamp(min=beta_clamp_min,max=beta_clamp_max) if beta_clamp==True else beta_used
        
        
        hard_negative_sample_list=[]
        batch_simpo_policy_hard_neg_list=[]
        batch_simpo_policy_easy_neg_list=[]
        # policy_logps_list=[]

        # 1.dpo rejected logratios
        batch_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        # 逆序记录：样本，rejected1,2,3,4的logratio

        #遍历 rejected logratio字典:    key:rejected1/2/3/4,   value:batchsize的rejected1样本概率
        for (key,value) in rejected_logratios.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(rejected_logratios['rejected1'])):
                batch_rejected.setdefault(i, []).append(value[i])

        # 2.policy rejected logps(simpo -- nll)
        
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
       
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)


        # 3.reference rejected logs
        batch_reference_rejected_logps={}
        for (key,value) in reference_rejected_logps.items():
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(reference_rejected_logps['rejected1'])):
                batch_reference_rejected_logps.setdefault(i, []).append(value[i])
               
        # avg_logratio=[] # batchsize,avg_negsample_reward
        easy_neg_logratio=[]

        # 负样本过滤/动态选择 最粗糙方案:log ratio小于平均值的加入logratio
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
           
            key = num  # 直接使用 num 作为键,(batch-size,)
            
            sample_neg_list=[]
            easy_neg_list=[]
            simpo_policy_hard_neg_list=[]
            simpo_policy_easy_neg_list=[]
            dynamic_flag=True

            # simpo reward 作为筛选，实际添加dpo的reward项
            if dynamic_flag==True:
                # 1.存在reward大于正样本
                for idx,value in enumerate(batch_policy_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= policy_chosen_logps[key]:
                        sample_neg_list.append(batch_rejected[key][idx])
                        
            
                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
                    ####原始数据准备（假设batch_policy_rejected[key]是reward列表）
                    rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0]

                    # 构建结果列表
                    sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                    simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                # 如果reward有大于正样本的，则进入easyneg的补充处理
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_rejected[key]) if val in sample_neg_list]
                    # 构建simpo_policy_hard_neg_list
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    
                   # 构建easy_neg_list和simpo_policy_easy_neg_list
                    
                    
                    if len(hard_indices)==len(batch_policy_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    

            else:
                rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                # 使用K-Means聚类（分为3类）
                kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                # 找到最高reward的类（按均值判断）
                cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0]

                # 构建结果列表
                sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
            # dpo reward related
            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
            # simpo reward related
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            
            # simpo reward related
            #batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            #batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}

      
      

        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本
           
            
            
            K=len(hard_negative_sample_list[i])
            DMPO_loss_item=-F.logsigmoid(beta_used*chosen_logratios-sum(beta_used*neg_sample_logratio for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i]) )/K)
            batch_losses_list.append(DMPO_loss_item)

        # 将所有样本的 loss 堆叠成一个 tensor
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]

        # 2.dpo loss:
     
        return (losses,None),A,A,chosen_rewards,rejected_rewards,(beta_used,beta_used,beta_used)

    elif filter_mode == 'SDPO':
        #A_filtered = A
        temp = sum(torch.exp(beta * (rejected_logratios[key] - chosen_logratios)) for key in rejected_logratios)
        temp1 = -torch.log(temp)
        losses = -F.logsigmoid(temp1)
        return (losses,None),A,None,chosen_rewards,rejected_rewards,(beta, beta, beta)    #torch.tensor(beta_total_record)

  
    elif filter_mode == 'reward_ratio_visual':
        A_filtered = A
        
        # 统计负样本 logratios 高于正样本的情况
        higher_neg_count = 0
        total_neg_count = 0
        
        # 记录所有正负样本对的 logratios
        all_pairs_data = []
        
        for key in rejected_logratios:
            # 计算当前负样本中 logratios 高于正样本的数量
            higher_mask = rejected_logratios[key] > chosen_logratios
            higher_count = higher_mask.sum().item()
            higher_neg_count += higher_count
            total_neg_count += len(rejected_logratios[key])
            
            # 收集所有样本对的数据
            for i in range(len(chosen_logratios)):
                all_pairs_data.append({
                    "rank": dist.get_rank(),
                    "batch_idx": i,
                    "neg_key": key,
                    "chosen_logratio": chosen_logratios[i].item(),
                    "rejected_logratio": rejected_logratios[key][i].item(),
                    "is_higher": bool(higher_mask[i].item())
                })
        
        # 收集所有 GPU 上的统计数据
        local_stats = torch.tensor([higher_neg_count, total_neg_count], dtype=torch.float32, device=chosen_logratios.device)
        gathered_stats = [torch.zeros_like(local_stats) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_stats, local_stats)
        
        # 汇总所有 GPU 的统计数据
        global_higher_neg_count = sum(stats[0].item() for stats in gathered_stats)
        global_total_neg_count = sum(stats[1].item() for stats in gathered_stats)
        
        # 保存统计结果到文件
       
               # 统计数据保存
        if dist.get_rank() == 0:  # 只在主进程保存统计摘要
            import json
            import os
            from datetime import datetime
            
            # 创建保存目录
            os.makedirs("./logratio_stats", exist_ok=True)
            
            # 获取当前时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存统计摘要到单一文件
            with open(f"./logratio_stats/summary.txt", "a") as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Total negative samples: {global_total_neg_count}\n")
                f.write(f"Negative samples with higher logratios: {global_higher_neg_count}\n")
                f.write(f"Percentage: {global_higher_neg_count/global_total_neg_count*100:.2f}%\n\n")
        
        # 每个进程将自己的数据保存到同一个文件中
        import json
        import os
        from datetime import datetime
        
        os.makedirs("./logratio_stats", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 使用文件锁确保多进程写入安全
        import fcntl
        
        with open(f"./logratio_stats/all_pairs_data.jsonl", "a") as f:
            # 获取文件锁
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for item in all_pairs_data:
                    item["timestamp"] = timestamp  # 添加时间戳以便区分不同批次
                    f.write(json.dumps(item) + "\n")
            finally:
                # 释放文件锁
                fcntl.flock(f, fcntl.LOCK_UN)
        
        temp = sum(torch.exp(beta * (rejected_logratios[key] - chosen_logratios)) for key in rejected_logratios)
        temp1 = -torch.log(temp)
        losses = -F.logsigmoid(temp1)
        return (losses, None), A, A_filtered, chosen_rewards, rejected_rewards, beta
    elif filter_mode == 'gap_visual':
        ##（把最小的loss过滤掉-> loss 低 <-> 高gap数据，不出意外：应该与过滤高gap的方法等价，猜测其beta曲线与高gap近似等价。其根据A-used和gap的差值来调整beta都不够合理，因为原始的double方法是双边过滤，过滤离群gap：低gap 高gap，所以设定该基准是合理的。
        # 尤其是过滤比例较高的情况下，整体A_used偏低，造成beta值整体异常偏大：基准——当前的gap mean以全体gap来更新是否合理？？？）
        
        losses = -F.logsigmoid(beta * (chosen_logratios - rejected_logratios['rejected1']))
        losses_global = losses#
        k = int(losses_global.size(0) * (1 - mode_weight))
        lower_bound_value = losses_global.topk(k, largest=True).values[-1]
        global_mask = losses_global >= lower_bound_value # loss较高的样本为1

        # 使用 global_mask 来选择参与计算的 A：loss较高的样本，A_filtered返回的是实际参与计算的，而不是被过滤的
        A_filtered = A[global_mask]
        A_used =  A

        gathered_A_used = [torch.zeros_like(A_used) for _ in range(4)]
        dist.all_gather(gathered_A_used, A_used)  # 收集所有 GPU 的 A_used

        # 仅主进程写入文件
        if dist.get_rank() == 0:
            with open("./pos_neg_gap_discrepancy_cu0.txt", "a") as f:
                for gpu_A_used in gathered_A_used:
                    f.write(f"{gpu_A_used.tolist()}\n")
        beta_used = beta * (1 -alpha_a * (A_used - gap_mean)+torch.where(A_used - gap_mean < 0, 1, 0))


        beta_used = torch.tensor(beta_used).to(A_used.device)
        beta_used = beta_used.detach()
        beta_used = beta_used.clamp(min=2e-1)
        beta_used = beta_used.clamp(min=beta_clamp_min,max=beta_clamp_max) if beta_clamp==True else beta_used
        return (losses,global_mask),A,A_filtered,chosen_rewards,rejected_rewards,beta_used

  
  

   # dpo loss:按照 policy/ref 的reward来筛选（虽不够合理，但也有提升）
    elif filter_mode == 'SDPO_reward_as_filter':

        #reward discrepancy 过大的样本，没有信息增量，在推荐里，也应认同。
        # 但discrepancy较小的数据，推荐这边的数据，负样本来自于真实的用户交互，似乎不应该被视为标签错误？
        # rejected_num,batch_size


        hard_negative_sample_list=[]
        batch_dpo_hard_neg_list=[]
        batch_dpo_easy_neg_list=[]
 
        # 1.dpo rejected logratios
        batch_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        # 逆序记录：样本，rejected1,2,3,4的logratio

        #遍历 rejected logratio字典:    key:rejected1/2/3/4,   value:batchsize的rejected1样本概率
        for (key,value) in rejected_logratios.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(rejected_logratios['rejected1'])):
                batch_rejected.setdefault(i, []).append(value[i])


        # 3.reference rejected logs
        batch_reference_rejected_logps={}
        for (key,value) in reference_rejected_logps.items():
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(reference_rejected_logps['rejected1'])):
                batch_reference_rejected_logps.setdefault(i, []).append(value[i])
               
        easy_neg_logratio=[]

        for num in range(len(rejected_logratios['rejected1'])):# batch_size
           
            key = num  # 直接使用 num 作为键,(batch-size,)
            
            sample_neg_list=[]
            easy_neg_list=[]

            dynamic_flag=False

            # dpo reward 作为筛选，实际添加dpo的reward项
            if dynamic_flag==True:
                # 1.存在reward大于正样本
                for idx,value in enumerate(batch_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= chosen_logratios[key]:
                        sample_neg_list.append(batch_rejected[key][idx])
                        
            
                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
                    ####原始数据准备（假设 batch_rejected [key]是reward列表）
                    rewards = np.array(batch_rejected[key])

                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0]

                    # 构建结果列表
                    sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                    
                    
                    easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]

                # 如果reward有大于正样本的，则进入easyneg的补充处理
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_rejected[key]) if val in sample_neg_list]
                    
                    # 构建easy_neg_list和simpo_policy_easy_neg_list
                    
                    
                    if len(hard_indices)==len(batch_policy_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    
                   

            else:
                rewards = np.array(batch_rejected[key])

                # 使用K-Means聚类（分为3类）
                kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                # 找到最高reward的类（按均值判断）
                cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0]

                # 构建结果列表
                sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]

            # dpo reward related
            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
     
           
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}


    

        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本
            
            temp_b = sum(torch.exp(beta * (neg_sample_logratio - chosen_logratios[i])) 
                    for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i]))
            temp1_b = -torch.log(temp_b)
            losses_b = -F.logsigmoid(temp1_b)
           
            batch_losses_list.append(losses_b)

        # 将所有样本的 loss 堆叠成一个 tensor
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]
        
        return (losses,None),A,None,chosen_rewards,rejected_rewards,(beta, beta, beta)    #torch.tensor(beta_total_record)

    ## dpo loss:按照policy筛选
    elif filter_mode == 'SDPO_policy_filter':
        # ### 按照policylogps进行筛选， 去掉原本减去ref model的值

        hard_negative_sample_list=[]
        batch_simpo_policy_hard_neg_list=[]
        batch_simpo_policy_easy_neg_list=[]
        # policy_logps_list=[]

        # 1.dpo rejected logratios
        batch_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        # 逆序记录：样本，rejected1,2,3,4的logratio

        #遍历 rejected logratio字典:    key:rejected1/2/3/4,   value:batchsize的rejected1样本概率
        for (key,value) in rejected_logratios.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(rejected_logratios['rejected1'])):
                batch_rejected.setdefault(i, []).append(value[i])

        # 2.policy rejected logps(simpo -- nll)
        
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
       
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 添加新代码：将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)

        # 3.reference rejected logs
        batch_reference_rejected_logps={}
        for (key,value) in reference_rejected_logps.items():
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(reference_rejected_logps['rejected1'])):
                batch_reference_rejected_logps.setdefault(i, []).append(value[i])
               
        # avg_logratio=[] # batchsize,avg_negsample_reward
        easy_neg_logratio=[]

        # 负样本过滤/动态选择 最粗糙方案:log ratio小于平均值的加入logratio
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
           
            key = num  # 直接使用 num 作为键,(batch-size,)
            
            sample_neg_list=[]
            easy_neg_list=[]
            simpo_policy_hard_neg_list=[]
            simpo_policy_easy_neg_list=[]
            dynamic_flag=True

            # simpo reward 作为筛选，实际添加dpo的reward项
            if dynamic_flag==True:
                # 1.存在reward大于正样本
                for idx,value in enumerate(batch_policy_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= policy_chosen_logps[key]:
                        sample_neg_list.append(batch_rejected[key][idx])
                        

                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
                    ####原始数据准备（假设batch_policy_rejected[key]是reward列表）
                    rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0]

                    # 构建结果列表
                    sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                    simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]

                # 如果reward有大于正样本的，则进入easyneg的补充处理
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_rejected[key]) if val in sample_neg_list]
                    # 构建simpo_policy_hard_neg_list
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    
                    # 构建easy_neg_list和simpo_policy_easy_neg_list
                    
                    
                    if len(hard_indices)==len(batch_policy_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    

            else:

                rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                # 将所有张量移到CPU并转换为numpy数组
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                # 使用K-Means聚类（分为3类）
                kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                # 找到最高reward的类（按均值判断）
                cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0]

                # 构建结果列表
                sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]


            # dpo reward related
            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
            # simpo reward related
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            
            
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}


        sample_level_beta_record=[]
        delta_pos2hardneg_record=[]  # 新增记录
        delta_hard_neg2easy_neg_record=[]  # 新增记录
        #[可选]:动态beta计算相关

        for i in range(chosen_logratios.size(0)):# batch遍历
         
            A=A_gap_sum
            A_used = torch.mean(A) if adjust_level == 'batch_level' else A
            #[负样本level-动态调整beta：
            delta_pos2hardneg=[]
            
            beta_record=[]
            pos2hardneg_record=[]  # 每个样本的delta_pos2hardneg记录
            hardneg2easy_record=[]  # 每个样本的delta_hard_neg2easy_neg记录
    

            # 对于dpo，这里的动态beta可以做两种尝试：1.DPO的reward来调节beta 2.SimPO的reward来调节beta
            # hard neg 当前seq 的<chosen,rej1,rej2..>loss计算
            for idx, neg_sample_logratio in enumerate(hard_negative_sample_list[i]):
              
                # instance/batch level选择的beta值不同
                beta_value=beta
                
                # # method1:（1).dpo reward 作为beta调整
                # ######### 1.正样本相对于负样本的生成优势：
                # delta_pos2hardneg=chosen_logratios[i]-neg_sample_logratio


                
                
                # ######### 2.hard neg相对于其他easy neg的均值，其生成优势:
                # delta_hard_neg2easy_neg=neg_sample_logratio-easy_neg_logratio[i]   

                
                ###### method2: (2). simpo reward作为beta调整
                delta_pos2hardneg=policy_chosen_logps[i]-batch_simpo_policy_hard_neg_list[i][idx]
                delta_hard_neg2easy_neg=batch_simpo_policy_hard_neg_list[i][idx]-batch_simpo_policy_easy_neg_list[i]
                
              
                # 记录这两个值
                pos2hardneg_record.append(float(delta_pos2hardneg.detach().cpu().item()))
                hardneg2easy_record.append(float(delta_hard_neg2easy_neg.detach().cpu().item()))
                

                
                # 计算差值
                #delta_diff = (delta_pos2hardneg -gap_mean) -( delta_hard_neg2easy_neg-delta_hard_neg2easy_neg_mean) #-2.0
                
                delta_diff = (delta_pos2hardneg - delta_pos2hardneg_mean) - (delta_hard_neg2easy_neg - delta_hard_neg2easy_neg_mean) - 6.0


                delta_beta = delta_diff /torch.sqrt(delta_pos2hardneg**2 + delta_hard_neg2easy_neg**2 + 1e-8)
                delta_beta=torch.tanh(delta_beta ) * 0.2
                
                
                #动态beta启动这行代码
                beta_value=beta_value*(1+delta_beta)

                beta_value=float(beta_value.detach().cpu().item())
                beta_record.append(beta_value)
                
            sample_level_beta_record.append(beta_record)
            delta_pos2hardneg_record.append(pos2hardneg_record)  # 添加到记录列表
            delta_hard_neg2easy_neg_record.append(hardneg2easy_record)  # 添加到记录列表


        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本
            # sample_level_beta_record[i][idx]
            
            temp_b = sum(torch.exp(beta * (neg_sample_logratio - chosen_logratios[i])) 
                    for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i]))
            # temp_b = sum(torch.exp(sample_level_beta_record[i][idx] * (neg_sample_logratio - chosen_logratios[i])) 
            #         for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i]))
            temp1_b = -torch.log(temp_b)
            losses_b = -F.logsigmoid(temp1_b)
            batch_losses_list.append(losses_b)

  
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]

        return (losses,None),batch_simpo_policy_hard_neg_list,batch_policy_rejected_list,chosen_rewards,rejected_rewards,(sample_level_beta_record, delta_pos2hardneg_record, delta_hard_neg2easy_neg_record)  #(sample_level_beta_record, delta_pos2hardneg_record, delta_hard_neg2easy_neg_record)    #torch.tensor(beta_total_record)

    elif filter_mode=='SDPO_beta_adjust_only':
        # ### 按照policylogps进行筛选，保留所有负样本但使用动态beta调整

        # 2.policy rejected logps(simpo -- nll)
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
       
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)

        # 1.dpo rejected logratios - 保留所有负样本
        batch_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        
        #遍历 rejected logratio字典:    key:rejected1/2/3/4,   value:batchsize的rejected1样本概率
        for (key,value) in rejected_logratios.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(rejected_logratios['rejected1'])):
                batch_rejected.setdefault(i, []).append(value[i])

        # 3.reference rejected logs
        batch_reference_rejected_logps={}
        for (key,value) in reference_rejected_logps.items():
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(reference_rejected_logps['rejected1'])):
                batch_reference_rejected_logps.setdefault(i, []).append(value[i])

        # 负样本分类：用于beta调整计算
        hard_negative_indices_list = []  # 记录每个样本的hard negative索引
        easy_neg_logratio = []  # 记录每个样本的easy negative平均值
        batch_simpo_policy_hard_neg_list = []
        batch_simpo_policy_easy_neg_list = []
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
            key = num  # 直接使用 num 作为键,(batch-size,)
            
            hard_indices = []
            easy_indices = []
            simpo_policy_hard_neg_list = []
            simpo_policy_easy_neg_list = []
            dynamic_flag = True

            # simpo reward 作为筛选标准，用于beta调整
            if dynamic_flag == True:
                # 1.存在reward大于正样本的情况
                for idx, value in enumerate(batch_policy_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                    #负样本的logratios大于正样本的logratios，则标记为hard-neg
                    if value >= policy_chosen_logps[key]:
                        hard_indices.append(idx)
                    else:
                        easy_indices.append(idx)

                #  2.如果没有大于正样本的，采用聚类方法划分
                if len(hard_indices) < 1:
                    ####原始数据准备（假设batch_policy_rejected[key]是reward列表）
                    rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0].tolist()
                    easy_indices = [i for i in range(len(rewards)) if i not in hard_indices]

            else:
                # 使用聚类方法
                rewards_tensors = batch_policy_rejected[key]
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0].tolist()
                easy_indices = [i for i in range(len(rewards)) if i not in hard_indices]

            # 构建hard和easy负样本列表（用于beta调整计算）
            simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
            simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in easy_indices]
            
            # 计算easy negative的平均值
            if len(easy_indices) > 0:
                easy_neg_avg = sum([batch_rejected[key][i] for i in easy_indices]) / len(easy_indices)
            else:
                # 如果没有easy negative，使用所有negative的平均值
                easy_neg_avg = sum(batch_rejected[key]) / len(batch_rejected[key])
            
            # 记录结果
            hard_negative_indices_list.append(hard_indices)
            easy_neg_logratio.append(easy_neg_avg)
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list) if len(simpo_policy_easy_neg_list) > 0 else 0)

        # 动态beta计算
        sample_level_beta_record = []
        delta_pos2hardneg_record = []
        delta_hard_neg2easy_neg_record = []

        for i in range(chosen_logratios.size(0)):# batch遍历
            beta_record = []
            pos2hardneg_record = []
            hardneg2easy_record = []

            # 为每个负样本计算beta值
            for idx in range(len(batch_rejected[i])):  # 遍历所有负样本
                # 判断当前负样本是否为hard negative
                if idx in hard_negative_indices_list[i]:
                    # 对于hard negative，进行beta调整
                    beta_value = beta
                    neg_sample_logratio = batch_rejected[i][idx]
                    
                    # method1: dpo reward 作为beta调整
                    delta_pos2hardneg = chosen_logratios[i] - neg_sample_logratio
                    delta_hard_neg2easy_neg = neg_sample_logratio - easy_neg_logratio[i]
                    
                    # method2: simpo reward作为beta调整（可选）
                    # hard_neg_idx_in_list = hard_negative_indices_list[i].index(idx)
                    # delta_pos2hardneg = policy_chosen_logps[i] - batch_simpo_policy_hard_neg_list[i][hard_neg_idx_in_list]
                    # delta_hard_neg2easy_neg = batch_simpo_policy_hard_neg_list[i][hard_neg_idx_in_list] - batch_simpo_policy_easy_neg_list[i]
                    
                    # 记录这两个值
                    pos2hardneg_record.append(float(delta_pos2hardneg.detach().cpu().item()))
                    hardneg2easy_record.append(float(delta_hard_neg2easy_neg.detach().cpu().item()))
                    
                    # 计算差值并调整beta
                    delta_diff = delta_pos2hardneg - delta_hard_neg2easy_neg
                    delta_beta = delta_diff / (abs(delta_hard_neg2easy_neg) + abs(delta_pos2hardneg))
                    delta_beta = torch.tanh(delta_beta) * 0.5
                    
                    # 动态beta调整
                    beta_value = beta_value * (1 + delta_beta)
                    beta_value = float(beta_value.detach().cpu().item())
                else:
                    # 对于easy negative，使用原始beta值
                    beta_value = beta  # 直接使用原始beta，不进行任何调整
                    pos2hardneg_record.append(0.0)  # 占位符
                    hardneg2easy_record.append(0.0)  # 占位符
                
                beta_record.append(beta_value)
                
            sample_level_beta_record.append(beta_record)
            delta_pos2hardneg_record.append(pos2hardneg_record)
            delta_hard_neg2easy_neg_record.append(hardneg2easy_record)

        # 计算所有负样本的loss，使用对应的动态beta
        batch_losses_list = []

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本
            # 计算与所有负样本的loss
            temp_b = sum(torch.exp(sample_level_beta_record[i][idx] * (batch_rejected[i][idx] - chosen_logratios[i])) 
                    for idx in range(len(batch_rejected[i])))  # 使用所有负样本
            temp1_b = -torch.log(temp_b)
            losses_b = -F.logsigmoid(temp1_b)
            batch_losses_list.append(losses_b)

        losses = torch.stack(batch_losses_list)  # shape: [batch_size]

        return (losses,None),batch_simpo_policy_hard_neg_list,batch_policy_rejected_list,chosen_rewards,rejected_rewards,(sample_level_beta_record, delta_pos2hardneg_record, delta_hard_neg2easy_neg_record)

    # 仅限simpo ，无ref model
    elif filter_mode== 'mppo_multineg_loss':
        N=len(rejected_logratios)
        losses=-F.logsigmoid(beta*N*torch.exp(chosen_logratios)-sum(beta*torch.exp(rejected_logratios[key]) for key in rejected_logratios))
        return (losses,None),A,A,chosen_rewards,rejected_rewards,(beta,beta,beta)
    
    elif filter_mode== 'mppo_hard_negative_no_dynamic_beta':
           
        ### 按照 policylogps 进行筛选， 去掉原本减去 ref model的值
        # rejected_num,batch_size

      
      
        # 2.policy rejected logps(simpo -- nll)
        
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        batch_simpo_policy_hard_neg_list=[]
        batch_simpo_policy_easy_neg_list=[]
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 添加新代码：将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)

        # 3.reference rejected logs
        
        
        hard_negative_sample_list=[]
        

        easy_neg_logratio=[]

        # 负样本过滤/动态选择 最粗糙方案:log ratio小于平均值的加入logratio
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
        
            key = num  # 直接使用 num 作为键
            values = batch_policy_rejected[key]  # 某条序列的数据，包含正负样本
            sample_neg_list=[]
            easy_neg_list=[]
            dynamic_flag=False

            if dynamic_flag==True:
                #  1.存在reward大于正样本
                for idx,value in enumerate(batch_policy_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= chosen_logratios[key]:
                        sample_neg_list.append(batch_policy_rejected[key][idx])
                        

                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
                    ####原始数据准备（假设batch_policy_rejected[key]是reward列表）
                    rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))  # init='k-means++',
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0]

                    # 构建结果列表
                    sample_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                    simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                # 如果reward有大于正样本的，则进入easyneg的补充处理
                
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_policy_rejected[key]) if val in sample_neg_list]
                    # 构建simpo_policy_hard_neg_list
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    
                    # 需要添加 如果所有负样本reward都大于正样本，该怎么处理的逻辑。
                    if len(hard_indices)==len(batch_policy_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    

            else:

                rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                # 将所有张量移到CPU并转换为numpy数组
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                # 使用K-Means聚类（分为3类）
                kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                # 找到最高reward的类（按均值判断）
                cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0]

                # 构建结果列表
                sample_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
            # print(chosen_logratios[key])
            # print('easy',easy_neg_list)
            # print('hard',simpo_policy_hard_neg_list)

            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
            # simpo reward related
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            
            
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}


        sample_level_beta_record=[]
        delta_pos2hardneg_record=[]  # 新增记录
        delta_hard_neg2easy_neg_record=[]  # 新增记录

        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本

        #    # fixed beta
            N=len(hard_negative_sample_list[i])
            losses_b=-F.logsigmoid(beta*N*torch.exp(chosen_logratios)-sum(beta*torch.exp(neg_sample_logratio) for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i])))
          
        
            batch_losses_list.append(losses_b)

        # 将所有样本的 loss 堆叠成一个 tensor
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]


        return (losses,None),batch_simpo_policy_hard_neg_list,batch_policy_rejected_list,chosen_rewards,rejected_rewards,(sample_level_beta_record, delta_pos2hardneg_record, delta_hard_neg2easy_neg_record)  #(sample_level_beta_record, delta_pos2hardneg_record, delta_hard_neg2easy_neg_record)    #torch.tensor(beta_total_record)

    elif filter_mode== 'mppo_hard_negative_dynamic_beta_dynamic_margin':
           
        ### 按照 policylogps 进行筛选， 去掉原本减去 ref model的值
        # rejected_num,batch_size

      
      
        # 2.policy rejected logps(simpo -- nll)
        
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        batch_simpo_policy_hard_neg_list=[]
        batch_simpo_policy_easy_neg_list=[]
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 添加新代码：将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)

        # 3.reference rejected logs
        
        
        hard_negative_sample_list=[]
        

        easy_neg_logratio=[]

        # 负样本过滤/动态选择 最粗糙方案:log ratio小于平均值的加入logratio
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
        
            key = num  # 直接使用 num 作为键
            #values = batch_policy_rejected[key]  # 某条序列的数据，包含正负样本
            sample_neg_list=[]
            easy_neg_list=[]
            dynamic_flag=True

            if dynamic_flag==True:
                #  1.存在reward大于正样本
                for idx,value in enumerate(batch_policy_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= chosen_logratios[key]:
                        sample_neg_list.append(batch_policy_rejected[key][idx])
                        

                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
                    ####原始数据准备（假设batch_policy_rejected[key]是reward列表）
                    rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))  # init='k-means++',
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0]

                    # 构建结果列表
                    sample_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                    simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                # 如果reward有大于正样本的，则进入easyneg的补充处理
                
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_policy_rejected[key]) if val in sample_neg_list]
                    # 构建simpo_policy_hard_neg_list
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    
                    # 需要添加 如果所有负样本reward都大于正样本，该怎么处理的逻辑。
                    if len(hard_indices)==len(batch_policy_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    

            else:

                rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                # 将所有张量移到CPU并转换为numpy数组
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                # 使用K-Means聚类（分为3类）
                kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                # 找到最高reward的类（按均值判断）
                cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0]

                # 构建结果列表
                sample_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
            # print(chosen_logratios[key])
            # print('easy',easy_neg_list)
            # print('hard',simpo_policy_hard_neg_list)

            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
            # simpo reward related
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            
            
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}

        #:动态beta计算相关
        
        sample_level_beta_record=[]
        delta_pos2hardneg_record=[]  # 新增记录
        delta_hard_neg2easy_neg_record=[]  # 新增记录
        margin_record=[]
        
       
        for i in range(chosen_logratios.size(0)):# batch遍历
         
            
            
            #[负样本level-动态调整beta：
            delta_pos2hardneg=[]
            
            beta_record=[]
            pos2hardneg_record=[]  # 每个样本的delta_pos2hardneg记录
            hardneg2easy_record=[]  # 每个样本的delta_hard_neg2easy_neg记录
    

            # 对于dpo，这里的动态beta可以做两种尝试：1.DPO的reward来调节beta 2.SimPO的reward来调节beta
            # hard neg 当前seq 的<chosen,rej1,rej2..>loss计算

            margin_list=[]
            for idx, neg_sample_logratio in enumerate(hard_negative_sample_list[i]):
              
                
                
                #######   1.正样本相对于负样本的生成优势：
                delta_pos2hardneg=chosen_logratios[i]-neg_sample_logratio
                ######### 2.hard neg相对于其他easy neg的均值，其生成优势:
                delta_hard_neg2easy_neg=neg_sample_logratio-easy_neg_logratio[i]   #delta_hard_neg2easy_neg=neg_sample_logratio-easy_neg_logratio[idx]
      
                margin = (delta_pos2hardneg-delta_hard_neg2easy_neg).detach()
                margin_list.append(margin)

            margin=torch.mean(torch.stack(margin_list))
            margin_record.append(margin)

            for idx, neg_sample_logratio in enumerate(hard_negative_sample_list[i]):
              
                # instance/batch level选择的beta值不同
                beta_value=beta
                
                # # method1:（1).dpo reward 作为beta调整
                # ######### 1.正样本相对于负样本的生成优势：
                # delta_pos2hardneg=chosen_logratios[i]-neg_sample_logratio


                
                
                # ######### 2.hard neg相对于其他easy neg的均值，其生成优势:
                # delta_hard_neg2easy_neg=neg_sample_logratio-easy_neg_logratio[i]   

                
                ###### method2: (2). simpo reward作为beta调整
                delta_pos2hardneg=policy_chosen_logps[i]-batch_simpo_policy_hard_neg_list[i][idx]
                delta_hard_neg2easy_neg=batch_simpo_policy_hard_neg_list[i][idx]-batch_simpo_policy_easy_neg_list[i]
                
              
                # 记录这两个值
                pos2hardneg_record.append(float(delta_pos2hardneg.detach().cpu().item()))
                hardneg2easy_record.append(float(delta_hard_neg2easy_neg.detach().cpu().item()))
                

                
                # 计算差值
                delta_diff = delta_pos2hardneg - delta_hard_neg2easy_neg-margin
              
                
                delta_beta = delta_diff /(abs(delta_hard_neg2easy_neg)+abs(delta_pos2hardneg))
                delta_beta=torch.tanh(delta_beta ) * 0.5
                
                
                #动态beta启动这行代码
                beta_value=beta_value*(1+delta_beta)

                beta_value=float(beta_value.detach().cpu().item())
                beta_record.append(beta_value)
                
            sample_level_beta_record.append(beta_record)
            delta_pos2hardneg_record.append(pos2hardneg_record)  # 添加到记录列表
            delta_hard_neg2easy_neg_record.append(hardneg2easy_record)  # 添加到记录列表


        
        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本

        #    # fixed beta
        #     N=len(hard_negative_sample_list[i])
        #     losses_b=-F.logsigmoid(beta*N*torch.exp(chosen_logratios)-sum(beta*torch.exp(neg_sample_logratio) for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i])))
            
            # dynamic adjust beta
            losses_b=-F.logsigmoid(sum(sample_level_beta_record[i][idx]*torch.exp(chosen_logratios)-sample_level_beta_record[i][idx]*torch.exp(neg_sample_logratio) for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i])))
        
            batch_losses_list.append(losses_b)

        # 将所有样本的 loss 堆叠成一个 tensor
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]


        return (losses,margin_record),batch_simpo_policy_hard_neg_list,batch_policy_rejected_list,chosen_rewards,rejected_rewards,(sample_level_beta_record, delta_pos2hardneg_record, delta_hard_neg2easy_neg_record)  #(sample_level_beta_record, delta_pos2hardneg_record, delta_hard_neg2easy_neg_record)    #torch.tensor(beta_total_record)

    elif filter_mode== 'mppo_hard_negative_dynamic_beta_fixed_margin':
           
        ### 按照 policylogps 进行筛选， 去掉原本减去 ref model的值
        # rejected_num,batch_size

      
      
        # 2.policy rejected logps(simpo -- nll)
        
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        batch_simpo_policy_hard_neg_list=[]
        batch_simpo_policy_easy_neg_list=[]
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 添加新代码：将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)

        # 3.reference rejected logs
        
        
        hard_negative_sample_list=[]
        

        easy_neg_logratio=[]

        # 负样本过滤/动态选择 最粗糙方案:log ratio小于平均值的加入logratio
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
        
            key = num  # 直接使用 num 作为键
            values = batch_policy_rejected[key]  # 某条序列的数据，包含正负样本
            sample_neg_list=[]
            easy_neg_list=[]
            dynamic_flag=True

            if dynamic_flag==True:
                #  1.存在reward大于正样本
                for idx,value in enumerate(batch_policy_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= chosen_logratios[key]:
                        sample_neg_list.append(batch_policy_rejected[key][idx])
                        

                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
                    ####原始数据准备（假设batch_policy_rejected[key]是reward列表）
                    rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))  # init='k-means++',
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0]

                    # 从 batch_policy_rejected 字典中根据 key 和 hard_indices 索引构建负样本列表
                    # 遍历 hard_indices 中的索引 i,获取对应的负样本,组成新的列表
                    sample_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                    simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                # 如果reward有大于正样本的，则进入easyneg的补充处理
                
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_policy_rejected[key]) if val in sample_neg_list]
                    # 构建simpo_policy_hard_neg_list
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    
                    # 需要添加 如果所有负样本reward都大于正样本，该怎么处理的逻辑。
                    if len(hard_indices)==len(batch_policy_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    

            else:

                rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                # 将所有张量移到CPU并转换为numpy数组
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                # 使用K-Means聚类（分为3类）
                kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                # 找到最高reward的类（按均值判断）
                cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0]

                # 构建结果列表
                sample_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
            # print(chosen_logratios[key])
            # print('easy',easy_neg_list)
            # print('hard',simpo_policy_hard_neg_list)

            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
            # simpo reward related
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            
            
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}


        sample_level_beta_record=[]
        delta_pos2hardneg_record=[]  # 新增记录
        delta_hard_neg2easy_neg_record=[]  # 新增记录
        #[可选]:动态beta计算相关

        for i in range(chosen_logratios.size(0)):# batch遍历
         
            
            
            #[负样本level-动态调整beta：
            delta_pos2hardneg=[]
            
            beta_record=[]
            pos2hardneg_record=[]  # 每个样本的delta_pos2hardneg记录
            hardneg2easy_record=[]  # 每个样本的delta_hard_neg2easy_neg记录
    

            # 对于dpo，这里的动态beta可以做两种尝试：1.DPO的reward来调节beta 2.SimPO的reward来调节beta
            # hard neg 当前seq 的<chosen,rej1,rej2..>loss计算
            for idx, neg_sample_logratio in enumerate(hard_negative_sample_list[i]):
              
                # instance/batch level选择的beta值不同
                beta_value=beta
                
                # # method1:（1).dpo reward 作为beta调整
                # ######### 1.正样本相对于负样本的生成优势：
                # delta_pos2hardneg=chosen_logratios[i]-neg_sample_logratio


                
                
                # ######### 2.hard neg相对于其他easy neg的均值，其生成优势:
                # delta_hard_neg2easy_neg=neg_sample_logratio-easy_neg_logratio[i]   

                
                ###### method2: (2). simpo reward作为beta调整
                delta_pos2hardneg=policy_chosen_logps[i]-batch_simpo_policy_hard_neg_list[i][idx]
                delta_hard_neg2easy_neg=batch_simpo_policy_hard_neg_list[i][idx]-batch_simpo_policy_easy_neg_list[i]
                
              
                # 记录这两个值
                pos2hardneg_record.append(float(delta_pos2hardneg.detach().cpu().item()))
                hardneg2easy_record.append(float(delta_hard_neg2easy_neg.detach().cpu().item()))
                

                
                # 计算差值
                delta_diff = delta_pos2hardneg - delta_hard_neg2easy_neg -2.0
              
                
                delta_beta = delta_diff /(abs(delta_hard_neg2easy_neg)+abs(delta_pos2hardneg))
                delta_beta=torch.tanh(delta_beta ) * 0.5
                
                #delta_beta = torch.sign(delta_beta) * torch.log(1 + torch.abs(delta_beta)) * 0.2

                #动态beta启动这行代码
                beta_value=beta_value*(1+delta_beta)

                beta_value=float(beta_value.detach().cpu().item())
                beta_record.append(beta_value)
                
            sample_level_beta_record.append(beta_record)
            delta_pos2hardneg_record.append(pos2hardneg_record)  # 添加到记录列表
            delta_hard_neg2easy_neg_record.append(hardneg2easy_record)  # 添加到记录列表


        
        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本

        #    # fixed beta
        #     N=len(hard_negative_sample_list[i])
        #     losses_b=-F.logsigmoid(beta*N*torch.exp(chosen_logratios)-sum(beta*torch.exp(neg_sample_logratio) for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i])))
            
            # dynamic adjust beta
            losses_b=-F.logsigmoid(sum(sample_level_beta_record[i][idx]*torch.exp(chosen_logratios)-sample_level_beta_record[i][idx]*torch.exp(neg_sample_logratio) for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i])))
        
            batch_losses_list.append(losses_b)

        # 将所有样本的 loss 堆叠成一个 tensor
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]


        return (losses,None),batch_simpo_policy_hard_neg_list,batch_policy_rejected_list,chosen_rewards,rejected_rewards,(sample_level_beta_record, delta_pos2hardneg_record, delta_hard_neg2easy_neg_record)  #(sample_level_beta_record, delta_pos2hardneg_record, delta_hard_neg2easy_neg_record)    #torch.tensor(beta_total_record)


     

    elif filter_mode== 'DMPO':
        K=len(rejected_logratios.keys())
        losses=-F.logsigmoid(beta*chosen_logratios-sum(beta*rejected_logratios[key] for key in rejected_logratios)/K)
        return (losses,None),A,A,chosen_rewards,rejected_rewards,(beta,beta,beta)
    
    
    elif filter_mode=="DMPO_hard_negative_cluster_3":
            
        
        
        
        
        hard_negative_sample_list=[]
        batch_simpo_policy_hard_neg_list=[]
        batch_simpo_policy_easy_neg_list=[]
        # policy_logps_list=[]

        # 1.dpo rejected logratios
        batch_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        # 逆序记录：样本，rejected1,2,3,4的logratio

        #遍历 rejected logratio字典:    key:rejected1/2/3/4,   value:batchsize的rejected1样本概率
        for (key,value) in rejected_logratios.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(rejected_logratios['rejected1'])):
                batch_rejected.setdefault(i, []).append(value[i])

        # 2.policy rejected logps(simpo -- nll)
        
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
       
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)


        # 3.reference rejected logs
        batch_reference_rejected_logps={}
        for (key,value) in reference_rejected_logps.items():
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(reference_rejected_logps['rejected1'])):
                batch_reference_rejected_logps.setdefault(i, []).append(value[i])
               
        # avg_logratio=[] # batchsize,avg_negsample_reward
        easy_neg_logratio=[]

        # 负样本过滤/动态选择 最粗糙方案:log ratio小于平均值的加入logratio
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
           
            key = num  # 直接使用 num 作为键,(batch-size,)
            
            sample_neg_list=[]
            easy_neg_list=[]
            simpo_policy_hard_neg_list=[]
            simpo_policy_easy_neg_list=[]
            dynamic_flag=False

            # simpo reward 作为筛选，实际添加dpo的reward项
            if dynamic_flag==True:
                # 1.存在reward大于正样本
                for idx,value in enumerate(batch_policy_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= policy_chosen_logps[key]:
                        sample_neg_list.append(batch_rejected[key][idx])
                        
            
                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
                    ####原始数据准备（假设batch_policy_rejected[key]是reward列表）
                    rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0]

                    # 构建结果列表
                    sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                    simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                # 如果reward有大于正样本的，则进入easyneg的补充处理
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_rejected[key]) if val in sample_neg_list]
                    # 构建simpo_policy_hard_neg_list
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    
                   # 构建easy_neg_list和simpo_policy_easy_neg_list
                    
                    
                    if len(hard_indices)==len(batch_policy_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    

            else:
                rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                # 使用K-Means聚类（分为3类）
                kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                # 找到最高reward的类（按均值判断）
                cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0]

                # 构建结果列表
                sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
            # dpo reward related
            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
            # simpo reward related
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            
            # simpo reward related
            #batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            #batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}

      
      

        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本
           
            
            
            K=len(hard_negative_sample_list[i])
            DMPO_loss_item=-F.logsigmoid(beta*chosen_logratios-sum(beta*neg_sample_logratio for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i]) )/K)
            batch_losses_list.append(DMPO_loss_item)

        # 将所有样本的 loss 堆叠成一个 tensor
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]

        
        return (losses,None),A,A,chosen_rewards,rejected_rewards,(beta,beta,beta)

    elif filter_mode=="DMPO_hard_negative_dynamic_beta_fixed_margin":
            
   
        hard_negative_sample_list=[]
        batch_simpo_policy_hard_neg_list=[]
        batch_simpo_policy_easy_neg_list=[]
        # policy_logps_list=[]

        # 1.dpo rejected logratios
        batch_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        # 逆序记录：样本，rejected1,2,3,4的logratio

        #遍历 rejected logratio字典:    key:rejected1/2/3/4,   value:batchsize的rejected1样本概率
        for (key,value) in rejected_logratios.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(rejected_logratios['rejected1'])):
                batch_rejected.setdefault(i, []).append(value[i])

        # 2.policy rejected logps(simpo -- nll)
        
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
       
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)


        # 3.reference rejected logs
        batch_reference_rejected_logps={}
        for (key,value) in reference_rejected_logps.items():
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(reference_rejected_logps['rejected1'])):
                batch_reference_rejected_logps.setdefault(i, []).append(value[i])
               
        # avg_logratio=[] # batchsize,avg_negsample_reward
        easy_neg_logratio=[]

        # 负样本过滤/动态选择 最粗糙方案:log ratio小于平均值的加入logratio
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
           
            key = num  # 直接使用 num 作为键,(batch-size,)
            
            sample_neg_list=[]
            easy_neg_list=[]
            simpo_policy_hard_neg_list=[]
            simpo_policy_easy_neg_list=[]
            dynamic_flag=True

            # simpo reward 作为筛选，实际添加dpo的reward项
            if dynamic_flag==True:
                # 1.存在reward大于正样本
                for idx,value in enumerate(batch_policy_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= policy_chosen_logps[key]:###chosen_logratios[key]:
                        sample_neg_list.append(batch_rejected[key][idx])
                        
            
                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
                    ####原始数据准备（假设batch_policy_rejected[key]是reward列表）
                    rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0]

                    # 构建结果列表
                    sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                    simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    
                # 如果reward有大于正样本的，则进入easyneg的补充处理
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_rejected[key]) if val in sample_neg_list]
                    # 构建simpo_policy_hard_neg_list
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    # 构建 easy_neg_list 和 simpo_policy_easy_neg_list
                    
                    
                    if len(hard_indices)==len(batch_policy_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    

            else:
                rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                # 使用K-Means聚类（分为3类）
                kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                # 找到最高reward的类（按均值判断）
                cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0]

                # 构建结果列表
                sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    
            # dpo reward related
            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
            # simpo reward related
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            
            # simpo reward related
            #batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            #batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}
        
        # 动态beta
        sample_level_beta_record=[]
        delta_pos2hardneg_record=[]  # 新增记录
        delta_hard_neg2easy_neg_record=[]  # 新增记录

        for i in range(chosen_logratios.size(0)):# batch遍历
         
        
            #[负样本level-动态调整beta：
            delta_pos2hardneg=[]
            
            beta_record=[]
            pos2hardneg_record=[]  # 每个样本的delta_pos2hardneg记录
            hardneg2easy_record=[]  # 每个样本的delta_hard_neg2easy_neg记录
    

            # 对于dpo，这里的动态beta可以做两种尝试：1.DPO的reward来调节beta 2.SimPO的reward来调节beta
            # hard neg 当前seq 的<chosen,rej1,rej2..>loss计算
            for idx, neg_sample_logratio in enumerate(hard_negative_sample_list[i]):
              
                # instance/batch level选择的beta值不同
                beta_value=beta
                
                # # method1:（1).dpo reward 作为beta调整
                # ######### 1.正样本相对于负样本的生成优势：
                # delta_pos2hardneg=chosen_logratios[i]-neg_sample_logratio


                
                
                # ######### 2.hard neg相对于其他easy neg的均值，其生成优势:
                # delta_hard_neg2easy_neg=neg_sample_logratio-easy_neg_logratio[i]   

                
                ###### method2: (2). simpo reward作为beta调整
                delta_pos2hardneg=policy_chosen_logps[i]-batch_simpo_policy_hard_neg_list[i][idx]
                delta_hard_neg2easy_neg=batch_simpo_policy_hard_neg_list[i][idx]-batch_simpo_policy_easy_neg_list[i]
                
              
                # 记录这两个值
                pos2hardneg_record.append(float(delta_pos2hardneg.detach().cpu().item()))
                hardneg2easy_record.append(float(delta_hard_neg2easy_neg.detach().cpu().item()))
                
                # 计算差值
                delta_diff = delta_pos2hardneg - delta_hard_neg2easy_neg -6.0
              
                delta_beta = delta_diff /(abs(delta_hard_neg2easy_neg)+abs(delta_pos2hardneg))
                delta_beta=torch.tanh(delta_beta ) * 0.5
           
                
                #动态beta启动这行代码
                beta_value=beta_value*(1+delta_beta)

                beta_value=float(beta_value.detach().cpu().item())
                beta_record.append(beta_value)
                
            sample_level_beta_record.append(beta_record)
            delta_pos2hardneg_record.append(pos2hardneg_record)  # 添加到记录列表
            delta_hard_neg2easy_neg_record.append(hardneg2easy_record)  # 添加到记录列表

      
      

        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本
           
            K=len(hard_negative_sample_list[i])
            DMPO_loss_item=-F.logsigmoid( sum(sample_level_beta_record[i][idx]*chosen_logratios[i]/K - sample_level_beta_record[i][idx]*neg_sample_logratio/K for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i]) ) )
            
            batch_losses_list.append(DMPO_loss_item)

        # 将所有样本的 loss 堆叠成一个 tensor
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]

        return (losses,None),batch_simpo_policy_hard_neg_list,batch_policy_rejected_list,chosen_rewards,rejected_rewards,(sample_level_beta_record,delta_pos2hardneg_record,delta_hard_neg2easy_neg_record)

    elif filter_mode=="DMPO_topk_hard_negative_dynamic_beta_fixed_margin":
            
   
        hard_negative_sample_list=[]
        batch_simpo_policy_hard_neg_list=[]
        batch_simpo_policy_easy_neg_list=[]
        # policy_logps_list=[]

        # 1.dpo rejected logratios
        batch_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        # 逆序记录：样本，rejected1,2,3,4的logratio

        #遍历 rejected logratio字典:    key:rejected1/2/3/4,   value:batchsize的rejected1样本概率
        for (key,value) in rejected_logratios.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(rejected_logratios['rejected1'])):
                batch_rejected.setdefault(i, []).append(value[i])

        # 2.policy rejected logps(simpo -- nll)
        
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
       
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)


        # 3.reference rejected logs
        batch_reference_rejected_logps={}
        for (key,value) in reference_rejected_logps.items():
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(reference_rejected_logps['rejected1'])):
                batch_reference_rejected_logps.setdefault(i, []).append(value[i])
               
        # avg_logratio=[] # batchsize,avg_negsample_reward
        easy_neg_logratio=[]

        # 负样本过滤/动态选择 最粗糙方案:log ratio小于平均值的加入logratio
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
           
            key = num  # 直接使用 num 作为键,(batch-size,)
            
            sample_neg_list=[]
            easy_neg_list=[]
            simpo_policy_hard_neg_list=[]
            simpo_policy_easy_neg_list=[]
            dynamic_flag=False

            # simpo reward 作为筛选，实际添加dpo的reward项
            if dynamic_flag==True:
                # 1.存在reward大于正样本
                for idx,value in enumerate(batch_policy_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= policy_chosen_logps[key]:###chosen_logratios[key]:
                        sample_neg_list.append(batch_rejected[key][idx])
                        
            
                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
            ####原始数据准备
                    rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                    # 使用topk选择最高的k个样本作为hard negative
                    k = 4 #min(3, len(rewards))  # 选择top3，如果样本数不足3则选择全部
                    top_k_indices = np.argsort(rewards)[-k:]  # 获取最高k个reward的索引
                    
                    # 构建结果列表
                    sample_neg_list = [batch_rejected[key][i] for i in top_k_indices]
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in top_k_indices]
                    easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in top_k_indices]
                    simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in top_k_indices]
                    
                # 如果reward有大于正样本的，则进入easyneg的补充处理
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_rejected[key]) if val in sample_neg_list]
                    # 构建simpo_policy_hard_neg_list
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    # 构建 easy_neg_list 和 simpo_policy_easy_neg_list
                    
                    
                    if len(hard_indices)==len(batch_policy_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        

            else:
                rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                # 将所有张量移到CPU并转换为numpy数组
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])

                # 使用topk选择最高的k个样本作为hard negative
                k = 4#min(3, len(rewards))  # 选择top3，如果样本数不足3则选择全部
                top_k_indices = np.argsort(rewards)[-k:]  # 获取最高k个reward的索引

                # 构建结果列表
                sample_neg_list = [batch_rejected[key][i] for i in top_k_indices]
                simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in top_k_indices]
                easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in top_k_indices]
                simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in top_k_indices]
                    
            # dpo reward related
            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
            # simpo reward related
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            
            # simpo reward related
            #batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            #batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}
        
        # 动态beta
        sample_level_beta_record=[]
        delta_pos2hardneg_record=[]  # 新增记录
        delta_hard_neg2easy_neg_record=[]  # 新增记录

        for i in range(chosen_logratios.size(0)):# batch遍历
         
        
            #[负样本level-动态调整beta：
            delta_pos2hardneg=[]
            
            beta_record=[]
            pos2hardneg_record=[]  # 每个样本的delta_pos2hardneg记录
            hardneg2easy_record=[]  # 每个样本的delta_hard_neg2easy_neg记录
    

            # 对于dpo，这里的动态beta可以做两种尝试：1.DPO的reward来调节beta 2.SimPO的reward来调节beta
            # hard neg 当前seq 的<chosen,rej1,rej2..>loss计算
            for idx, neg_sample_logratio in enumerate(hard_negative_sample_list[i]):
              
                # instance/batch level选择的beta值不同
                beta_value=beta
                
                # # method1:（1).dpo reward 作为beta调整
                # ######### 1.正样本相对于负样本的生成优势：
                # delta_pos2hardneg=chosen_logratios[i]-neg_sample_logratio


                
                
                # ######### 2.hard neg相对于其他easy neg的均值，其生成优势:
                # delta_hard_neg2easy_neg=neg_sample_logratio-easy_neg_logratio[i]   

                
                ###### method2: (2). simpo reward作为beta调整
                delta_pos2hardneg=policy_chosen_logps[i]-batch_simpo_policy_hard_neg_list[i][idx]
                delta_hard_neg2easy_neg=batch_simpo_policy_hard_neg_list[i][idx]-batch_simpo_policy_easy_neg_list[i]
                
              
                # 记录这两个值
                pos2hardneg_record.append(float(delta_pos2hardneg.detach().cpu().item()))
                hardneg2easy_record.append(float(delta_hard_neg2easy_neg.detach().cpu().item()))
                

                
                # 计算差值
                delta_diff = delta_pos2hardneg - delta_hard_neg2easy_neg -2.0
              
                
                delta_beta = delta_diff /(abs(delta_hard_neg2easy_neg)+abs(delta_pos2hardneg))
                delta_beta=torch.tanh(delta_beta ) * 0.5
                #delta_beta = torch.sign(delta_beta) * torch.log(1 + torch.abs(delta_beta)) * 0.2

                
                #动态beta启动这行代码
                beta_value=beta_value*(1+delta_beta)

                beta_value=float(beta_value.detach().cpu().item())
                beta_record.append(beta_value)
                
            sample_level_beta_record.append(beta_record)
            delta_pos2hardneg_record.append(pos2hardneg_record)  # 添加到记录列表
            delta_hard_neg2easy_neg_record.append(hardneg2easy_record)  # 添加到记录列表

      
      

        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本
           
            K=len(hard_negative_sample_list[i])
            DMPO_loss_item=-F.logsigmoid( sum(sample_level_beta_record[i][idx]*chosen_logratios/K - sample_level_beta_record[i][idx]*neg_sample_logratio/K for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i]) ) )
            
            batch_losses_list.append(DMPO_loss_item)

        # 将所有样本的 loss 堆叠成一个 tensor
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]

        return (losses,None),batch_simpo_policy_hard_neg_list,batch_policy_rejected_list,chosen_rewards,rejected_rewards,(sample_level_beta_record,delta_pos2hardneg_record,delta_hard_neg2easy_neg_record)

    elif filter_mode=="DMPO_hard_negative_dynamic_beta_dynamic_margin":
            
        
        # mc_po_loss=-F.logsigmoid(beta*chosen_logratios-torch.log(sum(torch.exp(beta*rejected_logratios[key]) for  key in rejected_logratios)))
        
        # losses=mc_po_loss
        
        hard_negative_sample_list=[]
        batch_simpo_policy_hard_neg_list=[]
        batch_simpo_policy_easy_neg_list=[]
        # policy_logps_list=[]

        # 1.dpo rejected logratios
        batch_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        # 逆序记录：样本，rejected1,2,3,4的logratio

        #遍历 rejected logratio字典:    key:rejected1/2/3/4,   value:batchsize的rejected1样本概率
        for (key,value) in rejected_logratios.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(rejected_logratios['rejected1'])):
                batch_rejected.setdefault(i, []).append(value[i])

        # 2.policy rejected logps(simpo -- nll)
        
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
       
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)


        # 3.reference rejected logs
        batch_reference_rejected_logps={}
        for (key,value) in reference_rejected_logps.items():
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(reference_rejected_logps['rejected1'])):
                batch_reference_rejected_logps.setdefault(i, []).append(value[i])
               
        # avg_logratio=[] # batchsize,avg_negsample_reward
        easy_neg_logratio=[]

        # 负样本过滤/动态选择 最粗糙方案:log ratio小于平均值的加入logratio
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
           
            key = num  # 直接使用 num 作为键,(batch-size,)
            
            sample_neg_list=[]
            easy_neg_list=[]
            simpo_policy_hard_neg_list=[]
            simpo_policy_easy_neg_list=[]
            dynamic_flag=True

            # simpo reward 作为筛选，实际添加dpo的reward项
            if dynamic_flag==True:
                # 1.存在reward大于正样本
                for idx,value in enumerate(batch_policy_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= policy_chosen_logps[key]:#########chosen_logratios[key]:
                        sample_neg_list.append(batch_rejected[key][idx])
                        
            
                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
                    ####原始数据准备（假设batch_policy_rejected[key]是reward列表）
                    rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0]

                    # 构建结果列表
                    sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                    simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    
                # 如果reward有大于正样本的，则进入easyneg的补充处理
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_rejected[key]) if val in sample_neg_list]
                    # 构建simpo_policy_hard_neg_list
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    # 构建 easy_neg_list 和 simpo_policy_easy_neg_list
                    
                    
                    if len(hard_indices)==len(batch_policy_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    

            else:
                rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                # 使用K-Means聚类（分为3类）
                kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                # 找到最高reward的类（按均值判断）
                cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0]

                # 构建结果列表
                sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]

            # dpo reward related
            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
            # simpo reward related
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            
            # simpo reward related
            #batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            #batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}
        
        # 动态beta
        sample_level_beta_record=[]
        delta_pos2hardneg_record=[]  # 新增记录
        delta_hard_neg2easy_neg_record=[]  # 新增记录
        margin_record=[]
        for i in range(chosen_logratios.size(0)):# batch遍历
         
        
            #[负样本level-动态调整beta：
            delta_pos2hardneg=[]
            
            beta_record=[]
            pos2hardneg_record=[]  # 每个样本的delta_pos2hardneg记录
            hardneg2easy_record=[]  # 每个样本的delta_hard_neg2easy_neg记录
            margin_list=[]
            for idx, neg_sample_logratio in enumerate(hard_negative_sample_list[i]):
              
                
                
                #######   1.正样本相对于负样本的生成优势：
                delta_pos2hardneg=chosen_logratios[i]-neg_sample_logratio
                ######### 2.hard neg相对于其他easy neg的均值，其生成优势:
                delta_hard_neg2easy_neg=neg_sample_logratio-easy_neg_logratio[i]   #delta_hard_neg2easy_neg=neg_sample_logratio-easy_neg_logratio[idx]
      
                margin = (delta_pos2hardneg-delta_hard_neg2easy_neg).detach()
                margin_list.append(margin)

            margin=torch.mean(torch.stack(margin_list))
            margin_record.append(margin)

            # 对于dpo，这里的动态beta可以做两种尝试：1.DPO的reward来调节beta 2.SimPO的reward来调节beta
            # hard neg 当前seq 的<chosen,rej1,rej2..>loss计算
            for idx, neg_sample_logratio in enumerate(hard_negative_sample_list[i]):
              
                # instance/batch level选择的beta值不同
                beta_value=beta
                
                # # method1:（1).dpo reward 作为beta调整
                # ######### 1.正样本相对于负样本的生成优势：
                # delta_pos2hardneg=chosen_logratios[i]-neg_sample_logratio


                
                
                # ######### 2.hard neg相对于其他easy neg的均值，其生成优势:
                # delta_hard_neg2easy_neg=neg_sample_logratio-easy_neg_logratio[i]   

                
                ###### method2: (2). simpo reward作为beta调整
                delta_pos2hardneg=policy_chosen_logps[i]-batch_simpo_policy_hard_neg_list[i][idx]
                delta_hard_neg2easy_neg=batch_simpo_policy_hard_neg_list[i][idx]-batch_simpo_policy_easy_neg_list[i]
                
              
                # 记录这两个值
                pos2hardneg_record.append(float(delta_pos2hardneg.detach().cpu().item()))
                hardneg2easy_record.append(float(delta_hard_neg2easy_neg.detach().cpu().item()))
                

                
                # 计算差值
                delta_diff = delta_pos2hardneg - delta_hard_neg2easy_neg-margin
              
                
                delta_beta = delta_diff /(abs(delta_hard_neg2easy_neg)+abs(delta_pos2hardneg))
                delta_beta=torch.tanh(delta_beta ) * 0.5
                
                
                #动态beta启动这行代码
                beta_value=beta_value*(1+delta_beta)

                beta_value=float(beta_value.detach().cpu().item())
                beta_record.append(beta_value)
                
            sample_level_beta_record.append(beta_record)
            delta_pos2hardneg_record.append(pos2hardneg_record)  # 添加到记录列表
            delta_hard_neg2easy_neg_record.append(hardneg2easy_record)  # 添加到记录列表
            margin_record.append(margin_list)
      
      

        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本
           
            
            
            # K=len(hard_negative_sample_list[i])
            # DMPO_loss_item=-F.logsigmoid(beta*chosen_logratios-sum(beta*neg_sample_logratio for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i]) )/K)
            
            K=len(hard_negative_sample_list[i])
            DMPO_loss_item=-F.logsigmoid( sum(sample_level_beta_record[i][idx]*chosen_logratios/K - sample_level_beta_record[i][idx]*neg_sample_logratio/K for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i]) ) )
            
            batch_losses_list.append(DMPO_loss_item)

        # 将所有样本的 loss 堆叠成一个 tensor
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]

        
        return (losses,margin_record),batch_simpo_policy_hard_neg_list,batch_policy_rejected_list,chosen_rewards,rejected_rewards,(sample_level_beta_record,delta_pos2hardneg_record,delta_hard_neg2easy_neg_record)
       
          

    elif filter_mode=="DMPO_hard_negative_cluster_4":
            
       
       
        
        hard_negative_sample_list=[]
        batch_simpo_policy_hard_neg_list=[]
        batch_simpo_policy_easy_neg_list=[]
        # policy_logps_list=[]

        # 1.dpo rejected logratios
        batch_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        # 逆序记录：样本，rejected1,2,3,4的logratio

        #遍历 rejected logratio字典:    key:rejected1/2/3/4,   value:batchsize的rejected1样本概率
        for (key,value) in rejected_logratios.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(rejected_logratios['rejected1'])):
                batch_rejected.setdefault(i, []).append(value[i])

        # 2.policy rejected logps(simpo -- nll)
        
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
       
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)


        # 3.reference rejected logs
        batch_reference_rejected_logps={}
        for (key,value) in reference_rejected_logps.items():
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(reference_rejected_logps['rejected1'])):
                batch_reference_rejected_logps.setdefault(i, []).append(value[i])
               
        # avg_logratio=[] # batchsize,avg_negsample_reward
        easy_neg_logratio=[]

        # 负样本过滤/动态选择 最粗糙方案:log ratio小于平均值的加入logratio
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
           
            key = num  # 直接使用 num 作为键,(batch-size,)
            
            sample_neg_list=[]
            easy_neg_list=[]
            simpo_policy_hard_neg_list=[]
            simpo_policy_easy_neg_list=[]
            dynamic_flag=False

            # simpo reward 作为筛选，实际添加dpo的reward项
            if dynamic_flag==True:
                # 1.存在reward大于正样本
                for idx,value in enumerate(batch_policy_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= policy_chosen_logps[key]:
                        sample_neg_list.append(batch_rejected[key][idx])
                        
            
                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
                    ####原始数据准备（假设batch_policy_rejected[key]是reward列表）
                    rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=4, random_state=42).fit(rewards.reshape(-1, 1))
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(4)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0]

                    # 构建结果列表
                    sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                    simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                # 如果reward有大于正样本的，则进入easyneg的补充处理
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_rejected[key]) if val in sample_neg_list]
                    # 构建simpo_policy_hard_neg_list
                    simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                    
                   # 构建easy_neg_list和simpo_policy_easy_neg_list
                    
                    
                    if len(hard_indices)==len(batch_policy_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(batch_policy_rejected[key])) if i not in hard_indices]
                    

            else:
                rewards_tensors = batch_policy_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                # 使用K-Means聚类（分为3类）
                kmeans = KMeans(n_clusters=4, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                # 找到最高reward的类（按均值判断）
                cluster_means = [np.mean(rewards[labels == i]) for i in range(4)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0]

                # 构建结果列表
                sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                simpo_policy_hard_neg_list = [batch_policy_rejected[key][i] for i in hard_indices]
                easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                simpo_policy_easy_neg_list = [batch_policy_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
            # dpo reward related
            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
            # simpo reward related
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            
            # simpo reward related
            #batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            #batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}

      
      

        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本
           
            
            
            K=len(hard_negative_sample_list[i])
            DMPO_loss_item=-F.logsigmoid(beta*chosen_logratios-sum(beta*neg_sample_logratio for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i]) )/K)
            batch_losses_list.append(DMPO_loss_item)

        # 将所有样本的 loss 堆叠成一个 tensor
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]

        
        return (losses,None),A,A,chosen_rewards,rejected_rewards,(beta,beta,beta)


    elif filter_mode=="DMPO_hard_negative_reward_filter":
            
        
        
        hard_negative_sample_list=[]
        batch_simpo_policy_hard_neg_list=[]
        batch_simpo_policy_easy_neg_list=[]
        # policy_logps_list=[]

        # 1.dpo rejected logratios
        batch_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
        # 逆序记录：样本，rejected1,2,3,4的logratio

        #遍历 rejected logratio字典:    key:rejected1/2/3/4,   value:batchsize的rejected1样本概率
        for (key,value) in rejected_logratios.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(rejected_logratios['rejected1'])):
                batch_rejected.setdefault(i, []).append(value[i])

        # 2.policy rejected logps(simpo -- nll)
        
        batch_policy_rejected={}# 字典格式：键的数量：batchsize，值的数量：neg num. 样本:[rejected1,2,3,4...]
       
        for (key,value) in policy_rejected_logps.items(): 
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(policy_rejected_logps['rejected1'])):
                batch_policy_rejected.setdefault(i, []).append(value[i])

        # 将字典转换为二维列表
        batch_policy_rejected_list = []
        for i in range(len(batch_policy_rejected)):
            # 将每个样本的所有rejected值添加到列表中
            sample_rejected_values = []
            for tensor in batch_policy_rejected[i]:
                if isinstance(tensor, torch.Tensor):
                    # 如果是张量，转换为Python标量
                    sample_rejected_values.append(float(tensor.detach().cpu().item()))
                else:
                    # 如果已经是标量，直接添加
                    sample_rejected_values.append(float(tensor))
            batch_policy_rejected_list.append(sample_rejected_values)


        # 3.reference rejected logs
        batch_reference_rejected_logps={}
        for (key,value) in reference_rejected_logps.items():
            # 获得每个样本的rejected1,2,3,4...
            for i in range(len(reference_rejected_logps['rejected1'])):
                batch_reference_rejected_logps.setdefault(i, []).append(value[i])
               
        # avg_logratio=[] # batchsize,avg_negsample_reward
        easy_neg_logratio=[]

        # 负样本过滤/动态选择 最粗糙方案:log ratio小于平均值的加入logratio
        
        for num in range(len(rejected_logratios['rejected1'])):# batch_size
           
            key = num  # 直接使用 num 作为键,(batch-size,)
            
            sample_neg_list=[]
            easy_neg_list=[]
            simpo_policy_hard_neg_list=[]
            simpo_policy_easy_neg_list=[]
            dynamic_flag=True

            # dpo reward 作为筛选，实际添加dpo的reward项
            if dynamic_flag==True:
                # 1.存在reward大于正样本
                for idx,value in enumerate(batch_rejected[key]): # 遍历每个样本的rejected 1,2,3,4 ...logratio
                
                    #负样本的logratios大于正样本的logratios，则加入hard-neg列表【非固定的neg3】
                    if value>= chosen_logratios[key]:
                        sample_neg_list.append(batch_rejected[key][idx])
                        
            
                #  2.如果logratios没有大于正样本的，采用【固定的hard-top neg3】
                if len(sample_neg_list)<1:
                    ####原始数据准备（假设batch_rejected[key]是reward列表）
                    rewards_tensors = batch_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                    rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                    # 使用K-Means聚类（分为3类）
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                    labels = kmeans.labels_

                    # 找到最高reward的类（按均值判断）
                    cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                    high_reward_cluster = np.argmax(cluster_means)
                    hard_indices = np.where(labels == high_reward_cluster)[0]

                    # 构建结果列表
                    sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                    simpo_policy_hard_neg_list = [batch_rejected[key][i] for i in hard_indices]
                    easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]
                    simpo_policy_easy_neg_list = easy_neg_list
                # 如果reward有大于正样本的，则进入easyneg的补充处理
                else:
                    hard_indices = [idx for idx, val in enumerate(batch_rejected[key]) if val in sample_neg_list]
                    # 构建simpo_policy_hard_neg_list
                    simpo_policy_hard_neg_list = [batch_rejected[key][i] for i in hard_indices]
                    
                   # 构建easy_neg_list和simpo_policy_easy_neg_list
                    
                    
                    if len(hard_indices)==len(batch_rejected[key]):
                        # 说明所有负样本reward都大于正样本
                        simpo_policy_easy_neg_list=simpo_policy_hard_neg_list
                        easy_neg_list=sample_neg_list
                    else:
                        # 构建easy_neg_list和simpo_policy_easy_neg_list
                        easy_neg_list = [batch_rejected[key][i] for i in range(len(batch_rejected[key])) if i not in hard_indices]
                        simpo_policy_easy_neg_list = [batch_rejected[key][i] for i in range(len(batch_rejected[key])) if i not in hard_indices]
                    

            else:
                rewards_tensors = batch_rejected[key]  # 假设这是PyTorch张量
                    # 将所有张量移到CPU并转换为numpy数组
                rewards = np.array([tensor.detach().cpu().numpy() for tensor in rewards_tensors])


                # 使用K-Means聚类（分为3类）
                kmeans = KMeans(n_clusters=3, random_state=42).fit(rewards.reshape(-1, 1))
                labels = kmeans.labels_

                # 找到最高reward的类（按均值判断）
                cluster_means = [np.mean(rewards[labels == i]) for i in range(3)]
                high_reward_cluster = np.argmax(cluster_means)
                hard_indices = np.where(labels == high_reward_cluster)[0]

                # 构建结果列表
                sample_neg_list = [batch_rejected[key][i] for i in hard_indices]
                simpo_policy_hard_neg_list = [batch_rejected[key][i] for i in hard_indices]
                easy_neg_list = [batch_rejected[key][i] for i in range(len(rewards)) if i not in hard_indices]

            # dpo reward related
            hard_negative_sample_list.append(sample_neg_list)
            easy_neg_logratio.append(sum(easy_neg_list)/len(easy_neg_list))
            
            # simpo reward related
            batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
            batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            
            # simpo reward related
            #batch_simpo_policy_easy_neg_list.append(sum(simpo_policy_easy_neg_list)/len(simpo_policy_easy_neg_list))
            #batch_simpo_policy_hard_neg_list.append(simpo_policy_hard_neg_list)
        # hard_negative_sample_list 二维列表：{样本,选择的neg_sample}

      
      

        batch_losses_list=[]

        for i in range(chosen_logratios.size(0)):  # 遍历 batch 中的每个样本
           
            
            
            K=len(hard_negative_sample_list[i])
            DMPO_loss_item=-F.logsigmoid(beta*chosen_logratios-sum(beta*neg_sample_logratio for idx,neg_sample_logratio in enumerate(hard_negative_sample_list[i]) )/K)
            batch_losses_list.append(DMPO_loss_item)

        # 将所有样本的 loss 堆叠成一个 tensor
        losses = torch.stack(batch_losses_list)  # shape: [batch_size]

        
        return (losses,None),A,A,chosen_rewards,rejected_rewards,(beta,beta,beta)

    # # dpo simpo均适用,top1当前 按照dpo policy model筛选
   
   


    
    elif filter_mode == 'single_losses_tail':
        ##（把最小的loss过滤掉-> loss 低 <-> 高gap数据，不出意外：应该与过滤高gap的方法等价，猜测其beta曲线与高gap近似等价。其根据A-used和gap的差值来调整beta都不够合理，因为原始的double方法是双边过滤，过滤离群gap：低gap 高gap，所以设定该基准是合理的。
        # 尤其是过滤比例较高的情况下，整体A_used偏低，造成beta值整体异常偏大：基准——当前的gap mean以全体gap来更新是否合理？？？）
        
        losses = -F.logsigmoid(beta * (chosen_logratios - rejected_logratios['rejected1']))
        losses_global = losses#
        k = int(losses_global.size(0) * (1 - mode_weight))
        lower_bound_value = losses_global.topk(k, largest=True).values[-1]
        global_mask = losses_global >= lower_bound_value # loss较高的样本为1

        # 使用 global_mask 来选择参与计算的 A：loss较高的样本，A_filtered返回的是实际参与计算的，而不是被过滤的
        A_filtered = A[global_mask]
        A_used = torch.mean(A_filtered) if adjust_level == 'batch_level' else A

        beta_used = beta * (1 -alpha_a * (A_used - gap_mean))  #+torch.where(A_used - gap_mean < 0, 1, 0)


        beta_used = torch.tensor(beta_used).to(A_used.device)
        beta_used = beta_used.detach()
        beta_used = beta_used.clamp(min=2e-1)
        beta_used = beta_used.clamp(min=beta_clamp_min,max=beta_clamp_max) if beta_clamp==True else beta_used
         
        # 2.dpo loss:
        losses= -F.logsigmoid( beta_used*(chosen_logratios-rejected_logratios['rejected1']) )
        return (losses,global_mask),A,A_filtered,chosen_rewards,rejected_rewards,beta_used

         ### 只进行beta过滤 不进行dpo loss的过滤
        #global_mask=None


   ### 与double在mode_weight=0.2时的过滤样本数量不一致：一个是向下取整，一个是向上取整
    elif filter_mode == 'single_gt_mean_filter':
    #reward discrepancy过大的样本，没有信息增量，在推荐里，也应认同。
    # 但discrepancy较小的数据，推荐这边的数据，负样本来自于真实的用户交互，似乎不应该被视为标签错误？

        filtered_mask = torch.ones_like(A, dtype=torch.bool)
        A_used = A

        # 按照 A_used 从高到低排序
        sorted_order = torch.argsort(A_used, descending=True)

        # 计算需要过滤的样本数量
        total_samples = A.numel()
        sample_num = int(mode_weight * total_samples)

        if sample_num > 0:
            # 过滤掉前 sample_num 个最高 A_used 样本
            filter_indices = sorted_order[:sample_num]
            filtered_mask[filter_indices] = False
        A_filtered=A[filtered_mask]
        # 使用过滤后的数据来 调整 beta
        A_used = torch.mean(A[filtered_mask]) if adjust_level == 'batch_level' else A
        # 把较高的gap过滤掉 -> 当前beta_used的计算是否还合理？？？
        # 修改后的代码
        beta_used = beta * (1 + alpha_a * (A_used - gap_mean + torch.where(A_used - gap_mean < 0, 1, 0)))

        beta_used = beta_used.detach()
        beta_used = beta_used.clamp(min=2e-1)
        beta_used = beta_used.clamp(min=beta_clamp_min, max=beta_clamp_max) if beta_clamp else beta_used

        # 计算损失
        global_mask = filtered_mask.detach()
        losses = -F.logsigmoid(beta_used * (chosen_logratios - rejected_logratios['rejected1']))
        return (losses,global_mask),A,A_filtered,chosen_rewards,rejected_rewards,beta_used

## gap较大的分配 较大的beta，gap较低的分配小beta
  
    return (losses,global_mask),A,A_filtered,chosen_rewards,rejected_rewards,beta_used


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
        alpha:float= 0.6,
        mode_weight:float=0.2,
        filter_mode:str='double',
        adjust_level:str='batch_level',
        beta_clamp:bool=False,
        beta_clamp_min:float=0.2,
        beta_clamp_max:float=5.0,
        betadpo_delay:bool=False,
        gap_update_filter_only:bool=False,
        loss_type:str='dpo',
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
        self.world_size=int(os.environ["WORLD_SIZE"])
        
        ### 新增调整
        self.mode_weight=mode_weight
        self.alpha_a=alpha
        self.filter_mode=filter_mode
        self.adjust_level=adjust_level
        self.beta_clamp=beta_clamp
        self.beta_clamp_min=beta_clamp_min
        self.beta_clamp_max=beta_clamp_max
        self.betadpo_delay=betadpo_delay
        self.gap_update_filter_only=gap_update_filter_only
        self.loss_type=loss_type
## 避开softmax分布 可以加一个极小值
        self.gap_mean = torch.zeros(1, device='cuda')#+1e-3
        self.gap_std = torch.zeros(1, device='cuda')#+6e-3
        self.loss_mean = torch.zeros(1, device='cuda')#+6e-3
        self.loss_std = torch.zeros(1, device='cuda')#+6e-3
        self.delta_hard_neg2easy_neg_mean = torch.zeros(1, device='cuda')
        self.delta_hard_neg2easy_neg_std = torch.zeros(1, device='cuda')

        self.delta_pos2hardneg_mean = torch.zeros(1, device='cuda')
        self.delta_pos2hardneg_std = torch.zeros(1, device='cuda')
        # high gap high beta /high gap small beta 4阶段探究

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
        # 把 chosen 和 rejected response 拼接起来
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
            ## 查探 per_token_logps * loss_mask 的形状，以想办法计算chosen  answer/rejected answer 单独的logps
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor], torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        # print(concatenated_batch["concatenated_input_ids"].shape)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False#self.loss_type in ["ipo", "simpo"], ###对长度求平均的操作
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
    

    ####  用于动态更新gap_mean，gap_std，loss_mean和loss_std
    def update_and_sync_tensor_mean(self, gap_local, loss_local, delta_hard_neg2easy_neg_local=None, delta_pos2hardneg_local=None, gamma=0.9):
        with torch.no_grad():
            batch_gap_mean = gap_local.mean()
            batch_gap_std = gap_local.std()
            batch_loss_mean = loss_local.mean()
            batch_loss_std = loss_local.std()
            
            # 更新现有的统计值
            self.gap_mean.mul_(gamma).add_(batch_gap_mean, alpha=1-gamma)
            self.gap_std.mul_(gamma).add_(batch_gap_std, alpha=1-gamma)
            self.loss_mean.mul_(gamma).add_(batch_loss_mean, alpha=1-gamma)
            self.loss_std.mul_(gamma).add_(batch_loss_std, alpha=1-gamma)
            
            # 如果提供了 delta_hard_neg2easy_neg_local，则更新其统计值
            if delta_hard_neg2easy_neg_local is not None:
                batch_delta_mean = delta_hard_neg2easy_neg_local.mean()
                batch_delta_std = delta_hard_neg2easy_neg_local.std()
                self.delta_hard_neg2easy_neg_mean.mul_(gamma).add_(batch_delta_mean, alpha=1-gamma)
                self.delta_hard_neg2easy_neg_std.mul_(gamma).add_(batch_delta_std, alpha=1-gamma)
            
            # 如果提供了 delta_pos2hardneg_local，则更新其统计值
            if delta_pos2hardneg_local is not None:
                batch_pos2hardneg_mean = delta_pos2hardneg_local.mean()
                batch_pos2hardneg_std = delta_pos2hardneg_local.std()
                self.delta_pos2hardneg_mean.mul_(gamma).add_(batch_pos2hardneg_mean, alpha=1-gamma)
                self.delta_pos2hardneg_std.mul_(gamma).add_(batch_pos2hardneg_std, alpha=1-gamma)
            
            # 如果使用了分布式训练，同步所有统计值
            if self.world_size > 1:
                dist.all_reduce(self.gap_mean, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.gap_std, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.loss_mean, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.loss_std, op=dist.ReduceOp.SUM)
                
                if delta_hard_neg2easy_neg_local is not None:
                    dist.all_reduce(self.delta_hard_neg2easy_neg_mean, op=dist.ReduceOp.SUM)
                    dist.all_reduce(self.delta_hard_neg2easy_neg_std, op=dist.ReduceOp.SUM)
                
                if delta_pos2hardneg_local is not None:
                    dist.all_reduce(self.delta_pos2hardneg_mean, op=dist.ReduceOp.SUM)
                    dist.all_reduce(self.delta_pos2hardneg_std, op=dist.ReduceOp.SUM)
                
                self.gap_mean /= self.world_size
                self.gap_std /= self.world_size
                self.loss_mean /= self.world_size
                self.loss_std /= self.world_size
                
                if delta_hard_neg2easy_neg_local is not None:
                    self.delta_hard_neg2easy_neg_mean /= self.world_size
                    self.delta_hard_neg2easy_neg_std /= self.world_size
                
                if delta_pos2hardneg_local is not None:
                    self.delta_pos2hardneg_mean /= self.world_size
                    self.delta_pos2hardneg_std /= self.world_size

    # def update_and_sync_tensor_mean(self, gap_local, loss_local, gamma=0.9):
    #     with torch.no_grad():
    #         batch_gap_mean = gap_local.mean()
    #         batch_gap_std = gap_local.std()
    #         batch_loss_mean = loss_local.mean()
    #         batch_loss_std = loss_local.std()
    #         # 更新loss_mean
    #         self.gap_mean.mul_(gamma).add_(batch_gap_mean, alpha=1-gamma)
    #         self.gap_std.mul_(gamma).add_(batch_gap_std, alpha=1-gamma)
    #         self.loss_mean.mul_(gamma).add_(batch_loss_mean, alpha=1-gamma)
    #         self.loss_std.mul_(gamma).add_(batch_loss_std, alpha=1-gamma)
            
    #         ###  如果使用了分布式训练，同步loss_mean
    #         if self.world_size > 1:
    #             # 我们使用SUM操作进行all_reduce，然后将结果除以大小来取平均
    #             dist.all_reduce(self.gap_mean, op=dist.ReduceOp.SUM)
    #             dist.all_reduce(self.gap_std, op=dist.ReduceOp.SUM)
    #             dist.all_reduce(self.loss_mean, op=dist.ReduceOp.SUM)
    #             dist.all_reduce(self.loss_std, op=dist.ReduceOp.SUM)
    #             self.gap_mean /= self.world_size
    #             self.gap_std /= self.world_size
    #             self.loss_mean /= self.world_size
    #             self.loss_std /= self.world_size


    def beta_DPO_get_batch_metrics(
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
        #####
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

#### global_mask待使用
        (losses, margin_record), hard_neg_reward,all_neg_reward,chosen_rewards, rejected_rewards,(beta_used, delta_pos2hardneg_record, delta_hard_neg2easy_neg_record) = preference_loss(
            ref_model_enabled=self.ref_model is not None,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            filter_mode=self.filter_mode,
            adjust_level=self.adjust_level,
            beta_clamp=self.beta_clamp,
            beta_clamp_min=self.beta_clamp_min,
            beta_clamp_max=self.beta_clamp_max,
            beta=self.beta,
            mode_weight=self.mode_weight,
            alpha_a=self.alpha_a,
            epoch=0,#self.state.epoch,
            betadpo_delay=self.betadpo_delay,
            gap_mean=self.gap_mean, gap_std=self.gap_std,
            delta_hard_neg2easy_neg_mean=self.delta_hard_neg2easy_neg_mean,  # 传递全局均值
            delta_pos2hardneg_mean=self.delta_pos2hardneg_mean  # 新增参数
        )



        # reward_accuracies 记录 chosen 比所有 rejected 的收益都大的比例是多少
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

        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
           

        for key in rejected_rewards:
            metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        for key in policy_rejected_logps:
            metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        # for key in policy_rejected_logits:
        #     metrics[f"{prefix}logits/rejected-{key}"] = policy_rejected_logits[key].detach().cpu().numpy().mean()
        # metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()

     
        
      

        def convert_tensors_to_list(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            elif isinstance(obj, list):
                return [convert_tensors_to_list(item) for item in obj]
            else:
                return obj  # 如果不是 Tensor 或 List，直接返回（如 float/int）

        # 使用方式
        metrics["hard_neg_reward"] = convert_tensors_to_list(hard_neg_reward)
        metrics["all_neg_reward"] = convert_tensors_to_list(all_neg_reward)
        metrics[f"margin_record"] = convert_tensors_to_list(margin_record)

        ### betadpo metrics record


        # gap_local实际上是DPO中的log ratio的差值，计算完一个batch的gap之后，动态更新基准 M0,标准差std- sigma。

        chosen_logratios = policy_chosen_logps - reference_chosen_logps if self.ref_model is not None else policy_chosen_logps
        rejected_logratios = {}
        for key in policy_rejected_logps:
            rejected_logratios[key] = policy_rejected_logps[key] - reference_rejected_logps[key] if self.ref_model is not None else policy_rejected_logps[key]
        # 进行 multi-sample 的处理，gap_local，gap-std等一系列依赖于 gap的计算
        
    
        multi_neg_gap_sum=0
        multi_neg_gap_sum = sum((chosen_logratios - rejected_logratios[key]).detach() for key in rejected_logratios)

        gap_local = multi_neg_gap_sum / len(rejected_logratios)

        loss_local = -F.logsigmoid(self.beta * gap_local)

         # 更新全局统计值，包括 delta_hard_neg2easy_neg
          # 从返回的记录中提取 delta_hard_neg2easy_neg 值
        if isinstance(delta_hard_neg2easy_neg_record, list) and len(delta_hard_neg2easy_neg_record) > 0:
            # 将嵌套列表展平为一维张量
            flattened_deltas = []
            for sample_deltas in delta_hard_neg2easy_neg_record:
                if isinstance(sample_deltas, list):
                    flattened_deltas.extend(sample_deltas)
                else:
                    flattened_deltas.append(sample_deltas)
            
            if flattened_deltas:
                delta_hard_neg2easy_neg_local = torch.tensor(flattened_deltas, device=self.gap_mean.device)
            else:
                delta_hard_neg2easy_neg_local = None
        else:
            delta_hard_neg2easy_neg_local = None
# 处理 delta_hard_neg2easy_neg_record（保持原有逻辑）
        if isinstance(delta_pos2hardneg_record, list) and len(delta_pos2hardneg_record) > 0:
            # 将嵌套列表展平为一维张量
            flattened_pos2hardneg = []
            
            for sample_deltas in delta_pos2hardneg_record:
                if isinstance(sample_deltas, list):
                    flattened_pos2hardneg.extend(sample_deltas)
                else:
                    flattened_pos2hardneg.append(sample_deltas)
            
            if flattened_pos2hardneg:
                delta_pos2hardneg_local = torch.tensor(flattened_pos2hardneg, device=self.gap_mean.device)
            else:
                delta_pos2hardneg_local = None
        else:
            delta_pos2hardneg_local = None

        self.update_and_sync_tensor_mean(gap_local, loss_local, delta_hard_neg2easy_neg_local,delta_pos2hardneg_local)
        
        #self.update_and_sync_tensor_mean(gap_local, loss_local)
        metrics[f"{prefix}/gap_mean"] = self.gap_mean.cpu().numpy().tolist()
        metrics[f'{prefix}/gap_std'] = self.gap_std.cpu().numpy().tolist()
        metrics[f'{prefix}/loss_mean'] = self.loss_mean.cpu().numpy().tolist()
        metrics[f'{prefix}/loss_std'] = self.loss_std.cpu().numpy().tolist()
        
        metrics[f'{prefix}/delta_hard_neg2easy_neg_mean'] = self.delta_hard_neg2easy_neg_mean.cpu().numpy().tolist()
        metrics[f'{prefix}/delta_hard_neg2easy_neg_std'] = self.delta_hard_neg2easy_neg_std.cpu().numpy().tolist()
        
            # 添加 delta_pos2hardneg 的统计值
        metrics[f'{prefix}/delta_pos2hardneg_mean'] = self.delta_pos2hardneg_mean.cpu().numpy().tolist()
        metrics[f'{prefix}/delta_pos2hardneg_std'] = self.delta_pos2hardneg_std.cpu().numpy().tolist()
        if isinstance(beta_used, float):
            beta_used_list_or_float = beta_used
            
        elif isinstance(beta_used, list):
            beta_used_list_or_float = beta_used
        else:
            beta_used_list_or_float = beta_used.cpu().numpy().tolist()
            delta_pos2hardneg_record= delta_pos2hardneg_record.cpu().numpy().tolist()
            delta_hard_neg2easy_neg_record = delta_hard_neg2easy_neg_record.cpu().numpy().tolist()
        if isinstance(beta_used_list_or_float, list):
            metrics[f"{prefix}/beta_used"] = beta_used_list_or_float
            metrics[f"{prefix}/delta_pos2hardneg_record"] = delta_pos2hardneg_record
            metrics[f"{prefix}/delta_hard_neg2easy_neg_record"] = delta_hard_neg2easy_neg_record
        elif isinstance(beta_used_list_or_float, float):
            metrics[f"{prefix}/beta_used"] = [beta_used_list_or_float]
            metrics[f"{prefix}/delta_pos2hardneg_record"] = delta_pos2hardneg_record
            metrics[f"{prefix}/delta_hard_neg2easy_neg_record"] = delta_hard_neg2easy_neg_record

        
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
        loss, metrics = self.beta_DPO_get_batch_metrics(model, inputs, train_eval="train")

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
            loss, metrics = self.beta_DPO_get_batch_metrics(model, inputs, train_eval="eval")

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
##### 处理 平均记录的情况


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
            if key in ['filtered_losses', 'margin_record','losses_list', 'weight_sample','hard_neg_reward',
                       'actual_losses_list','all_neg_reward','local_mask','eval_/beta_used','train_/beta_used','train_/delta_hard_neg2easy_neg_record','train_/delta_pos2hardneg_record']:
                logs[key] = metrics  # 保留完整列表
            else:
                logs[key] = torch.tensor(metrics).float().mean().item()



        del self._stored_metrics[train_eval]
        return super().log(logs)


