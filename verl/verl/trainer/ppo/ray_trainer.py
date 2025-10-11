# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Portions of this file are modifications by OPPO PersonalAI Team.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                        RayWorkerGroup)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           process_validation_metrics)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import (BaseCheckpointManager,
                                                      find_latest_ckpt_path)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import (get_seqlen_balanced_partitions,
                                         log_seqlen_unbalance)
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        
        # Initialize tool metrics tracking
        self.tool_metrics_history = []
        self.global_tool_metrics = {
            "total_steps": 0,
            "cumulative_tool_calls": 0,
            "cumulative_trajectories": 0,
            "tool_usage_per_step": {},
        }

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            # assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import \
                collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")



    def _maybe_log_val_generations(self, inputs, outputs, scores, data_sources=None, ground_truths=None):  
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        import pickle


        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np
        if data_sources is not None and ground_truths is not None:  
            samples = list(zip(inputs, outputs, scores, data_sources, ground_truths))  
        elif data_sources is not None:  
            samples = list(zip(inputs, outputs, scores, data_sources))  
        else:  
            samples = list(zip(inputs, outputs, scores))  
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to configured logger
        # self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

        # Save to local JSONL file if path is configured  
        jsonl_save_path = self.config.trainer.get("val_generations_jsonl_path", None)  
        if jsonl_save_path:  
            self._save_val_generations_to_jsonl(samples, jsonl_save_path)

    def _process_tool_metrics_from_rollout(self, rollout_data):
        """Extract detailed tool metrics from rollout data during training"""
        tool_metrics = {
            "total_tool_calls": 0,
            "tool_calls_by_type": {},
            "tool_execution_times": {},
            "tool_success_rates": {},
            "trajectory_turns": [],
            "avg_turns_per_trajectory": 0,
            "trajectory_tool_usage": [],
            "unique_tools_per_trajectory": [],
            "tool_call_sequences": [],
        }
        
        # Check if rollout data contains detailed tool metrics
        if hasattr(rollout_data, 'non_tensor_batch') and 'detailed_tool_metrics' in rollout_data.non_tensor_batch:
            detailed_metrics_batch = rollout_data.non_tensor_batch['detailed_tool_metrics']
            
            for trajectory_metrics in detailed_metrics_batch:
                if isinstance(trajectory_metrics, dict):
                    # Aggregate metrics from detailed tracking
                    tool_metrics["total_tool_calls"] += trajectory_metrics.get("total_tool_calls", 0)
                    tool_metrics["trajectory_turns"].append(trajectory_metrics.get("conversation_turns", 0))
                    
                    # Aggregate tool calls by type
                    for tool_name, count in trajectory_metrics.get("tool_calls_by_type", {}).items():
                        if tool_name not in tool_metrics["tool_calls_by_type"]:
                            tool_metrics["tool_calls_by_type"][tool_name] = 0
                        tool_metrics["tool_calls_by_type"][tool_name] += count
                    
                    # Track unique tools per trajectory
                    unique_tools = trajectory_metrics.get("unique_tools_used", [])
                    tool_metrics["unique_tools_per_trajectory"].append(len(unique_tools))
                    
                    # Store trajectory-level usage
                    tool_metrics["trajectory_tool_usage"].append({
                        "total_tools": trajectory_metrics.get("total_tool_calls", 0),
                        "tools_used": trajectory_metrics.get("tool_calls_by_type", {}),
                        "turns": trajectory_metrics.get("conversation_turns", 0),
                        "unique_tools": unique_tools,
                        "unique_tools_count": trajectory_metrics.get("unique_tools_count", 0)
                    })
                    
                    # Store tool call sequences for analysis
                    tool_call_sequence = trajectory_metrics.get("tool_call_sequence", [])
                    if tool_call_sequence:
                        tool_metrics["tool_call_sequences"].append(tool_call_sequence)
        
        
        # Calculate averages and additional metrics
        num_trajectories = len(tool_metrics["trajectory_turns"])
        if num_trajectories > 0:
            tool_metrics["avg_turns_per_trajectory"] = round(
                sum(tool_metrics["trajectory_turns"]) / num_trajectories, 2
            )
            tool_metrics["avg_tool_calls_per_trajectory"] = round(
                tool_metrics["total_tool_calls"] / num_trajectories, 2
            )
            
            # Calculate per-tool averages
            tool_metrics["avg_tool_calls_by_type"] = {
                tool: round(count / num_trajectories, 2) 
                for tool, count in tool_metrics["tool_calls_by_type"].items()
            }
            
            # Calculate unique tools per trajectory stats
            if tool_metrics["unique_tools_per_trajectory"]:
                tool_metrics["avg_unique_tools_per_trajectory"] = round(
                    sum(tool_metrics["unique_tools_per_trajectory"]) / num_trajectories, 2
                )
                tool_metrics["max_unique_tools_per_trajectory"] = max(tool_metrics["unique_tools_per_trajectory"])
                tool_metrics["min_unique_tools_per_trajectory"] = min(tool_metrics["unique_tools_per_trajectory"])
            else:
                tool_metrics["avg_unique_tools_per_trajectory"] = 0
                tool_metrics["max_unique_tools_per_trajectory"] = 0
                tool_metrics["min_unique_tools_per_trajectory"] = 0
            
            # Calculate tool diversity (how many different tools used across all trajectories)
            all_tools_used = set()
            for usage in tool_metrics["trajectory_tool_usage"]:
                all_tools_used.update(usage.get("unique_tools", []))
            tool_metrics["total_unique_tools_used"] = len(all_tools_used)
            tool_metrics["tool_diversity_score"] = len(all_tools_used) / len(tool_metrics["tool_calls_by_type"]) if tool_metrics["tool_calls_by_type"] else 0
            
        else:
            tool_metrics["avg_turns_per_trajectory"] = 0
            tool_metrics["avg_tool_calls_per_trajectory"] = 0
            tool_metrics["avg_unique_tools_per_trajectory"] = 0
            tool_metrics["max_unique_tools_per_trajectory"] = 0
            tool_metrics["min_unique_tools_per_trajectory"] = 0
            tool_metrics["total_unique_tools_used"] = 0
            tool_metrics["tool_diversity_score"] = 0
            tool_metrics["avg_tool_calls_by_type"] = {}
        
        return tool_metrics
    
    
    def _save_val_generations_to_jsonl(self, samples, jsonl_path):  
        """Save validation samples to JSONL files (both step-specific and combined)"""  
        import json
        import os

        # Create directory if it doesn't exist  
        base_dir = os.path.dirname(jsonl_path)
        os.makedirs(base_dir, exist_ok=True)  
        
        # Prepare data for JSONL  
        jsonl_data = []  
        for sample in samples:  
            if len(sample) == 5:  # input, output, score, data_source, ground_truth  
                entry = {  
                    "step": self.global_steps,  
                    "input": sample[0],  
                    "output": sample[1],   
                    "score": sample[2],  
                    "data_source": sample[3],  
                    "ground_truth": sample[4]  
                }  
            elif len(sample) == 4:  # input, output, score, data_source  
                entry = {  
                    "step": self.global_steps,  
                    "input": sample[0],  
                    "output": sample[1],  
                    "score": sample[2],   
                    "data_source": sample[3]  
                }  
            else:  # input, output, score  
                entry = {  
                    "step": self.global_steps,  
                    "input": sample[0],  
                    "output": sample[1],  
                    "score": sample[2]  
                }  
            jsonl_data.append(entry)  
        
        # Save to step-specific file
        base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
        step_specific_path = os.path.join(base_dir, f"{base_name}_step_{self.global_steps}.jsonl")
        
        with open(step_specific_path, 'w', encoding='utf-8') as f:  
            for entry in jsonl_data:  
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Also append to the main combined file (for backward compatibility)
        # Clear the main file if this is the first validation of this training session
        # We track this using a marker file to distinguish new training from resumed training
        # marker_file = jsonl_path + ".training_session"
        # is_new_training_session = not os.path.exists(marker_file)
        
        # if is_new_training_session:
        #     # Create marker file to indicate this training session has started
        #     with open(marker_file, 'w') as f:
        #         f.write(f"Training session started at step {self.global_steps}\n")
        #     main_file_mode = 'w'  # Clear the main file for new training session
        # else:
        #     main_file_mode = 'a'  # Append to existing file
        
        # with open(jsonl_path, main_file_mode, encoding='utf-8') as f:  
        #     for entry in jsonl_data:  
        #         f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
        print(f"Saved {len(jsonl_data)} validation samples to:")
        print(f"  - Step-specific: {step_specific_path}")
        # print(f"  - Combined: {jsonl_path} (mode: {main_file_mode})") 


    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        sample_data_sources = []  
        sample_ground_truths = []  

        # limit=3
        # loop=0

        for test_data in self.val_dataloader:
            # loop+=1
            # if loop>limit:
            #     break

            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()
            # breakpoint()
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            batch_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(test_batch))  
            batch_ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in test_batch]  
            
            sample_data_sources.extend(batch_data_sources)  
            sample_ground_truths.extend(batch_ground_truths)  


        # breakpoint()
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, data_sources=sample_data_sources, ground_truths=sample_ground_truths)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict
    
    def _calculate_actual_response_lengths(self, output_ids):
        """Calculate actual response lengths excluding padding tokens and observation content"""
        response_lengths = []
        
        for ids in output_ids:
            # Convert to list if it's a tensor
            if hasattr(ids, 'cpu'):
                ids = ids.cpu().tolist()
            elif hasattr(ids, 'tolist'):
                ids = ids.tolist()
            
            # First decode the full text
            full_text = self.tokenizer.decode(ids, skip_special_tokens=True)
            
            # Remove observation content
            cleaned_text = self._remove_observation_content(full_text)
            
            # Re-encode the cleaned text to get actual token count
            if cleaned_text.strip():  # Only if there's content left
                cleaned_ids = self.tokenizer.encode(cleaned_text, add_special_tokens=False)
                actual_length = len(cleaned_ids)
            else:
                actual_length = 0
            
            response_lengths.append(actual_length)
        
        return response_lengths

    def _remove_observation_content(self, text):
        """Remove content between <observation> and </observation> tags"""
        import re
        
        # Remove all observation blocks (case-insensitive, multiline)
        pattern = r'<observation>.*?</observation>'
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up extra whitespace that might be left
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)  # Remove multiple empty lines
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def _calculate_response_length_stats(self, response_lengths):
        """Calculate comprehensive statistics for response lengths"""
        if not response_lengths:
            return {
                "avg_response_length": 0,
                "median_response_length": 0,
                "max_response_length": 0,
                "min_response_length": 0,
                "std_response_length": 0,
                "percentile_25_response_length": 0,
                "percentile_75_response_length": 0,
                "percentile_90_response_length": 0,
                "percentile_95_response_length": 0,
                "total_responses": 0,
            }
        
        import numpy as np
        
        response_lengths = np.array(response_lengths)
        
        stats = {
            "avg_response_length": round(np.mean(response_lengths), 2),
            "median_response_length": round(np.median(response_lengths), 2),
            "max_response_length": int(np.max(response_lengths)),
            "min_response_length": int(np.min(response_lengths)),
            "std_response_length": round(np.std(response_lengths), 2),
            "percentile_25_response_length": round(np.percentile(response_lengths, 25), 2),
            "percentile_75_response_length": round(np.percentile(response_lengths, 75), 2),
            "percentile_90_response_length": round(np.percentile(response_lengths, 90), 2),
            "percentile_95_response_length": round(np.percentile(response_lengths, 95), 2),
            "total_responses": len(response_lengths),
        }
        
        # Calculate distribution statistics
        stats["response_length_variance"] = round(np.var(response_lengths), 2)
        stats["response_length_range"] = stats["max_response_length"] - stats["min_response_length"]
        
        # Calculate length distribution buckets
        length_buckets = {
            "short_responses_count": int(np.sum(response_lengths < 50)),  # < 50 tokens
            "medium_responses_count": int(np.sum((response_lengths >= 50) & (response_lengths < 200))),  # 50-200 tokens
            "long_responses_count": int(np.sum((response_lengths >= 200) & (response_lengths < 500))),  # 200-500 tokens
            "very_long_responses_count": int(np.sum(response_lengths >= 500)),  # >= 500 tokens
        }
        
        # Calculate percentages
        total_responses = len(response_lengths)
        length_buckets.update({
            "short_responses_pct": round(length_buckets["short_responses_count"] / total_responses * 100, 2),
            "medium_responses_pct": round(length_buckets["medium_responses_count"] / total_responses * 100, 2),
            "long_responses_pct": round(length_buckets["long_responses_count"] / total_responses * 100, 2),
            "very_long_responses_pct": round(length_buckets["very_long_responses_count"] / total_responses * 100, 2),
        })
        
        stats.update(length_buckets)
        
        return stats

    def _validate1(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        sample_data_sources = []  
        sample_ground_truths = []  

        # Tool metrics aggregation
        all_tool_metrics = []
        
        # Response length tracking
        all_response_lengths = []

        # limit=3
        # loop=0

        for test_data in self.val_dataloader:
            # loop+=1
            # if loop>limit:
            #     break

            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()
            # breakpoint()
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs and calculate actual response lengths
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            
            # Calculate actual response lengths (excluding padding tokens)
            batch_response_lengths = self._calculate_actual_response_lengths(output_ids)
            all_response_lengths.extend(batch_response_lengths)

            test_batch = test_batch.union(test_output_gen_batch)

            # Extract tool metrics from the current batch
            batch_tool_metrics = self._process_tool_metrics_from_rollout(test_output_gen_batch)
            all_tool_metrics.append(batch_tool_metrics)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            batch_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(test_batch))  
            batch_ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in test_batch]  
            
            sample_data_sources.extend(batch_data_sources)  
            sample_ground_truths.extend(batch_ground_truths)  

        # Aggregate tool metrics across all batches
        aggregated_tool_metrics = self._aggregate_validation_tool_metrics(all_tool_metrics)
        
        # Calculate response length statistics
        response_length_stats = self._calculate_response_length_stats(all_response_lengths)

        # breakpoint()
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, data_sources=sample_data_sources, ground_truths=sample_ground_truths)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Add tool metrics to the final metric dictionary
        for metric_name, metric_value in aggregated_tool_metrics.items():
            if isinstance(metric_value, dict):
                # For nested dictionaries (like tool_calls_by_type)
                for sub_key, sub_value in metric_value.items():
                    metric_dict[f"val-tool/{metric_name}/{sub_key}"] = sub_value
            else:
                # For simple metrics
                metric_dict[f"val-tool/{metric_name}"] = metric_value
        
        # Add response length statistics to the final metric dictionary
        for metric_name, metric_value in response_length_stats.items():
            metric_dict[f"val-response/{metric_name}"] = metric_value

        return metric_dict

    def _aggregate_validation_tool_metrics(self, all_tool_metrics):
        """Aggregate tool metrics across all validation batches"""
        if not all_tool_metrics:
            return {}
        
        aggregated = {
            "total_tool_calls": 0,
            "tool_calls_by_type": {},
            "trajectory_turns": [],
            "trajectory_tool_usage": [],
            "unique_tools_per_trajectory": [],
            "tool_call_sequences": [],
        }
        
        # Aggregate metrics from all batches
        for batch_metrics in all_tool_metrics:
            aggregated["total_tool_calls"] += batch_metrics.get("total_tool_calls", 0)
            aggregated["trajectory_turns"].extend(batch_metrics.get("trajectory_turns", []))
            aggregated["trajectory_tool_usage"].extend(batch_metrics.get("trajectory_tool_usage", []))
            aggregated["unique_tools_per_trajectory"].extend(batch_metrics.get("unique_tools_per_trajectory", []))
            aggregated["tool_call_sequences"].extend(batch_metrics.get("tool_call_sequences", []))
            
            # Aggregate tool calls by type
            for tool_name, count in batch_metrics.get("tool_calls_by_type", {}).items():
                if tool_name not in aggregated["tool_calls_by_type"]:
                    aggregated["tool_calls_by_type"][tool_name] = 0
                aggregated["tool_calls_by_type"][tool_name] += count
        
        # Calculate final aggregated metrics
        num_trajectories = len(aggregated["trajectory_turns"])
        if num_trajectories > 0:
            aggregated["avg_turns_per_trajectory"] = round(
                sum(aggregated["trajectory_turns"]) / num_trajectories, 2
            )
            aggregated["avg_tool_calls_per_trajectory"] = round(
                aggregated["total_tool_calls"] / num_trajectories, 2
            )
            
            # Calculate per-tool averages
            aggregated["avg_tool_calls_by_type"] = {
                tool: round(count / num_trajectories, 2) 
                for tool, count in aggregated["tool_calls_by_type"].items()
            }
            
            # Calculate unique tools per trajectory stats
            if aggregated["unique_tools_per_trajectory"]:
                aggregated["avg_unique_tools_per_trajectory"] = round(
                    sum(aggregated["unique_tools_per_trajectory"]) / num_trajectories, 2
                )
                aggregated["max_unique_tools_per_trajectory"] = max(aggregated["unique_tools_per_trajectory"])
                aggregated["min_unique_tools_per_trajectory"] = min(aggregated["unique_tools_per_trajectory"])
            else:
                aggregated["avg_unique_tools_per_trajectory"] = 0
                aggregated["max_unique_tools_per_trajectory"] = 0
                aggregated["min_unique_tools_per_trajectory"] = 0
            
            # Calculate tool diversity
            all_tools_used = set()
            for usage in aggregated["trajectory_tool_usage"]:
                all_tools_used.update(usage.get("unique_tools", []))
            aggregated["total_unique_tools_used"] = len(all_tools_used)
            aggregated["tool_diversity_score"] = round(
                len(all_tools_used) / len(aggregated["tool_calls_by_type"]) if aggregated["tool_calls_by_type"] else 0, 2
            )
            
            # Calculate tool usage distribution statistics
            tool_usage_counts = [usage.get("total_tools", 0) for usage in aggregated["trajectory_tool_usage"]]
            if tool_usage_counts:
                aggregated["tool_usage_std"] = round(np.std(tool_usage_counts), 2)
                aggregated["tool_usage_median"] = round(np.median(tool_usage_counts), 2)
                aggregated["tool_usage_percentile_75"] = round(np.percentile(tool_usage_counts, 75), 2)
                aggregated["tool_usage_percentile_25"] = round(np.percentile(tool_usage_counts, 25), 2)
            
            # Calculate tool call sequence statistics
            if aggregated["tool_call_sequences"]:
                sequence_lengths = [len(seq) for seq in aggregated["tool_call_sequences"]]
                aggregated["avg_tool_sequence_length"] = round(sum(sequence_lengths) / len(sequence_lengths), 2)
                aggregated["max_tool_sequence_length"] = max(sequence_lengths)
                aggregated["min_tool_sequence_length"] = min(sequence_lengths)
            else:
                aggregated["avg_tool_sequence_length"] = 0
                aggregated["max_tool_sequence_length"] = 0
                aggregated["min_tool_sequence_length"] = 0
                
        else:
            # Set default values when no trajectories
            aggregated.update({
                "avg_turns_per_trajectory": 0,
                "avg_tool_calls_per_trajectory": 0,
                "avg_unique_tools_per_trajectory": 0,
                "max_unique_tools_per_trajectory": 0,
                "min_unique_tools_per_trajectory": 0,
                "total_unique_tools_used": 0,
                "tool_diversity_score": 0,
                "avg_tool_calls_by_type": {},
                "tool_usage_std": 0,
                "tool_usage_median": 0,
                "tool_usage_percentile_75": 0,
                "tool_usage_percentile_25": 0,
                "avg_tool_sequence_length": 0,
                "max_tool_sequence_length": 0,
                "min_tool_sequence_length": 0,
            })
        
        return aggregated
    
    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(OmegaConf.select(self.config.trainer, "worker_nsight_options"))

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)
    
    def normalize_answer_format(self, data_proto):
        if 'answer' in data_proto.non_tensor_batch:
            answer = data_proto.non_tensor_batch['answer']
            
            # 如果是嵌套结构 (1, N)，展开为 (N,)
            if len(answer.shape) == 2 and answer.shape[0] == 1:
                answer = answer[0]
            
            # 确保每个元素都是list格式
            normalized = []
            for item in answer.flat:
                if isinstance(item, list):
                    normalized.append(item)
                else:
                    normalized.append([str(item)])
            
            data_proto.non_tensor_batch['answer'] = np.array(normalized, dtype=object)
        
        return data_proto

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            #pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Log filter variables
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        timing_raw = {}

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                # timing_raw = {}
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                
                # 检查 batch_dict 中的 index 字段
                print("=== Debug batch_dict ===")
                if 'index' in batch_dict:
                    print(f"batch_dict['index']: {batch_dict['index']}")

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_inputs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
                if "raw_prompt" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = new_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        
                        # Process tool metrics from rollout data
                        tool_metrics = self._process_tool_metrics_from_rollout(gen_batch_output)
                        if tool_metrics["total_tool_calls"] > 0:
                            # Core tool metrics
                            metrics.update({
                                f"tools/total_calls": tool_metrics["total_tool_calls"],
                                f"tools/avg_calls_per_traj": tool_metrics["avg_tool_calls_per_trajectory"],
                                f"tools/avg_turns_per_traj": tool_metrics["avg_turns_per_trajectory"],
                                f"tools/avg_unique_tools_per_traj": tool_metrics["avg_unique_tools_per_trajectory"],
                                f"tools/max_unique_tools_per_traj": tool_metrics["max_unique_tools_per_trajectory"],
                                f"tools/total_unique_tools_used": tool_metrics["total_unique_tools_used"],
                                f"tools/tool_diversity_score": tool_metrics["tool_diversity_score"],
                            })
                            
                            # Add per-tool metrics
                            for tool_name, count in tool_metrics["tool_calls_by_type"].items():
                                metrics[f"tools/{tool_name}_total_calls"] = count
                                metrics[f"tools/{tool_name}_avg_per_traj"] = tool_metrics["avg_tool_calls_by_type"].get(tool_name, 0)
                            
                            # Track global cumulative metrics
                            self.global_tool_metrics["total_steps"] += 1
                            self.global_tool_metrics["cumulative_tool_calls"] += tool_metrics["total_tool_calls"]
                            self.global_tool_metrics["cumulative_trajectories"] += len(tool_metrics["trajectory_tool_usage"])
                            
                            # Update per-step tool usage tracking
                            step_tool_usage = {}
                            for tool_name, count in tool_metrics["tool_calls_by_type"].items():
                                step_tool_usage[tool_name] = count
                                if tool_name not in self.global_tool_metrics["tool_usage_per_step"]:
                                    self.global_tool_metrics["tool_usage_per_step"][tool_name] = []
                                self.global_tool_metrics["tool_usage_per_step"][tool_name].append(count)
                            
                            # Add cumulative metrics to tracking
                            metrics.update({
                                f"tools/cumulative_calls": self.global_tool_metrics["cumulative_tool_calls"],
                                f"tools/cumulative_trajectories": self.global_tool_metrics["cumulative_trajectories"],
                                f"tools/avg_calls_per_step": round(self.global_tool_metrics["cumulative_tool_calls"] / self.global_tool_metrics["total_steps"], 2),
                            })
                            
                            # Store detailed metrics for this step
                            self.tool_metrics_history.append({
                                "step": self.global_steps,
                                "metrics": tool_metrics,
                                "step_tool_usage": step_tool_usage
                            })
                            
                            # Log comprehensive summary to console every N steps
                            if self.global_steps % 10 == 0:  # Log every 10 steps
                                print(f"\n=== Detailed Tool Usage Statistics (Step {self.global_steps}) ===")
                                print(f"Total tool calls this step: {tool_metrics['total_tool_calls']}")
                                print(f"Avg tool calls per trajectory: {tool_metrics['avg_tool_calls_per_trajectory']}")
                                print(f"Avg turns per trajectory: {tool_metrics['avg_turns_per_trajectory']}")
                                print(f"Avg unique tools per trajectory: {tool_metrics['avg_unique_tools_per_trajectory']}")
                                print(f"Max unique tools in any trajectory: {tool_metrics['max_unique_tools_per_trajectory']}")
                                print(f"Tool diversity score: {tool_metrics['tool_diversity_score']:.3f}")
                                print(f"--- Per-Tool Breakdown ---")
                                for tool_name, avg_count in tool_metrics["avg_tool_calls_by_type"].items():
                                    total_count = tool_metrics['tool_calls_by_type'][tool_name]
                                    print(f"  {tool_name}: {total_count} total, {avg_count} avg/traj")
                                print(f"--- Cumulative Stats ---")
                                print(f"Total cumulative calls: {self.global_tool_metrics['cumulative_tool_calls']}")
                                print(f"Total trajectories processed: {self.global_tool_metrics['cumulative_trajectories']}")
                                print(f"Avg calls per training step: {metrics['tools/avg_calls_per_step']}")
                                print("==============================================================\n")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = new_batch.non_tensor_batch.get('index', np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object))

                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(new_batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                        new_batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if self.config.actor_rollout_ref.rollout.n == 1:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        print("std val ", new_batch.non_tensor_batch[metric_name])
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        # Whether to filter unsatified trajectories
                        if self.config.algorithm.filter_groups.enable:
                            kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                            # kept_prompt_uids = []
                            # for uid, std in prompt_uid2metric_std.items():
                            #     if std > 0:
                            #         print(f"UID {uid}: 保留 - 标准差 > 0 (std={std:.4f})")
                            #         kept_prompt_uids.append(uid)
                            #     elif len(prompt_uid2metric_vals[uid]) == 1:
                            #         print(f"UID {uid}: 保留 - 只有一个数据点")
                            #         kept_prompt_uids.append(uid)
                            #     else:
                            #         print(f"UID {uid}: 丢弃 - 标准差为0且有多个数据点")
                            # print("len(prompt_uid2metric_vals[uid]) - ", len(prompt_uid2metric_vals[uid]))
                            print("len(kept_prompt_uids) - ", len(kept_prompt_uids))
                            if self.config.curriculum_learning.enable:
                                # Curriculum Learning
                                abandon_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std == 0]
                                total_abandon_award = 0
                                total_abandon_query = 0
                                for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                                    if uid in abandon_prompt_uids:
                                        total_abandon_award += metric_val
                                        total_abandon_query += 1
                                print(f"[INFO] The {num_gen_batches}th generation, a total of {self.config.data.gen_batch_size} queries, "
                                    f"abandon {total_abandon_query//self.config.actor_rollout_ref.rollout.n} queries, "
                                    f"total score of abandoned queries: {total_abandon_award//self.config.actor_rollout_ref.rollout.n}, "
                                    f"difficulty rate: {1.0 - total_abandon_award/total_abandon_query}")
                                
                                directory_path = os.path.join(self.config.trainer.default_local_dir, f"curriculum_learning", f"epoch_{epoch}")
                                os.makedirs(directory_path, exist_ok=True)

                                difficult_file_path = os.path.join(directory_path, "difficult_query.txt")
                                with open(difficult_file_path, 'a', encoding='utf-8') as f:
                                    for uid, std in prompt_uid2metric_std.items():
                                        if std == 0 and prompt_uid2metric_vals[uid][0] == 0:
                                            f.write(f"{uid}\n")
                                
                                easy_file_path = os.path.join(directory_path, "easy_query.txt")
                                with open(easy_file_path, 'a', encoding='utf-8') as f:
                                    for uid, std in prompt_uid2metric_std.items():
                                        if std == 0 and prompt_uid2metric_vals[uid][0] == 1:
                                            f.write(f"{uid}\n")
                        else:
                            kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items()]
            
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        # 在报错的代码行前添加调试信息
                        print("=== DataProto Concat Debug ===")
                        print(f"batch is None: {batch is None}")

                        # 调试 new_batch
                        print(f"new_batch length: {len(new_batch)}")
                        if new_batch.batch is not None:
                            print(f"new_batch.batch.batch_size: {new_batch.batch.batch_size}")
                            print(f"new_batch.batch keys: {list(new_batch.batch.keys())}")
                        else:
                            print("new_batch.batch: None")

                        if new_batch.non_tensor_batch:
                            print(f"new_batch.non_tensor_batch keys: {list(new_batch.non_tensor_batch.keys())}")
                        else:
                            print("new_batch.non_tensor_batch: empty")

                        # 调试 batch (如果不为 None)
                        if batch is not None:
                            print(f"batch length: {len(batch)}")
                            if batch.batch is not None:
                                print(f"batch.batch.batch_size: {batch.batch.batch_size}")
                                print(f"batch.batch keys: {list(batch.batch.keys())}")
                            else:
                                print("batch.batch: None")
                            
                            if batch.non_tensor_batch:
                                print(f"batch.non_tensor_batch keys: {list(batch.non_tensor_batch.keys())}")
                            else:
                                print("batch.non_tensor_batch: empty")
                            
                            # 检查维度不匹配
                            print("--- Dimension Mismatch Check ---")
                            if batch.batch is not None and new_batch.batch is not None:
                                for key in set(batch.batch.keys()) | set(new_batch.batch.keys()):
                                    if key in batch.batch and key in new_batch.batch:
                                        old_shape = batch.batch[key].shape
                                        new_shape = new_batch.batch[key].shape
                                        if len(old_shape) != len(new_shape):
                                            print(f"⚠️ TENSOR MISMATCH '{key}': {old_shape} vs {new_shape}")                     
                            
                            for key in set(batch.non_tensor_batch.keys()) | set(new_batch.non_tensor_batch.keys()):
                                if key in batch.non_tensor_batch and key in new_batch.non_tensor_batch:
                                    old_shape = batch.non_tensor_batch[key].shape
                                    new_shape = new_batch.non_tensor_batch[key].shape
                                    if len(old_shape) != len(new_shape):
                                        print(f"⚠️ NON_TENSOR MISMATCH '{key}': {old_shape} vs {new_shape}")
                                        print(f"new_batch answer sample: {new_batch.non_tensor_batch['answer'][:3]}")
                                        print(f"batch answer sample: {batch.non_tensor_batch['answer'][:3]}")
                                        new_batch.non_tensor_batch[key] = [item[0] for item in new_batch.non_tensor_batch[key]]
                                        batch.non_tensor_batch[key] = [item[0] for item in batch.non_tensor_batch[key]]
                                        # new_batch = self.normalize_answer_format(new_batch) 
                        
                                       
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === CONTINUE WITH NORMAL TRAINING FLOW ===
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data ipn the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    # if self.config.trainer.balance_batch:
                    #     self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # compute advantages, executed on the driver process
                        if self.config.algorithm.use_kl_in_reward and self.config.actor_rollout_ref.rollout.n == 1:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                timing_raw = {} # reset time log

                batch = None
                
                if self.config.actor_rollout_ref.rollout.n > 1:
                    metrics["train/num_gen_batches"] = num_gen_batches
                    num_prompt_in_batch = 0
                    num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
