# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Generate responses given a dataset of prompts
"""

import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
import os
from pathlib import Path
import json
@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    # Used for multimodal LLM, could be None
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # Create the same dataloader with trainer
    dataset = RLHFDataset(
        data_files=config.data.path, 
        tokenizer=tokenizer, 
        processor=processor, 
        config=config.data
    )
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        shuffle=config.data.get("shuffle", True),
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Load Rollout Worker
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    # store outputs
    sample_trajectories = []
    sample_data_sources = []
    sample_indexs = []

    # Main loop
    for batch_dict in dataloader:
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        # generate
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, wg.world_size)
        gen_batch_output_padded = wg.generate_sequences(gen_batch_padded)
        gen_batch_output = unpad_dataproto(gen_batch_output_padded, pad_size=pad_size)
        batch.union(gen_batch_output)

        # log output
        trajectory_ids = batch.batch["input_ids"]
        trajectory_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in trajectory_ids]
        sample_trajectories.extend(trajectory_texts)
        sample_data_sources.extend(batch.non_tensor_batch["data_source"].tolist())
        sample_indexs.extend(batch.non_tensor_batch["index"].tolist())

    # Write to files
    output_dir = os.path.join(config.data.output_path, config.data.experiment_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(len(trajectory_texts)):
        output_file = os.path.join(output_dir, f"{sample_data_sources[i]}.jsonl")
        with open(output_file, "a") as f:
            entry = {
                "trajectories":sample_trajectories[i],
                "index": sample_indexs[i],
            }
            f.write(json.dumps(entry,ensure_ascii=False) + "\n")

    print(f"[End] Rollout has done! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
