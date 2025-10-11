# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

import multiprocessing
import os
from functools import partial

import ray

from verl import DataProto
from verl.utils.reward_score import default_compute_score


def get_custom_reward_fn(config, is_valid):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}

    if not is_valid:
        file_path = reward_fn_config.get("train_path")
    else:
        file_path = reward_fn_config.get("val_path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not is_valid:
        function_name = reward_fn_config.get("train_name")
    else:
        function_name = reward_fn_config.get("val_name")
    
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


def load_reward_manager(config, tokenizer, num_examine, is_valid, **reward_kwargs):
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """
    from verl.workers.reward_manager import get_reward_manager_cls

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    if reward_manager_name == 'afm':
        if is_valid:
            if config.trainer.get('val_only', False):
                reward_record_file_name = config.trainer.experiment_name
                reward_record_file_dir = 'evaluation/verl_val'
                reward_record_file_dir = os.path.join(reward_record_file_dir, f"{reward_record_file_name}")
                os.makedirs(reward_record_file_dir, exist_ok=True)
                output_file_dir = reward_record_file_dir
                # print()
                print(f'[val only] the val generations, code, test exec results will be output to dir:{output_file_dir}')
                # logging.info(f'[val only] the val generations, code, test exec results will be output to dir:{output_file_dir}')
            else:
                output_file_dir = None
            return reward_manager_cls(tokenizer, num_examine, is_valid=True, output_file_dir=output_file_dir, config=config.get('AFM_reward_fn', {}), **reward_kwargs)
        else:
            return reward_manager_cls(tokenizer, num_examine, config=config.get('AFM_reward_fn', {}), **reward_kwargs)

    else:
        # Try to get a custom reward function based on the configuration
        compute_score = get_custom_reward_fn(config, is_valid)
        final_compute_score = compute_score
        if final_compute_score is None:
            final_compute_score = default_compute_score
        return reward_manager_cls(tokenizer, num_examine, final_compute_score, **reward_kwargs)


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result["reward_extra_info"]
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)
