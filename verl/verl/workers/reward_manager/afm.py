#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
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

import datetime
from collections import defaultdict
from verl.utils.reward_score import qa_em, codeforces, mathverify, grm_simple
from verl.utils.reward_score.livecodebench import skywork
import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.reward_score.qa_em import extract_solution
from verl.utils.reward_score.length_penalty import calculate_response_length_penalty
from verl.utils.reward_score.format_reward import FormatReward

import re
import numpy as np
import json
import time
import ray
from typing import Dict, Any
import logging
import os
import torch
from filelock import FileLock
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from verl.workers.reward_manager import register
def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle','mercury', 'tq']:
        return qa_em.compute_score_em
    elif data_source in ['math_dapo','aime24','aime25','MATH','MATH500','gms8k','amc23','OlympiadBench']:
        return mathverify.compute_score
    elif data_source in ['livecodebench','lcbv5','lcbv4']:
        return lcb.compute_score
    elif data_source in ['codeforces']:
        return codeforces.compute_score
    elif data_source in ['GPQA_dimand']:
        return gpqa.compute_score
    elif data_source in ['skywork', 'skywork_livecodebench_v4', 'skywork_livecodebench_v6', 'skywork_livecodebench_v5', 'skywork_livecodebench_v1_to_v3', 'skywork_math', "codeforces_areal_test", "codecontests_hf_test", "codecontests_hf_train","codecontests_areal_test"]:
        return skywork.compute_score
    elif data_source in ['search']:
        return grm_simple.compute_score
    else:
        print('[_select_rm_score_fn]',data_source,' reward fnx not found')
        raise NotImplementedError


def multi_compute_score(
    response_str, 
    ground_truth, 
    valid_response_length, 
    data_source,
    is_val: bool = False,
    format_fn_type = 'no_penalty', 
    context_max_response_length: int = 8192, 
    idx=None,
    output_file_dir: str = None
    ):
    compute_score_fn = _select_rm_score_fn(data_source)
    if is_val:
        score, code, output, index = compute_score_fn(
            index=idx,
            solution_str=response_str,
            ground_truth=ground_truth,
            format_score=0,
        )
    elif format_fn_type=='no_penalty':
        score, code, output, index = compute_score_fn(
            index=idx,
            solution_str=response_str,
            ground_truth=ground_truth,
            format_score=0,
        )
    elif format_fn_type=='overlong_buffer':
        score, code, output, index = compute_score_fn(
            index=idx,
            solution_str=response_str,
            ground_truth=ground_truth,
            format_score=0,
        )
        length_penalty = calculate_response_length_penalty(valid_response_length, context_max_response_length)
        score += length_penalty
    elif format_fn_type=='overlong_buffer_toolstar':
        score, code, output, index = compute_score_fn(
            index=idx,
            solution_str=response_str,
            ground_truth=ground_truth,
            format_score=0,
        )
        length_penalty = calculate_response_length_penalty(valid_response_length, context_max_response_length)
        score += length_penalty
        format_checker = FormatReward()
        if not format_checker.check_format_correctness(response_str):
            score = -1.0
    elif format_fn_type=='overlong_buffer_codetag':
        score, code, output, index = compute_score_fn(
            index=idx,
            solution_str=response_str,
            ground_truth=ground_truth,
            format_score=0,
        )
        length_penalty = calculate_response_length_penalty(valid_response_length, context_max_response_length)
        score += length_penalty
        format_checker = FormatReward()
        if not format_checker.check_code_format_correctness(response_str):
            score = 0.0
    else:
        raise NotImplementedError("During training, please specify a method for computing the format score.")
    
    # write to files
    if output_file_dir:
        output_file = os.path.join(output_file_dir, f"{data_source}_.jsonl")
        with FileLock(output_file + ".lock"):
            with open(output_file, "a") as f:
                entry = {
                    "solution_str":response_str,
                    "index": index,
                    "code": code,
                    "output": output,
                    "score": score,
                    "ground_truth":ground_truth,
                }
                f.write(json.dumps(entry,ensure_ascii=False) + "\n")

    return score

def parallel_compute_score(
    evaluation_func, 
    response_str, 
    ground_truth, 
    valid_response_length, 
    data_sources, 
    idx,
    is_val: bool = False,
    format_fn_type = 'no_penalty',
    context_max_response_length: int = 8192, 
    timeout=3, 
    max_workers=48,
    output_file_dir=None
    ):

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluation_func, 
                response_str[index], 
                ground_truth[index], 
                valid_response_length[index], 
                data_sources[index], 
                is_val, 
                format_fn_type, 
                context_max_response_length,
                idx[index],
                output_file_dir
            ): index
            for index in range(len(response_str))
        }
        results = {}
        metadata = {}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()

    return [results[i] for i in range(len(response_str))]



@register("afm")
class AFMRewardManager:
    def __init__(self, tokenizer, num_examine, config=None, format_score=0.,output_file_dir=None,is_valid=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.output_file_dir = output_file_dir
        self.is_val = is_valid
        self.config = config

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        start_time = time.time()
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch["responses"]

        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth']  for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        idx = data.non_tensor_batch['index']

        assert len(response_str) == len(ground_truth) == len(data_sources)

        scores = parallel_compute_score(
                multi_compute_score,
                response_str,
                ground_truth,
                valid_response_length,
                data_sources,
                idx,
                is_val = self.is_val,
                format_fn_type = self.config.get('format_fn_type', 'no_penalty'),
                context_max_response_length = self.config.get('context_max_response_length', 8192),
                output_file_dir=self.output_file_dir
            )
        
        assert len(scores) == len(response_str)



        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        # print computational time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[Reward] len(data): {len(data)}, total time: {elapsed_time:.3f}s")

        reward_extra_info = defaultdict(list)
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
