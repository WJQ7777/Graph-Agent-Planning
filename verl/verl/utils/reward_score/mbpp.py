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


import re
from typing import Tuple, Optional
import time
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from verl.utils.executors.utils import parse_code_blobs,extract_last_code_from_string
from verl.utils.executors import code_exec
_MAX_CHAR_DISPLAY = 2048  # 输出截断长度


def remote_check_stdio(code, stdin, stdout):
    # stdin 是测试用例的一部分（通过 ground_truth["inputs"] 提供），用于模拟用户输入
    # stdout 是测试用例中预期的正确输出（通过 ground_truth["outputs"] 提供）
    # succ（成功标志）​True/False，表示代码是否成功执行（无报错）
    # ​​执行成功时​​（succ = True）：output 是程序打印到控制台的内容（即 stdout 的实际值）。
    # 执行失败时​​（succ = False）：output 是错误信息（如 Traceback (most recent call last): ...）。代码中可能以 _ERROR_MSG_PREFIX 开头的错误提示（如 "ERROR: ..."）。
    python_executor = LocalPythonInterpreter(
                    additional_authorized_imports=[],
                    tools={},
                    max_print_outputs_length=None,)
    succ = test_single_code(python_executor, code)
    output = single_code(python_executor, code)
    # succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


def validate_response_structure(processed_str: str) -> bool:
    pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>$', re.DOTALL)
    return bool(pattern.match(processed_str.strip()))





def compute_score(index, solution_str, ground_truth, format_score=0.0, answer_reward=1., debug=False):
    score = 0.
    # print(f'[compute_score]solution_str\n:{solution_str}')
    # solution_str是包含prompt的整个轨迹
    # print(f'[compute_score_end]')
    try:
        # <answer>···pyt
        code_blob = extract_last_code_from_string(solution_str)
        
        ground_truth = json.loads(ground_truth)
        code = code_blob + "\n" + ground_truth["functional"]
        #code = solution_str + "\n" + ground_truth["functional"]  debug用
        if "functional" in ground_truth:
            succ, output = code_exec(code)
            # print('code:', code)
            # print('output:', output)
            if not succ:
                # print('code:', code)
                # print('output:', output)
                return format_score, code, output, index
        else:
            raise ValueError(
                f"Current supports for ground-truth are ['functional', 'inputs/outputs'] -- No idea what's: {ground_truth}"
            )
    except Exception as e:
        #print(f'[kodcode reward compute error]')
        return format_score, None, None, None
    return format_score + answer_reward, code, output, index