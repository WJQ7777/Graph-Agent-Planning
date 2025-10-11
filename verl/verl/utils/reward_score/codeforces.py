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
import sys
import numpy as np
import datasets

from verl.tools.utils.code_executors.utils import remove_from_solution_line,extract_zero_arg_functions,remove_main_block,parse_code_blobs,try_extract_solution,extract_last_code_from_string

_MAX_CHAR_DISPLAY = 2048  

from verl.tools.utils.code_executors.nsjail_executor_codeforces import exec_nsjail_testoutput
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

codeforces_testcases_path = os.environ.get("CODEFORCES_TESTCASES_PATH", None)
# If you want to use the CodeForces evaluation, please make sure that you have downloaded the corresponding test cases to the correct location.
# if codeforces_testcases_path is None:
#     raise ValueError("CODEFORCES_TESTCASES_PATH is not set")

codeforces_checker_file_path = os.environ.get("CODEFORCES_CHECKER_FILE_PATH", None)
# If you want to use the CodeForces evaluation, please make sure that you have downloaded the corresponding checker file to the correct location.
# if codeforces_checker_file_path is None:
#     raise ValueError("CODEFORCES_CHECKER_FILE_PATH is not set")

def validate_response_structure(processed_str: str) -> bool:
    pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>$', re.DOTALL)
    return bool(pattern.match(processed_str.strip()))

def get_generated_output(jsonl_path, target_index):
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['index'] == target_index:
                return data['generated_checker']
    return None

def compute_score(index, solution_str, ground_truth, format_score=0.0, answer_reward=1., debug=False):
    score = 0.
    sol = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"
    try:
        code_blob = extract_last_code_from_string(solution_str)
        ground_truth = json.loads(ground_truth)
    
        # Find full testcases
        pkl_dir = codeforces_testcases_path
        pkl_path = os.path.join(pkl_dir, f"{index}.pkl")

        if os.path.exists(pkl_path):
            # Load full testcases
            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f)
            if isinstance(ground_truth.get('stdin'), list) and isinstance(pkl_data.get('stdin'), list):
                ground_truth['stdin'].extend(pkl_data['stdin'])
            if isinstance(ground_truth.get('expected_output'), list) and isinstance(pkl_data.get('expected_output'), list):
                ground_truth['expected_output'].extend(pkl_data['expected_output'])
        
        testtype = ground_truth
        checker_file = codeforces_checker_file_path
        checker_code = get_generated_output(checker_file, index) 

        if isinstance(ground_truth["stdin"],str):
            test_code = ground_truth["stdin"]
            expected_output = ground_truth["expected_output"]
            succ, output = exec_nsjail_testoutput(sol + code_blob, test_code, expected_output, checker_code)
            if not succ:
                return format_score, code_blob, output, index
        else:
            test_code_list = ground_truth["stdin"]
            expected_output_list = ground_truth["expected_output"]
            for test_code,expected_output in zip(test_code_list,expected_output_list):
                succ, output = exec_nsjail_testoutput(sol + code_blob, test_code, expected_output, checker_code)
                if not succ:
                    return format_score, code_blob, output, index
    except Exception as e:
        return format_score, None, str(e), index
    return format_score + answer_reward, code_blob, output, index