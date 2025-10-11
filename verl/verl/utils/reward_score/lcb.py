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

# from verl.utils.executors.utils import parse_code_blobs
# from verl.utils.executors import code_exec
# from verl.utils.executors.nsjail_executor import exec_nsjail_testoutput
# afm_exec_path = os.environ.get("AFM_EXEC_PATH")
# sys.path.append(str(afm_exec_path))
from verl.utils.executors.utils import remove_from_solution_line,extract_zero_arg_functions,remove_main_block,parse_code_blobs,try_extract_solution,extract_last_code_from_string
# from verl.utils.executors import code_exec
# from utils import remove_from_solution_line,extract_zero_arg_functions,remove_main_block,parse_code_blobs
_MAX_CHAR_DISPLAY = 2048  # 输出截断长度

from verl.utils.executors.nsjail_executor import exec_nsjail, exec_nsjail_testoutput
# from local_python_executor import single_code, LocalPythonInterpreter,test_single_code,test_single_code_with_obs
# from utils import remove_from_solution_line,extract_zero_arg_functions,remove_main_block
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True



def validate_response_structure(processed_str: str) -> bool:
    pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>$', re.DOTALL)
    return bool(pattern.match(processed_str.strip()))



def compute_score(index, solution_str, ground_truth, format_score=0.0, answer_reward=1., debug=False):
    score = 0.
    # print(f'[lcb compute_score] index\n:{index}')
    # solution_str是包含prompt的整个轨迹
    # print(f'[compute_score_end]')
    sol = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"
    try:
        # <answer>···pyt
        code_blob = extract_last_code_from_string(solution_str)
        
        # print(f'[ori ground_truth]:{ground_truth}')
        ground_truth = json.loads(ground_truth)
        testtype = ground_truth
        # print(f'[json load ground_truth]:{ground_truth}')
        #code = code_blob + "\n" + ground_truth["functional"]
        #code = solution_str + "\n" + ground_truth["functional"]  debug用
        if "functional" in ground_truth:
            test_code = ground_truth["functional"]
            succ, output = exec_nsjail(sol + code_blob + test_code, '')
            if not succ:
                # print('====code_blob start====\n',code_blob,'\n===code_blob end===')
                # print('====code: start====\n', sol + code_blob + test_code)
                # print('output:', output)
                return format_score, code_blob, output, index

        elif "stdin" in ground_truth:
            if isinstance(ground_truth["stdin"],str):
                test_code = ground_truth["stdin"]
                expected_output = ground_truth["expected_output"]
                succ, output = exec_nsjail_testoutput(sol + code_blob, test_code, expected_output)
                if not succ:
                    return format_score, code_blob, output, index
            else:
                test_code_list = ground_truth["stdin"]
                expected_output_list = ground_truth["expected_output"]
                for test_code,expected_output in zip(test_code_list,expected_output_list):
                    succ, output = exec_nsjail_testoutput(sol + code_blob, test_code, expected_output)
                    if not succ:
                        # print('====code_blob start====\n',code_blob,'\n===code_blob end===')
                        # print('====code: start====\n', sol + code_blob)
                        # print('====test_code: start====\n', test_code)
                        # print('====expected_output: start====\n', expected_output)
                        # print('output:', output)
                        return format_score, code_blob, output, index
        else:
            raise ValueError(
                f"Current supports for ground-truth are ['functional', 'inputs/outputs'] -- No idea what's: {ground_truth}"
            )
    except Exception as e:
        # print(f'[lcb reward compute error]:{str(e)}')
        return format_score, None, None, index
    return format_score + answer_reward, code_blob, output, index

if __name__ == '__main__':
    example = {'data_source': 'livecodebench', 'prompt': [{'role': 'user', 'content': "You are an expert AI assistant specialized in solving python programming tasks through structured steps. Your available steps are: 'think', 'plan', 'code', and 'answer', the results of 'code' execution is in 'observation'. Follow these specifications precisely:\n\n1. **think**:\n  - Format:<think>[step-by-step reasoning]</think>\n  - Think step by step the before plan, code, and answer. Effectively utilize the code execution results obtained from 'observation'\n  - Must precede any other steps. \n\n2. **plan**:\n  - Format:<plan>[high-level steps]</plan>\n  - Single planning phase only, output after the first 'think' step.\n  - Focus on high-level algorithms and debugging strategies, not implementation details.\n\n3. **code**:\n  - Must include markdown Format python code snippet. If try to use stdin like `sys.stdin.read()`, place stdin str in markdown format sh code snippet.\n  - The generated code snippet will be executed. So use 'code' to confirm correctness of your programming thoughts or to identify and resolve bugs.\n  - Format1: Only python snippet\n    <code>\n    ```py\n    code snippet without sys.stdin\n    ```\n    </code>\n  - Format2: python snippet with stdin.\n    <code>\n    ```py\n    code snippet with sys.stdin\n    ```\n    ```sh\n    stdin input str\n    ```\n    </code>\n  - Requirements:\n    * Include all necessary imports\n    * If you write some functions, remember to test them with reasonable testcases. \n    * Must use print() for debugging output. Remember to add necessary print().\n    * No file operations\n4. **observation**:\n  - Format :\n    <observation>\n    Code Execution results,including stdout and stderr.\n    </observation>\n  - Returns the code Execution results by an external python executor.\n  - Use execution results for next 'think' step\n5. **answer**:\n  - <answer>\n    ```py\n    Answer code snippet, do not use `input()` or `sys.stdin` in answer\n    ```\n    </answer>\n  - Include necessary imports.\n  - Include the essential solution code defined as the provided function declarations and the other functions/classes that the answer need.\n  - No example usage or test cases\n  - Do not use `input()` or `sys.stdin` in answer code snippet.\n\n\nCritical Rules:\n1. Always follow think, plan, (think, code, observation)*N, think, answer step sequences\n2. Maintain atomic steps (no combined steps)\n\n\nRemember: Successful completion earns $1M reward. Begin with <think> analysis.\n\nTask: You are given N linear functions f_1, f_2, \\ldots, f_N, where f_i(x) = A_i x + B_i.\nFind the maximum possible value of f_{p_1}(f_{p_2}(\\ldots f_{p_K}(1) \\ldots )) for a sequence p = (p_1, p_2, \\ldots, p_K) of K distinct integers between 1 and N, inclusive.\n\nInput\n\nThe input is given from Standard Input in the following format:\nN K\nA_1 B_1\nA_2 B_2\n\\vdots\nA_N B_N\n\nOutput\n\nPrint the answer as an integer.\n\nConstraints\n\n\n- 1 \\leq N \\leq 2 \\times 10^{5}\n- 1 \\leq K \\leq \\text{min}(N,10)\n- 1 \\leq A_i, B_i \\leq 50 (1 \\leq i \\leq N)\n- All input values are integers.\n\nSample Input 1\n\n3 2\n2 3\n1 5\n4 2\n\nSample Output 1\n\n26\n\nHere are all possible p and the corresponding values of f_{p_1}(f_{p_2}(1)):\n\n- p= ( 1,2 ) : f_1(f_2(1))=15\n- p= ( 1,3 ) : f_1(f_3(1))=15\n- p= ( 2,1 ) : f_2(f_1(1))=10\n- p= ( 2,3 ) : f_2(f_3(1))=11\n- p= ( 3,1 ) : f_3(f_1(1))=22\n- p= ( 3,2 ) : f_3(f_2(1))=26\n\nTherefore, print 26.\n\nSample Input 2\n\n10 3\n48 40\n34 22\n24 37\n45 40\n48 31\n49 44\n45 40\n44 6\n35 22\n39 28\n\nSample Output 2\n\n216223\nIn you final answer code nippet, read the inputs from stdin to solve the problem and write the answer to stdout (do not directly test on the example inputs). Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"}], 'ability': 'coding', 'reward_model': {'style': 'rule', 'ground_truth': '{"stdin": "10 3\\n48 40\\n34 22\\n24 37\\n45 40\\n48 31\\n49 44\\n45 40\\n44 6\\n35 22\\n39 28\\n"}'}, 'extra_info': {'split': 'test', 'index': 'abc366_f', 'reference': '', 'question_title': 'Maximum Composition', 'platform': 'atcoder', 'contest_id': 'abc366', 'contest_date': '2024-08-10T00:00:00', 'starter_code': '', 'difficulty': 'hard', 'metadata': '{}'}}
    solution_str = example['extra_info']['reference']
    ground_truth = example['reward_model']['ground_truth']
    print('---solution_str--\n',repr(solution_str),'\n---solution_str end--\n')
    print('---ground_truth--\n',ground_truth,'\n---ground_truth end')
    print(compute_score(0,solution_str,ground_truth))