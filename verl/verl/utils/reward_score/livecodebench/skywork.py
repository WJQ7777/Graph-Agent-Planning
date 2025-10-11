# Copyright 2024 PRIME team and/or its affiliates
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

import json
import re
import traceback
from verl.utils.reward_score.livecodebench.unit_test import lcb_compute_score, prepare_unit_test_data
import os, pickle
from verl.utils.reward_score.livecodebench.lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics, check_correctness
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.pass_k_utils import extract_instance_results
from math_verify import parse, verify
import tempfile
import subprocess
from contextlib import contextmanager
import signal
import ast
import numpy as np
from verl.tools.utils.code_executors.utils import remove_from_solution_line,extract_zero_arg_functions,remove_main_block,parse_code_blobs,try_extract_solution,extract_last_code_from_string


IMPORT_PROMPT='''from typing import *

from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
import math
import datetime
inf = float('inf')

'''

livecodebench_dir = os.environ.get("LIVECODEBENCH_DATA_PATH", None)
# If you want to use the LiveCodeBench evaluation, please make sure that you have downloaded the corresponding test cases to the correct location.
# if livecodebench_dir is None:
#     raise ValueError("LIVECODEBENCH_DATA_PATH is not set")


@contextmanager
def timeout_run(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Code execution timed out")
    
    # Register signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

def convert_function_to_class_method(raw_code: str, function_name: str) -> str:
    # Parse raw code into AST
    tree = ast.parse(raw_code)
    target_func = None
    new_body = []
    # Traverse top-level nodes, keeping code that is not the target function
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target_func = node
        else:
            new_body.append(node)
    
    if target_func is None:
        return None

    if not (target_func.args.args and target_func.args.args[0].arg == "self"):
        self_arg = ast.arg(arg="self", annotation=None)
        target_func.args.args.insert(0, self_arg)    
    class_def = ast.ClassDef(
        name="Solution",
        bases=[],
        keywords=[],
        body=[target_func],
        decorator_list=[]
    )
    
    new_body.append(class_def)
    tree.body = new_body
    
    # Use ast.unparse to convert AST to code string (Python 3.9+ supported)
    new_code = ast.unparse(tree)
    return new_code


def math_verify_reward_function(solution_str, ground_truth):

    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return 0.0



def compute_score(index, 
                  solution_str, 
                  ground_truth, 
                  task=None, 
                  timeout=6, 
                  is_long_penalty=False, 
                  is_binary_reward=True, 
                  is_power4_reward=False,
                  format_score=0.0, 
                  answer_reward=1.,):
    ground_truth = json.loads(ground_truth)

    # Entering math task
    if isinstance(ground_truth, list) :
        assert len(ground_truth) > 0, f"In skywork_math's datasource, reward_model field's ground_truth field has no value for index {index}"
        extracted_solution_str = try_extract_solution(solution_str)
        if extracted_solution_str:
            try:
                return math_verify_reward_function(extracted_solution_str, ground_truth), solution_str, None, index
            except:
                # traceback.print_exc(10)
                return 0.0, solution_str, "math_verify_reward_function(extracted_solution_str, ground_truth)执行失败", index
        else:
            return 0.0, solution_str, "没有从数学task的solution_str中提取到<answer></answer>的tag", index

    # Entering code task
    assert isinstance(ground_truth, dict), f"In skywork_code's datasource, when computing score, ground_truth must be dict type, failed data is {index}"
    
    # Extract the answer code segment and add delimiters for subsequent interface use
    try:
        solution_str = f"```python\n" + extract_last_code_from_string(solution_str) + f"\n```"
    except Exception as e:
        return format_score, None, None, index

    if "question_id" in ground_truth and ground_truth["question_id"]:
        try:
            benchmark = pickle.load(open(os.path.join(livecodebench_dir, "{}.pkl".format(ground_truth["question_id"])), "rb"))
            custom_output = ground_truth.copy()
            custom_output["output_list"] = [solution_str]
            return lcb_compute_score([custom_output], [benchmark]), solution_str, "Correctly returned score", index
        except:
            # traceback.print_exc(10)
            return 0.0, None, "Failed to calculate LCB score", index
    
    elif 'import_prefix' in ground_truth and ground_truth['import_prefix']:
        solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if len(solutions) == 0:
            return 0.0, None, f"No delimiters found in solution_str", index
        try:
            solution = solutions[-1]
            tree = ast.parse(solution)
            solution = ground_truth["import_prefix"] + solution

            test_code = [x for x in ground_truth['test_code'].split("\n") if x != ""]

        
            unit_test_result = []
            unit_test_metadata = []
            for i in range(1, len(test_code)):
                cur_solution = solution
                cur_solution += "\n" + test_code[0] + test_code[i]
                cur_solution += "\ncheck({})".format(ground_truth['entry_point'])

                try:
                    success = False
                    message = None
                    with timeout_run(seconds=2):
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as temp_file:
                            temp_file.write(cur_solution)
                            temp_file.flush()
                            result = subprocess.run(
                                ['python', temp_file.name],
                                capture_output=True,
                                text=True,
                                timeout=timeout
                            )
                            if result.returncode != 0:
                                unit_test_result.append(False)
                                unit_test_metadata.append(f"Execution error: {result.stderr}")
                            else:
                                unit_test_result.append(True)
                                unit_test_metadata.append(f"Success")
                except TimeoutError:
                    # traceback.print_exc(10)
                    unit_test_result.append(False)
                    unit_test_metadata.append("Code execution timed out")
                except Exception as e:
                    unit_test_result.append(False)
                    unit_test_metadata.append("Execution exception")
                    
            if is_binary_reward:
                final_score = 1.0 if all(unit_test_result) else 0.0
                return final_score, solution_str, unit_test_metadata, index
            else:
                if is_power4_reward:
                    return (sum(unit_test_result)/len(unit_test_result))**4, solution_str, unit_test_metadata, index
                else:
                    return sum(unit_test_result)/len(unit_test_result), solution_str, unit_test_metadata, index

        except Exception as e:
            # traceback.print_exc(10)
            return 0.0, solution_str, f"Code parsing error: {str(e)}", index

    elif "inputs" in ground_truth and ground_truth["inputs"]:
        try:
            solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
            if len(solutions) == 0 :
                return 0.0, solution_str, f"No delimiters were parsed out of the solution_str", index
            else:
                solution = solutions[-1]
                try:
                    tree = ast.parse(solution)
                except:
                    # traceback.print_exc(10)
                    return 0.0, solution_str, traceback.format_exc(), index
            if isinstance(ground_truth, str):
                input_output = json.loads(ground_truth)
            elif isinstance(ground_truth, dict):
                input_output = ground_truth
                ground_truth = json.dumps(ground_truth)
                
            else:
                assert False
            if "fn_name" in input_output and input_output["fn_name"] and "class Solution" not in solution:
                solution = convert_function_to_class_method(solution, input_output["fn_name"])
                if not isinstance(solution, str):
                    return 0.0, solution, "solution is not a string", index
            
            metrics = check_correctness(
                {"input_output":ground_truth},
                solution,
                debug=False,
                timeout=timeout,
            )

            metrics = list(metrics)
            fixed = []
            for e in metrics[0]:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            metrics[0] = fixed

            if is_binary_reward:
                final_score = 1.0 if sum(metrics[0]) == len(metrics[0]) else 0.0
                return final_score, solution_str, metrics, index
            else:
                if is_power4_reward:
                    return (sum((x if x in [False, True] else False) for x in metrics[0])/len(metrics[0]))**4, solution_str, metrics, index
                else:
                    return sum((x if x in [False, True] else False) for x in metrics[0])/len(metrics[0]), solution_str, metrics, index

        except Exception as e:
            # traceback.print_exc(10)
            return 0.0, solution_str, traceback.format_exc(), index

    elif "assert_case" in ground_truth:
        raise ValueError("In data_source=skywork dataset, there are data whose reward_model.ground_truth has neither import_prefix field nor inputs field")
        solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if len(solutions) == 0:
            return False, None
        try:
            solution = solutions[-1]
            tree = ast.parse(solution)

            test_code = ground_truth['assert_case']
            unit_test_result = []
            unit_test_metadata = []
            for i in range(0, len(test_code)):
                cur_solution = solution
                cur_solution += "\n" + test_code[i]
                cur_solution = IMPORT_PROMPT + cur_solution

                try:
                    success = False
                    message = None
                    with timeout_run(seconds=2):
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as temp_file:
                            temp_file.write(cur_solution)
                            temp_file.flush()
                            result = subprocess.run(
                                ['python', temp_file.name],
                                capture_output=True,
                                text=True,
                                timeout=timeout
                            )
                            if result.returncode != 0:
                                unit_test_result.append(False)
                                unit_test_metadata.append(f"Execution error: {result.stderr}")
                            else:
                                unit_test_result.append(True)
                                unit_test_metadata.append(f"Success")
                except TimeoutError:
                    print("Code execution timed out")
                    traceback.print_exc(10)
                    unit_test_result.append(False)
                    unit_test_metadata.append("Code execution timed out")
                except Exception as e:
                    print(f"Execution exception: {str(e)}")
                    unit_test_result.append(False)
                    unit_test_metadata.append("Execution exception")
                    
            if is_binary_reward:
                return all(unit_test_result), unit_test_metadata
            else:
                if is_power4_reward:
                    return (sum(unit_test_result)/len(unit_test_result))**4, unit_test_metadata
                else:
                    return sum(unit_test_result)/len(unit_test_result), unit_test_metadata

        except Exception as e:
            traceback.print_exc(10)
            return False, f"Code parsing error: {str(e)}"

    else:
        raise ValueError("In data_source=skywork_code dataset, there are data whose reward_model.ground_truth has neither import_prefix field nor inputs field")
        try:
            return math_verify_reward_function(solution_str, ground_truth), None
        except:
            traceback.print_exc(10)
            return False, None
