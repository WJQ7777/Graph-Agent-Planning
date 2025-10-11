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


import os
import subprocess
import sys
import tempfile
from pathlib import Path
from io import StringIO
import re
from unittest.mock import patch
import shutil
nsjail_path = os.environ.get("NSJAILPATH")
def get_conda_env_paths():
    """Get PYTHONPATH and LD_LIBRARY_PATH of the current conda environment"""
    python_exec = sys.executable
    if "conda" not in python_exec:
        raise RuntimeError("Not running in a conda environment")
    
    conda_env_path = str(Path(python_exec).parent.parent)
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    
    python_path = f"{conda_env_path}/lib/{python_version}/site-packages"
    ld_path = f"{conda_env_path}/lib:/lib:/usr/lib"
    
    return python_path, ld_path

# New addition
def validate_with_checker(checker_code, user_code, input_data, correct_output=None):
    """
    Validate user code with Checker
    Args:
        checker_code (str): code from checker.py
        user_code (str): code submitted by user
        input_data (str): input data
        correct_output (str): standard answer (optional, some Checkers need it)
    Returns:
        bool: whether validation passed
        str: Checker output (score or validation result)
    """
    temp_dir = tempfile.mkdtemp(prefix="checker_")
    try:
        # Save related files
        checker_path = os.path.join(temp_dir, "checker.py")
        user_code_path = os.path.join(temp_dir, "solution.py")
        input_path = os.path.join(temp_dir, "input.txt")
        output_path = os.path.join(temp_dir, "solution_output.txt")
        correct_output_path = os.path.join(temp_dir, "correct_output.txt") if correct_output else "/dev/null"

        with open(checker_path, "w") as f:
            f.write(checker_code)
        with open(user_code_path, "w") as f:
            f.write(user_code)
        with open(input_path, "w") as f:
            f.write(input_data)
        if correct_output:
            with open(correct_output_path, "w") as f:
                f.write(correct_output)

        # Run user code
        try:
            with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
                subprocess.run(
                    [sys.executable, user_code_path],
                    stdin=f_in,
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    timeout=10,
                    check=True
                )
        except subprocess.TimeoutExpired:
            # print("❌ User code execution timeout (over 10 seconds)")
            return False, f"[code forces]validate_with_checker User code execution timeout (over 10 seconds)"
        except subprocess.CalledProcessError as e:
            # print("❌ User code execution failed, error message:", e.stderr.decode())
            return False, f"[code forces] User code execution failed, error message"
        
        # Check user output
        # with open(output_path, "r") as f:
            # user_output = f.read()
        # print("🔍 User code output:", user_output.strip())

        # Run Checker for validation
        result = subprocess.run(
            [sys.executable, checker_path, input_path, correct_output_path, output_path],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Parse Checker output (may be 0-1 or 0-100)
        checker_output = result.stdout.strip()
        try:
            score = float(checker_output)
            return (score > 0), checker_output
        except ValueError:
            return False, f"Checker returned invalid output: {checker_output}"

    except subprocess.TimeoutExpired:
        return False, "Checker validation timed out"
    except subprocess.CalledProcessError as e:
        return False, f"Checker: Solution execution failed: {e.stderr}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
def get_env_paths():
    """
    Get PYTHONPATH and LD_LIBRARY_PATH of the current environment
    Automatically detect Conda environment or system Python environment and return corresponding paths
    """
    python_exec = sys.executable
    # print('python_exec:',python_exec)
    # More reliable Conda environment detection method
    is_conda = (
        "conda" in python_exec or 
        "CONDA_PREFIX" in os.environ or
        "CONDA_DEFAULT_ENV" in os.environ
    )
    
    if is_conda:
        # print('is conda')
        # Conda environment path handling
        conda_env_path = os.environ.get("CONDA_PREFIX", str(Path(python_exec).parent.parent))
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        
        python_path = f"{conda_env_path}/lib/{python_version}/site-packages"
        ld_path = f"{conda_env_path}/lib:/usr/local/lib:/usr/lib"
        
        # Add Conda environment specific library paths
        if "CONDA_PREFIX" in os.environ:
            ld_path = f"{os.environ['CONDA_PREFIX']}/lib:{ld_path}"
    else:
        # print('syspath')
        # System Python environment path handling
        python_path = ":".join([
            f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages",
            f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        ])
        
        # Standard Linux library paths
        ld_path = "/usr/local/lib:/usr/lib"
        
        # If Python is installed in non-standard location
        if python_exec.startswith("/usr/local"):
            python_path = f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages:{python_path}"
            ld_path = f"/usr/local/lib:{ld_path}"
    
    return python_path, ld_path
def run_in_nsjail(code_str, has_input=False, checker_code=None, input_data="", correct_output=None):
    """Safely execute Python code in nsjail"""
    # Get current conda environment paths
    python_path, ld_path = get_env_paths()
    # print(f'python_path:{python_path}')
    # print(f'ld_path:{ld_path}')
    # New addition: checker
    if checker_code:
        is_valid, validation_msg = validate_with_checker(   # This function might error: cannot unpack non-iterable bool object
            checker_code=checker_code,
            user_code=code_str,
            input_data=input_data,
            correct_output=correct_output
        )
        if not is_valid:
            return {
                "success": False,
                "returncode": 1,
                "stdout": "",
                "stderr": f"Checker validation failed: {validation_msg}",
                "checker_output": validation_msg
            }

    temp_dir = tempfile.mkdtemp(prefix="nsjail_")
    temp_work_dir = os.path.join(temp_dir, "workspace")
    os.makedirs(temp_work_dir, exist_ok=True)
    try:
        # Build nsjail command
        code_path = os.path.join(temp_dir, "code.py")
        with open(code_path, 'w') as f:
            f.write(code_str)
        
        # print(f'code_path:{code_path}')
        # print(f'nsjail_path:{nsjail_path}')
        cmd = [
            nsjail_path,
            "--disable_proc",
            "--mode", "o",
            "--user", "nobody",
            "--group", "nogroup",
            "--chroot", "/",
            "--cwd", "/tmp/workspace",
            "--rlimit_as", "50000",  # 50GB memory limit
            "--rlimit_cpu", "30",    # 30 seconds CPU time limit
            "--bindmount_ro", "/opt:/opt",
            "--bindmount_ro", f"{code_path}:/tmp/code.py",
            "--bindmount", f"{temp_work_dir}:/tmp/workspace",  # Writable working directory
            "--bindmount_ro", "/tmp/empty:/mnt",  # Isolate /mnt
            "--env", f"PYTHONPATH={python_path}",
            "--env", f"LD_LIBRARY_PATH={ld_path}",
            "--really_quiet",
            "--",
            sys.executable,
            # "-c",
            "/tmp/code.py"#code_str
        ]
        
        # Create empty /mnt isolation directory
        os.makedirs("/tmp/empty", exist_ok=True)
        
        # Execute command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=25  # Total timeout
        )
        entry = {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        error_message = str(e)
        if 'timed out' in error_message:
            error_output= f"Time out. Check for infinite loops, blocked I/O (e.g., input()), or deadlocks."
        else:
            error_output =  f"{error_message}"
        # if has_input:
        #     error_message = 
        entry = {
            "success": False,
            "returncode": 1,
            "stdout": "",
            "stderr": error_output
        }
    finally:
        # Clean up temporary files
        # 6. Force cleanup of temporary directory (including all subdirectories)
        try:
            # print('remove',temp_dir)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f'[nsjail shutil.rmtree error]',e)
            
    return entry
def validate_output(result, expected_output_str, epsilon=1e-9):
    try:
        # Parse expected output
        expected_lines = expected_output_str.splitlines()
        actual_lines = result["stdout"].strip().splitlines()

        # Check line count matches
        assert len(expected_lines) == len(actual_lines), \
            f"Line count mismatch: expected {len(expected_lines)} lines, got {len(actual_lines)} lines"

        # Compare line by line
        for i, (exp_line, act_line) in enumerate(zip(expected_lines, actual_lines)):
            exp_val = parse_value_enhanced(exp_line)
            act_val = parse_value_enhanced(act_line.strip())

            if not is_equal_enhanced(exp_val, act_val, epsilon):
                raise AssertionError(
                    f"Mismatch at line {i+1}: expected {exp_line} ({type(exp_val)}), "
                    f"got {act_line} ({type(act_val)})"
                )

        return True
    except AssertionError as e:
        # print(f"Test failed: {e}")
        return False

def parse_value_enhanced(s):
    """Enhanced type inference (supports boolean, null)"""
    s = s.strip().lower()
    if s == "true":
        return True
    elif s == "false":
        return False
    elif s == "null":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s  # Preserve original string (case sensitive)

def is_equal_enhanced(expected, actual, epsilon=1e-9):
    """Enhanced comparison logic"""
    # Compare only if types match
    if type(expected) != type(actual):
        return False

    if isinstance(expected, bool):
        return expected == actual
    elif isinstance(expected, (int, float)):
        return abs(expected - actual) < epsilon
    elif isinstance(expected, str):
        return expected == actual
    else:
        return False
def mock_input_in_code(code_blob, input_str):
    # Process input string, ensure it's a string format
    if isinstance(input_str, list):
        input_str = "\n".join(input_str)
    
    # Create iterator for input lines
    input_lines = input_str.splitlines(keepends=True)
    input_lines_without_ends = [line.rstrip('\n') for line in input_lines]
    
    # Replace input() calls
    def replace_input(match):
        prompt = match.group(1) if match.group(1) else ''
        return f'next(input_iterator)'
    
    # Replace sys.stdin.readline() calls
    def replace_stdin_readline(match):
        return f'sys_stdin_readline()'
    
    # Replace sys.stdin.readlines() calls
    def replace_stdin_readlines(match):
        return f'sys_stdin_readlines()'
    
    # Replace sys.stdin.read() calls
    def replace_stdin_read(match):
        return f'sys_stdin_read()'
    
    # Replace open() calls (for standard input reading cases)
    def replace_open(match):
        filename = match.group(1)
        mode = match.group(2) if match.group(2) else 'r'
        if filename in ('sys.stdin', '/dev/stdin'):
            return f'mock_open_stdin()'
        return f'open({filename}, {mode})'
    
    # Handle input() calls
    modified_code = re.sub(
        r'input\(([^)]*)\)',
        replace_input,
        code_blob
    )
    
    # Handle sys.stdin.readline() calls
    modified_code = re.sub(
        r'sys\.stdin\.readline\(\)',
        replace_stdin_readline,
        modified_code
    )
    
    # Handle sys.stdin.readlines() calls
    modified_code = re.sub(
        r'sys\.stdin\.readlines\(\)',
        replace_stdin_readlines,
        modified_code
    )
    
    # Handle sys.stdin.read() calls
    modified_code = re.sub(
        r'sys\.stdin\.read\(\)',
        replace_stdin_read,
        modified_code
    )
    
    # Handle open() calls (for special sys.stdin cases)
    modified_code = re.sub(
        r'open\((sys\.stdin|/dev/stdin)(?:,\s*([\'\"]\w+[\'\"]))??\)',
        replace_open,
        modified_code
    )
    # noneed setup
    no_need_setup=f"""
from unittest.mock import mock_open, patch

input_iterator = iter({input_lines_without_ends})
def sys_stdin_readline():
    try:
        return next(input_iterator) + '\\n'
    except StopIteration:
        return ''

def sys_stdin_readlines():
    return list(input_iterator)

def sys_stdin_read():
    return {input_str!r}

def mock_open_stdin():
    return StringIO({input_str!r})
"""
    # Add code for input simulation
    input_setup = f"""
import sys
from io import StringIO
sys_stdin_content = StringIO({input_str!r})

# Replace sys.stdin
sys.stdin = sys_stdin_content
"""
    
    # Combine final code
    final_code = input_setup + modified_code
    return final_code
def exec_nsjail(code_blob, input_str=''):
    if input_str == '':
        result = run_in_nsjail(code_blob, has_input=False)
    else:
        new_code_blob = mock_input_in_code(code_blob, input_str)
        result = run_in_nsjail(new_code_blob, has_input=True)
    
    succ = result["success"]
    if succ:
        if not result["stdout"]:
            observation = '[EXECUTED] Code exited with status 0 (no output).'
        else:
            observation = (
                '[EXECUTED] Code exited with status 0.\n'
                '[STDOUT:BEGIN]\n'
                f'{result["stdout"].strip()}\n'
                '[STDOUT:END]'
            )
    else:
        exit_code = result.get("returncode", 1)
        if not result["stdout"] and result["stderr"]:
            observation = (
                f'[FAILED] Code exited with status {exit_code}.\n'
                '[STDERR:BEGIN]\n'
                f'{result["stderr"].strip()}\n'
                '[STDERR:END]'
            )
        elif not result["stdout"] and not result["stderr"]:
            observation = f'[FAILED] Code exited with status {exit_code} (no output).'
        elif result["stdout"] and not result["stderr"]:
            observation = (
                f'[FAILED] Code exited with status {exit_code}.\n'
                '[STDOUT:BEGIN]\n'
                f'{result["stdout"].strip()}\n'
                '[STDOUT:END]'
            )
        else:
            observation = (
                f'[FAILED] Code exited with status {exit_code} with mixed output:\n'
                '[STDOUT:BEGIN]\n'
                f'{result["stdout"].strip()}\n'
                '[STDOUT:END]\n'
                '[STDERR:BEGIN]\n'
                f'{result["stderr"].strip()}\n'
                '[STDERR:END]'
            )
    return succ, observation
def exec_nsjail_testoutput(code_blob, input_str='', expected_output=None, checker_code=None, ):
    if input_str=='':
        result = run_in_nsjail(code_blob, False, checker_code, input_str, expected_output)
    else:
        new_code_blob = mock_input_in_code(code_blob,input_str)
        # print('new code blob,',new_code_blob) 
        result = run_in_nsjail(new_code_blob, True, checker_code, input_str, expected_output)
    succ = result["success"]
    if succ:
        if not result["stdout"] and not expected_output:  # Empty stdout
            observation = '[EXECUTED] Code exited with status 0 (no output).'
        else:
            observation = (
                '[EXECUTED] Code exited with status 0.\n'
                '[STDOUT:BEGIN]\n'
                f'{result["stdout"].strip()}\n'
                '[STDOUT:END]'
            )
            ###########Assume output is not empty, must follow this path to be correct
            if expected_output:
                succ = validate_output(result, expected_output)
                # succ = ('\n'.join(expected_output.split('\n'))=='\n'.join(result["stdout"].split('\n')))
                if not succ:
                    observation = f'[Validation:Failed]Test case failed.\n[expected_output:BEGIN]\n{expected_output.strip()}\n[expected_output:END]\n[realoutput:BEGIN]\n{result["stdout"].strip()}\n[realoutput:END]'

    if succ:  # Code executed successfully (exit code 0)
        if not result["stdout"] and not expected_output:  # Empty stdout
            observation = '[RESULT:SUCCESS] The code executed successfully with no output.'
        else:  # Has stdout
            observation = f'[RESULT:SUCCESS] The code executed successfully.\n[STDOUT:BEGIN]\n{result["stdout"].strip()}\n[STDOUT:END]'
            ###########Assume output is not empty, must follow this path to be correct
            if expected_output:
                succ = validate_output(result, expected_output)
                # succ = ('\n'.join(expected_output.split('\n'))=='\n'.join(result["stdout"].split('\n')))
                if not succ:
                    observation = f'[Validation:Failed]Test case failed.\n[expected_output:BEGIN]\n{expected_output.strip()}\n[expected_output:END]\n[realoutput:BEGIN]\n{result["stdout"].strip()}\n[realoutput:END]'
            # dictonl list 
    else:
        exit_code = result.get("returncode", 1)
        if not result["stdout"] and result["stderr"]:
            observation = (
                f'[FAILED] Code exited with status {exit_code}.\n'
                '[STDERR:BEGIN]\n'
                f'{result["stderr"].strip()}\n'
                '[STDERR:END]'
            )
        elif not result["stdout"] and not result["stderr"]:
            observation = f'[FAILED] Code exited with status {exit_code} (no output).'
        elif result["stdout"] and not result["stderr"]:
            observation = (
                f'[FAILED] Code exited with status {exit_code}.\n'
                '[STDOUT:BEGIN]\n'
                f'{result["stdout"].strip()}\n'
                '[STDOUT:END]'
            )
        else:
            observation = (
                f'[FAILED] Code exited with status {exit_code} with mixed output:\n'
                '[STDOUT:BEGIN]\n'
                f'{result["stdout"].strip()}\n'
                '[STDOUT:END]\n'
                '[STDERR:BEGIN]\n'
                f'{result["stderr"].strip()}\n'
                '[STDERR:END]'
            )
    return succ, observation