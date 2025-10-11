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
import shutil
import ast


class NsjailSandbox:
    """Nsjail sandbox executor, used to safely execute Python code in an isolated environment"""

    def __init__(self, nsjail_path=None):
        """
        Initialize NsjailSandbox instance
        
        Args:
            nsjail_path: Path to the nsjail executable file
        """
        self.nsjail_path = nsjail_path
        if not self.nsjail_path:
            raise ValueError("nsjail_path not set")
    
    def parse_code_blobs_stdin_answer(self, code_blob: str) -> tuple[bool,str,str]:
        """Parses the LLM's output to get any code blob inside. Will return the code directly if it's code."""
        parts = code_blob.split('<answer>', 1)
        is_final_answer = False
        if len(parts) > 1:
            # print('[parse_code_blobs] get answer!')
            is_final_answer = True
            # Get everything after Answer:, then split by code block markers if needed
            code_blob = parts[1].strip()
        pattern = r"```(?:py|python)\n(.*?)\n```"
        # print('[parse_code_blobs]',code_blob,'\n[end parse_code_blobs]')
        pattern_stdin = r"```(?:sh|bash)\n(.*?)```"
        matches = re.findall(pattern, code_blob, re.DOTALL)
        matches_stdin = re.findall(pattern_stdin, code_blob, re.DOTALL)
        # print('matches',matches)
        # print('matches_stdin',matches_stdin)
        if len(matches) == 0:
            try:  # Maybe the LLM outputted a code blob directly
                # print('[parse_code_blobs] direct code')
                
                ast.parse(code_blob)
                return code_blob, ""
            except SyntaxError:
                try:
                    if '```py\n' in code_blob:
                        code_blob = code_blob.split('```py\n')[1]
                        ast.parse(code_blob)
                        return code_blob, ""
                except SyntaxError:
                    pass

            if "final" in code_blob and "answer" in code_blob:
                
                print('[parse_code_blobs]',1)
                raise ValueError(
                    f"""
    Your code snippet is invalid, because the regex pattern {pattern} was not found in it. \
    Here is your code snippet: \
    {code_blob} \
    It seems like you're trying to return the final answer, you can give it out of code snippet as follows:
    <answer>
    ```py
    YOUR FINAL ANSWER HERE
    ```</answer>""".strip()
                )
            raise ValueError(
                f"""
    Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
    Here is your code snippet:
    {code_blob}
    Make sure to include code with the correct pattern, for instance:
    <think>
    Your thoughts
    </think>

    <code>
    ```py
    # Your python code here
    ```
    </code>""".strip()
            )
        if len(matches_stdin)==0:
            stdin_str=''
        else:
            stdin_str = "\n".join(match for match in matches_stdin)
        # print(f'[code blobs parse stdin]:{repr(stdin_str)}')
        # print('[parse_code_blobs] right')
        return is_final_answer, "\n\n".join(match.strip() for match in matches), stdin_str
    
    def get_conda_env_paths(self):
        """Get the PYTHONPATH and LD_LIBRARY_PATH of the current conda environment"""
        python_exec = sys.executable
        if "conda" not in python_exec:
            raise RuntimeError("Not running in a conda environment")
        
        conda_env_path = str(Path(python_exec).parent.parent)
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        
        python_path = f"{conda_env_path}/lib/{python_version}/site-packages"
        ld_path = f"{conda_env_path}/lib:/lib:/usr/lib"
        
        return python_path, ld_path
    
    def get_env_paths(self):
        """
        Get the PYTHONPATH and LD_LIBRARY_PATH of the current environment
        Automatically detect the Conda environment or system Python environment and return the corresponding paths
        """
        python_exec = sys.executable
        is_conda = (
            "conda" in python_exec or 
            "CONDA_PREFIX" in os.environ or
            "CONDA_DEFAULT_ENV" in os.environ
        )
        
        if is_conda:
            conda_env_path = os.environ.get("CONDA_PREFIX", str(Path(python_exec).parent.parent))
            python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            
            python_path = f"{conda_env_path}/lib/{python_version}/site-packages"
            ld_path = f"{conda_env_path}/lib:/usr/local/lib:/usr/lib"
            
            if "CONDA_PREFIX" in os.environ:
                ld_path = f"{os.environ['CONDA_PREFIX']}/lib:{ld_path}"
        else:
            python_path = ":".join([
                f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages",
                f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            ])
            
            ld_path = "/usr/local/lib:/usr/lib"
            
            if python_exec.startswith("/usr/local"):
                python_path = f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages:{python_path}"
                ld_path = f"/usr/local/lib:{ld_path}"
        
        return python_path, ld_path
    
    def _cleanup_temp_dir(self, temp_dir):
        """Clean up the temporary directory"""
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f'[nsjail cleanup error] {e}')
    
    def run_in_nsjail(self, code_str, has_input=False):
        """
        Safely execute Python code in nsjail

        Args:
            code_str: Python code to be executed
            has_input: Whether there is input

        Returns:
            dict: A dictionary containing execution results, including success, returncode, stdout, stderr
        """
        python_path, ld_path = self.get_env_paths()
        temp_dir = tempfile.mkdtemp(prefix="nsjail_")
        temp_work_dir = os.path.join(temp_dir, "workspace")
        os.makedirs(temp_work_dir, exist_ok=True)
        
        try:
            code_path = os.path.join(temp_dir, "code.py")
            with open(code_path, 'w') as f:
                f.write(code_str)
            
            cmd = [
                self.nsjail_path,
                "--disable_proc",
                "--mode", "o",
                "--user", "nobody",
                "--group", "nogroup",
                "--chroot", "/",
                "--cwd", "/tmp/workspace",
                "--rlimit_as", "5000",  # (MB) 4.88GB memory limit 
                "--rlimit_cpu", "5",    # 5-second CPU time limit
                "--bindmount_ro", "/opt:/opt",
                "--bindmount_ro", f"{code_path}:/tmp/code.py",
                "--bindmount", f"{temp_work_dir}:/tmp/workspace",
                "--bindmount_ro", "/tmp/empty:/mnt",
                "--env", f"PYTHONPATH={python_path}",
                "--env", f"LD_LIBRARY_PATH={ld_path}",
                "--really_quiet",
                "--",
                sys.executable,
                "/tmp/code.py"
            ]
            
            os.makedirs("/tmp/empty", exist_ok=True)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7
            )
            entry = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            error_message = str(e)
            error_output = f"Time out. Check for infinite loops, blocked I/O (e.g., input()), or deadlocks." if 'timed out' in error_message else f"{error_message}"
            entry = {
                "success": False,
                "returncode": 1,
                "stdout": "",
                "stderr": error_output
            }
        finally:
            self._cleanup_temp_dir(temp_dir)
            
        return entry
    
    def parse_value_enhanced(self, s):
        """
        Enhanced type inference (supports boolean and null values)
        
        Args:
            s: The string to be parsed
        
        Returns:
            The inferred value, which may be of type bool, int, float, or str
        """
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
                return s  # Preserve the original string (case-sensitive)
    
    def is_equal_enhanced(self, expected, actual, epsilon=1e-9):
        """
        Enhance comparison logic to support comparison of numbers, booleans, and strings
        
        Args:
            expected: Expected value
            actual: Actual value
            epsilon: Tolerance for numerical comparison
        
        Returns:
            bool: Comparison result
        """
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
    
    def mock_input_in_code(self, code_blob, input_str):
        
        """
        Simulate input in the code, replacing input() and stdin-related calls
        
        Args:
            code_blob: Original code
            input_str: Input string

        Returns:
            str: Modified code containing input simulation logic
        """
        if isinstance(input_str, list):
            input_str = "\n".join(input_str)
        
        input_lines = input_str.splitlines(keepends=True)
        input_lines_without_ends = [line.rstrip('\n') for line in input_lines]
        
        def replace_input(match):
            prompt = match.group(1) if match.group(1) else ''
            return f'next(input_iterator)'
        
        def replace_stdin_readline(match):
            return f'sys_stdin_readline()'
        
        def replace_stdin_readlines(match):
            return f'sys_stdin_readlines()'
        
        def replace_stdin_read(match):
            return f'sys_stdin_read()'
        
        def replace_open(match):
            filename = match.group(1)
            mode = match.group(2) if match.group(2) else 'r'
            if filename in ('sys.stdin', '/dev/stdin'):
                return f'mock_open_stdin()'
            return f'open({filename}, {mode})'
        
        modified_code = re.sub(
            r'input\s*\(\s*([\'"]).*?\1\s*\)',
            replace_input,
            code_blob
        )
        
        modified_code = re.sub(
            r'sys\.stdin\.readline\s*\(\s*\)',
            replace_stdin_readline,
            modified_code
        )
        
        modified_code = re.sub(
            r'sys\.stdin\.readlines\s*\(\s*\)',
            replace_stdin_readlines,
            modified_code
        )
        
        modified_code = re.sub(
            r'sys\.stdin\.read\s*\(\s*\)',
            replace_stdin_read,
            modified_code
        )
        
        modified_code = re.sub(
            r'open\s*\(\s*(sys\.stdin|/dev/stdin)(?:,\s*([\'"])\w+[\'"])?\s*\)',
            replace_open,
            modified_code
        )
        
        input_setup = f"""
import sys
from io import StringIO
sys_stdin_content = StringIO({input_str!r})
sys.stdin = sys_stdin_content

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
        final_code = input_setup + modified_code
        return final_code
    
    def exec(self, code_blob, input_str=''):
        """
        Execute code in nsjail with input simulation support

        Args:
            code_blob: Code to be executed
            input_str: Input string (optional)

        Returns:
            tuple: (Whether execution was successful, execution result description)
        """
        if input_str == '':
            result = self.run_in_nsjail(code_blob, has_input=False)
        else:
            new_code_blob = self.mock_input_in_code(code_blob, input_str)
            result = self.run_in_nsjail(new_code_blob, has_input=True)
        
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
    
    def exec_nsjail_testoutput(self, code_blob, input_str='', expected_output=None):
        
        """
        Execute code in nsjail and verify if the output meets expectations

        Args:
            code_blob: Code to be executed
            input_str: Input string (optional)
            expected_output: Expected output (optional)
        
        Returns:
            tuple: (Whether the verification passed, execution result description)
        """
        if input_str == '':
            result = self.run_in_nsjail(code_blob, has_input=False)
        else:
            new_code_blob = self.mock_input_in_code(code_blob, input_str)
            result = self.run_in_nsjail(new_code_blob, has_input=True)
        
        succ = result["success"]
        if succ:
            if not result["stdout"] and not expected_output:
                observation = '[RESULT:SUCCESS] The code executed successfully with no output.'
            else:
                observation = f'[RESULT:SUCCESS] The code executed successfully.\n[STDOUT:BEGIN]\n{result["stdout"].strip()}\n[STDOUT:END]'
                if expected_output:
                    succ = self.validate_output(result, expected_output)
                    if not succ:
                        observation = f'[Validation:Failed] Test case failed.\n[expected_output:BEGIN]\n{expected_output.strip()}\n[expected_output:END]\n[realoutput:BEGIN]\n{result["stdout"].strip()}\n[realoutput:END]'
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
    
    def validate_output(self, result, expected_output_str, epsilon=1e-9):
        """
        Verify whether the code execution output meets expectations
        
        Args:
            result: Execution result dictionary
            expected_output_str: Expected output string
            epsilon: Tolerance for numerical comparison
        
        Returns:
            bool: Verification result
        """
        try:
            expected_lines = expected_output_str.splitlines()
            actual_lines = result["stdout"].strip().splitlines()

            if len(expected_lines) != len(actual_lines):
                raise AssertionError(
                    f"Line count mismatch: expected {len(expected_lines)} lines, actual {len(actual_lines)} lines"
                )

            for i, (exp_line, act_line) in enumerate(zip(expected_lines, actual_lines)):
                exp_val = self.parse_value_enhanced(exp_line)
                act_val = self.parse_value_enhanced(act_line.strip())

                if not self.is_equal_enhanced(exp_val, act_val, epsilon):
                    raise AssertionError(
                        f"Line {i+1} mismatch: expected {exp_line} ({type(exp_val)}), "
                        f"actual {act_line} ({type(act_val)})"
                    )

            return True
        except AssertionError:
            return False