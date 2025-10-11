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


import json
import logging
import os
import ray
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

from verl.tools.utils.code_executors.nsjail_sandbox import NsjailSandbox
from verl.tools.utils.code_executors.utils import truncate_content
from concurrent.futures import ProcessPoolExecutor
import resource
import ray
from multiprocessing import Process, Queue
import signal
import time
import sympy

# logger config
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("CODEEXEC_LOGGING_LEVEL", "WARN"))

class CodeExecutor(BaseTool):
    """Run the written code in nsjail sandbox env and return the result"""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize CodeExecutor and config"""
        super().__init__(config, tool_schema)

        # state save
        self._instance_dict = {}    
        
        # check execution config
        if "timeout" not in config:
            raise ValueError(f"[Error] Lack param 'timeout' in code_executor.yaml")
        if "memory_limit" not in config:
            raise ValueError(f"[Error] Lack param 'memory_limit' in code_executor.yaml")
        if "nsjail_path" not in config:
            raise ValueError(f"[Error] Lack param 'nsjail_path' in code_executor.yaml")
        if "max_obs_length" not in config:
            raise ValueError(f"[Error] Lack param 'max_obs_length' in code_executor.yaml")
        
        self.timeout = int(config["timeout"])
        self.memory_limit = int(sympy.sympify(config["memory_limit"])) 
        self.nsjail_path = config["nsjail_path"]
        self.max_obs_length = int(config["max_obs_length"]) * 3     # token-level length limit
        
        # print config
        logger.info(f"Initialized CodeExecutor with config: {config}")
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""

        if self.tool_schema:
            return self.tool_schema
            
        return OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "code",
                "description": "Execute the code blob in nsjail sandbox and return corresponding results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code_blob": {
                            "type": "string",
                            "description": "The code blob written by LLMs"
                        },
                    },
                    "required": ["code_blob"]
                }
            }
        )
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": 0,
            "results": []
        }
        return instance_id
    
    def _worker(self, q, query):
        # Set memory limitation for the execution subprocess
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, hard))
        nsjail_sandbox = NsjailSandbox(self.nsjail_path)

        try:
            _, code_blob, stdin_str = nsjail_sandbox.parse_code_blobs_stdin_answer(query)
            succ, output = nsjail_sandbox.exec(code_blob, stdin_str)
            q.put((succ, output))
        except MemoryError:
            q.put((False, "MEMORY_LIMIT_EXCEEDED"))
        except Exception as e:
            q.put((False, f"ERROR: {str(e)}"))

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the code_executor tool.
        
        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing code_blob
            
        Returns:
            Tuple of (tool_response, tool_reward_score, tool_metrics)
        """
        
        query = parameters.get("query")

        # Check delimiters
        # if not query.startswith("```py\n") and not query.endswith("\n```"):
        #     query = f"```py\n{query}\n```"
  
        # Init queue
        q = Queue()
        p = Process(target=self._worker, args=(q, query))
        p.start()
        
        # Set timeout for the execution subprocess and start it
        p.join(timeout=self.timeout)
        
        result = None
        if p.is_alive():
            p.terminate()  # Timeout, kill the subprocess immediately
            result = (False, "PROCESS_TIMEOUT")
        if not result:
            result = q.get() if not q.empty() else (False, "NO_RESULT")
        
        # Get result and cut obs length
        succ, output = result
        output = truncate_content(output, self.max_obs_length)

        # TODO: prm reward and extra_info
        prm_reward = 0.0
        metrics = {}

        return output, prm_reward, metrics

    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward for the tool instance."""
        return 0.0 # TODO: self._instance_dict[instance_id]["reward"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]