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


import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import aiohttp

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



class SerperCacheAPITool(BaseTool):
    """Serper Cache API tool that calls external serper cache server API v2.
    
    This tool provides web search functionality with caching by calling an external 
    FastAPI server that handles Google search via Serper API.
    
    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the web search tool by calling server API
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        """Initialize SerperCacheAPITool with configuration and schema.
        
        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Config options:
            - timeout: Request timeout in seconds (default: 60)
            - num_results: Default number of search results (default: 10)
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        server_host = os.environ.get("SERVER_HOST")
        if not server_host:
            raise ValueError(f"SERVER_HOST({server_host}) is not set and server_host.tmp file not found")
        self.serper_cache_url = f"http://{server_host}:9001/search"
        self.serper_api_key = os.environ.get("WEB_SEARCH_SERPER_API_KEY")
        self.timeout = 500
        self.num_results = config.get("num_results", 10)

        # Validate required environment variables
        if not self.serper_api_key:
            logger.warning("WEB_SEARCH_SERPER_API_KEY not set - requests must include X-API-KEY header")

        logger.info(f"Initialized SerperCacheAPITool with server: {self.serper_cache_url}")
    
    def _format_results_to_string(self, serper_json: Dict[str, Any], query: str = None) -> str:
        """Formats the Serper JSON result into a structured string."""
        if "organic" not in serper_json or not serper_json["organic"]:
            return f"No results found for query: '{query}'. Use a less specific query."

        web_snippets = []
        for idx, page in enumerate(serper_json["organic"], 1):
            title = page.get("title", "No Title")
            link = page.get("link", "#")
            date_published = f"\nDate published: {page['date']}" if "date" in page else ""
            source = f"\nSource: {page.get('source', '')}" if "source" in page else ""
            snippet = f"\n{page.get('snippet', '')}".replace("Your browser can't play this video.", "")

            formatted_entry = (
                f"{idx}. [{title}]({link})"
                f"{date_published}{source}"
                f"\n{link}{snippet}"
            )
            web_snippets.append(formatted_entry.strip())
        
        num_results = len(web_snippets)
        return (
            f"Found {num_results} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        if self.tool_schema:
            return self.tool_schema
        
        return OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "web_search",
                "description": "Search the web using Google via Serper API with caching. Returns search results with title, link, and snippet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": f"Number of search results to return (default: {self.num_results})",
                            "default": self.num_results
                        }
                    },
                    "required": ["query"]
                }
            }
        )
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.
        
        Args:
            instance_id: The instance id of the tool.
            
        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": 0.0,
            "metrics": {}
        }
        return instance_id


    def _format_search_results(self, results: dict) -> str:
        """Format search results into a readable string.
        
        Args:
            results: Search results from serper API
            
        Returns:
            Formatted string representation of the results
        """
        # Format organic search results
        organic_results = results.get("organic", [])
        if not organic_results:
            return "No search results found."
        
        formatted_results = []
        for item in organic_results:
            title = item.get('title', 'No title')
            link = item.get('link', 'No link')
            snippet = item.get('snippet', 'No snippet')
            formatted_results.append(f"[url:{link}]: (Title: {title}) (Content: {snippet})")
        
        return "\n".join(formatted_results)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the web search tool by calling server API.
        
        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query and optional settings
            kwargs: Additional arguments

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool (search results).
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        query = parameters.get("query")
        if not query or not isinstance(query, str):
            error_msg = "Error: 'query' is missing, empty, or not a string in parameters."
            logger.error(f"[SerperCacheAPITool] {error_msg} Received parameters: {parameters}")
            return json.dumps({"error": error_msg}), 0.0, {"error": "invalid_parameters"}

        num_results = parameters.get("num_results", self.num_results)
        
        try:
            # 构建请求payload
            payload = {
                "q": query,
                "num": num_results
            }

            # Prepare headers
            headers = {"Content-Type": "application/json"}
            if self.serper_api_key:
                headers["X-API-KEY"] = self.serper_api_key
            else:
                raise ValueError("Serper API key is not set")

            logger.info(f"[SerperCacheAPITool] Searching for: {query}")

            # 调用服务器API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.serper_cache_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_msg = f"Server returned status {response.status}, fail to call {self.serper_cache_url}"
                        logger.error(f"[SerperCacheAPITool] {error_msg}")
                        return json.dumps({"error": error_msg}), 0.0, {"error": "server_error", "status": response.status}

                    result = await response.json()
            
            # 格式化结果
            # breakpoint()
            search_results = self._format_results_to_string(serper_json=result)
            
            # 准备返回的metrics
            result_count = 0
            metrics = {
                "search_query": query,
                "num_results": num_results,
                "result_count": result_count
            }

            # 存储结果到实例字典
            self._instance_dict[instance_id]["response"] = search_results
            self._instance_dict[instance_id]["metrics"] = metrics

            logger.info(f"[SerperCacheAPITool] Found {result_count} results for query: {query}")

            return search_results, 0.0, metrics

        except asyncio.TimeoutError:
            error_msg = f"Request to serper cache server timed out after {self.timeout}s, fail to call {self.serper_cache_url}"
            logger.error(f"[SerperCacheAPITool] {error_msg}")
            return json.dumps({"error": error_msg}), 0.0, {"error": "timeout"}

        except aiohttp.ClientError as e:
            error_msg = f"Failed to connect to serper cache server: {str(e)}, fail to call {self.serper_cache_url}"
            logger.error(f"[SerperCacheAPITool] {error_msg}")
            return json.dumps({"error": error_msg}), 0.0, {"error": "connection_error"}

        except Exception as e:
            error_msg = f"Web search execution failed: {str(e)}, fail to call {self.serper_cache_url}"
            logger.error(f"[SerperCacheAPITool] Unexpected error: {error_msg}")
            return json.dumps({"error": error_msg}), 0.0, {"error": "unexpected_error"}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward for the tool instance.

        Args:
            instance_id: The instance ID of the tool

        Returns:
            The reward for the tool instance
        """
        if instance_id in self._instance_dict:
            return self._instance_dict[instance_id]["reward"]
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release resources for the tool instance.

        Args:
            instance_id: The instance ID of the tool
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            logger.debug(f"[SerperCacheAPITool] Released instance {instance_id}")