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
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class CrawlPageV2Tool(BaseTool):
    """Crawl Page v2 tool that calls external crawl page server API v2.
    
    This tool provides web page crawling and AI summarization functionality 
    by calling an external FastAPI server.
    
    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the crawl page tool by calling server API
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        """Initialize CrawlPageV2Tool with configuration and schema.
        
        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Config options:
            - summary_type: Type of summarization (default: "once")
            - chunk_size: Size of chunks for chunk-based summarization (default: 8192)
            - do_last_summary: Whether to do final summary (default: False)
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        server_host = os.environ.get("SERVER_HOST")
        if not server_host:
            raise ValueError(f"SERVER_HOST({server_host}) is not set and server_host.tmp file not found")
        self.crawl_page_endpoint = f"http://{server_host}:9000/crawl_page"
        self.summary_type = config.get("summary_type", "once")
        self.chunk_size = config.get("chunk_size", 8192)
        self.do_last_summary = config.get("do_last_summary", False)
        
        self.api_url = os.environ.get("SUMMARY_OPENAI_API_BASE_URL")
        self.api_key = os.environ.get("SUMMARY_OPENAI_API_KEY") 
        self.model = os.environ.get("SUMMARY_MODEL")

        if not self.api_url or not self.api_key or not self.model:
            raise ValueError("Summary API configuration not complete")
        if not self.crawl_page_endpoint:
            raise ValueError("Crawl page endpoint not set")

        logger.info(f"Initialized CrawlPageV2Tool with server: {self.crawl_page_endpoint}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        if self.tool_schema:
            return self.tool_schema
        
        return OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "crawl_page",
                "description": "Crawl web pages and get AI-powered summary of their content. Use this after web_search to get detailed information from specific pages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "URLs to crawl, separated by '|'. You can use URLs from web_search results. Supports plain URLs or markdown format '[title](url)'"
                        },
                        "summary_type": {
                            "type": "string",
                            "description": f"Type of summarization: 'none', 'once', 'chunk', or 'page' (default: {self.summary_type})",
                            "enum": ["none", "once", "chunk", "page"],
                            "default": self.summary_type
                        },
                        "chunk_size": {
                            "type": "integer",
                            "description": f"Size of chunks for chunk-based summarization (default: {self.chunk_size})",
                            "default": self.chunk_size
                        },
                        "do_last_summary": {
                            "type": "boolean",
                            "description": f"Whether to do final summary for multi-chunk/page results (default: {self.do_last_summary})",
                            "default": self.do_last_summary
                        }
                    },
                    "required": ["query"]
                }
            }
        )

    async def create(self, instance_id: str, **kwargs) -> bool:
        """Create a tool instance for a trajectory.
        
        Args:
            instance_id: The instance ID of the tool
            
        Returns:
            True if the tool instance is created successfully
        """
        self._instance_dict[instance_id] = {}
        return True

    def _extract_context_from_messages(self, messages: list) -> tuple[str, str, str]:
        """Extract task, web_search_query, and think_content from message history.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Tuple of (task, web_search_query, think_content)
        """
        # Get task from user message
        task = ""
        if len(messages) > 1 and messages[1].get('role') == "user":
            task = messages[1].get('content', '')

        # Get web search query from recent tool calls
        web_search_query = ""
        for m in reversed(messages):
            tool_calls = m.get("tool_calls", None)
            if tool_calls and tool_calls[0]['function']['name'] == 'web_search':
                web_search_query = tool_calls[0]['function']['arguments']['query']
                break

        # Get think content from recent messages
        think_content = ""
        for m in reversed(messages):
            if 'content' in m and isinstance(m['content'], str):
                import re
                matches = re.findall(r'<think>(.*?)</think>', m['content'], re.DOTALL)
                if matches:
                    think_content = matches[0].strip()
                    break

        return task, web_search_query, think_content

    def _parse_urls_from_query(self, query: str) -> List[str]:
        """Parse URLs from query string.
        
        Args:
            query: Query string containing URLs separated by '|'
            
        Returns:
            List of URLs
        """
        # Split by '|' and strip whitespace
        urls = [url.strip() for url in query.split('|') if url.strip()]
        return urls

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the crawl page tool by calling server API.
        
        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query and optional settings
            kwargs: Additional arguments including message history

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool (crawled and summarized content).
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        query = parameters.get("query")
        if not query or not isinstance(query, str):
            error_msg = "Error: 'query' is missing, empty, or not a string in parameters."
            logger.error(f"[CrawlPageV2Tool] {error_msg} Received parameters: {parameters}")
            return json.dumps({"error": error_msg}), 0.0, {"error": "invalid_parameters"}

        # Parse URLs from query
        urls = self._parse_urls_from_query(query)
        if not urls:
            error_msg = "Error: No valid URLs found in query."
            logger.error(f"[CrawlPageV2Tool] {error_msg} Query: {query}")
            return json.dumps({"error": error_msg}), 0.0, {"error": "no_urls"}

        # Get optional parameters
        summary_type = parameters.get("summary_type", self.summary_type)
        chunk_size = parameters.get("chunk_size", self.chunk_size)
        do_last_summary = parameters.get("do_last_summary", self.do_last_summary)
        
        # Extract context from message history if available
        task, web_search_query, think_content = "", "", ""
        if '_messages_list_of_dic' in kwargs:
            messages = kwargs['_messages_list_of_dic']
            task, web_search_query, think_content = self._extract_context_from_messages(messages)
        
        try:
            # 构建请求payload
            payload = {
                "urls": urls,
                "task": task,
                "web_search_query": web_search_query or "general search",
                "think_content": think_content or "general content extraction",
                "summary_type": summary_type,
                "chunk_size": chunk_size,
                "do_last_summary": do_last_summary,
                "api_url": self.api_url,
                "api_key": self.api_key,
                "model": self.model
            }

            # Prepare headers
            headers = {"Content-Type": "application/json"}

            logger.info(f"[CrawlPageV2Tool] Crawling {len(urls)} URLs with summary_type={summary_type}")

            # 调用服务器API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.crawl_page_endpoint,
                    json=payload,
                    headers=headers,
                ) as response:
                    
                    if response.status != 200:
                        error_msg = f"Server returned status {response.status}, fail to call {self.crawl_page_endpoint}"
                        logger.error(f"[CrawlPageV2Tool] {error_msg}")
                        return json.dumps({"error": error_msg}), 0.0, {"error": "server_error", "status": response.status}

                    result = await response.json()
            
            # 检查服务器响应
            if not result.get("success", False):
                error_msg = result.get("error_message", "Unknown server error") + f"fail to use {self.crawl_page_endpoint}"
                logger.error(f"[CrawlPageV2Tool] Server error: {error_msg}")
                return json.dumps({"error": error_msg}), 0.0, {"error": "server_processing_error"}

            # 获取结果
            crawl_result = result.get("obs", "")
            processing_time = result.get("processing_time", 0)
            
            # 准备返回的metrics
            metrics = {
                "urls_count": len(urls),
                "summary_type": summary_type,
                "chunk_size": chunk_size,
                "do_last_summary": do_last_summary,
                "processing_time": processing_time
            }

            # 存储结果到实例字典
            self._instance_dict[instance_id]["response"] = crawl_result
            self._instance_dict[instance_id]["metrics"] = metrics

            logger.info(f"[CrawlPageV2Tool] Successfully crawled {len(urls)} URLs in {processing_time:.2f}s")

            return crawl_result, 0.0, metrics

        except asyncio.TimeoutError:
            error_msg = f"Request to crawl page server timed out, fail to use {self.crawl_page_endpoint}"
            logger.error(f"[CrawlPageV2Tool] {error_msg}")
            return json.dumps({"error": error_msg}), 0.0, {"error": "timeout"}

        except aiohttp.ClientError as e:
            error_msg = f"Failed to connect to crawl page server: {str(e)}, fail to use {self.crawl_page_endpoint}"
            logger.error(f"[CrawlPageV2Tool] {error_msg}")
            return json.dumps({"error": error_msg}), 0.0, {"error": "connection_error"}

        except Exception as e:
            error_msg = f"Crawl page execution failed: {str(e)}, fail to use {self.crawl_page_endpoint}"
            logger.error(f"[CrawlPageV2Tool] Unexpected error: {error_msg}")
            return json.dumps({"error": error_msg}), 0.0, {"error": "unexpected_error"}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward with respect to tool state.
        
        Args:
            instance_id: The instance ID of the tool
            
        Returns:
            The reward score
        """
        # Simple reward calculation based on successful execution
        if instance_id in self._instance_dict and "response" in self._instance_dict[instance_id]:
            response = self._instance_dict[instance_id]["response"]
            # Give reward based on content length and quality
            if response and len(response) > 100:  # Meaningful content
                return 1.0
            elif response:
                return 0.5
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.
        
        Args:
            instance_id: The instance ID of the tool
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            logger.debug(f"[CrawlPageV2Tool] Released instance {instance_id}")