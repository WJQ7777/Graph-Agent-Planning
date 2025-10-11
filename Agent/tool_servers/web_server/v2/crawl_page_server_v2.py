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
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
import uvicorn
from cachetools import TTLCache
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from keys import get_qwen_api, get_jina_api

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CRAWL_PAGE_TIMEOUT = 500

class CrawlPageRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to crawl")
    task: str = Field(..., description="Task description")
    web_search_query: str = Field(..., description="Web search query")
    think_content: str = Field(..., description="Thinking content for summarization")
    messages: Optional[List[Dict]] = Field(None, description="Message history (optional, for future use)")
    summary_type: Optional[str] = Field("page", description="Summary type")
    summary_prompt_type: Optional[str] = "webthinker_with_goal" # webthinker or webthinker_with_goal


class CrawlPageResponse(BaseModel):
    success: bool
    obs: str
    error_message: Optional[str] = None
    processing_time: float

class CrawlPageServer:
    def __init__(self):
        logger.info("Initializing CrawlPageServer")
        self.cache = TTLCache(maxsize=10000, ttl=3*3600)  
        self.jina_timeout = 30
        self.summary_timeout = 300
        self.jina_token_budget = 80000
        self.max_retries = 5
        logger.info(f"CrawlPageServer initialized with jina_timeout={self.jina_timeout}s, summary_timeout={self.summary_timeout}s, token_budget={self.jina_token_budget}, max_retries={self.max_retries}")

    async def read_page_async(self, session: aiohttp.ClientSession, url: str) -> tuple[str, str]:
        logger.debug(f"Starting to read page: {url}")
        
        if url in self.cache:
            return self.cache[url]
            
        attempt = 0
        last_exc = None
        
        while attempt < 2:
            attempt += 1
            try:
                # Choose a Jina API
                selected_jina_api = get_jina_api()
                jina_key = selected_jina_api["key"]
                logger.info(f"[Attempt {attempt}/{2}] Fetching {url} with Jina key: {jina_key[:10]}...")
                
                jina_url = f'https://r.jina.ai/{url}'
                headers = {
                    'Authorization': f'Bearer {jina_key}',
                    'X-Engine': 'browser',
                    'X-Return-Format': 'text',
                    "X-Remove-Selector": "header, .class, #id",
                    'X-Timeout': f"{self.jina_timeout}",
                    "X-Retain-Images": "none",
                    'X-Token-Budget': f"{self.jina_token_budget}"
                }
                
                async with session.get(jina_url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.jina_timeout)) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Successfully fetched {url}, content length: {len(content)} chars")
                        self.cache[url] = (content, url)
                        return (content, url)
                    else:
                        last_exc = f"HTTP {response.status}"
                        logger.warning(f"[Attempt {attempt}] Failed to fetch {url}: HTTP {response.status}")
                        
            except asyncio.TimeoutError:
                last_exc = f"Timeout after {self.jina_timeout}s"
                logger.warning(f"[Attempt {attempt}] Timeout for {url}")
            except Exception as e:
                last_exc = str(e)
                logger.warning(f"[Attempt {attempt}] Error for {url}: {e}")
            
            if attempt < 2:
                delay = 30 * attempt  # 30s
                logger.info(f"Wait for {delay}s...")
                await asyncio.sleep(delay)
        
        logger.error(f"All Jina attempts fail for url: {url}")
        return (f"[Page content not accessible: {last_exc}]", url)


    def validate_urls(self, urls: List[str]) -> List[str]:
        logger.debug(f"Validating {len(urls)} URLs")
        processed_urls = []
        for url in urls:
            url = url.strip()
            if not url:
                continue
                
            markdown_match = re.search(r'\[.*?\]\((https?://[^\s\)]+)\)', url)
            if markdown_match:
                processed_urls.append(markdown_match.group(1))
            elif url.startswith(('http://', 'https://')):
                processed_urls.append(url)
            else:
                url_match = re.search(r'(https?://[^\s]+)', url)
                if url_match:
                    processed_urls.append(url_match.group(1))
                else:
                    logger.warning(f"Invalid URL format: {url}")
        
        logger.info(f"Validated URLs: {len(processed_urls)} valid out of {len(urls)} total")
        return processed_urls
    

    def get_click_intent_instruction(self, prev_reasoning: str) -> str:
        return f"""Based on the previous thoughts below, provide the detailed intent of the latest click action.
Previous thoughts: {prev_reasoning}
Please provide the current click intent."""

    async def get_summary_prompt(self, task: str, web_search_query: str, think_content: str, page_contents: str, summary_prompt_type: str = "webthinker") -> str:
        click_intent = ""
        if summary_prompt_type == "webthinker_with_goal":
            intent_prompt = self.get_click_intent_instruction(think_content)
            click_intent = await self.call_ai_api_async(
                "You are a summary agent robot.", intent_prompt
            )
            logger.info(f"[INFO] Get click intent: {click_intent}")

        return f"""Target: Extract all content from a web page that matches a specific web search query, clues and ideas, ensuring completeness and relevance. (No response/analysis required.)

web search query: {web_search_query}

Clues and ideas: {click_intent or think_content}

Searched Web Page: {page_contents}

Important Notes:
- Summarize all content (text, tables, lists, code blocks) into concise points that directly address query, clues and ideas.
- Preserve and list all relevant links ([text](url)) from the web page.
- Summarize in three points: web search query-related information, clues and ideas-related information, and relevant links with descriptions.
- If no relevant information exists, Just output "No relevant information"
"""
    
    async def call_ai_api_async(self, system_prompt: str, user_prompt: str) -> str:
        selected_qwen_api = get_qwen_api()
        api_url, api_key, model = selected_qwen_api["url"], selected_qwen_api["key"], selected_qwen_api["model"]
        
        logger.info(f"Calling AI API with model: {model}, API URL: {api_url}, max_retries: {self.max_retries}")
        
        # LLM API retry：60s, 120s, 240s, 480s
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            attempt += 1
            try:
                logger.info(f"[Attempt {attempt}/{self.max_retries}] Calling AI API...")
                client = AsyncOpenAI(base_url=api_url, api_key=api_key)
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=False,
                    timeout=self.summary_timeout
                )
                content = completion.choices[0].message.content
                logger.info(f"AI API response received, length: {len(content)} chars")
                return content
            except Exception as e:
                last_error = str(e)

                if "data_inspection_failed" in last_error:
                    logger.error(f"AI API call failed due to data inspection: {last_error}")
                    return f"AI Process Fail due to data inspection:  ({last_error})"
                
                logger.warning(f"[Attempt {attempt}] AI API Fail: {last_error}")
                if attempt < self.max_retries:
                    delay = 60 * (2 ** (attempt - 1))  # 60s -> 120s -> 240s -> 480s
                    logger.info(f"Wait for {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("All attmpt fail for API call.")
        
        return f"AI API fail after {self.max_retries} attmpts: {last_error}"
        
    async def summarize_content(self, content: str, request: CrawlPageRequest) -> str:
        """Helper function to summarize content"""
        logger.debug(f"Summarizing content of length: {len(content)} chars")
        detailed_prompt = await self.get_summary_prompt(
            request.task, 
            request.web_search_query, 
            request.think_content, 
            content,
            request.summary_prompt_type
        )
        return await self.call_ai_api_async("You are a summary agent robot.", detailed_prompt)
    
    async def process_crawl_page(self, request: CrawlPageRequest) -> CrawlPageResponse:
        start_time = time.time()
        
        try:
            logger.info("--------- Start crawl_page process ---------")
            logger.info(f"Request URLs count: {len(request.urls)}, web_search_query: {request.web_search_query}")
            # task = request.task
            # think_content = request.think_content
            # web_search_query = request.web_search_query
            # messages = request.messages
            urls = self.validate_urls(request.urls)
            if not urls:
                logger.warning("No valid URLs found after validation")
                return CrawlPageResponse(
                    success=False,
                    obs="",
                    error_message="No valid URL found",
                    processing_time=time.time() - start_time
                )
            
            logger.info(f"Start process {len(urls)} URLs...")
            for i, url in enumerate(urls):
                logger.debug(f"URL {i+1}: {url}")
            
            page_contents = ""
            logger.info("Creating aiohttp session for page fetching")
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=100)) as session:
                tasks = [self.read_page_async(session, url) for url in urls]
                try:
                    logger.info(f"Fetching {len(tasks)} pages concurrently...")
                    page_results = await asyncio.gather(*tasks, return_exceptions=True)
                    logger.info(f"Completed fetching {len(page_results)} pages")
                    
                    processed_results = []
                    for i, result in enumerate(page_results):
                        if isinstance(result, Exception):
                            logger.error(f"Exception for URL {urls[i]}: {str(result)}")
                            processed_results.append((f"[Page content not accessible: {str(result)}]", urls[i]))
                        else:
                            processed_results.append(result)
                    page_results = processed_results
                    
                except Exception as e:
                    logger.error(f"An unexpected error occurred during page fetching: {e}", exc_info=True)
                    page_results = [(f"[Page content not accessible: Server error during fetch]", url) for url in urls]
            
            ##### End Jina read page #####
            logger.info("Page fetching completed, starting summarization")
            ##### Start page summary #####

            logger.info(f"Using summary type: 'page'")
            page_tasks = []
            page_indices = []
            for i, (content, url) in enumerate(page_results):
                logger.info(f"Creating task for page {i+1}/{len(page_results)}, URL: {url}, content length: {len(content) / 1000:.2f}k characters")
                if content.startswith("[Page content not accessible:"):
                    page_indices.append((i, False))
                else:
                    task = self.summarize_content(content, request)
                    page_tasks.append(task)
                    page_indices.append((i, True))
            
            if page_tasks:
                logger.info(f"Processing {len(page_tasks)} pages concurrently")
                page_results_summary = await asyncio.gather(*page_tasks, return_exceptions=True)
            else:
                page_results_summary = []
            
            page_summaries = []
            summary_idx = 0
            for i, needs_summary in page_indices:
                content, url = page_results[i]
                if not needs_summary:
                    page_summaries.append(f"Page {i+1} [{url}]: {content}")
                else:
                    result = page_results_summary[summary_idx]
                    summary_idx += 1
                    if isinstance(result, Exception):
                        logger.error(f"Error processing page {i+1} [{url}]: {str(result)}")
                        page_summaries.append(f"Page {i+1} [{url}] Summary:\n[Error: {str(result)}]")
                    else:
                        page_summaries.append(f"Page {i+1} [{url}] Summary:\n{result}")

            logger.info("Concatenating page summaries without final summary")
            summary_result = "\n\n".join(page_summaries)
            
            processing_time = time.time() - start_time
            logger.info(f"Process done: {processing_time:.2f}sec, length: {len(summary_result)} chars")
            logger.info("--------- Successful ---------")
            
            return CrawlPageResponse(
                success=True,
                obs=summary_result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error: {str(e)}", exc_info=True)
            logger.error("--------- Fail ---------")
            return CrawlPageResponse(
                success=False,
                obs="",
                error_message=f"Error: {str(e)}",
                processing_time=processing_time
            )


crawl_server = CrawlPageServer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CrawlPage Start")
    yield
    logger.info("CrawlPage Stopped")

app = FastAPI(
    title="CrawlPage Server",
    description="CrawlPage Server",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/crawl_page", response_model=CrawlPageResponse)
async def crawl_page_endpoint(request: CrawlPageRequest):
    logger.info(f"Received crawl_page request from client")
    """
    CrawlPage Endpoint
    
    Param:
    - urls: List[str] - url list to be crawled
    - task: str - original question
    - think_content: str - thinking content
    - messages: Optional[List[Dict]] - message history (optional)
    """
    try:
        result = await asyncio.wait_for(
            crawl_server.process_crawl_page(request),
            timeout=CRAWL_PAGE_TIMEOUT
        )
        logger.info(f"Request completed successfully, success={result.success}")
        return result
    except asyncio.TimeoutError:
        logger.error(f"Request timeout after {CRAWL_PAGE_TIMEOUT}s")
        raise HTTPException(status_code=504, detail=f"Timeout: {CRAWL_PAGE_TIMEOUT}s")
    except Exception as e:
        logger.error(f"Server Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


@app.get("/health")
async def health_check():
    logger.debug("Health check requested")
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/")
async def root():
    return {
        "message": "CrawlPage Server",
        "version": "1.0.0",
        "endpoints": {
            "crawl_page": "/crawl_page",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    host = "0.0.0.0"
    port = int(os.getenv("CRAWL_PAGE_PORT", 9000))
    
    logger.info(f"Start CrawlPage Server... http://{host}:{port}")
    uvicorn.run(
        "crawl_page_server_v2:app",
        host=host, 
        port=port, 
        workers=50
    )