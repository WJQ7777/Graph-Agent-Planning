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
import atexit
import json
import logging
import os
import pickle
import threading
import time
from typing import Dict, Optional
from contextlib import asynccontextmanager

import httpx
import uvicorn
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request

from keys import get_serper_api

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SERPER_TIMEOUT = 60

# --- Cache Backend --- #


class CacheBackend:
    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    def __init__(self, file_path: str):
        self._cache = TTLCache(maxsize=1000000, ttl=3600)
        self._lock = threading.Lock()
        self.file_path = file_path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        # Open the file in append mode, creating it if it doesn't exist.
        self._file_handler = open(self.file_path, "ab")
        self._load_from_file()
        atexit.register(self._close_file)

    def _load_from_file(self):
        """Loads the cache from a pickle file."""
        try:
            with open(self.file_path, "rb") as f:
                i = 0
                while True:
                    try:
                        entry = pickle.load(f)
                        key, value = list(entry.items())[0]
                        self._cache[key] = value
                        i += 1
                    except EOFError:
                        break
                    except (IndexError) as e:
                        logger.warning(
                            f"Skipping malformed entry {i+1} in {self.file_path}: {e}"
                        )
            logger.info(
                f"Cache loaded from {self.file_path}, containing {len(self._cache)} items."
            )
        except FileNotFoundError:
            logger.info(
                f"Cache file {self.file_path} not found. A new one will be created."
            )
        except Exception as e:
            logger.warning(f"Could not load cache from {self.file_path}: {e}")

    def _close_file(self):
        """Closes the file handler on exit."""
        if self._file_handler:
            self._file_handler.close()
            logger.info("Cache file handler closed.")

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Sets a value in the cache and appends it to the pickle file."""
        with self._lock:
            if key not in self._cache:
                self._cache[key] = value
                try:
                    pickle.dump({key: value}, self._file_handler)
                    self._file_handler.flush()  # Ensure it's written to disk
                except IOError as e:
                    logger.error(f"Could not write to cache file {self.file_path}: {e}")



class SerperProxyServer:
    def __init__(self, cache_backend: CacheBackend, max_concurrent_requests: int = 2, requests_per_second: float = 1.0):
        self.cache = cache_backend
        # 限制并发请求数
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        # 速率限制
        self.requests_per_second = requests_per_second
        self.last_request_time = 0
        self.request_lock = asyncio.Lock()
        
        # 统计信息
        self.total_requests = 0
        self.cache_hits = 0
        self._stats_lock = threading.Lock()
    
    async def _rate_limit(self):
        """实现速率限制"""
        async with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            min_interval = 1.0 / self.requests_per_second
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                # logger.info(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = time.time()

    def _generate_cache_key(self, query: str) -> str:
        normalized_query = query.strip()
        return f"serper:{normalized_query}"

    def _log_hit_rate(self, *, hit: bool):
        """Logs the cache hit rate."""
        with self._stats_lock:
            self.total_requests += 1
            if hit:
                self.cache_hits += 1

            hit_rate = (
                (self.cache_hits / self.total_requests) * 100
                if self.total_requests > 0
                else 0
            )
            status = "HIT" if hit else "MISS"
            logger.info(
                f"Cache {status}. Rate: {hit_rate:.2f}% ({self.cache_hits}/{self.total_requests})"
            )

    async def process_request(self, request_data: dict, headers: dict) -> dict:
        start_time = time.time()
        query = request_data.get("q", "")
        original_num = request_data.get("num", 10)
        cache_key = self._generate_cache_key(query)

        try:
            # 先检查缓存
            cached_result_str = self.cache.get(cache_key)
            if cached_result_str:
                cached_result = json.loads(cached_result_str)
                cached_count = len(cached_result.get("organic", []))
                
                if cached_count >= original_num:
                    self._log_hit_rate(hit=True)
                    if "organic" in cached_result:
                        cached_result["organic"] = cached_result["organic"][:original_num]
                    logger.info(f"Request processed from cache in {time.time() - start_time:.2f} seconds.")
                    return cached_result

            # 需要调用 API 时进行速率限制
            async with self.semaphore:  # 限制并发数
                await self._rate_limit()  # 限制请求频率
                
                self._log_hit_rate(hit=False)
                logger.info(f"Forwarding to Serper API for key: {cache_key}")

                # 准备 API 请求
                api_request_data = request_data.copy()
                if cached_result_str:
                    api_request_data["num"] = 100
                else:
                    api_request_data["num"] = 10 if original_num <= 10 else 100

                selected_serper = get_serper_api()
                serper_url = selected_serper["url"]
                serper_key = selected_serper["key"]

                api_headers = {"Content-Type": "application/json", "X-API-KEY": serper_key}
                logger.info(f"Using Serper API: {serper_url} with key: {serper_key[:10]}...")

                async with httpx.AsyncClient(limits=httpx.Limits(max_connections=100)) as client:
                    response = await client.post(
                        serper_url,
                        json=api_request_data,
                        headers=api_headers,
                        timeout=request_data.get("timeout", SERPER_TIMEOUT),
                    )
                    response.raise_for_status()
                    full_result = response.json()

                    # 缓存完整结果
                    self.cache.set(cache_key, json.dumps(full_result))

                    # 返回裁剪后的结果
                    result_to_return = full_result.copy()
                    if "organic" in result_to_return:
                        result_to_return["organic"] = result_to_return["organic"][:original_num]

                    logger.info(f"Request processed via API in {time.time() - start_time:.2f} seconds.")
                    return result_to_return

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling Serper API: {e.response.status_code} {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"Unexpected error during Serper API request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


proxy_server: SerperProxyServer

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Serper Cache Start")
    # --- Application Initialization and Shutdown ---
    CACHE_FILE = "/home/jiaqi/Agent_Foundation_Models/AFM/tool_servers/cache/serper_api_cache.pkl"  # Switched to .pkl
    cache_backend = InMemoryCache(file_path=CACHE_FILE)
    app.state.proxy_server = SerperProxyServer(cache_backend=cache_backend)
    yield
    logger.info("Serper Cache Stop")

app = FastAPI(title="Serper API Proxy with Cache", lifespan=lifespan)

# The atexit registration is now handled within the InMemoryCache class


@app.post("/search")
async def serper_proxy_endpoint(request: Request):
    proxy_server: SerperProxyServer = request.app.state.proxy_server
    try:
        request_data = await request.json()
        headers = dict(request.headers)
        return await proxy_server.process_request(request_data, headers)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body.")
    except Exception as e:
        # Unhandled exceptions are caught and logged by the proxy_server now,
        # but we keep a general handler here as a fallback.
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Unhandled exception in endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    host = "0.0.0.0"
    port = int(os.getenv("WEBSEARCH_PORT", 9001))

    logger.info(f"Start Serper Server... http://{host}:{port}")
    uvicorn.run(
        "cache_serper_server_v2:app",
        host=host, 
        port=port, 
        workers=8,
    )