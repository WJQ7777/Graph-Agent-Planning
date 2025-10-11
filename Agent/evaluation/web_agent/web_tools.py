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

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import requests
import re
from openai import OpenAI
import json
import os
import time
import random

######################################################
model = os.environ.get("SUMMARY_MODEL")

# Serper API
serper_key_str = os.getenv("WEB_SEARCH_SERPER_API_KEYS", "")

jina_key = os.getenv("JINA_API_KEY", "")

UNI_API_KEY = os.getenv("UNI_API_KEY", "")
MTU_API_KEY = os.getenv("MTU_API_KEY", "")

uni_urls_str = os.getenv("UNI_API_URLS", "")
uni_urls = [url.strip() for url in uni_urls_str.split("|") if url.strip()]
mtu_urls_str = os.getenv("MTU_API_URLS", "")
mtu_urls = [url.strip() for url in mtu_urls_str.split("|") if url.strip()]

api_pairs = []
for url in uni_urls:
    api_pairs.append({"api_url": url, "api_key": UNI_API_KEY})
for url in mtu_urls:
    api_pairs.append({"api_url": url, "api_key": MTU_API_KEY})


# ##################################################################################################################
def _format_results_to_string(serper_json: Dict[str, Any], query: str = None) -> str:
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


def WebSearchTool(web_search_url, api_key, api_url, model, task, query, history, topk=10):
    # think_content
    think_content = extract_last_tag(history, '<think>', '</think>')
    # web_search_query
    web_search_query = extract_last_tag(history, '<web_search>', '</web_search>')

    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": serper_key_str,
    }

    if topk > 20:
        topk = 20

    selected_url_key_group = random.choice(api_pairs)
    payload = {
        "q": query,
        "num": topk,
    }

    try:
        response = requests.post(web_search_url, headers=headers, json=payload)
        response.raise_for_status() 
        result = response.json()
        result = _format_results_to_string(result)
    except requests.exceptions.RequestException as e:
        result = f"An error occurred: {e}"
    return result


######################################################################################################
def extract_last_tag(text, start_tag, end_tag):
    end_index = text.rfind(end_tag)
    if end_index == -1:
        return ""
    start_index = text.rfind(start_tag, 0, end_index)
    if start_index == -1:
        return ""
    start_index += len(start_tag)
    return text[start_index:end_index]


def CrawlPageTool(crawl_page_url, api_key, api_url, model, task, urls, history):
    if isinstance(urls, str):
        urls = urls.split("|")
    # think_content
    think_content = extract_last_tag(history, '<think>', '</think>')
    # web_search_query
    web_search_query = extract_last_tag(history, '<web_search>', '</web_search>')
    selected_url_key_group = random.choice(api_pairs)
    data = {
        "urls": urls,
        "web_search_query": web_search_query,
        "think_content": think_content,
        "summary_prompt_type": "webthinker_with_goal", # support webthinker or webthinker_with_goal
        "summary_type": "page",
        "task": "", 
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(crawl_page_url, json=data, timeout=500, headers=headers)
    result = response.json()
    if result.get("success"):
        crawl_page_result = result["obs"]
    else:
        crawl_page_result = result.get("error_message")
    return crawl_page_result
