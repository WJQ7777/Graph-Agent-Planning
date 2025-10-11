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


"""
Test CrawlPage Server
Usage:
python test_crawl_page_simple_v2.py <endpoint_url>
e.g. python test_crawl_page_simple_v2.py http://127.0.0.1:9000/crawl_page
"""

import argparse
import json
import os
import time

import requests

parser = argparse.ArgumentParser(description='Crawl page test script v2.')
parser.add_argument('endpoint_url', type=str, help='The endpoint URL to test against (e.g., http://127.0.0.1:9000/crawl_page)')

args = parser.parse_args()

api_url = os.environ.get("SUMMARY_OPENAI_API_BASE_URL")
api_key = os.environ.get("SUMMARY_OPENAI_API_KEY")
model = os.environ.get("SUMMARY_MODEL")

if not all([api_url, api_key, model]):
    print("Error: Environment variable is not set.")
    print(f"api_url: {api_url}")
    print(f"api_key: {'set' if api_key else 'not set'}")
    print(f"model: {model}")
    exit(1)

data_once = {
    "urls": ["https://en.wikipedia.org/wiki/Qwen", "https://en.wikipedia.org/wiki/Alibaba_Cloud"],
    "task": "qwen is developed by?", 
    "web_search_query": "qwen is developed by?",
    "think_content": "to think...",
    "api_url": api_url,
    "api_key": api_key,
    "model": model,
    "summary_type": "once",
}

data_page = {
    "urls": ["https://en.wikipedia.org/wiki/Qwen", "https://en.wikipedia.org/wiki/Alibaba_Cloud"],
    "task": "qwen is developed by?", 
    "web_search_query": "qwen is developed by?",
    "think_content": "to think...",
    "api_url": api_url,
    "api_key": api_key,
    "model": model,
    "summary_type": "page",
}

print("Test start...")
print(f"API URL: {api_url}")
print(f"Model: {model}")

all_data = [data_once, data_page]

for data in all_data:
    print("\n" + "="*20)
    print(f"Testing Summary Type: {data.get('summary_type')}")
    print("="*20)
    try:
        url = args.endpoint_url
        response = requests.post(
            url,
            json=data,
            timeout=60,
            headers={"Content-Type": "application/json"}
        )
        
        response.raise_for_status()
        
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Response content: {response.text}")  # Print first 500 chars of response
            continue

        if result.get("success"):
            print("Success!")
            print(f"Process time: {result.get('processing_time'):.1f} second.")
            print("\nResult:")
            print("-" * 50)
            print(result.get('obs'))
            print("-" * 50)
        else:
            print(f"Fail: {result.get('error_message', 'unknown error')}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Connot connect to server!")
        break # Stop testing if connection fails
    except requests.exceptions.Timeout:
        print("Error: Request Timeout")
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unknown Error: {str(e)}")

print("\nAll test done.")