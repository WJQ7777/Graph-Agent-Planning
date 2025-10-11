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
Simple test for Wiki RAG server
Demonstrates basic input/output usage
"""

import requests
import json

def test_wiki_rag_simple():
    """Simple test for Wiki RAG server"""
    
    url = "http://127.0.0.1:8000/retrieve"
    
    query = "machine learning algorithms"
    payload = {
        "queries": [query],
        "topk": 3,
        "return_scores": True
    }
    
    print("=== Request ===")
    print(f"Query: {query}")
    print(f"Param: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n=== Result ===")
        print(f"Status: {response.status_code}")
        print(f"Doc: {len(result['result'][0])}")

        contents = [doc["document"]["contents"] for doc in result["result"][0]]
        merged_content = "\n\n".join(contents)
        
        print(f"\n=== Merged Result ===")
        print(merged_content)
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Fail: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Wiki RAG Test")
    print("=" * 40)
    
    success = test_wiki_rag_simple()
    
    if success:
        print("\nTest Seccessful!")
    else:
        print("\nTest Fail!")
