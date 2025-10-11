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
import re
from typing import List, Dict, Any, Union
from openai import OpenAI
import tiktoken


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error in read_jsonl: {str(e)}")
    return data


def write_jsonl(
    data: List[Dict[str, Any]], 
    file_path: str, 
    append: bool = False, 
    ensure_ascii: bool = False
) -> bool:
    try:
        mode = 'a' if append else 'w'
        with open(file_path, mode, encoding='utf-8') as file:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=ensure_ascii) + '\n'
                file.write(json_line)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def read_json(file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def write_json(
    data: Union[Dict[str, Any], List[Dict[str, Any]]], 
    file_path: str, 
    indent: int = 2, 
    ensure_ascii: bool = False,
    sort_keys: bool = False
) -> bool:
    try:
        with open(file_path, "w", encoding='utf-8') as file:
            json.dump(
                data, 
                file, 
                indent=indent, 
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys
            )
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def count_tokens(text, tokenizer):
    if not text:
        return 0
    return len(tokenizer.encode(text))


def retry_predict(api_key, api_url, model, prompt, developer_prompt=None):
    client = OpenAI(api_key=api_key, base_url=api_url, timeout=180)
    messages = []
    if developer_prompt:
        messages.append({
            "role": "system",
            "content": developer_prompt
        })
    messages.append({
        "role": "user",
        "content": prompt
    })

    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.6,
                max_tokens=8192,
                top_p=0.9,
                frequency_penalty=0,
                presence_penalty=0
            )
            content = response.choices[0].message.content.strip()
            if content:
                return content
        except Exception as e:
            print(f"OpenAI API Error: {e}, retrying...")
    return ""


def truncate_special_tokens(text: str, max_tokens: int, tokenizer: str) -> str:
    token_pattern = r'<[^>]+>|</[^>]+>'
    
    segments = re.split(f'({token_pattern})', text)
    segments = [seg for seg in segments if seg] 
    
    if not segments:
        return ""
    
    segment_tokens = []
    for segment in segments:
        tokens = tokenizer.encode(segment)
        segment_tokens.append((segment, len(tokens)))
    
    keep_tags = ['plan', 'reflection']
    keep_tag_indices = []
    
    for i, (segment, _) in enumerate(segment_tokens):
        tag_match = re.match(r'<(/?)([^>]+)>', segment)
        if tag_match:
            is_closing = tag_match.group(1) == '/'
            tag_name = tag_match.group(2)
            if tag_name in keep_tags:
                keep_tag_indices.append((i, tag_name, is_closing))
    
    special_indices = set()
    
    for i, tag_name, is_closing in keep_tag_indices:
        if is_closing:
            start_idx = -1
            depth = 0
            for j in range(i, -1, -1):
                seg = segment_tokens[j][0]
                m = re.match(r'<(/?)([^>]+)>', seg)
                if m:
                    current_tag = m.group(2)
                    if current_tag == tag_name:
                        if m.group(1) == '/':
                            depth += 1
                        else:
                            depth -= 1
                            if depth == 0:
                                start_idx = j
                                break
            if start_idx != -1:
                for idx in range(start_idx, i + 1):
                    special_indices.add(idx)
        else:
            end_idx = -1
            depth = 0
            for j in range(i, len(segments)):
                seg = segment_tokens[j][0]
                m = re.match(r'<(/?)([^>]+)>', seg)
                if m:
                    current_tag = m.group(2)
                    if current_tag == tag_name:
                        if m.group(1) == '/':
                            depth -= 1
                            if depth == 0:
                                end_idx = j
                                break
                        else:
                            depth += 1
            if end_idx != -1:
                for idx in range(i, end_idx + 1):
                    special_indices.add(idx)
    
    special_segments = [segment_tokens[i] for i in sorted(special_indices)]
    special_tokens = sum(token_count for _, token_count in special_segments)
    
    if special_tokens > max_tokens:
        result = []
        current_tokens = 0
        
        for segment, token_count in reversed(special_segments):
            if current_tokens + token_count <= max_tokens:
                result.insert(0, segment)
                current_tokens += token_count
            else:
                if re.match(r'</[^>]+>', segment):
                    tag_name = re.search(r'</([^>]+)>', segment).group(1)
                    start_tag_found = False
                    for s, _ in reversed(result):
                        if re.match(rf'<{tag_name}>', s):
                            start_tag_found = True
                            break
                    if not start_tag_found:
                        for s, tc in reversed(special_segments):
                            if re.match(rf'<{tag_name}>', s):
                                if current_tokens + tc <= max_tokens:
                                    result.insert(0, s)
                                    current_tokens += tc
                                break
        return ''.join(result) if result else segments[0]
    
    remaining_tokens = max_tokens - special_tokens
    additional_segments = []
    
    for i in range(len(segments) - 1, -1, -1):
        if i in special_indices:
            continue 
        
        segment, token_count = segment_tokens[i]
        
        if remaining_tokens <= 0:
            break
        
        if token_count <= remaining_tokens:
            additional_segments.insert(0, segment)
            remaining_tokens -= token_count
        else:
            if not re.match(r'<[^>]+>|</[^>]+>', segment):
                tokens = tokenizer.encode(segment)
                partial_tokens = tokens[:remaining_tokens]
                partial_text = tokenizer.decode(partial_tokens)
                if partial_text:
                    additional_segments.insert(0, partial_text)
                    remaining_tokens = 0
    
    return ''.join([seg for seg, _ in special_segments] + additional_segments)