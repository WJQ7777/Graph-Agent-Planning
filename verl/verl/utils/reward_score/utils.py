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


import re

def verify_format_simple(response):
    """
    1. 提取 xml tag 必须是成对的，不能有未闭合的 tag，不允许 tag 嵌套。
    2. 相邻的 tag 必须成对出现
    """
    # breakpoint()
    opening_pattern = r'<(\w+)>'
    closing_pattern = r'</(\w+)>'
    opening_tags = re.findall(opening_pattern, response)
    closing_tags = re.findall(closing_pattern, response)

    if len(opening_tags) == 0 or len(closing_tags) == 0:
        return False

    if len(opening_tags) != len(closing_tags):
        print("[INFO] 格式错误! opening & closing tags 不匹配")
        return False
    for i in range(len(opening_tags)):
        if opening_tags[i] != closing_tags[i]:
            print("[INFO] 格式错误! opening & closing tags 不一一对应(可能存在标签嵌套或者错位)")
            return False
    if opening_tags[-1] != 'answer':
        print("[INFO] 格式错误! 最后一个tag 必须是 answer!")
        return False
    
    return True


def _content_repetition(response):
    """
    检查response中所有XML标签内容是否存在重复
    如果存在重复内容，返回False；否则返回True
    """
    # 提取所有 <tag>content</tag> 格式的内容, 除了 answer 和 suggested_answer
    pattern = r'<(?!answer|suggested_answer)\b(\w+)\b>(.*?)</\1>'
    contents = re.findall(pattern, response, re.DOTALL)
    
    # 清理内容：去除首尾空白字符
    # findall with two groups returns a list of tuples, we need the second element (the content)
    cleaned_contents = [content.strip() for tag_name, content in contents if content.strip()]

    # 检查是否有重复内容
    if len(cleaned_contents) != len(set(cleaned_contents)):
        print("[INFO] 发现重复内容，格式错误")
        return False
    return True


def verify_format_repetition(response):
    """
    1. 不允许连续出现 not_rep = ["plan", "reflection", "suggested_answer"]
    2. 不允许重复的 web_search query & crawl_page query
    """
    # # 基础格式错误，直接返回
    # if not verify_format_simple(response):
    #     return False

    # 不允许 web_search query & crawl_page query 重复
    web_search_queries = re.findall(r'<web_search>(.*?)</web_search>', response, re.DOTALL)
    crawl_page_queries = re.findall(r'<crawl_page>(.*?)</crawl_page>', response, re.DOTALL)
    web_search_queries = [query.strip() for query in web_search_queries if query.strip()]
    crawl_page_queries = [query.strip() for query in crawl_page_queries if query.strip()]
    
    # 检查是否有重复的查询
    if len(web_search_queries) != len(set(web_search_queries)):
        print("[INFO] 发现重复的 web_search 查询")
        return False
    if len(crawl_page_queries) != len(set(crawl_page_queries)):
        print("[INFO] 发现重复的 crawl_page 查询")
        return False
    
    if not _content_repetition(response):
        return False

    return True