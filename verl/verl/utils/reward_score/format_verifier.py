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
from typing import List, Set


def validate_format(text: str, allow_tags_name: List[str]) -> bool:
    """Validate text format
    
    Args:
        text: Text to be validated
        allow_tags_name: List of allowed tag names
    
    Returns:
        bool: Whether the format is valid
    """
    if not text.strip():
        return False
    
    # Extract all tag information
    tag_pattern = r'<(/?)(\w+)>'
    matches = list(re.finditer(tag_pattern, text))
    
    if not matches:
        return False

    # Count tags and build tag sequence
    tag_counts = {}
    tag_sequence = []
    allowed_tags_set = set(allow_tags_name)
    
    for match in matches:
        is_close = match.group(1) == '/'
        tag_name = match.group(2)
        position = match.start()
        
        # Check if tag name is within allowed range
        if tag_name not in allowed_tags_set:
            return False
        
        # Count tags
        if tag_name not in tag_counts:
            tag_counts[tag_name] = {'open': 0, 'close': 0}
        
        if is_close:
            tag_counts[tag_name]['close'] += 1
            tag_sequence.append((position, tag_name, 'close'))
        else:
            tag_counts[tag_name]['open'] += 1
            tag_sequence.append((position, tag_name, 'open'))

    # Sort tag sequence by position
    tag_sequence.sort(key=lambda x: x[0])
    
    # Check if the last tag is answer's closing tag
    if not (tag_sequence and tag_sequence[-1][1] == 'answer' and tag_sequence[-1][2] == 'close'):
        return False
    
    # Check answer tag count (must be exactly one)
    if tag_counts.get('answer', {}).get('open', 0) != 1:
        return False
    
    # Check all tags appear in pairs
    for tag_name, counts in tag_counts.items():
        if counts['open'] != counts['close']:
            return False

    # Check adjacent open tags are not repeated
    open_tags = [tag_name for _, tag_name, tag_type in tag_sequence if tag_type == 'open']
    for i in range(len(open_tags) - 1):
        if open_tags[i] == open_tags[i + 1]:
            return False
    
    # Check tag pair content is non-empty and nesting is not allowed
    tag_stack = []
    
    for pos, tag_name, tag_type in tag_sequence:
        if tag_type == 'open':
            # If nesting is not allowed, stack should not have other tags
            if tag_stack:
                return False  # Nesting not allowed
            tag_stack.append((tag_name, pos))
        else:  # close tag
            if not tag_stack or tag_stack[-1][0] != tag_name:
                return False  # Tags don't match
            
            open_pos = tag_stack.pop()[1]
            # Calculate tag content (excluding the tags themselves)
            open_end = text.find('>', open_pos) + 1
            close_start = pos
            content = text[open_end:close_start].strip()
            
            if not content:
                return False  # Tag content is empty
    
    return True