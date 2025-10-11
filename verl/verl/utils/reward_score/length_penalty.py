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


from transformers import AutoTokenizer
def calculate_length_penalty(solution_str: str, max_length: int = 8192) -> float:
    """
    计算文本长度惩罚值。
    
    Args:
        solution_str: 需要计算惩罚值的文本字符串
        max_length: 最大允许的token长度，默认为8192
        
    Returns:
        float: 惩罚值，范围在0到1之间
        1.0 表示没有惩罚
        接近0的值表示严重超出长度限制
    """    
    tokenizer = AutoTokenizer.from_pretrained("/home/notebook/code/group/open_source_model/qwen2.5/Qwen2.5-7B-Instruct")
    token_len = len(tokenizer.encode(solution_str))
    
    # 计算7/8的阈值
    threshold = int(max_length * 7 / 8)
    
    # 如果文本长度小于阈值，返回基础惩罚值1.0
    if token_len <= threshold:
        return 1.0
    
    # 计算超出阈值的部分和可用剩余空间
    excess = token_len - threshold
    remaining_space = max_length - threshold
    
    # 计算惩罚值：线性递减，从1.0到0.0
    penalty = 1.0 - (excess / remaining_space)
    
    # 确保惩罚值不会小于0
    return max(0.0, penalty)

def calculate_response_length_penalty(text_length: int, max_length: int = 8192) -> float:
    """
    计算文本长度的惩罚值。
    当文本长度超过最大长度的7/8时，惩罚值线性增大。
    
    参数:
        text_length (int): 当前文本长度
        max_length (int): 最大允许长度，默认为8192
        
    返回:
        float: 惩罚值，范围0~-1，随着长度增加而加大惩罚
    """
    # 计算7/8的阈值
    threshold = int(max_length * 7 / 8)
    
    # 计算超出阈值的部分
    excess = text_length - threshold
    
    # 计算可用的剩余空间
    remaining_space = max_length - threshold
    
    # 线性增加惩罚值
    # 当达到最大长度时，惩罚值将达到2.0
    # penalty = 1.0 - (excess / remaining_space)
    penalty = min(0.0, -(excess / remaining_space))

    return penalty