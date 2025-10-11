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
from typing import List, Tuple, Dict
from verl.tools.utils.code_executors.utils import try_extract_solution
import json

class FormatReward:
    def __init__(self):
        # Update tag set
        self.required_tags = ['<plan>', '<think>', '<code>', '<observation>', '<reflection>', '<answer>']
        self.paired_tags = {
            '<plan>': '</plan>',
            '<think>': '</think>',
            '<code>': '</code>',
            '<observation>': '</observation>',
            '<reflection>':  '</reflection>',
            '<answer>': '</answer>'
        }
        
    def check_tag_presence(self, text: str) -> Tuple[bool, List[str]]:
        """Check if all required tags are present"""
        missing_tags = []
        for tag in self.required_tags:
            if tag not in text:
                missing_tags.append(tag)
        return len(missing_tags) == 0, missing_tags
    
    def check_paired_tags(self, text: str) -> Tuple[bool, List[str]]:
        """Check if tags appear in correct pairs and are properly closed (including quantity matching and nesting order)"""
        unpaired_tags = []
        stack = []
        line_num = 1  # For error location (by line number)
        last_pos = 0   # Record last scan position
        
        # Preprocessing: split text by lines and record line numbers (for error location)
        lines = text.split('\n')
        
        # Scan all tags (including opening and closing tags)
        tag_pattern = re.compile(r'(</?(?:plan|think|code|observation|reflection|answer)>)')
        for line in lines:
            for match in tag_pattern.finditer(line):
                tag = match.group(1)
                pos = match.start() + last_pos
                
                # Determine if it's an opening or closing tag
                if tag.startswith('</'):
                    # Closing tag: check if it matches the opening tag at the top of the stack
                    if not stack:
                        unpaired_tags.append(f"Line {line_num} found extra closing tag: {tag}")
                    else:
                        expected_close = self.paired_tags.get(stack[-1])
                        if tag == expected_close:
                            stack.pop()  # Correctly closed
                        else:
                            unpaired_tags.append(
                                f"Line {line_num} tag closure mismatch: expected {expected_close}, actual {tag}"
                            )
                else:
                    # Opening tag: push to stack
                    stack.append(tag)
            
            line_num += 1
            last_pos += len(line) + 1  # +1 is for newline character
        
        # Check unclosed tags
        for tag in stack:
            unpaired_tags.append(f"Unclosed opening tag: {tag} (expected closing tag: {self.paired_tags[tag]})")
        
        # Check if quantities match (retain original logic)
        for open_tag, close_tag in self.paired_tags.items():
            open_count = text.count(open_tag)
            close_count = text.count(close_tag)
            if open_count != close_count:
                unpaired_tags.append(
                    f"Tag quantity mismatch: {open_tag} (appears {open_count} times) vs {close_tag} (appears {close_count} times)"
                )
        
        return len(unpaired_tags) == 0, unpaired_tags
    def check_paired_code_tags(self, text: str) -> Tuple[bool, List[str]]:
        """Check if tags appear in correct pairs and are properly closed (including quantity matching and nesting order)"""
        unpaired_tags = []
        stack = []
        line_num = 1  # For error location (by line number)
        last_pos = 0   # Record last scan position
        
        # Preprocessing: split text by lines and record line numbers (for error location)
        lines = text.split('\n')
        
        # Scan all tags (including opening and closing tags)
        tag_pattern = re.compile(r'(</?code>)')
        for line in lines:
            for match in tag_pattern.finditer(line):
                tag = match.group(1)
                pos = match.start() + last_pos
                
                # Determine if it's an opening or closing tag
                if tag.startswith('</'):
                    # Closing tag: check if it matches the opening tag at the top of the stack
                    if not stack:
                        unpaired_tags.append(f"Line {line_num} found extra closing tag: {tag}")
                    else:
                        expected_close = self.paired_tags.get(stack[-1])
                        if tag == expected_close:
                            stack.pop()  # Correctly closed
                        else:
                            unpaired_tags.append(
                                f"Line {line_num} tag closure mismatch: expected {expected_close}, actual {tag}"
                            )
                else:
                    # Opening tag: push to stack
                    stack.append(tag)
            
            line_num += 1
            last_pos += len(line) + 1  # +1 is for newline character
        
        # Check unclosed tags
        for tag in stack:
            unpaired_tags.append(f"Unclosed opening tag: {tag} (expected closing tag: {self.paired_tags[tag]})")
        
        # Check if quantities match (retain original logic)
        for open_tag, close_tag in self.paired_tags.items():
            open_count = text.count(open_tag)
            close_count = text.count(close_tag)
            if open_count != close_count:
                unpaired_tags.append(
                    f"Tag quantity mismatch: {open_tag} (appears {open_count} times) vs {close_tag} (appears {close_count} times)"
                )
        
        return len(unpaired_tags) == 0, unpaired_tags
    def check_code_tag_content(self, text: str) -> Tuple[bool, List[str]]:
        """
        Under the premise that <code> and </code> appear in pairs, check their internal format:
        1) Must exist and only exist one pair of ```py\n ... ```, and ``` must be the closest one to ```py\n.
        2) If </code> doesn't immediately appear after this pair of ```, then must exist one pair of ```sh\n ... ```, and ``` must be the closest one to ```sh\n.
        Returns (bool, [error message list])
        """
        errors = []

        # 1. Extract all content between <code>...</code>
        code_blocks = re.findall(r'<code>(.*?)</code>', text, flags=re.DOTALL)
        if not code_blocks:
            errors.append("No <code>...</code> content found")
            return False, errors

        # Iterate through each <code> block
        for idx, block in enumerate(code_blocks, start=1):
            # 2. Check ```py\n ... ``` is paired and unique
            py_start = '```py\n'
            py_end = '```\n'

            # Find all starting positions of ```py\n
            py_starts = [m.start() for m in re.finditer(re.escape(py_start), block)]
            if len(py_starts) != 1:
                errors.append(f"In <code> block {idx}, ```py\\n must appear exactly once, actual {len(py_starts)} times")
                continue

            start_pos = py_starts[0]

            # Find the next closest ```\n
            next_end_pos = block.find(py_end, start_pos + len(py_start))
            if next_end_pos == -1:
                errors.append(f"In <code> block {idx}, no matching ```\\n found for ```py\\n")
                continue

            # Ensure no other ```\n in between
            middle = block[start_pos + len(py_start): next_end_pos]
            if '```\n' in middle:
                errors.append(f"In <code> block {idx}, extra ```\\n found between ```py\\n and its corresponding ```\\n")
                continue

            # 3. Check if </code> immediately follows this pair of ```
            after_py = block[next_end_pos + len(py_end):]
            # Remove whitespace characters
            after_py_stripped = after_py.strip()
            if after_py_stripped != '':
                # Must exist one pair of ```sh\n ... ```
                sh_start = '```sh\n'
                sh_end = '```\n'

                sh_starts = [m.start() for m in re.finditer(re.escape(sh_start), after_py)]
                if len(sh_starts) != 1:
                    errors.append(f"In <code> block {idx}, after ```\\n, must find exactly one ```sh\\n")
                    continue

                sh_start_pos = sh_starts[0]

                sh_next_end_pos = after_py.find(sh_end, sh_start_pos + len(sh_start))
                if sh_next_end_pos == -1:
                    errors.append(f"In <code> block {idx}, no matching ```\\n found for ```sh\\n")
                    continue

                # Ensure no other ```\n in between
                sh_middle = after_py[sh_start_pos + len(sh_start): sh_next_end_pos]
                if '```\n' in sh_middle:
                    errors.append(f"In <code> block {idx}, extra ```\\n found between ```sh\\n and its corresponding ```\\n")
                    continue

                # Ensure no content after ```sh\n ... ```
                after_sh = after_py[sh_next_end_pos + len(sh_end):].strip()
                if after_sh != '':
                    errors.append(f"In <code> block {idx}, extra content found after ```sh\\n ... ```\\n")
                    continue

        return len(errors) == 0, errors
    def check_tag_content(self, text: str) -> Tuple[bool, List[str]]:
        """Check if tag content is valid (not empty, no illegal nesting of other tags)"""
        issues = []
        for open_tag, close_tag in self.paired_tags.items():
            pattern = f"{open_tag}(.*?){close_tag}"
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                content = match.group(1)
                if not content.strip():
                    issues.append(f"{open_tag} tag content is empty")
                for other_tag in self.required_tags:
                    if other_tag in content:
                        issues.append(f"{open_tag} tag contains {other_tag} tag")
        return len(issues) == 0, issues
    
    def check_sequence(self, text: str) -> Tuple[bool, List[str]]:
        """Check if tag sequence is correct"""
        # Extract all tags
        tags = re.findall(r'<(?:think|code|observation|reflection|answer)>', text)
        
        # Check basic sequence rules
        if not tags:
            return False, []
            
        # Check if starts with <think>
        if tags[0] != '<think>':
            return False, tags
            
        # Check if ends with <answer>
        if tags[-1] != '<answer>':
            return False, tags
            
        # Check middle sequence
        for i in range(1, len(tags)-1):
            current_tag = tags[i]
            next_tag = tags[i+1]
            
            # Check that code must be followed by observation
            if current_tag == '<code>':
                if next_tag != '<observation>':
                    return False, tags
            # Check that observation must be followed by reflection
            elif current_tag == '<observation>':
                if next_tag != '<reflection>':
                    return False, tags
            # Check that reflection must be followed by think
            elif current_tag == '<reflection>':
                if next_tag != '<think>':
                    return False, tags
            # Check that think can be followed by code or answer
            elif current_tag == '<think>':
                if next_tag not in ['<code>', '<answer>']:
                    return False, tags
                    
        return True, tags
    
    def check_format_correctness(self, text: str) -> bool:
        """Check if tag format is correct (including tag pairing and content correctness)"""
        paired_correct, _ = self.check_paired_tags(text)
        content_correct, _ = self.check_tag_content(text)
        return paired_correct and content_correct
    def check_code_format_correctness(self, text: str) -> bool:
        """Check if tag format is correct (including tag pairing and content correctness)"""
        paired_correct, a = self.check_paired_code_tags(text)
        if paired_correct:
            content_correct, b = self.check_code_tag_content(text)
        else:
            content_correct = False
        return paired_correct and content_correct
    def calculate_reward(self, text: str) -> Dict:
        # Extract
        
        """Calculate format reward score"""
        # Check tag presence
        tags_present, missing_tags = self.check_tag_presence(text)
        
        # Check sequence correctness
        sequence_correct, tag_sequence = self.check_sequence(text)
        
        # Check tag pairing
        paired_correct, unpaired_tags = self.check_paired_tags(text)
        
        # Check content between tags
        content_correct, content_issues = self.check_tag_content(text)
        
        # Calculate reward score
        reward = 0.0
        if tags_present:
            reward += 0.25
        if sequence_correct:
            reward += 0.25
        if paired_correct:
            reward += 0.25
        if content_correct:
            reward += 0.25

            
        return {
            'reward': reward,
            'tags_present': tags_present,
            'sequence_correct': sequence_correct,
            'paired_correct': paired_correct,
            'content_correct': content_correct,
        }

def main():
    perfect_text = '''
    <think>I need to solve this problem step by step</think>
    <code>print("Hello World")</code>
    <observation>The program output Hello World</observation>
    <think>Now I need to calculate the sum</think>
    <code>sum = 1 + 2</code>
    <observation>The calculation result is 3</observation>
    <answer>The final answer is 3</answer>
    '''

    missing_tags_text = '''
    <think>Text missing required tags</think>
    <code>x = 5</code>
    <observation>The value of x is 5</observation>
    '''
    
    unpaired_tags_text = '''
    <think>Text with unpaired tags</think>
    <code>y = 10</code>
    <observation>The value of y is 10
    <think>Proceed to next step</think>
    <answer>The result is 10</answer>
    '''

    nested_tags_text = '''
    <think>Text with incorrect tag nesting<code>print(1)</code></think>
    <observation>Output 1</observation>
    <answer>Complete</answer>
    '''

    empty_content_text = '''
    <think>Text with empty tag content</think>
    <code></code>
    <observation>   </observation>
    <answer>Answer</answer>
    '''

    wrong_sequence_text = '''
    <code>print("Wrong start")</code>
    <think>Text with incorrect tag sequence</think>
    <observation>Observation result</observation>
    <answer>End</answer>
    '''

    mismatched_count_text = '''
    <think>Text with mismatched tag counts</think>
    <code>a = 1</code>
    <observation>a is 1</observation>
    <code>b = 2</code>
    <observation>b is 2</observation>
    <think>Need more thinking</think>
    <answer>Result</answer>
    </answer>
    '''

    complex_correct_text = '''
    <think>First phase thinking</think>
    <code>result = 0</code>
    <observation>Initialize result to 0</observation>
    <think>Second phase thinking</think>
    <code>result += 5</code>
    <observation>result is now 5</observation>
    <code>result *= 2</code>
    <observation>result is now 10</observation>
    <think>Final confirmation</think>
    <answer>The final result is 10</answer>
    '''

    # Initialize FormatReward class
    reward_checker = FormatReward()
    
    # Define all test texts
    test_cases = [
        ("Perfect format text", perfect_text),
        ("Text missing required tags", missing_tags_text),
        ("Text with unpaired tags", unpaired_tags_text),
        ("Text with incorrect tag nesting", nested_tags_text),
        ("Text with empty tag content", empty_content_text),
        ("Text with incorrect tag sequence", wrong_sequence_text),
        ("Text with mismatched tag counts", mismatched_count_text),
        ("Complex but correct text", complex_correct_text)
    ]
    
    # Test each text
    for name, text in test_cases:
        print(f"\n{'='*50}")
        print(f"Test case: {name}")
        print(f"{'='*50}")
        
        # Check tag presence
        tags_present, missing_tags = reward_checker.check_tag_presence(text)
        print(f"\n1. Tag presence check:")
        print(f"Result: {'Pass' if tags_present else 'Fail'}")
        if missing_tags:
            print(f"Missing tags: {missing_tags}")
        
        # Check tag pairing
        paired_correct, unpaired_tags = reward_checker.check_paired_tags(text)
        print(f"\n2. Tag pairing check:")
        print(f"Result: {'Pass' if paired_correct else 'Fail'}")
        if unpaired_tags:
            print("Issues found:")
            for issue in unpaired_tags:
                print(f"- {issue}")
        
        # Check tag content
        content_correct, content_issues = reward_checker.check_tag_content(text)
        print(f"\n3. Tag content check:")
        print(f"Result: {'Pass' if content_correct else 'Fail'}")
        if content_issues:
            print("Issues found:")
            for issue in content_issues:
                print(f"- {issue}")
        
        # Check tag sequence
        sequence_correct, tag_sequence = reward_checker.check_sequence(text)
        print(f"\n4. Tag sequence check:")
        print(f"Result: {'Pass' if sequence_correct else 'Fail'}")
        if not sequence_correct:
            print(f"Actual tag sequence: {tag_sequence}")
        
        # Calculate comprehensive reward score
        result = reward_checker.calculate_reward(text)
        print(f"\n5. Comprehensive reward score:")
        print(f"Total score: {result['reward']:.2f}")
        print(f"Detailed results:")
        print(f"- Tag presence: {'Pass' if result['tags_present'] else 'Fail'}")
        print(f"- Tag sequence: {'Pass' if result['sequence_correct'] else 'Fail'}")
        print(f"- Tag pairing: {'Pass' if result['paired_correct'] else 'Fail'}")
        print(f"- Tag content: {'Pass' if result['content_correct'] else 'Fail'}")

        
    
