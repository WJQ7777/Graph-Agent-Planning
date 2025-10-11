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
Convert parquet or jsonl files into a valid training format with schema validation.
This script is refactored for clarity, flexibility, and robustness.
"""

import argparse
import json
import os
from typing import Any, Dict, List
import uuid

import datasets
from sys_prompts import (SR1_SYS_PROMPT,
                        function_2_tool_reflection_doublecheck,
                        AFM_TOOLS)

# --- Constants for maintainability ---
# A mapping for system prompt keys to their actual content
SYS_PROMPT_MAP = {
    'SR1': SR1_SYS_PROMPT,
    'DOUBLE_CHECK_RULE': function_2_tool_reflection_doublecheck,
    'AFM': AFM_TOOLS
}

# Define tool names as constants
WIKI_SEARCH = "wiki_search"
WEB_SEARCH = "web_search"
CRAWL_PAGE = "crawl_page"
CODE = "code"
VISUAL_INSPECTOR = "visual_inspector"

# --- Core Processing Functions ---

def process_example(
    example: Dict[str, Any],
    sys_prompt: str,
    tool_type: str = 'single',
    filter_none: bool = False
) -> Dict[str, Any]:
    """
    Validates and converts a single example to the target schema.
    
    Args:
        example: The input data dictionary.
        sys_prompt: The system prompt string to use.
        tool_type: 'single' for just wiki_search, 'three' for all three tools.
        filter_none: If True, return None for examples with None/empty values.
        
    Returns:
        A dictionary structured according to the target schema, or None if filtered.
    """
    # 1. Validate input data
    for field in ['question', 'answer', 'data_source']:
        if field not in example:
            if filter_none:
                return None
            raise ValueError(f"Validation failed: Missing required field '{field}' in example: {example}")
    
    # Check for None or empty values
    for k, v in example.items():
        if k in ['question', 'answer'] and (v is None or v == ""):
            if filter_none:
                return None
            else:
                print(example)
                assert v != None and v != ""

    question = example['question']
    data_source = example['data_source']
    file_name = example.get('file_name', None)
    if file_name:
        abs_file_name = os.path.join('experiments/mm_data', file_name)
        question += f"  The image path you need to solve the question is {abs_file_name}"
    
    # Normalize answer to a list of strings
    answer = example['answer']
    if isinstance(answer, str):
        answer = [answer]

    # 2. Dynamically create tools_kwargs based on tool_type
    common_create_kwargs = {
        "ground_truth": answer,
        "question": question,
        "data_source": data_source
    }
    
    if tool_type == 'single':
        tool_names = [WIKI_SEARCH]
    elif tool_type == 'three':
        tool_names = [WIKI_SEARCH, WEB_SEARCH, CRAWL_PAGE]
    elif tool_type == 'two':
        tool_names = [WEB_SEARCH, CRAWL_PAGE]
    elif tool_type == 'afm':
        tool_names = [WEB_SEARCH, CRAWL_PAGE, CODE, VISUAL_INSPECTOR]
    else:
        # This case should be prevented by argparse choices, but good for safety
        raise ValueError(f"Invalid tool_type: {tool_type}")

    tools_kwargs = {
        name: {"create_kwargs": common_create_kwargs} for name in tool_names
    }

    # 3. Build the final structured dictionary
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "user", "content": sys_prompt.strip() + "\n\nQuestion: " + question.strip()}
        ],
        "reward_model": {
            "ground_truth": json.dumps({"target": answer, "question": question}, ensure_ascii=False)
        },
        "extra_info": {
            "need_tools_kwargs": True,
            "index": data_source + '-' + str(uuid.uuid4()),
            "tools_kwargs": tools_kwargs,
            "question": question,
            "answer": answer
        }
    }

# --- Data Loading and Saving ---

def load_dataset_from_file(input_path: str) -> datasets.Dataset:
    """Loads a dataset from a json, jsonl, or parquet file."""
    file_extension = os.path.splitext(input_path)[1].lower()
    
    try:
        if file_extension in ['.json', '.jsonl']:
            return datasets.load_dataset("json", data_files=input_path, split='train')
        elif file_extension == '.parquet':
            return datasets.load_dataset("parquet", data_files=input_path, split='train')
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise IOError(f"Failed to load dataset from {input_path}: {e}")

# --- Utility Functions ---

def print_statistics(processed_data: datasets.Dataset, input_path: str, output_path: str) -> None:
    """Prints summary statistics of the processed dataset."""
    num_examples = len(processed_data)
    if num_examples == 0:
        print("Warning: The processed dataset is empty.")
        return

    print("\n" + "="*50)
    print("      Dataset Conversion Statistics")
    print("="*50)
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total examples processed: {num_examples:,}")
    
    # BUG FIX: Calculate stats from the correct fields in the processed data
    avg_question_len = sum(len(ex['extra_info']['question']) for ex in processed_data) / num_examples
    # 'answer' is now guaranteed to be a list in 'extra_info'
    avg_answer_len = sum(len(ans) for ex in processed_data for ans in ex['extra_info']['answer']) / num_examples

    print(f"\nAverage question length: {avg_question_len:.1f} chars")
    print(f"Average answer length:   {avg_answer_len:.1f} chars")
    
    print("\n--- Sample of the first record ---")
    # Pretty print the first example's JSON structure for easy inspection
    sample_output = json.dumps(processed_data[0], indent=2, ensure_ascii=False)
    print(sample_output)
    print("="*50 + "\n")


# --- Main Execution ---

def main():
    """Main function to parse arguments and orchestrate the data preparation."""
    parser = argparse.ArgumentParser(
        description="Convert parquet or jsonl files into a valid training format with schema validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument("input_path", help="Path to the input file (e.g., data.jsonl, data.parquet).")
    parser.add_argument("-o", "--output", required=True, help="Path to the output directory.")
    parser.add_argument(
        "--sys-prompt-key",
        default='SR1',
        choices=SYS_PROMPT_MAP.keys(),
        help="The key for the system prompt to use."
    )
    parser.add_argument(
        "--tool-type",
        default='single',
        choices=['single', 'two', 'three', 'afm'],
        help="Specify the tool configuration: 'single' for wiki_search only, 'three' for all three tools, 'afm' for all tools."
    )
    parser.add_argument(
        "--filter-none",
        action='store_true',
        help="Filter out examples with None or empty values instead of raising errors."
    )
    args = parser.parse_args()

    try:
        # 1. Load data
        print(f"Loading dataset from '{args.input_path}'...")
        source_dataset = load_dataset_from_file(args.input_path)
        
        # 2. Select system prompt
        sys_prompt = SYS_PROMPT_MAP[args.sys_prompt_key]
        print(f"Using system prompt key: '{args.sys_prompt_key}'")
        print(f"Using tool type: '{args.tool_type}'")

        # 3. Process the dataset
        print("Processing examples...")
        original_count = len(source_dataset)
        
        if args.filter_none:
            # Process with filtering enabled
            processed_examples = []
            filtered_examples = []
            
            for example in source_dataset:
                result = process_example(example, sys_prompt=sys_prompt, tool_type=args.tool_type, filter_none=True)
                if result is None:
                    filtered_examples.append(example)
                else:
                    processed_examples.append(result)
            
            # Create dataset from processed examples
            processed_dataset = datasets.Dataset.from_list(processed_examples)
            
            # Print filtering statistics
            filtered_count = len(filtered_examples)
            print(f"\nFiltering Statistics:")
            print(f"Original examples: {original_count:,}")
            print(f"Filtered out: {filtered_count:,}")
            print(f"Remaining: {len(processed_examples):,}")
            
            # Print sample of filtered data
            if filtered_examples:
                print(f"\nSample of filtered examples:")
                for i, example in enumerate(filtered_examples[:3]):
                    print(f"  {i+1}. {example}")
                if len(filtered_examples) > 3:
                    print(f"  ... and {len(filtered_examples) - 3} more")
        else:
            # Process without filtering (original behavior)
            original_columns = source_dataset.column_names
            processed_dataset = source_dataset.map(
                lambda x: process_example(x, sys_prompt=sys_prompt, tool_type=args.tool_type),
                num_proc=os.cpu_count(), # Use multiple cores for faster processing
                remove_columns=original_columns
            )
        
        # 4. Prepare output path and save
        os.makedirs(args.output, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(args.input_path))[0]
        output_path = os.path.join(args.output, f"{base_filename}.parquet")
        
        print(f"Saving processed data to '{output_path}'...")
        processed_dataset.to_parquet(output_path)
        
        # 5. Print final statistics
        print_statistics(processed_dataset, args.input_path, output_path)
        print("✅ Script finished successfully.")

    except (ValueError, IOError, KeyError) as e:
        print(f"\n❌ An error occurred: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()