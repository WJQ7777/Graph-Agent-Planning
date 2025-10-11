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
import json
import logging
import os
import random
import re
import traceback
from typing import Dict, List

import openai

from verl.utils.reward_score.qa_em import compute_score_em_batch

# Setup logging
logger = logging.getLogger(__name__)

# LLM评估模板
LLM_EVALUATION_PROMPT_TEMPLATE = """
Please determine if the predicted answer is equivalent to the labeled answer. 
Question:  {question} 
Labeled Answer:  {gt_answer} 
Predicted Answer: {pred_answer}  

**Rules**:
If the prediction and answer are semantically equivalent despite the expression order, the description format, and the use of measurement units and the order, then your judgement will be correct.
{{  
"rationale": "your rationale for the judgement, as a text", 
"judgement": "your judgement result, can only be 'correct' or 'incorrect' 
}}
"""

# Environment variables for LLM judge
LLM_JUDGE_API_KEY = os.getenv("GRM_API_KEY")
LLM_JUDGE_BASE_URL = os.getenv("GRM_BASE_URL", "https://api.openai.com/v1")
LLM_JUDGE_MODEL_NAME = "gpt-4.1-mini"

client = openai.AsyncOpenAI(
    api_key=LLM_JUDGE_API_KEY,
    base_url=LLM_JUDGE_BASE_URL,
)

def extract_answer(solution_str: str) -> str:
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if not matches:
        return ""
    
    # Return the last answer if multiple exist
    return matches[-1].group(1).strip()

async def llm_judge_single(question: str, pred_answer: str, gt_answer: str, max_retries: int = 2) -> float:
    """
    Judge a single prediction using LLM.
    
    Args:
        question: The original question
        pred_answer: The predicted answer
        gt_answer: The ground truth answer
        max_retries: Maximum number of retries
        
    Returns:
        Score: 1.0 if correct, 0.0 if incorrect
    """
    do_print = random.randint(1, 64) == 1
    
    formatted_prompt = LLM_EVALUATION_PROMPT_TEMPLATE.format(
        question=question,
        pred_answer=pred_answer,
        gt_answer=gt_answer
    )
    
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=LLM_JUDGE_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": formatted_prompt},
                ],
                temperature=0.0,
            )
            response_text = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                response_json = json.loads(response_text)
                judgement = response_json.get("judgement", "").lower()
                
                if judgement == "correct":
                    score = 1.0
                elif judgement == "incorrect":
                    score = 0.0
                else:
                    raise ValueError(f"Invalid judgement: {judgement}")
                
                if do_print:
                    print("--- LLM Judge Evaluation ---")
                    print(f"Question: {question}")
                    print(f"Predicted: {pred_answer}")
                    print(f"Ground Truth: {gt_answer}")
                    print(f"Score: {score}")
                    print(f"Rationale: {response_json.get('rationale', '')}")
                    print("---------------------------\n")

                return score

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not parse LLM judge response on attempt {attempt + 1}: {response_text}")
                print(f"Error: {e}")

        except Exception as e:
            print(f"An error occurred during LLM judge API call on attempt {attempt + 1}: {e}")
            traceback.print_exc()
        
        # If not the last attempt, wait before retrying
        if attempt < max_retries:
            print(f"Retrying LLM judge in 1 second...")
            await asyncio.sleep(1)
    
    print(f"All {max_retries + 1} LLM judge attempts failed. Returning default score of 0.")
    return 0.0

def compute_score_llm_judge_batch(
    questions: List[str],
    ground_truths: List[Dict],
    responses: List[str], 
    prompts: List[str],
    data_sources: List[str],
    extra_infos: List[Dict],
    **kwargs
) -> List[float]:
    """
    Compute scores using both LLM judge and EM, with flag control.
    
    Args:
        data_sources: List of data source identifiers
        prompts: List of questions/prompts
        responses: List of solution strings containing answers
        ground_truths: List of ground truth dictionaries
        extra_infos: List of extra information dictionaries
    Returns:
        List of scores
    """
    if len(responses) != len(ground_truths) != len(prompts):
        raise ValueError("The number of responses, ground_truths, and prompts must be equal.")
    
    # Extract predicted answers
    pred_answers = [extract_answer(solution_str) for solution_str in responses]
    gt_answers = [gt['target'] for gt in ground_truths]
    
    # Compute EM scores
    em_scores = compute_score_em_batch(data_sources, prompts, responses, ground_truths, extra_infos)

    # Compute LLM judge scores
    async def _async_batch_llm_judge():
        tasks = [
            llm_judge_single(question, pred_answer, gt_answer)
            for question, pred_answer, gt_answer in zip(prompts, pred_answers, gt_answers)
        ]
        return await asyncio.gather(*tasks)
    
    llm_judge_scores = asyncio.run(_async_batch_llm_judge())
    
    # Return dictionary with both metrics
    return [
        {
            "score": em,
            "em": em,
            "llm_judge": laj
        } for em, laj in zip(em_scores, llm_judge_scores)
    ]