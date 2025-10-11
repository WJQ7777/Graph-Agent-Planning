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
from verl.utils.reward_score.utils import verify_format_simple, verify_format_repetition
import time

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
LLM_JUDGE_MODEL_NAME = os.getenv("GRM_MODEL_NAME", "gpt-4.1-mini")

client = openai.AsyncOpenAI(
    api_key=LLM_JUDGE_API_KEY,
    base_url=LLM_JUDGE_BASE_URL,
)

client_sync = openai.OpenAI(
    api_key=LLM_JUDGE_API_KEY,
    base_url=LLM_JUDGE_BASE_URL,
)

def extract_answer(response: str) -> str:
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, response, re.DOTALL))
    
    if not matches:
        return ""
    
    # Return the last answer if multiple exist
    return matches[-1].group(1).strip()

async def llm_judge_single(question: str, pred_answer: str, gt_answer: str, max_retries: int = 3) -> float:
    """
    Judge a single prediction using LLM.
    Returns:
        Score: 1.0 if correct, 0.0 if incorrect
    """
    do_print = random.randint(1, 32) == 1
    
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
            )
            response_text = response.choices[0].message.content

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
                print(f"[WARNING] Could not parse LLM judge response on attempt {attempt + 1}: {response_text}")
                print(f"[WARNING] Got {e}")

        except Exception as e:
            print(f"[WARNING] An error occurred during LLM judge API call on attempt {attempt + 1}: {e}")
            traceback.print_exc()

        # If not the last attempt, wait before retrying
        if attempt < max_retries:
            print(f"[INFO] LLM Judge fail, Retrying LLM judge in 1 second...")
            await asyncio.sleep(1)
    
    print(f"[ERROR] All {max_retries + 1} LLM judge attempts failed. Returning default score of 0.")
    return 0.0

def compute_score_grm_batch(
    questions: List[str],
    ground_truths: List[Dict],
    responses: List[str], 
    prompts: List[str],
    data_sources: List[str],
    extra_infos: List[Dict],
    **kwargs
) -> List[float]:
    """
    Compute scores using LLM judge.
    Returns:
        List of scores
    """
    # Extract predicted answers
    pred_answers = [extract_answer(response) for response in responses]
    gt_answers = [gt['target'] for gt in ground_truths]
    
    # Compute LLM judge scores
    async def _async_batch_llm_judge():
        tasks = [
            llm_judge_single(question, pred_answer, gt_answer)
            for question, pred_answer, gt_answer in zip(questions, pred_answers, gt_answers)
        ]
        return await asyncio.gather(*tasks)
    
    llm_judge_scores = asyncio.run(_async_batch_llm_judge())
    
    # Return dictionary with both metrics
    return [
        {
            "score": laj,
            "llm_judge": laj
        } for laj in llm_judge_scores
    ]



def compute_score_grm_batch_simple_fmt( questions: List[str], ground_truths: List[Dict], responses: List[str], prompts: List[str],
    data_sources: List[str], extra_infos: List[Dict], **kwargs) -> List[Dict]:
    """
    Compute combined scores using format checking and LLM judge.
    
    The final score is calculated as:
    score = 0.1 * format_score + 0.9 * llm_judge_score
    """
    pred_answers = [extract_answer(response) for response in responses]
    gt_answers = [gt['target'] for gt in ground_truths]
    
    # Check format for each response
    format_scores = [1.0 if verify_format_simple(response) else 0.0 for response in responses]
    
    # Compute LLM judge scores
    async def _async_batch_llm_judge():
        tasks = [
            llm_judge_single(question, pred_answer, gt_answer)
            for question, pred_answer, gt_answer in zip(questions, pred_answers, gt_answers)
        ]
        return await asyncio.gather(*tasks)
    
    llm_judge_scores = asyncio.run(_async_batch_llm_judge())
    
    # Calculate combined scores
    results = []
    for fmt_score, laj_score in zip(format_scores, llm_judge_scores):
        combined_score = 0.1 * fmt_score + 0.9 * laj_score
        results.append({
            'score': combined_score,
            'format': fmt_score,
            'llm_judge': laj_score
        })
    
    return results


def compute_score_grm_batch_repetition_fmt(questions: List[str], ground_truths: List[Dict], responses: List[str], prompts: List[str],
    data_sources: List[str], extra_infos: List[Dict], **kwargs) -> List[Dict]:
    pred_answers = [extract_answer(response) for response in responses]
    gt_answers = [gt['target'] for gt in ground_truths]
    
    # Check format for each response
    format_scores = [0 if verify_format_repetition(response) else -0.5 for response in responses]
    
    # Compute LLM judge scores
    async def _async_batch_llm_judge():
        tasks = [
            llm_judge_single(question, pred_answer, gt_answer)
            for question, pred_answer, gt_answer in zip(questions, pred_answers, gt_answers)
        ]
        return await asyncio.gather(*tasks)
    
    llm_judge_scores = asyncio.run(_async_batch_llm_judge())
    
    # Calculate combined scores
    results = []
    for fmt_score, laj_score in zip(format_scores, llm_judge_scores):
        combined_score = fmt_score + laj_score
        results.append({
            'score': combined_score,
            'format': fmt_score,
            'llm_judge': laj_score
        })
    
    return results

def compute_score_grm_batch_mul_fmt(
    questions: List[str],
    ground_truths: List[Dict],
    responses: List[str],
    prompts: List[str],
    data_sources: List[str],
    extra_infos: List[Dict],
    **kwargs
) -> List[Dict]:
    pred_answers = [extract_answer(response) for response in responses]
    gt_answers = [gt['target'] for gt in ground_truths]
    
    # check format for each response
    format_scores = [1.0 if verify_format_repetition(response) else 0.0 for response in responses]
    
    # only use llm for those format=1
    valid_indices = [i for i, fmt in enumerate(format_scores) if fmt == 1.0]
    
    async def _async_batch_llm_judge():
        tasks = [
            llm_judge_single(questions[i], pred_answers[i], gt_answers[i])
            for i in valid_indices
        ]
        return await asyncio.gather(*tasks) if tasks else []
    
    llm_judge_partial_scores = asyncio.run(_async_batch_llm_judge())
    
    # compute final score
    llm_judge_scores = [-1.0] * len(responses)
    for idx, score in zip(valid_indices, llm_judge_partial_scores):
        llm_judge_scores[idx] = score
    
    # calculate combined scores
    results = [
        {
            'score': -1 if fmt == 0 else laj,
            'format': fmt,
            'llm_judge': laj
        }
        for fmt, laj in zip(format_scores, llm_judge_scores)
    ]
    
    return results

def llm_judge_single_sync(question: str, pred_answer: str, gt_answer: str, max_retries: int = 3) -> float:
    """
    Judge a single prediction using LLM - Sync for afm RewardManager。
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
            response = client_sync.chat.completions.create(
                model=LLM_JUDGE_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": formatted_prompt},
                ],
            )
            response_text = response.choices[0].message.content

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
                    print("---------------------------")

                return score

            except (json.JSONDecodeError, ValueError) as e:
                print(f"[WARNING] Could not parse LLM judge response on attempt {attempt + 1}: {response_text}")
                print(f"[WARNING] Got {e}")

        except Exception as e:
            print(f"[WARNING] An error occurred during LLM judge API call on attempt {attempt + 1}: {e}")
            traceback.print_exc()

        # If not the last attempt, wait before retrying
        if attempt < max_retries:
            print(f"[INFO] LLM Judge fail, Retrying LLM judge in 1 second...")
            time.sleep(1)
    
    print(f"[ERROR] All {max_retries + 1} LLM judge attempts failed. Returning default score of 0.")
    return 0.0

def compute_score(index=None, solution_str=None, ground_truth=None, format_score=0):
    """
    Interface for AFMRewardManager
    """
    ground_truth = json.loads(ground_truth)
    question = ground_truth.get('question', '')
    gt_answer = ground_truth.get('target', '')
    pred_answer = extract_answer(solution_str)
    
    # use sync llm judge function
    score = llm_judge_single_sync(question, pred_answer, gt_answer)
    
    return score, pred_answer, gt_answer, index