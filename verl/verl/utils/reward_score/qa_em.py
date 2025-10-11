# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Portions of this file are modifications by OPPO PersonalAI Team.
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

import datetime
import random
import re
import string
from typing import List


def _log_print(data_source, prompt_str, solution_str, ground_truth, extracted_ans, reward, extra_info=None):
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using EM scoring method")
        print(f"Data source: {data_source}")
        print(f"Prompt string: {prompt_str}")
        print(f"Solution string: {solution_str}")
        print(f"Extracted answer: {extracted_ans}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Reward: {reward}")
        if extra_info is not None:
            print(f"Extra info: {extra_info}")
        print(f"--------------------------------")

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    if len(matches) == 0:
        return None
    # return the last one
    return matches[-1].group(1).strip()


def compute_score_em(data_source, prompt_str, solution_str, ground_truth, extra_info=None, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    
    rw = 0
    if answer is None:
        rw = 0
    else:
        if em_check(answer, ground_truth['target']):
            rw = score
        else:
            rw = format_score

    if do_print:
        _log_print(data_source, prompt_str, solution_str, ground_truth, answer, rw, extra_info)

    return {
        "score": rw,
        "em": rw
    }

def compute_score_em_batch(
    questions: List[str],
    ground_truths: list[dict],
    prompts: list[str],
    responses: list[str], 
    data_sources: list[str],
    extra_infos: list[dict],
    **kwargs
) -> list[float]:

    scores = []
    for (ds, p, s, gt) in zip(data_sources, prompts, responses, ground_truths):
        score = compute_score_em(ds, p, s, gt).get("score")
        scores.append(score)
    
    return scores
def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
