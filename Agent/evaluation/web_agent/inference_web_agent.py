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
import string
import time
import random
import os
import argparse
from openai import OpenAI
from queue import Queue, Empty
from transformers import AutoTokenizer
import logging
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts import llm_evaluation_webthinker, function_2_tool_cqb_reflection_doublecheck_rule
from web_tools import WebSearchTool, CrawlPageTool
from utils import read_jsonl, write_jsonl, read_json, write_json, count_tokens, truncate_special_tokens, retry_predict


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


### Your Own Model
URL = "http://10.77.209.251:10000/v1" # SET URL
MODEL = "AFM-WebAgent-32B-RL"
KEY = "empty"
SYSTEM_PROMPT = function_2_tool_cqb_reflection_doublecheck_rule

try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B", trust_remote_code=True)
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer from 'Qwen/Qwen2.5-32B': {str(e)}") from e


INFER_KWARGS = {
    "temperature": 1.0,
    "top_p": 0.9,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "max_tokens": 4096,
    "total_tokens": 32768,
    "web_topk": 10,
    "trajectory_len": 36,
    "parallel": 30,
    "round": 5,
}


def request_service(system, prompt, current_answer, url, key, model, stop_words=None, **kwargs):
    if stop_words is None:
        stop_words = [
            "</web_search>", 
            "</crawl_page>",
            "</answer>", 
        ]

    client = OpenAI(base_url=url, api_key=key)
    try:
        system_token_count, prompt_token_count, current_answer_token_count = count_tokens(system, tokenizer), count_tokens(prompt, tokenizer), count_tokens(current_answer, tokenizer)
        max_tokens_for_answer = kwargs.get("total_tokens", 32768) - kwargs.get("max_tokens", 4096) - system_token_count - prompt_token_count - current_answer_token_count - 512
        truncate_current_answer = truncate_special_tokens(current_answer, max_tokens_for_answer, tokenizer)
        model_output_message = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": f"user\n\n{system}\n\n{prompt}"},
                {"role": "assistant", "content": truncate_current_answer}
            ],
            stream=False,
            stop=stop_words,
            temperature=kwargs.get("temperature", 1.0), 
            top_p=kwargs.get("top_p", 0.9),        
            presence_penalty=kwargs.get("presence_penalty", 0), 
            frequency_penalty=kwargs.get("frequency_penalty", 0), 
            max_tokens=kwargs.get("max_tokens", 4096), 
            n=1, 
        )
        collected_content = []
        model_output = model_output_message.choices[0].message.content
        stop_tag = model_output_message.choices[0].model_extra["stop_reason"]
        tag = stop_tag.lstrip("</").rstrip(">")
        return tag, model_output + stop_tag
    except Exception as e:
        error_msg = str(e)
        return "error", error_msg


def extract_specific_tag(text):
    from collections import deque
    ALLOWED_TAGS = {'think', 'plan', 'reflection', 'suggested_answer', 'answer', 'web_search', 'crawl_page', 'double_check'}
    tag_stack = deque()
    tag_pairs = []
    
    split_pattern = re.compile(r'(<\/?(?:{})>)'.format('|'.join(ALLOWED_TAGS)))
    segments = split_pattern.split(text)
    segments = [s for s in segments if s.strip()]
    content_buffer = []
    
    for seg in segments:
        if seg.startswith('<'):
            is_close_tag = seg.startswith('</')
            tag_name = seg.strip('<>/').lower()
            if tag_name not in ALLOWED_TAGS:
                content_buffer.append(seg)
                continue
            if not is_close_tag:
                tag_stack.append((tag_name, len(content_buffer)))
            else:
                if not tag_stack:
                    continue
                open_tag, content_start_idx = tag_stack.pop()
                if open_tag == tag_name:
                    paired_content = ''.join(content_buffer[content_start_idx:])
                    tag_pairs.append({
                        "tool": f"{open_tag}",
                        "content": paired_content.strip()
                    })
                    content_buffer = content_buffer[:content_start_idx]
                    think_content = ''
                    if "think" in open_tag or "reasoning_content" in open_tag:
                        think_content = paired_content.strip()
        else:
            content_buffer.append(seg)
    if not tag_pairs:
        return "", "", ""
    return think_content, f"</{tag_pairs[-1]['tool']}>", tag_pairs[-1]['content']


def get_search_results_with_format(task, response, history, **kwargs):
    try:
        _, tool, query = extract_specific_tag(response)
        search_results = ""
        wiki_topk = kwargs.get("wiki_topk", 10)
        web_topk = kwargs.get("web_topk", 10)

        if tool == '</web_search>':
            search_results = WebSearchTool(
                f'http://{os.getenv("SERVER_HOST")}:{os.getenv("WEBSEARCH_PORT")}/search',
                api_key=os.getenv("SUMMARY_OPENAI_API_KEY"),
                api_url=os.getenv("SUMMARY_OPENAI_API_BASE_URL"),
                model=os.getenv("SUMMARY_MODEL"),
                task=task,
                query=query,
                history=history,
                topk=wiki_topk
            )
        elif tool == "</crawl_page>":
            search_results = CrawlPageTool(
                f'http://{os.getenv("SERVER_HOST")}:{os.getenv("CRAWL_PAGE_PORT")}/crawl_page',
                api_key=os.getenv("SUMMARY_OPENAI_API_KEY"),
                api_url=os.getenv("SUMMARY_OPENAI_API_BASE_URL"),
                model=os.getenv("SUMMARY_MODEL"),
                task=task,
                urls=query,
                history=history,
            )
        else:
            search_results = ''
        return True, search_results
    except Exception as err:
        return False, err


def process_single_data(query, **kwargs):
    global URL, KEY, MODEL

    system_prompt = SYSTEM_PROMPT.strip()
    current_answer = ""
    trajectory_len = kwargs.get("trajectory_len", 12)
    result_list = []
    attempt = 0
    error_count = 0
    while attempt < trajectory_len and error_count < 10:
        step_list = [elem["type"] for elem in result_list]
        time.sleep(random.random() * 0.1)
        current_answer += "<think>"
        item_type, content = request_service(system_prompt, query, current_answer, URL, KEY, MODEL, **kwargs)
        content_wo_think = content.split("</think>")[-1].strip()
        logging.info(f"step {attempt+1}: {item_type}")
        if item_type == "error" or content is None:
            error_count += 1
            continue
        elif len(step_list) >= 5 and len(set(step_list[-5:])) == 1:
            content = f"|<BEGIN_OF_DUPLICATE_FUNCTION>|{content}|<END_OF_DUPLICATE_FUNCTION>|You have previsouly output the same function. Please try to think differently with no more duplications."
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            attempt += 1
        elif content_wo_think in "".join(current_answer):
            content = f"|<BEGIN_OF_DUPLICATE_CONTENT>|{content}|<END_OF_DUPLICATE_CONTENT>|You have previsouly output the same content. Please try to think differently with no more duplications."
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            attempt += 1
        elif item_type == "answer":
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            attempt += 1
            return result_list, None
        elif item_type in ["suggested_answer"]:
            historic_suggested_answer_list = [item for item in result_list if item["type"] == "suggested_answer"]
            if len(historic_suggested_answer_list) >= 2:
                result_list.append({
                    "type": "answer",
                    "content": content.replace("<suggested_answer>", "<answer>").replace("</suggested_answer>", "</answer>")
                })
                current_answer += content
                attempt += 1
                return result_list, None
            else:
                result_list.append({
                    "type": item_type,
                    "content": content
                })
                current_answer += content
                attempt += 1
        elif item_type in ["web_search", "crawl_page"]:
            current_answer_with_current_content = current_answer + content
            is_success, obs = get_search_results_with_format(query, content, current_answer_with_current_content, **kwargs)
            result_list.append({
                "type": item_type,
                "content": f"\n{content}\n<observation>\n{obs}\n</observation>"
            })
            current_answer += f"\n{content}\n<observation>\n{obs}\n</observation>"
            attempt += 1

    suggested_answer_list = [item for item in result_list if item["type"] == "suggested_answer"]
    if suggested_answer_list:
        result_list.append({
            "type": "answer",
            "content": suggested_answer_list[-1]["content"].replace("<suggested_answer>", "<answer>").replace("</suggested_answer>", "</answer>")
        })
        return result_list, None
    else:
        return result_list, "exceed max_attempts, and no answer/suggested_answer found."


def decode_response(response):
    try:
        if isinstance(response, str):
            return json.loads(response)
        return response
    except:
        return {"judgement": "incorrect"}


def process_queries(infile, outfile, q_key, a_key, **kwargs):
    if infile.endswith(".json"):
        questions_data = read_json(infile)
    elif infile.endswith(".jsonl"):
        questions_data = read_jsonl(infile)
    else:
        raise ValueError(f"Invalid file format: {infile}")

    stats = {"total": len(questions_data), "success": 0, "failed": 0}
    task_queue = Queue()
    result_queue = Queue()
    write_lock = Lock()

    def producer():
        for idx, question_data in enumerate(questions_data):
            task_queue.put((idx, question_data))
        for _ in range(kwargs.get("parallel", 4)):
            task_queue.put(None)

    def consumer():
        nonlocal stats
        while True:
            task = task_queue.get()
            if task is None:
                break
            
            idx, question_data = task
            question = question_data[q_key]
            golden_answer = question_data[a_key]
            level = question_data.get('Level', '-1')
            max_retry = 3
            result = 0
            trace = None
            
            for retry in range(max_retry):
                trace = {
                    "question_id": str(idx),
                    "question": question,
                    "Level": level,
                    "golden_answer": golden_answer,
                    "prediction": None,
                    "llm_judge": 0,
                    "steps": [],
                    "status": None,
                    "error": None,
                }

                # process single data
                result_list, failed_reason = process_single_data(question, **kwargs)
                if failed_reason:
                    trace["status"] = "error"
                    trace["error"] = failed_reason
                else:
                    trace["steps"] = result_list
                    if any([result_dict["type"] == "error" for result_dict in result_list]):
                        trace["status"] = "error"
                        trace["error"] = "Error in processing"
                    else:
                        if result_list[-1]["type"] == "answer":
                            prediction = re.findall(r'<answer>(.*?)</answer>', 
                                                        result_list[-1]["content"], 
                                                        re.DOTALL)[0].strip()
                            trace["prediction"] = prediction
                            trace["status"] = "completed"
                        else:
                            trace["error"] = f"Last step is not of dtype = answer"
                            trace["status"] = "invalid_format"

                # postprocess result
                if not trace["error"]:
                    llm_evaluation_prompt = llm_evaluation_webthinker.format(
                        question=question, 
                        gt_answer=golden_answer, 
                        pred_answer=trace["prediction"]
                    )
                    output = retry_predict(
                        os.getenv("GRM_API_KEY"),
                        os.getenv("GRM_BASE_URL"),
                        os.getenv("GRM_MODEL_NAME"),
                        llm_evaluation_prompt,
                        developer_prompt="You are an evaluation assistant."
                    )
                    json_output = decode_response(output)
                    if (json_output and isinstance(json_output, dict) and 
                        "judgement" in json_output and 
                        json_output['judgement'].lower() == "correct"):
                        trace['llm_judge'] = 1
                    else:
                        trace['llm_judge'] = 0
                    logging.info("#" * 50)
                    logging.info(f"-- Level: {trace['Level']}")
                    logging.info(f"-- question: {question}")
                    logging.info(f"-- predicted_answer: {trace['prediction']}")
                    logging.info(f"-- golden_answer: {golden_answer}")
                    logging.info(f"-- llm_judge: {trace['llm_judge']}")
                    logging.info(f"Successfully Processing {idx}, Status: {trace['status']}")
                    logging.info("#" * 50)
                    trace["raw"] = question_data
                    result = 1
                    break
                logging.info(f"Query Processing Error. Index: {idx}, Error: {trace['error']}")
            result_queue.put((result, trace))
            task_queue.task_done()

    def result_writer():
        nonlocal stats
        with write_lock:
            existing_data = []
            if os.path.exists(outfile):
                existing_data = read_jsonl(outfile)
            while True:
                result_item = result_queue.get()
                if result_item is None:
                    break
                result, trace = result_item
                if trace.get("prediction") is not None:
                    if result == 1:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                    existing_data.append(trace)
                    write_jsonl([trace], outfile, "a")
                else:
                    logging.info("Skip writting file: prediction is None")
                result_queue.task_done()

    num_workers = kwargs.get("parallel", 8)
    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        executor.submit(producer)
        consumer_futures = [executor.submit(consumer) for _ in range(num_workers)]
        writer_future = executor.submit(result_writer)
        for future in as_completed(consumer_futures):
            future.result()
        result_queue.put(None)
        writer_future.result()

    logging.info(f"Running Successed: {stats['success']}, Failed: {stats['failed']}, Total: {len(questions_data)}")
    return outfile


def main():
    parser = argparse.ArgumentParser(description='Process queries with multiple rounds.')
    parser.add_argument('--infile', required=True, help='Input file path')
    parser.add_argument('--outfile', required=True, help='Output file path (base name)')
    parser.add_argument('--q-key', default='question', help='Key for question field (default: question)')
    parser.add_argument('--a-key', default='answer', help='Key for answer field (default: answer)')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.expanduser(args.outfile))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    start_time = time.time()
    # Run N times
    round = INFER_KWARGS['round']
    for current_round in range(round):
        if args.outfile.endswith(".jsonl"):
            current_outfile = args.outfile.replace(".jsonl", f".round_{current_round}.jsonl")
        else:
            current_outfile = f"{args.outfile}.round_{current_round}.jsonl" 
        process_queries(
            args.infile, 
            current_outfile, 
            args.q_key, 
            args.a_key, 
            **INFER_KWARGS
        )
    cost_time = time.time() - start_time
    print(f"Time cost: {cost_time:.2f} seconds")

if __name__ == "__main__":
    main()