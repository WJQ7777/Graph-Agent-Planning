# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py

import argparse
import asyncio
import json
import time
import warnings
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple
import os

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")
    
class RateLimiter:
    def __init__(self, max_concurrent_requests: int = 10, requests_per_second: float = 10.0):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.requests_per_second = requests_per_second
        self.last_request_time = 0
        self.request_lock = asyncio.Lock()
        
    async def acquire(self):
        """获取访问权限，包括并发控制和速率限制"""
        async with self.semaphore:  # 控制并发数
            await self._rate_limit()  # 控制请求频率
            
    async def _rate_limit(self):
        """实现速率限制"""
        async with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            min_interval = 1.0 / self.requests_per_second
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f}s")
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = time.time()


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(inputs["input_ids"].device)
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        torch.cuda.empty_cache()

        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!", stacklevel=2)
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [{"title": content.split("\n")[0].strip('"'), "text": "\n".join(content.split("\n")[1:]), "contents": content} for content in all_contents]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(model_name=self.retrieval_method, model_path=config.retrieval_model_path, pooling_method=config.retrieval_pooling_method, max_length=config.retrieval_query_max_length, use_fp16=config.retrieval_use_fp16)
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: "):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()

        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


retriever = None
config = None
logger = None
rate_limiter = None  # 添加速率限制器

async def batching_worker():
    """修改后的批处理工作器，包含访问控制"""
    while True:
        try:
            first_request_item = await asyncio.wait_for(request_queue.get(), BATCH_TIMEOUT)
        except asyncio.TimeoutError:
            continue

        batch = [first_request_item]
        start_time = time.time()
        
        # 收集批次请求
        while len(batch) < MAX_BATCH_SIZE and (time.time() - start_time) < BATCH_TIMEOUT:
            try:
                request_item = request_queue.get_nowait()
                batch.append(request_item)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)

        if not batch:
            continue

        # 应用速率限制
        try:
            await rate_limiter.acquire()
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            # 如果速率限制器出错，继续处理但记录错误
            pass

        all_queries = []
        futures_info = []  # (future, num_queries, topk, return_scores)

        for req, future in batch:
            all_queries.extend(req.queries)
            topk = req.topk if req.topk is not None else config.retrieval_topk
            futures_info.append((future, len(req.queries), topk, req.return_scores))

        max_topk = max(info[2] for info in futures_info)
        
        # 记录批次信息
        logger.info(f"Processing batch: {len(batch)} requests, {len(all_queries)} queries, max_topk={max_topk}")
        batch_start_time = time.time()

        try:
            results, scores = retriever.batch_search(query_list=all_queries, num=max_topk, return_score=True)
            
            batch_process_time = time.time() - batch_start_time
            logger.info(f"Batch processed in {batch_process_time:.3f}s")

            current_idx = 0
            for future, num_queries, topk, return_scores_flag in futures_info:
                req_results = results[current_idx : current_idx + num_queries]
                req_scores = scores[current_idx : current_idx + num_queries]
                current_idx += num_queries

                trimmed_results = [r[:topk] for r in req_results]
                trimmed_scores = [s[:topk] for s in req_scores]

                if return_scores_flag:
                    resp = []
                    for i, single_result in enumerate(trimmed_results):
                        combined = [{"document": doc, "score": score} for doc, score in zip(single_result, trimmed_scores[i])]
                        resp.append(combined)
                else:
                    resp = trimmed_results

                if not future.done():
                    future.set_result({"result": resp})

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for future, _, _, _ in futures_info:
                if not future.done():
                    future.set_exception(e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rate_limiter
    logger.info("Retrieval server starting up...")
    
    # 初始化速率限制器
    max_concurrent = int(os.getenv("MAX_CONCURRENT_REQUESTS", 10))
    requests_per_sec = float(os.getenv("REQUESTS_PER_SECOND", 10.0))
    rate_limiter = RateLimiter(max_concurrent, requests_per_sec)
    
    logger.info(f"Rate limiter configured: max_concurrent={max_concurrent}, requests_per_second={requests_per_sec}")
    
    # 启动批处理工作器
    asyncio.create_task(batching_worker())
    yield
    logger.info("Retrieval server shutting down.")


app = FastAPI(lifespan=lifespan)
request_queue = asyncio.Queue()
MAX_BATCH_SIZE = 512
BATCH_TIMEOUT = 0.01  # 10ms


@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """检索端点，现在包含队列管理"""
    future = asyncio.Future()
    
    # 检查队列大小，避免内存溢出
    queue_size = request_queue.qsize()
    max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", 1000))
    
    if queue_size > max_queue_size:
        logger.warning(f"Queue size ({queue_size}) exceeds limit ({max_queue_size})")
        raise HTTPException(status_code=503, detail="Server too busy, please retry later")
    
    await request_queue.put((request, future))
    
    try:
        # 设置超时以避免长时间等待
        timeout = float(120.0)
        result = await asyncio.wait_for(future, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        logger.error("Request timeout")
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8008)

    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=128,
    )

    retriever = get_retriever(config)

    uvicorn.run(app, host=args.host, port=args.port, workers=1)
