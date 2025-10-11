# Copyright 2025 the LlamaFactory team.
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

# Portions of this file are modifications by OPPO PersonalAI Team.
# Licensed under the Apache License, Version 2.0.

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")

@dataclass
class IgnoreObsSupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        audios: Sequence["AudioInput"],
    ) -> Tuple[List[int], List[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        is_valid = True
        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                # target_label = target_ids
                target_label, is_valid, search_regions = self._mask_search_result_tokens(target_ids)

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels, is_valid
    
    def _mask_search_result_tokens(self, token_ids: List[int]) -> List[int]:
        start_marker = self.tokenizer.encode(f"<{self.data_args.ignore_observation_token}", add_special_tokens=False)
        end_marker = self.tokenizer.encode(f"{self.data_args.ignore_observation_token}>\n\n", add_special_tokens=False)

        labels = token_ids.copy()
        
        inside_search_result = False
        search_regions = []  # 记录所有搜索结果区域
        
        text = self.tokenizer.decode(token_ids)
        start_count = text.count(f"<{self.data_args.ignore_observation_token}>")
        
        i = 0
        start_pos, end_pos = -1, -1
        while i < len(token_ids):
            if (not inside_search_result and 
                i <= len(token_ids) - len(start_marker) and 
                all(token_ids[i+j] == start_marker[j] for j in range(len(start_marker)))):

                labels[i:i+len(start_marker)] = [IGNORE_INDEX] * len(start_marker)
                inside_search_result = True
                start_pos = i
                i += len(start_marker)
                continue
                
            if (inside_search_result and 
                i <= len(token_ids) - len(end_marker) and 
                all(token_ids[i+j] == end_marker[j] for j in range(len(end_marker)))):

                labels[i:i+len(end_marker)] = [IGNORE_INDEX] * len(end_marker)
                inside_search_result = False
                i += len(end_marker)
                end_pos = i
                
                search_regions.append((start_pos, end_pos))
                continue

            if inside_search_result:
                labels[i] = IGNORE_INDEX
                
            i += 1
            
        # 检查是否有未闭合的搜索标签
        is_valid = not inside_search_result
        
        return labels, is_valid, search_regions
    
    def _validate_example(self, examples, i):
        """验证单个示例是否有效"""
        # 检查对话轮数是否正确
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                f"Dropped invalid example {i}: 对话轮数不正确 - prompt轮数={len(examples['_prompt'][i])}, response轮数={len(examples['_response'][i])}"
            )
            return False
            
        response_text = examples["_response"][i][0]["content"]
        
        # 检查标签数量是否匹配
        start_count = response_text.count(f"<{self.data_args.ignore_observation_token}>")
        end_count = response_text.count(f"</{self.data_args.ignore_observation_token}>")
        
        if start_count != end_count:
            logger.warning_rank0(
                f"Dropped invalid example {i}: 标签数量不匹配 - start={start_count}, end={end_count}"
            )
            return False
            
        # 检查标签嵌套是否正确
        stack = []
        start_tag = f"<{self.data_args.ignore_observation_token}>"
        end_tag = f"</{self.data_args.ignore_observation_token}>"
        
        index = 0
        while index < len(response_text):
            if response_text.startswith(start_tag, index):
                stack.append(start_tag)
                index += len(start_tag)
            elif response_text.startswith(end_tag, index):
                if not stack or stack.pop() != start_tag:
                    logger.warning_rank0(
                        f"Dropped invalid example {i}: 标签嵌套不正确"
                    )
                    return False
                index += len(end_tag)
            else:
                index += 1
                
        if stack:  # 栈不为空，说明有未闭合的标签
            logger.warning_rank0(
                f"Dropped invalid example {i}: 有未闭合的标签"
            )
            return False
            
        return True

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        valid_indices = []
        
        for i in range(len(examples["_prompt"])):
            # 验证示例是否有效
            if not self._validate_example(examples, i):
                continue
                
            # 进一步验证掩码处理结果
            try:
                response_text = examples["_response"][i][0]["content"]
                token_ids = self.tokenizer.encode(response_text)
                _, is_valid, search_regions = self._mask_search_result_tokens(token_ids)
                
                if not is_valid:
                    logger.warning_rank0(
                        f"Dropped invalid example {i}: 掩码处理后发现未闭合的标签"
                    )
                    continue
                    
                # 检查搜索区域是否合理
                for start_pos, end_pos in search_regions:
                    region_text = self.tokenizer.decode(token_ids[start_pos:end_pos])
                    # 可以添加更多特定的检查逻辑
                    
            except Exception as e:
                logger.warning_rank0(
                    f"Dropped invalid example {i}: 处理过程中发生异常 - {str(e)}"
                )
                continue
                
            # 如果所有验证都通过，则处理数据
            # 在处理过程中，如果被切割，并且通过验证，则收集这些数据
            input_ids, labels, is_valid = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            ) 
            if is_valid: # 如果通过验证
                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append([1] * len(input_ids))
                model_inputs["labels"].append(labels)
                model_inputs["images"].append(examples["_images"][i])
                model_inputs["videos"].append(examples["_videos"][i])
                model_inputs["audios"].append(examples["_audios"][i])
                valid_indices.append(i)
            
        # 记录过滤统计信息
        original_count = len(examples["_prompt"])
        filtered_count = len(valid_indices)
        logger.info(f"数据集过滤统计: 原始样本数={original_count}, 有效样本数={filtered_count}, 过滤率={(original_count-filtered_count)/original_count*100:.2f}%")
        logger.info(f"如果过滤率过高，请检查数据集的标签是否正确或者数据的长度是否过长导致特殊符号被截断？？？？？")
        
        return model_inputs

    # def print_data_example(self, example: Dict[str, List[int]]) -> None:
    #     valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    #     logger.info("Tokenized Example:")
    #     logger.info("input_ids:\n{}".format(example["input_ids"]))
    #     logger.info("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    #     logger.info("label_ids:\n{}".format(example["labels"]))
    #     logger.info(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")
    #############################################################################################################
    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        
        logger.info("Tokenized Example:")
        
        # 分块打印input_ids，每块500个token
        logger.info("input_ids:")
        input_ids = example["input_ids"]
        for i in range(0, len(input_ids), 500):
            logger.info(f"  [{i}-{i+len(input_ids[i:i+500])-1}]: {input_ids[i:i+500]}")
        
        # 分块打印解码后的输入文本，每块500个字符
        logger.info("inputs:")
        decoded_inputs = self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        for i in range(0, len(decoded_inputs), 500):
            logger.info(f"  [字符 {i}-{min(i+500, len(decoded_inputs))-1}]:\n{decoded_inputs[i:i+500]}")
        
        # 分块打印label_ids
        logger.info("label_ids:")
        label_ids = example["labels"]
        for i in range(0, len(label_ids), 500):
            logger.info(f"  [{i}-{i+len(label_ids[i:i+500])-1}]: {label_ids[i:i+500]}")
        
        # 分块打印过滤后的标签文本，每块500个字符
        logger.info("labels:")
        decoded_labels = self.tokenizer.decode(valid_labels, skip_special_tokens=False)
        for i in range(0, len(decoded_labels), 500):
            logger.info(f"  [字符 {i}-{min(i+500, len(decoded_labels))-1}]:\n{decoded_labels[i:i+500]}")

@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs