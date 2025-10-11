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
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class PairwiseDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        # consider the response is more important
        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        vocab_size = self.tokenizer.vocab_size        
        if any(t < 0 or t >= vocab_size for t in example["chosen_input_ids"]):
            logger.error(f"Invalid token IDs in chosen_ids: min={min(example['chosen_input_ids'])}, max={max(example['chosen_input_ids'])}, vocab_size={vocab_size}")

        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")


class IgnoreObsPairwiseDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        is_valid = True
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        # consider the response is more important
        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        # 做observation mask
        chosen_labels, is_valid, search_regions = self._mask_search_result_tokens(chosen_ids)
        rejected_labels, is_valid, search_regions = self._mask_search_result_tokens(rejected_ids)

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_labels
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_labels

        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels, is_valid
    def _validate_example(self, examples, i):
            """验证单个示例是否有效"""
            # 检查对话轮数是否正确
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 2:
                logger.warning_rank0(
                    f"Dropped invalid example {i}: RM格式不正确 - prompt轮数={len(examples['_prompt'][i])}, response轮数={len(examples['_response'][i])}"
                )
                return False
            
            def validata_single_example(examples, response_ind):
                response_text = examples["_response"][i][response_ind]["content"]
                
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
            
            # 需要正负样本都符合格式
            return validata_single_example(examples, 0) and validata_single_example(examples, 1)

    def _mask_search_result_tokens(self, token_ids: list[int]) -> list[int]:
        start_marker = self.tokenizer.encode(f"<{self.data_args.ignore_observation_token}", add_special_tokens=False)
        end_marker = self.tokenizer.encode(f"{self.data_args.ignore_observation_token}>\n\n", add_special_tokens=False)

        labels = token_ids.copy()
        
        inside_search_result = False
        search_regions = []  # 记录所有搜索结果区域
        
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
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        valid_indices = []
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue
            if not self._validate_example(examples, i):
                continue
            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels, is_valid = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            if is_valid:
                model_inputs["chosen_input_ids"].append(chosen_input_ids)
                model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
                model_inputs["chosen_labels"].append(chosen_labels)
                model_inputs["rejected_input_ids"].append(rejected_input_ids)
                model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
                model_inputs["rejected_labels"].append(rejected_labels)
                model_inputs["images"].append(examples["_images"][i])
                model_inputs["videos"].append(examples["_videos"][i])
                model_inputs["audios"].append(examples["_audios"][i])
                valid_indices.append(i)

        original_count = len(examples["_prompt"])
        filtered_count = len(valid_indices)        
        logger.info(f"数据集过滤统计: 原始样本数={original_count}, 有效样本数={filtered_count}, 过滤率={(original_count-filtered_count)/original_count*100:.2f}%")
        logger.info(f"如果过滤率过高，请检查数据集的标签是否正确或者数据的长度是否过长导致特殊符号被截断？？？？？")
        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        # valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        # valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        # vocab_size = self.tokenizer.vocab_size        
        # if any(t < 0 or t >= vocab_size for t in example["chosen_input_ids"]):
        #     logger.error(f"Invalid token IDs in chosen_ids: min={min(example['chosen_input_ids'])}, max={max(example['chosen_input_ids'])}, vocab_size={vocab_size}")
        
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
