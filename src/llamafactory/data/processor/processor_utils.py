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

import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


@dataclass
class DatasetProcessor(ABC):
    r"""A class for data processors."""

    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Build model inputs from the examples."""
        # 获取批量数据
        conversations = examples["conversations"]
        images_list = examples["images"]
        
        # 存储处理后的数据
        processed_batch = {
            "input_ids": [],
            "attention_mask": [],
            # 其他需要处理的字段...
            "valid_flag": []  # 用于标记有效样本
        }

        # 遍历批次中的每个样本
        for conv, images in zip(conversations, images_list):
            valid_sample = True  # 初始假设样本有效
            
            # 1. 统计<image>占位符数量
            image_token_count = 0
            for turn in conv:
                content = turn["content"]
                if isinstance(content, str):
                    image_token_count += content.count("<image>")
                elif isinstance(content, list):
                    for item in content:
                        if item["type"] == "text":
                            image_token_count += item["text"].count("<image>")
                        elif item["type"] == "image":
                            image_token_count += 1  # 直接图像项也视为占位符
            
            # 2. 验证图像数量与占位符数量
            if image_token_count != len(images):
                valid_sample = False
            
            # 3. 仅处理有效样本
            if valid_sample:
                # 原有处理逻辑（根据你的实际代码实现）
                # 例如：processed = self.tokenizer(conv, ...)
                # processed_batch["input_ids"].append(processed["input_ids"])
                # ...
                processed_batch["valid_flag"].append(True)
            else:
                # 对于无效样本，添加空值并标记
                processed_batch["valid_flag"].append(False)
                # 添加空值以保持批次结构
                for key in processed_batch:
                    if key not in ["valid_flag"] and key in self.expected_outputs:
                        processed_batch[key].append([])  # 或适当空值
        
        return processed_batch

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        r"""Print a data example to stdout."""
        ...


def search_for_fit(numbers: list[int], capacity: int) -> int:
    r"""Find the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    r"""Implement efficient greedy algorithm with binary search for the knapsack problem."""
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    r"""Compute the real sequence length after truncation by the cutoff_len."""
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len
