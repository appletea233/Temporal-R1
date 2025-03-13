# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import math
import os
import yaml
import json
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from qwen_vl_utils import process_vision_info

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        self.dataset = []
        self.data_folders = {}

        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.dataset.extend(cur_data_dict)

                    source = cur_data_dict[0].get("dataset", None)
                    data_folder = dataset["data_folder"]
                    if source and data_folder:
                        self.data_folders[source] = data_folder
        else:
            print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_item = self.dataset[index]
        source = data_item['dataset']
        row_dict = dict()

        if source in ['charades_train']:
            video_path = os.path.join(self.data_folders[source], data_item['video'])
            problem = data_item["problem"]
            # problem = f"{problem} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            messages = [
                {"role": "user", "content": [
                    {
                        "type": "video", 
                        "total_pixels": self.max_pixels, 
                        "min_pixels": self.min_pixels,
                        "video": video_path,
                    },
                    {
                        "type": "text", 
                        "text": problem
                    },
                ]},
            ]
            row_dict['problem_type'] = 'tvg'
        elif source in ['youcook_rerank_train']:
            images_path = data_item['sampled_frame_paths']
            # image_content = 
            messages = [
                {"role": "user", "content": [
                    {
                        "type": "video",                         
                        "total_pixels": self.max_pixels // 8, 
                        "max_pixels": self.max_pixels // 8, 
                        "min_pixels": self.min_pixels // 8,
                        "video": images_path,
                        "fps": 1.0,
                    },
                    {
                        "type": "text", 
                        "text": data_item["problem"]
                    },
                ]},
            ]
            row_dict['problem_type'] = 'rerank'
        else:
            video_path = os.path.join(self.data_folders[source], data_item['video'])
            problem = data_item["problem"]
            # problem = f"{problem} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            messages = [
                {"role": "user", "content": [
                    {
                        "type": "video", 
                        "total_pixels": self.max_pixels, 
                        "min_pixels": self.min_pixels,
                        "video": video_path,
                    },
                    {
                        "type": "text", 
                        "text": problem
                    },
                ]},
            ]
            row_dict['problem_type'] = 'tvg'
        
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        prompt = maybe_apply_chat_template({'prompt': messages}, self.processor)["prompt"]

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        row_dict["multi_modal_data"] = {
            'video': video_inputs
        }
        
        if source in ['charades_train']:
            fps_inputs = video_kwargs['fps'][0]
            nframe_inputs = video_inputs[0].shape[0]
            length_inputs = nframe_inputs / fps_inputs
            row_dict['video_length'] = length_inputs
        else:
            row_dict['video_length'] = data_item['video_length']

        model_inputs = self.processor(
            text=prompt,
            videos=video_inputs,
            return_tensors="pt",
        )
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        row_dict['multi_modal_inputs'] = dict(model_inputs)
        if "video_grid_thw" in model_inputs:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                video_grid_thw=model_inputs["video_grid_thw"],
                second_per_grid_ts=model_inputs["second_per_grid_ts"],
                attention_mask=attention_mask,
            )
        elif "image_grid_thw" in model_inputs:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = data_item['solution']
        return row_dict

from .tokenizer import get_processor, get_tokenizer

if __name__ == "__main__":
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/lihongyu/Qwen/Qwen2.5-VL-3B-Instruct"
    data_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/video_llm/codes/VLM-R1/src/EasyR1/scripts/tvg.yaml'
    tokenizer = get_tokenizer(model_path)
    processor = get_processor(model_path, use_fast=False)

    dataset = RLHFDataset(
        data_path,
        tokenizer,
        processor,
        min_pixels=2592,
        max_pixels=2592*2
        # max_pixels=14 * 14 * 1024 * 8,
    )
    # Qwen-VL patch size 2 * 14 * 14
    for data in dataset:
        pass
