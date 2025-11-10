import copy
from dataclasses import dataclass
import json
from typing import Any, Dict, Iterator, List, Optional, Sequence, TYPE_CHECKING, Union
from PIL import Image, ImageFile
import os

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *

import traceback
import transformers
import torch
from torch.utils.data import Dataset, IterableDataset
import random

import boto3
from io import BytesIO
import requests

from datasets import load_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(image_file, s3_client=None):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    elif image_file.startswith("s3://"):
        if s3_client is None:
            raise ValueError("S3 client is required for loading s3:// images.")
        bucket_name = image_file.split("s3://")[1].split("/")[0]
        image_key = "/".join(image_file.split("s3://")[1].split("/")[1:])
        response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
        image_data = response['Body'].read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    return image


def build_s3_client(data_args: DataArguments):
    if getattr(data_args, "s3_config", None) is None:
        return None
    s3_config = json.load(open(data_args.s3_config, "r"))
    return boto3.client(
        service_name='s3',
        endpoint_url=s3_config['endpoint_url'],
        aws_access_key_id=s3_config['aws_access_key_id'],
        aws_secret_access_key=s3_config['aws_secret_access_key'],
    )


def zero_image_tensor(data_args: DataArguments):
    crop_size = getattr(data_args.image_processor, 'crop_size',
                        getattr(data_args.image_processor, 'size'))
    height = crop_size['height'] if isinstance(crop_size, dict) else crop_size
    width = crop_size['width'] if isinstance(crop_size, dict) else crop_size
    return torch.zeros(3, height, width)


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

        # --- setup S3 client if provided ---
        self.s3_client = build_s3_client(self.data_args)

        # --- load data_path JSON from local, HTTP, or S3 ---
        if data_path.startswith("http://") or data_path.startswith("https://"):
            response = requests.get(data_path)
            list_data_dict = response.json()

        elif data_path.startswith("s3://"):
            if self.s3_client is None:
                raise ValueError("S3 client not initialized; provide data_args.s3_config.")
            bucket_name = data_path.split("s3://")[1].split("/")[0]
            key = "/".join(data_path.split("s3://")[1].split("/")[1:])
            obj = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            list_data_dict = json.loads(obj['Body'].read().decode('utf-8'))

        else:
            list_data_dict = json.load(open(data_path, "r"))

        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))

        if 'image' in sources:
            try:
                if isinstance(sources['image'], str):
                    images = [sources['image']]
                else:
                    images = sources['image']

                data_dict['image'] = []
                image_folder = self.data_args.image_folder

                for image_file in images:
                    # join path only if not s3/http
                    if not (image_file.startswith("http") or image_file.startswith("https") or image_file.startswith("s3://")):
                        image_file = os.path.join(image_folder, image_file)

                    image = load_image(image_file, self.s3_client)
                    image = self.image_preprocess(image)
                    data_dict['image'].append(image)

            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encountered error when reading image {image_file}, using {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)

        elif self.data_args.is_multimodal:
            data_dict['image'] = [zero_image_tensor(self.data_args)]

        return data_dict


class StreamingHFDataset(IterableDataset):
    """Streaming dataset backed by Hugging Face datasets (Parquet or hub-hosted)."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        if load_dataset is None:
            raise ImportError("datasets library is required for streaming mode. Please install `datasets`.")
        assert data_args.hf_dataset_name or data_args.hf_data_files, \
            "Provide `hf_dataset_name` or `hf_data_files` for streaming."
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)
        self.s3_client = build_s3_client(self.data_args)

        dataset_kwargs = dict(
            path=data_args.hf_dataset_name if data_args.hf_dataset_name else "parquet",
            split=data_args.hf_dataset_split,
            streaming=True
        )
        if data_args.hf_dataset_config:
            dataset_kwargs["name"] = data_args.hf_dataset_config
        if data_args.hf_data_files:
            dataset_kwargs["data_files"] = data_args.hf_data_files
        self.dataset = load_dataset(**dataset_kwargs)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for raw_sample in self.dataset:
            try:
                conversations = raw_sample.get(self.data_args.hf_conversation_column)
                if conversations is None:
                    continue
                if isinstance(conversations, str):
                    conversations = json.loads(conversations)
                data_dict = self.text_preprocess(copy.deepcopy(conversations))
            except Exception as err:
                traceback.print_exc()
                print(f"[StreamingHFDataset] skipping sample due to text preprocessing error: {err}")
                continue

            images = self._prepare_images(raw_sample)
            if images:
                data_dict['image'] = images
            elif self.data_args.is_multimodal:
                data_dict['image'] = [zero_image_tensor(self.data_args)]
            yield data_dict

    def _prepare_images(self, sample: Dict[str, Any]) -> List[torch.Tensor]:
        processed_images: List[torch.Tensor] = []
        image_entries = sample.get(self.data_args.hf_image_column)
        image_paths = sample.get(self.data_args.hf_image_path_column)

        entries_list = ensure_list(image_entries)
        paths_list = ensure_list(image_paths)

        max_len = max(len(entries_list), len(paths_list))
        if max_len == 0:
            return processed_images
        if len(entries_list) == 0:
            entries_list = [None] * max_len
        if len(paths_list) == 0:
            paths_list = [None] * max_len

        for idx in range(max_len):
            image_obj = entries_list[idx] if idx < len(entries_list) else None
            path_obj = paths_list[idx] if idx < len(paths_list) else None
            pil_image = self._resolve_image(image_obj, path_obj)
            if pil_image is None:
                continue
            try:
                processed_images.append(self.image_preprocess(pil_image))
            except Exception as err:
                traceback.print_exc()
                print(f"[StreamingHFDataset] skipping image idx {idx} due to preprocessing error: {err}")
        return processed_images

    def _resolve_image(self, image_entry: Any, path_entry: Optional[Union[str, Dict[str, Any]]]):
        if isinstance(image_entry, Image.Image):
            return image_entry
        if isinstance(image_entry, dict):
            if image_entry.get('bytes') is not None:
                return Image.open(BytesIO(image_entry['bytes'])).convert("RGB")
            if image_entry.get('path'):
                resolved_path = self._resolve_path(image_entry['path'])
                return self._load_image_from_path(resolved_path)
        if isinstance(image_entry, str):
            resolved = self._resolve_path(image_entry)
            return self._load_image_from_path(resolved)
        if path_entry:
            if isinstance(path_entry, dict) and path_entry.get('path'):
                resolved_path = self._resolve_path(path_entry['path'])
            else:
                resolved_path = self._resolve_path(path_entry)
            return self._load_image_from_path(resolved_path)
        return None

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        if path.startswith("http://") or path.startswith("https://") or path.startswith("s3://"):
            return path
        if self.data_args.image_folder:
            return os.path.join(self.data_args.image_folder, path)
        return path

    def _load_image_from_path(self, path: Optional[str]):
        if path is None:
            return None
        try:
            return load_image(path, self.s3_client)
        except Exception as err:
            traceback.print_exc()
            print(f"[StreamingHFDataset] failed to load image at {path}: {err}")
            return None


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = []
            for instance in instances:
                images.extend(instance['image'])
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if getattr(data_args, "use_hf_streaming", False):
        train_dataset = StreamingHFDataset(tokenizer=tokenizer,
                                           data_args=data_args)
    else:
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                              data_path=data_args.data_path,
                                              data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
