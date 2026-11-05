import copy
import bisect
import hashlib
import itertools
import json
import os
import random
import time
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
import transformers
from decord import VideoReader
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from mr0.dataloader.qwenvl_llavajson.qwen_data_config import data_list
from mr0.dataloader.qwenvl_llavajson.rope2d import (
    get_rope_index_2,
    get_rope_index_25,
    get_rope_index_3vl,
    get_rope_index_35,
)

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>\n"
DEFAULT_VIDEO_TOKEN = "<video>\n"

local_rank = None
_JSONL_INDEX_CACHE_VERSION = 1


def safe_hash(input_tuple):
    return int(hashlib.sha256(str(input_tuple).encode("utf-8")).hexdigest(), 16) % (10**8)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _jsonl_index_cache_dir() -> str:
    cache_dir = os.environ.get("MR0_VLM_INDEX_CACHE")
    if cache_dir:
        return cache_dir
    return os.path.join(os.getcwd(), "outputs", "vlm_jsonl_index")


def _jsonl_index_cache_path(path: str) -> str:
    stat = os.stat(path)
    fingerprint = f"{_JSONL_INDEX_CACHE_VERSION}:{os.path.abspath(path)}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
    return os.path.join(_jsonl_index_cache_dir(), f"{digest}.offsets.npy")


def _build_jsonl_offsets(path: str) -> np.ndarray:
    offsets = array("Q", [0])
    file_size = os.path.getsize(path)
    absolute_pos = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            search_from = 0
            while True:
                newline_pos = chunk.find(b"\n", search_from)
                if newline_pos < 0:
                    break
                next_offset = absolute_pos + newline_pos + 1
                if next_offset < file_size:
                    offsets.append(next_offset)
                search_from = newline_pos + 1
            absolute_pos += len(chunk)
    return np.frombuffer(offsets, dtype=np.uint64).copy()


def load_jsonl_offsets(path: str) -> np.ndarray:
    cache_path = _jsonl_index_cache_path(path)
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")

    lock_path = cache_path + ".lock"
    stale_lock_seconds = int(os.environ.get("MR0_VLM_INDEX_LOCK_TIMEOUT", "3600"))
    while True:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if os.path.exists(cache_path):
                return np.load(cache_path, mmap_mode="r")
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_lock_seconds:
                    os.unlink(lock_path)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(1)

    try:
        offsets = _build_jsonl_offsets(path)
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, offsets)
        os.replace(tmp_path, cache_path)
    finally:
        os.close(lock_fd)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass

    return np.load(cache_path, mmap_mode="r")


def _coprime_stride(length: int, seed: int) -> int:
    if length <= 1:
        return 1
    stride = seed % length
    if stride == 0:
        stride = 1
    while np.gcd(stride, length) != 1:
        stride += 1
        if stride >= length:
            stride = 1
    return stride


def _affine_permute(index: int, length: int, seed: int) -> int:
    if length <= 1:
        return 0
    stride = _coprime_stride(length, seed)
    offset = (seed // max(length, 1)) % length
    return int((index * stride + offset) % length)


def _apply_chat_template_ids(tokenizer, messages):
    try:
        encoded = tokenizer.apply_chat_template(messages, return_dict=False)
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages)
    if hasattr(encoded, "__contains__") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    return list(encoded)


def _resolve_special_token_id(tokenizer, token_name: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token_name)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Token '{token_name}' not found in tokenizer vocabulary")
    return int(token_id)


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List | None = None,
    visual_type: str = "image",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")
    grid_thw = list(grid_thw or [])

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += _apply_chat_template_ids(tokenizer, [{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        if visual_replicate_index >= len(grid_thw):
                            raise ValueError(
                                f"Found more <{visual_type}> tags than processed {visual_type} inputs "
                                f"({visual_replicate_index + 1} > {len(grid_thw)})."
                            )
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>" * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = _apply_chat_template_ids(tokenizer, conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    if visual_replicate_index != len(grid_thw):
        raise ValueError(
            f"Found {visual_replicate_index} <{visual_type}> tags but processed {len(grid_thw)} {visual_type} inputs."
        )

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
        self.model_type = data_args.model_type
        self._image_token_id = _resolve_special_token_id(tokenizer, "<|image_pad|>")
        self._video_token_id = _resolve_special_token_id(tokenizer, "<|video_pad|>")
        self._vision_start_token_id = _resolve_special_token_id(tokenizer, "<|vision_start|>")
        _model_type = (data_args.model_type or "").lower()
        if _model_type == "qwen2.5vl":
            base_rope_fn = get_rope_index_25
        elif _model_type in ("qwen3.5vl", "qwen35vl", "qwen3.5"):
            base_rope_fn = get_rope_index_35
        elif _model_type in ("qwen3vl", "qwen3-vl"):
            base_rope_fn = get_rope_index_3vl
        else:
            base_rope_fn = get_rope_index_2
        self.get_rope_index = lambda *args, **kwargs: base_rope_fn(
            *args,
            **kwargs,
            image_token_id=self._image_token_id,
            video_token_id=self._video_token_id,
            vision_start_token_id=self._vision_start_token_id,
        )

        self._jsonl_sources = []
        self._jsonl_cum_lengths = []
        self._jsonl_total_length = 0
        self._jsonl_handles = {}
        self._eager_samples = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                offsets = load_jsonl_offsets(data["annotation_path"])
                num_lines = len(offsets)
                sampling_rate = data.get("sampling_rate", 1.0)
                sample_count = int(num_lines * sampling_rate) if sampling_rate < 1.0 else num_lines
                if sample_count <= 0:
                    continue
                source = {
                    "annotation_path": data["annotation_path"],
                    "data_path": data["data_path"],
                    "offsets": offsets,
                    "num_lines": num_lines,
                    "sample_count": sample_count,
                    "sampling_rate": sampling_rate,
                    "sample_seed": safe_hash((data["annotation_path"], sampling_rate, "sample")),
                }
                self._jsonl_sources.append(source)
                self._jsonl_total_length += sample_count
                self._jsonl_cum_lengths.append(self._jsonl_total_length)
                if sampling_rate < 1.0:
                    print(f"sampling {sample_count} examples from dataset {data}")
                else:
                    rank0_print(f"dataset name: {data}")
                continue
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if data["data_path"] != "":
                    ann["data_path"] = data["data_path"]
                elif "raw_data" in ann.keys():
                    ann["data_path"] = ann["raw_data"]["data_root"]
            self._eager_samples += annotations

        self._eager_samples = self.pre_filter_long_case(
            self._eager_samples, max_words=tokenizer.max_len_single_sentence
        )
        random.shuffle(self._eager_samples)  # Randomly shuffle the data for training

        self.tokenizer = tokenizer
        self.list_data_dict = self._eager_samples
        self.data_args = data_args
        self._total_length = len(self._eager_samples) + self._jsonl_total_length
        self._shuffle_seed = safe_hash((getattr(data_args, "dataset_use", ""), "lazy_vlm_shuffle"))

        rank0_print(f"Total training samples: {self._total_length}")
        rank0_print("Formatting inputs...Skip in lazy mode")

        # self.data_args.image_processor.max_pixels = data_args.max_pixels
        # self.data_args.image_processor.min_pixels = data_args.min_pixels
        # self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        # self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return self._total_length

    def _open_jsonl(self, path: str):
        handle = self._jsonl_handles.get(path)
        if handle is None or handle.closed:
            handle = open(path, "rb")
            self._jsonl_handles[path] = handle
        return handle

    def _get_raw_sample(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        if self._jsonl_total_length:
            index = _affine_permute(index, len(self), self._shuffle_seed)

        if index < len(self._eager_samples):
            return self._eager_samples[index]

        jsonl_index = index - len(self._eager_samples)
        source_index = bisect.bisect_right(self._jsonl_cum_lengths, jsonl_index)
        previous = 0 if source_index == 0 else self._jsonl_cum_lengths[source_index - 1]
        local_index = jsonl_index - previous
        source = self._jsonl_sources[source_index]
        if source["sampling_rate"] < 1.0:
            line_index = _affine_permute(local_index, source["num_lines"], source["sample_seed"])
        else:
            line_index = local_index
        handle = self._open_jsonl(source["annotation_path"])
        handle.seek(int(source["offsets"][line_index]))
        sample = json.loads(handle.readline())
        if source["data_path"] != "":
            sample["data_path"] = source["data_path"]
        elif "raw_data" in sample.keys():
            sample["data_path"] = sample["raw_data"]["data_root"]
        return sample

    def pre_filter_long_case(self, list_data_dict, max_words=1024):
        """filter out conversations with total words exceeding max_words"""

        def count_total_words(convs):
            total = 0
            for entry in convs:
                value = entry.get("value", "")
                total += len(value.strip().split())
            return total

        return [item for item in list_data_dict if count_total_words(item.get("conversations", [])) <= max_words]

    @staticmethod
    def _as_media_list(media):
        if media is None:
            return []
        if isinstance(media, list):
            return media
        return [media]

    @staticmethod
    def _count_visual_tags(conversations, visual_type: str) -> int:
        tag = f"<{visual_type}>"
        total = 0
        for conv in conversations:
            content = conv.get("content", conv.get("value", ""))
            if isinstance(content, str):
                total += content.count(tag)
        return total

    def _validate_visual_inputs(self, sample: dict) -> tuple[list, list]:
        conversations = sample.get("conversations", [])
        image_files = self._as_media_list(sample.get("images"))
        video_files = self._as_media_list(sample.get("videos"))
        image_tags = self._count_visual_tags(conversations, "image")
        video_tags = self._count_visual_tags(conversations, "video")

        if image_files and video_files:
            raise ValueError("Samples containing both images and videos are not supported by this VLM dataloader.")
        if image_tags and video_tags:
            raise ValueError("Samples containing both <image> and <video> tags are not supported.")
        if len(image_files) != image_tags:
            raise ValueError(f"Image count/tag mismatch: {len(image_files)} image files vs {image_tags} <image> tags.")
        if len(video_files) != video_tags:
            raise ValueError(f"Video count/tag mismatch: {len(video_files)} video files vs {video_tags} <video> tags.")
        return image_files, video_files

    @property
    def lengths(self):
        length_list = []
        for sample in self._eager_samples:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        if self._jsonl_total_length:
            length_list.extend([1] * self._jsonl_total_length)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self._eager_samples:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if ("images" in sample) or ("videos" in sample) else -cur_len
            length_list.append(cur_len)
        if self._jsonl_total_length:
            length_list.extend([1] * self._jsonl_total_length)
        return length_list

    @property
    def pre_calculated_length(self):
        if self._eager_samples and "num_tokens" in self._eager_samples[0]:
            length_list = [sample["num_tokens"] for sample in self._eager_samples]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"File not exist: {image_file}")
        image = Image.open(image_file).convert("RGB")
        # if fix image size?
        if getattr(self.data_args, "fix_image_size", None) is not None:
            image = image.resize(
                self.data_args.fix_image_size,
                resample=Image.BICUBIC,
            )
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {video_file}")
        if avg_fps is None or avg_fps <= 0:
            raise ValueError(f"Video has invalid fps ({avg_fps}): {video_file}")
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(images=None, videos=video, return_tensors="pt")
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_t = self.data_args.image_processor.temporal_patch_size / fps
        return video_tensor, grid_thw, second_per_grid_t

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_final_retries):
            try:
                next_index = random.randrange(len(self))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self._get_raw_sample(i)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_files, video_files = self._validate_visual_inputs(sample)
        video = None
        image = None
        grid_thw = None
        if image_files:
            image_folder = sample["data_path"]
            image_paths = [os.path.join(image_folder, file) for file in image_files]
            results = [self.process_image_unified(file) for file in image_paths]
            image, grid_thw = zip(*results)
            image = list(image)
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image")
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),  # (1,16,16)
            )
        elif video_files:
            video_folder = sample["data_path"]
            video_paths = [os.path.join(video_folder, file) for file in video_files]
            results = [self.process_video(file) for file in video_paths]
            video, grid_thw, second_per_grid_ts = zip(*results)
            video = list(video)
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video")
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer)
            position_ids = torch.arange(0, data_dict["input_ids"].size(1)).view(1, -1).unsqueeze(0).expand(3, -1, -1)

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                position_ids=position_ids,
            )
        if image is not None:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
        # video exist in the data
        elif video is not None:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw

        max_len = min(self.tokenizer.max_len_single_sentence, self.tokenizer.model_max_length)
        if data_dict["input_ids"].shape[0] > max_len:
            raise ValueError(
                f"Tokenized sample length {data_dict['input_ids'].shape[0]} exceeds model limit {max_len}; "
                "dropping instead of truncating multimodal tokens."
            )

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side=self.tokenizer.padding_side,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side=self.tokenizer.padding_side
        )
        position_ids = pad_and_cat(position_ids)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[..., : self.tokenizer.model_max_length]  # 3,bs,length

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            itertools.chain(*(instance["pixel_values"] for instance in instances if "pixel_values" in instance))
        )
        videos = list(
            itertools.chain(
                *(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(*(instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance))
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(*(instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance))
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )

        seq_lens = torch.tensor([0] + [len(seq) for seq in input_ids], dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            itertools.chain(*(instance["pixel_values"] for instance in instances if "pixel_values" in instance))
        )
        videos = list(
            itertools.chain(
                *(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(*(instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance))
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(*(instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance))
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # load training dataset
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    # load evaluation dataset (if specified eval dataset path)
    eval_dataset = None
    if hasattr(data_args, "eval_dataset") and data_args.eval_dataset:
        eval_data_args = copy.deepcopy(data_args)
        eval_data_args.dataset_use = data_args.eval_dataset
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=eval_data_args)

    # select appropriate collator based on whether data needs to be flattened
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def make_vlm_dataloader(cfg):
    data_args = cfg.datasets.vlm_data
    image_processor = AutoProcessor.from_pretrained(
        cfg.framework.qwenvl.base_vlm,
    ).image_processor

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.framework.qwenvl.base_vlm,
        model_max_length=data_args.model_max_length,
        padding_side="left",  # flash Attention version of Qwen2.5_VL. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input.
        use_fast=False,
    )

    # avoid processing these in dataset
    image_processor.max_pixels = int(data_args.max_pixels)
    image_processor.min_pixels = int(data_args.min_pixels)
    image_processor.size["longest_edge"] = int(data_args.max_pixels)
    image_processor.size["shortest_edge"] = int(data_args.min_pixels)
    data_args_ns = SimpleNamespace(**OmegaConf.to_container(data_args, resolve=True))
    data_args_ns.image_processor = image_processor  # TODO later remove the logic bound to model
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args_ns)

    #
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    from torch.utils.data import DataLoader

    num_workers = getattr(cfg.datasets.vlm_data, "num_workers", 4)
    prefetch_factor = getattr(cfg.datasets.vlm_data, "prefetch_factor", 4)
    persistent_workers = getattr(cfg.datasets.vlm_data, "persistent_workers", True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.datasets.vlm_data.per_device_batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    return {
        "train_dataloader": train_dataloader,
    }


from transformers import AutoProcessor

if __name__ == "__main__":
    import argparse

    import debugpy

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    data_args = cfg.datasets.vlm_data
    image_processor = AutoProcessor.from_pretrained(
        cfg.framework.qwenvl.base_vlm,
    ).image_processor

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.framework.qwenvl.base_vlm,
        model_max_length=data_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    # avoid processing these in dataset
    image_processor.max_pixels = data_args.max_pixels
    image_processor.min_pixels = data_args.min_pixels
    image_processor.size["longest_edge"] = data_args.max_pixels
    image_processor.size["shortest_edge"] = data_args.min_pixels

    data_args_ns = SimpleNamespace(**OmegaConf.to_container(data_args, resolve=True))
    data_args_ns.image_processor = image_processor
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args_ns)

    #
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.datasets.vlm_data.per_device_batch_size,
        collate_fn=data_collator,
    )
    batchs = iter(train_dataloader)
    batch_samples = next(batchs)
    # skip the first 99 batches, get the 100th batch

    # batch_samples = next(islice(batchs, 99, 100))
    count = 0
    while count < 100:
        batch_samples = next(batchs)  # for debug
        print(count)
        count += 1
    pass
