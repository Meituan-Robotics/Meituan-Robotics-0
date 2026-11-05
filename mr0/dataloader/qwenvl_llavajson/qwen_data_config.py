"""VQA / multimodal dataset registry for VLM cotrain.

Register multimodal datasets here with short nicknames. The general format
follows Qwen-VL fine-tuning conventions:
https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md

Each entry needs an ``annotation_path`` (json/jsonl) and a ``data_path``
(image root or "/" if annotation files carry absolute image paths).
"""

import os
import re

# ---- Example: ShareGPT4V-COCO ------------------------------------------------
# Override roots via environment variables or edit these directly.
json_root = os.environ.get("VQA_JSON_ROOT", "/PATH/TO/vqa_json_root")
image_root = os.environ.get("VQA_IMAGE_ROOT", "/PATH/TO/vqa_image_root")

SHAREGPT4V_COCO = {
    "annotation_path": f"{json_root}/sharegpt4v_coco.json",
    "data_path": f"{image_root}/",
}

# ---- Register datasets -------------------------------------------------------
# Add your datasets here. Nicknames can be looked up by data_list([...]).
data_dict = {
    "sharegpt4v_coco": SHAREGPT4V_COCO,
}


def parse_sampling_rate(dataset_name):
    """Optional per-dataset sampling rate suffix: ``name%50`` → 0.5."""
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    """Resolve a list of dataset nicknames to full config dicts.

    Pass ``["all"]`` to load every registered dataset. Each entry gets a
    ``sampling_rate`` key parsed from a trailing ``%N`` suffix if present.
    """
    if dataset_names == ["all"]:
        dataset_names = list(data_dict.keys())
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict:
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"Dataset '{dataset_name}' not registered in data_dict")
    return config_list


if __name__ == "__main__":
    print(data_list)
