# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].

import copy
import inspect
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from mr0.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from mr0.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from mr0.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
)


def collate_fn(batch):
    return batch


class FastActionTokenCollator:
    """Wrap a VLA collate_fn and precompute FAST action-token strings in workers."""

    def __init__(
        self,
        base_collate_fn,
        fast_tokenizer_path: str,
        action_dim: int | None = None,
        time_horizon: int | None = None,
    ):
        self.base_collate_fn = base_collate_fn
        self.fast_tokenizer_path = fast_tokenizer_path
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.fast_tokenizer = None

    def _get_fast_tokenizer(self):
        if self.fast_tokenizer is None:
            from mr0.model.modules.action_model.fast_ActionHeader import _load_fast_processor

            self.fast_tokenizer = _load_fast_processor(self.fast_tokenizer_path)
            if self.time_horizon is not None:
                self.fast_tokenizer.time_horizon = self.time_horizon
            if self.action_dim is not None:
                self.fast_tokenizer.action_dim = self.action_dim
        return self.fast_tokenizer

    @staticmethod
    def _map_fast_tokens_to_vlm_action(tokens) -> str:
        return "".join(f"<robot_action_{int(token)}>" for token in tokens)

    def __call__(self, batch):
        batch = self.base_collate_fn(batch)
        if not batch or "action" not in batch[0] or "vlm_action_tokens" in batch[0]:
            return batch

        from mr0.model.modules.action_model.fast_ActionHeader import encode_fast_actions_by_shape

        actions = [sample["action"] for sample in batch]
        batch_fast_tokens, fitted_actions = encode_fast_actions_by_shape(
            self._get_fast_tokenizer(),
            actions,
            target_time_horizon=self.time_horizon,
        )

        for sample, fast_tokens, action in zip(batch, batch_fast_tokens, fitted_actions):
            sample["action"] = action
            sample["fast_action_time_horizon"] = int(action.shape[0])
            sample["fast_action_dim"] = int(action.shape[1])
            sample["fast_action_tokens"] = [int(token) for token in fast_tokens]
            sample["vlm_action_tokens"] = self._map_fast_tokens_to_vlm_action(fast_tokens)
        return batch


def make_fast_action_collate_fn(
    base_collate_fn,
    fast_tokenizer_path: str,
    action_dim: int | None = None,
    time_horizon: int | None = None,
):
    return FastActionTokenCollator(
        base_collate_fn=base_collate_fn,
        fast_tokenizer_path=fast_tokenizer_path,
        action_dim=action_dim,
        time_horizon=time_horizon,
    )


def make_padding_collate_fn(action_dim: int, action_horizon: int, state_dim: int | None = None):
    """Create a collate_fn that pads action (and optionally state) to uniform dimensions.

    Pads with zeros on the dim axis (right) and chunk/time axis (end).
    Raises ValueError if the source dimensions exceed the target dimensions.

    Args:
        action_dim: Target action dimension (second axis).
        action_horizon: Target action chunk length (first axis).
        state_dim: Target state dimension. If None, state is not padded.
    """

    def _pad_array(arr: np.ndarray, target_time: int, target_dim: int, name: str) -> np.ndarray:
        """Pad/truncate a [T, D] array to [target_time, target_dim] with zeros.

        Time axis is truncated (take first target_time) when source is longer —
        used for action_horizon ablations where dataset emits a fixed long chunk
        (e.g. 50) but training needs only the first N steps.
        """
        t, d = arr.shape
        if d > target_dim:
            raise ValueError(
                f"{name} dim ({d}) exceeds target dim ({target_dim}). "
                f"Check your config or dataset — source data should not be wider than the target."
            )
        if t > target_time:
            arr = arr[:target_time]
            t = target_time
        if t == target_time and d == target_dim:
            return arr
        padded = np.zeros((target_time, target_dim), dtype=arr.dtype)
        padded[:t, :d] = arr
        return padded

    def padding_collate_fn(batch):
        for sample in batch:
            if "action" in sample:
                _, action_source_dim = sample["action"].shape
                sample["action"] = _pad_array(sample["action"], action_horizon, action_dim, "action")
                action_mask = np.zeros((action_dim,), dtype=bool)
                action_mask[:action_source_dim] = True
                sample["action_mask"] = action_mask
            if state_dim is not None and "state" in sample:
                state_time = sample["state"].shape[0]  # keep original time dim for state
                sample["state"] = _pad_array(sample["state"], state_time, state_dim, "state")
        return batch

    return padding_collate_fn


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    lerobot_version: str | None = None,
    modality_root_dir: Path | str | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param lerobot_version: Explicit lerobot version override ("v2.0" or "v3.0"). If None, auto-detected from dataset file structure.
    :param modality_root_dir: Per-dataset shadow modality root. Takes precedence over
        ``data_cfg.modality_root_dir`` for this dataset. Use this when a mixture
        carries datasets whose modality.json lives under a different root than the
        rest of the mix (e.g. synthetic data with its own modality shadow).
    :return: A LeRobotSingleDataset object.
    """

    # Registry instances are shared across datasets. Copy before applying run-specific settings.
    data_config = copy.deepcopy(ROBOT_TYPE_CONFIG_MAP[robot_type])
    action_horizon = data_cfg.get("action_horizon", None) if data_cfg else None
    if action_horizon is not None:
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        data_config.action_indices = list(range(action_horizon))
    modality_config = data_config.modality_config()
    binary_gripper = data_cfg.get("binary_gripper", True) if data_cfg else True
    image_augmentation = data_cfg.get("image_augmentation", False) if data_cfg else False
    normalization_mode = data_cfg.get("normalization_mode", "min_max") if data_cfg else "min_max"
    # Defensive call: only pass kwargs the config's transform() actually accepts. Some example
    # configs define transform(self) with no args, or only binary_gripper; passing an unknown
    # kwarg would break them. AgileX's transform() accepts image_augmentation and opts in.
    transform_params = inspect.signature(data_config.transform).parameters
    transform_kwargs = {}
    if "binary_gripper" in transform_params:
        transform_kwargs["binary_gripper"] = binary_gripper
    if "image_augmentation" in transform_params:
        transform_kwargs["image_augmentation"] = image_augmentation
    if "normalization_mode" in transform_params:
        transform_kwargs["normalization_mode"] = normalization_mode
    transforms = data_config.transform(**transform_kwargs)
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(
            f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default"
        )
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]

    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    extra_kwargs = {}
    if modality_root_dir is not None:
        extra_kwargs["modality_root_dir"] = Path(modality_root_dir)
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,  # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
        data_root_dir=data_root_dir,
        lerobot_version=lerobot_version,
        **extra_kwargs,
    )


def _parse_mixture_entry(entry, default_data_root_dir: Path | str):
    """Return (dataset_name, weight, robot_type, data_root_dir, modality_root_dir).

    Supported tuple shapes:
      - 3: (dataset, weight, robot_type) — inherits default data_root_dir and the
        cfg-wide modality_root_dir.
      - 4: (dataset, weight, robot_type, data_root_dir) — overrides data_root_dir
        only.
      - 5: (dataset, weight, robot_type, data_root_dir, modality_root_dir) —
        overrides both. modality_root_dir may be None to fall back to cfg.
    """
    if len(entry) == 3:
        d_name, d_weight, robot_type = entry
        return d_name, d_weight, robot_type, Path(default_data_root_dir), None
    if len(entry) == 4:
        d_name, d_weight, robot_type, data_root_dir = entry
        return d_name, d_weight, robot_type, Path(data_root_dir), None
    if len(entry) == 5:
        d_name, d_weight, robot_type, data_root_dir, modality_root_dir = entry
        mod_root = Path(modality_root_dir) if modality_root_dir is not None else None
        return d_name, d_weight, robot_type, Path(data_root_dir), mod_root
    raise ValueError(f"Invalid mixture entry {entry!r}; expected 3, 4 or 5 fields.")


def _discover_mixture(data_cfg: dict, data_root_dir: Path | str) -> list[tuple[str, float, str]]:
    """Discover immediate LeRobot child datasets under ``data_root_dir``."""
    robot_type = data_cfg.get("auto_discover_robot_type", None)
    if not robot_type:
        raise ValueError("data_mix='auto' requires datasets.vla_data.auto_discover_robot_type.")

    root = Path(data_root_dir)
    mixture_spec = [
        (child.name, 1.0, robot_type)
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "meta" / "info.json").is_file()
    ]
    if not mixture_spec:
        raise FileNotFoundError(f"No LeRobot child datasets found under {root}.")
    print(f"Auto-discovered {len(mixture_spec)} datasets under {root}: {[entry[0] for entry in mixture_spec]}")
    return mixture_spec


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    balance_dataset_weights = data_cfg.get("balance_dataset_weights", balance_dataset_weights)
    balance_trajectory_weights = data_cfg.get("balance_trajectory_weights", balance_trajectory_weights)
    mixture_spec = _discover_mixture(data_cfg, data_root_dir) if data_mix == "auto" else DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for entry in mixture_spec:
        d_name, d_weight, robot_type, item_data_root_dir, item_modality_root_dir = _parse_mixture_entry(entry, data_root_dir)
        dataset_key = (str(item_data_root_dir), d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type, item_data_root_dir, item_modality_root_dir))

    dataset_mixture = []
    for d_name, d_weight, robot_type, item_data_root_dir, item_modality_root_dir in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_LeRobotSingleDataset(
                    item_data_root_dir,
                    d_name,
                    robot_type,
                    delete_pause_frame=delete_pause_frame,
                    data_cfg=data_cfg,
                    modality_root_dir=item_modality_root_dir,
                ),
                d_weight,
            )
        )

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


if __name__ == "__main__":

    # import debugpy
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path to YAML config",
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        default=None,
        help="Override data_mix in config (e.g. libero_goal)",
    )
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()
    args.config_yaml = args.config_yaml  # use CLI arg or default
    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    if hasattr(args, 'data_mix') and args.data_mix:
        vla_dataset_cfg.data_mix = args.data_mix
    vla_dataset_cfg.task_id = "all"
    print(f"Config: {args.config_yaml}")
    print(f"Data mix: {vla_dataset_cfg.data_mix}")
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm

    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        # print(1)
        if count > 100:
            break
        count += 1
        pass
