import json
import os
from pathlib import Path

import numpy as np
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

from mr0.dataloader.dataloader_manager import DataLoaderManager
from mr0.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")


def build_dataloader(
    cfg, dataset_py="lerobot_datasets_oxe"
):  # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from mr0.dataloader.lerobot_datasets import (
            collate_fn,
            get_vla_dataset,
            make_fast_action_collate_fn,
            make_padding_collate_fn,
        )

        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

        action_model_cfg = getattr(getattr(cfg, "framework", None), "action_model", None)
        framework_cfg = getattr(cfg, "framework", None)
        framework_name = getattr(framework_cfg, "name", None) if framework_cfg is not None else None

        # Regression/diffusion heads require fixed-width tensors. QwenFast
        # tokenizes native action shapes, so only its time axis is normalized
        # inside FastActionTokenCollator.
        if (
            framework_name != "QwenFast"
            and action_model_cfg is not None
            and hasattr(action_model_cfg, "action_dim")
            and hasattr(action_model_cfg, "action_horizon")
        ):
            state_dim = getattr(action_model_cfg, "state_dim", None)
            chosen_collate_fn = make_padding_collate_fn(
                action_dim=action_model_cfg.action_dim,
                action_horizon=action_model_cfg.action_horizon,
                state_dim=state_dim,
            )
        else:
            chosen_collate_fn = collate_fn

        if framework_name == "QwenFast" and vla_dataset_cfg.get("fast_tokenize_in_dataloader", True):
            fast_tokenizer_path = action_model_cfg.get("fast_tokenizer_path", "physical-intelligence/fast")
            action_dim = getattr(action_model_cfg, "action_dim", None)
            future_action_window_size = getattr(action_model_cfg, "future_action_window_size", None)
            time_horizon = None if future_action_window_size is None else future_action_window_size + 1
            chosen_collate_fn = make_fast_action_collate_fn(
                chosen_collate_fn,
                fast_tokenizer_path=fast_tokenizer_path,
                action_dim=action_dim,
                time_horizon=time_horizon,
            )

        num_workers = getattr(cfg.datasets.vla_data, "num_workers", 4)
        prefetch_factor = getattr(cfg.datasets.vla_data, "prefetch_factor", 4)
        persistent_workers = getattr(cfg.datasets.vla_data, "persistent_workers", True)
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=chosen_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():

            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]

        return vlm_train_dataloader


def build_dataloader_manager(cfg) -> DataLoaderManager:
    """Convenience factory: build a DataLoaderManager from the full config.

    Equivalent to ``DataLoaderManager.from_config(cfg)`` but exposed at
    package level for symmetry with ``build_dataloader``.
    """
    return DataLoaderManager.from_config(cfg)
