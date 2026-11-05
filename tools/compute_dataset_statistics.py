#!/usr/bin/env python3
"""Build a MR0 dataset and save its merged normalization statistics."""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from mr0.dataloader.lerobot_datasets import get_vla_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--data-mix", required=True)
    parser.add_argument("--modality-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--action-horizon", type=int, required=True)
    parser.add_argument("--advantage-root")
    parser.add_argument("--advantage-filename")
    parser.add_argument("--gripper-mask-root", required=True)
    parser.add_argument("--gripper-mask-mode", default="loss_mask")
    args = parser.parse_args()

    if args.action_horizon <= 0:
        raise ValueError(f"action horizon must be positive, got {args.action_horizon}")
    if (args.advantage_root is None) != (args.advantage_filename is None):
        raise ValueError("advantage root and filename must be provided together")

    cfg = OmegaConf.load(args.config_yaml).datasets.vla_data
    cfg.data_root_dir = args.data_root
    cfg.data_mix = args.data_mix
    cfg.modality_root_dir = args.modality_root
    cfg.action_horizon = args.action_horizon
    cfg.advantage_root_dir = args.advantage_root
    cfg.advantage_sidecar_filename = args.advantage_filename
    cfg.gripper_mask_root_dir = args.gripper_mask_root
    cfg.gripper_mask_mode = args.gripper_mask_mode

    dataset = get_vla_dataset(data_cfg=cfg)
    output = Path(args.output)
    dataset.save_dataset_statistics(output)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"dataset statistics were not written: {output}")
    print(f"Validated {len(dataset.datasets)} subsets and {len(dataset)} sampled windows")


if __name__ == "__main__":
    main()
