# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""

import copy
import gc
import hashlib
import io
import json
import math
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch.distributed as dist
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from mr0.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from mr0.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from mr0.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from mr0.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps

# LeRobot v2.0 dataset file names
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_EPISODE_STATS_FILENAME = "meta/episodes_stats.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats_gr00t.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
LE_ROBOT_STATS_FORMAT_VERSION = 3
EPSILON = 5e-4
QUANTILE_HISTOGRAM_BINS = 5000

#  LeRobot v3.0 dataset file names
LE_ROBOT3_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/*/*.parquet"


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    # parquet_paths = parquet_paths[:3]
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)

    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in tqdm(all_low_dim_data.columns, desc="Processing modalities"):
        print(le_modality)
        if "task_info" in le_modality:
            continue
        print(f"Computing statistics for {le_modality}...")
        # Check if data is empty or invalid
        try:
            np_data = np.vstack([np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]])
        except Exception as e:
            print(f"Warning: Failed to process modality {le_modality} due to error: {e}")
            continue

        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


def _normalize_action_mode(mode: str) -> str:
    """Normalize action mode names to {abs, delta, rel}."""
    mode = str(mode).lower()
    if mode in {"absolute", "raw"}:
        mode = "abs"
    if mode not in {"abs", "delta", "rel"}:
        mode = "abs"
    return mode


def _normalize_lerobot_version(version: str | None) -> str:
    """Normalize LeRobot version labels to the loaders implemented here."""
    if version is None:
        return "v2.0"

    version = str(version).strip().lower()
    aliases = {
        "2": "v2.0",
        "2.0": "v2.0",
        "v2": "v2.0",
        "v2.0": "v2.0",
        "2.1": "v2.0",
        "v2.1": "v2.0",
        "3": "v3.0",
        "3.0": "v3.0",
        "v3": "v3.0",
        "v3.0": "v3.0",
    }
    if version not in aliases:
        raise ValueError(f"Unsupported lerobot_version: {version}. Expected v2.0/v2.1 or v3.0.")
    return aliases[version]


def _normalize_action_mode_apply_keys(
    action_mode_apply_keys: Sequence[str] | None,
    fallback_keys: Sequence[str] | None = None,
) -> list[str]:
    source_keys = action_mode_apply_keys if action_mode_apply_keys else (fallback_keys or [])
    normalized = []
    for key in source_keys:
        key = str(key)
        if not key.startswith("action."):
            key = f"action.{key}"
        normalized.append(key)
    return normalized


def _normalize_action_mode_state_map(action_mode_state_map: dict[str, str] | None) -> dict[str, str]:
    normalized = {}
    for action_key, state_key in (action_mode_state_map or {}).items():
        action_key = str(action_key)
        state_key = str(state_key)
        if not action_key.startswith("action."):
            action_key = f"action.{action_key}"
        if not state_key.startswith("state."):
            state_key = f"state.{state_key}"
        normalized[action_key] = state_key
    return normalized


def _build_stats_cache_config(
    action_mode: str,
    lerobot_version: str,
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
    parquet_paths: list[Path],
    dataset_path: Path,
    lerobot_modality_meta: "LeRobotModalityMetadata",
) -> dict:
    parquet_fingerprint = []
    for parquet_path in sorted(parquet_paths):
        stat = parquet_path.stat()
        try:
            rel_path = parquet_path.relative_to(dataset_path).as_posix()
        except ValueError:
            rel_path = parquet_path.as_posix()
        parquet_fingerprint.append(
            {
                "path": rel_path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    return {
        "mode": action_mode,
        "lerobot_version": lerobot_version,
        "action_keys_full": list(action_keys_full),
        "state_keys_full": list(state_keys_full),
        "action_indices": list(action_indices) if action_indices is not None else None,
        "state_indices": list(state_indices) if state_indices is not None else None,
        "action_mode_apply_keys": list(action_mode_apply_keys or []),
        "action_mode_state_map": dict(sorted((action_mode_state_map or {}).items())),
        "modality_metadata": lerobot_modality_meta.model_dump(mode="json"),
        "parquet_files": parquet_fingerprint,
    }


def _invalidate_legacy_stats_cache(stats_path: Path, reason: str) -> None:
    if not stats_path.exists():
        return
    print(f"Removing stale dataset statistics cache at {stats_path}: {reason}")
    stats_path.unlink()


def _load_stats_cache(
    stats_path: Path,
    expected_config: dict,
    *,
    invalidate_legacy: bool,
) -> dict | None:
    if not stats_path.exists():
        return None

    try:
        with open(stats_path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, f"failed to load JSON ({exc})")
        return None

    if not isinstance(payload, dict):
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "unexpected top-level format")
        return None

    format_version = payload.get("__format_version")
    cache_config = payload.get("__cache_config")
    statistics = payload.get("statistics")
    if format_version != LE_ROBOT_STATS_FORMAT_VERSION or cache_config is None or statistics is None:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "legacy statistics format detected")
        return None

    if cache_config != expected_config:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "statistics config mismatch, rebuilding cache")
        return None

    return statistics


def _save_stats_cache(stats_path: Path, cache_config: dict, statistics: dict) -> None:
    payload = {
        "__format_version": LE_ROBOT_STATS_FORMAT_VERSION,
        "__cache_config": cache_config,
        "statistics": statistics,
    }
    tmp_path = stats_path.with_suffix(".tmp")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=4)
    os.replace(tmp_path, stats_path)


def _count_from_stats(stats: dict) -> int:
    count = stats.get("count", 0)
    if isinstance(count, list):
        count = count[0] if count else 0
    return int(count)


def _aggregate_episode_statistics(episode_stats_path: Path) -> dict | None:
    """Aggregate LeRobot episode-level statistics into dataset-level statistics."""
    if not episode_stats_path.exists():
        return None

    accum: dict[str, dict] = {}
    with episode_stats_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for modality, stats in row.get("stats", {}).items():
                count = _count_from_stats(stats)
                if count <= 0:
                    continue
                mean = np.asarray(stats["mean"], dtype=np.float64)
                std = np.asarray(stats["std"], dtype=np.float64)
                min_value = np.asarray(stats["min"], dtype=np.float64)
                max_value = np.asarray(stats["max"], dtype=np.float64)

                item = accum.setdefault(
                    modality,
                    {
                        "count": 0,
                        "sum": np.zeros_like(mean),
                        "sum_sq": np.zeros_like(mean),
                        "min": min_value.copy(),
                        "max": max_value.copy(),
                    },
                )
                item["count"] += count
                item["sum"] += mean * count
                item["sum_sq"] += (std**2 + mean**2) * count
                item["min"] = np.minimum(item["min"], min_value)
                item["max"] = np.maximum(item["max"], max_value)

    if not accum:
        return None

    statistics = {}
    for modality, item in accum.items():
        count = item["count"]
        mean = item["sum"] / count
        variance = np.maximum(item["sum_sq"] / count - mean**2, 0.0)
        std = np.sqrt(variance)
        statistics[modality] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": item["min"].tolist(),
            "max": item["max"].tolist(),
            # Exact dataset quantiles are not recoverable from per-episode
            # moments. When a mixture uses min_max normalization these keys
            # are still present but conservatively set to min/max.
            "q01": item["min"].tolist(),
            "q99": item["max"].tolist(),
        }
    return statistics


def _fill_quantiles_from_parquet(
    statistics: dict,
    parquet_paths: list[Path],
    quantiles: tuple[float, ...] = (0.01, 0.99),
) -> None:
    """Add real q01/q99 to per-episode-aggregated statistics via a streaming histogram pass.

    Per-episode moments carry no distributional shape, so quantiles need the raw frames. This is
    openpi's histogram trick (see eva-openpi shared/normalize.py RunningStats), minus the rebinning:
    the aggregate already knows the exact global min/max, so the bin edges can be pinned up front
    and one pass suffices.
    """
    modalities = [m for m in statistics if "task_info" not in m]
    edges = {}
    hists = {}
    counts = {m: 0 for m in modalities}
    for modality in modalities:
        lo = np.asarray(statistics[modality]["min"], dtype=np.float64)
        hi = np.asarray(statistics[modality]["max"], dtype=np.float64)
        # Degenerate dims (lo == hi) would produce zero-width bins; nudge so np.histogram accepts them.
        hi = np.where(hi > lo, hi, lo + 1e-10)
        edges[modality] = [
            np.linspace(lo[i] - 1e-10, hi[i] + 1e-10, QUANTILE_HISTOGRAM_BINS + 1) for i in range(lo.size)
        ]
        hists[modality] = np.zeros((lo.size, QUANTILE_HISTOGRAM_BINS), dtype=np.int64)

    for parquet_path in tqdm(sorted(parquet_paths), desc="Quantile histogram pass"):
        frame = pd.read_parquet(parquet_path)
        for modality in modalities:
            if modality not in frame.columns:
                continue
            np_data = np.vstack([np.asarray(x, dtype=np.float64) for x in frame[modality]])
            counts[modality] += np_data.shape[0]
            for i in range(np_data.shape[1]):
                hists[modality][i] += np.histogram(np_data[:, i], bins=edges[modality][i])[0]

    for modality in modalities:
        if counts[modality] == 0:
            continue
        cumsum = np.cumsum(hists[modality], axis=1)
        for q in quantiles:
            target = q * counts[modality]
            idx = [np.searchsorted(cumsum[i], target) for i in range(cumsum.shape[0])]
            statistics[modality][f"q{int(round(q * 100)):02d}"] = [
                float(edges[modality][i][idx[i]]) for i in range(len(idx))
            ]


def _compute_statistics_for_mode(
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
    episode_stats_path: Path | None = None,
) -> dict:
    print(f"[RANK 0] Calculating dataset statistics for {dataset_name} (mode={action_mode})")

    base_stats = None
    if action_mode == "abs" and episode_stats_path is not None:
        base_stats = _aggregate_episode_statistics(episode_stats_path)
        if base_stats is not None:
            print(f"[RANK 0] Loaded dataset statistics for {dataset_name} from {episode_stats_path}")
            _fill_quantiles_from_parquet(base_stats, parquet_paths)

    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    if action_mode == "abs":
        return base_stats

    if action_indices is None or state_indices is None:
        raise ValueError(
            "Both action and state modalities are required to compute " f"{action_mode} action mode statistics."
        )

    if action_mode == "delta":
        return calculate_delta_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    if action_mode == "rel":
        return calculate_rel_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    raise ValueError(f"Unsupported action mode for statistics: {action_mode}")


def _load_or_compute_statistics(
    stats_path: Path,
    stats_cache_config: dict,
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
    episode_stats_path: Path | None = None,
) -> dict:
    le_statistics = _load_stats_cache(
        stats_path,
        stats_cache_config,
        invalidate_legacy=True,
    )
    if le_statistics is not None:
        return le_statistics

    le_statistics = _compute_statistics_for_mode(
        parquet_paths=parquet_paths,
        dataset_name=dataset_name,
        action_mode=action_mode,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        state_keys_full=state_keys_full,
        action_indices=action_indices,
        state_indices=state_indices,
        action_mode_apply_keys=action_mode_apply_keys,
        action_mode_state_map=action_mode_state_map,
        episode_stats_path=episode_stats_path,
    )
    _save_stats_cache(stats_path, stats_cache_config, le_statistics)
    return le_statistics


def _get_action_col_slices(
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
) -> dict[str, list[tuple[tuple[int, int], str, tuple[int, int], str, str]]]:
    apply_keys = _normalize_action_mode_apply_keys(action_mode_apply_keys, action_keys_full)
    action_mode_state_map = _normalize_action_mode_state_map(action_mode_state_map)

    action_meta = lerobot_modality_meta.action
    state_meta = lerobot_modality_meta.state

    # Build per-column mapping: action column -> list of (action_slice, state_column, state_slice)
    action_col_slices: dict[str, list[tuple[tuple[int, int], str, tuple[int, int]]]] = {}
    for action_key in apply_keys:
        if not action_key.startswith("action."):
            raise ValueError(f"Invalid action key {action_key}. Expected prefix 'action.'.")
        state_key = action_mode_state_map.get(action_key, action_key.replace("action.", "state.", 1))
        if state_key not in state_keys_full:
            raise ValueError(
                f"State key {state_key} not found for action key {action_key}. "
                f"Add it to action_mode_state_map or remove {action_key} from action_mode_apply_keys."
            )

        action_subkey = action_key.replace("action.", "", 1)
        state_subkey = state_key.replace("state.", "", 1)
        if action_subkey not in action_meta or state_subkey not in state_meta:
            raise ValueError(f"Action/state key missing in metadata: {action_key} -> {state_key}")

        action_cfg = action_meta[action_subkey]
        state_cfg = state_meta[state_subkey]
        action_col = action_cfg.original_key or action_subkey
        state_col = state_cfg.original_key or state_subkey
        action_slice = (action_cfg.start, action_cfg.end)
        state_slice = (state_cfg.start, state_cfg.end)
        action_padding = "first_last" if action_cfg.absolute else "zero"
        state_padding = "first_last" if state_cfg.absolute else "zero"
        action_col_slices.setdefault(action_col, []).append(
            (action_slice, state_col, state_slice, action_padding, state_padding)
        )

    return action_col_slices


def calculate_delta_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    """
    Calculate action statistics using delta mode.

    Rule:
      - For t>0: a_t - a_{t-1}
      - For t=0: a_0 - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting delta action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk.copy()
                    if len(out) > 1:
                        out[1:] = action_part_chunk[1:] - action_part_chunk[:-1]
                    out[0] = action_part_chunk[0] - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    delta_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        delta_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return delta_stats


def calculate_rel_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    """
    Calculate action statistics using rel mode.

    Rule:
      - For all t: a_t - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting rel action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    rel_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        rel_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return rel_stats


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg=None,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        self.data_cfg = data_cfg
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        # indict letobot version
        version = kwargs.get("lerobot_version", None)
        if version is None and self.data_cfg is not None:
            version = self.data_cfg.get("lerobot_version", None)
        self._lerobot_version = _normalize_lerobot_version(version)

        self._action_mode = None
        self._action_mode_state_map = {}
        self._action_mode_apply_keys = None

        self.delete_pause_frame = delete_pause_frame

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = transforms if transforms is not None else ComposedModalityTransform(transforms=[])

        self._dataset_path = Path(dataset_path)
        self._data_root_dir = Path(kwargs["data_root_dir"]) if "data_root_dir" in kwargs else None
        self._modality_root_dir = Path(kwargs["modality_root_dir"]) if kwargs.get("modality_root_dir") is not None else None
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        self._init_action_mode()
        self._metadata = self._get_metadata(EmbodimentTag(self.tag))

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._init_gripper_mask()
        self._init_advantage_sidecar()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        # self._episodes = self._get_episode_info() # TODO why we need this func
        self.curr_traj_data = None
        self.curr_traj_id = None

        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._maybe_apply_episode_subset()
        self._validate_advantage_coverage()
        self._num_steps = int(np.sum(self._trajectory_lengths))
        self._trajectory_end_offsets = np.cumsum(self._trajectory_lengths)
        lazy_cfg = self.data_cfg.get("lazy_step_sampling", False) if self.data_cfg is not None else False
        self._lazy_step_sampling = lazy_cfg not in ["False", False]
        # Static-frame masking (ported from openpi's collector_full_chunk_only): keep a start
        # frame only if the whole H-frame action window is `collector==1` (moving) and stays
        # within one episode. No-op for datasets without a `collector` column.
        cfc_cfg = self.data_cfg.get("collector_full_chunk_only", False) if self.data_cfg is not None else False
        self._collector_full_chunk_only = cfc_cfg not in ["False", False]
        # Window length = action horizon (number of future action steps sampled per step).
        self._collector_chunk_horizon = self._get_action_horizon()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._all_steps = self._get_all_steps()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _init_gripper_mask(self) -> None:
        """Load an optional per-frame RECAP gripper-mask sidecar.

        ``base_sample`` filtering drops a training window when its base frame has
        ``mask_all`` set. ``mask_gripper`` is deliberately not used for filtering:
        it is reserved for the later loss-mask integration, where only gripper
        dimensions (9 and 19) are excluded from the loss.
        """
        cfg = self.data_cfg or {}
        mode = cfg.get("gripper_mask_mode", "none")
        if mode not in {"none", "base_sample", "loss_mask", "base_sample_and_loss_mask"}:
            raise ValueError(
                f"Unsupported gripper_mask_mode={mode!r}; expected 'none', 'base_sample', "
                "'loss_mask', or 'base_sample_and_loss_mask'."
            )
        self._gripper_mask_mode = mode
        self._gripper_mask_indices = tuple(int(index) for index in cfg.get("gripper_mask_indices", (9, 19)))
        self._gripper_mask_lookup = {}
        if mode == "none":
            return

        mask_root = cfg.get("gripper_mask_root_dir", None)
        if mask_root:
            if self._data_root_dir is not None:
                try:
                    relative_path = self.dataset_path.relative_to(self._data_root_dir)
                except ValueError:
                    relative_path = Path(self.dataset_name)
            else:
                relative_path = Path(self.dataset_name)
            mask_path = Path(mask_root) / relative_path / "meta" / "gripper_mask.parquet"
        else:
            mask_path = self.dataset_path / "meta" / "gripper_mask.parquet"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"gripper_mask_mode={mode!r} requires {mask_path}. "
                "Generate the sidecar first or set gripper_mask_mode='none'."
            )

        table = pd.read_parquet(mask_path, columns=["episode_index", "frame_index", "mask_gripper", "mask_all"])
        required_columns = {"episode_index", "frame_index", "mask_gripper", "mask_all"}
        missing_columns = required_columns - set(table.columns)
        if missing_columns:
            raise ValueError(f"Invalid gripper mask sidecar {mask_path}: missing {sorted(missing_columns)}")
        self._gripper_mask_lookup = {
            (int(row.episode_index), int(row.frame_index)): (bool(row.mask_gripper), bool(row.mask_all))
            for row in table.itertuples(index=False)
        }
        print(f"[gripper-mask] loaded {len(self._gripper_mask_lookup)} frame labels from {mask_path} (mode={mode})")

    def _init_advantage_sidecar(self) -> None:
        """Load an optional per-frame boolean advantage sidecar for prompt routing."""
        cfg = self.data_cfg or {}
        filename = cfg.get("advantage_sidecar_filename", None)
        self._advantage_lookup = None
        self._advantage_sidecar_path = None
        if filename in (None, ""):
            return

        filename = str(filename)
        if Path(filename).name != filename:
            raise ValueError(
                f"advantage_sidecar_filename must be a filename, got {filename!r}. "
                "Use advantage_root_dir to select the sidecar root."
            )

        advantage_root = cfg.get("advantage_root_dir", None)
        if advantage_root:
            if self._data_root_dir is not None:
                try:
                    relative_path = self.dataset_path.relative_to(self._data_root_dir)
                except ValueError:
                    relative_path = Path(self.dataset_name)
            else:
                relative_path = Path(self.dataset_name)
            sidecar_path = Path(advantage_root) / relative_path / "meta" / filename
        else:
            sidecar_path = self.dataset_path / "meta" / filename
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"advantage_sidecar_filename={filename!r} requires {sidecar_path}. "
                "Generate the sidecar first or unset advantage_sidecar_filename."
            )

        import pyarrow.parquet as pq

        metadata = pq.read_schema(sidecar_path).metadata or {}
        encoded_n_step = metadata.get(b"n_step")
        if encoded_n_step is None:
            raise ValueError(
                f"Advantage sidecar {sidecar_path} has no n_step metadata; regenerate it for "
                f"action_horizon={self._get_action_horizon()}."
            )
        sidecar_n_step = int(encoded_n_step.decode())
        action_horizon = self._get_action_horizon()
        if sidecar_n_step != action_horizon:
            raise ValueError(
                f"Advantage sidecar horizon mismatch for {sidecar_path}: "
                f"n_step={sidecar_n_step}, action_horizon={action_horizon}"
            )

        table = pd.read_parquet(sidecar_path, columns=["episode_index", "frame_index", "advantage"])
        required_columns = {"episode_index", "frame_index", "advantage"}
        missing_columns = required_columns - set(table.columns)
        if missing_columns:
            raise ValueError(f"Invalid advantage sidecar {sidecar_path}: missing {sorted(missing_columns)}")
        if table[list(required_columns)].isnull().any().any():
            null_counts = table[list(required_columns)].isnull().sum().to_dict()
            raise ValueError(f"Invalid advantage sidecar {sidecar_path}: null values {null_counts}")
        duplicate_rows = table.duplicated(["episode_index", "frame_index"], keep=False)
        if duplicate_rows.any():
            examples = table.loc[duplicate_rows, ["episode_index", "frame_index"]].head(5).to_dict("records")
            raise ValueError(
                f"Invalid advantage sidecar {sidecar_path}: duplicate (episode_index, frame_index) keys; "
                f"examples={examples}"
            )
        if not pd.api.types.is_bool_dtype(table["advantage"].dtype):
            raise ValueError(
                f"Invalid advantage sidecar {sidecar_path}: advantage must be boolean, got {table['advantage'].dtype}"
            )

        self._advantage_lookup = {
            (int(row.episode_index), int(row.frame_index)): bool(row.advantage)
            for row in table.itertuples(index=False)
        }
        self._advantage_sidecar_path = sidecar_path
        print(
            f"[advantage] loaded {len(self._advantage_lookup)} unique frame labels "
            f"from {sidecar_path} for subset={self.dataset_name}"
        )

    def _validate_advantage_coverage(self) -> None:
        """Require the sidecar to cover every source frame before sampling starts."""
        if self._advantage_lookup is None:
            return

        source_keys = set()
        for trajectory_id in self.trajectory_ids:
            if self._lerobot_version == "v2.0":
                data = self.get_trajectory_data(trajectory_id)
            else:
                data = self.get_trajectory_data_lerobot_v3(trajectory_id)
            source_keys.update(
                (int(row.episode_index), int(row.frame_index))
                for row in data[["episode_index", "frame_index"]].itertuples(index=False)
            )

        sidecar_keys = set(self._advantage_lookup)
        missing_keys = source_keys - sidecar_keys
        extra_keys = sidecar_keys - source_keys
        if missing_keys or extra_keys:
            raise ValueError(
                f"Advantage sidecar coverage mismatch for subset={self.dataset_name!r}: "
                f"source_frames={len(source_keys)}, labels={len(sidecar_keys)}, "
                f"missing={len(missing_keys)} (examples={sorted(missing_keys)[:5]}), "
                f"extra={len(extra_keys)} (examples={sorted(extra_keys)[:5]}), "
                f"path={self._advantage_sidecar_path}"
            )
        print(f"[advantage] verified 100% coverage for {len(source_keys)} frames in subset={self.dataset_name}")

    def _get_modality_meta_path(self) -> Path:
        """Return dataset modality metadata path, optionally from a shadow metadata root."""
        shadow_path = self._get_shadow_meta_path("modality.json")
        if shadow_path is not None and shadow_path.exists():
            return shadow_path
        return self.dataset_path / LE_ROBOT_MODALITY_FILENAME

    def _get_shadow_meta_path(self, filename: str) -> Path | None:
        """Return a file path under modality_root_dir mirroring the dataset layout.

        Per-instance ``self._modality_root_dir`` (set from a 5-tuple mixture entry)
        takes precedence over the cfg-wide ``data_cfg.modality_root_dir`` so a
        single mixture can pull datasets from different modality shadow roots.
        """
        modality_root_dir = self._modality_root_dir
        if modality_root_dir is None and self.data_cfg is not None:
            cfg_mod_root = self.data_cfg.get("modality_root_dir", None)
            if cfg_mod_root:
                modality_root_dir = Path(cfg_mod_root)
        if modality_root_dir is None:
            return None

        data_root_dirs = []
        if self._data_root_dir is not None:
            data_root_dirs.append(self._data_root_dir)
        if self.data_cfg is not None:
            cfg_data_root_dir = self.data_cfg.get("data_root_dir", None)
            if cfg_data_root_dir:
                data_root_dirs.append(Path(cfg_data_root_dir))

        if data_root_dirs:
            seen_roots = set()
            for data_root_dir in data_root_dirs:
                data_root_dir = Path(data_root_dir)
                root_key = str(data_root_dir)
                if root_key in seen_roots:
                    continue
                seen_roots.add(root_key)
                try:
                    rel_dataset_path = self.dataset_path.relative_to(data_root_dir)
                except ValueError:
                    continue
                return Path(modality_root_dir) / rel_dataset_path / "meta" / filename
            rel_dataset_path = Path(self.dataset_name)
            return Path(modality_root_dir) / rel_dataset_path / "meta" / filename
        return None

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self._get_modality_meta_path()
        assert modality_meta_path.exists(), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file for {self.dataset_path}"
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(le_modality_meta, modality)
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [le_state_action_meta[subkey].end - le_state_action_meta[subkey].start],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert le_info_path.exists(), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                try:
                    channels = le_video_meta["info"]["video.channels"]
                    fps = le_video_meta["info"]["video.fps"]
                except KeyError:
                    # Image features (dtype=="image") have no video_info/info; use shape and dataset-level fps
                    channel_key = "channel" if "channel" in le_video_meta.get("names", []) else "channels"
                    channels = le_video_meta["shape"][le_video_meta["names"].index(channel_key)]
                    fps = le_info.get("fps", 10)
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0

        action_mode = _normalize_action_mode(self.data_cfg.get("action_mode", "abs") if self.data_cfg else "abs")

        shadow_stats_path = self._get_shadow_meta_path("stats_gr00t.json")
        stats_path = shadow_stats_path if shadow_stats_path is not None else self.dataset_path / LE_ROBOT_STATS_FILENAME
        action_cfg = self.modality_configs.get("action")
        state_cfg = self.modality_configs.get("state")
        action_keys_full = list(action_cfg.modality_keys) if action_cfg else []
        state_keys_full = list(state_cfg.modality_keys) if state_cfg else []
        action_indices = list(action_cfg.delta_indices) if action_cfg else None
        state_indices = list(state_cfg.delta_indices) if state_cfg else None

        apply_keys = _normalize_action_mode_apply_keys(
            self.data_cfg.get("action_mode_apply_keys", None) if self.data_cfg else None,
            action_keys_full,
        )
        normalized_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) if self.data_cfg else {}
        )
        parquet_files = list(self.dataset_path.glob(LE_ROBOT_DATA_FILENAME))
        parquet_files_filtered = sorted(parquet_files)
        stats_cache_config = _build_stats_cache_config(
            action_mode=action_mode,
            lerobot_version=self._lerobot_version,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=apply_keys,
            action_mode_state_map=normalized_state_map,
            parquet_paths=parquet_files_filtered,
            dataset_path=self.dataset_path,
            lerobot_modality_meta=le_modality_meta,
        )

        if is_main():
            le_statistics = _load_or_compute_statistics(
                stats_path,
                stats_cache_config=stats_cache_config,
                parquet_paths=parquet_files_filtered,
                dataset_name=self.dataset_name,
                action_mode=action_mode,
                lerobot_modality_meta=le_modality_meta,
                action_keys_full=action_keys_full,
                state_keys_full=state_keys_full,
                action_indices=action_indices,
                state_indices=state_indices,
                action_mode_apply_keys=apply_keys,
                action_mode_state_map=normalized_state_map,
                episode_stats_path=self.dataset_path / LE_ROBOT_EPISODE_STATS_FILENAME,
            )
        else:
            le_statistics = None

        if dist.is_initialized():
            dist.barrier()

        if le_statistics is None:
            le_statistics = _load_stats_cache(
                stats_path,
                stats_cache_config,
                invalidate_legacy=False,
            )
            if le_statistics is None:
                raise RuntimeError(f"Dataset statistics cache is missing or invalid after sync: {stats_path}")

        for stat in le_statistics.values():
            DatasetStatisticalValues.model_validate(stat)

        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        # v2.0
        if self._lerobot_version == "v2.0":
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
            return np.array(trajectory_ids), np.array(trajectory_lengths)
        # v3.0
        elif self._lerobot_version == "v3.0":
            file_paths = sorted(list((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME)))
            trajectory_ids = []
            trajectory_lengths = []
            # data_chunck_index = []
            # data_file_index = []
            # vido_from_index = []
            self.trajectory_ids_to_metadata = {}
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                timestamp_cols = [
                    c
                    for c in episodes_data.columns
                    if str(c).startswith("videos/") and str(c).endswith("/from_timestamp")
                ]
                for index, episode in episodes_data.iterrows():
                    trajectory_ids.append(episode["episode_index"])
                    trajectory_lengths.append(episode["length"])

                    from_timestamps = {}
                    for col in timestamp_cols:
                        value = episode[col]
                        if pd.isna(value):
                            continue
                        # videos/{video_key}/from_timestamp -> {video_key}
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        from_timestamps[video_key] = float(value)

                    # TODO auto map key? just map to file_path and file_from_index
                    episode_meta = {
                        "data/chunk_index": episode["data/chunk_index"],
                        "data/file_index": episode["data/file_index"],
                        "data/file_from_index": index,
                        "videos/from_timestamps": from_timestamps,
                    }
                    self.trajectory_ids_to_metadata[trajectory_ids[-1]] = episode_meta

            # Should be able to read the saved index info directly here
            return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _maybe_apply_episode_subset(self) -> None:
        """Optionally keep only a random subset of episodes per leaf dataset.

        Triggered by ``data_cfg.episode_ratio`` (float in (0, 1]). Selection is
        seeded by ``data_cfg.episode_subset_seed`` (default 42) combined with
        the leaf dataset name so the same yaml deterministically picks the same
        episodes across runs and workers. Always keeps at least one episode
        even when ``ceil(N * ratio) < 1`` due to a tiny leaf.
        """
        if self.data_cfg is None:
            return
        ratio = self.data_cfg.get("episode_ratio", None)
        if ratio is None:
            return
        try:
            ratio = float(ratio)
        except (TypeError, ValueError) as e:
            raise ValueError(f"episode_ratio must be a float, got {ratio!r}") from e
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"episode_ratio must be in (0, 1], got {ratio}")
        if ratio == 1.0:
            return

        n = len(self._trajectory_ids)
        if n == 0:
            return
        keep = max(1, int(math.ceil(n * ratio)))
        if keep >= n:
            return

        base_seed = int(self.data_cfg.get("episode_subset_seed", 42))
        # Fold leaf name into seed so different leaves get different subsets
        # while a given (yaml, leaf) pair is deterministic across runs.
        name_seed = int(hashlib.sha1(self._dataset_name.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(base_seed ^ name_seed)
        sel = np.sort(rng.choice(n, size=keep, replace=False))

        self._trajectory_ids = self._trajectory_ids[sel]
        self._trajectory_lengths = self._trajectory_lengths[sel]
        print(
            f"[LeRobotSingleDataset] {self._dataset_name}: "
            f"episode_ratio={ratio} -> kept {keep}/{n} episodes"
        )

    def _get_action_horizon(self) -> int:
        """Number of future action steps sampled per step (length of the action delta window).

        Used as the window length for collector_full_chunk_only static-frame masking.
        Falls back to 1 if no action modality is configured.
        """
        action_cfg = self.modality_configs.get("action", None)
        if action_cfg is None or not action_cfg.delta_indices:
            return 1
        return len(action_cfg.delta_indices)

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.
        """

        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0

        if self._lazy_step_sampling:
            return []

        config_key = self._get_steps_config_key()
        steps_filename = "steps_data_index.pkl"
        shadow_steps_path = self._get_shadow_meta_path(steps_filename)
        steps_path = shadow_steps_path if shadow_steps_path is not None else self.dataset_path / "meta" / steps_filename

        # ---------- try to read from cache  ----------
        if steps_path.exists():
            try:
                with open(steps_path, "rb") as f:
                    cached_data = pickle.load(f)
                cached_key = cached_data.get("config_key", None)
                cached_num_traj = cached_data.get("num_trajectories", None)
                if cached_key == config_key and cached_num_traj == len(self.trajectory_ids):
                    return cached_data["steps"]
                else:
                    print(
                        f"[RANK {os.environ.get('RANK', 'NA')}] "
                        f"Steps cache invalidated (key: {cached_key} vs {config_key}, "
                        f"num_traj: {cached_num_traj} vs {len(self.trajectory_ids)}), rebuilding."
                    )
            except Exception as e:
                print(f"[RANK {os.environ.get('RANK', 'NA')}] " f"Failed to load cached steps ({e}), will rebuild.")

        # ---------- only build by rank0  ----------
        if is_main():
            all_steps = self._get_all_steps_single_process()

            cache_data = {
                "config_key": config_key,
                "steps": all_steps,
                "num_trajectories": len(self.trajectory_ids),
                "total_steps": len(all_steps),
                "computed_timestamp": pd.Timestamp.now().isoformat(),
                "delete_pause_frame": self.delete_pause_frame,
            }

            steps_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = steps_path.with_suffix(".tmp")

            with open(tmp_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, steps_path)

            print(f"[RANK 0] Cached steps saved to {steps_path}")

        # ---------- sync after rank0  ----------
        if dist.is_initialized():
            dist.barrier()

        # ---------- read by all rank ----------
        with open(steps_path, "rb") as f:
            cached_data = pickle.load(f)

        return cached_data["steps"]

    def _get_steps_config_key(self) -> str:
        """Generate a configuration key for steps caching."""
        config_dict = {
            "delete_pause_frame": self.delete_pause_frame,
            "dataset_name": self.dataset_name,
            "split": getattr(self, "split", None),
            "filter": getattr(self, "filter_key", None),
            "collector_full_chunk_only": self._collector_full_chunk_only,
            "collector_chunk_horizon": self._collector_chunk_horizon if self._collector_full_chunk_only else None,
        }
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _valid_base_indices(self, trajectory_id: int, trajectory_length: int, data: "pd.DataFrame | None" = None):
        """Return the list of valid start indices for a trajectory.

        Default: every frame is a valid start (0 .. trajectory_length-1).

        When ``collector_full_chunk_only`` is enabled and the trajectory carries a
        per-frame ``collector`` column (1=moving, 0=static), a start ``i`` is kept only if
        the whole H-frame action window ``i .. i+H-1`` is all-moving AND fits inside the
        trajectory (no episode-boundary crossing / padding). This mirrors openpi's
        ``_CollectorFullChunkFilteredDataset``. Trajectories without a ``collector`` column
        fall back to the default (no-op).
        """
        if not self._collector_full_chunk_only:
            return range(trajectory_length)

        if data is None:
            if self._lerobot_version == "v2.0":
                data = self.get_trajectory_data(trajectory_id)
            elif self._lerobot_version == "v3.0":
                data = self.get_trajectory_data_lerobot_v3(trajectory_id)
        if data is None or "collector" not in data.columns:
            return range(trajectory_length)

        collector = np.asarray(data["collector"]).reshape(-1).astype(np.int64)
        H = self._collector_chunk_horizon
        n = len(collector)
        # moving_run[i] = number of consecutive moving frames starting at i.
        moving_run = np.zeros(n, dtype=np.int64)
        run = 0
        for i in range(n - 1, -1, -1):
            run = run + 1 if collector[i] == 1 else 0
            moving_run[i] = run
        # Valid start: window i..i+H-1 all moving AND fully in-trajectory (i+H <= length).
        limit = min(n, trajectory_length)
        valid = [i for i in range(limit) if i + H <= trajectory_length and moving_run[i] >= H]
        return valid

    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        """Original single-process implementation as fallback."""
        all_steps: list[tuple[int, int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0

        # Check if language modality is configured
        has_language_modality = "language" in self.modality_keys and len(self.modality_keys["language"]) > 0
        skip_empty_language_check = False
        if self.data_cfg is not None:
            skip_empty_language_check = self.data_cfg.get("skip_empty_language_check", False) not in ["False", False]
        # TODO why trajectory_length here, why not use data length?
        for trajectory_id, trajectory_length in tqdm(
            zip(self.trajectory_ids, self.trajectory_lengths), total=len(self.trajectory_ids), desc="Getting All Step"
        ):
            if skip_empty_language_check and not self._collector_full_chunk_only:
                processed_trajectories += 1
                for base_index in range(trajectory_length):
                    all_steps.append((trajectory_id, base_index))
                continue

            try:
                if self._lerobot_version == "v2.0":
                    data = self.get_trajectory_data(trajectory_id)
                elif self._lerobot_version == "v3.0":
                    data = self.get_trajectory_data_lerobot_v3(trajectory_id)

                trajectory_skipped = False

                # Check if trajectory has valid language instruction (if language modality is configured)
                if has_language_modality and not skip_empty_language_check:
                    self.curr_traj_data = data  # Set current trajectory data for get_language to work

                    language_instruction = self.get_language_with_fallback(trajectory_id, 0)
                    if not language_instruction or language_instruction[0] == "":
                        print(f"Skipping trajectory {trajectory_id} due to empty language instruction")
                        skipped_trajectories += 1
                        trajectory_skipped = True
                        continue

            except Exception as e:
                print(f"Skipping trajectory {trajectory_id} due to read error: {e}")
                skipped_trajectories += 1
                trajectory_skipped = True
                continue

            if not trajectory_skipped:
                processed_trajectories += 1

            for base_index in self._valid_base_indices(trajectory_id, trajectory_length, data=data):
                if self._gripper_mask_mode in {"base_sample", "base_sample_and_loss_mask"}:
                    row = data.iloc[base_index]
                    key = (int(row["episode_index"]), int(row["frame_index"]))
                    _, mask_all = self._gripper_mask_lookup.get(key, (False, False))
                    if mask_all:
                        continue
                all_steps.append((trajectory_id, base_index))

        # Print summary statistics
        print(
            f"Single-process summary: Processed {processed_trajectories} trajectories, skipped {skipped_trajectories} empty trajectories"
        )
        if self._collector_full_chunk_only:
            print(
                f"[collector] full-chunk-filter (H={self._collector_chunk_horizon}): "
                f"kept {len(all_steps)} all-moving in-trajectory start windows"
            )
        print(f"Total steps: {len(all_steps)} from {len(self.trajectory_ids)} trajectories")

        return all_steps

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get("action", [])

        # Extract position data
        delta_position_values = None
        position_candidates = ["delta_eef_position"]
        coordinate_candidates = ["x", "y", "z"]

        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue

        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == "x":
                                    x_data = coord_data
                                elif coord == "y":
                                    y_data = coord_data
                                elif coord == "z":
                                    z_data = coord_data
                    except Exception:
                        continue

            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()

        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if "action.delta_eef_position" in data.columns:
                delta_position_values = data["action.delta_eef_position"].to_numpy().tolist()
            elif all(col in data.columns for col in ["action.x", "action.y", "action.z"]):
                x_vals = data["action.x"].to_numpy()
                y_vals = data["action.y"].to_numpy()
                z_vals = data["action.z"].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")

        # Extract gripper data
        gripper_values = None
        gripper_candidates = ["gripper_close", "gripper"]

        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue

        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if "action.gripper_close" in data.columns:
                gripper_values = data["action.gripper_close"].to_numpy().tolist()
            elif "action.gripper" in data.columns:
                gripper_values = data["action.gripper"].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")

        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _init_action_mode(self) -> None:
        if self.data_cfg is None:
            self._action_mode = "abs"
            return

        action_mode = self.data_cfg.get("action_mode", "abs")
        if action_mode is None:
            action_mode = "abs"
        action_mode = _normalize_action_mode(action_mode)
        if action_mode not in {"abs", "delta", "rel"}:
            raise ValueError(f"Invalid action_mode: {action_mode}. Expected one of: abs, delta, rel.")
        self._action_mode = action_mode

        apply_keys = _normalize_action_mode_apply_keys(self.data_cfg.get("action_mode_apply_keys", None))
        if apply_keys:
            self._action_mode_apply_keys = apply_keys

        self._action_mode_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) or {}
        )

    def _infer_state_key_for_action(self, action_key: str) -> str | None:
        if action_key in self._action_mode_state_map:
            return self._action_mode_state_map[action_key]

        if not action_key.startswith("action."):
            return None
        base = action_key.replace("action.", "", 1)
        if f"state.{base}" in self.modality_keys.get("state", []):
            return f"state.{base}"
        return None

    def _apply_action_mode(self, data: dict) -> dict:
        if self._action_mode in (None, "abs"):
            return data

        action_keys = self._action_mode_apply_keys or self.modality_keys.get("action", [])
        for action_key in action_keys:
            if action_key not in data:
                raise ValueError(f"Action key {action_key} not found in data for action_mode={self._action_mode}.")
            state_key = self._infer_state_key_for_action(action_key)

            # for safety, check if the state key is valid
            if state_key is None or state_key not in data:
                raise ValueError(
                    f"State key for {action_key} not found in data for action_mode={self._action_mode}. "
                    "Add action_mode_state_map or remove this key from action_mode_apply_keys."
                )

            action_values = np.asarray(data[action_key])
            state_values = np.asarray(data[state_key])
            if action_values.ndim != 2 or state_values.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays for action/state, got {action_key}: {action_values.shape}, {state_key}: {state_values.shape}"
                )
            if action_values.shape[1] != state_values.shape[1]:
                raise ValueError(
                    f"Action/state dim mismatch for {action_key} vs {state_key}: {action_values.shape} vs {state_values.shape}"
                )

            state0 = state_values[0]
            if self._action_mode == "delta":
                out = action_values.copy()
                if len(out) > 1:
                    out[1:] = action_values[1:] - action_values[:-1]
                out[0] = action_values[0] - state0
            elif self._action_mode == "rel":
                out = action_values - state0
            else:
                out = action_values

            data[action_key] = out

        return data

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self._get_modality_meta_path()
        assert modality_meta_path.exists(), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file for {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str | None:
        """Get the video path pattern, if this LeRobot dataset stores videos.

        Image-feature datasets store per-frame image bytes/paths directly in
        parquet and legitimately omit ``video_path`` from ``meta/info.json``.
        """
        return self.lerobot_info_meta.get("video_path")

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        if self._lerobot_version == "v2.0":
            tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            df = pd.DataFrame(tasks)
            return df.set_index("task_index")

        elif self._lerobot_version == "v3.0":
            tasks_path = self.dataset_path / LE_ROBOT3_TASKS_FILENAME
            df = pd.read_parquet(tasks_path)
            df = df.reset_index()  # Convert the index into a column (typically named 'index')
            df = df.rename(columns={"index": "task"})  # Rename the 'index' column to 'task'
            df = df[["task_index", "task"]]  # Reorder columns
            return df.set_index("task_index")

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}")

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        if self._lazy_step_sampling:
            return self._num_steps
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        if self._lazy_step_sampling:
            trajectory_id, base_index = self.get_step_from_flat_index(index)
        else:
            trajectory_id, base_index = self.all_steps[index]
        raw_data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(raw_data)
        source_indices = self._get_source_indices(trajectory_id, base_index)
        if self._advantage_lookup is not None:
            source_indices["advantage"] = self._get_advantage(trajectory_id, base_index)
        if self._gripper_mask_mode in {"loss_mask", "base_sample_and_loss_mask"}:
            source_indices["action_loss_mask"] = self._get_action_loss_mask(trajectory_id, base_index)
        return self._pack_sample(data, source_indices=source_indices)

    def get_step_from_flat_index(self, index: int) -> tuple[int, int]:
        if self._num_steps <= 0:
            raise ValueError(f"Dataset {self.dataset_name} has no steps.")
        index = int(index) % self._num_steps
        trajectory_index = int(np.searchsorted(self._trajectory_end_offsets, index, side="right"))
        prev_end = 0 if trajectory_index == 0 else int(self._trajectory_end_offsets[trajectory_index - 1])
        return int(self.trajectory_ids[trajectory_index]), index - prev_end

    @staticmethod
    def _as_numpy(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _get_source_indices(self, trajectory_id: int, base_index: int) -> dict:
        """Return immutable source identifiers for a base frame.

        These identifiers are passed through to the training sample so sidecars
        can be joined without changing episode/frame/index semantics.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        row = self.curr_traj_data.iloc[base_index]
        result = {"dataset_name": self.dataset_name}
        for key in ("episode_index", "frame_index", "index", "timestamp", "task_index"):
            if key in row.index:
                value = row[key]
                result[key] = value.item() if hasattr(value, "item") else value
        return result

    def _get_advantage(self, trajectory_id: int, base_index: int) -> bool:
        """Return a frame label without falling back to negative on a missing join."""
        assert self._advantage_lookup is not None
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        row = self.curr_traj_data.iloc[base_index]
        key = (int(row["episode_index"]), int(row["frame_index"]))
        try:
            return self._advantage_lookup[key]
        except KeyError as error:
            raise KeyError(
                f"Missing advantage label for subset={self.dataset_name!r}, "
                f"episode_index={key[0]}, frame_index={key[1]} in {self._advantage_sidecar_path}"
            ) from error

    def _get_action_loss_mask(self, trajectory_id: int, base_index: int) -> np.ndarray:
        """Build a [H, D] loss mask from the gripper-mask sidecar for one action chunk."""
        horizon = self._get_action_horizon()
        action_dim = sum(
            meta.end - meta.start for meta in self.lerobot_modality_meta.action.values()
        )
        mask = np.ones((horizon, action_dim), dtype=bool)
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = int(self.trajectory_lengths[trajectory_index])
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        for offset in range(horizon):
            source_index = base_index + offset
            if source_index >= trajectory_length:
                continue  # Absolute-action padding is intentionally kept in the loss.
            row = self.curr_traj_data.iloc[source_index]
            mask_gripper, mask_all = self._gripper_mask_lookup.get(
                (int(row["episode_index"]), int(row["frame_index"])), (False, False)
            )
            if mask_all:
                mask[offset] = False
            elif mask_gripper:
                mask[offset, self._gripper_mask_indices] = False
        return mask

    def _pack_sample(self, data: dict, source_indices: dict | None = None) -> dict:
        """Pack transformed modality data into training sample format."""
        img_size = self.data_cfg.get("image_size", None) if self.data_cfg else None

        images = []
        if "video" in data:
            video_frame = self._as_numpy(data["video"])[0]
            if video_frame.ndim == 3:
                video_frame = video_frame[None, ...]
            for image_arr in video_frame:
                image = Image.fromarray(image_arr)
                if img_size is not None:
                    if not hasattr(self, "_logged_dataloader_resize"):
                        print(f"[DataLoader] Resizing images from {image.size} to {tuple(img_size)}")
                        self._logged_dataloader_resize = True
                    image = image.resize(tuple(img_size))
                images.append(np.array(image))
        else:
            for video_key in self.modality_keys["video"]:
                image = self._as_numpy(data[video_key])[0]
                image = Image.fromarray(image)
                if img_size is not None:
                    if not hasattr(self, "_logged_dataloader_resize"):
                        print(f"[DataLoader] Resizing images from {image.size} to {tuple(img_size)}")
                        self._logged_dataloader_resize = True
                    image = image.resize(tuple(img_size))
                images.append(np.array(image))

        language_keys = self.modality_keys["language"]
        order = self._sample_language_key_order(language_keys)
        _INVALID = {"null", "none", "nan"}
        language = ""
        for language_key in order:
            values = data.get(language_key, [])
            if len(values) == 0:
                continue
            value = values[0]
            if value is None:
                continue
            value = str(value).strip()
            if not value or value.lower() in _INVALID:
                continue
            language = value
            break
        if "action" in data:
            action = self._as_numpy(data["action"]).astype(np.float16)
        else:
            action = []
            for action_key in self.modality_keys["action"]:
                action.append(self._as_numpy(data[action_key]))
            action = np.concatenate(action, axis=1).astype(np.float16)

        sample = {
            "action": action,
            "image": images,
            "lang": language,
            "language": language,
        }
        if source_indices:
            sample.update(source_indices)

        if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
            if "state" in data:
                state = self._as_numpy(data["state"]).astype(np.float16)
            else:
                state = []
                for state_key in self.modality_keys["state"]:
                    state.append(self._as_numpy(data[state_key]))
                state = np.concatenate(state, axis=1).astype(np.float16)
            sample["state"] = state

        return sample

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities # just for action base data
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # TODO @JinhuiYE The logic below is poorly implemented. Data reading should be directly based on curr_traj_data.
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        data = self._apply_action_mode(data)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self._lerobot_version == "v2.0":

            if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
                return self.curr_traj_data
            else:
                chunk_index = self.get_episode_chunk(trajectory_id)
                parquet_path = self.dataset_path / self.data_path_pattern.format(
                    episode_chunk=chunk_index, episode_index=trajectory_id
                )
                assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
                self.curr_traj_data = pd.read_parquet(parquet_path)
                self.curr_traj_id = trajectory_id
                return self.curr_traj_data
        elif self._lerobot_version == "v3.0":
            return self.get_trajectory_data_lerobot_v3(trajectory_id)

    def get_trajectory_data_lerobot_v3(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory from lerobot v3."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            chunk_index = episode_meta["data/chunk_index"]
            file_index = self.get_episode_file_index(trajectory_id)

            parquet_path = self.dataset_path / self.data_path_pattern.format(
                chunk_index=chunk_index, file_index=file_index
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            file_data = pd.read_parquet(parquet_path)

            # filter by trajectory_id
            episode_data = file_data.loc[file_data["episode_index"] == trajectory_id].copy()
            self.curr_traj_data = episode_data
            self.curr_traj_id = trajectory_id
            return episode_data

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}")
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def get_episode_file_index(self, ep_index: int) -> int:
        """Get the file index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_index"]

    def get_episode_file_from_index(self, ep_index: int) -> int:
        """Get the file from index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_from_index"]

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self._lerobot_version == "v2.0":
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        elif self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            video_filename = self.video_path_pattern.format(
                video_key=original_key,
                chunk_index=episode_meta["data/chunk_index"],
                file_index=episode_meta["data/file_index"],
            )
        return self.dataset_path / video_filename

    def _is_image_feature(self, key: str) -> bool:
        """Whether a configured video modality is stored as image paths/bytes in parquet."""
        subkey = key.removeprefix("video.")
        original_key = self.lerobot_modality_meta.video[subkey].original_key or subkey
        return self.lerobot_info_meta.get("features", {}).get(original_key, {}).get("dtype") == "image"

    def get_image(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Read image-feature frames stored as bytes or paths in a LeRobot parquet."""
        step_indices = self.delta_indices[key] + base_index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        step_indices = np.clip(step_indices, 0, self.trajectory_lengths[trajectory_index] - 1)
        subkey = key.removeprefix("video.")
        original_key = self.lerobot_modality_meta.video[subkey].original_key or subkey
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        frames = []
        for step_index in step_indices:
            payload = self.curr_traj_data[original_key].iloc[int(step_index)]
            if isinstance(payload, dict):
                image_bytes = payload.get("bytes")
                image_path = payload.get("path")
            else:
                image_bytes = getattr(payload, "bytes", None)
                image_path = getattr(payload, "path", None)
            if image_bytes:
                image_source = io.BytesIO(image_bytes)
            elif image_path:
                image_path = Path(image_path)
                image_source = image_path if image_path.is_absolute() else self.dataset_path / image_path
            else:
                raise ValueError(f"Image feature {original_key!r} has neither bytes nor path at step {step_index}")
            with Image.open(image_source) as image:
                frames.append(np.asarray(image.convert("RGB")))
        return np.stack(frames, axis=0)

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]
        if self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata.get(trajectory_id, {})
            from_timestamps = episode_meta.get("videos/from_timestamps", {})
            original_video_key = self.lerobot_modality_meta.video[key].original_key
            if original_video_key is None:
                original_video_key = key
            from_timestamp = float(from_timestamps.get(original_video_key, 0.0))
            video_timestamp = video_timestamp + from_timestamp

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,  # TODO
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got key {le_key} is{data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
            # padding_strategy="zero",           # HACK for realdata
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith("annotation."), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)):  #
            # task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
            value = self.curr_traj_data[original_key].iloc[step_indices[i]]  # TODO check v2.0
            if isinstance(value, str):
                task_indices.append(value)
            elif isinstance(value, (int, float)):
                task_indices.append(value)
            else:
                task_indices.append(value.item())

        # If task_indices are strings (direct language), return them directly
        if task_indices and isinstance(task_indices[0], str):
            return task_indices
        return self.tasks.loc[task_indices]["task"].tolist()

    def get_language_with_fallback(self, trajectory_id: int, base_index: int) -> list[str]:
        """Return the first non-empty configured language value for each requested step.

        With ``data_cfg.language_sampling`` (a ``{key: weight}`` dict) set,
        each call picks one preferred language key at random per the given
        weights, then falls back to the strict order if that key is empty.
        Keys can be short (``current_instruction``) or full
        (``annotation.human.action.current_instruction``).
        """
        language_keys = self.modality_keys.get("language", [])
        if not language_keys:
            return []

        candidates = [self.get_language(trajectory_id, key, base_index) for key in language_keys]
        if not candidates:
            return []

        # Some datasets store the literal string "null" (e.g. Geniesim3.0
        # current_instruction) where the column exists but the value is
        # missing — treat those as empty so the fallback chain advances.
        _INVALID = {"null", "none", "nan"}

        order = self._sample_language_key_order(language_keys)
        # Map key -> position in original candidates list.
        key_to_pos = {k: i for i, k in enumerate(language_keys)}

        max_len = max(len(values) for values in candidates)
        merged: list[str] = []
        for i in range(max_len):
            chosen = ""
            for key in order:
                values = candidates[key_to_pos[key]]
                if i >= len(values):
                    continue
                value = values[i]
                if value is None:
                    continue
                value = str(value).strip()
                if not value or value.lower() in _INVALID:
                    continue
                chosen = value
                break
            merged.append(chosen)
        return merged

    def _sample_language_key_order(self, language_keys: list[str]) -> list[str]:
        """Return the language-key fallback order for this sample.

        With ``data_cfg.language_sampling`` set, a preferred key is drawn
        per call by weight and moved to the front; remaining keys keep
        their original order as fallback. Without ``language_sampling``,
        returns ``language_keys`` unchanged (strict fallback).
        """
        sampling = self.data_cfg.get("language_sampling", None) if self.data_cfg is not None else None
        if not sampling:
            return list(language_keys)
        key_to_idx = {}
        for i, k in enumerate(language_keys):
            key_to_idx[k] = i
            key_to_idx[k.rsplit(".", 1)[-1]] = i
        names, weights = [], []
        for name, w in sampling.items():
            if key_to_idx.get(str(name)) is not None and float(w) > 0:
                names.append(key_to_idx[str(name)])
                weights.append(float(w))
        if not weights:
            return list(language_keys)
        preferred = random.choices(names, weights=weights, k=1)[0]
        if preferred == 0:
            return list(language_keys)
        return [language_keys[preferred]] + [k for i, k in enumerate(language_keys) if i != preferred]

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            if self._is_image_feature(key):
                return self.get_image(trajectory_id, key, base_index)
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def _save_dataset_statistics_(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the dataset.
        Gripper-related keys will be placed at the end.

        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the data structure to save
        statistics_data = {}

        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)

        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}

        # Process action statistics (only for used keys)
        if hasattr(self.metadata.statistics, "action") and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action

            # Filter to only include used action keys and reorder: non-gripper first, gripper last
            non_gripper_keys = []
            gripper_keys = []

            for key in action_stats.keys():
                if key in used_action_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_action_stats = {}
            for key in reordered_keys:
                filtered_action_stats[key] = action_stats[key]

            if filtered_action_stats:
                # Combine statistics from filtered action sub-keys
                combined_action_stats = combine_modality_stats(filtered_action_stats)

                # Add mask field based on whether it's gripper or not
                mask = generate_action_mask_for_used_keys(self.metadata.modalities.action, filtered_action_stats.keys())
                combined_action_stats["mask"] = mask

                tag_stats["action"] = combined_action_stats

        # Process state statistics (only for used keys)
        if hasattr(self.metadata.statistics, "state") and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state

            # Filter to only include used state keys, optionally reorder gripper to end
            non_gripper_keys = []
            gripper_keys = []

            for key in state_stats.keys():
                if key in used_state_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_state_stats = {}
            for key in reordered_keys:
                filtered_state_stats[key] = state_stats[key]

            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats

        # Add dataset counts
        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)

        statistics_data[tag] = tag_stats

        # Save as JSON file
        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(used_action_keys)}")
        print(f"Used state keys (reordered): {list(used_state_keys)}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            original_key = key
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"

                # Apply image cropping if enabled and the video key is base_view
                # Note: crop_obs_camera functionality has been removed

                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics


def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.

    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.

    Returns:
        dict: Combined statistics
    """
    combined_stats = {"mean": [], "std": [], "max": [], "min": [], "q01": [], "q99": []}

    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object

        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:
                # Handle NDArray case - convert to list
                if hasattr(stat_value, "tolist"):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))

    return combined_stats


def generate_action_mask_for_used_keys(action_modalities: dict, used_action_keys_ordered) -> list[bool]:
    """
    Generate mask based on action modalities, but only for used keys.
    Gripper-related are False, others are True.

    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys in the correct order.

    Returns:
        list[bool]: List of mask values
    """
    mask = []

    # Generate mask in the same order as the statistics were combined
    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]

            # Get dimension count from shape
            if hasattr(subkey_config, "shape") and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1

            # Check if it's gripper-related
            is_gripper = "gripper" in subkey.lower()

            # Generate mask value for each dimension
            for _ in range(dim_count):
                mask.append(not is_gripper)  # gripper is False, others are True

    return mask


def get_used_modality_keys(modality_keys: dict) -> tuple[list, list]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys = []
    used_state_keys = []

    # Extract action keys (remove "action." prefix)
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.append(clean_key)

    # Extract state keys (remove "state." prefix)
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.append(clean_key)

    return used_action_keys, used_state_keys


class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
        **kwargs,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            # Check if dataset is valid and has data
            if len(dataset) == 0:
                print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)

        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")

        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode
        self.data_cfg = kwargs["data_cfg"] if "data_cfg" in kwargs else None

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights, dtype=np.float64)

        # `dataset_weight_scheme` (if set in data_cfg) overrides the older
        # `balance_dataset_weights` bool. Options:
        #   "frames"       — weight ∝ dataset frame count (== balance_dataset_weights=True)
        #   "leaves"       — every leaf equal (== balance_dataset_weights=False)
        #   "sqrt_frames"  — weight ∝ sqrt(frames), a temperature α=0.5
        #                    trade-off between frame-fair and leaf-uniform.
        #                    Useful when a few huge leaves (e.g. buy/kuavo_4pro
        #                    at 16% of frames) dominate a "frames"-weighted
        #                    mixture but "leaves" would over-sample tiny sets.
        scheme = None
        if self.data_cfg is not None:
            scheme = self.data_cfg.get("dataset_weight_scheme", None)
        if scheme is not None:
            scheme = str(scheme).lower()
            if scheme not in ("frames", "leaves", "sqrt_frames"):
                raise ValueError(
                    f"Unknown dataset_weight_scheme={scheme!r}; "
                    "expected one of: frames | leaves | sqrt_frames"
                )
            if scheme == "frames":
                self._dataset_sampling_weights *= self._dataset_lengths
            elif scheme == "sqrt_frames":
                self._dataset_sampling_weights *= np.sqrt(self._dataset_lengths.astype(np.float64))
            # "leaves": leave weights as-is (initial per-entry weight, 1.0 by default)
            print(f"[LeRobotMixtureDataset] dataset_weight_scheme={scheme}")
        elif self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths

        # Check for zero or negative weights before normalization
        if np.any(self._dataset_sampling_weights <= 0):
            print(f"Warning: Found zero or negative sampling weights: {self._dataset_sampling_weights}")
            # Set minimum weight to prevent division issues
            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)

        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print("Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths

            # Check for zero or negative weights before normalization
            if np.any(trajectory_sampling_weights <= 0):
                print(f"Warning: Dataset {i} has zero or negative trajectory weights")
                trajectory_sampling_weights = np.maximum(trajectory_sampling_weights, 1e-8)

            # Normalize weights
            weights_sum = trajectory_sampling_weights.sum()
            if weights_sum == 0 or np.isnan(weights_sum):
                print(f"Error: Dataset {i} has invalid trajectory weights sum: {weights_sum}")
                # Fallback to equal weights
                trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths)) / len(dataset.trajectory_lengths)
            else:
                trajectory_sampling_weights /= weights_sum

            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            print(f"Warning: No dataset with weight 1.0 found. Original weights: {dataset_sampling_weights}")
            # Fallback: use the dataset(s) with maximum weight as primary
            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight
            print(f"Using datasets with maximum weight {max_weight} as primary: {self._primary_dataset_indices}")

        if not np.any(self._primary_dataset_indices):
            # This should never happen, but just in case
            print("Error: Still no primary dataset found. Using first dataset as primary.")
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self._sequential_step_sampling = False
        if self.data_cfg is not None:
            seq_cfg = self.data_cfg.get("sequential_step_sampling", True)
            self._sequential_step_sampling = seq_cfg not in ["False", False]

        self._step_order: list[np.ndarray] = []
        self._step_pos: list[int] = []
        self._step_cycle: list[int] = []
        if self._sequential_step_sampling:
            if len(self.datasets) != 1:
                raise ValueError(
                    "sequential_step_sampling=True is only supported for a single dataset. "
                    "Use sequential_step_sampling=False for multi-dataset mixtures."
                )
            for dataset in self.datasets:
                self._step_order.append(np.arange(len(dataset.all_steps)))
                if self.mode == "train":
                    rng = np.random.default_rng(self.seed)
                    rng.shuffle(self._step_order[-1])
                self._step_pos.append(0)
                self._step_cycle.append(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def _init_worker_shard(self) -> None:
        """Restrict each DataLoader worker to its own shard of _step_order.

        Without this, every worker forks a private copy of _step_pos/_step_order
        and walks the same order from 0, so each step is sampled num_workers times.
        Called lazily on first sample_step() in the worker process.
        """
        if getattr(self, "_worker_shard_ready", False):
            return
        try:
            import torch.utils.data as torch_data
            worker_info = torch_data.get_worker_info()
        except Exception:
            worker_info = None
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        if dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
        else:
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers
        for i, dataset in enumerate(self.datasets):
            order = self._step_order[i]
            if len(order) < global_num_workers:
                raise ValueError(
                    f"sequential_step_sampling=True requires at least one step per global worker for "
                    f"{dataset.dataset_name}: {len(order)} steps < {global_num_workers} global workers."
                )
            self._step_order[i] = order[global_worker_id::global_num_workers]
            self._step_pos[i] = 0
        self._worker_shard_ready = True
        self._global_worker_id = global_worker_id
        self._global_num_workers = global_num_workers

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        # return self.sampled_steps[index]

        # Set seed
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        # trajectory_index = rng.choice(
        #     len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        # )
        # trajectory_id = dataset.trajectory_ids[trajectory_index]

        # # Sample step
        # base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        # return dataset, trajectory_id, base_index
        if len(dataset) == 0:
            raise ValueError(f"Dataset {dataset.dataset_name} has no steps.")

        if not self._sequential_step_sampling:
            single_step_index = rng.choice(len(dataset))
        else:
            self._init_worker_shard()
            step_pos = self._step_pos[dataset_index]
            if step_pos >= len(self._step_order[dataset_index]):
                # Use a per-dataset cycle counter (not step_pos) as the reshuffle
                # seed, so all workers compute the same global permutation each
                # cycle even when shard lengths differ (len % num_workers != 0).
                self._step_cycle[dataset_index] += 1
                global_worker_id = getattr(self, "_global_worker_id", 0)
                global_num_workers = getattr(self, "_global_num_workers", 1)
                order = np.arange(len(dataset.all_steps))
                if self.mode == "train":
                    shuffle_seed = safe_hash(
                        (self.epoch, dataset_index, self.seed, self._step_cycle[dataset_index])
                    )
                    np.random.default_rng(shuffle_seed).shuffle(order)
                order = order[global_worker_id::global_num_workers]
                self._step_order[dataset_index] = order
                step_pos = 0

            single_step_index = self._step_order[dataset_index][step_pos]
            self._step_pos[dataset_index] = step_pos + 1
        if getattr(dataset, "_lazy_step_sampling", False):
            trajectory_id, base_index = dataset.get_step_from_flat_index(single_step_index)
        else:
            trajectory_id, base_index = dataset.all_steps[single_step_index]
        return dataset, trajectory_id, base_index

    _getitem_count = 0

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        LeRobotMixtureDataset._getitem_count += 1
        if LeRobotMixtureDataset._getitem_count % 1000 == 0:
            gc.collect()

        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                while True:  # @DUG
                    dataset, trajectory_id, step = self.sample_step(index)
                    video_key = dataset.modality_keys["video"][0]
                    if dataset._is_image_feature(video_key):
                        break
                    video_path = dataset.get_video_path(trajectory_id, video_key.replace("video.", ""))
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)

                raw_data = dataset.get_step_data(trajectory_id, step)
                data = dataset.transforms(raw_data)
                source_indices = dataset._get_source_indices(trajectory_id, step)
                if dataset._advantage_lookup is not None:
                    source_indices["advantage"] = dataset._get_advantage(trajectory_id, step)
                if dataset._gripper_mask_mode in {"loss_mask", "base_sample_and_loss_mask"}:
                    source_indices["action_loss_mask"] = dataset._get_action_loss_mask(trajectory_id, step)
                sample = dataset._pack_sample(data, source_indices=source_indices)
                sample["robot_tag"] = dataset.tag
                return sample

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Log the error but continue trying
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    print("Retrying with new sample...")
                    # For retry, we can use a slightly different index to get a new sample
                    # This helps avoid getting stuck on the same problematic sample
                    index = random.randint(0, len(self) - 1)
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    # Return a dummy sample or re-raise the exception
                    raise last_exception

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        # Check for potential issues
        if len(self.datasets) == 0:
            return 0

        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)

        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0

        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(
                f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}"
            )
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0

        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]

        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()

        result = int(max_ratio)
        if result == 0:
            print("Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        if len(per_task_stats) != len(dataset_sampling_weights):
            raise ValueError(
                f"Statistics/weight length mismatch: {len(per_task_stats)} stats vs "
                f"{len(dataset_sampling_weights)} weights."
            )
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Number of dimensions (assuming consistent across tasks)
            num_dims = len(per_task_stats[0][modality]["mean"])

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            if modality == "video" and len(configs) > 1:
                merged_metadata["modalities"][modality] = json.loads(sorted(configs)[0])
                continue
            assert len(configs) == 1, f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                print(f"Failed to load cached statistics: {e}")
                print("Falling back to computing statistics from scratch...")

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        all_weights: dict[str, list[float]] = {}
        for dataset, dataset_weight in zip(self.datasets, self.dataset_sampling_weights):
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
                all_weights[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
            all_weights[dataset.tag].append(float(dataset_weight))
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=all_weights[tag],
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the datasets.
        Gripper-related keys will be placed at the end.

        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the data structure to save
        statistics_data = {}

        # Collect actually used keys from all datasets
        all_used_action_keys = []
        all_used_state_keys = []

        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)

        # Organize statistics by tag
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}

            # Process action statistics
            if hasattr(merged_metadata.statistics, "action") and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action

                # Filter and reorder keys - iterate in all_used_action_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_action_keys:
                    if key in action_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_action_stats = {}
                for key in reordered_keys:
                    filtered_action_stats[key] = action_stats[key]

                if filtered_action_stats:
                    combined_action_stats = combine_modality_stats(filtered_action_stats)

                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action, filtered_action_stats.keys()
                    )
                    combined_action_stats["mask"] = mask

                    tag_stats["action"] = combined_action_stats

            # Process state statistics
            if hasattr(merged_metadata.statistics, "state") and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state

                # Filter and reorder keys - iterate in all_used_state_keys order
                # Filter and reorder keys - iterate in all_used_state_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_state_keys:
                    if key in state_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]

                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats

            # Add dataset counts
            tag_stats.update(self._get_dataset_counts(tag))

            statistics_data[tag] = tag_stats

        # Save file
        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(all_used_action_keys)}")
        print(f"Used state keys (reordered): {list(all_used_state_keys)}")

    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified tag.

        Args:
            tag (str): embodiment tag

        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0

        # Count dataset information belonging to this tag
        for dataset in self.datasets:
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)

        return {"num_transitions": num_transitions, "num_trajectories": num_trajectories}

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.

        Args:
            load_path (Path | str): Path to the statistics file

        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")

        if load_path.suffix.lower() == ".json":
            with open(load_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif load_path.suffix.lower() == ".pkl":
            import pickle

            with open(load_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.

        Args:
            cached_statistics (dict): Statistics loaded from file
        """
        # Validate that cached statistics match current datasets
        if "metadata" in cached_statistics:
            cached_dataset_names = set(cached_statistics["metadata"]["dataset_names"])
            current_dataset_names = set(dataset.dataset_name for dataset in self.datasets)

            if cached_dataset_names != current_dataset_names:
                print("Warning: Cached statistics dataset names don't match current datasets.")
                print(f"Cached: {cached_dataset_names}")
                print(f"Current: {current_dataset_names}")
                return

        # Apply cached statistics
        self.merged_metadata = {}
        for tag, stats_data in cached_statistics.items():
            if tag == "metadata":  # Skip metadata field
                continue

            # Convert back to DatasetMetadata format
            metadata_dict = {"embodiment_tag": tag, "statistics": {"action": {}, "state": {}}, "modalities": {}}

            # Convert action statistics back
            if "action" in stats_data:
                action_data = stats_data["action"]
                # This is simplified - you may need to split back to sub-keys
                metadata_dict["statistics"]["action"] = action_data

            # Convert state statistics back
            if "state" in stats_data:
                state_data = stats_data["state"]
                metadata_dict["statistics"]["state"] = state_data

            self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)

        # Update transforms metadata for each dataset
        for dataset in self.datasets:
            if dataset.tag in self.merged_metadata:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

        print(f"Applied cached statistics for {len(self.merged_metadata)} embodiment tags.")
