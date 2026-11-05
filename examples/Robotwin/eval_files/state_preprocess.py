from __future__ import annotations

from typing import Mapping

import numpy as np


ROBOTWIN_STATE_DIM = 14

# RoboTwin env joint_action vector:
#   [left_joints(6), left_gripper, right_joints(6), right_gripper]
# Training concat order from AgilexData50Config.state_keys:
#   [left_joints(6), right_joints(6), left_gripper, right_gripper]
ROBOTWIN_ENV_TO_TRAIN_STATE_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13],
    dtype=np.int64,
)

# Model action output is in the same training concat order; RoboTwin expects env order.
ROBOTWIN_TRAIN_TO_ENV_ACTION_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13],
    dtype=np.int64,
)

ROBOTWIN_TRAIN_STATE_CONTINUOUS_MASK = np.array([True] * 12 + [False, False], dtype=bool)
ROBOTWIN_GRIPPER_BINARY_THRESHOLD = 0.49


def resolve_robotwin_use_state(use_state: bool | str | None, model_config: Mapping[str, object]) -> bool:
    """
    Resolve eval-time state usage.

    `auto` is intentionally conservative: it only enables state when the
    checkpoint config says the training dataloader included state.
    """
    if isinstance(use_state, str):
        normalized = use_state.strip().lower()
        if normalized in {"auto", "none", "null"}:
            use_state = None
        elif normalized in {"true", "1", "yes", "y"}:
            return True
        elif normalized in {"false", "0", "no", "n"}:
            return False
        else:
            raise ValueError(f"Invalid use_state value: {use_state}. Expected true, false, or auto.")

    if use_state is not None:
        return bool(use_state)

    datasets_cfg = model_config.get("datasets", {})
    if not isinstance(datasets_cfg, Mapping):
        return False
    vla_data_cfg = datasets_cfg.get("vla_data", {})
    if not isinstance(vla_data_cfg, Mapping):
        return False

    include_state = vla_data_cfg.get("include_state", False)
    if isinstance(include_state, str):
        return include_state.strip().lower() in {"true", "1", "yes", "y"}
    return bool(include_state)


def _as_robotwin_vector(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1:] != (ROBOTWIN_STATE_DIM,):
        raise ValueError(
            f"{name} must have last dimension {ROBOTWIN_STATE_DIM}, got shape {array.shape}."
        )
    return array


def reorder_robotwin_env_state_to_train(state: np.ndarray) -> np.ndarray:
    """Reorder raw RoboTwin env state into the concat order used by training."""
    state = _as_robotwin_vector(state, name="state")
    return np.take(state, ROBOTWIN_ENV_TO_TRAIN_STATE_INDICES, axis=-1)


def reorder_robotwin_train_action_to_env(action: np.ndarray) -> np.ndarray:
    """Reorder model/train-order action into RoboTwin env joint_action order."""
    action = np.asarray(action)
    if action.shape[-1:] != (ROBOTWIN_STATE_DIM,):
        raise ValueError(
            f"action must have last dimension {ROBOTWIN_STATE_DIM}, got shape {action.shape}."
        )
    return np.take(action, ROBOTWIN_TRAIN_TO_ENV_ACTION_INDICES, axis=-1)


def _get_state_normalization_bounds(
    state_norm_stats: Mapping[str, object],
    *,
    normalization_mode: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if normalization_mode == "min_max":
        high_key, low_key, clip = "max", "min", False
    elif normalization_mode == "q99":
        high_key, low_key, clip = "q99", "q01", True
    else:
        raise ValueError(
            f"Unsupported normalization_mode: {normalization_mode}. Expected one of ['min_max', 'q99']."
        )

    missing_keys = [key for key in (high_key, low_key) if key not in state_norm_stats]
    if missing_keys:
        raise KeyError(
            f"State stats for normalization_mode='{normalization_mode}' are missing keys: {missing_keys}."
        )

    high = _as_robotwin_vector(state_norm_stats[high_key], name=f"state_norm_stats['{high_key}']")
    low = _as_robotwin_vector(state_norm_stats[low_key], name=f"state_norm_stats['{low_key}']")
    return high, low, clip


def normalize_robotwin_train_state(
    train_order_state: np.ndarray,
    state_norm_stats: Mapping[str, object],
    *,
    normalization_mode: str = "min_max",
    binary_threshold: float = ROBOTWIN_GRIPPER_BINARY_THRESHOLD,
) -> np.ndarray:
    """
    Normalize state after it has already been reordered into training concat order.

    This mirrors StateActionTransform for AgilexData50Config:
    - first 12 joint dims use min_max/q99 continuous normalization,
    - final 2 gripper dims use binary thresholding,
    - min == max continuous dims become 0 for min_max, matching training;
      q01 == q99 dims keep the raw value and are then clipped, matching training.
    """
    state = _as_robotwin_vector(train_order_state, name="train_order_state")
    high, low, clip_continuous = _get_state_normalization_bounds(
        state_norm_stats,
        normalization_mode=normalization_mode,
    )

    continuous_mask = ROBOTWIN_TRAIN_STATE_CONTINUOUS_MASK
    valid_mask = continuous_mask & (high != low)
    normalized = np.zeros_like(state, dtype=np.float32)

    normalized[..., valid_mask] = (
        2.0
        * (state[..., valid_mask] - low[valid_mask])
        / (high[valid_mask] - low[valid_mask])
        - 1.0
    )
    if clip_continuous:
        invalid_continuous_mask = continuous_mask & ~valid_mask
        normalized[..., invalid_continuous_mask] = state[..., invalid_continuous_mask]

    normalized[..., ~continuous_mask] = (
        state[..., ~continuous_mask] > binary_threshold
    ).astype(np.float32)

    if clip_continuous:
        normalized[..., continuous_mask] = np.clip(normalized[..., continuous_mask], -1.0, 1.0)

    if not np.isfinite(normalized).all():
        raise ValueError("Normalized RobotWin state contains NaN or Inf.")
    return normalized


def preprocess_robotwin_state_for_model(
    raw_env_state: np.ndarray,
    state_norm_stats: Mapping[str, object],
    *,
    normalization_mode: str = "min_max",
    binary_threshold: float = ROBOTWIN_GRIPPER_BINARY_THRESHOLD,
) -> np.ndarray:
    """
    Convert a raw RoboTwin env state vector into the `(1, 14)` normalized state
    expected by state-conditioned MR0 RobotWin checkpoints.
    """
    train_order_state = reorder_robotwin_env_state_to_train(raw_env_state)
    normalized_state = normalize_robotwin_train_state(
        train_order_state,
        state_norm_stats,
        normalization_mode=normalization_mode,
        binary_threshold=binary_threshold,
    )

    if normalized_state.size != ROBOTWIN_STATE_DIM:
        raise ValueError(
            "preprocess_robotwin_state_for_model expects one RobotWin state vector, "
            f"got shape {normalized_state.shape}."
        )
    return normalized_state.reshape(1, ROBOTWIN_STATE_DIM)
