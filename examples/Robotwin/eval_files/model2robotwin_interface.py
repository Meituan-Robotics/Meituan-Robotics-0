from __future__ import annotations
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from mr0.model.tools import read_mode_config


# ─── Inference mode controls (read from env vars at __init__) ───────────────
# All default-safe: when env vars are unset, behavior matches the original
# "predict every chunk_size steps, use raw_actions[step%chunk_size]" baseline.
#
#   ROBOTWIN_INFERENCE_MODE   chunk (default) | per_step | half_chunk
#   ROBOTWIN_REPLAN_EVERY     int; if set, overrides the mode default
#   ROBOTWIN_ENSEMBLE         none (default) | cos | mean | decay
#   ROBOTWIN_ENSEMBLE_ALPHA   float, default 0.1 (cos)
#   ROBOTWIN_ENSEMBLE_DECAY   float, default 0.5 (decay; weight = decay^age)
#   ROBOTWIN_ENSEMBLE_HORIZON int; cap on history length, default = chunk_size
#   ROBOTWIN_NUM_DDIM_STEPS   int; overrides num_ddim_steps (Euler steps for
#                             flow matching denoising). Default = ckpt's value
#                             (typically 10 from deploy_policy.yml).


def _env(name: str, default=None, cast=str):
    v = os.environ.get(name, None)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except Exception:
        return default

try:
    from examples.Robotwin.eval_files.state_preprocess import (
        ROBOTWIN_STATE_DIM,
        normalize_robotwin_train_state,
        preprocess_robotwin_state_for_model,
        reorder_robotwin_env_state_to_train,
        reorder_robotwin_train_action_to_env,
        resolve_robotwin_use_state,
    )
except ImportError:
    from state_preprocess import (
        ROBOTWIN_STATE_DIM,
        normalize_robotwin_train_state,
        preprocess_robotwin_state_for_model,
        reorder_robotwin_env_state_to_train,
        reorder_robotwin_train_action_to_env,
        resolve_robotwin_use_state,
    )

from examples.Robotwin.eval_files.adaptive_ensemble import AdaptiveEnsembler


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
        use_state: bool | str | None = "auto",
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.model_config, _ = read_mode_config(policy_ckpt_path)
        self.use_state = self.resolve_use_state(use_state, self.model_config)

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode}, "
            f"use_state: {self.use_state} ***"
        )
        self.use_ddim = use_ddim
        # Allow env-var override (W1: vary inference timesteps without retraining)
        self.num_ddim_steps = _env("ROBOTWIN_NUM_DDIM_STEPS", num_ddim_steps, int)
        if self.num_ddim_steps != num_ddim_steps:
            print(f"[INFER] ROBOTWIN_NUM_DDIM_STEPS override: {num_ddim_steps} -> {self.num_ddim_steps}")
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(
            self.unnorm_key, policy_ckpt_path=policy_ckpt_path, action_mode=action_mode
        )
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        self.state_norm_stats = (
            self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
            if self.use_state
            else None
        )
        self.raw_actions = None

        # ─── Inference mode wiring (env vars override defaults) ───────────────
        mode = _env("ROBOTWIN_INFERENCE_MODE", "chunk", str)
        if mode not in ("chunk", "per_step", "half_chunk"):
            raise ValueError(f"Unknown ROBOTWIN_INFERENCE_MODE={mode!r}")
        self.inference_mode = mode

        default_replan = {
            "chunk": self.action_chunk_size,
            "per_step": 1,
            "half_chunk": max(1, self.action_chunk_size // 2),
        }[mode]
        self.replan_every = _env("ROBOTWIN_REPLAN_EVERY", default_replan, int)
        if self.replan_every < 1:
            raise ValueError(f"ROBOTWIN_REPLAN_EVERY must be >=1, got {self.replan_every}")

        ens_kind = _env("ROBOTWIN_ENSEMBLE", "none", str)
        if ens_kind not in ("none", "cos", "mean", "decay"):
            raise ValueError(f"Unknown ROBOTWIN_ENSEMBLE={ens_kind!r}")
        self.ensemble_kind = ens_kind
        self.ensemble_alpha = _env("ROBOTWIN_ENSEMBLE_ALPHA", 0.1, float)
        self.ensemble_decay_gamma = _env("ROBOTWIN_ENSEMBLE_DECAY", 0.5, float)
        self.ensemble_horizon = _env("ROBOTWIN_ENSEMBLE_HORIZON", self.action_chunk_size, int)

        # History of recent prediction chunks for ensembling
        # each entry: {"start_step": int, "actions": np.ndarray (T, D)}
        self._chunk_history: List[dict] = []
        self._last_chunk_start: int = -1

        print(
            f"[INFER] mode={self.inference_mode} replan_every={self.replan_every} "
            f"ensemble={self.ensemble_kind} alpha={self.ensemble_alpha} "
            f"decay={self.ensemble_decay_gamma} horizon={self.ensemble_horizon} "
            f"chunk_size={self.action_chunk_size}"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        # Reset state tracking for delta/rel modes
        self.initial_state = None
        self.prev_action = None
        # Reset chunk history for ensembling
        self._chunk_history = []
        self._last_chunk_start = -1

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        raw_env_state = example.get("state", None)
        train_order_state = (
            reorder_robotwin_env_state_to_train(raw_env_state)
            if raw_env_state is not None
            else None
        )
        if train_order_state is not None:
            if train_order_state.size != ROBOTWIN_STATE_DIM:
                raise ValueError(
                    f"Expected one RobotWin state vector for eval, got shape {train_order_state.shape}"
                )
            train_order_state = train_order_state.reshape(ROBOTWIN_STATE_DIM)

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if train_order_state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(train_order_state).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                # Re-store initial state after reset if in delta/rel mode
                if self.action_mode in ["delta", "rel"] and train_order_state is not None:
                    self.initial_state = np.array(train_order_state).copy()

        images = [self._resize_image(image) for image in images]
        example_copy = example.copy()
        example_copy["image"] = images

        if self.use_state:
            if raw_env_state is None:
                raise ValueError("use_state=True requires `state` to be provided in example")
            if self.state_norm_stats is None:
                raise ValueError("use_state=True requires state normalization stats in dataset_statistics.json")
            example_copy["state"] = preprocess_robotwin_state_for_model(
                raw_env_state,
                self.state_norm_stats,
                normalization_mode=self.normalization_mode,
            )
        else:
            example_copy.pop("state", None)

        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        # ── Decide whether to call the policy server this step ──
        # Replan when the most recent chunk no longer covers `step`,
        # or every replan_every steps relative to its start.
        need_replan = self.raw_actions is None or (
            (step - self._last_chunk_start) % self.replan_every == 0
            if self._chunk_history else True
        )

        if need_replan:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            normalized_actions = normalized_actions[0]
            # Unnormalize to get delta/rel values
            raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
                normalization_mode=self.normalization_mode,
            )

            # Convert delta/rel to absolute actions
            if self.action_mode == "delta":
                raw_actions_abs = self._delta_to_absolute(raw_actions, train_order_state)
            elif self.action_mode == "rel":
                raw_actions_abs = self._rel_to_absolute(raw_actions)
            else:
                raw_actions_abs = raw_actions

            self.raw_actions = raw_actions_abs  # kept for backward compat / debug
            self._chunk_history.append({"start_step": step, "actions": raw_actions_abs})
            self._last_chunk_start = step

            # Drop chunks that no longer cover the current step (their tail ran out)
            self._chunk_history = [
                h for h in self._chunk_history
                if h["start_step"] + len(h["actions"]) > step
            ]
            # Cap history length to ensemble_horizon (keep the most recent)
            if len(self._chunk_history) > self.ensemble_horizon:
                self._chunk_history = self._chunk_history[-self.ensemble_horizon:]

        # ── Select / fuse the action for the current env step ──
        current_action = self._select_action_for_step(step)

        # Update prev_action for delta mode (for cross-chunk continuity)
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        current_action = reorder_robotwin_train_action_to_env(current_action)
        return current_action

    def _select_action_for_step(self, step: int) -> np.ndarray:
        """
        Compute the action to execute at env step `step`, using the chunk
        history. Each chunk h covers env steps [h.start_step, h.start_step+T).
        For each h that covers `step`, the relevant prediction is
        h.actions[step - h.start_step]. We collect those candidates and fuse
        them according to self.ensemble_kind.
        """
        candidates = []
        for h in self._chunk_history:
            idx = step - h["start_step"]
            if 0 <= idx < len(h["actions"]):
                candidates.append(h["actions"][idx])

        if not candidates:
            # Should not happen: step's covering chunk was just appended.
            raise RuntimeError(
                f"No candidate actions for step={step}. "
                f"history={[(h['start_step'], len(h['actions'])) for h in self._chunk_history]}"
            )

        # Single candidate (e.g. baseline chunk mode): just return it.
        if len(candidates) == 1 or self.ensemble_kind == "none":
            # When ensemble=none, follow the most recently emitted chunk
            # (i.e., the last appended history item). This matches both the
            # original baseline ("predict every chunk_size steps and walk
            # through the chunk") and per_step_fresh ("always use freshest
            # chunk's [0]").
            h = self._chunk_history[-1]
            idx = step - h["start_step"]
            return h["actions"][idx]

        preds = np.stack(candidates, axis=0)  # (N, D)

        if self.ensemble_kind == "mean":
            return preds.mean(axis=0)

        if self.ensemble_kind == "decay":
            # Newest candidate = last in `candidates` (chunk_history is ordered
            # oldest→newest; older chunks → smaller idx in `preds`).
            n = preds.shape[0]
            ages = np.arange(n - 1, -1, -1)  # newest=0, oldest=n-1
            w = np.power(self.ensemble_decay_gamma, ages)
            w = w / w.sum()
            return (w[:, None] * preds).sum(axis=0)

        if self.ensemble_kind == "cos":
            ref = preds[-1]
            num = (preds * ref).sum(axis=1)
            denom = np.linalg.norm(preds, axis=1) * np.linalg.norm(ref) + 1e-7
            cos_sim = num / denom
            w = np.exp(self.ensemble_alpha * cos_sim)
            w = w / w.sum()
            return (w[:, None] * preds).sum(axis=0)

        raise ValueError(f"Unknown ensemble_kind={self.ensemble_kind!r}")

    @staticmethod
    def normalize_state(
        state: np.ndarray,
        state_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> np.ndarray:
        """
        Normalize a RobotWin state that is already in training concat order.
        """
        return normalize_robotwin_train_state(
            state,
            state_norm_stats,
            normalization_mode=normalization_mode,
        )

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> np.ndarray:
        action_high, action_low = ModelClient._get_normalization_bounds(
            action_norm_stats, normalization_mode=normalization_mode
        )
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
        normalized_actions = np.clip(normalized_actions, -1, 1)

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """
        Convert delta actions to absolute actions.

        Training: delta[0] = a[0] - s[0], delta[t] = a[t] - a[t-1]
        Deployment: a[0] = delta[0] + base, a[t] = delta[t] + a[t-1]

        Where base is:
        - First chunk: initial_state (s_0)
        - Subsequent chunks: prev_action (last action from previous chunk)
        """
        if current_state is None:
            raise ValueError("delta action_mode requires a current state in training concat order")

        abs_actions = np.zeros_like(delta_actions)
        mask = self.action_norm_stats.get("mask", np.ones(delta_actions.shape[-1], dtype=bool))

        # Determine base action
        base = self.prev_action if self.prev_action is not None else self.initial_state

        for i in range(len(delta_actions)):
            abs_actions[i] = np.where(mask, delta_actions[i] + base, delta_actions[i])
            base = abs_actions[i]

        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """
        Convert relative actions to absolute actions.

        Training: rel[t] = a[t] - s[0]
        Deployment: a[t] = rel[t] + s[0]
        """
        if self.initial_state is None:
            raise ValueError("rel action_mode requires an initial state in training concat order")

        abs_actions = np.zeros_like(rel_actions)
        mask = self.action_norm_stats.get("mask", np.ones(rel_actions.shape[-1], dtype=bool))

        for i in range(len(rel_actions)):
            abs_actions[i] = np.where(mask, rel_actions[i] + self.initial_state, rel_actions[i])

        return abs_actions

    @staticmethod
    def resolve_use_state(use_state: bool | str | None, model_config: dict) -> bool:
        return resolve_robotwin_use_state(use_state, model_config)

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path, action_mode: str = "abs") -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)

        stats = norm_stats[unnorm_key]

        # Support two formats:
        # New format: {"robotwin": {"abs": {...}, "delta": {...}, "rel": {...}}}
        # Old format: {"robotwin": {"action": {...}, "state": {...}}}

        if action_mode in stats:
            # New format: directly use the corresponding mode stats
            mode_stats = stats[action_mode]
            return mode_stats.get("action", mode_stats)
        if "action" in stats:
            # Old format: only supports abs mode
            if action_mode != "abs":
                raise ValueError(
                    f"Statistics key `{unnorm_key}` only provides `abs` action stats, "
                    f"but action_mode=`{action_mode}` was requested."
                )
            return stats["action"]
        raise ValueError(
            f"Invalid statistics file format for key `{unnorm_key}`. "
            f"Available top-level keys: {sorted(stats.keys())}"
        )

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        available_keys = sorted(norm_stats.keys())
        if unnorm_key is None:
            if len(available_keys) == 1:
                return available_keys[0]
            raise ValueError(
                "`unnorm_key` must be provided when multiple normalization statistics are available. "
                f"Available keys: {available_keys}"
            )

        if unnorm_key not in norm_stats:
            raise KeyError(
                f"Unknown `unnorm_key`: `{unnorm_key}`. Available keys: {available_keys}"
            )

        return unnorm_key

    @staticmethod
    def _get_normalization_bounds(
        norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> tuple[np.ndarray, np.ndarray]:
        if normalization_mode == "q99":
            if "q01" not in norm_stats or "q99" not in norm_stats:
                raise KeyError(
                    "Normalization mode `q99` requires statistics keys `q01` and `q99`."
                )
            return np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
        if normalization_mode == "min_max":
            if "min" not in norm_stats or "max" not in norm_stats:
                raise KeyError(
                    "Normalization mode `min_max` requires statistics keys `min` and `max`."
                )
            return np.array(norm_stats["max"]), np.array(norm_stats["min"])
        raise ValueError(
            f"Unsupported normalization_mode: {normalization_mode}. Expected one of ['min_max', 'q99']."
        )


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )
    use_state = usr_args.get("use_state", "auto")

    image_size = usr_args.get("image_size", [224, 224])

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
        image_size=image_size,
        use_state=use_state,
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    if not hasattr(eval, "_logged_frame_size"):
        print(f"[EVAL CLIENT] Raw frame sizes from env: head={head_img.shape}, left={left_img.shape}, right={right_img.shape}")
        print(f"[EVAL CLIENT] Client will resize to: {model.image_size}")
        eval._logged_frame_size = True

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action)
