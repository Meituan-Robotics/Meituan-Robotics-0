# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Fast Action Tokenizer Adapter
"this file is adapted from https://huggingface.co/physical-intelligence/fast"

Overview:
    This module encapsulates a lightweight "action → language model-readable sequence" converter (Fast_Action_Tokenizer).
    Its core objective is to convert continuous/discrete raw robot actions (raw_actions) into
    pseudo-natural language token strings like <robot_action_12><robot_action_3><robot_action_87> ...
    This facilitates direct integration into multimodal large models (VLM/LLM) dialogue templates,
    leveraging their language modeling capabilities for action prediction.
"""

import importlib.util
import json
import os
from collections import Counter

import numpy as np
import torch.nn as nn
from scipy.fft import idct
from transformers import AutoProcessor, PreTrainedTokenizerFast

_FAST_DECODE_WARN_COUNT = 0
_FAST_DECODE_WARN_LIMIT = 20

# Per-process diagnostics for token / coefficient length distribution.
# Keys: ("tok", n_tokens), ("coeff", n_coeffs), ("expected", time_horizon*action_dim)
# Always-on; printed every N calls so the eval log shows the distribution.
_FAST_DECODE_STATS = Counter()
_FAST_DECODE_CALLS = 0
_FAST_DECODE_STATS_EVERY = 200


def _as_action_array(action) -> np.ndarray:
    action_arr = np.asarray(action)
    if action_arr.ndim != 2:
        raise ValueError(f"FAST actions must be shaped [T, D], got {action_arr.shape}")
    return action_arr


def fit_fast_action_time(action: np.ndarray, target_time_horizon: int | None = None) -> np.ndarray:
    """Pad/truncate only the time axis for FAST tokenization."""
    action = _as_action_array(action)
    if target_time_horizon is None:
        return action

    source_time, action_dim = action.shape
    if source_time > target_time_horizon:
        return action[:target_time_horizon]
    if source_time == target_time_horizon:
        return action

    padded = np.zeros((target_time_horizon, action_dim), dtype=action.dtype)
    padded[:source_time] = action
    return padded


def encode_fast_actions_by_shape(fast_tokenizer, raw_actions, target_time_horizon: int | None = None):
    """Encode actions without forcing mixed action dimensions into one stack."""
    actions = [fit_fast_action_time(action, target_time_horizon) for action in raw_actions]
    grouped_indices: dict[tuple[int, int], list[int]] = {}
    for idx, action in enumerate(actions):
        grouped_indices.setdefault(tuple(action.shape), []).append(idx)

    batch_fast_tokens = [None] * len(actions)
    for indices in grouped_indices.values():
        batch_actions = np.stack([actions[idx] for idx in indices], axis=0)
        group_tokens = fast_tokenizer(batch_actions)
        for idx, tokens in zip(indices, group_tokens):
            batch_fast_tokens[idx] = [int(token) for token in tokens]

    return batch_fast_tokens, actions


def _safe_decode_one(fast_tokenizer, token_ids, time_horizon, action_dim):
    """Decode a single FAST token sequence with truncate-then-pad fallback.

    The upstream FAST decoder asserts the decoded DCT coefficients reshape
    exactly to (time_horizon, action_dim) and otherwise returns zeros. At
    inference, the model can over- or under-generate tokens, producing a
    coefficient count that is not a multiple of action_dim, or a time axis
    that does not match time_horizon. Both fall back to zeros today, which
    means the robot freezes (delta mode) or jumps to origin (abs mode).

    Here we recover what we can: truncate the coefficient stream to the
    nearest multiple of action_dim, then pad the time axis by repeating the
    last decoded frame (or zeros if we got nothing) up to time_horizon.
    Returns (decoded_actions[T, D], fallback_reason_or_None).
    """
    expected = time_horizon * action_dim
    _FAST_DECODE_STATS[("expected", expected)] += 1

    if token_ids is None:
        _FAST_DECODE_STATS[("tok", 0)] += 1
        _FAST_DECODE_STATS[("coeff", 0)] += 1
        return np.zeros((time_horizon, action_dim), dtype=np.float32), "no_tokens"

    n_tokens = len(token_ids)
    _FAST_DECODE_STATS[("tok", n_tokens)] += 1

    try:
        decoded_tokens = fast_tokenizer.bpe_tokenizer.decode(token_ids)
        coeff = np.array(list(map(ord, decoded_tokens))) + fast_tokenizer.min_token
    except Exception as e:
        _FAST_DECODE_STATS[("coeff", -1)] += 1
        return np.zeros((time_horizon, action_dim), dtype=np.float32), f"bpe_decode_error:{e}"

    n = coeff.shape[0]
    _FAST_DECODE_STATS[("coeff", n)] += 1

    reason = None
    n_kept = (n // action_dim) * action_dim
    if n_kept != n:
        reason = f"truncated_{n}_to_{n_kept}"
    if n_kept == 0:
        return np.zeros((time_horizon, action_dim), dtype=np.float32), reason or "empty_after_truncate"

    coeff = coeff[:n_kept].reshape(-1, action_dim)
    T = coeff.shape[0]
    if T < time_horizon:
        pad = np.repeat(coeff[-1:], time_horizon - T, axis=0)
        coeff = np.concatenate([coeff, pad], axis=0)
        reason = (reason + "|" if reason else "") + f"padded_{T}_to_{time_horizon}"
    elif T > time_horizon:
        coeff = coeff[:time_horizon]
        reason = (reason + "|" if reason else "") + f"timecut_{T}_to_{time_horizon}"

    actions = idct(coeff / fast_tokenizer.scale, axis=0, norm="ortho").astype(np.float32)
    return actions, reason


def decode_fast_actions_by_shape(
    fast_tokenizer,
    batch_fast_token_ids,
    time_horizons,
    action_dims,
):
    """Decode FAST token ids with per-sample time horizon and action dim.

    Uses truncate-then-pad recovery instead of the upstream all-zero fallback.
    """
    global _FAST_DECODE_WARN_COUNT, _FAST_DECODE_CALLS
    decoded_actions = []
    for token_ids, time_horizon, action_dim in zip(batch_fast_token_ids, time_horizons, action_dims):
        actions, reason = _safe_decode_one(
            fast_tokenizer, token_ids, int(time_horizon), int(action_dim)
        )
        if reason is not None and _FAST_DECODE_WARN_COUNT < _FAST_DECODE_WARN_LIMIT:
            print(f"[FAST decode] recovered with {reason} (time_horizon={time_horizon}, action_dim={action_dim})")
            _FAST_DECODE_WARN_COUNT += 1
            if _FAST_DECODE_WARN_COUNT == _FAST_DECODE_WARN_LIMIT:
                print(f"[FAST decode] further warnings suppressed (limit={_FAST_DECODE_WARN_LIMIT})")
        decoded_actions.append(actions)

    _FAST_DECODE_CALLS += len(batch_fast_token_ids)
    if _FAST_DECODE_CALLS // _FAST_DECODE_STATS_EVERY > (
        _FAST_DECODE_CALLS - len(batch_fast_token_ids)
    ) // _FAST_DECODE_STATS_EVERY:
        _print_decode_stats(prefix=f"calls={_FAST_DECODE_CALLS}")

    first_shape = decoded_actions[0].shape if decoded_actions else None
    if first_shape is not None and all(action.shape == first_shape for action in decoded_actions):
        return np.stack(decoded_actions, axis=0)
    return decoded_actions


def _print_decode_stats(prefix=""):
    tok = sorted(((k[1], v) for k, v in _FAST_DECODE_STATS.items() if k[0] == "tok"), key=lambda x: -x[1])
    coeff = sorted(((k[1], v) for k, v in _FAST_DECODE_STATS.items() if k[0] == "coeff"), key=lambda x: -x[1])
    expected = sorted(((k[1], v) for k, v in _FAST_DECODE_STATS.items() if k[0] == "expected"), key=lambda x: -x[1])
    top = lambda items, n=8: ", ".join(f"{k}:{v}" for k, v in items[:n])
    print(f"[FAST decode stats {prefix}] tokens(top): {top(tok)}")
    print(f"[FAST decode stats {prefix}] coeffs(top): {top(coeff)}")
    print(f"[FAST decode stats {prefix}] expected(top): {top(expected)}")


def _load_fast_processor(pretrained_path: str = "physical-intelligence/fast"):
    """Load the FAST UniversalActionProcessor with compatibility for transformers >= 5.x.

    transformers 5.x changed AutoProcessor internals which breaks the default
    loading path for the physical-intelligence/fast custom processor. This
    helper manually loads the custom class and its BPE tokenizer component.
    """
    try:
        return AutoProcessor.from_pretrained(pretrained_path, trust_remote_code=True)
    except Exception as e:
        # Any failure here (ValueError/OSError on transformers 5.x, or
        # AttributeError when a node's stale HF dynamic-module cache for
        # transformers_modules.fast lacks UniversalActionProcessor) falls through
        # to the manual load below, which reads the source file directly and is
        # immune to dynamic-module cache staleness.
        print(f"[fast] AutoProcessor.from_pretrained failed ({type(e).__name__}: {e}); "
              f"falling back to manual load from source")

    # Fallback: manual load
    if os.path.isdir(pretrained_path):
        local_dir = pretrained_path
    else:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(pretrained_path)

    spec = importlib.util.spec_from_file_location(
        "processing_action_tokenizer",
        os.path.join(local_dir, "processing_action_tokenizer.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    UniversalActionProcessor = mod.UniversalActionProcessor

    bpe_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(local_dir, "tokenizer.json"),
        clean_up_tokenization_spaces=False,
    )

    with open(os.path.join(local_dir, "processor_config.json"), "r") as f:
        cfg = json.load(f)

    processor = UniversalActionProcessor(
        bpe_tokenizer=bpe_tokenizer,
        scale=cfg.get("scale", 10),
        vocab_size=cfg.get("vocab_size", 2048),
        min_token=cfg.get("min_token", -354),
        action_dim=cfg.get("action_dim"),
        time_horizon=cfg.get("time_horizon"),
    )
    return processor


class Fast_Action_Tokenizer(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, fast_tokenizer_name="playground/Pretrained_models/fast"):
        super().__init__()

        self.fast_tokenizer = _load_fast_processor(
            fast_tokenizer_name
        )  # load https://huggingface.co/physical-intelligence/fast

    def encoder_action2fastoken(self, raw_actions):
        batch_fast_tokens, _ = encode_fast_actions_by_shape(self.fast_tokenizer, raw_actions)
        return batch_fast_tokens  # List[str]

    def decoder_action(self, generated_ids):
        # api https://huggingface.co/physical-intelligence/fast
        # return: (batch_size, chunck, dim)
        pred_actions = self.fast_tokenizer.decode([generated_ids - self._ACTION_TOKEN_MIN])
        return pred_actions

    def fit_tokenizer_on_datasets(
        self,
        action_dataset,
        datasets_path="<your_local_path>",
    ):
        # If datasets_path exists, load directly
        if os.path.exists(datasets_path):

            self.fast_tokenizer = AutoProcessor.from_pretrained(datasets_path, trust_remote_code=True)
            return
        else:
            # If not found, Fit the tokenizer on the new dataset
            new_tokenizer = self.fast_tokenizer.tokenizer.fit(action_dataset)
            self.fast_tokenizer = new_tokenizer

            # Save the new tokenizer, optionally push it to the Hugging Face model hub
            self.fast_tokenizer.save_pretrained(datasets_path)


def get_action_model(config=None):
    """
    Factory: build ActionModel from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).
    Returns:
        ActionModel: Initialized diffusion action head.
    """
    fast_path = getattr(config.framework.action_model, "fast_tokenizer_path", None) if config else None
    if fast_path:
        action_model = Fast_Action_Tokenizer(fast_tokenizer_name=fast_path)
    else:
        action_model = Fast_Action_Tokenizer()

    return action_model


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10094))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10094 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":

    start_debugpy_once()

    fast_tokenizer_name = "physical-intelligence/fast"
    fast_tokenizer = Fast_Action_Tokenizer(fast_tokenizer_name=fast_tokenizer_name)
    raw_actions = [np.random.randn(16, 7), np.random.randn(16, 7)]

    # Load the tokenizer from the Hugging Face hub
    tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_name, trust_remote_code=True)

    # basic test
    # Tokenize & decode action chunks (we use dummy data here)
    action_data = np.random.rand(2, 16, 7)  # one batch of action chunks
    tokens = tokenizer(action_data)  # tokens = list[int]
    decoded_actions = tokenizer.decode(tokens)

    # self func test
    vlm_tokens = fast_tokenizer.encoder_action2vlmtoken(raw_actions)
    print(vlm_tokens)
    pred_actions = fast_tokenizer.decoder_action(np.array([12, 3, 45, 87]))
    print(pred_actions)
