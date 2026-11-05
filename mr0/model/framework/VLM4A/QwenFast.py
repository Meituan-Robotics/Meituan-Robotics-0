# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-Fast Framework

A lightweight implementation for autoregressive discrete action prediction conditioned on multi-view images + instruction.
fast tokenizer is copyright from physical-intelligence/fast

Key Points:
  - Qwen3.5 vision-language backbone
  - Unified action learning via next-token prediction (fast tokenizer)
  - Autoregressive action tokens derived from discretized / symbolized continuous actions

Prepare action tokens with mr0/model/modules/vlm/tools/add_qwen_special_tokens/.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import os
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from deployment.model_server.tools.image_tools import to_pil_preserve
from mr0.model.tools import FRAMEWORK_REGISTRY
from mr0.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from mr0.model.framework.base_framework import baseframework
from mr0.model.framework.share_tools import merge_framework_config
from mr0.model.modules.action_model.fast_ActionHeader import (
    decode_fast_actions_by_shape,
    encode_fast_actions_by_shape,
    get_action_model,
)
from mr0.model.modules.vlm import get_vlm_model


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenFast
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenFastDefaultConfig:
    """QwenFast framework default parameters.

    Autoregressive discrete action prediction via FAST tokenizer.
    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.
    """

    # --- Registry identifier ---
    name: str = "QwenFast"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL with action special tokens) ===
    qwenvl: dict = field(default_factory=lambda: {
        # Path to base VLM checkpoint (must include FAST action tokens)
        "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
        # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
        "attn_implementation": "flash_attention_2",
    })

    # === Action head (FAST tokenizer — discrete next-token prediction) ===
    action_model: dict = field(default_factory=lambda: {
        # Action head architecture type
        "action_model_type": "FAST",
        # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
        "action_dim": 7,
        # How many future steps to predict
        "future_action_window_size": 15,
        # How many past steps included in action chunk (usually 0)
        "past_action_window_size": 0,
        # FAST tokenizer pretrained path (override in your training config)
        "fast_tokenizer_path": "physical-intelligence/fast",
    })

    # === Observation image size (optional resize before encoding) ===
    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("QwenFast")
class Qwenvl_Fast(baseframework):
    """
    Multimodal vision-language-action model (FAST variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - FAST tokenizer for discretized / symbolized continuous action encoding
      - Autoregressive next-token prediction over action tokens

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenFastDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        # self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.action_model.fast_tokenizer.time_horizon = self.future_action_window_size + 1
        self.action_model.fast_tokenizer.action_dim = self.config.framework.action_model.action_dim
        self._validate_action_token_setup()

        # Optional per-step instrumentation (toggled with MR0_PROFILE_STEP=1).
        # Measures main-thread time spent in build_qwenvl_inputs vs. the actual
        # GPU forward wall time inside Qwenvl_Fast.forward(). Rank 0 only.
        self._profile_step = os.getenv("MR0_PROFILE_STEP", "0") == "1"
        self._profile_window = int(os.getenv("MR0_PROFILE_WINDOW", "100"))
        self._t_build: List[float] = []
        self._t_fwd_call: List[float] = []
        self._profile_idx = 0

    def _profile_emit(self) -> None:
        if not self._t_build:
            return
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_rank0:
            def _stats(xs: List[float]) -> str:
                a = np.asarray(xs)
                return (f"mean={a.mean():6.2f}ms p50={np.median(a):6.2f} "
                        f"p95={np.percentile(a, 95):6.2f} max={a.max():6.2f}")
            logger.info(
                f"[QwenFast profile] step ~{self._profile_idx} "
                f"(last {len(self._t_build)} forwards):"
            )
            logger.info(f"  build_qwenvl_inputs (main thread)  {_stats(self._t_build)}")
            logger.info(f"  qwen_vl_interface forward (GPU)    {_stats(self._t_fwd_call)}")
        self._t_build.clear()
        self._t_fwd_call.clear()

    def _validate_action_token_setup(self) -> None:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        token_ids = []
        missing = []
        for idx in range(2048):
            token = f"<robot_action_{idx}>"
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == tokenizer.unk_token_id:
                missing.append(token)
            else:
                token_ids.append(int(token_id))
        if missing:
            preview = ", ".join(missing[:3])
            raise ValueError(
                "QwenFast requires FAST action tokens in the VLM tokenizer. "
                f"Missing {len(missing)} tokens, e.g. {preview}."
            )

        action_token_min = token_ids[0]
        expected_ids = list(range(action_token_min, action_token_min + len(token_ids)))
        if token_ids != expected_ids:
            raise ValueError(
                "QwenFast requires contiguous <robot_action_i> token ids. "
                f"Got first={token_ids[0]}, last={token_ids[-1]}."
            )
        self.qwen_vl_interface._ACTION_TOKEN_MIN = action_token_min
        self.qwen_vl_interface._ACTION_TOKEN_MAX = token_ids[-1]

        input_vocab_size = self.qwen_vl_interface.model.get_input_embeddings().weight.shape[0]
        if token_ids[-1] >= input_vocab_size:
            raise ValueError(
                f"Action token id {token_ids[-1]} exceeds input embedding size {input_vocab_size}. "
                "Regenerate the -Action checkpoint with resized token embeddings."
            )

        output_embed = self.qwen_vl_interface.model.get_output_embeddings()
        if output_embed is None or not hasattr(output_embed, "weight"):
            raise ValueError("QwenFast requires a VLM lm_head/output embedding for action-token prediction.")
        output_vocab_size = output_embed.weight.shape[0]
        if token_ids[-1] >= output_vocab_size:
            raise ValueError(
                f"Action token id {token_ids[-1]} exceeds lm_head size {output_vocab_size}. "
                "Regenerate the -Action checkpoint with resized output embeddings."
            )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly predict future actions via next-token prediction (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        if all("vlm_action_tokens" in example for example in examples):
            vlm_action_tokens = [example["vlm_action_tokens"] for example in examples]
        else:
            actions = [example["action"] for example in examples]  # label [B, len, 7]

            # step 0: map_raw_action_to_vlm_action
            batch_fast_tokens, _ = encode_fast_actions_by_shape(
                self.action_model.fast_tokenizer,
                actions,
                target_time_horizon=self.chunk_len,
            )

            # batch_fast_tokens = [self.fast_tokenizer(raw_action)[0] for raw_action in raw_actions]
            vlm_action_tokens = [self.map_fast_token_to_vlm_action(fast_tokens) for fast_tokens in batch_fast_tokens]

        # Step 1: QWenVL input format
        if self._profile_step:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, solutions=vlm_action_tokens
        )
        if self._profile_step:
            _t1 = time.perf_counter()
            self._t_build.append((_t1 - _t0) * 1000)
        labels = qwen_inputs.get("labels")
        action_token_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        action_token_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        if labels is None or not ((labels >= action_token_min) & (labels <= action_token_max)).any():
            raise ValueError(
                "QwenFast built a batch without action-token labels. "
                "Check the FAST tokenizer output and the VLM action-token vocabulary."
            )

        if self._profile_step:
            _t2 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        if self._profile_step:
            torch.cuda.synchronize()
            _t3 = time.perf_counter()
            self._t_fwd_call.append((_t3 - _t2) * 1000)
            self._profile_idx += 1
            if self._profile_idx % self._profile_window == 0:
                self._profile_emit()

        vlm_action_loss = qwenvl_outputs.loss
        if vlm_action_loss is None:
            raise RuntimeError("QwenFast forward did not return an LM loss.")
        if not torch.isfinite(vlm_action_loss):
            raise FloatingPointError(f"QwenFast action loss is not finite: {vlm_action_loss}")

        return {"action_loss": vlm_action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: single forward pass to obtain future actions (no diffusion sampling).
        # can be batch forward
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        # train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        # if train_obs_image_size:
        #     batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = [instruction for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        max_new_tokens = int(kwargs.get("fast_max_new_tokens", kwargs.get("max_new_tokens", 2048)))
        generation_kwargs = {"max_new_tokens": max_new_tokens}
        if "do_sample" in kwargs:
            generation_kwargs["do_sample"] = kwargs["do_sample"]
        for key in ("temperature", "top_p", "top_k", "num_beams"):
            if key in kwargs:
                generation_kwargs[key] = kwargs[key]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(
                **qwen_inputs,
                **generation_kwargs,
            )
        # --- Diagnostic: how many new tokens were generated, did we hit the cap? ---
        _prompt_len = qwen_inputs["input_ids"].shape[1]
        _new_len = generated_ids.shape[1] - _prompt_len
        if _new_len >= max_new_tokens:
            logger.warning(
                "[QwenFast] generation hit max_new_tokens=%d (no EOS); prompt_len=%d. "
                "This is the slow path — model never emitted a stop token.",
                max_new_tokens, _prompt_len,
            )
        else:
            logger.info(
                "[QwenFast] generated %d new tokens (prompt_len=%d, cap=%d)",
                _new_len, _prompt_len, max_new_tokens,
            )
        # --- Extract and decoder vlm_action to continue actions ---
        # --- extrace token (index based on VLM) ---
        batch_vlm_action_token_ids = self._extract_action_token_ids(generated_ids)
        self._validate_generated_action_tokens(
            batch_vlm_action_token_ids,
            generated_ids=generated_ids,
            qwen_inputs=qwen_inputs,
            max_new_tokens=max_new_tokens,
        )
        # --- map index to fast tokenizer index space ---
        batch_fast_action_token_idx = self._decode_action_tokens(batch_vlm_action_token_ids)
        # --- decode fast tokenizer index to action semantic ---
        time_horizons, action_dims = self._resolve_decode_schema(examples, kwargs)
        normalized_actions = decode_fast_actions_by_shape(
            self.action_model.fast_tokenizer,
            batch_fast_action_token_idx,
            time_horizons,
            action_dims,
        )

        return {"normalized_actions": normalized_actions}

    def _extract_action_token_ids(
        self,
        generated_ids: torch.LongTensor,
    ) -> List[List[int]]:
        """
        Extract action tokens (with offset) from the generated token sequence and return a 2D list:
        ret[b] = [vlm_action_token_id_0, vlm_action_token_id_1, ...]
        Rule: keep the first contiguous action-token segment. Qwen chat models can
        continue with non-action text after the action chunk; collecting later
        action-like tokens would corrupt FAST BPE decoding.
        """
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        act_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        mask = (generated_ids >= act_min) & (generated_ids <= act_max)  # [B, L]
        results = []
        for b in range(generated_ids.size(0)):
            idx = mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
                continue
            first = idx[0].item()
            last = first
            for pos in idx[1:].tolist():
                if pos != last + 1:
                    break
                last = pos
            tokens = generated_ids[b, first : last + 1].tolist()
            results.append(tokens)
        return results

    def _validate_generated_action_tokens(
        self,
        batch_vlm_action_token_ids: List[List[int]],
        generated_ids: torch.LongTensor,
        qwen_inputs: dict,
        max_new_tokens: int,
    ) -> None:
        missing = [idx for idx, tokens in enumerate(batch_vlm_action_token_ids) if not tokens]
        if not missing:
            return

        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        act_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        attention_mask = qwen_inputs.get("attention_mask")
        input_lengths = None
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
        generated_lengths = [generated_ids.size(1)] * generated_ids.size(0)
        raise RuntimeError(
            "QwenFast generation produced no FAST action tokens for sample indices "
            f"{missing}. action_token_range=[{act_min}, {act_max}], "
            f"input_lengths={input_lengths}, generated_lengths={generated_lengths}, "
            f"max_new_tokens={max_new_tokens}. Check the VLM chat template and action-token checkpoint."
        )

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[Any]:
        """
        Decode the offset VLM action token list back to fast tokenizer semantics.
        fast_tokenizer.decode expects the original fast token id sequence (without offset).
        """
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        batch_fast_token_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_fast_token_ids.append(None)
                continue
            fast_ids = [t - act_min for t in seq]

            batch_fast_token_ids.append(fast_ids)

        return batch_fast_token_ids

    def _resolve_decode_schema(self, examples: List[dict], kwargs: dict) -> tuple[list[int], list[int]]:
        default_time_horizon = self.future_action_window_size + 1
        default_action_dim = self.config.framework.action_model.action_dim

        time_horizon_override = kwargs.get("fast_action_time_horizon", kwargs.get("time_horizon"))
        action_dim_override = kwargs.get("fast_action_dim", kwargs.get("action_dim"))

        time_horizons = []
        action_dims = []
        for idx, example in enumerate(examples):
            if "fast_action_time_horizon" in example:
                time_horizon = example["fast_action_time_horizon"]
            elif "time_horizon" in example:
                time_horizon = example["time_horizon"]
            else:
                time_horizon = self._pick_schema_override(time_horizon_override, idx)

            if "fast_action_dim" in example:
                action_dim = example["fast_action_dim"]
            elif "action_dim" in example:
                action_dim = example["action_dim"]
            else:
                action_dim = self._pick_schema_override(action_dim_override, idx)

            if time_horizon is None and "action" in example:
                time_horizon = np.asarray(example["action"]).shape[0]
            if action_dim is None and "action" in example:
                action_dim = np.asarray(example["action"]).shape[-1]

            time_horizons.append(int(time_horizon or default_time_horizon))
            action_dims.append(int(action_dim or default_action_dim))

        return time_horizons, action_dims

    @staticmethod
    def _pick_schema_override(value, idx: int):
        if isinstance(value, (list, tuple, np.ndarray)):
            return value[idx]
        return value

    def map_fast_token_to_vlm_action(self, tokens) -> str:
        """Maps fast action tokens to the VLM action format.
        Action token 0 is mapped to the string <robot_action_0>  ... and so on
        """
        return "".join(
            [f"<robot_action_{token}>" for token in tokens]
        )  # you should add <robot_action_{token}> to VLM as special tokens,


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action"

    # try get model
    model = Qwenvl_Fast(cfg)
    print(model)

    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 14)).astype(np.float16),  # action_chunk, action_dim
        "image": [image, image],  # two views
        "lang": "This is a fake instruction for testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 14)).astype(np.float16),  # action_chunk, action_dim
        "image": [image, image],  # two views
        "lang": "The fake instruction for testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # test predict action. for new model, it didn't learn to predict action token, so you would meet empty action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    # # test with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from mr0.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
    # vla_dataset_cfg = cfg.datasets.vla_data
    # vla_dataset_cfg.video_backend = "torchvision_av"
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    # from torch.utils.data import DataLoader
    # train_dataloader = DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=collate_fn)
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)
    # action = model.predict_action(batch[0])
    print("Finished")
