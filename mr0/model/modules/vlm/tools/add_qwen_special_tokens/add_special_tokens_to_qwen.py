# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].


import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
)


def add_new_tokens(
    model,
    tokenizer,
    new_tokens: List[str],
    init_strategy: str = "avg",
    as_special: bool = True,
) -> Tuple[Dict[str, int], int, int, int]:
    """
    Add new tokens into the model and tokenizer (if they don't already exist).
    init_strategy: avg / normal / zero
    Returns:
      - mapping: token_id mapping for all target tokens
      - added_now: number of tokens actually added to the tokenizer this time
      - action_token_start_idx: start index of newly added embeddings (based on old embedding size)
      - action_token_end_idx: end index of newly added embeddings (if none added, equals start_idx - 1)
    Notes:
      - tokenizer.vocab_size is the base vocabulary size (excluding already added special/added tokens)
      - len(tokenizer) is the total vocabulary size (including added/special tokens)
      - The old embedding size of the model is model.get_input_embeddings().weight.shape[0]
    """
    # 1) Compute tokens to add (relative to current tokenizer vocab)
    vocab = tokenizer.get_vocab()  # includes existing special tokens
    to_add_tokens = [t for t in new_tokens if t not in vocab]

    # 2) Record current embedding/head size of the model.
    # Qwen checkpoints may have reserved model rows that are not present in the
    # tokenizer yet. Newly added tokens can land inside those rows, so
    # initialization must use token ids, not only [old_size, new_size).
    old_embed = model.get_input_embeddings()
    old_embed_size = old_embed.weight.shape[0]  # includes Qwen reserved tokens
    old_embed_ref = old_embed.weight.detach().mean(dim=0, keepdim=True).clone()
    old_output = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    old_output_ref = None
    if old_output is not None and hasattr(old_output, "weight"):
        old_output_ref = old_output.weight.detach().mean(dim=0, keepdim=True).clone()

    # 3) If needed, add tokens into tokenizer first
    added_now = 0
    if to_add_tokens:
        if as_special:
            added_now = tokenizer.add_special_tokens({"additional_special_tokens": to_add_tokens})
        else:
            added_now = tokenizer.add_tokens(to_add_tokens)

    # 4) Target total size is exactly the tokenizer size. This preserves Qwen's
    # reserved rows when token ids fall into them, and only extends the model
    # when token ids exceed the current embedding/head size.
    target_size = len(tokenizer)
    action_token_start_idx = old_embed_size
    action_token_end_idx = old_embed_size - 1
    if target_size > old_embed_size:
        model.resize_token_embeddings(target_size)  # resizing to target size

    # 6) Build mapping (return ids for requested tokens)
    mapping = {t: tokenizer.convert_tokens_to_ids(t) for t in new_tokens}
    added_token_ids = [int(mapping[t]) for t in to_add_tokens]
    if added_token_ids:
        action_token_start_idx = min(added_token_ids)
        action_token_end_idx = max(added_token_ids)
        _init_token_rows(
            model.get_input_embeddings().weight,
            added_token_ids,
            init_strategy,
            old_embed_ref,
        )
        output_embed = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embed is not None and hasattr(output_embed, "weight"):
            input_weight = model.get_input_embeddings().weight
            if output_embed.weight.data_ptr() != input_weight.data_ptr():
                ref = old_output_ref if old_output_ref is not None else old_embed_ref
                _init_token_rows(output_embed.weight, added_token_ids, init_strategy, ref)

    return mapping, added_now, action_token_start_idx, action_token_end_idx


def _init_token_rows(
    weight: torch.Tensor,
    token_ids: List[int],
    init_strategy: str,
    avg_ref: torch.Tensor,
) -> None:
    unique_ids = sorted(set(int(idx) for idx in token_ids))
    if not unique_ids:
        return
    if unique_ids[-1] >= weight.shape[0]:
        raise ValueError(
            f"Token id {unique_ids[-1]} exceeds embedding/head size {weight.shape[0]}; "
            "resize_token_embeddings did not cover the tokenizer."
        )
    with torch.no_grad():
        if init_strategy == "avg":
            ref_vec = avg_ref.to(device=weight.device, dtype=weight.dtype)
            for idx in unique_ids:
                weight[idx].copy_(ref_vec[0])
        elif init_strategy == "zero":
            for idx in unique_ids:
                weight[idx].zero_()
        elif init_strategy == "normal":
            for idx in unique_ids:
                nn.init.normal_(weight[idx], mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")


def _load_qwen_vl_model(model_id: str, device: str, attn_implementation: str):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    if model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        model_cls = Qwen3_5ForConditionalGeneration
    elif model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
    elif model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model_type for token extension: {model_type}")

    kwargs = {
        "attn_implementation": attn_implementation,
        "torch_dtype": torch.bfloat16,
    }
    if device == "auto":
        kwargs["device_map"] = "auto"
    elif device.startswith("cuda"):
        kwargs["device_map"] = device
    model = model_cls.from_pretrained(model_id, **kwargs)
    if device == "cpu":
        model = model.to("cpu")
    return model


def save_bundle(
    model,
    tokenizer,
    mapping: Dict[str, int],
    save_dir: str,
    processor_src: str | None = None,
    padding_side: str | None = None,
):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "added_custom_token_id_map.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved to: {save_dir}")

    # Additionally save AutoProcessor (generate preprocessor_config.json) so AutoProcessor.from_pretrained(...) can load
    try:
        src = processor_src or save_dir
        processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
        # Sync processor.tokenizer
        processor.tokenizer = tokenizer
        processor.save_pretrained(save_dir)
        print(f"[OK] AutoProcessor saved to: {save_dir}")
    except Exception as e:
        print(f"[WARN] Failed to save AutoProcessor: {e}")


def reload_and_check(save_dir: str, tokens: List[str]) -> bool:
    tok = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
    vocab = tok.get_vocab()
    missing = [t for t in tokens if t not in vocab]
    if missing:
        print(f"[WARN] Still missing after reload: {missing}")
        return False
    print("[OK] Reload check passed, all tokens exist.")
    return True


def parse_tokens(args) -> List[str]:
    tokens: List[str] = []
    if args.tokens:
        tokens.extend([t.strip() for t in args.tokens.split(",") if t.strip()])
    if args.tokens_file:
        with open(args.tokens_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens.append(line)
    # De-duplicate while keeping order
    seen = set()
    ordered = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def main():
    parser = argparse.ArgumentParser(description="Add special tokens to Qwen-VL compatible models and save locally.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct", help="HF Hub model ID or local path")
    parser.add_argument("--save-dir", required=True, help="Output directory to save")
    parser.add_argument("--tokens", default="", help="Comma-separated tokens, e.g., <loc_x>,<loc_y>")
    parser.add_argument("--tokens-file", help="Text file containing tokens to add (one per line)")
    parser.add_argument(
        "--init-strategy",
        default="avg",
        choices=["avg", "normal", "zero"],
        help="Initialization strategy for newly added embeddings",
    )
    parser.add_argument("--as-special", action="store_true", help="Whether to add as special tokens")
    parser.add_argument("--no-as-special", dest="as_special", action="store_false")
    parser.set_defaults(as_special=True)
    parser.add_argument("--padding-side", default="left", choices=["left", "right"])
    parser.add_argument("--device", default="cuda", help="cuda / cuda:N / cpu / auto")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    args = parser.parse_args()

    tokens = parse_tokens(args)
    if not tokens:
        print("No tokens provided, use --tokens or --tokens-file")
        return

    print(f"[INFO] Tokens to process: {len(tokens)}")
    print(f"[INFO] Token preview: first={tokens[:3]}, last={tokens[-3:]}")

    print(f"[INFO] Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = args.padding_side
    model = _load_qwen_vl_model(
        args.model_id,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    # Print sizes for diagnosis
    base_tok_size = tokenizer.vocab_size  # base vocab size
    total_tok_size = len(tokenizer)  # total vocab size
    model_embed_size = model.get_input_embeddings().weight.shape[0]  # current model embedding size
    print(f"[DEBUG] tokenizer.vocab_size(base) = {base_tok_size}")
    print(f"[DEBUG] len(tokenizer)(total)     = {total_tok_size}")
    print(f"[DEBUG] model.embed_size(before)  = {model_embed_size}")
    print(f"[DEBUG] added_in_tokenizer        = {total_tok_size - base_tok_size}")

    mapping, added, action_token_start_idx, action_token_end_idx = add_new_tokens(
        model=model,
        tokenizer=tokenizer,
        new_tokens=tokens,
        init_strategy=args.init_strategy,
        as_special=args.as_special,
    )
    new_model_embed_size = model.get_input_embeddings().weight.shape[0]

    save_bundle(model, tokenizer, mapping, args.save_dir, processor_src=args.model_id, padding_side=args.padding_side)

    # Re-validate
    reload_and_check(args.save_dir, tokens)

    print(f"[INFO] Newly added to tokenizer: {added}")
    # print(f"[INFO] Token mapping: {mapping}")
    print(f"[INFO] Action token idx range: [{action_token_start_idx}, {action_token_end_idx}]")
    print(f"[DEBUG] model.embed_size(after)   = {new_model_embed_size}")


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    main()
