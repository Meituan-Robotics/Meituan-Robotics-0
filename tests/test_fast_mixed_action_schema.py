from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from mr0.model.framework.VLM4A.QwenFast import Qwenvl_Fast
from mr0.dataloader.lerobot_datasets import FastActionTokenCollator
from mr0.model.modules.action_model.fast_ActionHeader import (
    decode_fast_actions_by_shape,
    encode_fast_actions_by_shape,
)


class FakeFastTokenizer:
    def __init__(self):
        self.encoded_shapes = []
        self.decoded_schema = []
        # New decode path reads these directly instead of calling self.decode().
        self.scale = 10.0
        self.min_token = 0
        self.bpe_tokenizer = SimpleNamespace(decode=self._bpe_decode)
        self._bpe_calls = []

    def __call__(self, batch_actions):
        self.encoded_shapes.append(tuple(batch_actions.shape))
        return [[batch_actions.shape[-1], i] for i in range(batch_actions.shape[0])]

    def _bpe_decode(self, token_ids):
        # Fake: emit `time_horizon * action_dim` characters that reshape cleanly.
        # We don't know the schema here, so just echo enough chars driven by
        # token count; the safe decoder will truncate/pad as needed.
        # Use codepoint = 1 for every produced char so DCT inversion is stable.
        self._bpe_calls.append(tuple(token_ids))
        return "".join(chr(1) for _ in range(max(1, len(token_ids) * 16)))

    def decode(self, tokens, *, time_horizon=None, action_dim=None):
        # Retained for any callers that still invoke the legacy method.
        self.decoded_schema.append((tuple(tokens[0]), int(time_horizon), int(action_dim)))
        return np.full((1, int(time_horizon), int(action_dim)), fill_value=action_dim, dtype=np.float32)


class FastMixedActionSchemaTest(unittest.TestCase):
    def test_encode_groups_by_native_action_dim_without_dim_padding(self):
        tokenizer = FakeFastTokenizer()
        actions = [
            np.ones((16, 10), dtype=np.float16),
            np.ones((16, 20), dtype=np.float16),
        ]

        tokens, fitted_actions = encode_fast_actions_by_shape(tokenizer, actions, target_time_horizon=16)

        self.assertEqual(tokens, [[10, 0], [20, 0]])
        self.assertEqual([action.shape for action in fitted_actions], [(16, 10), (16, 20)])
        self.assertEqual(tokenizer.encoded_shapes, [(1, 16, 10), (1, 16, 20)])

    def test_collator_keeps_fast_schema_fields_for_mixed_action_dims(self):
        tokenizer = FakeFastTokenizer()
        collator = FastActionTokenCollator(
            base_collate_fn=lambda batch: batch,
            fast_tokenizer_path="unused",
            action_dim=20,
            time_horizon=16,
        )
        collator.fast_tokenizer = tokenizer
        batch = [
            {"action": np.ones((8, 10), dtype=np.float16)},
            {"action": np.ones((16, 20), dtype=np.float16)},
        ]

        out = collator(batch)

        self.assertEqual(out[0]["action"].shape, (16, 10))
        self.assertEqual(out[1]["action"].shape, (16, 20))
        self.assertEqual(out[0]["fast_action_time_horizon"], 16)
        self.assertEqual(out[0]["fast_action_dim"], 10)
        self.assertEqual(out[1]["fast_action_dim"], 20)
        self.assertEqual(out[0]["fast_action_tokens"], [10, 0])
        self.assertEqual(out[1]["fast_action_tokens"], [20, 0])
        self.assertEqual(out[0]["vlm_action_tokens"], "<robot_action_10><robot_action_0>")
        self.assertEqual(tokenizer.encoded_shapes, [(1, 16, 10), (1, 16, 20)])

    def test_decode_uses_per_sample_fast_schema(self):
        tokenizer = FakeFastTokenizer()

        decoded = decode_fast_actions_by_shape(
            tokenizer,
            batch_fast_token_ids=[[1, 2, 3], [4, 5]],
            time_horizons=[16, 16],
            action_dims=[10, 20],
        )

        self.assertIsInstance(decoded, list)
        self.assertEqual([action.shape for action in decoded], [(16, 10), (16, 20)])
        self.assertEqual(tokenizer._bpe_calls, [(1, 2, 3), (4, 5)])

    def test_decode_stacks_when_schema_matches(self):
        tokenizer = FakeFastTokenizer()

        decoded = decode_fast_actions_by_shape(
            tokenizer,
            batch_fast_token_ids=[[1], [2]],
            time_horizons=[16, 16],
            action_dims=[10, 10],
        )

        self.assertEqual(decoded.shape, (2, 16, 10))

    def test_decode_handles_none_token_ids(self):
        tokenizer = FakeFastTokenizer()

        decoded = decode_fast_actions_by_shape(
            tokenizer,
            batch_fast_token_ids=[None, [1, 2]],
            time_horizons=[4, 4],
            action_dims=[7, 7],
        )

        self.assertEqual(decoded.shape, (2, 4, 7))
        self.assertTrue(np.all(decoded[0] == 0))

    def test_qwenfast_rejects_generation_without_action_tokens(self):
        model = Qwenvl_Fast.__new__(Qwenvl_Fast)
        model.qwen_vl_interface = SimpleNamespace(_ACTION_TOKEN_MIN=100, _ACTION_TOKEN_MAX=200)

        with self.assertRaisesRegex(RuntimeError, "no FAST action tokens"):
            model._validate_generated_action_tokens(
                [[]],
                generated_ids=torch.tensor([[1, 2, 3]]),
                qwen_inputs={"attention_mask": torch.tensor([[1, 1, 1]])},
                max_new_tokens=128,
            )

    def test_qwenfast_extracts_first_contiguous_action_segment(self):
        model = Qwenvl_Fast.__new__(Qwenvl_Fast)
        model.qwen_vl_interface = SimpleNamespace(_ACTION_TOKEN_MIN=100, _ACTION_TOKEN_MAX=200)

        extracted = model._extract_action_token_ids(
            torch.tensor(
                [
                    [9, 100, 101, 102, 7, 103, 104],
                    [9, 8, 105, 4, 106, 107, 3],
                ]
            )
        )

        self.assertEqual(extracted, [[100, 101, 102], [105]])


if __name__ == "__main__":
    unittest.main()
