from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight CI without torch
    torch = None

from mr0.dataloader.lerobot_datasets import make_padding_collate_fn


@unittest.skipIf(torch is None, "torch is required for MMDiT masked loss tests")
class MMDiTActionMaskTest(unittest.TestCase):
    def test_padding_collate_adds_action_mask_for_mixed_action_widths(self):
        collate = make_padding_collate_fn(action_dim=20, action_horizon=16, state_dim=20)
        batch = [
            {
                "action": np.ones((16, 10), dtype=np.float16),
                "state": np.ones((1, 10), dtype=np.float16),
            },
            {
                "action": np.ones((16, 20), dtype=np.float16),
                "state": np.ones((1, 20), dtype=np.float16),
            },
        ]

        out = collate(batch)

        self.assertEqual(out[0]["action"].shape, (16, 20))
        self.assertEqual(out[1]["action"].shape, (16, 20))
        self.assertEqual(out[0]["state"].shape, (1, 20))
        self.assertEqual(out[1]["state"].shape, (1, 20))
        np.testing.assert_array_equal(
            out[0]["action_mask"],
            np.array([True] * 10 + [False] * 10),
        )
        np.testing.assert_array_equal(out[1]["action_mask"], np.ones(20, dtype=bool))
        self.assertEqual(float(out[0]["action"][:, 10:].sum()), 0.0)

    def test_recap_routing_matrix(self):
        from mr0.model.framework.VLM4A.QwenMMDiT import route_recap_prompts

        cases = [
            (True, True, 0.0, "task\nAdvantage: positive", "positive_conditional_count"),
            (True, True, 1.0, "task", "positive_unconditional_count"),
            (False, True, 0.0, "task", "negative_unconditional_count"),
            (False, True, 1.0, "task", "negative_unconditional_count"),
            (True, False, 0.0, "task\nAdvantage: positive", "positive_conditional_count"),
            (False, False, 0.0, "task\nAdvantage: negative", "negative_conditional_count"),
            (False, False, 1.0, "task", "negative_unconditional_count"),
        ]
        for advantage, positive_only, dropout, expected_prompt, expected_branch in cases:
            with self.subTest(advantage=advantage, positive_only=positive_only, dropout=dropout):
                prompts, metrics = route_recap_prompts(
                    ["task"],
                    [advantage],
                    positive_only_conditional=positive_only,
                    unconditional_prob=dropout,
                )
                self.assertEqual(prompts, [expected_prompt])
                self.assertEqual(metrics[expected_branch], 1)
                self.assertEqual(metrics["positive_count"] + metrics["negative_count"], 1)
                self.assertEqual(
                    metrics["positive_conditional_count"]
                    + metrics["positive_unconditional_count"]
                    + metrics["negative_conditional_count"]
                    + metrics["negative_unconditional_count"],
                    1,
                )

    def test_recap_routing_is_per_sample_and_seed_reproducible(self):
        from mr0.model.framework.VLM4A.QwenMMDiT import route_recap_prompts

        instructions = [f"task {index}" for index in range(32)]
        advantages = [index % 2 == 0 for index in range(32)]
        first = route_recap_prompts(
            instructions,
            advantages,
            unconditional_prob=0.5,
            generator=torch.Generator().manual_seed(123),
        )
        second = route_recap_prompts(
            instructions,
            advantages,
            unconditional_prob=0.5,
            generator=torch.Generator().manual_seed(123),
        )
        self.assertEqual(first, second)
        self.assertTrue(any(prompt.endswith("Advantage: positive") for prompt in first[0]))
        self.assertTrue(any(prompt == instruction for prompt, instruction in zip(first[0], instructions, strict=True)))

    def test_recap_label_changes_routed_prompt(self):
        from mr0.model.framework.VLM4A.QwenMMDiT import route_recap_prompts

        positive, _ = route_recap_prompts(["task"], [True], unconditional_prob=0.0)
        negative, _ = route_recap_prompts(["task"], [False], unconditional_prob=0.0)
        self.assertNotEqual(positive, negative)

    def test_cfg_reuses_noise_and_obeys_beta_boundaries(self):
        from mr0.model.framework.VLM4A.QwenMMDiT import Qwen_MMDiT

        calls = []
        conditional = np.full((1, 4, 2), 3.0, dtype=np.float32)
        unconditional = np.full((1, 4, 2), 1.0, dtype=np.float32)

        def predict_action(examples, noise):
            calls.append((examples[0]["lang"], noise))
            actions = conditional if examples[0]["lang"].endswith("Advantage: positive") else unconditional
            return {"normalized_actions": actions}

        fake_model = SimpleNamespace(
            action_model=SimpleNamespace(
                device=torch.device("cpu"), dtype=torch.float32, action_horizon=4, action_dim=2
            ),
            predict_action=predict_action,
        )
        noise = torch.randn(1, 4, 2)
        examples = [{"lang": "task"}]

        beta_zero = Qwen_MMDiT.predict_action_cfg(fake_model, examples, beta=0.0, noise=noise)
        self.assertIs(calls[0][1], noise)
        self.assertIs(calls[1][1], noise)
        np.testing.assert_array_equal(beta_zero["normalized_actions"], unconditional)

        calls.clear()
        beta_one = Qwen_MMDiT.predict_action_cfg(fake_model, examples, beta=1.0, noise=noise)
        self.assertIs(calls[0][1], calls[1][1])
        np.testing.assert_array_equal(beta_one["normalized_actions"], conditional)
        self.assertEqual(beta_one["normalized_actions"].shape, (1, 4, 2))
        self.assertTrue(np.isfinite(beta_one["normalized_actions"]).all())

    def test_mmdit_explicit_noise_validation(self):
        from mr0.model.modules.action_model.MMDiT_ActionHeader import MMDiTFlowmatchingActionHead

        head = MMDiTFlowmatchingActionHead.__new__(MMDiTFlowmatchingActionHead)
        torch.nn.Module.__init__(head)
        head.action_horizon = 4
        head.action_dim = 2
        vl_embs = torch.zeros(1, 3, 8)
        with self.assertRaisesRegex(ValueError, "noise must have shape"):
            head.predict_action(vl_embs, noise=torch.zeros(1, 3, 2))

    def test_masked_velocity_mse_ignores_padded_action_dimensions(self):
        from mr0.model.modules.action_model.MMDiT_ActionHeader import masked_velocity_mse

        pred_velocity = torch.zeros(2, 4, 20)
        velocity_target = torch.ones(2, 4, 20)
        velocity_target[0, :, 10:] = 1000.0
        action_mask = torch.ones(2, 20, dtype=torch.bool)
        action_mask[0, 10:] = False

        masked_loss = masked_velocity_mse(pred_velocity, velocity_target, action_mask)
        unmasked_loss = masked_velocity_mse(pred_velocity, velocity_target, None)

        self.assertAlmostEqual(masked_loss.item(), 1.0, places=6)
        self.assertGreater(unmasked_loss.item(), 1_000.0)


if __name__ == "__main__":
    unittest.main()
