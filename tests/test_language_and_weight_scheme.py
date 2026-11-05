"""Unit tests for language_sampling and dataset_weight_scheme.

These tests don't spin up a real LeRobot dataset — they exercise the two
pure helpers directly by calling them on a lightweight stand-in that
mimics the ``LeRobotSingleDataset`` interface used by the helpers under
test.
"""
from __future__ import annotations

import random
import unittest
from collections import Counter

import numpy as np


LANG_KEYS = [
    "annotation.human.action.current_instruction",
    "annotation.human.action.global_instruction",
    "annotation.human.action.global_instruction_dense",
    "annotation.human.action.task_description",
]


class _StubLangDataset:
    """Minimal stand-in that carries just what the helpers under test read."""

    def __init__(self, data_cfg=None):
        self.data_cfg = data_cfg
        self.modality_keys = {"language": LANG_KEYS}

    from mr0.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

    _sample_language_key_order = LeRobotSingleDataset._sample_language_key_order


def _pack_language(ds, data):
    """Mirror of the language-select block in LeRobotSingleDataset._pack_sample."""
    order = ds._sample_language_key_order(LANG_KEYS)
    invalid = {"null", "none", "nan"}
    for key in order:
        vals = data.get(key, [])
        if len(vals) == 0:
            continue
        v = vals[0]
        if v is None:
            continue
        v = str(v).strip()
        if not v or v.lower() in invalid:
            continue
        return v
    return ""


class LanguageSamplingTests(unittest.TestCase):
    SAMPLE = {
        LANG_KEYS[0]: ["curr"],
        LANG_KEYS[1]: ["global"],
        LANG_KEYS[2]: ["dense"],
        LANG_KEYS[3]: ["task"],
    }

    def setUp(self):
        random.seed(0)

    def test_strict_fallback_default(self):
        ds = _StubLangDataset(data_cfg=None)
        for _ in range(100):
            self.assertEqual(_pack_language(ds, self.SAMPLE), "curr")

    def test_50_50_curr_task(self):
        ds = _StubLangDataset(data_cfg={
            "language_sampling": {"current_instruction": 0.5, "task_description": 0.5}
        })
        counts = Counter(_pack_language(ds, self.SAMPLE) for _ in range(20000))
        # Expect roughly 50/50 with a wide tolerance
        self.assertAlmostEqual(counts["curr"] / 20000, 0.5, delta=0.03)
        self.assertAlmostEqual(counts["task"] / 20000, 0.5, delta=0.03)
        self.assertEqual(set(counts.keys()), {"curr", "task"})

    def test_70_30_curr_task(self):
        ds = _StubLangDataset(data_cfg={
            "language_sampling": {"current_instruction": 0.7, "task_description": 0.3}
        })
        counts = Counter(_pack_language(ds, self.SAMPLE) for _ in range(20000))
        self.assertAlmostEqual(counts["curr"] / 20000, 0.7, delta=0.03)
        self.assertAlmostEqual(counts["task"] / 20000, 0.3, delta=0.03)

    def test_full_key_names_accepted(self):
        ds = _StubLangDataset(data_cfg={
            "language_sampling": {
                "annotation.human.action.current_instruction": 0.5,
                "annotation.human.action.task_description": 0.5,
            }
        })
        counts = Counter(_pack_language(ds, self.SAMPLE) for _ in range(2000))
        self.assertEqual(set(counts.keys()), {"curr", "task"})

    def test_null_string_is_invalid(self):
        sample = dict(self.SAMPLE, **{LANG_KEYS[0]: ["null"]})
        ds = _StubLangDataset(data_cfg={
            "language_sampling": {"current_instruction": 1.0}
        })
        # curr is "null" -> falls back to global
        counts = Counter(_pack_language(ds, sample) for _ in range(200))
        self.assertEqual(dict(counts), {"global": 200})

    def test_preferred_empty_falls_back(self):
        sample = dict(self.SAMPLE, **{LANG_KEYS[3]: [""]})
        ds = _StubLangDataset(data_cfg={
            "language_sampling": {"current_instruction": 0.5, "task_description": 0.5}
        })
        # task preferred half the time but empty -> falls back per new order
        # [task, curr, global, dense] -> curr; curr preferred -> curr
        counts = Counter(_pack_language(ds, sample) for _ in range(2000))
        self.assertEqual(dict(counts), {"curr": 2000})

    def test_unknown_keys_dropped(self):
        ds = _StubLangDataset(data_cfg={
            "language_sampling": {"nonexistent_key": 1.0, "current_instruction": 1.0}
        })
        # Only current_instruction survives; all draws should be 'curr'.
        counts = Counter(_pack_language(ds, self.SAMPLE) for _ in range(500))
        self.assertEqual(dict(counts), {"curr": 500})


class DatasetWeightSchemeTests(unittest.TestCase):
    """Verify frames/leaves/sqrt_frames compute the expected weight arrays.

    We reproduce the exact expressions used in LeRobotMixtureDataset.__init__:
      frames      : w := initial_weights * lengths
      leaves      : w := initial_weights          (no length multiplication)
      sqrt_frames : w := initial_weights * sqrt(lengths)
    """

    LENGTHS = np.array([1, 4, 9], dtype=np.float64)
    INITIAL = np.array([1.0, 1.0, 1.0])

    def _norm(self, w):
        return w / w.sum()

    def test_frames_equals_old_true(self):
        # Old: balance_dataset_weights=True -> multiplies by lengths.
        w_new = self.INITIAL * self.LENGTHS
        expected = np.array([1, 4, 9]) / 14.0
        np.testing.assert_allclose(self._norm(w_new), expected)

    def test_leaves_equals_old_false(self):
        # Old: balance_dataset_weights=False -> initial weights unchanged.
        w_new = self.INITIAL.copy()
        expected = np.array([1, 1, 1]) / 3.0
        np.testing.assert_allclose(self._norm(w_new), expected)

    def test_sqrt_frames_new(self):
        w_new = self.INITIAL * np.sqrt(self.LENGTHS)
        expected = np.array([1, 2, 3]) / 6.0
        np.testing.assert_allclose(self._norm(w_new), expected)

    def test_extreme_length_imbalance(self):
        """Reality-check at production scale: with one mega-leaf
        (~5x the size of the smaller ones combined), `frames` gives it
        near-total share, `sqrt_frames` dampens but doesn't crush it."""
        lengths = np.concatenate([np.ones(4000) * 500, [242_000_000]])
        initial = np.ones_like(lengths)
        p_frames = self._norm(initial * lengths)
        p_sqrt = self._norm(initial * np.sqrt(lengths))
        # In `frames`, the mega-leaf dominates strongly (>90% of gradient).
        self.assertGreater(p_frames[-1], 0.90)
        # In `sqrt_frames`, its share drops to a moderate ~15% (from >99%),
        # but doesn't get crushed to zero.
        self.assertLess(p_sqrt[-1], 0.25)
        self.assertGreater(p_sqrt[-1], 0.05)


if __name__ == "__main__":
    unittest.main()
