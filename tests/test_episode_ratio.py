"""Unit tests for LeRobotSingleDataset.episode_ratio subsetting.

Doesn't spin up a real LeRobot dataset — exercises the subsetting helper
directly on a stand-in that carries just the fields it touches.
"""
from __future__ import annotations

import hashlib
import math
import unittest

import numpy as np


class _StubDataset:
    def __init__(self, name: str, n_episodes: int, data_cfg: dict | None = None):
        self._dataset_name = name
        self._trajectory_ids = np.arange(n_episodes, dtype=np.int64)
        self._trajectory_lengths = np.full(n_episodes, 100, dtype=np.int64)
        self.data_cfg = data_cfg

    from mr0.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

    _maybe_apply_episode_subset = LeRobotSingleDataset._maybe_apply_episode_subset


class EpisodeRatioTests(unittest.TestCase):
    def test_no_cfg_is_noop(self):
        ds = _StubDataset("foo", 1000, data_cfg=None)
        ds._maybe_apply_episode_subset()
        self.assertEqual(len(ds._trajectory_ids), 1000)

    def test_ratio_none_is_noop(self):
        ds = _StubDataset("foo", 1000, data_cfg={})
        ds._maybe_apply_episode_subset()
        self.assertEqual(len(ds._trajectory_ids), 1000)

    def test_ratio_one_is_noop(self):
        ds = _StubDataset("foo", 1000, data_cfg={"episode_ratio": 1.0})
        ds._maybe_apply_episode_subset()
        self.assertEqual(len(ds._trajectory_ids), 1000)

    def test_ratio_zero_rejected(self):
        ds = _StubDataset("foo", 1000, data_cfg={"episode_ratio": 0.0})
        with self.assertRaises(ValueError):
            ds._maybe_apply_episode_subset()

    def test_ratio_gt_one_rejected(self):
        ds = _StubDataset("foo", 1000, data_cfg={"episode_ratio": 1.5})
        with self.assertRaises(ValueError):
            ds._maybe_apply_episode_subset()

    def test_ratio_bad_type_rejected(self):
        ds = _StubDataset("foo", 1000, data_cfg={"episode_ratio": "not-a-number"})
        with self.assertRaises(ValueError):
            ds._maybe_apply_episode_subset()

    def test_ratio_10pct_keeps_ceil_fraction(self):
        ds = _StubDataset("foo", 1000, data_cfg={"episode_ratio": 0.1})
        ds._maybe_apply_episode_subset()
        self.assertEqual(len(ds._trajectory_ids), 100)
        # ids must be unique + sorted + in [0, n)
        ids = ds._trajectory_ids.tolist()
        self.assertEqual(sorted(set(ids)), ids)
        self.assertTrue(all(0 <= i < 1000 for i in ids))

    def test_ratio_ceil_rounding(self):
        # 25 * 0.1 = 2.5 → ceil → 3
        ds = _StubDataset("foo", 25, data_cfg={"episode_ratio": 0.1})
        ds._maybe_apply_episode_subset()
        self.assertEqual(len(ds._trajectory_ids), 3)

    def test_tiny_leaf_keeps_at_least_one(self):
        # 3 * 0.1 = 0.3 → ceil → 1 (still >= 1 by max clamp)
        ds = _StubDataset("foo", 3, data_cfg={"episode_ratio": 0.1})
        ds._maybe_apply_episode_subset()
        self.assertEqual(len(ds._trajectory_ids), 1)

    def test_single_episode_leaf_survives(self):
        ds = _StubDataset("foo", 1, data_cfg={"episode_ratio": 0.1})
        ds._maybe_apply_episode_subset()
        self.assertEqual(len(ds._trajectory_ids), 1)

    def test_empty_leaf_stays_empty(self):
        ds = _StubDataset("foo", 0, data_cfg={"episode_ratio": 0.1})
        ds._maybe_apply_episode_subset()
        self.assertEqual(len(ds._trajectory_ids), 0)

    def test_deterministic_same_seed(self):
        d1 = _StubDataset("foo", 1000, data_cfg={"episode_ratio": 0.1})
        d2 = _StubDataset("foo", 1000, data_cfg={"episode_ratio": 0.1})
        d1._maybe_apply_episode_subset()
        d2._maybe_apply_episode_subset()
        self.assertListEqual(d1._trajectory_ids.tolist(), d2._trajectory_ids.tolist())

    def test_different_leaf_names_different_subsets(self):
        d1 = _StubDataset("leaf_a", 1000, data_cfg={"episode_ratio": 0.1})
        d2 = _StubDataset("leaf_b", 1000, data_cfg={"episode_ratio": 0.1})
        d1._maybe_apply_episode_subset()
        d2._maybe_apply_episode_subset()
        # With 100 picks out of 1000 and different name-derived seeds,
        # the two subsets must not be identical.
        self.assertNotEqual(d1._trajectory_ids.tolist(), d2._trajectory_ids.tolist())

    def test_different_seed_different_subset(self):
        d1 = _StubDataset("foo", 1000, data_cfg={"episode_ratio": 0.1, "episode_subset_seed": 42})
        d2 = _StubDataset("foo", 1000, data_cfg={"episode_ratio": 0.1, "episode_subset_seed": 7})
        d1._maybe_apply_episode_subset()
        d2._maybe_apply_episode_subset()
        self.assertNotEqual(d1._trajectory_ids.tolist(), d2._trajectory_ids.tolist())

    def test_lengths_align_with_ids(self):
        ds = _StubDataset("foo", 1000, data_cfg={"episode_ratio": 0.1})
        ds._trajectory_lengths = (np.arange(1000) + 1).astype(np.int64)  # length = id+1
        ds._maybe_apply_episode_subset()
        for tid, tlen in zip(ds._trajectory_ids, ds._trajectory_lengths):
            self.assertEqual(int(tlen), int(tid) + 1)


if __name__ == "__main__":
    unittest.main()
