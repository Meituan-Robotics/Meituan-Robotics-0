from __future__ import annotations

import unittest

import numpy as np

from examples.Robotwin.eval_files.state_preprocess import (
    ROBOTWIN_ENV_TO_TRAIN_STATE_INDICES,
    normalize_robotwin_train_state,
    preprocess_robotwin_state_for_model,
    reorder_robotwin_env_state_to_train,
    reorder_robotwin_train_action_to_env,
    resolve_robotwin_use_state,
)


def _stats(low, high):
    return {
        "min": np.asarray(low, dtype=np.float32).tolist(),
        "max": np.asarray(high, dtype=np.float32).tolist(),
        "q01": np.asarray(low, dtype=np.float32).tolist(),
        "q99": np.asarray(high, dtype=np.float32).tolist(),
    }


class RobotWinStatePreprocessTest(unittest.TestCase):
    def test_robotwin_state_reorder_matches_training_concat_order(self):
        raw_env_state = np.arange(14, dtype=np.float32)

        train_order_state = reorder_robotwin_env_state_to_train(raw_env_state)

        np.testing.assert_array_equal(train_order_state, raw_env_state[ROBOTWIN_ENV_TO_TRAIN_STATE_INDICES])
        np.testing.assert_array_equal(
            reorder_robotwin_train_action_to_env(train_order_state),
            raw_env_state,
        )


    def test_robotwin_state_preprocess_reorders_before_normalizing(self):
        raw_env_state = np.arange(14, dtype=np.float32)
        state_stats = _stats(low=np.zeros(14), high=np.array([10.0] * 12 + [1.0, 1.0]))

        normalized_state = preprocess_robotwin_state_for_model(
            raw_env_state,
            state_stats,
            normalization_mode="min_max",
        )

        train_order_state = raw_env_state[ROBOTWIN_ENV_TO_TRAIN_STATE_INDICES]
        expected = np.zeros(14, dtype=np.float32)
        expected[:12] = 2.0 * train_order_state[:12] / 10.0 - 1.0
        expected[12:] = 1.0
        np.testing.assert_allclose(normalized_state, expected.reshape(1, 14))


    def test_robotwin_state_min_max_zeroes_degenerate_continuous_dims(self):
        train_order_state = np.arange(14, dtype=np.float32)
        low = np.zeros(14, dtype=np.float32)
        high = np.array([0.0] + [10.0] * 11 + [1.0, 1.0], dtype=np.float32)

        normalized_state = normalize_robotwin_train_state(
            train_order_state,
            _stats(low=low, high=high),
            normalization_mode="min_max",
        )

        self.assertEqual(normalized_state[0], 0.0)
        np.testing.assert_allclose(normalized_state[1:12], 2.0 * train_order_state[1:12] / 10.0 - 1.0)


    def test_robotwin_state_q99_clips_continuous_dims_like_training_normalizer(self):
        train_order_state = np.array([20.0] * 12 + [0.2, 0.8], dtype=np.float32)
        state_stats = _stats(low=np.zeros(14), high=np.array([10.0] * 12 + [1.0, 1.0]))

        normalized_state = normalize_robotwin_train_state(
            train_order_state,
            state_stats,
            normalization_mode="q99",
        )

        np.testing.assert_allclose(normalized_state[:12], np.ones(12, dtype=np.float32))
        np.testing.assert_array_equal(normalized_state[12:], np.array([0.0, 1.0], dtype=np.float32))

    def test_robotwin_state_q99_degenerate_continuous_dims_keep_raw_then_clip(self):
        train_order_state = np.array([0.25, 2.0] + [1.0] * 10 + [0.0, 1.0], dtype=np.float32)
        low = np.array([0.0, 0.0] + [0.0] * 10 + [0.0, 0.0], dtype=np.float32)
        high = np.array([0.0, 0.0] + [2.0] * 10 + [1.0, 1.0], dtype=np.float32)

        normalized_state = normalize_robotwin_train_state(
            train_order_state,
            _stats(low=low, high=high),
            normalization_mode="q99",
        )

        np.testing.assert_allclose(normalized_state[:2], np.array([0.25, 1.0], dtype=np.float32))


    def test_robotwin_state_preprocess_rejects_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "last dimension 14"):
            preprocess_robotwin_state_for_model(np.zeros(13, dtype=np.float32), _stats(np.zeros(14), np.ones(14)))

    def test_robotwin_use_state_auto_reads_training_include_state(self):
        self.assertTrue(
            resolve_robotwin_use_state(
                "auto",
                {"datasets": {"vla_data": {"include_state": True}}},
            )
        )
        self.assertFalse(
            resolve_robotwin_use_state(
                "auto",
                {"datasets": {"vla_data": {}}},
            )
        )

    def test_robotwin_use_state_can_be_forced(self):
        self.assertTrue(resolve_robotwin_use_state("true", {}))
        self.assertFalse(resolve_robotwin_use_state("false", {"datasets": {"vla_data": {"include_state": True}}}))
        with self.assertRaisesRegex(ValueError, "Invalid use_state"):
            resolve_robotwin_use_state("maybe", {})


if __name__ == "__main__":
    unittest.main()


def test_adaptive_ensemble_aligns_chunks_and_resets():
    from examples.Robotwin.eval_files.adaptive_ensemble import AdaptiveEnsembler

    ensemble = AdaptiveEnsembler(3, adaptive_ensemble_alpha=0.0)
    first = np.array([[1., 2.], [3., 4.], [5., 6.]])
    second = np.array([[7., 8.], [9., 10.], [11., 12.]])
    np.testing.assert_allclose(ensemble.ensemble_action(first), first[0])
    np.testing.assert_allclose(ensemble.ensemble_action(second), (first[1] + second[0]) / 2)
    ensemble.reset()
    np.testing.assert_allclose(ensemble.ensemble_action(second), second[0])
