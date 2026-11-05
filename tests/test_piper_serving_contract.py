import json
from pathlib import Path

import numpy as np
import pytest

from deployment.model_server.openpi_compat.eva_adapter import (
    EvaAgileXAdapter,
    _matrix_to_rot6d,
    _rot6d_to_matrix,
    _wxyz_to_matrix,
    joint_14d_to_eef_quat_16d,
)


def _make_adapter(tmp_path: Path) -> EvaAgileXAdapter:
    mask = [True] * 20
    # q01/q99 mirror min/max, matching stats written before real quantiles existed. Bounds are
    # then identical in either normalization mode, so these tests stay mode-agnostic.
    bounds = {"min": [-1.0] * 20, "max": [1.0] * 20, "q01": [-1.0] * 20, "q99": [1.0] * 20}
    stats = {
        "test": {
            "state": {**bounds, "mask": mask},
            "action": {**bounds, "mask": mask},
        }
    }
    (tmp_path / "dataset_statistics.json").write_text(json.dumps(stats))
    return EvaAgileXAdapter(tmp_path)


def _make_images(adapter: EvaAgileXAdapter) -> dict[str, np.ndarray]:
    return {cam: np.zeros((32, 32, 3), dtype=np.uint8) for cam in adapter.camera_obs_keys}


def test_qpos_and_eef_inputs_produce_identical_model_state(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    qpos = np.array(
        [0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.05, -0.1, 0.2, -0.3, 0.4, -0.2, 0.1, 0.01],
        dtype=np.float32,
    )
    eef = joint_14d_to_eef_quat_16d(qpos)
    images = _make_images(adapter)

    qpos_state = adapter.build_examples({"images": images, "qpos": qpos, "prompt": "test"})[0]["state"]
    eef_state = adapter.build_examples({"images": images, "eef": eef, "prompt": "test"})[0]["state"]
    legacy_state = adapter.build_examples({"images": images, "state": eef, "prompt": "test"})[0]["state"]

    np.testing.assert_allclose(qpos_state, eef_state, atol=1e-6)
    np.testing.assert_allclose(qpos_state, legacy_state, atol=1e-6)
    np.testing.assert_allclose(qpos_state[0, [9, 19]], qpos[[6, 13]], atol=1e-6)


def test_piper_fk_emits_finite_unit_quaternions() -> None:
    eef = joint_14d_to_eef_quat_16d(np.zeros(14, dtype=np.float32))

    assert eef.shape == (16,)
    assert np.isfinite(eef).all()
    np.testing.assert_allclose(np.linalg.norm(eef[3:7]), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(eef[11:15]), 1.0, atol=1e-6)


def test_rot6d_reconstruction_is_finite_and_orthogonal() -> None:
    quaternions = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    matrices = _rot6d_to_matrix(_matrix_to_rot6d(_wxyz_to_matrix(quaternions)))

    assert np.isfinite(matrices).all()
    identities = np.broadcast_to(np.eye(3), matrices.shape)
    np.testing.assert_allclose(matrices.swapaxes(-1, -2) @ matrices, identities, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(matrices), 1.0, atol=1e-6)


def test_observation_schema_rejects_ambiguous_or_invalid_state(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    images = _make_images(adapter)
    qpos = np.zeros(14, dtype=np.float32)
    eef = joint_14d_to_eef_quat_16d(qpos)

    with pytest.raises(ValueError, match="exactly one"):
        adapter.build_examples({"images": images, "qpos": qpos, "eef": eef})
    with pytest.raises(KeyError, match="missing robot state"):
        adapter.build_examples({"images": images})
    with pytest.raises(ValueError, match="must be 14D"):
        adapter.build_examples({"images": images, "qpos": np.zeros(13, dtype=np.float32)})
    with pytest.raises(ValueError, match="NaN/Inf"):
        bad_eef = eef.copy()
        bad_eef[0] = np.nan
        adapter.build_examples({"images": images, "eef": bad_eef})


def test_q99_uses_quantile_bounds_and_clips_state(tmp_path: Path) -> None:
    mask = [True] * 20
    stats = {
        "test": {
            "state": {
                "min": [-10.0] * 20, "max": [10.0] * 20,
                "q01": [-1.0] * 20, "q99": [1.0] * 20, "mask": mask,
            },
            "action": {
                "min": [-10.0] * 20, "max": [10.0] * 20,
                "q01": [-1.0] * 20, "q99": [1.0] * 20, "mask": mask,
            },
        }
    }
    (tmp_path / "dataset_statistics.json").write_text(json.dumps(stats))
    adapter = EvaAgileXAdapter(tmp_path, normalization_mode="q99")
    images = _make_images(adapter)

    example = adapter.build_examples({"images": images, "state": np.full(20, 2.0)})[0]
    np.testing.assert_allclose(example["state"], 1.0)

    normalized = np.full((1, 20), 0.5, dtype=np.float32)
    actions = adapter.postprocess_actions(normalized)
    np.testing.assert_allclose(actions[0, [0, 1, 2, 8, 9, 10]], 0.5)
    np.testing.assert_allclose(actions[0, [7, 15]], 0.5)


def test_normalization_mode_defaults_to_q99_and_rejects_unknown(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    assert adapter.normalization_mode == "q99"
    with pytest.raises(ValueError, match="normalization_mode"):
        EvaAgileXAdapter(tmp_path, normalization_mode="unknown")
