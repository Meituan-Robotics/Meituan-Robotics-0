"""OpenPI adapter for dual-arm xyz + rot6d + gripper (20D) policies.

Accepts Piper joint states (14D), canonical EEF quaternion states (16D),
or model-space rot6d states (20D). Uses checkpoint q99 or min_max bounds
and returns 16D EEF actions with continuous grippers and wxyz rotations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# data_config.action_keys order (model output, 20-dim). Stats are saved in
# this same order for AgileX runs.
_MODEL_SLICES = {
    "left_xyz":      slice(0, 3),
    "left_rot6d":    slice(3, 9),
    "left_gripper":  slice(9, 10),
    "right_xyz":     slice(10, 13),
    "right_rot6d":   slice(13, 19),
    "right_gripper": slice(19, 20),
}
# Positions of binary gripper dims in the 20-dim model layout.
_MODEL_GRIPPER_IDXS = (9, 19)
# Positions of binary gripper dims in the 16-dim canonical EEF layout.
# Must match AgileXFoldEef6dDataConfig.GRIPPER_BINARY_THRESHOLD. Piper gripper
# joint angle is in [0, 0.1] rad; the training-time threshold splits closed/open.
_STATE_GRIPPER_THRESHOLD = 0.04
_STATE_INPUT_KEYS = ("eef", "qpos", "state")


class PiperDhChain:
    """Single-arm Piper DH forward kinematics used by the deployment client contract."""

    def __init__(self) -> None:
        self._a = (0.0, 0.0, 285.03, -21.98, 0.0, 0.0)
        self._alpha = (0.0, -math.pi / 2.0, 0.0, math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0)
        # dh_is_offset=0x01, matching the offline training-data conversion.
        self._theta = (0.0, -math.pi * 172.22 / 180.0, -math.pi * 102.78 / 180.0, 0.0, 0.0, 0.0)
        self._d = (123.0, 0.0, 0.0, 250.75, 0.0, 91.0)

    def fk_arm(self, qpos_7d: np.ndarray) -> np.ndarray:
        """Convert six arm joints plus gripper to xyz, wxyz, gripper."""
        qpos = np.asarray(qpos_7d, dtype=np.float64).reshape(-1)
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise ValueError(f"Piper arm qpos must be finite shape (7,), got {qpos.shape}")

        transform = np.eye(4, dtype=np.float64)
        for i in range(6):
            ca, sa = math.cos(self._alpha[i]), math.sin(self._alpha[i])
            ct = math.cos(float(qpos[i]) + self._theta[i])
            st = math.sin(float(qpos[i]) + self._theta[i])
            link = np.array(
                [
                    [ct, -st, 0.0, self._a[i]],
                    [st * ca, ct * ca, -sa, -sa * self._d[i]],
                    [st * sa, ct * sa, ca, ca * self._d[i]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            transform = transform @ link

        quat = _matrix_to_wxyz(transform[None, :3, :3])[0]
        return np.array(
            [
                transform[0, 3] / 1000.0,
                transform[1, 3] / 1000.0,
                transform[2, 3] / 1000.0,
                *quat,
                qpos[6],
            ],
            dtype=np.float32,
        )


def joint_14d_to_eef_quat_16d(joint_14d: np.ndarray) -> np.ndarray:
    """Convert dual-arm Piper qpos to canonical EEF quaternion state."""
    joint = np.asarray(joint_14d, dtype=np.float64).reshape(-1)
    if joint.shape != (14,) or not np.isfinite(joint).all():
        raise ValueError(f"Piper qpos must be finite shape (14,), got {joint.shape}")
    chain = PiperDhChain()
    return np.concatenate((chain.fk_arm(joint[:7]), chain.fk_arm(joint[7:]))).astype(np.float32)


def _rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Robust Gram-Schmidt 6D -> proper rotation matrix. d6 shape (..., 6)."""
    d6 = np.asarray(d6, dtype=np.float32)
    if d6.shape[-1] != 6 or not np.isfinite(d6).all():
        raise ValueError(f"rot6d must be finite with last dim 6, got {d6.shape}")

    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    a1_norm = np.linalg.norm(a1, axis=-1, keepdims=True)
    fallback_b1 = np.broadcast_to(np.array([1.0, 0.0, 0.0], dtype=np.float32), a1.shape)
    b1 = np.where(a1_norm > 1e-6, a1 / np.maximum(a1_norm, 1e-6), fallback_b1)

    b2_raw = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2_norm = np.linalg.norm(b2_raw, axis=-1, keepdims=True)
    # For zero/collinear model outputs, choose the canonical axis least aligned
    # with b1 and orthogonalize it instead of emitting a singular matrix.
    fallback_axis = np.eye(3, dtype=np.float32)[np.argmin(np.abs(b1), axis=-1)]
    fallback_b2 = fallback_axis - np.sum(b1 * fallback_axis, axis=-1, keepdims=True) * b1
    fallback_b2 /= np.maximum(np.linalg.norm(fallback_b2, axis=-1, keepdims=True), 1e-6)
    b2 = np.where(b2_norm > 1e-6, b2_raw / np.maximum(b2_norm, 1e-6), fallback_b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)


def _wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Batched wxyz quaternion -> 3x3 rotation matrix. q shape (..., 4)."""
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], axis=-1)
    return out.reshape(q.shape[:-1] + (3, 3)).astype(np.float32)


def _matrix_to_rot6d(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> 6D representation (first two columns flat)."""
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1).astype(np.float32)


def _matrix_to_wxyz(m: np.ndarray) -> np.ndarray:
    """Batched 3x3 rotation matrix -> wxyz quaternion. m shape (..., 3, 3)."""
    flat = m.reshape(-1, 3, 3)
    out = np.zeros((flat.shape[0], 4), dtype=np.float32)
    for i in range(flat.shape[0]):
        r = flat[i]
        trace = r[0, 0] + r[1, 1] + r[2, 2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
        out[i] = (w, x, y, z)
    return out.reshape(m.shape[:-2] + (4,))


class EvaAgileXAdapter:
    """Per-checkpoint adapter: caches stats and eva-client wiring details.

    Args:
        ckpt_dir: directory containing `dataset_statistics.json` (run_dir).
        unnorm_key: top-level key in stats (e.g. "new_embodiment"). If None and
            only one key is present, it is auto-resolved.
        camera_obs_keys: ordered camera names eva-client will send in
            `observation["images"]`, mapped to model image slots in order.
            Defaults to `("cam_high", "cam_left_wrist", "cam_right_wrist")`,
            matching AgileX modality.json (primary_image / left_wrist /
            right_wrist).
    """

    def __init__(
        self,
        ckpt_dir: str | Path,
        unnorm_key: str | None = None,
        normalization_mode: str = "q99",
        camera_obs_keys: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist"),
    ) -> None:
        ckpt_dir = Path(ckpt_dir)
        stats_path = ckpt_dir / "dataset_statistics.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"dataset_statistics.json not found in {ckpt_dir}")
        with open(stats_path) as f:
            all_stats = json.load(f)

        if unnorm_key is None:
            if len(all_stats) != 1:
                raise ValueError(
                    f"unnorm_key required: {sorted(all_stats.keys())}"
                )
            unnorm_key = next(iter(all_stats))
        if unnorm_key not in all_stats:
            raise KeyError(f"unnorm_key '{unnorm_key}' not in {sorted(all_stats)}")
        self.unnorm_key = unnorm_key

        if normalization_mode not in {"min_max", "q99"}:
            raise ValueError(
                f"normalization_mode must be 'min_max' or 'q99', got {normalization_mode!r}"
            )
        self.normalization_mode = normalization_mode
        low_key, high_key = ("q01", "q99") if normalization_mode == "q99" else ("min", "max")

        action_stats = all_stats[unnorm_key]["action"]
        self._action_max = np.asarray(action_stats[high_key], dtype=np.float32)
        self._action_min = np.asarray(action_stats[low_key], dtype=np.float32)
        self._action_mask = np.asarray(
            action_stats.get("mask", np.ones_like(self._action_max, dtype=bool)),
            dtype=bool,
        )
        if self._action_max.shape[0] != 20:
            raise ValueError(
                f"Expected 20-dim action stats, got {self._action_max.shape[0]} for "
                f"key '{unnorm_key}'"
            )
        # Gripper values remain continuous and participate in min-max unnormalization.
        for idx in _MODEL_GRIPPER_IDXS:
            self._action_mask[idx] = True

        state_stats = all_stats[unnorm_key].get("state")
        if state_stats is not None:
            self._state_max = np.asarray(state_stats[high_key], dtype=np.float32)
            self._state_min = np.asarray(state_stats[low_key], dtype=np.float32)
            raw_state_mask = state_stats.get("mask")
            if raw_state_mask is None:
                # Continuous state dimensions, including both grippers, use min-max normalization.
                self._state_mask = np.ones(self._state_max.shape[0], dtype=bool)
            else:
                self._state_mask = np.asarray(raw_state_mask, dtype=bool)
        else:
            self._state_max = self._state_min = self._state_mask = None

        self.camera_obs_keys = tuple(camera_obs_keys)

    # ── obs prep ───────────────────────────────────────────────────────────

    def build_examples(self, obs: dict) -> list[dict]:
        """Map a qpos- or EEF-based eva-client observation to one model example."""
        if not isinstance(obs, dict):
            raise ValueError(f"observation must be a dict, got {type(obs)}")

        images_dict = obs.get("images") or obs.get("image") or {}
        if not isinstance(images_dict, dict):
            raise ValueError(f"observation['images'] must be a dict, got {type(images_dict)}")
        image_list = []
        for cam in self.camera_obs_keys:
            if cam not in images_dict:
                raise KeyError(
                    f"missing camera '{cam}' in observation; got {list(images_dict.keys())}"
                )
            image = np.asarray(images_dict[cam])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"camera '{cam}' must have shape (H, W, 3), got {image.shape}")
            image_list.append(image)

        prompt = obs.get("prompt") or obs.get("lang") or ""
        example = {"image": image_list, "lang": str(prompt)}

        provided_state_keys = [key for key in _STATE_INPUT_KEYS if obs.get(key) is not None]
        if len(provided_state_keys) > 1:
            raise ValueError(f"provide exactly one of {_STATE_INPUT_KEYS}, got {provided_state_keys}")
        if self._state_max is not None and not provided_state_keys:
            raise KeyError(f"missing robot state; provide exactly one of {_STATE_INPUT_KEYS}")

        if provided_state_keys and self._state_max is not None:
            state_key = provided_state_keys[0]
            state_arr = np.asarray(obs[state_key], dtype=np.float32).reshape(-1)
            if not np.isfinite(state_arr).all():
                raise ValueError(f"observation['{state_key}'] contains NaN/Inf")
            if state_key == "qpos":
                if state_arr.shape[0] != 14:
                    raise ValueError(f"observation['qpos'] must be 14D, got {state_arr.shape[0]}D")
                state_arr = joint_14d_to_eef_quat_16d(state_arr)
            elif state_key == "eef" and state_arr.shape[0] != 16:
                raise ValueError(f"observation['eef'] must be 16D, got {state_arr.shape[0]}D")
            elif state_key == "state" and state_arr.shape[0] == 14:
                state_arr = joint_14d_to_eef_quat_16d(state_arr)
            if state_arr.shape[0] == 16:
                # Canonical EEF quaternion state -> model 20-dim rot6d.
                state_arr = self._eva_state_to_model_state(state_arr)
            if state_arr.shape[0] != self._state_max.shape[0]:
                raise ValueError(
                    f"observation['{state_key}'] dim {state_arr.shape[0]} != stats dim "
                    f"{self._state_max.shape[0]} (expected qpos 14D, EEF quaternion 16D, "
                    "or model rot6d 20D)"
                )
            # Stats are in model order. q99 matches training by clipping outliers.
            normed = self._minmax_normalize(state_arr, self._state_min, self._state_max, self._state_mask)
            if self.normalization_mode == "q99":
                normed = np.clip(normed, -1.0, 1.0)
            example["state"] = normed[None, :]
        return [example]

    @staticmethod
    def _eva_state_to_model_state(state_16: np.ndarray) -> np.ndarray:
        """Convert eva-client 16-dim canonical EEF state -> 20-dim model rot6d state.

        Input layout:  [L_xyz(3), L_wxyz(4), L_grip(1), R_xyz(3), R_wxyz(4), R_grip(1)]
        Output layout: [L_xyz(3), L_rot6d(6), L_grip(1), R_xyz(3), R_rot6d(6), R_grip(1)]
        """
        out = np.zeros(20, dtype=np.float32)
        l_xyz = state_16[0:3]
        l_wxyz = state_16[3:7]
        l_grip = state_16[7:8]
        r_xyz = state_16[8:11]
        r_wxyz = state_16[11:15]
        r_grip = state_16[15:16]
        l_rot6d = _matrix_to_rot6d(_wxyz_to_matrix(l_wxyz))
        r_rot6d = _matrix_to_rot6d(_wxyz_to_matrix(r_wxyz))
        out[0:3] = l_xyz
        out[3:9] = l_rot6d
        out[9:10] = l_grip
        out[10:13] = r_xyz
        out[13:19] = r_rot6d
        out[19:20] = r_grip
        return out

    # ── action post-processing ─────────────────────────────────────────────

    def postprocess_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        """Take model output (T, 20) in [-1,1] -> canonical eva-client (T, 16).

        Output layout: [L_xyz(3), L_wxyz(4), L_grip(1), R_xyz(3), R_wxyz(4), R_grip(1)].
        Gripper dims (positions 7 and 15) remain continuous after min-max unnormalization.
        """
        if normalized_actions.ndim != 2 or normalized_actions.shape[-1] != 20:
            raise ValueError(
                f"expected (T, 20) normalized actions, got {normalized_actions.shape}"
            )
        clipped = np.clip(normalized_actions, -1.0, 1.0)
        # Stats are already in model order, including continuous gripper dimensions.
        unnorm = self._minmax_unnormalize(
            clipped, self._action_min, self._action_max, self._action_mask
        )

        out = np.empty((unnorm.shape[0], 16), dtype=np.float32)
        # Left arm.
        l_xyz = unnorm[:, _MODEL_SLICES["left_xyz"]]
        l_rot6d = unnorm[:, _MODEL_SLICES["left_rot6d"]]
        l_grip = unnorm[:, _MODEL_SLICES["left_gripper"]]
        l_wxyz = _matrix_to_wxyz(_rot6d_to_matrix(l_rot6d))
        out[:, 0:3] = l_xyz
        out[:, 3:7] = l_wxyz
        out[:, 7:8] = l_grip

        # Right arm.
        r_xyz = unnorm[:, _MODEL_SLICES["right_xyz"]]
        r_rot6d = unnorm[:, _MODEL_SLICES["right_rot6d"]]
        r_grip = unnorm[:, _MODEL_SLICES["right_gripper"]]
        r_wxyz = _matrix_to_wxyz(_rot6d_to_matrix(r_rot6d))
        out[:, 8:11] = r_xyz
        out[:, 11:15] = r_wxyz
        out[:, 15:16] = r_grip

        return out

    @staticmethod
    def _minmax_unnormalize(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.where(mask, 0.5 * (x + 1.0) * (hi - lo) + lo, x)

    @staticmethod
    def _minmax_normalize(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        denom = np.where(np.abs(hi - lo) > 1e-8, hi - lo, 1.0)
        return np.where(mask, 2.0 * (x - lo) / denom - 1.0, x)
