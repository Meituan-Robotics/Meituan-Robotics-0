"""End-to-end smoke test for the OpenPI-compat server with a real ckpt.

Stages:
    1. Load policy + adapter, call adapter+policy in-process (no socket).
    2. Start the WS server on a port, drive it with a msgpack client, verify
       metadata frame + actions frame.

Validation:
    - actions returned have shape (T, 16) and finite values
    - rotation block per arm has unit norm (canonical wxyz)
    - gripper dims (cols 7, 15) ∈ {0, 1}
    - xyz coords land within the stats min/max range (safety, not strict equality)

Run:
    python -m deployment.model_server.openpi_compat.e2e_test \\
        --ckpt_path .../<run>/checkpoints/steps_10000_pytorch_model.pt
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _make_obs(adapter):
    rng = np.random.default_rng(42)
    images = {
        cam: rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
        for cam in adapter.camera_obs_keys
    }
    state_16 = np.zeros(16, dtype=np.float32)
    state_16[3:7] = [1, 0, 0, 0]
    state_16[11:15] = [1, 0, 0, 0]
    state_16[7] = 0.05
    state_16[15] = 0.6
    return {
        "images": images,
        "prompt": "fold the cloth with both arms",
        "state": state_16,
    }


def _build_examples_and_validate_state(adapter, obs: dict, expected_grip: tuple[float, float]):
    examples = adapter.build_examples(obs)
    state = examples[0].get("state")
    assert state is not None, "adapter did not emit normalized state"
    assert state.shape == (1, 20), f"expected normalized state shape (1,20), got {state.shape}"
    grip = state[0, [9, 19]]
    expected = np.asarray(expected_grip, dtype=np.float32)
    np.testing.assert_array_equal(
        grip,
        expected,
        err_msg=f"adapter state gripper threshold mismatch: got {grip}, expected {expected}",
    )
    return examples


def _validate_actions(actions: np.ndarray, adapter) -> dict:
    assert actions.ndim == 2, f"expected 2D (T,16), got {actions.shape}"
    T, D = actions.shape
    assert D == 16, f"expected 16-dim canonical EEF, got D={D}"
    assert np.isfinite(actions).all(), "actions contain NaN/Inf"

    l_wxyz = actions[:, 3:7]
    r_wxyz = actions[:, 11:15]
    l_norms = np.linalg.norm(l_wxyz, axis=-1)
    r_norms = np.linalg.norm(r_wxyz, axis=-1)
    assert np.allclose(l_norms, 1.0, atol=1e-3), f"L wxyz not unit: {l_norms}"
    assert np.allclose(r_norms, 1.0, atol=1e-3), f"R wxyz not unit: {r_norms}"

    l_grip = actions[:, 7]
    r_grip = actions[:, 15]
    assert np.all((l_grip == 0.0) | (l_grip == 1.0)), f"L grip non-binary: {l_grip}"
    assert np.all((r_grip == 0.0) | (r_grip == 1.0)), f"R grip non-binary: {r_grip}"

    # XYZ in stats range (per-arm). Stats are in model order: L_xyz @0:3, R_xyz @10:13.
    l_lo = adapter._action_min[0:3]; l_hi = adapter._action_max[0:3]
    r_lo = adapter._action_min[10:13]; r_hi = adapter._action_max[10:13]
    l_xyz = actions[:, 0:3]; r_xyz = actions[:, 8:11]
    assert (l_xyz >= l_lo - 1e-4).all() and (l_xyz <= l_hi + 1e-4).all(), \
        f"L_xyz out of stats range: min {l_xyz.min(0)} max {l_xyz.max(0)} vs [{l_lo}, {l_hi}]"
    assert (r_xyz >= r_lo - 1e-4).all() and (r_xyz <= r_hi + 1e-4).all(), \
        f"R_xyz out of stats range: min {r_xyz.min(0)} max {r_xyz.max(0)} vs [{r_lo}, {r_hi}]"

    return {
        "T": T,
        "L_grip_unique": sorted(set(l_grip.tolist())),
        "R_grip_unique": sorted(set(r_grip.tolist())),
        "L_xyz_range": (l_xyz.min(0).tolist(), l_xyz.max(0).tolist()),
        "R_xyz_range": (r_xyz.min(0).tolist(), r_xyz.max(0).tolist()),
        "L_wxyz_norm": (float(l_norms.min()), float(l_norms.max())),
        "R_wxyz_norm": (float(r_norms.min()), float(r_norms.max())),
    }


def stage1_inproc(ckpt_path: Path):
    from deployment.model_server.openpi_compat.eva_adapter import EvaAgileXAdapter
    from mr0.model.framework.base_framework import baseframework

    print("\n" + "=" * 70)
    print("STAGE 1: in-process adapter + policy + postprocess (no socket)")
    print("=" * 70)

    run_dir = ckpt_path.parents[1]
    adapter = EvaAgileXAdapter(ckpt_dir=run_dir)
    print(f"  adapter ok: unnorm_key={adapter.unnorm_key} cams={adapter.camera_obs_keys}")

    print("  loading policy from", ckpt_path)
    t0 = time.monotonic()
    vla = baseframework.from_pretrained(str(ckpt_path)).to("cuda").eval()
    print(f"  policy loaded in {time.monotonic() - t0:.1f}s")

    obs = _make_obs(adapter)
    examples = _build_examples_and_validate_state(adapter, obs, expected_grip=(1.0, 1.0))
    print("  example keys:", list(examples[0].keys()),
          "image[0].shape:", np.asarray(examples[0]["image"][0]).shape,
          "state.shape:", examples[0]["state"].shape)

    print("  running predict_action ...")
    t0 = time.monotonic()
    out = vla.predict_action(examples=examples)
    print(f"  predict_action {(time.monotonic() - t0) * 1000:.0f} ms;"
          f" normalized_actions.shape={out['normalized_actions'].shape}")

    normalized = out["normalized_actions"][0]
    print(f"  normalized stats: min={normalized.min():.3f} max={normalized.max():.3f} "
          f"mean={normalized.mean():.3f}")

    actions = adapter.postprocess_actions(normalized)
    info = _validate_actions(actions, adapter)
    print(f"  ✓ canonical actions: T={info['T']} dim=16")
    print(f"    L grip values seen: {info['L_grip_unique']}")
    print(f"    R grip values seen: {info['R_grip_unique']}")
    print(f"    L_xyz range: {info['L_xyz_range']}")
    print(f"    R_xyz range: {info['R_xyz_range']}")
    print(f"    L wxyz unit norm: {info['L_wxyz_norm']}")
    print(f"    R wxyz unit norm: {info['R_wxyz_norm']}")
    return vla, adapter


async def _client_roundtrip(host: str, port: int, obs: dict):
    import websockets.asyncio.client
    from deployment.model_server.tools import msgpack_numpy

    packer = msgpack_numpy.Packer()
    async with websockets.asyncio.client.connect(f"ws://{host}:{port}", max_size=None) as ws:
        meta = msgpack_numpy.unpackb(await ws.recv())
        print("  ↳ metadata received:", {k: meta[k] for k in
            ("server", "unnorm_key", "action_format", "camera_obs_keys") if k in meta})
        await ws.send(packer.pack(obs))
        resp = msgpack_numpy.unpackb(await ws.recv())
        return meta, resp


def stage2_ws(vla, adapter, port: int):
    from deployment.model_server.openpi_compat.server import OpenPiCompatServer
    import threading

    print("\n" + "=" * 70)
    print(f"STAGE 2: WebSocket roundtrip on 127.0.0.1:{port}")
    print("=" * 70)

    metadata = {
        "server": "mr0-openpi-compat",
        "unnorm_key": adapter.unnorm_key,
        "action_format": {"arms": "dual", "rotation": "wxyz", "include_gripper": True, "total_dim": 16},
        "camera_obs_keys": list(adapter.camera_obs_keys),
    }
    server = OpenPiCompatServer(
        policy=vla, adapter=adapter, host="127.0.0.1", port=port, metadata=metadata,
    )

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(2.0)  # let asyncio loop warm up
    print("  server started")

    obs = _make_obs(adapter)
    _build_examples_and_validate_state(adapter, obs, expected_grip=(1.0, 1.0))
    t0 = time.monotonic()
    meta, resp = asyncio.run(_client_roundtrip("127.0.0.1", port, obs))
    print(f"  ↳ roundtrip {(time.monotonic() - t0) * 1000:.0f} ms")

    actions = resp["actions"]
    timing = resp.get("server_timing", {})
    print(f"  ↳ actions shape: {actions.shape} dtype: {actions.dtype}, server infer_ms={timing.get('infer_ms', '?'):.0f}")

    info = _validate_actions(actions, adapter)
    print(f"  ✓ canonical actions over wire: T={info['T']}")
    print(f"    L grip unique: {info['L_grip_unique']}")
    print(f"    R grip unique: {info['R_grip_unique']}")

    # second turn — same connection in client closed already, open another
    obs2 = _make_obs(adapter)
    obs2["state"][7] = 0.08  # L grip > 0.04 (open)
    obs2["state"][15] = 0.01 # R grip < 0.04 (closed)
    _build_examples_and_validate_state(adapter, obs2, expected_grip=(1.0, 0.0))
    meta2, resp2 = asyncio.run(_client_roundtrip("127.0.0.1", port, obs2))
    info2 = _validate_actions(resp2["actions"], adapter)
    print(f"  ✓ second roundtrip ok, server infer_ms={resp2['server_timing']['infer_ms']:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=9123)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    ckpt_path = Path(args.ckpt_path)
    assert ckpt_path.exists(), f"ckpt not found: {ckpt_path}"

    vla, adapter = stage1_inproc(ckpt_path)
    stage2_ws(vla, adapter, args.port)
    print("\nALL E2E CHECKS PASSED")


if __name__ == "__main__":
    main()
