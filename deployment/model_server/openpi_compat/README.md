# MR0 OpenPI-compatible WebSocket server

This server uses the OpenPI flat-msgpack protocol: it sends metadata on connection, accepts observations, and returns an action chunk. It is separate from the envelope-based RoboTwin server.

## Start the server

```bash
python -m deployment.model_server.openpi_compat.server \
  --ckpt_path /path/to/run/checkpoints/model.pt \
  --port 9100 --normalization_mode q99
```

The run directory must contain the matching configuration and `dataset_statistics.json`. Select `q99` or `min_max` to match training. This adapter expects continuous grippers: for new AgileX training runs, set `MR0_NORMALIZATION_MODE=q99` (or `min_max`) on every node so the dataset registry applies the same normalization to grippers. A checkpoint trained with binary grippers needs a matching adapter. You can also use `CHECKPOINT_PATH=/path/to/model.pt bash deployment/model_server/openpi_compat/serve_eva.sh`; optional environment variables include `MR0_PYTHON`, `PORT`, `CUDA_DEVICES`, `UNNORM_KEY`, and `USE_BF16`.

## Observation and action format

The included `EvaAgileXAdapter` serves dual-arm policies trained with 20-dimensional xyz + rot6d + gripper actions. It is not the 14-dimensional RoboTwin adapter.

- Send `images` keyed by `cam_high`, `cam_left_wrist`, and `cam_right_wrist`, plus `prompt`.
- Supply exactly one state source: `qpos` (14 dimensions), `eef` (16 dimensions, wxyz quaternions), or `state` (14, 16, or 20 dimensions).
- Joint states use the bundled Piper forward kinematics. EEF states are converted from wxyz quaternions to the model's rot6d representation.
- Normalization uses the selected statistics bounds. Gripper commands remain continuous.
- Responses contain `actions` with shape `(T, 16)` in `[L_xyz, L_wxyz, L_gripper, R_xyz, R_wxyz, R_gripper]` order, plus server timing.

Read the connection metadata for camera keys and action format. For a different robot, provide an adapter with matching state conventions, action order, and normalization statistics.

## Local checks

```bash
python -m pytest tests/test_piper_serving_contract.py tests/test_policy_server_routing.py -q
```

The adapter tests exercise state conversion, normalization, and action output without moving a robot. Full model inference requires the matching weights and a CUDA runtime.
