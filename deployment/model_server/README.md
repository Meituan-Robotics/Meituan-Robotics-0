# MR0 policy servers

Run these commands from the repository root after installation.

## RoboTwin policy server

```bash
python -m deployment.model_server.server_policy \
  --ckpt_path /path/to/run/checkpoints/model.pt \
  --port 10093 --use_bf16
```

The checkpoint's run directory must contain its configuration and `dataset_statistics.json`.

Use `python -m deployment.model_server.tools.debug_server_policy --help` for the client options, or follow the [RoboTwin evaluation guide](../../examples/Robotwin/README.md).

## OpenPI-compatible server

See the [OpenPI server guide](openpi_compat/README.md) for the dual-arm EEF adapter and message format.
