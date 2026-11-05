"""OpenPI-compatible WebSocket policy server for eva-client.

Differences vs. ``deployment.model_server.server_policy``:
 - speaks the OpenPI flat-msgpack protocol (push metadata on connect, then
   receive a flat ``{state, images, prompt, ...}`` dict, return a flat dict
   containing ``{actions: ndarray (T, D)}``).
 - bundles AgileX-specific output canonicalization (rot6d -> wxyz quaternion,
   stats reorder) so eva-client can use ``rotation: wxyz`` without modification.

Usage::

    python -m deployment.model_server.openpi_compat.server \\
        --ckpt_path /path/to/<run>/checkpoints/steps_60000_pytorch_model.pt \\
        --port 9100
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import time
import traceback
from pathlib import Path

import torch
import websockets.asyncio.server
import websockets.frames

from deployment.model_server.openpi_compat.eva_adapter import EvaAgileXAdapter
from deployment.model_server.tools import msgpack_numpy
from mr0.model.framework.base_framework import baseframework

logger = logging.getLogger(__name__)


class OpenPiCompatServer:
    def __init__(
        self,
        policy,
        adapter: EvaAgileXAdapter,
        host: str = "0.0.0.0",
        port: int = 9100,
        metadata: dict | None = None,
        cfg_beta: float | None = None,
        cfg_suffix: str = "Advantage: positive",
    ) -> None:
        self._policy = policy
        self._adapter = adapter
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._cfg_beta = cfg_beta
        self._cfg_suffix = cfg_suffix

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=30,
            ping_timeout=600,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                t0 = time.monotonic()
                actions = await asyncio.to_thread(self._infer, obs)
                infer_ms = (time.monotonic() - t0) * 1000.0
                response = {"actions": actions, "server_timing": {"infer_ms": infer_ms}}
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                tb = traceback.format_exc()
                logger.exception("inference error")
                try:
                    await websocket.send(tb)
                except Exception:
                    pass
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise

    def _infer(self, obs: dict):
        examples = self._adapter.build_examples(obs)
        if self._cfg_beta is None:
            out = self._policy.predict_action(examples=examples)
        else:
            out = self._policy.predict_action_cfg(
                examples=examples,
                beta=self._cfg_beta,
                suffix=self._cfg_suffix,
            )
        normalized = out["normalized_actions"][0]  # (T, 20)
        return self._adapter.postprocess_actions(normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help=".pt checkpoint path; the run dir must hold dataset_statistics.json")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--normalization_mode", choices=("min_max", "q99"), default="q99",
                        help="Statistics bounds used for state/action scaling.")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--cfg_beta", type=float, default=None,
                        help="Enable final-action CFG with this guidance scale.")
    parser.add_argument("--cfg_suffix", type=str, default="Advantage: positive")
    parser.add_argument("--num_inference_timesteps", type=int, default=None,
                        help="Override the action model Euler denoising step count.")
    args = parser.parse_args()
    if args.num_inference_timesteps is not None and args.num_inference_timesteps < 1:
        parser.error("--num_inference_timesteps must be at least 1")

    logging.basicConfig(level=logging.INFO, force=True)

    ckpt_path = Path(args.ckpt_path)
    run_dir = ckpt_path.parents[1]
    adapter = EvaAgileXAdapter(
        ckpt_dir=run_dir,
        unnorm_key=args.unnorm_key,
        normalization_mode=args.normalization_mode,
    )

    vla = baseframework.from_pretrained(str(ckpt_path))
    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()
    if args.num_inference_timesteps is not None:
        vla.action_model.num_inference_timesteps = args.num_inference_timesteps
        logger.info("Overriding action-model denoising steps to %d", args.num_inference_timesteps)

    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        ip = args.host
    logger.info("OpenPI-compat server starting on %s:%d (host=%s)", ip, args.port, hostname)

    metadata = {
        "server": "mr0-openpi-compat",
        "ckpt": str(ckpt_path),
        "unnorm_key": adapter.unnorm_key,
        "normalization_mode": adapter.normalization_mode,
        "horizon": vla.action_model.action_horizon,
        "cfg": {
            "enabled": args.cfg_beta is not None,
            "beta": args.cfg_beta,
            "suffix": args.cfg_suffix if args.cfg_beta is not None else None,
        },
        "observation_format": {
            "state_inputs": {
                "qpos": {"total_dim": 14},
                "eef": {"rotation": "wxyz", "total_dim": 16},
                "state": {"accepted_dims": [14, 16, 20]},
            },
            "state_input_rule": "exactly_one",
        },
        "action_format": {
            "arms": "dual",
            "rotation": "wxyz",
            "include_gripper": True,
            "total_dim": 16,
        },
        "camera_obs_keys": list(adapter.camera_obs_keys),
    }
    server = OpenPiCompatServer(
        policy=vla,
        adapter=adapter,
        host=args.host,
        port=args.port,
        metadata=metadata,
        cfg_beta=args.cfg_beta,
        cfg_suffix=args.cfg_suffix,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
