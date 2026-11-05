"""Send a synthetic MR0 observation to a running policy server.

Run with python -m deployment.model_server.tools.debug_server_policy --help.
Match image sizes, camera count, and state dimension to the checkpoint.
"""
import argparse
import logging

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def _build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--state_dim", type=int, default=14)
    parser.add_argument("--cameras", type=int, default=3)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--prompt", default="Pick up the red block.")
    return parser


def _main():
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    try:
        logging.info("Server metadata: %s", client.get_server_metadata())
        example = {
            "image": [np.zeros((args.height, args.width, 3), dtype=np.uint8) for _ in range(args.cameras)],
            "lang": args.prompt,
            "state": np.zeros((1, args.state_dim), dtype=np.float32),
        }
        reply = client.predict_action({"examples": [example]})
        if not reply.get("ok"):
            raise RuntimeError(f"Policy rejected the observation: {reply.get('error')}")
        actions = np.asarray(reply["data"]["normalized_actions"])
        logging.info("Received actions: shape=%s dtype=%s", actions.shape, actions.dtype)
    finally:
        client.close()


if __name__ == "__main__":
    _main()
