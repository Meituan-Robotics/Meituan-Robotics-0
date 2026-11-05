"""Regression checks for the public policy server's message contract."""
from unittest.mock import Mock

import pytest

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


@pytest.mark.parametrize("envelope", [False, True])
def test_inference_routes_flat_and_enveloped_payloads(envelope):
    policy = Mock()
    policy.predict_action.return_value = {"normalized_actions": [1, 2]}
    server = WebsocketPolicyServer(policy)
    payload = {"examples": [{"lang": "place the block"}]}
    message = {"type": "infer", "request_id": "42", "payload": payload} if envelope else payload
    reply = server._route_message(message)
    policy.predict_action.assert_called_once_with(**payload)
    assert reply["ok"] is True
    assert reply["data"] == policy.predict_action.return_value
    assert reply["request_id"] == ("42" if envelope else "default")


@pytest.mark.parametrize("message", [None, [], {"type": "infer", "payload": []}])
def test_bad_messages_return_errors_without_calling_policy(message):
    policy = Mock()
    reply = WebsocketPolicyServer(policy)._route_message(message)
    assert reply["ok"] is False
    policy.predict_action.assert_not_called()


def test_ping_and_policy_errors_preserve_request_id():
    policy = Mock()
    server = WebsocketPolicyServer(policy)
    assert server._route_message({"type": "ping", "request_id": "ping"})["request_id"] == "ping"
    policy.predict_action.assert_not_called()
    policy.predict_action.side_effect = ValueError("missing images")
    reply = server._route_message({"request_id": "bad", "examples": []})
    assert reply["ok"] is False
    assert reply["request_id"] == "bad"
    assert reply["error"]["message"] == "missing images"


def test_websocket_client_server_roundtrip(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    import os
    import asyncio
    import numpy as np
    import websockets.asyncio.server
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    policy = Mock()
    expected = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    policy.predict_action.return_value = {"normalized_actions": expected}
    server = WebsocketPolicyServer(policy, metadata={"server": "mr0-test"})

    async def exercise():
        async with websockets.asyncio.server.serve(server._handler, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]

            def request():
                client = WebsocketClientPolicy(host="127.0.0.1", port=port)
                try:
                    assert client.get_server_metadata() == {"server": "mr0-test"}
                    reply = client.predict_action({"examples": [{"lang": "place", "state": np.zeros((1, 4), dtype=np.float32)}]})
                    assert reply["ok"]
                    np.testing.assert_array_equal(reply["data"]["normalized_actions"], expected)
                finally:
                    client.close()

            await asyncio.wait_for(asyncio.to_thread(request), timeout=15)

    asyncio.run(exercise())
    assert policy.predict_action.call_count == 1
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:1"
