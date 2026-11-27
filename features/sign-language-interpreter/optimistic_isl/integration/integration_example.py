"""Example integration showing how Optimistic smart glasses display code
consumes the decoupled local JSON / WebSocket recognition interface.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from integration.output_interface import OptimisticOutputInterface
from integration.websocket_server import OptimisticWebSocketServer


def python_api_example() -> None:
    """Direct in-process Python API usage for Raspberry Pi display loop."""
    output_hub = OptimisticOutputInterface()

    def render_on_glasses_hud(payload: dict) -> None:
        if payload["recognized"]:
            print(
                f"[HUD DISPLAY] {payload['gesture']} "
                f"({payload['confidence']*100:.1f}%)"
            )
        else:
            print("[HUD DISPLAY] ...")

    output_hub.subscribe(render_on_glasses_hud)
    output_hub.publish(
        {
            "gesture": "HELLO",
            "confidence": 0.94,
            "recognized": True,
        }
    )
    print("Latest JSON:", output_hub.get_latest_json())


async def websocket_client_example(uri: str) -> None:
    """Connect to the local WebSocket server and print Optimistic HUD updates."""
    async with websockets.connect(uri) as ws:
        print(f"Connected to {uri}. Waiting for gesture updates...")
        for _ in range(5):
            msg = await ws.recv()
            data = json.loads(msg)
            print("Received glasses payload:", data)


if __name__ == "__main__":
    python_api_example()

    # Self-contained local WebSocket round-trip demonstration (no camera needed)
    hub = OptimisticOutputInterface()
    srv = OptimisticWebSocketServer(hub, host="127.0.0.1", port=0)
    srv.start()

    async def _demo() -> None:
        uri = f"ws://127.0.0.1:{srv.actual_port}"
        async with websockets.connect(uri) as ws:
            initial = json.loads(await ws.recv())
            print("WS initial:", initial)
            hub.publish({"gesture": "THANK YOU", "confidence": 0.92, "recognized": True})
            updated = json.loads(await ws.recv())
            print("WS updated:", updated)

    asyncio.run(_demo())
    srv.stop()
