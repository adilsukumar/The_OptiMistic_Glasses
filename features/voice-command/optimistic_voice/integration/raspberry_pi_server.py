"""Raspberry Pi local status & companion bridge server."""

from __future__ import annotations

import asyncio
import json
from typing import Callable


class RaspberryPiBridgeServer:
    """Optional local WebSocket bridge on the Pi for HUD/status integration."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8766,
        on_status_request: Callable[[], dict] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.on_status_request = on_status_request or (lambda: {"status": "running"})

    async def _handler(self, websocket) -> None:
        async for _ in websocket:
            await websocket.send(json.dumps(self.on_status_request()))

    async def serve(self) -> None:
        import websockets

        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()
