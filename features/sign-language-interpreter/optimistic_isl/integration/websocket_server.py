"""Local offline WebSocket server streaming Optimistic ISL gesture JSON."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Set

import websockets

import config
from integration.output_interface import OptimisticOutputInterface

logger = logging.getLogger("websocket_server")


class OptimisticWebSocketServer:
    """Runs an offline local WebSocket server in a background thread."""

    def __init__(
        self,
        output_interface: OptimisticOutputInterface,
        host: str = config.INFERENCE_CONFIG.websocket_host,
        port: int = config.INFERENCE_CONFIG.websocket_port,
    ) -> None:
        self.output_interface = output_interface
        self.host = host
        self.port = port
        self._clients: Set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: asyncio.Event | None = None
        self._started = threading.Event()
        self.actual_port: int = port

    async def _handler(self, websocket: Any) -> None:
        self._clients.add(websocket)
        try:
            await websocket.send(self.output_interface.get_latest_json())
            async for _ in websocket:
                # Clients may send ping/keepalive text; respond with current state
                await websocket.send(self.output_interface.get_latest_json())
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)

    async def _broadcast_async(self, payload_json: str) -> None:
        if not self._clients:
            return
        stale = []
        for ws in list(self._clients):
            try:
                await ws.send(payload_json)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)

    def broadcast(self, payload: Dict[str, Any]) -> None:
        if self._loop is not None and self._loop.is_running():
            msg = json.dumps(payload)
            asyncio.run_coroutine_threadsafe(self._broadcast_async(msg), self._loop)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.output_interface.subscribe(self.broadcast)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started.wait(timeout=5.0)

    def _run_loop(self) -> None:
        async def _serve() -> None:
            self._stop_event = asyncio.Event()
            async with websockets.serve(self._handler, self.host, self.port) as server:
                sock = server.sockets[0]
                self.actual_port = int(sock.getsockname()[1])
                self._started.set()
                logger.info(
                    "Optimistic WebSocket server listening on ws://%s:%d",
                    self.host,
                    self.actual_port,
                )
                await self._stop_event.wait()

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(_serve())
        finally:
            self._loop.close()

    def stop(self) -> None:
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
