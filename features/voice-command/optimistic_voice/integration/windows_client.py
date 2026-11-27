"""Lightweight authenticated Windows companion server for Raspberry Pi control."""

from __future__ import annotations

import asyncio
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimistic_voice.assistant.command_router import CommandRouter
from optimistic_voice.assistant.intent_classifier import IntentType, ParsedCommand
from optimistic_voice.config import OptimisticConfig
from optimistic_voice.integration.communication import sign_message, verify_message


class WindowsCompanionServer:
    """Receives authenticated commands from Raspberry Pi and executes them on Windows."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, auth_token: str = "") -> None:
        self.host = host
        self.port = port
        self.auth_token = auth_token or secrets.token_urlsafe(24)
        cfg = OptimisticConfig(execution_mode="local")
        self.router = CommandRouter(cfg)

    async def handle_client(self, websocket) -> None:
        async for raw_message in websocket:
            try:
                payload = verify_message(self.auth_token, str(raw_message))
                cmd = ParsedCommand(
                    intent=IntentType(payload["intent"]),
                    raw_text=payload.get("raw_text", ""),
                    params=payload.get("params", {}),
                )
                result = await asyncio.to_thread(self.router.execute_local, cmd)
                reply = {"success": result.success, "message": result.message, "data": result.data}
            except Exception as exc:
                reply = {"success": False, "message": f"Rejected or failed: {exc}", "data": None}
            await websocket.send(sign_message(self.auth_token, reply))

    async def serve(self) -> None:
        import websockets

        print(f"[OPTIMISTIC WINDOWS CLIENT] Listening on ws://{self.host}:{self.port}")
        if not os.getenv("OPTIMISTIC_AUTH_TOKEN"):
            print(f"[OPTIMISTIC WINDOWS CLIENT] Generated session token: {self.auth_token}")
            print("Set OPTIMISTIC_AUTH_TOKEN to this value on your Raspberry Pi.")
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()


def main() -> None:
    host = os.getenv("OPTIMISTIC_WINDOWS_HOST", "0.0.0.0")
    port = int(os.getenv("OPTIMISTIC_REMOTE_PORT", "8765"))
    token = os.getenv("OPTIMISTIC_AUTH_TOKEN", "")
    server = WindowsCompanionServer(host=host, port=port, auth_token=token)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        server.router.close()


if __name__ == "__main__":
    main()
