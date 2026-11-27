"""Authenticated local WebSocket communication between Raspberry Pi and Windows."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from typing import Any

from optimistic_voice.assistant.intent_classifier import ParsedCommand
from optimistic_voice.automation import ExecutionResult


def sign_message(secret: str, payload: dict[str, Any]) -> str:
    """Create an HMAC-SHA256 authenticated JSON envelope."""
    envelope = {
        "timestamp": int(time.time()),
        "payload": payload,
    }
    body = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    sig = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    return json.dumps({"envelope": body, "signature": sig})


def verify_message(secret: str, raw_message: str, max_age_s: int = 30) -> dict[str, Any]:
    """Verify HMAC signature and freshness of an incoming JSON message."""
    data = json.loads(raw_message)
    body = data.get("envelope", "")
    sig = data.get("signature", "")
    expected = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise PermissionError("Invalid authentication signature.")

    envelope = json.loads(body)
    ts = int(envelope.get("timestamp", 0))
    if abs(time.time() - ts) > max_age_s:
        raise PermissionError("Expired command timestamp.")
    return envelope["payload"]


class RemoteWindowsClient:
    """Sends authenticated ParsedCommand payloads from Raspberry Pi to Windows."""

    def __init__(self, host: str, port: int, auth_token: str) -> None:
        if not auth_token:
            raise ValueError("OPTIMISTIC_AUTH_TOKEN must be set for remote_windows mode.")
        self.uri = f"ws://{host}:{port}"
        self.auth_token = auth_token

    def send_command(self, cmd: ParsedCommand) -> ExecutionResult:
        payload = {
            "intent": cmd.intent.value,
            "raw_text": cmd.raw_text,
            "params": cmd.params,
        }
        try:
            return asyncio.run(self._send_async(payload))
        except Exception as exc:
            return ExecutionResult(success=False, message=f"Remote Windows execution failed: {exc}")

    async def _send_async(self, payload: dict[str, Any]) -> ExecutionResult:
        import websockets

        signed = sign_message(self.auth_token, payload)
        async with websockets.connect(self.uri, open_timeout=5) as ws:
            await ws.send(signed)
            raw_reply = await asyncio.wait_for(ws.recv(), timeout=25)
        reply = verify_message(self.auth_token, str(raw_reply))
        return ExecutionResult(
            success=bool(reply.get("success", False)),
            message=str(reply.get("message", "")),
            data=reply.get("data"),
        )
