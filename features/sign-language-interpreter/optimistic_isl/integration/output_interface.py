"""Decoupled Optimistic Smart Glasses output interface.

Exposes the latest recognized gesture in the exact structured JSON format
required by Optimistic display hardware, completely decoupled from ML inference.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class OptimisticOutputInterface:
    """Thread-safe output state & subscriber hub for Optimistic smart glasses."""

    def __init__(self, json_sink_path: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self._latest: Dict[str, Any] = {
            "gesture": "Gesture not recognized",
            "confidence": 0.0,
            "recognized": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self._json_sink_path = json_sink_path

    @staticmethod
    def format_payload(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Project internal prediction dict onto Optimistic's clean display schema."""
        return {
            "gesture": str(prediction.get("gesture", "Gesture not recognized")),
            "confidence": round(float(prediction.get("confidence", 0.0)), 4),
            "recognized": bool(prediction.get("recognized", False)),
            "timestamp": str(
                prediction.get("timestamp") or datetime.now(timezone.utc).isoformat()
            ),
        }

    def publish(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Update latest state, optionally write JSON sink, and notify subscribers."""
        payload = self.format_payload(prediction)
        with self._lock:
            self._latest = payload
            subscribers = list(self._subscribers)

        if self._json_sink_path is not None:
            self._json_sink_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._json_sink_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._json_sink_path)

        for callback in subscribers:
            callback(payload)

        return payload

    def get_latest(self) -> Dict[str, Any]:
        """Return the latest Optimistic display payload as a dict."""
        with self._lock:
            return dict(self._latest)

    def get_latest_json(self) -> str:
        """Return the latest Optimistic display payload serialized as JSON."""
        return json.dumps(self.get_latest())

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a listener invoked whenever a new gesture state is published."""
        with self._lock:
            self._subscribers.append(callback)
