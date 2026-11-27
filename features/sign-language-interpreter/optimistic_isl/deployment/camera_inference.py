"""Standalone real-time camera & WebSocket server for Raspberry Pi deployment."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

import cv2
import numpy as np
import websockets

from gesture_predictor import PiGesturePredictor
from hand_detector import HAND_CONNECTIONS, PiHandDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pi_camera_inference")


class LocalBroadcastServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._clients: Set[Any] = set()
        self._latest: Dict[str, Any] = {
            "gesture": "Gesture not recognized",
            "confidence": 0.0,
            "recognized": False,
            "timestamp": "",
        }
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._thread: Optional[threading.Thread] = None

    async def _handler(self, ws: Any) -> None:
        self._clients.add(ws)
        try:
            await ws.send(json.dumps(self._latest))
            async for _ in ws:
                await ws.send(json.dumps(self._latest))
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)

    def start(self) -> None:
        ready = threading.Event()

        def _run() -> None:
            async def _serve() -> None:
                self._stop_event = asyncio.Event()
                async with websockets.serve(self._handler, self.host, self.port):
                    ready.set()
                    await self._stop_event.wait()

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(_serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        ready.wait(timeout=5.0)
        logger.info("Local glasses WebSocket running at ws://%s:%d", self.host, self.port)

    def publish(self, payload: Dict[str, Any]) -> None:
        clean = {
            "gesture": payload["gesture"],
            "confidence": payload["confidence"],
            "recognized": payload["recognized"],
            "timestamp": payload["timestamp"],
        }
        self._latest = clean
        if self._loop and self._loop.is_running() and self._clients:
            msg = json.dumps(clean)

            async def _send() -> None:
                for c in list(self._clients):
                    try:
                        await c.send(msg)
                    except Exception:
                        self._clients.discard(c)

            asyncio.run_coroutine_threadsafe(_send(), self._loop)

    def stop(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimistic Pi ISL real-time inference.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--picamera2", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--ws-host", default="127.0.0.1")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--json-sink", type=Path, default=None)
    args = parser.parse_args()

    detector = PiHandDetector()
    predictor = PiGesturePredictor(confidence_threshold=args.threshold)
    server = LocalBroadcastServer(host=args.ws_host, port=args.ws_port)
    server.start()

    picam = None
    cap = None
    if args.picamera2:
        from picamera2 import Picamera2  # type: ignore
        picam = Picamera2()
        picam.configure(picam.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
        picam.start()
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {args.camera}")

    frame_idx = 0
    try:
        while True:
            if picam is not None:
                frame = cv2.cvtColor(picam.capture_array(), cv2.COLOR_RGB2BGR)
            else:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

            frame_idx += 1
            feat, _, raw_hands = detector.process_bgr_frame(frame)
            if frame_idx % args.stride == 0:
                pred = predictor.push_frame(feat)
                server.publish(pred)
                if args.json_sink is not None:
                    args.json_sink.write_text(
                        json.dumps(
                            {
                                "gesture": pred["gesture"],
                                "confidence": pred["confidence"],
                                "recognized": pred["recognized"],
                                "timestamp": pred["timestamp"],
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            else:
                predictor.buffer.append(feat)

            if not args.headless:
                h, w = frame.shape[:2]
                for pts in raw_hands:
                    px = [(int(p[0] * w), int(p[1] * h)) for p in pts]
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(frame, px[a], px[b], (255, 180, 0), 2)
                    for cx, cy in px:
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 120), -1)
                cv2.imshow("Optimistic Pi ISL", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if picam is not None:
            picam.stop()
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        server.stop()
        detector.close()


if __name__ == "__main__":
    main()
