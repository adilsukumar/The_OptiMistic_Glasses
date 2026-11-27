"""Interactive web demo for Optimistic Smart Glasses ISL recognition.

Runs a local HTTP UI + MJPEG landmark stream on a custom non-standard port
(default HTTP 18472, WebSocket 18473) so users can test real-time inference on
INCLUDE test videos or a connected webcam while viewing the live Optimistic HUD.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from inference.gesture_predictor import GesturePredictor
from inference.hand_detector import RealTimeHandDetector
from integration.output_interface import OptimisticOutputInterface
from integration.websocket_server import OptimisticWebSocketServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo_server")


def load_demo_videos() -> List[Dict[str, str]]:
    """Return available INCLUDE test videos that exist on disk."""
    meta_path = config.TEST_DIR / "metadata.json"
    if not meta_path.exists():
        return []
    items = json.loads(meta_path.read_text(encoding="utf-8"))
    videos: List[Dict[str, str]] = []
    seen = set()
    for row in items:
        vp = row.get("video_path", "")
        abs_p = config.RAW_VIDEOS_DIR / vp
        if abs_p.exists() and vp not in seen:
            seen.add(vp)
            videos.append(
                {
                    "video_path": vp,
                    "label": row.get("display_label", row.get("label", "UNKNOWN")),
                }
            )
    return videos


class DemoEngine:
    """Thread-safe video/webcam runner feeding MJPEG frames and Optimistic WebSocket HUD."""

    def __init__(self, ws_port: int) -> None:
        self.ws_port = ws_port
        self.output_hub = OptimisticOutputInterface()
        self.ws_server = OptimisticWebSocketServer(
            self.output_hub, host="127.0.0.1", port=ws_port
        )
        self.ws_server.start()

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._stop_flag = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._source_desc = "Idle"
        self.videos = load_demo_videos()

        # Render initial placeholder frame
        self._set_placeholder("Select an INCLUDE test video or Start Webcam")

    def _set_placeholder(self, text: str) -> None:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas[:] = (18, 22, 32)
        cv2.putText(
            canvas,
            "OPTIMISTIC SMART GLASSES - ISL DEMO",
            (55, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (96, 165, 250),
            2,
        )
        cv2.putText(
            canvas,
            text,
            (95, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 225, 235),
            1,
        )
        ok, buf = cv2.imencode(".jpg", canvas)
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def stop_stream(self) -> None:
        self._stop_flag.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None

    def start_video(self, video_rel_path: str) -> None:
        abs_path = config.RAW_VIDEOS_DIR / video_rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Video not found: {abs_path}")
        self.stop_stream()
        self._stop_flag.clear()
        self._source_desc = f"Test Video: {video_rel_path}"
        self._worker = threading.Thread(
            target=self._run_capture,
            args=(str(abs_path), True),
            daemon=True,
        )
        self._worker.start()

    def start_webcam(self, camera_index: int = 0) -> None:
        self.stop_stream()
        self._stop_flag.clear()
        self._source_desc = f"Live Webcam #{camera_index}"
        self._worker = threading.Thread(
            target=self._run_capture,
            args=(camera_index, False),
            daemon=True,
        )
        self._worker.start()

    def _run_capture(self, source: Any, loop_video: bool) -> None:
        detector = RealTimeHandDetector()
        predictor = GesturePredictor()
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self._set_placeholder(f"Unable to open source: {source}")
            detector.close()
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_delay = 1.0 / min(max(fps, 10.0), 30.0) if loop_video else 0.0
        frame_idx = 0
        latest_payload = self.output_hub.get_latest()

        try:
            while not self._stop_flag.is_set():
                t_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        predictor.reset()
                        frame_idx = 0
                        continue
                    break

                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(
                        frame, (640, int(h * 640.0 / w)), interpolation=cv2.INTER_AREA
                    )

                frame_idx += 1
                feat, detected, raw_hands = detector.process_bgr_frame(frame)
                run_inf = (frame_idx % config.INFERENCE_CONFIG.inference_stride) == 0
                pred = predictor.push_frame(feat, run_inference=run_inf)
                if not pred.get("skipped_stride", False):
                    latest_payload = self.output_hub.publish(pred)

                RealTimeHandDetector.draw_landmarks(frame, raw_hands)
                status = (
                    f"RECOGNIZED: {latest_payload['gesture']} ({latest_payload['confidence']*100:.1f}%)"
                    if latest_payload["recognized"]
                    else f"{latest_payload['gesture']} ({latest_payload['confidence']*100:.1f}%)"
                )
                color = (34, 197, 94) if latest_payload["recognized"] else (59, 130, 246)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 56), (15, 23, 42), -1)
                cv2.putText(frame, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                cv2.putText(
                    frame,
                    f"Hands: {'YES' if detected else 'NO'} | Buffer: {pred['buffer_fill']}/{pred['buffer_capacity']}",
                    (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (203, 213, 225),
                    1,
                )

                enc_ok, buf = cv2.imencode(".jpg", frame)
                if enc_ok:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()

                if frame_delay > 0:
                    elapsed = time.perf_counter() - t_start
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)
        finally:
            cap.release()
            detector.close()

    def shutdown(self) -> None:
        self.stop_stream()
        self.ws_server.stop()


def build_html(ws_port: int, videos: List[Dict[str, str]]) -> str:
    options = "\n".join(
        f'<option value="{v["video_path"]}">{v["label"]} — {v["video_path"]}</option>'
        for v in videos
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Optimistic Smart Glasses — ISL Live Demo</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: #0b1120; color: #e2e8f0; display: flex; flex-direction: column; align-items: center; padding: 24px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 1.5rem; color: #60a5fa; }}
    .sub {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 18px; }}
    .controls {{
      display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: center;
      background: #111827; padding: 12px 16px; border-radius: 12px; border: 1px solid #1e293b; margin-bottom: 18px;
    }}
    select, button {{
      padding: 8px 12px; border-radius: 8px; border: 1px solid #334155;
      background: #1e293b; color: #f8fafc; font-size: 0.9rem; cursor: pointer;
    }}
    button.primary {{ background: #2563eb; border-color: #3b82f6; font-weight: 600; }}
    button.secondary {{ background: #0f766e; border-color: #14b8a6; font-weight: 600; }}
    .grid {{
      display: grid; grid-template-columns: minmax(320px, 640px) minmax(280px, 360px);
      gap: 20px; width: 100%; max-width: 1040px;
    }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .card {{
      background: #111827; border: 1px solid #1e293b; border-radius: 14px; padding: 16px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.35);
    }}
    .video-wrap img {{ width: 100%; border-radius: 10px; display: block; background: #020617; }}
    .hud-box {{
      background: #020617; border: 2px solid #1e293b; border-radius: 12px; padding: 18px; text-align: center;
      margin-bottom: 14px;
    }}
    .hud-box.active {{ border-color: #22c55e; box-shadow: 0 0 18px rgba(34,197,94,0.25); }}
    .hud-label {{ font-size: 0.75rem; letter-spacing: 0.1em; color: #94a3b8; text-transform: uppercase; }}
    .hud-word {{ font-size: 1.8rem; font-weight: 700; margin: 8px 0; color: #f8fafc; }}
    .hud-conf {{ font-size: 0.95rem; color: #38bdf8; }}
    pre {{
      background: #020617; padding: 12px; border-radius: 8px; font-size: 0.8rem;
      overflow-x: auto; border: 1px solid #1e293b; color: #a5f3fc;
    }}
  </style>
</head>
<body>
  <h1>Optimistic Smart Glasses — Sign Language Interpreter Demo</h1>
  <div class="sub">15-Class TFLite Model (116 KB) &bull; MediaPipe 21-Point Landmarks &bull; Local WebSocket ws://127.0.0.1:{ws_port}</div>

  <div class="controls">
    <select id="videoSelect">
      {options}
    </select>
    <button class="primary" onclick="startVideo()">Play Test Video</button>
    <button class="secondary" onclick="startWebcam()">Use Live Webcam</button>
    <button onclick="stopStream()">Stop</button>
  </div>

  <div class="grid">
    <div class="card video-wrap">
      <img id="stream" src="/stream.mjpg" alt="Live ISL Stream" />
    </div>
    <div class="card">
      <div id="hudBox" class="hud-box">
        <div class="hud-label">Optimistic Glasses HUD</div>
        <div id="hudWord" class="hud-word">Gesture not recognized</div>
        <div id="hudConf" class="hud-conf">Confidence: 0.0%</div>
      </div>
      <div class="hud-label">Live WebSocket JSON Payload</div>
      <pre id="jsonOut">{{}}</pre>
    </div>
  </div>

  <script>
    const ws = new WebSocket("ws://127.0.0.1:{ws_port}");
    ws.onmessage = (ev) => {{
      const data = JSON.parse(ev.data);
      document.getElementById("hudWord").textContent = data.gesture;
      document.getElementById("hudConf").textContent = "Confidence: " + (data.confidence * 100).toFixed(1) + "%";
      document.getElementById("hudBox").classList.toggle("active", Boolean(data.recognized));
      document.getElementById("jsonOut").textContent = JSON.stringify(data, null, 2);
    }};

    async function startVideo() {{
      const vp = document.getElementById("videoSelect").value;
      await fetch("/api/start_video?path=" + encodeURIComponent(vp), {{ method: "POST" }});
    }}
    async function startWebcam() {{
      await fetch("/api/start_webcam", {{ method: "POST" }});
    }}
    async function stopStream() {{
      await fetch("/api/stop", {{ method: "POST" }});
    }}
  </script>
</body>
</html>"""


def make_handler(engine: DemoEngine):
    html_bytes = build_html(engine.ws_port, engine.videos).encode("utf-8")

    class DemoHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
            elif parsed.path == "/api/status":
                body = json.dumps(engine.output_hub.get_latest()).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/stream.mjpg":
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                try:
                    while True:
                        jpg = engine.get_latest_jpeg()
                        if jpg:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                            )
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/start_video":
                qs = parse_qs(parsed.query)
                vp = qs.get("path", [""])[0]
                engine.start_video(vp)
                self._ok({"started": vp})
            elif parsed.path == "/api/start_webcam":
                engine.start_webcam(0)
                self._ok({"started": "webcam:0"})
            elif parsed.path == "/api/stop":
                engine.stop_stream()
                self._ok({"stopped": True})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _ok(self, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DemoHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimistic ISL Live Web Demo")
    parser.add_argument("--http-port", type=int, default=18472)
    parser.add_argument("--ws-port", type=int, default=18473)
    parser.add_argument("--autoplay", action="store_true", help="Autoplay first test video on startup.")
    args = parser.parse_args()

    engine = DemoEngine(ws_port=args.ws_port)
    if args.autoplay and engine.videos:
        engine.start_video(engine.videos[0]["video_path"])

    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), make_handler(engine))
    logger.info("Optimistic ISL Demo running at http://127.0.0.1:%d (WS port %d)", args.http_port, args.ws_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.shutdown()


if __name__ == "__main__":
    main()
