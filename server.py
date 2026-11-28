"""Local development web server for The OptiMistic Glasses ISL Live Translator & Control Center."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
VOICE_PARENT = ROOT_DIR / "features" / "voice-command"
ISL_DIR = ROOT_DIR / "features" / "sign-language-interpreter" / "optimistic_isl"

for path_entry in (str(ROOT_DIR), str(VOICE_PARENT), str(ISL_DIR)):
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

try:
    from optimistic_voice.assistant.command_parser import CommandParser

    _PARSER = CommandParser()
    _PARSER_ERROR = ""
except Exception as exc:  # pragma: no cover
    _PARSER = None
    _PARSER_ERROR = str(exc)

from isl_recognition_engine import (
    INCLUDE_VIDEOS, ISL_LESSON, ForegroundHands, PrecisionISLRecognizer, SigningFace, gesture_guide,
)
from isl_speech import WindowsSpeaker, format_transcript
import isl_vocabulary_db as db

_SPEAKER = WindowsSpeaker()


class ISLController:
    """Keep the latest camera frame and publish completed recognition events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_cond = threading.Condition()
        self._raw_ready = threading.Event()
        self._running = False
        self._mirror_mode = True
        self._capture_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None

        self._latest_raw_frame: Any = None
        self._frame_seq = 0
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_seq = 0

        self._latest_hands: List[Any] = []
        self._confidence_threshold = 0.75
        self._mode = "conversation"
        self._auto_speak = True
        self._revision = 0
        self._event_id = 0
        self._events = deque(maxlen=100)
        self._latest_prediction = {
            "running": False, "hands_detected": False, "fps": 0.0, "latency_ms": 0.0,
            "mirror_mode": True, "status": "Standby", "error": "",
        }

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def set_threshold(self, threshold: float) -> float:
        clamped = max(0.65, min(0.95, float(threshold)))
        with self._lock:
            self._confidence_threshold = clamped
        return clamped

    def set_mirror_mode(self, enabled: bool) -> bool:
        with self._lock:
            self._mirror_mode = bool(enabled)
            self._latest_prediction["mirror_mode"] = self._mirror_mode
        return self._mirror_mode

    def set_mode(self, mode: str) -> str:
        if mode not in ("conversation", "counting"):
            raise ValueError("Choose Conversation or Counting.")
        with self._lock:
            if mode != self._mode:
                self._mode = mode
                self._revision += 1
                self._latest_prediction["status"] = "Show your hands"
                _SPEAKER.clear()
        return mode

    def set_auto_speak(self, enabled: bool) -> bool:
        with self._lock:
            self._auto_speak = enabled
            if not enabled:
                _SPEAKER.clear()
        return enabled

    def request_reset(self, clear_transcript: bool = False) -> None:
        with self._lock:
            self._revision += 1
            self._latest_prediction["status"] = "Show your hands"
            if clear_transcript:
                self._events.clear()
                _SPEAKER.clear()

    def speak_transcript(self, event_ids: list[int]) -> bool:
        with self._lock:
            if self._mode != "conversation":
                return False
            phrases = {event["id"]: event["english"] for event in self._events
                       if event.get("category") != "Counting"}
            if (not isinstance(event_ids, list) or not 0 < len(event_ids) <= len(phrases)
                    or any(type(idx) is not int or idx not in phrases for idx in event_ids)):
                return False
            _SPEAKER.speak(format_transcript(phrases[idx] for idx in event_ids))
            return True

    def start(self, camera_index: int = 0) -> Dict[str, Any]:
        with self._lock:
            if self._running:
                return {"started": True, "already_running": True}
            if any(thread and thread.is_alive() for thread in (self._capture_thread, self._infer_thread)):
                return {"started": False, "error": "Camera is stopping. Try again in a moment."}
            self._running = True
            self._latest_hands = []
            self._latest_raw_frame = None
            self._latest_jpeg = None
            self._raw_ready.clear()
            self._latest_prediction.update(
                running=True, hands_detected=False, fps=0.0, latency_ms=0.0,
                status="Starting camera", error="",
            )
            self._capture_thread = threading.Thread(
                target=self._capture_loop, args=(camera_index,), daemon=True
            )
            self._infer_thread = threading.Thread(
                target=self._inference_loop, daemon=True
            )
            self._capture_thread.start()
            self._infer_thread.start()
        return {"started": True, "already_running": False}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            self._running = False
            self._raw_ready.set()
            _SPEAKER.clear()
            cap_t = self._capture_thread
            inf_t = self._infer_thread
        with self._frame_cond:
            self._frame_cond.notify_all()
        if cap_t is not None and cap_t.is_alive():
            cap_t.join(timeout=2.5)
        if inf_t is not None and inf_t.is_alive():
            inf_t.join(timeout=2.5)
        with self._lock:
            self._latest_jpeg = None
            self._latest_raw_frame = None
            self._latest_hands = []
            self._latest_prediction.update(running=False, hands_detected=False, status="Standby")
        return {"stopped": True}

    def get_prediction(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_prediction, mode=self._mode, auto_speak=self._auto_speak,
                        threshold=self._confidence_threshold, events=list(self._events))

    def wait_for_jpeg(self, last_seq: int, timeout: float = 0.5) -> tuple[Optional[bytes], int]:
        with self._frame_cond:
            if self._jpeg_seq == last_seq and self._running:
                self._frame_cond.wait(timeout=timeout)
            return (self._latest_jpeg if self._jpeg_seq != last_seq else None), self._jpeg_seq

    def _capture_loop(self, camera_index: int) -> None:
        import cv2  # type: ignore
        import config  # type: ignore
        from inference.hand_detector import RealTimeHandDetector  # type: ignore

        cap = None
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera index {camera_index}")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.INFERENCE_CONFIG.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.INFERENCE_CONFIG.frame_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            fps_t0 = time.perf_counter()
            fps_frames = 0
            current_fps = 0.0
            last_frame = fps_t0

            while self.is_running():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if time.perf_counter() - last_frame > 3:
                        raise RuntimeError("The camera stopped delivering frames.")
                    time.sleep(0.01)
                    continue

                fps_frames += 1
                now = time.perf_counter()
                last_frame = now
                if now - fps_t0 >= 1.0:
                    current_fps = fps_frames / (now - fps_t0)
                    fps_frames = 0
                    fps_t0 = now

                with self._lock:
                    mirror = self._mirror_mode
                    self._latest_raw_frame = frame
                    self._frame_seq += 1
                    hands = list(self._latest_hands)
                    self._latest_prediction["fps"] = round(current_fps, 1)
                    self._raw_ready.set()

                display_frame = cv2.flip(frame, 1) if mirror else frame.copy()
                if hands:
                    # In mirror mode, flip landmarks x coordinate to match display
                    if mirror:
                        mirrored_hands = []
                        for h in hands:
                            mh = h.copy()
                            mh[:, 0] = 1.0 - mh[:, 0]
                            mirrored_hands.append(mh)
                        RealTimeHandDetector.draw_landmarks(display_frame, mirrored_hands)
                    else:
                        RealTimeHandDetector.draw_landmarks(display_frame, hands)

                ok_jpg, encoded = cv2.imencode(
                    ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                )
                if ok_jpg:
                    with self._frame_cond:
                        self._latest_jpeg = encoded.tobytes()
                        self._jpeg_seq += 1
                        self._frame_cond.notify_all()
        except Exception as exc:
            with self._lock:
                self._running = False
                self._latest_prediction["running"] = False
                self._latest_prediction["error"] = str(exc)
            with self._frame_cond:
                self._frame_cond.notify_all()
        finally:
            if cap is not None:
                cap.release()

    def _inference_loop(self) -> None:
        detector = None
        try:
            from inference.hand_detector import RealTimeHandDetector

            foreground = ForegroundHands()
            face_tracker = SigningFace()
            detector = RealTimeHandDetector(detection_width=384, foreground_selector=foreground.select)
            recognizer = PrecisionISLRecognizer()
            recognizer.load_model()
            revision = -1
            last_seq = -1
            while self.is_running():
                self._raw_ready.wait(timeout=0.25)
                self._raw_ready.clear()
                with self._lock:
                    frame = self._latest_raw_frame
                    seq = self._frame_seq
                    mode = self._mode
                    threshold = self._confidence_threshold
                    current_revision = self._revision
                if frame is None or seq == last_seq:
                    continue
                last_seq = seq
                if revision != current_revision:
                    recognizer.reset()
                    foreground.reset()
                    face_tracker.reset()
                    revision = current_revision
                recognizer.mode = mode
                recognizer.threshold = threshold

                start = time.perf_counter()
                features, detected, hands = detector.process_bgr_frame(frame)
                face = face_tracker.locate(frame) if detected and mode == "conversation" else None
                match = recognizer.process(features, hands, face=face)
                elapsed = (time.perf_counter() - start) * 1000
                if mode == "counting":
                    status = "Hold a number pose" if detected else "Show a number pose"
                else:
                    status = recognizer.status
                preview = recognizer.get_live_sentence_preview()
                with self._lock:
                    if not self._running or revision != self._revision:
                        continue
                    self._latest_hands = hands
                    self._latest_prediction.update(
                        hands_detected=detected,
                        status=status,
                        latency_ms=round(elapsed, 1),
                        live_preview=preview,
                    )
                    if match:
                        is_complete = match.get("is_sentence_complete", False)
                        if mode == "conversation":
                            # Speak out loud when full connected sentence completes
                            if is_complete:
                                self._event_id += 1
                                self._events.append(dict(match, id=self._event_id))
                                if self._auto_speak:
                                    _SPEAKER.speak(match["english"])
                        else:
                            self._event_id += 1
                            self._events.append(dict(match, id=self._event_id))
        except Exception as exc:
            with self._lock:
                self._running = False
                self._latest_prediction.update(running=False, error=str(exc))
        finally:
            if detector is not None:
                detector.close()


_ISL_CONTROLLER = ISLController()


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class OptimisticHandler(SimpleHTTPRequestHandler):
    """Serves static files, ISL vocabulary, custom sign APIs, MJPEG camera, and voice command parser."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        if args and "/api/camera/prediction" in str(args[0]):
            return
        super().log_message(format, *args)

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/info":
            labels = _read_json(ISL_DIR / "deployment" / "labels.json")
            metrics = _read_json(ISL_DIR / "evaluation" / "metrics.json")
            self._send_json(
                {
                    "project": "The OptiMistic Glasses — Indian Sign Language Live Translator",
                    "root": str(ROOT_DIR),
                    "parser_ready": _PARSER is not None,
                    "parser_error": _PARSER_ERROR,
                    "labels": labels,
                    "metrics": metrics,
                    "gesture_guide": gesture_guide(),
                    "supported_translations": {
                        entry["gesture"]: {"english": entry["english"], "hindi": entry["hindi"]}
                        for entry in gesture_guide()
                    },
                    "total_signs": len(db.ISL_MASTER_VOCABULARY),
                    "categories": db.get_categories(),
                    "isl_lesson": ISL_LESSON,
                    "include_videos": INCLUDE_VIDEOS,
                }
            )
            return

        if self.path.startswith("/api/vocabulary"):
            from urllib.parse import parse_qs, urlparse
            query_params = parse_qs(urlparse(self.path).query)
            q = query_params.get("q", [""])[0].strip()
            cat = query_params.get("category", [""])[0].strip()
            if q:
                results = db.search_signs(q, limit=60)
            elif cat and cat in db.BY_CATEGORY:
                results = db.BY_CATEGORY[cat]
            else:
                results = db.ISL_MASTER_VOCABULARY
            self._send_json({
                "total": len(db.ISL_MASTER_VOCABULARY),
                "count": len(results),
                "categories": db.get_categories(),
                "vocabulary": results,
            })
            return

        if self.path == "/api/custom-signs":
            self._send_json({"custom_signs": []})
            return

        if self.path == "/api/camera/prediction":
            self._send_json(_ISL_CONTROLLER.get_prediction())
            return

        if self.path.startswith("/api/camera/stream"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_seq = -1
            try:
                while _ISL_CONTROLLER.is_running():
                    frame_bytes, last_seq = _ISL_CONTROLLER.wait_for_jpeg(last_seq, timeout=0.5)
                    if frame_bytes:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return

        if self.path == "/api/files":
            editable = [
                "web/index.html",
                "web/style.css",
                "web/app.js",
                "web/isl_vocabulary.json",
                "server.py",
                "isl_recognition_engine.py",
                "features/sign-language-interpreter/optimistic_isl/config.py",
                "features/voice-command/optimistic_voice/assistant/command_parser.py",
            ]
            self._send_json({"files": [f for f in editable if (ROOT_DIR / f).exists()]})
            return

        super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/api/camera/start":
            self._send_json(_ISL_CONTROLLER.start(camera_index=0))
            return

        if self.path == "/api/camera/stop":
            self._send_json(_ISL_CONTROLLER.stop())
            return

        if self.path in {"/api/camera/speak", "/api/camera/reset",
                         "/api/camera/mode", "/api/camera/auto-speak"}:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length) if length else "{}")
                if self.path == "/api/camera/speak":
                    if not _ISL_CONTROLLER.speak_transcript(data.get("event_ids", [])):
                        raise ValueError("Select completed Conversation translations to speak.")
                    result = {"spoken": True}
                elif self.path == "/api/camera/reset":
                    _ISL_CONTROLLER.request_reset(data.get("clear") is True)
                    result = {"reset": True}
                elif self.path == "/api/camera/mode":
                    result = {"mode": _ISL_CONTROLLER.set_mode(data.get("mode"))}
                else:
                    result = {"auto_speak": _ISL_CONTROLLER.set_auto_speak(data.get("enabled") is True)}
                self._send_json(result)
            except (ValueError, TypeError, AttributeError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/camera/mirror":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                data = json.loads(raw_body)
                val = _ISL_CONTROLLER.set_mirror_mode(bool(data.get("mirror", True)))
                self._send_json({"mirror": val})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/camera/threshold":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                data = json.loads(raw_body)
                val = _ISL_CONTROLLER.set_threshold(float(data.get("threshold", 0.55)))
                self._send_json({"threshold": val})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path in {"/api/custom-signs/record", "/api/custom-signs/delete"}:
            self._send_json({"success": False, "error": "Custom sign training is unavailable."},
                            status=HTTPStatus.NOT_IMPLEMENTED)
            return

        if self.path == "/api/parse-command":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                data = json.loads(raw_body)
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return

            text = str(data.get("text", "")).strip()
            if _PARSER is None:
                self._send_json(
                    {"error": f"CommandParser unavailable: {_PARSER_ERROR}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

            parsed = _PARSER.parse(text)
            self._send_json(
                {
                    "raw_text": parsed.raw_text,
                    "intent": parsed.intent.value
                    if hasattr(parsed.intent, "value")
                    else str(parsed.intent),
                    "confidence": parsed.confidence,
                    "requires_confirmation": getattr(parsed, "requires_confirmation", False),
                    "parameters": {
                        "target": getattr(parsed, "target", None),
                        "query": getattr(parsed, "query", None),
                        "text_to_type": getattr(parsed, "text_to_type", None),
                        "index": getattr(parsed, "index", None),
                        "direction": getattr(parsed, "direction", None),
                        "action": getattr(parsed, "action", None),
                    },
                }
            )
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), OptimisticHandler)
    print(f"The OptiMistic Glasses ISL Live Translator running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        _ISL_CONTROLLER.stop()
        _SPEAKER.close()
        server.server_close()


if __name__ == "__main__":
    main()
