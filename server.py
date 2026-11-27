"""Local development web server for The OptiMistic Glasses ISL Live Translator & Control Center."""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
VOICE_PARENT = ROOT_DIR / "features" / "voice-command"
ISL_DIR = ROOT_DIR / "features" / "sign-language-interpreter" / "optimistic_isl"
DATA_DIR = ROOT_DIR / "data"

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

from isl_recognition_engine import PrecisionISLRecognizer, analyze_hand


class RobustWindowsSpeaker:
    """Threaded Windows speaker using SAPI with PowerShell System.Speech fallback."""

    def __init__(self, rate: int = 1) -> None:
        self.q: queue.Queue[str] = queue.Queue(maxsize=5)
        self.rate = rate
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self) -> None:
        spk = None
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            spk = win32com.client.Dispatch("SAPI.SpVoice")
            spk.Rate = self.rate
        except Exception:
            spk = None

        while True:
            text = self.q.get()
            if not text:
                continue

            spoken = False
            if spk is not None:
                try:
                    spk.Speak(text, 0)
                    spoken = True
                except Exception:
                    pass

            if not spoken:
                try:
                    import subprocess

                    sanitized = text.replace('"', '\\"')
                    cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{sanitized}")'
                    subprocess.run(
                        ["powershell", "-NoProfile", "-Command", cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass

            self.q.task_done()

    def speak(self, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        while not self.q.empty():
            try:
                self.q.get_nowait()
                self.q.task_done()
            except Exception:
                break
        self.q.put(clean)


_SPEAKER = RobustWindowsSpeaker(rate=1)

# English & Hindi translation mappings for recognized signs
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "NAMASTE": {"english": "Namaste", "hindi": "नमस्ते"},
    "HELLO": {"english": "Hello", "hindi": "हैलो / नमस्ते"},
    "GOODBYE": {"english": "Goodbye", "hindi": "अलविदा"},
    "BYE": {"english": "Goodbye", "hindi": "अलविदा"},
    "HOW ARE YOU": {"english": "How are you?", "hindi": "आप कैसे हैं?"},
    "I AM FINE": {"english": "I am fine", "hindi": "मैं ठीक हूँ"},
    "ALRIGHT": {"english": "Alright", "hindi": "ठीक है"},
    "I AM A BOY": {"english": "I am a boy", "hindi": "मैं एक लड़का हूँ"},
    "I AM A GIRL": {"english": "I am a girl", "hindi": "मैं एक लड़की हूँ"},
    "GOOD MORNING": {"english": "Good morning", "hindi": "शुभ प्रभात"},
    "GOOD AFTERNOON": {"english": "Good afternoon", "hindi": "शुभ दोपहर"},
    "GOOD EVENING": {"english": "Good evening", "hindi": "शुभ संध्या"},
    "GOOD NIGHT": {"english": "Good night", "hindi": "शुभ रात्रि"},
    "THANK YOU": {"english": "Thank you", "hindi": "धन्यवाद"},
    "PLEASED": {"english": "Pleased to meet you", "hindi": "आपसे मिलकर अच्छा लगा"},
    "I LOVE YOU": {"english": "I love you", "hindi": "मैं तुमसे प्यार करता हूँ"},
    "YES": {"english": "Yes", "hindi": "हाँ"},
    "NO": {"english": "No", "hindi": "नहीं"},
    "STOP": {"english": "Stop", "hindi": "रुकिए / बंद करें"},
    "WAIT": {"english": "Wait", "hindi": "रुकिए / प्रतीक्षा करें"},
    "PLEASE": {"english": "Please", "hindi": "कृपया"},
    "HELP ME": {"english": "Help me", "hindi": "मेरी मदद करो"},
    "WHAT ARE YOU DOING": {"english": "What are you doing?", "hindi": "आप क्या कर रहे हैं?"},
    "ONE": {"english": "One (1)", "hindi": "एक"},
    "TWO": {"english": "Two (2)", "hindi": "दो"},
    "THREE": {"english": "Three (3)", "hindi": "तीन"},
    "FOUR": {"english": "Four (4)", "hindi": "चार"},
    "FIVE": {"english": "Five (5)", "hindi": "पाँच"},
    "SIX": {"english": "Six (6)", "hindi": "छह"},
    "SEVEN": {"english": "Seven (7)", "hindi": "सात"},
    "EIGHT": {"english": "Eight (8)", "hindi": "आठ"},
    "TEN": {"english": "Ten (10)", "hindi": "दस"},
    "SUMMER": {"english": "Summer", "hindi": "गर्मी का मौसम"},
    "SPRING": {"english": "Spring", "hindi": "वसंत ऋतु"},
    "WINTER": {"english": "Winter", "hindi": "सर्दी का मौसम"},
    "FALL": {"english": "Autumn / Fall", "hindi": "पतझड़"},
    "SEASON": {"english": "Season", "hindi": "मौसम"},
    "MONSOON": {"english": "Monsoon / Rainy Season", "hindi": "मानसून / बारिश"},
}


class ISLController:
    """Decoupled high-FPS camera capture + Hybrid MediaPipe/TFLite/Geometric ISL inference."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_cond = threading.Condition()
        self._running = False
        self._mirror_mode = True
        self._capture_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None

        self._latest_raw_frame: Any = None
        self._frame_seq = 0
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_seq = 0

        self._latest_hands: List[Any] = []
        self._hands_detected = False
        self._confidence_threshold = 0.55
        self._reset_requested = False

        self._recognizer = PrecisionISLRecognizer(stability_threshold=2)
        self._latest_buffer: Optional[Any] = None

        self._latest_prediction: Dict[str, Any] = {
            "running": False,
            "gesture": "CAMERA STOPPED",
            "english": "",
            "hindi": "",
            "recognized": False,
            "confidence": 0.0,
            "hands_detected": False,
            "buffer_fill": 0,
            "buffer_capacity": 30,
            "top_predictions": [],
            "source": "",
            "fps": 0.0,
            "latency_ms": 0.0,
            "mirror_mode": True,
            "error": "",
        }

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def set_threshold(self, threshold: float) -> float:
        clamped = max(0.35, min(0.95, float(threshold)))
        with self._lock:
            self._confidence_threshold = clamped
        return clamped

    def set_mirror_mode(self, enabled: bool) -> bool:
        with self._lock:
            self._mirror_mode = bool(enabled)
            self._latest_prediction["mirror_mode"] = self._mirror_mode
        return self._mirror_mode

    def request_reset(self) -> None:
        with self._lock:
            self._reset_requested = True

    def record_custom_sample(self, label: str) -> Dict[str, Any]:
        with self._lock:
            buf = self._latest_buffer
        if buf is None:
            return {"success": False, "error": "No landmark buffer available yet. Show hands to webcam first."}
        count = self._custom_classifier.add_sample(label, buf)
        return {"success": True, "label": label.upper(), "sample_count": count}

    def delete_custom_sign(self, label: str) -> Dict[str, Any]:
        deleted = self._custom_classifier.delete_sign(label)
        return {"success": deleted, "label": label.upper()}

    def get_custom_signs(self) -> List[Dict[str, Any]]:
        return self._custom_classifier.list_signs()

    def start(self, camera_index: int = 0) -> Dict[str, Any]:
        with self._lock:
            if self._running:
                return {"started": True, "already_running": True}
            self._running = True
            self._latest_hands = []
            self._hands_detected = False
            self._latest_raw_frame = None
            self._latest_jpeg = None
            self._latest_prediction.update(
                {
                    "running": True,
                    "gesture": "STARTING CAMERA...",
                    "english": "",
                    "hindi": "",
                    "recognized": False,
                    "confidence": 0.0,
                    "hands_detected": False,
                    "buffer_fill": 0,
                    "top_predictions": [],
                    "error": "",
                }
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
            self._latest_prediction["running"] = False
            self._latest_prediction["gesture"] = "CAMERA STOPPED"
            self._latest_prediction["english"] = ""
            self._latest_prediction["hindi"] = ""
            self._latest_prediction["recognized"] = False
            self._latest_prediction["hands_detected"] = False
            self._latest_prediction["top_predictions"] = []
        return {"stopped": True}

    def get_prediction(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_prediction)

    def wait_for_jpeg(self, last_seq: int, timeout: float = 0.5) -> tuple[Optional[bytes], int]:
        with self._frame_cond:
            if self._jpeg_seq == last_seq and self._running:
                self._frame_cond.wait(timeout=timeout)
            return self._latest_jpeg, self._jpeg_seq

    def _capture_loop(self, camera_index: int) -> None:
        import cv2  # type: ignore
        import config  # type: ignore
        from inference.hand_detector import RealTimeHandDetector  # type: ignore

        cap = None
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
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

            while self.is_running():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                fps_frames += 1
                now = time.perf_counter()
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
        import numpy as np  # type: ignore
        import config  # type: ignore
        from dataset.extract_landmarks import interpolate_short_dropouts  # type: ignore
        from inference.gesture_predictor import GesturePredictor  # type: ignore
        from inference.hand_detector import RealTimeHandDetector  # type: ignore

        detector = None
        try:
            detector = RealTimeHandDetector(detection_width=320)
            predictor = GesturePredictor(
                confidence_threshold=self._confidence_threshold,
                smoothing_window=5,
                min_consistent_frames=2,
            )
            predictor.smoother.ema_alpha = 0.60

            last_seq = -1
            active_spoken_sign: Optional[str] = None
            no_match_frames = 0
            while self.is_running():
                with self._lock:
                    frame = self._latest_raw_frame
                    seq = self._frame_seq
                    threshold = self._confidence_threshold
                    do_reset = self._reset_requested
                    self._reset_requested = False

                if do_reset:
                    predictor.reset()

                predictor.smoother.confidence_threshold = threshold

                if frame is None or seq == last_seq:
                    time.sleep(0.01)
                    continue
                last_seq = seq

                feat, detected, raw_hands = detector.process_bgr_frame(frame)
                match = self._recognizer.process(raw_hands)

                top_preds: List[Dict[str, Any]] = []
                gesture = "ANALYZING..."
                english = ""
                hindi = ""
                recognized = False
                confidence = 0.0
                latency_ms = 0.0
                source = "Precision ISL"

                buf_len = int(self._recognizer.candidate_count)
                cap_len = int(self._recognizer.stability_threshold)

                if match is not None:
                    gesture = match["gesture"]
                    english = match["english"]
                    hindi = match["hindi"]
                    confidence = float(match["confidence"])
                    recognized = bool(match.get("stable", False))
                    top_preds = [
                        {"label": gesture, "confidence": confidence},
                    ]
                    if recognized and english and gesture != active_spoken_sign:
                        active_spoken_sign = gesture
                        no_match_frames = 0
                        _SPEAKER.speak(english)
                else:
                    no_match_frames += 1
                    if no_match_frames > 15:
                        active_spoken_sign = None

                with self._lock:
                    self._latest_hands = raw_hands if detected else []
                    self._hands_detected = detected
                    self._latest_prediction.update(
                        {
                            "running": True,
                            "gesture": gesture,
                            "english": english,
                            "hindi": hindi,
                            "recognized": recognized,
                            "confidence": round(confidence, 3),
                            "hands_detected": detected,
                            "buffer_fill": buf_len,
                            "buffer_capacity": cap_len,
                            "top_predictions": top_preds,
                            "source": source,
                            "latency_ms": round(latency_ms, 1),
                            "error": "",
                        }
                    )
        except Exception as exc:
            with self._lock:
                self._latest_prediction["error"] = str(exc)
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
                    "supported_translations": TRANSLATIONS,
                }
            )
            return

        if self.path == "/api/vocabulary":
            vocab = _read_json(WEB_DIR / "isl_vocabulary.json")
            self._send_json({"vocabulary": vocab or []})
            return

        if self.path == "/api/custom-signs":
            self._send_json({"custom_signs": _ISL_CONTROLLER.get_custom_signs()})
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

        if self.path == "/api/camera/speak":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                data = json.loads(raw_body)
                text = str(data.get("text", "")).strip()
                if text:
                    _SPEAKER.speak(text)
                self._send_json({"spoken": True, "text": text})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/camera/stop":
            self._send_json(_ISL_CONTROLLER.stop())
            return

        if self.path == "/api/camera/reset":
            _ISL_CONTROLLER.request_reset()
            self._send_json({"reset": True})
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

        if self.path == "/api/custom-signs/record":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                data = json.loads(raw_body)
                label = str(data.get("label", "")).strip()
                if not label:
                    self._send_json({"error": "Missing sign label"}, status=HTTPStatus.BAD_REQUEST)
                    return
                res = _ISL_CONTROLLER.record_custom_sample(label)
                self._send_json(res)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/custom-signs/delete":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                data = json.loads(raw_body)
                label = str(data.get("label", "")).strip()
                res = _ISL_CONTROLLER.delete_sign(label)
                self._send_json(res)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
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
        _ISL_CONTROLLER.stop()
        server.server_close()


if __name__ == "__main__":
    main()
