"""Real-time camera ISL recognition application for workstation and Raspberry Pi."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
logger = logging.getLogger("camera_inference")


def open_camera(
    camera_index: int,
    width: int,
    height: int,
    use_picamera2: bool = False,
    video_path: Optional[Path] = None,
):
    """Open a video file, Raspberry Pi Picamera2, or USB/V4L2 OpenCV camera."""
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file {video_path}")
        return ("video", cap)
    if use_picamera2:
        try:
            from picamera2 import Picamera2  # type: ignore
            picam = Picamera2()
            cfg = picam.create_preview_configuration(
                main={"format": "RGB888", "size": (width, height)}
            )
            picam.configure(cfg)
            picam.start()
            return ("picamera2", picam)
        except Exception as exc:
            logger.warning("Picamera2 unavailable (%s); falling back to cv2.VideoCapture.", exc)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    return ("opencv", cap)


def read_camera_frame(cam_tuple) -> Optional[np.ndarray]:
    kind, handle = cam_tuple
    if kind == "picamera2":
        rgb = handle.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, frame = handle.read()
    return frame if ok else None


def close_camera(cam_tuple) -> None:
    kind, handle = cam_tuple
    if kind == "picamera2":
        handle.stop()
    else:
        handle.release()


def save_unrecognized_sample(
    sequence: np.ndarray,
    consent_granted: bool,
    out_dir: Path = config.UNRECOGNIZED_SAMPLES_DIR,
) -> Optional[Path]:
    """Save an unrecognized landmark sequence only when user consent is explicitly given."""
    if not consent_granted:
        logger.info("Sample save skipped: user consent not granted.")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    np_path = out_dir / f"unrecognized_{ts}.npy"
    meta_path = out_dir / f"unrecognized_{ts}.json"
    np.save(np_path, sequence.astype(np.float32))
    meta_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "consent_granted": True,
                "sequence_shape": list(sequence.shape),
                "notes": "Saved locally by user for future ISL dataset improvement.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved consented unrecognized sample to %s", np_path)
    return np_path


def run_camera_loop(
    camera_index: int = config.INFERENCE_CONFIG.camera_index,
    model_path: Path = config.TFLITE_MODEL_PATH,
    labels_path: Path = config.DEPLOYMENT_LABELS_PATH,
    confidence_threshold: float = config.INFERENCE_CONFIG.confidence_threshold,
    headless: bool = False,
    enable_websocket: bool = True,
    use_picamera2: bool = False,
    allow_save_unrecognized: bool = False,
    json_sink: Optional[Path] = None,
    video_path: Optional[Path] = None,
) -> None:
    """Execute the real-time camera recognition and Optimistic output loop."""
    detector = RealTimeHandDetector()
    predictor = GesturePredictor(
        model_path=model_path,
        labels_path=labels_path,
        confidence_threshold=confidence_threshold,
    )
    output_hub = OptimisticOutputInterface(json_sink_path=json_sink)
    ws_server = OptimisticWebSocketServer(output_hub) if enable_websocket else None
    if ws_server is not None:
        ws_server.start()

    cam = open_camera(
        camera_index,
        config.INFERENCE_CONFIG.frame_width,
        config.INFERENCE_CONFIG.frame_height,
        use_picamera2=use_picamera2,
        video_path=video_path,
    )

    frame_counter = 0
    latest_payload = output_hub.get_latest()
    logger.info("Real-time ISL recognition started. Press 'q' to quit, 's' to save consented unrecognized sample.")

    try:
        while True:
            frame = read_camera_frame(cam)
            if frame is None:
                if cam[0] == "video":
                    break
                logger.warning("Failed to grab camera frame; retrying...")
                time.sleep(0.02)
                continue

            frame_counter += 1
            feat, detected, raw_hands = detector.process_bgr_frame(frame)
            run_inf = (frame_counter % config.INFERENCE_CONFIG.inference_stride) == 0
            pred = predictor.push_frame(feat, run_inference=run_inf)

            if not pred.get("skipped_stride", False):
                latest_payload = output_hub.publish(pred)

            if not headless:
                RealTimeHandDetector.draw_landmarks(frame, raw_hands)
                status = (
                    f"RECOGNIZED: {latest_payload['gesture']} ({latest_payload['confidence']*100:.1f}%)"
                    if latest_payload["recognized"]
                    else f"{latest_payload['gesture']} ({latest_payload['confidence']*100:.1f}%)"
                )
                banner_color = (0, 170, 60) if latest_payload["recognized"] else (0, 100, 220)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 68), (20, 20, 20), -1)
                cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, banner_color, 2)
                buf_info = (
                    f"Hands: {'YES' if detected else 'NO'} | "
                    f"Buffer: {pred['buffer_fill']}/{pred['buffer_capacity']}"
                )
                cv2.putText(frame, buf_info, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
                cv2.imshow("Optimistic Smart Glasses - ISL Recognition", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s") and not latest_payload["recognized"]:
                    seq = predictor.get_buffered_sequence()
                    if seq is not None:
                        save_unrecognized_sample(seq, consent_granted=allow_save_unrecognized)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        close_camera(cam)
        if not headless:
            cv2.destroyAllWindows()
        if ws_server is not None:
            ws_server.stop()
        detector.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimistic real-time ISL camera inference.")
    parser.add_argument("--camera", type=int, default=config.INFERENCE_CONFIG.camera_index)
    parser.add_argument("--model", type=Path, default=config.TFLITE_MODEL_PATH)
    parser.add_argument("--labels", type=Path, default=config.DEPLOYMENT_LABELS_PATH)
    parser.add_argument("--threshold", type=float, default=config.INFERENCE_CONFIG.confidence_threshold)
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV GUI window (recommended on Pi glasses).")
    parser.add_argument("--no-ws", action="store_true", help="Disable local WebSocket server.")
    parser.add_argument("--picamera2", action="store_true", help="Use Raspberry Pi Picamera2 backend.")
    parser.add_argument(
        "--allow-save-unrecognized",
        action="store_true",
        help="User consent flag allowing 's' key to save unrecognized sequences locally.",
    )
    parser.add_argument("--json-sink", type=Path, default=None, help="Optional JSON file path updated with latest output.")
    parser.add_argument("--video", type=Path, default=None, help="Optional video file path for offline replay testing.")
    args = parser.parse_args()

    run_camera_loop(
        camera_index=args.camera,
        model_path=args.model,
        labels_path=args.labels,
        confidence_threshold=args.threshold,
        headless=args.headless,
        enable_websocket=not args.no_ws,
        use_picamera2=args.picamera2,
        allow_save_unrecognized=args.allow_save_unrecognized,
        json_sink=args.json_sink,
        video_path=args.video,
    )


if __name__ == "__main__":
    main()
