"""Unit and integration tests for prediction smoothing, TFLite inference, and Optimistic output."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.gesture_predictor import GesturePredictor
from inference.prediction_smoother import PredictionSmoother
from integration.output_interface import OptimisticOutputInterface
from integration.websocket_server import OptimisticWebSocketServer
from models.model_architecture import build_model


def test_prediction_smoother_rejects_low_confidence_and_single_spike() -> None:
    smoother = PredictionSmoother(
        class_names=["SUMMER", "WINTER", "SPRING"],
        confidence_threshold=0.70,
        window_size=5,
        min_consistent_frames=3,
        ema_alpha=0.5,
    )
    # Single high spike should NOT trigger recognition yet
    res1 = smoother.update(np.array([0.95, 0.03, 0.02], dtype=np.float32))
    assert not res1.recognized
    assert res1.gesture == "Gesture not recognized"

    # Consistent above-threshold frames stabilize into recognition
    smoother.update(np.array([0.95, 0.03, 0.02], dtype=np.float32))
    res3 = smoother.update(np.array([0.95, 0.03, 0.02], dtype=np.float32))
    assert res3.recognized
    assert res3.gesture == "SUMMER"
    assert res3.confidence >= 0.70

    # Low-confidence uncertain frames fall back to unrecognized
    smoother.reset()
    low = smoother.update(np.array([0.40, 0.35, 0.25], dtype=np.float32))
    assert not low.recognized
    assert low.gesture == "Gesture not recognized"


def test_gesture_predictor_tflite_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model = build_model("tcn_temporal", 30, 128, 3, batch_size=1)
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_path = tmp_path / "test.tflite"
        tflite_path.write_bytes(conv.convert())

        labels_path = tmp_path / "labels.json"
        labels_path.write_text(
            json.dumps(
                {
                    "num_classes": 3,
                    "id_to_display": {"0": "SUMMER", "1": "WINTER", "2": "SPRING"},
                }
            ),
            encoding="utf-8",
        )

        predictor = GesturePredictor(
            model_path=tflite_path,
            labels_path=labels_path,
            confidence_threshold=0.99,  # force threshold gating check
        )
        frame = np.zeros(128, dtype=np.float32)
        frame[126] = 1.0
        for i in range(29):
            out = predictor.push_frame(frame)
            assert not out["buffer_ready"]
        out_ready = predictor.push_frame(frame)
        assert out_ready["buffer_ready"]
        assert out_ready["gesture"] == "Gesture not recognized"
        assert not out_ready["recognized"]


def test_optimistic_output_and_websocket_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sink = Path(tmp) / "latest.json"
        hub = OptimisticOutputInterface(json_sink_path=sink)
        srv = OptimisticWebSocketServer(hub, host="127.0.0.1", port=0)
        srv.start()

        async def _check() -> None:
            uri = f"ws://127.0.0.1:{srv.actual_port}"
            async with websockets.connect(uri) as ws:
                msg0 = json.loads(await ws.recv())
                assert msg0["recognized"] is False
                hub.publish({"gesture": "HELLO", "confidence": 0.94, "recognized": True})
                msg1 = json.loads(await ws.recv())
                assert msg1["gesture"] == "HELLO"
                assert msg1["confidence"] == 0.94
                assert msg1["recognized"] is True

        try:
            asyncio.run(_check())
            saved = json.loads(sink.read_text(encoding="utf-8"))
            assert saved["gesture"] == "HELLO"
        finally:
            srv.stop()
