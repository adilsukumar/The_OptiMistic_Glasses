"""Unit tests for model architectures and TFLite export compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.model_architecture import build_model


def test_architectures_build_and_predict() -> None:
    x = np.random.default_rng(7).normal(0.0, 1.0, size=(2, 30, 128)).astype(np.float32)
    for arch in ("mlp_static", "gru_temporal", "tcn_temporal", "hybrid_cnn_gru"):
        model = build_model(
            architecture=arch,
            sequence_length=30,
            feature_dim=128,
            num_classes=6,
        )
        preds = model.predict(x, verbose=0)
        assert preds.shape == (2, 6)
        assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-4)


def test_tcn_converts_to_pure_tflite_builtins() -> None:
    model = build_model(
        architecture="tcn_temporal",
        sequence_length=30,
        feature_dim=128,
        num_classes=6,
        batch_size=1,
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_bytes = converter.convert()
    assert len(tflite_bytes) > 1000

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    sample = np.zeros((1, 30, 128), dtype=np.float32)
    interp.set_tensor(in_det["index"], sample)
    interp.invoke()
    out = interp.get_tensor(out_det["index"])
    assert out.shape == (1, 6)
    assert np.isclose(out.sum(), 1.0, atol=1e-4)
