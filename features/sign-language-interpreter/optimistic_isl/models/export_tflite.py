"""Export trained Keras model to quantized TensorFlow Lite for Raspberry Pi."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import keras
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_tflite")


def benchmark_tflite_model(
    tflite_bytes: bytes,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Run TFLite interpreter on X_test to verify accuracy and CPU latency."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    preds = []
    latencies_ms = []
    in_dtype = input_details["dtype"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    out_scale, out_zero = output_details.get("quantization", (0.0, 0))

    for i in range(len(X_test)):
        sample = X_test[i : i + 1].astype(np.float32)
        if in_dtype == np.int8 and in_scale > 0:
            sample = np.round(sample / in_scale + in_zero).clip(-128, 127).astype(np.int8)

        t0 = time.perf_counter()
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        if output_details["dtype"] == np.int8 and out_scale > 0:
            out = (out.astype(np.float32) - out_zero) * out_scale
        preds.append(int(np.argmax(out[0])))

    acc = float(np.mean(y_test == np.asarray(preds, dtype=np.int32)))
    return {
        "size_kb": round(len(tflite_bytes) / 1024.0, 2),
        "test_accuracy": round(acc, 4),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "input_dtype": str(in_dtype),
    }


def export_tflite_variants(
    model_path: Path = config.BEST_MODEL_PATH,
    out_tflite_path: Path = config.TFLITE_MODEL_PATH,
) -> Dict[str, Any]:
    """Export FP32, Dynamic Range Quantized, and Full-INT8 TFLite models and compare."""
    if not model_path.exists():
        raise FileNotFoundError(f"Trained Keras model not found: {model_path}")

    config.ensure_directories()
    base_model = keras.models.load_model(model_path)

    X_train = np.load(config.TRAIN_DIR / "X.npy").astype(np.float32)
    X_test = np.load(config.TEST_DIR / "X.npy").astype(np.float32)
    y_test = np.load(config.TEST_DIR / "y.npy").astype(np.int32)

    # Wrap with explicit batch=1 input signature for pure TFLITE_BUILTINS compatibility
    seq_len, feat_dim = X_test.shape[1], X_test.shape[2]
    serving_input = keras.Input(shape=(seq_len, feat_dim), batch_size=1, name="serving_input")
    serving_output = base_model(serving_input, training=False)
    serving_model = keras.Model(serving_input, serving_output)

    results: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_dir = Path(tmpdir) / "saved_model"
        serving_model.export(str(saved_dir))

        # 1. FP32 baseline
        conv_fp32 = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
        conv_fp32.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        fp32_bytes = conv_fp32.convert()
        results["fp32"] = benchmark_tflite_model(fp32_bytes, X_test, y_test)

        # 2. Dynamic range quantization (required default)
        conv_dyn = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
        conv_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
        conv_dyn.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        dyn_bytes = conv_dyn.convert()
        results["dynamic_range"] = benchmark_tflite_model(dyn_bytes, X_test, y_test)

        # 3. Full INT8 quantization check
        def rep_dataset():
            for i in range(min(100, len(X_train))):
                yield [X_train[i : i + 1]]

        int8_bytes = None
        try:
            conv_int8 = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
            conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
            conv_int8.representative_dataset = rep_dataset
            conv_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv_int8.inference_input_type = tf.int8
            conv_int8.inference_output_type = tf.int8
            int8_bytes = conv_int8.convert()
            results["full_int8"] = benchmark_tflite_model(int8_bytes, X_test, y_test)
        except Exception as exc:
            results["full_int8"] = {"error": str(exc)}

    # Choose dynamic_range unless INT8 matches/exceeds accuracy within 0.01 and shrinks further
    chosen_name = "dynamic_range"
    chosen_bytes = dyn_bytes
    if (
        int8_bytes is not None
        and "test_accuracy" in results.get("full_int8", {})
        and results["full_int8"]["test_accuracy"] >= results["dynamic_range"]["test_accuracy"] - 0.005
    ):
        logger.info("Full INT8 maintains accuracy (%.4f); selecting dynamic_range or int8.", results["full_int8"]["test_accuracy"])
        # Keep dynamic_range as primary float32-IO deployment artifact for seamless Pi compatibility
        # unless user wants int8; also save int8 separately if beneficial.
        (out_tflite_path.parent / "optimistic_model_int8.tflite").write_bytes(int8_bytes)

    out_tflite_path.write_bytes(chosen_bytes)
    if config.LABEL_MAPPING_PATH.exists():
        shutil.copy2(config.LABEL_MAPPING_PATH, config.DEPLOYMENT_LABELS_PATH)
    if config.HAND_LANDMARKER_TASK_PATH.exists():
        shutil.copy2(
            config.HAND_LANDMARKER_TASK_PATH,
            out_tflite_path.parent / config.HAND_LANDMARKER_TASK_PATH.name,
        )

    report = {
        "selected_variant": chosen_name,
        "tflite_path": str(out_tflite_path),
        "labels_path": str(config.DEPLOYMENT_LABELS_PATH),
        "benchmarks": results,
    }
    with open(config.EVALUATION_DIR / "tflite_export_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Exported TFLite model (%s, %.2f KB) to %s", chosen_name, len(chosen_bytes) / 1024.0, out_tflite_path)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained model to TFLite.")
    parser.add_argument("--model", type=Path, default=config.BEST_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=config.TFLITE_MODEL_PATH)
    args = parser.parse_args()
    export_tflite_variants(model_path=args.model, out_tflite_path=args.output)


if __name__ == "__main__":
    main()
