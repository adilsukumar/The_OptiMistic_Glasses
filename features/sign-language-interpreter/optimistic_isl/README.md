# Optimistic ISL — Indian Sign Language Recognition for Smart Glasses

End-to-end, offline-capable Indian Sign Language (ISL) recognition pipeline built for **Optimistic Smart Glasses** powered by **Raspberry Pi 4 / Raspberry Pi 5** and an attached USB or CSI camera module.

---

## 1. Primary Dataset & Inspection Findings

- **Dataset**: `ai4bharat/INCLUDE` ([Hugging Face](https://huggingface.co/datasets/ai4bharat/INCLUDE), videos hosted on [Zenodo Record 4010759](https://zenodo.org/records/4010759))
- **License**: `CC-BY-4.0`
- **Metadata Schema**: `parent_label` (string), `label` (string), `video_path` (string), `include_50` (bool)
- **Full Dataset Scale**: 5,250 raw metadata rows (`3,816` train / `425` val / `1,009` test), representing **4,257 unique videos** across **262 classes** and **15 parent categories** (plus the 50-class `INCLUDE-50` subset of 943 unique videos).
- **Critical Inspection Findings**:
  1. **Duplicate `video_path` Leakage**: 969 `video_path` entries appear multiple times in the Hugging Face table (creating 424 cross-split overlaps between raw `train`, `val`, and `test`).
  2. **Label/Folder Mismatches**: 49 Hugging Face rows disagree with their folder path (e.g., `label="35. Bank"` pointing to `Places/26. University/MVI_3376.MOV`).
  3. **Resolution**: Our pipeline deduplicates strictly by canonical `video_path`, file MD5, and landmark sequence hash, uses the verified folder hierarchy as ground truth, and builds fresh leakage-resistant splits (`train` / `validation` / `test`).

---

## 2. Project Structure

```text
optimistic_isl/
├── config.py                     # Central paths, dataset, training & inference configs
├── requirements.txt              # Workstation / training Python dependencies
├── dataset/
│   ├── inspect_dataset.py        # Inspect HF metadata, Zenodo manifest & local videos
│   ├── download_dataset.py       # Resumable Zenodo downloader with MD5 verification
│   ├── extract_landmarks.py      # MediaPipe 21-point extraction & wrist/scale normalization
│   ├── preprocess.py             # End-to-end preprocessing, stats & visualizations
│   └── split_dataset.py          # Leakage-resistant stratified split builder
├── models/
│   ├── model_architecture.py     # MLP, Unrolled GRU, 1D TCN & Hybrid architectures
│   ├── train.py                  # Reproducible Keras training & architecture comparison
│   ├── evaluate.py               # Independent test evaluation, F1 & confusion matrix
│   └── export_tflite.py          # FP32 / Dynamic-Range / INT8 TFLite export & benchmark
├── inference/
│   ├── hand_detector.py          # Low-latency MediaPipe hand detector & overlay
│   ├── gesture_predictor.py      # Rolling 30-frame buffer & TFLite predictor
│   ├── prediction_smoother.py    # EMA + consistency buffer + confidence gating
│   └── camera_inference.py       # Live webcam / Pi camera recognition app
├── integration/
│   ├── output_interface.py       # Decoupled thread-safe JSON state & callback hub
│   ├── websocket_server.py       # Offline local WebSocket broadcaster
│   └── integration_example.py    # Python API & WebSocket consumer examples
├── deployment/                   # Self-contained Raspberry Pi 4 / Pi 5 bundle
│   ├── optimistic_model.tflite
│   ├── labels.json
│   ├── hand_landmarker.task
│   ├── camera_inference.py
│   ├── gesture_predictor.py
│   ├── hand_detector.py
│   ├── requirements_pi.txt
│   └── setup_pi.sh
├── evaluation/                   # Metrics JSON, confusion matrix & training curves
├── processed_dataset/            # Train/val/test .npy splits, stats & visualizations
└── tests/                        # Pytest suite (preprocessing, model, inference)
```

---

## 3. Quickstart Commands (Workstation)

### Step 1: Inspect the Dataset
```bash
python dataset/inspect_dataset.py
```

### Step 2: Download Videos & MediaPipe Asset
Download default practical smart-glasses categories (`Seasons`, `Greetings`, `Pronouns`), or pass `--categories ...`, or `--all --delete-zips` for the full ~56 GB INCLUDE archive:
```bash
python dataset/download_dataset.py --categories Seasons Greetings
```

### Step 3: Preprocess & Extract Landmarks
```bash
python dataset/preprocess.py
```

### Step 4: Train & Compare Models
Compare `mlp_static`, `gru_temporal`, and `tcn_temporal`, then train the best model:
```bash
python models/train.py --compare --epochs 120 --batch-size 16
```
Or train a specific architecture directly:
```bash
python models/train.py --arch tcn_temporal
```

### Step 5: Evaluate on Independent Test Set
```bash
python models/evaluate.py
```

### Step 6: Export Quantized TensorFlow Lite Model
```bash
python models/export_tflite.py
```

### Step 7: Run Automated Tests
```bash
pytest tests -v
```

### Step 8: Run Real-Time Webcam Recognition
```bash
python inference/camera_inference.py --camera 0 --threshold 0.70
```
Press `q` to quit. Pass `--allow-save-unrecognized` and press `s` to save consented unrecognized sequences locally into `unrecognized_samples/`.

---

## 4. Raspberry Pi 4 / Pi 5 Deployment (Optimistic Glasses)

1. Copy the self-contained `deployment/` directory to your Raspberry Pi:
   ```bash
   scp -r optimistic_isl/deployment pi@raspberrypi.local:~/optimistic_deployment
   ```
2. Run setup on the Raspberry Pi (64-bit Raspberry Pi OS recommended):
   ```bash
   cd ~/optimistic_deployment
   chmod +x setup_pi.sh
   ./setup_pi.sh
   ```
3. Launch headless smart-glasses inference (100% offline, zero internet required):
   - **USB Camera**:
     ```bash
     source .venv_pi/bin/activate
     python3 camera_inference.py --headless --ws-host 127.0.0.1 --ws-port 8765
     ```
   - **Raspberry Pi CSI Camera Module**:
     ```bash
     python3 camera_inference.py --headless --picamera2 --ws-host 127.0.0.1 --ws-port 8765
     ```

---

## 5. Optimistic Smart Glasses Output Format

Both the local Python `OptimisticOutputInterface` and the local WebSocket server (`ws://127.0.0.1:8765`) emit structured JSON:

```json
{
  "gesture": "HELLO",
  "confidence": 0.94,
  "recognized": true,
  "timestamp": "2026-11-27T09:15:00+00:00"
}
```

When confidence falls below threshold (default `0.70`) or frames are unstable, it outputs:

```json
{
  "gesture": "Gesture not recognized",
  "confidence": 0.42,
  "recognized": false,
  "timestamp": "2026-11-27T09:15:01+00:00"
}
```

See `integration/integration_example.py` for Python callback and WebSocket client examples.

---

## 6. Observed Results (INCLUDE `Pronouns` + `Greetings` + `Seasons` — 23 Classes)

- **Processed Dataset**: 443 unique videos discovered -> **434 valid sequences** (9 rejected for `excessive_missing_hands`), split without leakage into **280 train / 79 validation / 75 test** (`(30, 128)` tensors, mean missing-hand ratio `0.1725`).
- **Architecture Comparison (Validation Set, 79 samples)**:
  - `mlp_static` (`26,775` params): `0.6456` val accuracy, `1.0617` val loss
  - `gru_temporal` (`58,487` params): `0.5823` val accuracy, `1.4162` val loss
  - `tcn_temporal` (`97,015` params, **selected**): `0.9114` val accuracy, `0.4292` val loss (`0.9186` val macro F1)
- **Independent Test Set (`tcn_temporal`, 75 samples)**:
  - **Accuracy**: `0.8533` (`85.33%`)
  - **Macro Precision / Recall / F1**: `0.8627` / `0.8514` / `0.8419` (`0.8395` weighted F1)
  - **Confidence-Gated (`>= 0.70`)**: `78.67%` coverage at **`96.61%` accuracy**
  - **Weakest Signs**: `SHE` (`F1=0.0000`), `YOU (PLURAL)` (`F1=0.2500`), `FALL` (`F1=0.5000`), `HE` (`F1=0.6667`), `YOU` (`F1=0.6667`)
- **TFLite Export (`deployment/optimistic_model.tflite`)**:
  - `fp32`: `387.18 KB`, `0.8533` test accuracy, `0.116 ms` mean CPU latency
  - `dynamic_range` (**selected**): `116.95 KB`, `0.8400` test accuracy, `0.086 ms` mean CPU latency (100% pure `TFLITE_BUILTINS`, zero Flex ops)
  - `full_int8` (`deployment/optimistic_model_int8.tflite`): `128.80 KB`, `0.8533` test accuracy, `0.103 ms` mean CPU latency
- **Automated Tests**: `pytest tests -v` -> **9 passed**
- **Environment Notes / Remaining Physical Hardware Checks**:
  - Live physical webcam & Raspberry Pi 4/5 hardware camera (`Picamera2` / USB V4L2) and HUD display execution remain to be tested on the target Raspberry Pi device.

