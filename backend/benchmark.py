"""
benchmark.py - Measure model performance metrics for the paper table.
Outputs: Accuracy, Precision, Recall, F1, Model Size, Inference Time, Latency, FPS
"""

import os, sys, time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from utils import IDX_TO_LABEL, NUM_CLASSES, preprocess
from model import ASLMLP

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model", "asl_mlp.pth")
WARMUP_RUNS = 50
TIMING_RUNS = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL_PATH, map_location=device)
input_size  = ckpt.get("input_size", 93)
num_classes = ckpt.get("num_classes", NUM_CLASSES)
model = ASLMLP(input_size=input_size, num_classes=num_classes).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ── Model size ────────────────────────────────────────────────────────────────
model_size_kb = os.path.getsize(MODEL_PATH) / 1024

# ── Inference time (single sample) ───────────────────────────────────────────
dummy = torch.randn(1, input_size).to(device)

# warmup
for _ in range(WARMUP_RUNS):
    with torch.no_grad():
        _ = model(dummy)

# timed runs
times = []
for _ in range(TIMING_RUNS):
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy)
    times.append((time.perf_counter() - t0) * 1000)  # ms

inference_ms = np.mean(times)
fps = 1000.0 / inference_ms

# ── Latency (preprocess + inference) ─────────────────────────────────────────
raw_lm = np.random.rand(63).tolist()
latency_times = []
for _ in range(TIMING_RUNS):
    t0 = time.perf_counter()
    feat = preprocess(raw_lm)
    x = torch.from_numpy(feat).float().unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(x)
    latency_times.append((time.perf_counter() - t0) * 1000)

latency_ms = np.mean(latency_times)
max_fps = 1000.0 / latency_ms

# ── Test set evaluation ───────────────────────────────────────────────────────
TEST_ROOT = os.path.join(os.path.dirname(__file__), "..", "asl_dataset",
                         "asl_alphabet_test", "asl_alphabet_test")

all_preds, all_labels = [], []

if os.path.exists(TEST_ROOT):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from utils import LABEL_TO_IDX

    base_options = mp_python.BaseOptions(
        model_asset_path=os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    for fname in sorted(os.listdir(TEST_ROOT)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        orig_label = fname.split("_")[0]
        label_key = orig_label.upper() if len(orig_label) == 1 else orig_label.lower()
        if label_key not in LABEL_TO_IDX:
            continue
        fpath = os.path.join(TEST_ROOT, fname)
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = detector.detect(mp_img)
        if not result.hand_landmarks:
            continue
        lm = result.hand_landmarks[0]
        raw = [coord for pt in lm for coord in (pt.x, pt.y, pt.z)]
        feat = preprocess(raw)
        x = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(1).item()
        all_preds.append(pred)
        all_labels.append(LABEL_TO_IDX[label_key])

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  TABLE I — MODEL PERFORMANCE METRICS")
print("="*50)
print(f"  {'Metric':<25} {'Value':>10}")
print("-"*50)

if all_labels:
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    print(f"  {'Classification Accuracy':<25} {acc*100:>9.2f}%")
    print(f"  {'Precision':<25} {prec*100:>9.2f}%")
    print(f"  {'Recall':<25} {rec*100:>9.2f}%")
    print(f"  {'F1 Score':<25} {f1*100:>9.2f}%")
else:
    print("  [SKIP] Test set not found — skipping accuracy metrics")

print(f"  {'Model Size':<25} {model_size_kb:>8.0f} KB")
print(f"  {'Inference Time':<25} {inference_ms:>8.2f} ms")
print(f"  {'Latency':<25} {latency_ms:>8.2f} ms")
print(f"  {'Maximum FPS':<25} {max_fps:>8.0f} FPS")
print("="*50)
print(f"\n  Device: {device} | Input size: {input_size} | Classes: {num_classes}")
print(f"  Test samples evaluated: {len(all_labels)}")
