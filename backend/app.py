"""
app.py - Flask REST API for ASL recognition.

Endpoints:
  POST /predict   { "landmarks": [63 floats] }  → prediction + confidence + confusion
  GET  /health    → health check

Novel features:
  1. Confusion-aware: returns top-2 if confidence gap < threshold
  2. Adaptive threshold: looser for visually similar pairs
  3. Stability status: stable / uncertain
"""

import os
import sys
import json
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
from utils import (IDX_TO_LABEL, NUM_CLASSES, preprocess,
                   get_confusion_labels, CONFUSION_PAIRS, compute_geometric_features)
from model import ASLMLP, load_model

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model", "asl_mlp.pth")
CONFIDENCE_THRESHOLD = 0.60   # minimum confidence to report stable
CONFUSION_GAP = 0.15          # if top-2 gap < this → confusion warning
ADAPTIVE_PAIRS = set()        # filled from CONFUSION_PAIRS keys
for k in CONFUSION_PAIRS:
    ADAPTIVE_PAIRS.add(k)

app = Flask(__name__)
CORS(app, origins="*")

# ─── Load model ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
input_size = 63


def load():
    global model, input_size
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model not found at {MODEL_PATH}. Train first with train.py")
        return
    ckpt = torch.load(MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict):
        input_size = ckpt.get("input_size", 63)
        num_classes = ckpt.get("num_classes", NUM_CLASSES)
    else:
        num_classes = NUM_CLASSES
    model = ASLMLP(input_size=input_size, num_classes=num_classes).to(device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"[INFO] Model loaded from {MODEL_PATH} (input_size={input_size})")


load()


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
        "input_size": input_size,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    data = request.get_json(force=True)
    landmarks = data.get("landmarks")

    # ── Input validation ──
    if not landmarks or len(landmarks) != 63:
        return jsonify({"error": f"Expected 63 landmarks, got {len(landmarks) if landmarks else 0}"}), 400

    try:
        # ── Preprocess ──
        features = preprocess(landmarks)                    # shape: (input_size,)
        if len(features) != input_size:
            # Pad or truncate to match trained model
            if len(features) < input_size:
                features = np.pad(features, (0, input_size - len(features)))
            else:
                features = features[:input_size]

        x = torch.from_numpy(features).float().unsqueeze(0).to(device)  # (1, input_size)

        # ── Inference ──
        with torch.no_grad():
            logits = model(x)                               # (1, num_classes)
            probs  = torch.softmax(logits, dim=1)[0]       # (num_classes,)

        top2_probs, top2_idx = torch.topk(probs, k=2)
        top1_idx   = top2_idx[0].item()
        top1_prob  = top2_probs[0].item()
        top2_idx_v = top2_idx[1].item()
        top2_prob  = top2_probs[1].item()

        pred_label = IDX_TO_LABEL.get(top1_idx, "?")
        conf = round(top1_prob, 4)
        gap  = top1_prob - top2_prob

        # ── Adaptive threshold for confusion-prone letters ──
        threshold = CONFIDENCE_THRESHOLD
        if pred_label in ADAPTIVE_PAIRS:
            threshold = CONFIDENCE_THRESHOLD + 0.05   # stricter for similar signs

        # ── Status & confusion ──
        status = "stable" if conf >= threshold else "uncertain"
        confusion = []
        if gap < CONFUSION_GAP or conf < threshold:
            runner_up = IDX_TO_LABEL.get(top2_idx_v, "?")
            confusion = [runner_up] + get_confusion_labels(pred_label)
            confusion = list(dict.fromkeys(confusion))  # deduplicate, preserve order
            status = "confused"

        # ── Geometric features for frontend visualization ──
        geo = compute_geometric_features(landmarks)
        num_distances = 10
        distances = [round(float(v), 4) for v in geo[:num_distances]]
        angles    = [round(float(v), 4) for v in geo[num_distances:]]

        return jsonify({
            "prediction": pred_label,
            "confidence": conf,
            "status": status,
            "possible_confusions": confusion,
            "adaptive": pred_label in ADAPTIVE_PAIRS,
            "top2": {
                IDX_TO_LABEL.get(top2_idx[0].item(), "?"): round(top2_probs[0].item(), 4),
                IDX_TO_LABEL.get(top2_idx[1].item(), "?"): round(top2_probs[1].item(), 4),
            },
            "geometric": {
                "distances": distances,
                "angles": angles,
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
