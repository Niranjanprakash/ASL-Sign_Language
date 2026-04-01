"""
evaluate_plots.py
Generates:
  1. Confusion Matrix
  2. Normalized Confusion Matrix
  3. Per-Class Accuracy Bar Chart
Uses the same val split (seed=42, 10%) as train.py.
"""

import os, sys, time
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from utils import LABELS, LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES, preprocess
from model import ASLMLP

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "asl_dataset",
                            "asl_alphabet_train", "asl_alphabet_train")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "saved_model", "asl_mlp.pth")
OUT_DIR      = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SAMPLES_PER_CLASS = 500   # increase for more accuracy; 500 → fast run ~5 min

# ── MediaPipe ─────────────────────────────────────────────────────────────────
def create_detector():
    base = mp_python.BaseOptions(
        model_asset_path=os.path.join(os.path.dirname(__file__), "hand_landmarker.task"))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base, num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE)
    return mp_vision.HandLandmarker.create_from_options(opts)

def extract(detector, path):
    img = cv2.imread(path)
    if img is None: return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks: return None
    lm = res.hand_landmarks[0]
    return [c for pt in lm for c in (pt.x, pt.y, pt.z)]

# ── Build feature array ───────────────────────────────────────────────────────
def build_features(detector):
    X, y = [], []
    for cls in sorted(os.listdir(DATASET_ROOT)):
        cls_dir = os.path.join(DATASET_ROOT, cls)
        if not os.path.isdir(cls_dir): continue
        key = cls.upper() if len(cls) == 1 else cls.lower()
        if key not in LABEL_TO_IDX: continue
        idx = LABEL_TO_IDX[key]
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg",".jpeg",".png"))][:MAX_SAMPLES_PER_CLASS]
        n = 0
        for f in files:
            raw = extract(detector, os.path.join(cls_dir, f))
            if raw is None: continue
            X.append(preprocess(raw)); y.append(idx); n += 1
        print(f"  {cls:8s}: {n}/{len(files)}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# ── Dataset ───────────────────────────────────────────────────────────────────
class ASLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_confusion(cm, labels, title, path, normalize=False):
    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(data, interpolation="nearest",
                   cmap="Blues" if not normalize else "YlOrRd")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted Label", ylabel="True Label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = data.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = f"{data[i,j]:.2f}" if normalize else str(int(cm[i,j]))
            ax.text(j, i, val, ha="center", va="center", fontsize=5,
                    color="white" if data[i,j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_per_class_accuracy(cm, labels, path):
    per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-9) * 100
    colors = ["#d62728" if v < 90 else "#ff7f0e" if v < 97 else "#2ca02c"
              for v in per_class]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(labels, per_class, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=np.mean(per_class), color="navy", linestyle="--",
               linewidth=1.2, label=f"Mean: {np.mean(per_class):.2f}%")
    ax.set(xlabel="ASL Gesture Class", ylabel="Accuracy (%)",
           title="Per-Class Accuracy of ASL Gesture Recognition",
           ylim=[max(0, per_class.min() - 5), 102])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=10)

    for bar, val in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=6.5, rotation=90)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nBuilding dataset (max {MAX_SAMPLES_PER_CLASS}/class)...\n")

    detector = create_detector()
    X, y = build_features(detector)
    print(f"\nTotal valid samples: {len(y)}")

    # Reproduce same val split as train.py
    dataset  = ASLDataset(X, y)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    _, val_ds = random_split(dataset, [train_size, val_size],
                             generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    print(f"Validation samples: {len(val_ds)}")

    # Load model
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = ASLMLP(input_size=ckpt["input_size"], num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded | val_acc from checkpoint: {ckpt.get('val_acc',0)*100:.2f}%\n")

    # Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Only keep classes that appear in val set
    present = sorted(set(all_labels) | set(all_preds))
    label_names = [IDX_TO_LABEL[i] for i in present]
    cm = confusion_matrix(all_labels, all_preds, labels=present)

    overall_acc = np.diag(cm).sum() / cm.sum() * 100
    print(f"Overall Accuracy: {overall_acc:.2f}%\n")
    print("Generating plots...")

    plot_confusion(cm, label_names,
                   "Confusion Matrix — ASL Gesture Recognition",
                   os.path.join(OUT_DIR, "confusion_matrix.png"))

    plot_confusion(cm, label_names,
                   "Normalized Confusion Matrix — ASL Gesture Recognition",
                   os.path.join(OUT_DIR, "confusion_matrix_normalized.png"),
                   normalize=True)

    plot_per_class_accuracy(cm, label_names,
                            os.path.join(OUT_DIR, "per_class_accuracy.png"))

    print(f"\nAll plots saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
