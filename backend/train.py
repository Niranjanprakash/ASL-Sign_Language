"""
train.py - Training pipeline for ASL MLP model.

Steps:
  1. Walk asl_dataset/asl_alphabet_train/<CLASS>/*.jpg
  2. Run MediaPipe Hands on each image to extract 21 landmarks (63 floats)
  3. Apply preprocess() (normalize + geometric features)
  4. Train MLP with CrossEntropyLoss + Adam
  5. Evaluate on asl_alphabet_test images
  6. Save checkpoint to saved_model/asl_mlp.pth

Usage:
  cd backend
  python train.py
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils import LABELS, LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES, preprocess
from model import ASLMLP

# ─── Constants ────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "asl_dataset", "asl_alphabet_train", "asl_alphabet_train"
)
TEST_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "asl_dataset", "asl_alphabet_test", "asl_alphabet_test"
)
SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "asl_mlp.pth")

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.1
MAX_SAMPLES_PER_CLASS = None   # None = use all samples (full 87,000 image dataset)


# ─── MediaPipe Hands setup ────────────────────────────────────────────────────
def create_mp_detector():
    """Create a MediaPipe Hands detector using the Tasks API (static image mode)."""
    base_options = mp_python.BaseOptions(
        model_asset_path=os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ─── Landmark extraction ──────────────────────────────────────────────────────
def extract_landmarks_from_image(detector, image_path: str):
    """
    Extract 63 raw landmark floats from an image file.
    Returns list of 63 floats or None if no hand detected.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    return [coord for pt in lm for coord in (pt.x, pt.y, pt.z)]


# ─── Dataset builder ──────────────────────────────────────────────────────────
def build_dataset(detector, root: str, max_per_class=None, split_name="train"):
    """
    Walk root/<CLASS>/*.jpg, extract landmarks, preprocess.
    Returns (features_list, labels_list, feature_size).
    """
    features, labels = [], []
    skipped_no_hand, skipped_no_dir = 0, 0
    total_images = 0

    class_dirs = sorted(os.listdir(root))
    print(f"\n[{split_name}] Processing {len(class_dirs)} classes in: {root}")

    for class_name in class_dirs:
        class_dir = os.path.join(root, class_name)
        
        # If it's a flat directory with files like A_test.jpg:
        if not os.path.isdir(class_dir):
            if class_name.lower().endswith((".jpg", ".jpeg", ".png")):
                # Extract label "A" from "A_test.jpg"
                orig_label = class_name.split("_")[0]
                label_key = orig_label.upper() if len(orig_label) == 1 else orig_label.lower()
                if label_key in LABEL_TO_IDX:
                    label_idx = LABEL_TO_IDX[label_key]
                    raw = extract_landmarks_from_image(detector, class_dir)
                    if raw is not None:
                        features.append(preprocess(raw))
                        labels.append(label_idx)
                        total_images += 1
                    else:
                        skipped_no_hand += 1
            continue

        # Map folder name to label index
        # Accept A-Z directly; skip del/nothing/space for 26-class model
        label_key = class_name.upper() if len(class_name) == 1 else class_name.lower()
        if label_key not in LABEL_TO_IDX:
            print(f"  [SKIP] Unknown class folder: {class_name}")
            skipped_no_dir += 1
            continue

        label_idx = LABEL_TO_IDX[label_key]
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if max_per_class:
            image_files = image_files[:max_per_class]

        n_class = 0
        t0 = time.time()
        for fname in image_files:
            total_images += 1
            fpath = os.path.join(class_dir, fname)
            raw = extract_landmarks_from_image(detector, fpath)
            if raw is None:
                skipped_no_hand += 1
                continue
            feat = preprocess(raw)
            features.append(feat)
            labels.append(label_idx)
            n_class += 1

        elapsed = time.time() - t0
        print(f"  [{split_name}] {class_name:8s}: {n_class:4d}/{len(image_files)} samples "
              f"({elapsed:.1f}s)")

    print(f"\n[{split_name}] Total images processed: {total_images}")
    print(f"[{split_name}] Skipped (no hand):       {skipped_no_hand}")
    print(f"[{split_name}] Valid samples:            {len(features)}")

    if not features:
        raise RuntimeError(f"No valid samples found in {root}. Check MediaPipe model file.")

    feat_size = len(features[0])
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64), feat_size


# ─── Custom Dataset ────────────────────────────────────────────────────────────
class ASLDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Training loop ────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ASL MLP Training Pipeline")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Build dataset
    detector = create_mp_detector()
    X, y, feat_size = build_dataset(
        detector, DATASET_ROOT,
        max_per_class=MAX_SAMPLES_PER_CLASS,
        split_name="TRAIN"
    )
    print(f"\nFeature size: {feat_size}")
    print(f"Classes found: {np.unique(y)}")

    # Train / val split
    dataset = ASLDataset(X, y)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = ASLMLP(input_size=feat_size, num_classes=NUM_CLASSES).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_val_acc = 0.0
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        train_acc = correct / total
        avg_loss = total_loss / total

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                val_correct += (preds == yb).sum().item()
                val_total += len(yb)
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": feat_size,
                "num_classes": NUM_CLASSES,
                "labels": LABELS,
                "val_acc": val_acc,
                "epoch": epoch,
            }, MODEL_PATH)
            print(f"  ✓ Model saved (val_acc={val_acc:.4f})")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # ── Test evaluation ──
    print("\nEvaluating on test set...")
    X_test, y_test, _ = build_dataset(detector, TEST_ROOT, split_name="TEST")
    test_ds = ASLDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load best checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    label_names = [IDX_TO_LABEL[i] for i in sorted(IDX_TO_LABEL.keys())]
    print("\n─── Classification Report ───")
    print(classification_report(
        all_labels, all_preds,
        target_names=label_names,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL MLP model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_PER_CLASS)
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    MAX_SAMPLES_PER_CLASS = args.max_samples

    train(args)
