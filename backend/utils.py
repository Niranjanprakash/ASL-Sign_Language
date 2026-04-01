"""
utils.py - Landmark preprocessing utilities for ASL recognition.
Handles normalization, geometric features extraction, and confusion pair detection.
"""

import numpy as np
import math

# ─── Label Mapping ────────────────────────────────────────────────────────────
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)  # 29

# ─── Confusion pairs (visually similar signs) ─────────────────────────────────
CONFUSION_PAIRS = {
    "A": ["S", "E"],
    "S": ["A", "E"],
    "E": ["A", "S"],
    "M": ["N"],
    "N": ["M"],
    "R": ["U"],
    "U": ["R"],
    "I": ["J"],
    "J": ["I"],
    "G": ["H"],
    "H": ["G"],
    "P": ["Q"],
    "Q": ["P"],
    "K": ["V"],
    "V": ["K"],
    "D": ["F"],
    "F": ["D"],
}

# ─── MediaPipe connection pairs (for skeleton drawing) ─────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


def normalize_landmarks(landmarks: list) -> np.ndarray:
    """
    Normalize 63 raw landmark values (21 * [x, y, z]).
    Steps:
      1. Reshape to (21, 3)
      2. Subtract wrist (landmark 0) to make hand-position invariant
      3. Scale by max absolute distance so values are in [-1, 1]
    Returns a flat np.ndarray of shape (63,).
    """
    lm = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    wrist = lm[0].copy()
    lm -= wrist                                    # translate to wrist origin
    scale = np.max(np.abs(lm)) + 1e-6            # prevent div-by-zero
    lm /= scale
    return lm.flatten()


def compute_geometric_features(landmarks: list) -> np.ndarray:
    """
    Compute geometric features from 63 raw landmark values.
    Uses wrist-centered, scale-normalized landmarks so distances are
    on the same scale as the normalized coordinate features.
      - 10 distances between all fingertip pairs (fingertips: 4,8,12,16,20)
      - 15 bend angles: 3 per finger (wrist-to-MCP, MCP-to-PIP, PIP-to-DIP)
    Returns a flat np.ndarray of shape (25,).
    """
    # Normalize to wrist origin + scale — same as normalize_landmarks
    lm = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    lm -= lm[0].copy()                              # wrist to origin
    scale = np.max(np.abs(lm)) + 1e-6
    lm /= scale

    # 10 fingertip-pair distances
    fingertips = [4, 8, 12, 16, 20]
    distances = []
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            diff = lm[fingertips[i]] - lm[fingertips[j]]
            distances.append(float(np.linalg.norm(diff)))

    # 15 bend angles: 3 per finger (wrist-to-MCP, MCP-to-PIP, PIP-to-DIP)
    # Each angle is at the middle joint of a triplet of consecutive landmarks
    finger_joints = [
        [0, 1, 2, 3, 4],    # thumb:  wrist→CMC→MCP→IP→tip
        [0, 5, 6, 7, 8],    # index:  wrist→MCP→PIP→DIP→tip
        [0, 9, 10, 11, 12], # middle: wrist→MCP→PIP→DIP→tip
        [0, 13, 14, 15, 16],# ring:   wrist→MCP→PIP→DIP→tip
        [0, 17, 18, 19, 20],# pinky:  wrist→MCP→PIP→DIP→tip
    ]
    angles = []
    for joints in finger_joints:
        for k in range(1, len(joints) - 1):   # k=1,2,3 → 3 angles per finger = 15 total
            a = lm[joints[k - 1]]
            b = lm[joints[k]]
            c = lm[joints[k + 1]]
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            angle = math.acos(float(np.clip(cos_angle, -1.0, 1.0)))
            angles.append(angle)

    return np.array(distances + angles, dtype=np.float32)


def preprocess(landmarks: list) -> np.ndarray:
    """
    Full preprocessing pipeline.
    Output shape: (88,) = 63 normalized landmarks + 10 fingertip distances + 15 bend angles
    """
    norm = normalize_landmarks(landmarks)
    geo  = compute_geometric_features(landmarks)
    return np.concatenate([norm, geo])


def get_confusion_labels(label: str) -> list:
    """Return visually similar labels for a predicted class."""
    return CONFUSION_PAIRS.get(label, [])


def majority_vote(buffer: list) -> str:
    """Return the most common prediction in a sliding buffer."""
    if not buffer:
        return "?"
    return max(set(buffer), key=buffer.count)
