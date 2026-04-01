"""
download_model.py - Downloads the MediaPipe hand_landmarker.task model file.
Run this ONCE before training or running the backend.

Usage:
  cd backend
  python download_model.py
"""

import os
import sys
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def download():
    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        if size > 1_000_000:
            print(f"[OK] hand_landmarker.task already exists ({size/1e6:.1f} MB)")
            return
        else:
            print(f"[WARN] Existing file seems too small ({size} bytes), re-downloading...")

    print(f"Downloading hand_landmarker.task from:")
    print(f"  {MODEL_URL}")
    print("  This may take a moment (~30 MB)...\n")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        sys.stdout.write(f"\r  [{bar}] {pct:.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, progress)
    print(f"\n\n✓ Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    download()
