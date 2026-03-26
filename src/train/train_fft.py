"""
train_fft.py
============
Train the FFT classifier for deepfake detection.

Usage:
    cd E:\Deepfake_Detection\deep
    python src/train/train_fft.py
"""

import os
import sys
import torch
sys.path.append(r"E:\Deepfake_Detection\deep")

from src.models.fft_analysis import train_fft_model

# ── Config ────────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "❌ GPU (CUDA) required but not available!"
DEVICE      = torch.device("cuda")

REAL_DIR    = r"E:\Deepfake_Detection\deep\data\images\Train\Real"
FAKE_DIR    = r"E:\Deepfake_Detection\deep\data\images\Train\Fake"
SAVE_PATH   = r"E:\Deepfake_Detection\deep\models\fft_classifier.pth"

EPOCHS      = 30
BATCH_SIZE  = 64
LR          = 1e-3
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Training FFT Classifier...")
    print(f"Real images: {REAL_DIR}")
    print(f"Fake images: {FAKE_DIR}")
    print(f"Save path: {SAVE_PATH}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LR}")
    print("-" * 50)

    # Check directories exist
    if not os.path.exists(REAL_DIR):
        print(f"❌ Real directory not found: {REAL_DIR}")
        sys.exit(1)
    if not os.path.exists(FAKE_DIR):
        print(f"❌ Fake directory not found: {FAKE_DIR}")
        sys.exit(1)

    # Train
    model = train_fft_model(
        real_dir=REAL_DIR,
        fake_dir=FAKE_DIR,
        save_path=SAVE_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR
    )

    print("✅ FFT training complete!")