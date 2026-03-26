"""
fft_analysis.py
================
FFT (Fast Fourier Transform) Analysis for Deepfake Detection
Plug this into your existing image pipeline.

Usage:
    from fft_analysis import FFTAnalyzer
    analyzer = FFTAnalyzer()
    score = analyzer.predict("path/to/image.jpg")
    features = analyzer.extract_features("path/to/image.jpg")
"""

import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────
# 1. CORE FFT UTILITIES
# ─────────────────────────────────────────────

def compute_fft_spectrum(image_path: str, size: int = 224) -> np.ndarray:
    """
    Compute the 2D FFT magnitude spectrum of an image.
    Returns a 2D numpy array (log-scaled, normalized to [0,1]).

    Why log scale? Raw FFT magnitudes span many orders of magnitude.
    Log compression makes the spectrum visually readable and
    numerically stable for ML features.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0

    # 2D FFT → shift zero-frequency to center
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)

    # Log magnitude spectrum
    magnitude = np.abs(fft_shift)
    log_spectrum = np.log1p(magnitude)

    # Normalize to [0, 1]
    log_spectrum = (log_spectrum - log_spectrum.min()) / (log_spectrum.max() - log_spectrum.min() + 1e-8)

    return log_spectrum


def compute_radial_profile(spectrum: np.ndarray) -> np.ndarray:
    """
    Collapse 2D spectrum into a 1D radial average.
    This gives you the 'energy at each frequency ring'.

    - Real images: smooth exponential decay
    - Deepfakes: bumps/spikes at specific frequencies (GAN artifacts)
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

    max_r = min(cx, cy)
    radial_mean = np.zeros(max_r)
    for radius in range(max_r):
        mask = r == radius
        if mask.sum() > 0:
            radial_mean[radius] = spectrum[mask].mean()

    return radial_mean


def extract_frequency_features(image_path: str, size: int = 224) -> np.ndarray:
    """
    Extract a feature vector from the FFT spectrum.
    Returns a 1D numpy array of 512 features:
        - Flattened 16x16 coarse spectrum (256 values)
        - 128-bin radial profile (128 values)
        - 16 band energy ratios + 4 peak stats + other stats (128 values)

    These features can be fed into:
        - A standalone MLP classifier (see FFTClassifier below)
        - Concatenated with EfficientNet features for fusion
    """
    spectrum = compute_fft_spectrum(image_path, size)
    h, w = spectrum.shape

    # ── Feature 1: Coarse spatial spectrum (16×16 downsampled = 256 values) ──
    coarse = cv2.resize(spectrum, (16, 16), interpolation=cv2.INTER_AREA)
    feat1 = coarse.flatten()  # 256 values

    # ── Feature 2: Radial profile (first 128 rings) ──
    radial = compute_radial_profile(spectrum)
    # Pad or truncate to exactly 128
    if len(radial) >= 128:
        feat2 = radial[:128]
    else:
        feat2 = np.pad(radial, (0, 128 - len(radial)))

    # ── Feature 3: Statistical features ──
    # Band energy ratios (split spectrum into 16 frequency bands)
    bands = np.array_split(radial, 16)
    band_energies = np.array([b.mean() for b in bands])  # 16 values

    # High-frequency to low-frequency energy ratio
    low_energy = spectrum[:h//4, :w//4].mean()
    high_energy = spectrum[3*h//4:, 3*w//4:].mean()
    center_energy = spectrum[h//4:3*h//4, w//4:3*w//4].mean()
    hf_lf_ratio = high_energy / (low_energy + 1e-8)

    # Peak detection in spectrum (GANs often have spike peaks)
    flat = spectrum.flatten()
    top_peaks = np.sort(flat)[-10:]  # top 10 pixel values
    peak_stats = np.array([top_peaks.mean(), top_peaks.std(), top_peaks.max(), hf_lf_ratio])

    # Variance across quadrants
    q1 = spectrum[:h//2, :w//2].var()
    q2 = spectrum[:h//2, w//2:].var()
    q3 = spectrum[h//2:, :w//2].var()
    q4 = spectrum[h//2:, w//2:].var()
    quadrant_var = np.array([q1, q2, q3, q4])

    # Overall stats
    global_stats = np.array([
        spectrum.mean(), spectrum.std(), spectrum.var(),
        low_energy, high_energy, center_energy,
        np.percentile(spectrum, 25), np.percentile(spectrum, 75),
        np.percentile(spectrum, 90), np.percentile(spectrum, 99),
    ])

    # Zero-padding filler to reach exactly 128 stat features
    stat_features = np.concatenate([band_energies, peak_stats, quadrant_var, global_stats])
    if len(stat_features) < 128:
        stat_features = np.pad(stat_features, (0, 128 - len(stat_features)))
    else:
        stat_features = stat_features[:128]

    # ── Combine all ──
    features = np.concatenate([feat1, feat2, stat_features])  # 512 total
    return features.astype(np.float32)


# ─────────────────────────────────────────────
# 2. FFT CLASSIFIER (Standalone MLP)
# ─────────────────────────────────────────────

class FFTClassifier(nn.Module):
    """
    Lightweight MLP that takes 512 FFT features → REAL/FAKE prediction.
    Trains in minutes (not hours) — a great complement to EfficientNet.

    Architecture:
        512 → 256 → 128 → 64 → 1 (sigmoid)
        Each hidden layer: Linear → BatchNorm → ReLU → Dropout
    """
    def __init__(self, input_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 3. DATASET CLASS
# ─────────────────────────────────────────────

from torch.utils.data import Dataset

class FFTDataset(Dataset):
    """
    Dataset that loads images and returns FFT feature vectors + labels.

    Folder structure expected:
        data/images/real/   ← label 0
        data/images/fake/   ← label 1
    """
    def __init__(self, real_dir: str, fake_dir: str, size: int = 224):
        self.samples = []
        self.size = size

        for path in Path(real_dir).glob("*"):
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.samples.append((str(path), 0))  # 0 = REAL

        for path in Path(fake_dir).glob("*"):
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.samples.append((str(path), 1))  # 1 = FAKE

        print(f"FFTDataset: {sum(1 for _, l in self.samples if l==0)} real, "
              f"{sum(1 for _, l in self.samples if l==1)} fake")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            features = extract_frequency_features(path, self.size)
        except Exception:
            features = np.zeros(512, dtype=np.float32)

        return torch.tensor(features), torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────
# 4. TRAINING SCRIPT
# ─────────────────────────────────────────────

def train_fft_model(
    real_dir: str,
    fake_dir: str,
    save_path: str = "models/fft_classifier.pth",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3
):
    """
    Full training loop for FFT classifier.
    Expected training time: ~5-10 minutes on CPU, ~1-2 min on GPU.

    Args:
        real_dir: path to folder with real images
        fake_dir: path to folder with fake images
        save_path: where to save the trained model weights
        epochs: number of training epochs (30 is usually enough)
        batch_size: 64 works well for FFT features
        lr: learning rate
    """
    from torch.utils.data import DataLoader, random_split
    from sklearn.metrics import accuracy_score
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Dataset & Split
    dataset = FFTDataset(real_dir, fake_dir)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, Loss, Optimizer
    model = FFTClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    train_losses, val_accs = [], []

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(features).squeeze()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # ── Validate ──
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                preds = model(features).squeeze().cpu().numpy()
                all_preds.extend((preds > 0.5).astype(int))
                all_labels.extend(labels.numpy().astype(int))

        val_acc = accuracy_score(all_labels, all_preds)
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_accs.append(val_acc)
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best saved ({val_acc:.4f})")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, color='#3b9eff', linewidth=2)
    ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch"); ax1.grid(alpha=0.3)
    ax2.plot(val_accs, color='#00e5a0', linewidth=2)
    ax2.set_title("Validation Accuracy"); ax2.set_xlabel("Epoch"); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fft_training_curves.png", dpi=150, bbox_inches='tight')
    print("Training curves saved to fft_training_curves.png")
    plt.show()

    return model


# ─────────────────────────────────────────────
# 5. ANALYZER CLASS (for inference + Streamlit)
# ─────────────────────────────────────────────

class FFTAnalyzer:
    """
    High-level interface for inference. Use this in your Streamlit app.

    Example:
        analyzer = FFTAnalyzer(model_path="models/fft_classifier.pth")
        score = analyzer.predict("test_image.jpg")
        print(f"Fake probability: {score:.2%}")
    """
    def __init__(self, model_path: str = None, threshold: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model = FFTClassifier().to(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"FFT model loaded from {model_path}")
        else:
            print("Warning: No model path provided or file missing. Using untrained weights.")

        self.model.eval()  # ALWAYS call eval() for inference

    def extract_features(self, image_path: str) -> np.ndarray:
        """Returns raw 512-dim FFT feature vector."""
        return extract_frequency_features(image_path)

    def predict(self, image_path: str) -> dict:
        """
        Run full FFT analysis on one image.
        Returns a dict with:
            fake_prob   : float 0→1 (higher = more likely fake)
            verdict     : "FAKE" or "REAL"
            confidence  : float 0→1
            spectrum    : 2D numpy array (for visualization)
            radial      : 1D numpy array (for chart)
        """
        features = extract_frequency_features(image_path)
        tensor = torch.tensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            fake_prob = self.model(tensor).item()

        verdict = "FAKE" if fake_prob > self.threshold else "REAL"
        confidence = fake_prob if verdict == "FAKE" else (1 - fake_prob)

        spectrum = compute_fft_spectrum(image_path)
        radial   = compute_radial_profile(spectrum)

        return {
            "fake_prob":  fake_prob,
            "verdict":    verdict,
            "confidence": confidence,
            "spectrum":   spectrum,
            "radial":     radial,
        }

    def batch_predict(self, image_paths: list) -> list:
        """Predict on a list of images. Returns list of dicts."""
        return [self.predict(p) for p in image_paths]


# ─────────────────────────────────────────────
# 6. VISUALIZATION HELPERS
# ─────────────────────────────────────────────

def plot_fft_comparison(real_path: str, fake_path: str, save_path: str = None):
    """
    Side-by-side comparison: real vs fake FFT spectrum.
    Perfect for your README and project report.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor('#0a0a0f')

    for col, (path, label, color) in enumerate([
        (real_path, "REAL", "#00e5a0"),
        (fake_path, "FAKE", "#ff5b5b"),
    ]):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))

        spectrum = compute_fft_spectrum(path)
        radial = compute_radial_profile(spectrum)

        # Column 0+1: original image
        axes[0, col].imshow(img_resized)
        axes[0, col].set_title(f"{label} — Original", color=color, fontsize=13, fontweight='bold')
        axes[0, col].axis('off')

        # Row 1 col 0+1: spectrum
        axes[1, col].imshow(spectrum, cmap='magma')
        axes[1, col].set_title(f"{label} — FFT Spectrum", color=color, fontsize=13, fontweight='bold')
        axes[1, col].axis('off')

    # Column 2: radial profiles overlaid
    real_spectrum = compute_fft_spectrum(real_path)
    fake_spectrum = compute_fft_spectrum(fake_path)
    real_radial = compute_radial_profile(real_spectrum)
    fake_radial = compute_radial_profile(fake_spectrum)

    length = min(len(real_radial), len(fake_radial), 100)
    axes[0, 2].plot(real_radial[:length], color='#00e5a0', linewidth=2, label='Real')
    axes[0, 2].plot(fake_radial[:length], color='#ff5b5b', linewidth=2, label='Fake')
    axes[0, 2].set_title("Radial Energy Profile", color='white', fontsize=13, fontweight='bold')
    axes[0, 2].set_xlabel("Frequency Ring", color='#9090a8')
    axes[0, 2].set_ylabel("Mean Energy", color='#9090a8')
    axes[0, 2].legend(facecolor='#18181f', labelcolor='white')
    axes[0, 2].set_facecolor('#111118')
    axes[0, 2].tick_params(colors='#9090a8')
    axes[0, 2].grid(alpha=0.2)

    # Difference spectrum
    diff = np.abs(real_spectrum - fake_spectrum)
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title("Spectrum Difference", color='white', fontsize=13, fontweight='bold')
    axes[1, 2].axis('off')

    for ax in axes.flat:
        ax.set_facecolor('#0a0a0f') if not ax.images else None

    plt.tight_layout(pad=1.5)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
        print(f"Comparison plot saved to {save_path}")
    plt.show()


def plot_single_analysis(image_path: str, result: dict, save_path: str = None):
    """
    Single image analysis visualization — use this in your Streamlit app.
    Shows: original image, FFT spectrum, radial profile.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#0a0a0f')

    color = "#ff5b5b" if result["verdict"] == "FAKE" else "#00e5a0"

    # Original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(cv2.resize(img_rgb, (224, 224)))
    axes[0].set_title(f"Input Image", color='white', fontsize=12)
    axes[0].axis('off')

    # FFT spectrum
    axes[1].imshow(result["spectrum"], cmap='magma')
    axes[1].set_title(
        f"FFT Spectrum\n{result['verdict']} ({result['confidence']:.1%})",
        color=color, fontsize=12, fontweight='bold'
    )
    axes[1].axis('off')

    # Radial profile
    radial = result["radial"][:100]
    axes[2].fill_between(range(len(radial)), radial, alpha=0.3, color=color)
    axes[2].plot(radial, color=color, linewidth=2)
    axes[2].set_title("Radial Energy Profile", color='white', fontsize=12)
    axes[2].set_facecolor('#111118')
    axes[2].tick_params(colors='#9090a8')
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
    plt.show()


# ─────────────────────────────────────────────
# 7. FUSION HELPER — connect to EfficientNet score
# ─────────────────────────────────────────────

def fuse_scores(efficientnet_prob: float, fft_prob: float,
                weight_cnn: float = 0.65, weight_fft: float = 0.35) -> dict:
    """
    Combine EfficientNet (spatial) + FFT (frequency) scores.
    Default weights: 65% CNN, 35% FFT (CNN is stronger — this is tunable).

    Args:
        efficientnet_prob: fake probability from your EfficientNet model
        fft_prob: fake probability from FFT classifier
        weight_cnn: trust weight for EfficientNet (default 0.65)
        weight_fft: trust weight for FFT (default 0.35)

    Returns:
        dict with fused_prob, verdict, confidence, breakdown
    """
    assert abs(weight_cnn + weight_fft - 1.0) < 1e-6, "Weights must sum to 1.0"

    fused = weight_cnn * efficientnet_prob + weight_fft * fft_prob
    verdict = "FAKE" if fused > 0.5 else "REAL"
    confidence = fused if verdict == "FAKE" else (1 - fused)

    return {
        "fused_prob":   fused,
        "verdict":      verdict,
        "confidence":   confidence,
        "breakdown": {
            "efficientnet": efficientnet_prob,
            "fft":          fft_prob,
            "weights":      {"cnn": weight_cnn, "fft": weight_fft}
        }
    }


# ─────────────────────────────────────────────
# 8. QUICK TEST — run this file directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("FFT Analysis Module — Quick Test")
    print("=" * 55)

    if len(sys.argv) >= 2:
        test_img = sys.argv[1]
        print(f"\nAnalyzing: {test_img}")
        features = extract_frequency_features(test_img)
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"Mean: {features.mean():.4f}, Std: {features.std():.4f}")
        print("\nTo train the FFT classifier, call:")
        print('  train_fft_model("data/images/real", "data/images/fake")')
    else:
        print("\nUsage: python fft_analysis.py <path_to_image.jpg>")
        print("\nModule is ready to import. Example:")
        print("  from fft_analysis import FFTAnalyzer, train_fft_model")
        print("  analyzer = FFTAnalyzer('models/fft_classifier.pth')")
        print("  result = analyzer.predict('test.jpg')")
        print(f"  print(result['verdict'], result['confidence'])")