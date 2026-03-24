import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class FrequencyDetector(nn.Module):
    def __init__(self, input_size=128*128):
        super(FrequencyDetector, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.classifier(x)


def extract_fft_features(img, size=128):
    """
    Convert image to frequency domain and extract features.
    Deepfakes leave unnatural frequency patterns (GAN artifacts).
    """
    # Convert to grayscale
    if isinstance(img, torch.Tensor):
        # Convert tensor to PIL
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img).convert('L')
    else:
        img = img.convert('L')

    # Resize to fixed size
    img = img.resize((size, size))
    arr = np.array(img, dtype=np.float32)

    # Apply FFT
    fft       = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # Log scale — makes GAN artifacts visible
    log_mag = np.log1p(magnitude)

    # Normalize
    log_mag = log_mag / (log_mag.max() + 1e-8)

    # Flatten to 1D feature vector
    features = log_mag.flatten().astype(np.float32)

    return torch.tensor(features)


def build_frequency_model():
    model = FrequencyDetector(input_size=128*128)
    return model


# ── Quick test ───────────────────────────────────
if __name__ == "__main__":
    print("Testing frequency detector...")

    # Test feature extraction
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    features = extract_fft_features(dummy_img)
    print(f"FFT features shape: {features.shape}")

    # Test model
    model  = build_frequency_model()
    output = model(features.unsqueeze(0))
    print(f"Model output shape: {output.shape}")
    print(f"Model output value: {output.item():.4f}")
    print("✅ Frequency detector working!")