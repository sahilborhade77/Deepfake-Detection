import torch
import torch.nn as nn

class AudioDetector(nn.Module):
    def __init__(self, n_mfcc=40):
        super(AudioDetector, self).__init__()

        # CNN layers — treat MFCC like a 2D image
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.3),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x shape: [batch, 1, n_mfcc, time]
        features = self.cnn(x)
        output   = self.classifier(features)
        return output


def build_audio_model():
    model = AudioDetector(n_mfcc=40)
    return model


# ── Quick test ───────────────────────────────────
if __name__ == "__main__":
    print("Building audio model...")
    model = build_audio_model()

    # Simulate batch of 4 audio clips
    # Shape: [batch, 1 channel, 40 mfcc, 100 time steps]
    dummy = torch.randn(4, 1, 40, 100)
    output = model(dummy)

    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Output values: {output.detach()}")
    print("✅ Audio model built successfully!")