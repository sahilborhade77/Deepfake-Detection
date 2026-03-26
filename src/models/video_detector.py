import torch
import torch.nn as nn
import timm

class VideoDetector(nn.Module):
    def __init__(self, pretrained=True, hidden_size=256, num_layers=2):
        super(VideoDetector, self).__init__()

        # Same EfficientNet-B4 as image model — extracts features per frame
        self.cnn = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0  # remove classifier
        )
        num_features = self.cnn.num_features  # 1792

        # LSTM — sees the sequence of frame features
        # and catches temporal inconsistencies
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: [batch, frames, channels, height, width]
        batch_size, frames, C, H, W = x.shape

        # Run CNN on each frame
        # Reshape to [batch*frames, C, H, W]
        x = x.view(batch_size * frames, C, H, W)
        features = self.cnn(x)  # [batch*frames, 1792]

        # Reshape back to [batch, frames, 1792]
        features = features.view(batch_size, frames, -1)

        # Run LSTM on sequence
        lstm_out, _ = self.lstm(features)

        # Take only the last frame output
        last_output = lstm_out[:, -1, :]

        # Classify
        output = self.classifier(last_output)
        return output


def build_video_model(pretrained=True):
    model = VideoDetector(pretrained=pretrained)
    return model


# ── Quick test ───────────────────────────────────
if __name__ == "__main__":
    print("Building video model...")
    model = build_video_model(pretrained=False)

    # Simulate batch of 2 videos, 10 frames each
    dummy = torch.randn(2, 10, 3, 224, 224)
    output = model(dummy)

    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Output values: {output.detach()}")
    print("✅ Video model built successfully!")