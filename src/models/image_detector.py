import torch
import torch.nn as nn
import timm

class ImageDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageDetector, self).__init__()

        # Load EfficientNet-B4 pretrained on ImageNet
        self.model = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0  # remove original classifier
        )

        # Get the number of features EfficientNet outputs
        num_features = self.model.num_features  # 1792 for B4

        # Our custom classifier — Real vs Fake
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)  # 1 output → REAL or FAKE
        )

    def forward(self, x):
        features = self.model(x)
        output = self.classifier(features)
        return output


def build_model(pretrained=True):
    model = ImageDetector(pretrained=pretrained)
    return model


# ── Quick test ───────────────────────────────────
if __name__ == "__main__":
    print("Building model...")
    model = build_model(pretrained=False)

    # Simulate a batch of 4 images (3 channels, 224x224)
    dummy = torch.randn(4, 3, 224, 224)
    output = model(dummy)

    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Output values: {output.detach()}")
    print("✅ Model built successfully!")