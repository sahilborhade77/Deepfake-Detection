import sys
sys.path.append(r"E:\Deepfake_Detection\deep")

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import seaborn as sns

from src.models.image_detector import build_model
from src.models.video_detector import build_video_model
from src.models.audio_detector import build_audio_model
from src.data.image_loader     import get_dataloader
from src.data.video_loader     import get_video_dataloader
from src.data.audio_loader     import get_audio_dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {
    "image": r"E:\Deepfake_Detection\deep\models\image_model.pth",
    "video": r"E:\Deepfake_Detection\deep\models\video_model.pth",
    "audio": r"E:\Deepfake_Detection\deep\models\audio_model.pth",
}


def evaluate(model, loader, model_name):
    model.eval()
    all_preds  = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(DEVICE)
            labels = labels.float()

            outputs = model(inputs)
            scores  = torch.sigmoid(outputs).cpu().squeeze().numpy()
            preds   = (scores > 0.5).astype(int)

            # Handle single sample batch
            if scores.ndim == 0:
                scores = np.array([scores])
                preds  = np.array([preds])

            all_scores.extend(scores.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_scores = np.array(all_scores)

    # ── Print report ──
    print(f"\n{'='*50}")
    print(f"  {model_name.upper()} MODEL — TEST RESULTS")
    print(f"{'='*50}")
    print(classification_report(
        all_labels, all_preds,
        target_names=["REAL", "FAKE"]
    ))

    # AUC score
    try:
        auc = roc_auc_score(all_labels, all_scores)
        print(f"  AUC Score: {auc:.4f}")
    except:
        pass

    # ── Confusion matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        cmap="Blues"
    )
    plt.title(f"{model_name} — Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(
        f"E:\\Deepfake_Detection\\deep\\models\\"
        f"{model_name}_confusion_matrix.png"
    )
    plt.show()
    print(f"  📊 Confusion matrix saved!")

    # ── ROC curve ──
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='#00e5a0',
                 label=f'AUC = {auc:.3f}')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} — ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"E:\\Deepfake_Detection\\deep\\models\\"
            f"{model_name}_roc_curve.png"
        )
        plt.show()
        print(f"  📊 ROC curve saved!")
    except:
        pass

    return all_preds, all_labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["image", "video", "audio", "all"],
        default="all"
    )
    args = parser.parse_args()

    # ── Image ──
    if args.model in ["image", "all"]:
        try:
            model = build_model(pretrained=False).to(DEVICE)
            model.load_state_dict(torch.load(
                MODELS["image"], map_location=DEVICE
            ))
            loader = get_dataloader("Test", batch_size=32)
            evaluate(model, loader, "image")
        except Exception as e:
            print(f"⚠️  Image model: {e}")

    # ── Video ──
    if args.model in ["video", "all"]:
        try:
            model = build_video_model(pretrained=False).to(DEVICE)
            model.load_state_dict(torch.load(
                MODELS["video"], map_location=DEVICE
            ))
            loader = get_video_dataloader("Test", batch_size=4)
            evaluate(model, loader, "video")
        except Exception as e:
            print(f"⚠️  Video model: {e}")

    # ── Audio ──
    if args.model in ["audio", "all"]:
        try:
            model = build_audio_model().to(DEVICE)
            model.load_state_dict(torch.load(
                MODELS["audio"], map_location=DEVICE
            ))
            loader = get_audio_dataloader("Test", batch_size=8)
            evaluate(model, loader, "audio")
        except Exception as e:
            print(f"⚠️  Audio model: {e}")