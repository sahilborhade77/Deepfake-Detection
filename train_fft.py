from src.models.fft_analysis import train_fft_model

train_fft_model(
    real_dir  = "data/images/real",
    fake_dir  = "data/images/fake",
    save_path = "models/fft_classifier.pth",
    epochs    = 30,
    batch_size= 64,
)