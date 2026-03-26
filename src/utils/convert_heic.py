"""
convert_heic.py — Converts all .HEIC files in the dataset to .jpg
Run this ONCE before fine-tuning.

Usage:
    cd E:\Deepfake_Detection\deep
    python src/utils/convert_heic.py
"""
import os
import sys

def convert_heic_files(base_dir: str):
    """Find and convert all .HEIC files under base_dir to .jpg."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        from PIL import Image
    except ImportError:
        print("❌  pillow-heif not installed. Run:")
        print("    pip install pillow-heif")
        sys.exit(1)

    heic_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".heic"):
                heic_files.append(os.path.join(root, f))

    if not heic_files:
        print("✅  No HEIC files found — nothing to convert.")
        return

    print(f"Found {len(heic_files)} HEIC file(s). Converting...\n")
    converted, failed = 0, 0

    for heic_path in heic_files:
        jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
        try:
            img = Image.open(heic_path).convert("RGB")
            img.save(jpg_path, "JPEG", quality=95)
            os.remove(heic_path)
            print(f"  ✔  {os.path.basename(heic_path)} → {os.path.basename(jpg_path)}")
            converted += 1
        except Exception as e:
            print(f"  ✘  Failed: {heic_path} — {e}")
            failed += 1

    print(f"\n✅  Done! Converted: {converted}  |  Failed: {failed}")


if __name__ == "__main__":
    BASE = r"E:\Deepfake_Detection\deep\data\images"
    convert_heic_files(BASE)
