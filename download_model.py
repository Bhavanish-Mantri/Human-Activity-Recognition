"""
Model Download Script for Human Activity Recognition

Downloads the pre-trained ResNet-34 3D CNN model (ONNX format) trained
on the Kinetics-400 dataset. This model is used for human activity
recognition from video streams.

Model Details:
    - Architecture: ResNet-34 (3D Convolutional Neural Network)
    - Training Data: Kinetics-400 dataset
    - Format: ONNX (Open Neural Network Exchange)
    - Input: 16 frames of 112x112 pixels
    - Output: 400 activity class probabilities
    - Size: ~200 MB

Usage:
    python download_model.py
"""

import os
import sys
import urllib.request
import hashlib

# Model download URL (hosted via public repository)
MODEL_URLS = [
    "https://github.com/spmallick/learnopencv/raw/master/Human-Activity-Recognition-Using-OpenCV/resnet-34_kinetics.onnx",
    "https://www.dropbox.com/s/065l4vr8bptzohb/resnet-34_kinetics.onnx?dl=1",
]

MODEL_FILENAME = "resnet-34_kinetics.onnx"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


class DownloadProgressBar:
    """Progress bar for file downloads."""
    
    def __init__(self):
        self.last_percent = -1
    
    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            sys.stdout.write(f"\rDownloading... {block_num * block_size / 1024 / 1024:.1f} MB")
            sys.stdout.flush()
            return
            
        percent = min(100, int(block_num * block_size * 100 / total_size))
        if percent != self.last_percent:
            self.last_percent = percent
            downloaded = block_num * block_size / 1024 / 1024
            total = total_size / 1024 / 1024
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\r  [{bar}] {percent:3d}% ({downloaded:.1f}/{total:.1f} MB)")
            sys.stdout.flush()


def download_model():
    """Download the ResNet-34 Kinetics ONNX model."""
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    
    # Check if model already exists
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[INFO] Model already exists: {model_path}")
        print(f"[INFO] File size: {file_size:.1f} MB")
        
        response = input("[?] Do you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("[INFO] Using existing model file.")
            return model_path
    
    print("=" * 60)
    print("  DOWNLOADING PRE-TRAINED MODEL")
    print("  Model: ResNet-34 3D CNN (ONNX Format)")
    print("  Dataset: Kinetics-400")
    print("  Expected size: ~200 MB")
    print("=" * 60)
    
    for i, url in enumerate(MODEL_URLS):
        try:
            print(f"\n[INFO] Attempting download from source {i + 1}/{len(MODEL_URLS)}...")
            print(f"[INFO] URL: {url[:80]}...")
            
            progress = DownloadProgressBar()
            urllib.request.urlretrieve(url, model_path, reporthook=progress)
            print()  # New line after progress bar
            
            # Verify download
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            if file_size < 10:  # Model should be at least 10 MB
                print(f"[WARNING] Downloaded file is too small ({file_size:.1f} MB)")
                os.remove(model_path)
                continue
            
            print(f"\n[SUCCESS] Model downloaded successfully!")
            print(f"[INFO] File: {model_path}")
            print(f"[INFO] Size: {file_size:.1f} MB")
            return model_path
            
        except Exception as e:
            print(f"\n[ERROR] Download failed from source {i + 1}: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
            continue
    
    # If all URLs fail, provide manual instructions
    print("\n" + "=" * 60)
    print("  MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("""
All automatic download attempts failed. Please download manually:

Option 1 - GitHub (LearnOpenCV):
  1. Go to: https://github.com/spmallick/learnopencv
  2. Navigate to: Human-Activity-Recognition-Using-OpenCV/
  3. Download: resnet-34_kinetics.onnx

Option 2 - Google Drive:
  1. Search for "resnet-34_kinetics.onnx" on Google
  2. Download from a trusted source

Option 3 - ONNX Model Zoo:
  1. Go to: https://github.com/onnx/models
  2. Look for video classification models

After downloading, place the file as:
  {model_path}
""".format(model_path=model_path))
    
    return None


def verify_setup():
    """Verify that all required files are present."""
    print("\n[INFO] Verifying project setup...")
    
    required_files = {
        MODEL_FILENAME: "Pre-trained ResNet-34 ONNX model",
        "actions.txt": "Kinetics-400 action class labels",
        "har.py": "Main HAR application script",
        "requirements.txt": "Python dependencies"
    }
    
    all_good = True
    for filename, description in required_files.items():
        filepath = os.path.join(MODEL_DIR, filename)
        exists = os.path.exists(filepath)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {filename:<35s} - {description}")
        if not exists:
            all_good = False
    
    if all_good:
        print("\n[SUCCESS] All files are present! You're ready to go.")
        print("\nRun the application:")
        print("  Webcam:  python har.py --model resnet-34_kinetics.onnx --classes actions.txt")
        print("  Video:   python har.py --model resnet-34_kinetics.onnx --classes actions.txt --input video.mp4")
    else:
        print("\n[WARNING] Some files are missing. Please check the setup.")
    
    return all_good


if __name__ == "__main__":
    print("Human Activity Recognition - Model Setup\n")
    model_path = download_model()
    verify_setup()
