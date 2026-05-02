# Human Activity Recognition with OpenCV & Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green?logo=opencv)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Overview

A **real-time Human Activity Recognition (HAR)** system that uses a pre-trained **ResNet-34 3D Convolutional Neural Network** to classify human activities from video streams. The model is trained on the **Kinetics-400 dataset** and can recognize **400 different human activities** including sports, daily activities, gestures, and more.

### Key Features
- 🎥 **Real-time recognition** from webcam or video files
- 🧠 **400 activity classes** from the Kinetics-400 dataset
- 📊 **Confidence scores** with softmax probability distribution
- 🖥️ **Styled HUD overlay** with prediction visualization
- 📹 **Video output support** for saving processed results
- ⚡ **GPU/CUDA acceleration** support for faster inference
- 📋 **Batch processing** mode for analyzing multiple videos
- 📈 **Performance metrics** with FPS and inference time tracking

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Human Activity Recognition                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Video Input ──→ Frame Extraction ──→ Temporal Window (16)   │
│       │                                      │               │
│       │              ┌───────────────────────┘               │
│       │              ▼                                       │
│       │         Blob Creation                                │
│       │         (Resize 112x112,                             │
│       │          Mean Subtraction,                           │
│       │          BGR→RGB, Crop)                              │
│       │              │                                       │
│       │              ▼                                       │
│       │      ┌───────────────┐                               │
│       │      │  ResNet-34 3D │                               │
│       │      │     CNN       │                               │
│       │      │   (ONNX)      │                               │
│       │      └───────┬───────┘                               │
│       │              │                                       │
│       │              ▼                                       │
│       │      Softmax Output                                  │
│       │      (400 classes)                                   │
│       │              │                                       │
│       │              ▼                                       │
│       └───── Activity Label + Confidence ──→ Display/Save    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Model Details

| Component | Details |
|-----------|---------|
| **Architecture** | ResNet-34 (3D CNN) |
| **Framework** | ONNX (Open Neural Network Exchange) |
| **Training Data** | Kinetics-400 Dataset |
| **Input Shape** | `[1, 3, 16, 112, 112]` (batch, channels, depth, height, width) |
| **Output** | 400 class probabilities |
| **Temporal Window** | 16 consecutive frames |
| **Spatial Resolution** | 112 × 112 pixels |

---

## 📁 Project Structure

```
ML-Project/
├── har.py                  # Main HAR application (real-time)
├── batch_analyze.py        # Batch video analysis script
├── download_model.py       # Model download helper script
├── actions.txt             # 400 Kinetics action class labels
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
├── resnet-34_kinetics.onnx # Pre-trained model (download required)
├── videos/                 # Input video files
├── output/                 # Processed output videos
└── results/                # Batch analysis reports (JSON/CSV)
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time mode)
- GPU with CUDA (optional, for faster inference)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Human-Activity-Recognition.git
cd Human-Activity-Recognition
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Model
```bash
python download_model.py
```

This will download the ResNet-34 ONNX model (~200 MB) automatically.

---

## 💻 Usage

### Real-time Webcam Mode
```bash
python har.py --model resnet-34_kinetics.onnx --classes actions.txt
```

### Video File Mode
```bash
python har.py --model resnet-34_kinetics.onnx --classes actions.txt \
    --input videos/example.mp4 --output output/result.mp4
```

### With GPU Acceleration
```bash
python har.py --model resnet-34_kinetics.onnx --classes actions.txt --gpu 1
```

### Batch Video Analysis
```bash
python batch_analyze.py --model resnet-34_kinetics.onnx --classes actions.txt \
    --input_dir videos/ --output_dir results/
```

### Command-line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model` | `-m` | Required | Path to ONNX model file |
| `--classes` | `-c` | Required | Path to class labels file |
| `--input` | `-i` | Webcam | Path to input video file |
| `--output` | `-o` | None | Path to output video file |
| `--display` | `-d` | 1 | Show output window (1=yes) |
| `--gpu` | `-g` | 0 | Use GPU/CUDA (1=yes) |

---

## 🧠 How It Works

### 1. Frame Collection
The system captures **16 consecutive frames** from the video stream, forming a temporal window that captures motion patterns.

### 2. Preprocessing (Blob Creation)
- Frames are resized to **112 × 112** pixels
- **Mean subtraction** is applied using ImageNet normalization values `(114.7748, 107.7354, 99.4750)`
- Color channels are swapped from **BGR to RGB**
- Frames are combined into a **5D tensor**: `[batch, channels, depth, height, width]`

### 3. Inference
The preprocessed blob is passed through the **ResNet-34 3D CNN**, which outputs raw scores for 400 activity classes.

### 4. Post-processing
- **Softmax** is applied to convert raw scores to probability distribution
- The class with the **highest probability** is selected as the predicted activity
- **Confidence score** indicates the model's certainty

### 5. Visualization
Results are rendered on the video frames with:
- **Activity label** with styled overlay
- **Confidence bar** with color-coded fill
- **FPS counter** for performance monitoring
- **Info bar** with system details

---

## 📊 Sample Activities (400 Classes)

The model can recognize activities across multiple categories:

| Category | Examples |
|----------|----------|
| **Sports** | playing basketball, swimming, skateboarding, archery |
| **Daily Activities** | reading book, writing, cooking, cleaning |
| **Music** | playing guitar, playing piano, playing drums |
| **Exercise** | yoga, push ups, squat, deadlifting |
| **Dance** | ballet, salsa, breakdancing, zumba |
| **Food** | eating burger, making pizza, cooking egg |
| **Outdoor** | rock climbing, surfing, skiing, paragliding |

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Inference Time (CPU)** | ~200-400ms per batch |
| **Inference Time (GPU)** | ~50-100ms per batch |
| **Throughput (CPU)** | ~40-80 frames/s |
| **Number of Classes** | 400 |
| **Top-1 Accuracy** | ~63% (Kinetics-400) |
| **Top-5 Accuracy** | ~83% (Kinetics-400) |

---

## 🔧 Technical Stack

- **Python 3.8+** — Core programming language
- **OpenCV (cv2)** — Video capture, frame processing, DNN inference
- **NumPy** — Numerical computations, array operations
- **imutils** — Frame resizing and video stream utilities
- **ONNX** — Model format for cross-platform deployment
- **ResNet-34 3D CNN** — Deep learning architecture for video understanding

---

## 📋 Future Improvements

- [ ] Add multi-person activity recognition
- [ ] Implement activity tracking over time
- [ ] Add web-based dashboard for monitoring
- [ ] Support for custom model training/fine-tuning
- [ ] Edge deployment optimization (TensorRT/OpenVINO)
- [ ] REST API for cloud-based inference

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Kinetics-400 Dataset](https://deepmind.com/research/open-source/kinetics) — DeepMind
- [OpenCV DNN Module](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
- [ONNX Model Format](https://onnx.ai/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385) — He et al.

---

**Built with ❤️ for AI/ML Portfolio**
