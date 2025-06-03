## Real-Time ASL Inference Using MediaPipe & TFLite

This module enables real-time ASL sign recognition using [MediaPipe Holistic](https://developers.google.com/mediapipe/solutions/vision/holistic) for landmark extraction and a TensorFlow Lite model for efficient sequence-level inference.

### 🧠 Overview

- Extracts pose, face, and hand landmarks from video or webcam.
- Aggregates frames into fixed-length sequences (e.g., 70 frames).
- Periodically runs predictions using `model.tflite`.
- Displays results live on video using OpenCV.

### 🎥 How to Run

```
python run_realtime.py
```

- Input: `query_combined.mp4` (can be replaced with webcam).
- Output: Live video window with predicted ASL text overlay.

### 📁 Files

| File                                  | Purpose                                      |
|---------------------------------------|----------------------------------------------|
| `./test/t4.py`                     | Main script for live landmark capture        |
| `output_landmarks.parquet`            | Saved landmark sequences                     |
| `test/model.tflite`                   | Exported TensorFlow Lite model               |
| `test/inference_args.json`            | Required input features for the model        |
| `test/character_to_prediction_index.json` | Mapping from model outputs to characters |

### 🔍 How It Works

1. Captures video frames and extracts landmarks using MediaPipe.
2. Stores landmarks per frame in a structured format.
3. Every N frames, saves a sequence to `.parquet` and runs inference.
4. Displays predictions on video using OpenCV overlay.

### 📦 Requirements

- `mediapipe`
- `opencv-python`
- `tensorflow`
- `pandas`
- `numpy`

### 🧪 Notes

- Use `cv2.VideoCapture(0)` to switch from video file to webcam.
- Assumes ~15 FPS; configurable via `num_frames` parameter.
- Built for efficient real-time inference and easy integration.

Part of the [`Real-Time-ASL`](https://github.com/mahmoudmokhiamar/Real-Time-ASL) project by Mahmoud Mokhiamar.

---

## ASL Fingerspelling Model Training and Conversion

This section explains how to reproduce the ASL fingerspelling model used in this project. The training pipeline is based on a PyTorch → TFLite workflow for efficient deployment.

### 🧾 Setup & Install

Clone the repository and install dependencies:

```
git clone https://github.com/mahmoudmokhiamar/Real-Time-ASL.git
cd Real-Time-ASL
pip install -r requirements.txt
```

### 📥 Download Preprocessed Data

```
cd datamount/
kaggle datasets download -d darraghdog/asl-fingerspelling-preprocessing-train-dataset
unzip -n asl-fingerspelling-preprocessing-train-dataset.zip
rm asl-fingerspelling-preprocessing-train-dataset.zip

kaggle datasets download -d darraghdog/asl-fingerspelling-preprocessed-supp-dataset
unzip -n asl-fingerspelling-preprocessed-supp-dataset.zip 
mv supplemental_landmarks/* train_landmarks_npy/
rm asl-fingerspelling-preprocessed-supp-dataset.zip
rm -rf supplemental_landmarks/
cd ..
```

---

## 🧪 Reproducing the Model

Training consists of two rounds. Round 1 generates OOF predictions, which are used in Round 2 as auxiliary targets.

### 1️⃣ Round 1 Training

Train all 4 folds using `cfg_1`:

```
python train.py -C cfg_1
python train.py -C cfg_1 --fold 1
python train.py -C cfg_1 --fold 2
python train.py -C cfg_1 --fold 3
```

Merge OOF predictions:

```
python scripts/get_train_folded_oof_supp.py
```

### 2️⃣ Round 2 Training

Train the final models using `cfg_2`:

```
python train.py -C cfg_2 --fold -1
python train.py -C cfg_2 --fold -1
```

### 3️⃣ Convert to TensorFlow Lite

Convert the PyTorch model to TensorFlow Lite for deployment:

```
python scripts/convert_cfg_2_to_tf_lite.py
```

### 📦 Output Files

- `datamount/weights/cfg_2/fold-1/model.tflite`
- `datamount/weights/cfg_2/fold-1/inference_args.json`

---

## 🧠 Model Architecture

- **Encoder**: Adapted Squeezeformer optimized for landmark sequences.
- **Decoder**: 2-layer Transformer.
- **Augmentations**: CutMix, FingerDropout, TimeStretch, DecoderInput Masking.
- **Confidence Head**: Identifies unreliable predictions.

The model is trained in PyTorch and converted to TensorFlow Lite for efficient real-time inference.

---

## 📚 References

- Squeezeformer (PyTorch): https://github.com/upskyy/Squeezeformer
- Squeezeformer Paper: https://arxiv.org/pdf/2206.00888.pdf
- Decoder adapted from: https://github.com/huggingface/transformers/

---

### 🙏 Acknowledgments

This work draws inspiration from the [1st place solution](https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434485) to the [Google - ASL Fingerspelling Recognition](https://www.kaggle.com/competitions/asl-fingerspelling) Kaggle competition.