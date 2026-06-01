# Task 3: Handwritten Character Recognition

**CodeAlpha ML Internship**

## Objective
Identify handwritten digits (0–9) and letters (A–Z) using Convolutional Neural Networks (CNNs).

## Datasets

### Part A — MNIST Digits
- **Source:** Built into `tf.keras.datasets.mnist`
- **Images:** 28×28 grayscale
- **Classes:** 10 (digits 0–9)
- **Size:** 60,000 train / 10,000 test

### Part B — EMNIST Letters
- **Source:** `tensorflow_datasets` (`emnist/letters`)
- **Images:** 28×28 grayscale
- **Classes:** 26 (letters A–Z)
- **Fallback:** Synthetic 28×28 patch images if `tensorflow_datasets` is not installed

## Model Architecture (shared for both datasets)

```
Input (28×28×1)
├── Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
├── Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
├── Conv2D(128) → BatchNorm → Dropout(0.25)
├── Flatten → Dense(256) → BatchNorm → Dropout(0.5)
└── Dense(n_classes, softmax)
```

- **Optimizer:** Adam (lr=1e-3)
- **Loss:** sparse_categorical_crossentropy
- **Callbacks:** EarlyStopping (patience=8), ReduceLROnPlateau (patience=4)

## Data Augmentation
Applied during MNIST and EMNIST training:
- Random rotation ±10°
- Width/height shift ±10%
- Zoom ±10%

## How to Run

### 1. Install dependencies
```bash
pip install tensorflow scikit-learn matplotlib seaborn
# For EMNIST (optional):
pip install tensorflow-datasets
```

### 2. Run the script
```bash
python task3_handwriting_recognition.py
```

## Outputs

| File | Description |
|---|---|
| `task3_mnist_training.png` | MNIST accuracy & loss curves |
| `task3_mnist_confusion.png` | MNIST confusion matrix |
| `task3_mnist_predictions.png` | Sample digit predictions grid |
| `task3_mnist_cnn.keras` | Saved MNIST model |
| `task3_emnist_training.png` | EMNIST accuracy & loss curves |
| `task3_emnist_confusion.png` | EMNIST confusion matrix |
| `task3_emnist_predictions.png` | Sample letter predictions grid |
| `task3_emnist_cnn.keras` | Saved EMNIST model |
| `task3_handwriting_recognition.csv` | Digit pixel dataset (CSV) |

## Project Structure
```
task3_handwriting_recognition.py
task3_handwriting_recognition.csv
task3_mnist_cnn.keras
task3_emnist_cnn.keras
task3_mnist_*.png
task3_emnist_*.png
README_task3_handwriting_recognition.md
```

## Expected Results
| Dataset | Expected Accuracy |
|---|---|
| MNIST (digits) | ~99% |
| EMNIST (letters) | ~88–92% |

## Extensibility
The pipeline can be extended to full word/sentence recognition using sequence models (CRNN — CNN + RNN/LSTM) for tasks like license plate or handwritten form recognition.

## Requirements
```
numpy
tensorflow>=2.10
scikit-learn
matplotlib
seaborn
tensorflow-datasets  (optional, for EMNIST)
```
