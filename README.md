# COMSYS-5 Project

Real-time gender classification and dataset analysis system.

# Evaluation Reports

## Task A: Gender Classification
```
Classification Report:
              precision    recall  f1-score   support

      female       0.92      0.84      0.87        79
        male       0.85      0.92      0.88        79

    accuracy                           0.88       158
   macro avg       0.88      0.88      0.88       158
weighted avg       0.88      0.88      0.88       158
```

## Task B: Face Recognition
```
📊 Evaluation Metrics:
Accuracy : 0.6383
Precision: 0.6383
Recall   : 1.0000
F1 Score : 0.7792
```

## Quick Start

- **Clone with git**
  ```bash
  git clone https://github.com/nishanroy561/COMSYS-5_Project
  cd COMSYS-5_Project
  ```
- **Or Download ZIP**
  1. Download and extract ZIP from GitHub
  2. Open a terminal in the extracted folder

- **Create Virtual Environment**
  ```bash
  python -m venv venv
  ```
- **Activate Virtual Environment**
  ```bash
  # Windows:
  .\venv\Scripts\activate
  # Linux/Mac:
  source venv/bin/activate
  ```
- **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```

---

# TASK A: Gender Classification

## Setup & Running
1. Change to the TASK_A directory:
   ```bash
   cd TASK_A
   ```
2. Static image gender classification:
   ```bash
   python scripts/photos.py
   ```
3. Real-time webcam gender classification:
   ```bash
   python scripts/real_time.py
   ```

## Evaluation
Run the evaluation script:
```bash
python scripts/eval.py
```

---

# TASK B: Face Recognition

## Setup & Running
1. Change to the TASK_B directory:
   ```bash
   cd TASK_B
   ```
2. Run dataset analysis:
   ```bash
   python test.py
   ```

## Evaluation
Run the evaluation script:
```bash
python eval.py
```

---

## Project Structure
- **TASK_A**: Gender classification using ResNet-18
- **TASK_B**: Face recognition dataset analysis
- **Shared dependencies**: PyTorch, OpenCV, scikit-learn

## Requirements
- Python 3.8+
- Webcam (for real-time detection)
- 100MB+ disk space

## Architecture Overview

**TASK_A: Gender Classification**
- Uses a pre-trained ResNet-18 convolutional neural network as the backbone.
- Fine-tuned for binary classification (male/female).
- Input images are resized to 224x224 and normalized using ImageNet statistics.
- Faces are detected using OpenCV's Haar Cascade classifier before classification.
- A bias correction threshold is applied to improve prediction reliability.

**TASK_B: Dataset Analysis**
- Loads and analyzes PyTorch embedding files for the training set.
- Scans directory structures for both training and validation datasets.
- Computes statistics: number of people, images, embeddings, and overlap between train/val sets.
- Reports results and highlights any missing or mismatched data.
