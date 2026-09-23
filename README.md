# Deep-Learning-Assignment-SE4050: Human Activity Recognition (HAR)

A collaborative deep learning project comparing four distinct neural architectures on the original [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) dataset.

## Architecture Comparison Overview

| Member | Architecture / Component | Implementation Scope |
| :--- | :--- | :--- |
| **Dharana** | **Transformer Encoder** (Branch: `dharana/componenet1`) | Multi-head self-attention with trainable positional embeddings in TensorFlow/Keras. |
| **Member 2** | **1D-CNN & EDA** | 1D Convolutional baseline and exploratory signal analysis. |
| **Member 3** | **BiLSTM & Shared Data Loader** | Bidirectional LSTM model and centralized dataset preprocessor. |
| **Member 4** | **CNN-LSTM & Shared Evaluation** | Hybrid CNN-LSTM and comparative evaluation pipeline. |

---

## Component 1: Transformer Encoder (Dharana)

Dharana's component implements a pure Transformer encoder trained from scratch on raw 9-channel sensor windows of length 128 (shape: `(N, 128, 9)`).

### Key Features
* **Trainable Positional Embeddings:** Learnable temporal position vectors registered for Keras serialization.
* **Pre-LayerNormalization:** 2-block Transformer encoder with multi-head attention (4 heads, `key_dim=16`) and GELU feed-forward sublayers.
* **Strict Leakage Prevention:** Built-in validation guaranteeing zero subject overlap between training and validation splits.
* **Self-Contained Run Logging:** Automatically saves `.keras` model checkpoints, `history.json`, `history.csv`, `config.json`, and `run_metadata.json`.

---

## Installation & Setup

### 1. Clone & Switch to Component Branch
```bash
git clone https://github.com/DharanaT567-del/Deep-Learning-Assignment-SE4050.git
cd Deep-Learning-Assignment-SE4050
git checkout dharana/componenet1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Start & Execution

### 1. Run Verification & Unit Tests
Run the test suite to verify model shapes, gradients, serialization, and leakage checks:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 2. Run Synthetic Smoke Training Test
Test the end-to-end training and artifact generation pipeline (2 epochs on synthetic data):
```bash
python src/train_transformer.py --smoke-test
```

### 3. Train on Processed HAR Dataset
Once Member 3 provides the preprocessed NPZ archive:
```bash
python src/train_transformer.py --config configs/transformer.json --data-path data/uci_har_processed.npz
```

### 4. Interactive Jupyter Notebook
Launch the demonstration notebook importing the modular components:
```bash
jupyter notebook notebooks/02_transformer.ipynb
```

---

## Project Structure

```text
├── configs/
│   └── transformer.json            # Baseline model & training hyperparameters
├── docs/
│   └── transformer.md              # Detailed architecture, tensor shapes & integration guide
├── notebooks/
│   └── 02_transformer.ipynb        # Interactive demonstration notebook
├── src/
│   ├── data_contract.py            # Shared data contract, leakage checks & NPZ loader
│   ├── train_transformer.py        # Complete training pipeline & run logger
│   └── models/
│       ├── __init__.py
│       └── transformer.py          # Custom layers, model builder & serialization
├── tests/
│   └── test_transformer.py         # Unit tests, gradient checks & contract validation
├── .gitignore                      # Excludes large data, weights & artifacts
├── README.md                       # Project and component documentation
└── requirements.txt                # Python dependencies
```

---

## Documentation

For full mathematical descriptions, tensor flow diagrams, and integration specifications for Member 3 and Member 4, refer to [`docs/transformer.md`](docs/transformer.md).
