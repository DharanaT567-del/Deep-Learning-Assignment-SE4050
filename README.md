# Human Activity Recognition from Smartphone Inertial Signals

**SE4050 – Deep Learning | Group Assignment | Group SE4050_G31 | 2026**

This project compares four deep learning architectures for Human Activity Recognition (HAR) on the [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) dataset. Each model is trained from scratch on raw inertial sensor windows of shape `(128 timesteps × 9 channels)` and classifies them into six activities. All four models use the same data loader, the same subject-disjoint splits and the same evaluation protocol, so the comparison is fair.

---

## Team Members

| # | Name with Initials | Registration Number | Individual Component |
| :-: | :--- | :--- | :--- |
| 1 | Thilakarathne M.G.T.D | IT23324442 | Transformer encoder |
| 2 | Wickramasinghe Y.M.K | IT23411944 | CNN-LSTM and shared evaluation |
| 3 | Abeyrathne E.D.V.N | IT23265110 | BiLSTM and shared data loader |
| 4 | Perera V.D.R | IT23428942 | 1D-CNN and exploratory data analysis |

---

## Dataset

| Property | Value |
| :--- | :--- |
| Source | UCI HAR Dataset (Anguita et al., 2013) |
| Subjects | 30 volunteers (19–48 years), smartphone worn on the waist |
| Signals | 9 channels: body acceleration (x, y, z), body gyroscope (x, y, z), total acceleration (x, y, z) |
| Sampling | 50 Hz, 2.56 s windows (128 samples) with 50% overlap |
| Classes | `0` WALKING, `1` WALKING_UPSTAIRS, `2` WALKING_DOWNSTAIRS, `3` SITTING, `4` STANDING, `5` LAYING |

### Subject-disjoint splits

The dataset is split by subject, so no person appears in more than one split. This prevents the model from memorising individual gait patterns.

| Split | Windows | Subjects |
| :--- | ---: | :--- |
| Train | 5,952 | 17 subjects |
| Validation | 1,400 | 4 subjects (3, 16, 22, 23), held out from the original training set |
| Test | 2,947 | 9 subjects (2, 4, 9, 10, 12, 13, 18, 20, 24), the official UCI test set |

Standardisation statistics are fitted on the **training split only**. The test split is used **once**, for the final comparison. It is never used for early stopping or hyperparameter tuning.

---

## Models

| Model | Owner | Core idea | Trainable parameters |
| :--- | :--- | :--- | ---: |
| **Transformer encoder** | Thilakarathne M.G.T.D | Dense projection → trainable positional embedding → 2 pre-LN encoder blocks (4-head self-attention, GELU FFN) → global average pooling → dense head | 80,454 |
| **Hybrid CNN-LSTM** | Wickramasinghe Y.M.K | Two Conv1D layers (64 filters, k=3) extract local motion features → MaxPool halves the sequence (128 → 64) → LSTM (64 units) models longer-range context | 52,230 |
| **Bidirectional LSTM** | Abeyrathne E.D.V.N | Two stacked BiLSTM layers (64 units per direction) read the sequence both forwards and backwards → dense head | 145,350 |
| **1D-CNN** | Perera V.D.R | Two Conv1D + BatchNorm layers (64 filters, k=5) → MaxPool + dropout → Conv1D (128 filters) → global average pooling → dense head | 66,502 |

All models use the Adam optimiser and cross-entropy loss. Training runs for up to 60 epochs, with early stopping on validation loss (patience 10, best weights restored) and `ReduceLROnPlateau` (factor 0.5, patience 5).

---

## Results

All four models were evaluated on the same held-out test set: 2,947 windows from 9 subjects the models never saw. The figures below come from the latest run of [`notebooks/model_comparison.ipynb`](notebooks/model_comparison.ipynb) on Google Colab (TensorFlow 2.20, Keras 3.13, GPU).

| Model | Test Accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC | Parameters | Training Time (s) | Latency (ms / 100 samples) |
| :--- | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| **Bidirectional LSTM** | **89.38%** | **0.8932** | **0.8934** | 0.9862 | 145,350 | 38.55 | 28.91 |
| **Hybrid CNN-LSTM** | 88.80% | 0.8887 | 0.8883 | **0.9874** | **52,230** | 28.12 | **23.61** |
| **1D-CNN** | 86.22% | 0.8541 | 0.8596 | 0.9862 | 66,502 | **14.03** | 29.07 |
| **Transformer Encoder** | 85.10% | 0.8478 | 0.8514 | 0.9748 | 80,454 | 30.01 | 96.38 |

> Macro F1 is the primary metric. It gives every activity equal weight, so a model cannot score well by getting the easy dynamic classes right while failing on the harder static postures.

### Key findings

- **Most accurate model:** the BiLSTM achieved the highest accuracy and Macro F1 (89.38%, 0.8932).
- **Best accuracy for its size:** the CNN-LSTM came within 0.58 percentage points of the BiLSTM while using only **35.9%** of its parameters. It also had the lowest inference latency and the highest ROC-AUC.
- **Transformer:** with limited training data (about 6K windows), self-attention generalised less well to unseen subjects. It was also the slowest model at inference time.
- **Main source of error:** **SITTING vs STANDING** was the most confused pair for all four models. Both are static postures in which the accelerometer mostly measures the same 1 g gravity vector. Dynamic activities and LAYING were separated reliably.

### Deployment recommendation

| Target | Recommended model | Reason |
| :--- | :--- | :--- |
| Wearables and low-power edge devices | **Hybrid CNN-LSTM** | Smallest model and lowest latency, with accuracy close to the best |
| Server or cloud inference | **Bidirectional LSTM** | Highest accuracy when memory and latency matter less |

> Training on a GPU is not fully deterministic, so repeated runs can differ by about ±1–2% accuracy. Report the numbers from the executed notebook you submit.

---

## Repository Structure

```text
Deep-Learning-Assignment-SE4050/
├── configs/                      # JSON hyperparameter configs (one per model / experiment)
│   ├── transformer.json
│   ├── cnn_lstm.json
│   ├── bilstm.json
│   ├── cnn.json
│   ├── cnn_kernel3.json          # 1D-CNN ablation: kernel size 3
│   └── cnn_dropout05.json        # 1D-CNN ablation: dropout 0.5
├── docs/
│   ├── transformer.md            # Architecture details and tensor shapes per model
│   ├── cnn_lstm.md
│   ├── bilstm.md
│   ├── cnn.md
│   ├── model_comparison.md       # Shared evaluation methodology and discussion
│   └── report/                   # Report sections (preprocessing, models, comparison)
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_transformer.ipynb
│   ├── 03_error_analysis.ipynb
│   ├── 04_bilstm.ipynb
│   ├── 05_cnn_lstm.ipynb
│   └── model_comparison.ipynb    # Four-model benchmark on the held-out test set
├── src/
│   ├── data_contract.py          # Shared data contract, validation and leakage checks
│   ├── data/
│   │   └── uci_har_loader.py     # Download + preprocess UCI HAR into a single .npz
│   ├── models/
│   │   ├── transformer.py
│   │   ├── cnn_lstm.py
│   │   ├── bilstm.py
│   │   └── cnn.py
│   ├── train_transformer.py      # Training pipelines (one per model)
│   ├── train_cnn_lstm.py
│   ├── train_bilstm.py
│   └── train_cnn.py
├── tests/                        # Unit tests (models, data loader, data contract)
├── requirements.txt
└── README.md
```

`data/`, `outputs/`, `*.npz` and `*.keras` are git-ignored. You generate them locally.

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/DharanaT567-del/Deep-Learning-Assignment-SE4050.git
cd Deep-Learning-Assignment-SE4050
pip install -r requirements.txt
```

Requires TensorFlow ≥ 2.15 and Keras 3. Run every command from the repository root.

### 2. Build the shared dataset

The following command downloads UCI HAR, builds the subject-disjoint train/validation/test splits, applies train-only standardisation, validates the result and saves it to `data/uci_har_processed.npz`:

```bash
python -m src.data.uci_har_loader --download
```

Options: `--n-val-subjects` (default 4), `--seed` (default 42), `--no-standardize`, `--out`.

### 3. Run the tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 4. Train the models

Each trainer reads its config from `configs/` and writes `best_model.keras`, `history.json`, `history.csv`, `config.json` and `run_metadata.json` to `outputs/<model>/run_<timestamp>/`.

```bash
# Transformer encoder
python src/train_transformer.py --config configs/transformer.json --data-path data/uci_har_processed.npz --eval-test

# Hybrid CNN-LSTM
python src/train_cnn_lstm.py --config configs/cnn_lstm.json --data-path data/uci_har_processed.npz --evaluate-test

# Bidirectional LSTM
python src/train_bilstm.py --config configs/bilstm.json --data-path data/uci_har_processed.npz

# 1D-CNN (plus ablations: configs/cnn_kernel3.json, configs/cnn_dropout05.json)
python src/train_cnn.py --config configs/cnn.json --data-path data/uci_har_processed.npz
```

Every trainer also accepts `--epochs`, `--batch-size`, `--lr`, `--seed` and `--output-dir` to override values from the config.

**Quick smoke test (synthetic data, no download needed):**

```bash
python src/train_transformer.py --smoke-test
python src/train_bilstm.py --smoke-test
python src/train_cnn.py --smoke-test
python src/train_cnn_lstm.py --synthetic-smoke
```

### 5. Run the comparison

Open [`notebooks/model_comparison.ipynb`](notebooks/model_comparison.ipynb) locally or in Google Colab. The notebook loads the latest checkpoint for each model from `outputs/`, trains any model that has no checkpoint (seed 42) and evaluates all four once on the test set. It reports accuracy, F1, ROC-AUC, parameter count, training time and latency, and plots side-by-side confusion matrices.

---

## Engineering Practices

- **Shared data contract.** `src/data_contract.py` defines the dataset format that every model uses. Validation rejects datasets with wrong shapes, NaN or Inf values, out-of-range labels, splits missing a class, or **any subject that appears in more than one split**.
- **No test-set leakage.** Normalisation is fitted on the training split only. Early stopping and tuning use the validation split only.
- **Reproducibility.** All runs use a fixed seed (42), store their hyperparameters in JSON configs and save a full config and metadata snapshot with each run.
- **Serialisable models.** Custom layers are registered for Keras serialisation, and the tests check that a saved and reloaded model gives the same predictions.

---

## Documentation

| Document | Description |
| :--- | :--- |
| [docs/transformer.md](docs/transformer.md) | Transformer encoder: architecture, tensor shapes, integration |
| [docs/cnn_lstm.md](docs/cnn_lstm.md) | Hybrid CNN-LSTM: design and tuning |
| [docs/bilstm.md](docs/bilstm.md) | Bidirectional LSTM: design and experiments |
| [docs/cnn.md](docs/cnn.md) | 1D-CNN baseline and ablations |
| [docs/model_comparison.md](docs/model_comparison.md) | Shared evaluation protocol, metrics and error analysis |
| [docs/report/](docs/report/) | Sections of the final report |

---

## Reference

D. Anguita, A. Ghio, L. Oneto, X. Parra and J. L. Reyes-Ortiz, "A Public Domain Dataset for Human Activity Recognition Using Smartphones," *ESANN*, 2013.
