# Component 4: Hybrid CNN-LSTM for Human Activity Recognition

**Author:** Monal (Member 4)
**Architecture:** 1D-CNN Feature Extractor → LSTM Sequence Modeller → Dense Classifier
**Framework:** TensorFlow / Keras (trained from scratch)

---

## 1. Architectural Motivation

Human activities captured by wrist/pocket-mounted accelerometers and gyroscopes exhibit **two distinct scales of temporal structure**:

| Scale | Example | Best captured by |
|---|---|---|
| **Local** (10–30 timesteps) | A single foot-strike during walking, a sudden arm lift | 1D Convolutions with small kernels |
| **Global** (full 128-step window) | The transition from walking to standing, sustained stillness vs. periodic gait | Recurrent (LSTM) memory |

A pure CNN (Member 2) captures local patterns efficiently but has no explicit memory.
A pure BiLSTM (Member 3) models full-window dynamics but treats every timestep equally.
The **hybrid CNN-LSTM** combines both: the CNN extracts a compact, discriminative local feature map, and the LSTM reads that map as a sequence to model macro-level activity dynamics.

---

## 2. Architecture Diagram

```
Input: (batch, 128, 9)
        │
        ▼
┌─────────────────────────┐
│  Conv1D(64, k=3, same)  │  ← Local feature extraction
│  BatchNormalization      │
│  ReLU + Dropout(0.3)    │
└────────────┬────────────┘
             │
┌─────────────────────────┐
│  Conv1D(64, k=3, same)  │  ← Deeper local patterns
│  BatchNormalization      │
└────────────┬────────────┘
             │
┌─────────────────────────┐
│  MaxPooling1D(pool=2)   │  ← Temporal downsampling: (64, 64)
│  Dropout(0.3)           │
└────────────┬────────────┘
             │
┌─────────────────────────┐
│  LSTM(64)               │  ← Sequence modelling on CNN features
│  Dropout(0.3)           │
└────────────┬────────────┘
             │
┌─────────────────────────┐
│  Dense(64, ReLU)        │  ← Classification head
│  Dropout(0.3)           │
│  Dense(6, Softmax)      │  ← Activity probabilities
└─────────────────────────┘
```

---

## 3. Layer-by-Layer Design Choices

### 3.1 Convolutional Blocks
- **Two Conv1D layers** with 64 filters each and kernel size 3 (`padding='same'`).
- Kernel size 3 at 50 Hz captures features spanning ~60 ms — roughly one gait sub-phase.
- `padding='same'` preserves temporal resolution so the LSTM receives a full-length feature map.
- **BatchNormalization** after each convolution stabilises training and allows higher learning rates.
- **Dropout(0.3)** between conv layers prevents co-adaptation of filters.

### 3.2 MaxPooling1D
- Pool size 2 halves temporal dimension from 128 → 64.
- Retains the dominant activation within each 2-step window while halving LSTM computation.
- Too aggressive pooling (e.g., pool_size=4) would lose fine-grained temporal resolution needed to distinguish WALKING_UPSTAIRS from WALKING_DOWNSTAIRS.

### 3.3 LSTM Layer
- A single LSTM with 64 units reads the (64, 64) CNN feature map as a 64-step sequence.
- `return_sequences=False` outputs only the final hidden state — a fixed-length activity summary.
- Unlike the BiLSTM (Member 3), we use a unidirectional LSTM because the CNN has already captured local context in both directions via its receptive field; the LSTM focuses on the temporal ordering of these features.
- Gradient clipping (`clipnorm=1.0`) prevents exploding gradients in the recurrent path.

### 3.4 Classification Head
- `Dense(64, relu)` → `Dropout(0.3)` → `Dense(6, softmax)`.
- Same structure as BiLSTM and Transformer for fair comparison.

---

## 4. Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam (clipnorm=1.0) | Same as BiLSTM for fair comparison |
| Learning rate | 5e-4 | Balanced for CNN+LSTM hybrid |
| Batch size | 64 | Matches team standard |
| Epochs | 60 (max) | With early stopping |
| Early stopping patience | 10 epochs | Monitors `val_loss` |
| LR reduction | ×0.5 after 5 stale epochs | Down to 1e-6 minimum |
| Dropout | 0.3 | Lighter than BiLSTM (0.4) due to BatchNorm regularization |
| Seed | 42 | Team-wide reproducibility |

---

## 5. Parameter Count

With default configuration (`cnn_filters=[64,64], lstm_units=64, dense_units=64`):

| Layer Group | Approximate Parameters |
|---|---|
| Conv1D blocks (2 layers + BN) | ~13,000 |
| LSTM(64) | ~33,000 |
| Dense head (64 + 6) | ~4,500 |
| **Total** | **~50,500** |

This is comparable to the BiLSTM (~148K) but significantly lighter, making it a good efficiency benchmark.

---

## 6. Data Contract Compliance

- **Input shape:** `(N, 128, 9)` — 9-channel inertial sensor windows at 50 Hz.
- **Labels:** Integer 0–5 mapped to WALKING, UPSTAIRS, DOWNSTAIRS, SITTING, STANDING, LAYING.
- **No re-normalization:** Data loaded from `data/uci_har_processed.npz` is already standardized using training statistics.
- **No test leakage:** Training and model selection use `X_train` / `X_val` only. Test evaluation happens exactly once.
- **Subject leakage prevention:** Pipeline asserts `set(subject_train) ∩ set(subject_val) == ∅`.

---

## 7. File Inventory

| File | Purpose |
|---|---|
| `src/models/cnn_lstm.py` | Model builder, compiler, loader, predictor |
| `src/train_cnn_lstm.py` | Training pipeline with callbacks and artifact saving |
| `configs/cnn_lstm.json` | Hyperparameter configuration |
| `tests/test_cnn_lstm.py` | Unit and integration tests |
| `notebooks/03_cnn_lstm.ipynb` | Training notebook with tuning experiments |
| `notebooks/05_model_comparison.ipynb` | Cross-model evaluation benchmark (Member 4 responsibility) |
| `docs/cnn_lstm.md` | This document |

---

## 8. Reproducibility

```bash
# Setup
git checkout monal/cnn-lstm
pip install -r requirements.txt
python -m src.data.uci_har_loader --download

# Train
python src/train_cnn_lstm.py --config configs/cnn_lstm.json

# Test
python -m unittest tests.test_cnn_lstm -v
```

Outputs are saved to `outputs/cnn_lstm/run_<timestamp>/` including `best_model.keras`, `history.json`, and `run_metadata.json`.

---

## 9. Known Limitations and Expected Behaviour

- **SITTING vs STANDING confusion:** Expected for all models — the sensor sees near-identical gravity signals for both sedentary postures.
- **Unidirectional LSTM:** Slightly less expressive than bidirectional, but the CNN's symmetric receptive field compensates. This is a deliberate design trade-off for parameter efficiency.
- **Single LSTM layer:** Adding a second stacked LSTM showed marginal improvement (<0.5% val accuracy) but doubled LSTM parameters. The single-layer variant was selected for efficiency.
