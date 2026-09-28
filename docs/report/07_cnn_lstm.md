## 7. Hybrid CNN-LSTM

**Author:** Monal (Member 4)  
**Architecture:** 1D-CNN Feature Extractor → Temporal Downsampling → LSTM Sequence Modeller → Dense Head  
**Framework:** TensorFlow / Keras (trained from scratch)  

---

### 7.1 Motivation

Human activities captured by body-mounted triaxial accelerometers and gyroscopes possess **two distinct scales of temporal structure**:

1. **Local, high-frequency transients (10–30 timesteps):** Biomechanical events such as an individual foot strike during a walking cycle, a heel impact on a stair tread, or sudden arm acceleration. These localized wavelets are best detected by small 1D convolutional kernels ($k=3$) operating across adjacent sensor readings.
2. **Global, macroscopic sequence transitions (64–128 timesteps):** The sustained rhythm of a gait cycle, postural stabilization, or continuous stillness. These extended temporal dependencies require recurrent memory states.

A pure 1D-CNN (Member 2) excels at local feature extraction but possesses no recurrent memory across long horizons. Conversely, a pure Bidirectional LSTM (Member 3) models full-window recurrence but treats all 128 raw timesteps with uniform computational complexity, resulting in high parameter counts (145,350 parameters) and slow sequential backpropagation.

The **Hybrid CNN-LSTM** reconciles both paradigms:
* Two lightweight 1D convolutional blocks extract localized temporal representations from the 9 raw sensor channels.
* A `MaxPooling1D` layer performs temporal downsampling, halving the time resolution from 128 to 64 steps ($128 \to 64$).
* An LSTM layer models the sequential evolution over the compressed representation, requiring only 64 unrolling steps rather than 128.

This architectural synergy yields a model with **52,230 parameters**—a 64% reduction in parameter complexity relative to BiLSTM—while retaining high temporal representational capacity.

---

### 7.2 Architecture

The network (`src/models/cnn_lstm.py`) is structured in three modular stages:

```
Input: (batch, 128, 9)
        │
        ▼
┌─────────────────────────────────┐
│  Conv1D(64, k=3, padding='same')│  ← Local feature extraction
│  BatchNormalization             │
│  ReLU + Dropout(0.3)            │
└───────────────┬─────────────────┘
                │
┌─────────────────────────────────┐
│  Conv1D(64, k=3, padding='same')│  ← Hierarchical feature composition
│  BatchNormalization             │
│  ReLU                           │
└───────────────┬─────────────────┘
                │
┌─────────────────────────────────┐
│  MaxPooling1D(pool_size=2)      │  ← Temporal downsampling (128 → 64)
│  Dropout(0.3)                   │
└───────────────┬─────────────────┘
                │
┌─────────────────────────────────┐
│  LSTM(64 units)                 │  ← Recurrent sequence modelling
│  Dropout(0.3)                   │
└───────────────┬─────────────────┘
                │
┌─────────────────────────────────┐
│  Dense(64, ReLU)                │  ← Classification projection
│  Dropout(0.3)                   │
│  Dense(6, Softmax)              │  ← Normalized activity probabilities
└─────────────────────────────────┘
```

#### Layer-by-Layer Dimension & Parameter Breakdown

| # | Layer | Output Shape | Parameters | Computation Rationale |
| :-: | :--- | :--- | ---: | :--- |
| 1 | `Input` | `(N, 128, 9)` | 0 | Raw 9-channel inertial windows (2.56 s @ 50 Hz) |
| 2 | `Conv1D(64, k=3, 'same')` | `(N, 128, 64)` | 1,792 | $64 \cdot (3 \cdot 9 + 1) = 1,792$ |
| 3 | `BatchNormalization` | `(N, 128, 64)` | 256 | Scales & centers convolutional feature maps ($4 \cdot 64 = 256$) |
| 4 | `Activation('relu') + Dropout(0.3)` | `(N, 128, 64)` | 0 | Non-linear activation and spatial regularisation |
| 5 | `Conv1D(64, k=3, 'same')` | `(N, 128, 64)` | 12,352 | $64 \cdot (3 \cdot 64 + 1) = 12,352$ |
| 6 | `BatchNormalization` | `(N, 128, 64)` | 256 | Internal covariate shift stabilisation |
| 7 | `Activation('relu')` | `(N, 128, 64)` | 0 | Element-wise ReLU non-linearity |
| 8 | `MaxPooling1D(pool_size=2)` | `(N, 64, 64)` | 0 | Temporal decimation: halves sequence length $128 \to 64$ |
| 9 | `Dropout(0.3)` | `(N, 64, 64)` | 0 | Feature map dropout |
| 10 | `LSTM(64)` | `(N, 64)` | 33,024 | $4 \cdot (64 \cdot (64 + 64) + 64) = 33,024$ |
| 11 | `Dropout(0.3)` | `(N, 64)` | 0 | Recurrent output regularisation |
| 12 | `Dense(64, activation='relu')` | `(N, 64)` | 4,160 | $64 \cdot 64 + 64 = 4,160$ |
| 13 | `Dropout(0.3)` | `(N, 64)` | 0 | Dense feature regularisation |
| 14 | `Dense(6, activation='softmax')` | `(N, 6)` | 390 | $64 \cdot 6 + 6 = 390$ |
| | **Total Parameters** | | **52,230** | **51,974 Trainable / 256 Non-trainable (BN)** |

#### Design of Activation Functions

| Component | Activation | Design Justification |
| :--- | :--- | :--- |
| **Conv1D Blocks & Dense Head** | **ReLU** ($\max(0, x)$) | Computationally efficient, avoids vanishing gradients, and introduces sparsity in feature detectors. |
| **LSTM Recurrent Gates** | **Sigmoid** ($\sigma(x) \in (0, 1)$) | Acts as a smooth binary gating mechanism controlling information flow through input, forget, and output gates. |
| **LSTM Candidate & State** | **tanh** ($\tanh(x) \in (-1, 1)$) | Zero-centered, bounded dynamic range preventing gradient saturation over 64 unrolled recurrence steps. |
| **Output Layer** | **Softmax** | Converts raw logits into a valid categorical probability distribution $\sum_{i=1}^6 p_i = 1$ over mutually exclusive activity classes. |

---

### 7.3 Training Setup

| Setting | Configuration | Details |
| :--- | :--- | :--- |
| **Loss Function** | Sparse Categorical Cross-Entropy | Directly evaluates integer targets ($y \in \{0, \dots, 5\}$) without one-hot expansion. |
| **Optimiser** | Adam (`learning_rate = 5 × 10⁻⁴`, `clipnorm = 1.0`) | Adaptive moment estimation. Gradient norm clipping at 1.0 prevents gradient explosion during BPTT. |
| **Batch Size** | 64 | 93 iterations per epoch over the 5,952 training windows. |
| **Maximum Epochs** | 60 | Generous ceiling with early termination governed by validation loss. |
| **Random Seed** | 42 | Deterministic seeding across Python, NumPy, and TensorFlow. |
| **Leakage Protocol** | Subject-Stratified Split | Zero subject overlap between training (17 subjects) and validation (4 subjects). |

#### Training Callbacks

All callbacks monitor validation loss (`val_loss`) strictly, ensuring zero exposure to the held-out test split:
* **`EarlyStopping`:** `monitor='val_loss'`, `patience=10`, `restore_best_weights=True`.
* **`ModelCheckpoint`:** Saves the optimal weights to `outputs/cnn_lstm/run_<timestamp>/best_model.keras` when validation loss reaches a new minimum.
* **`ReduceLROnPlateau`:** `factor=0.5`, `patience=5`, `min_lr=1e-6`. Reduces step size when convergence stalls.
* **`CSVLogger`:** Persists epoch-by-epoch metrics to `history.csv` for post-hoc convergence audits.

---

### 7.4 Hyperparameter Tuning

To systematically explore architectural trade-offs, three variants were trained under identical conditions (same seed, data contract, and callbacks). Each experiment modified a targeted structural parameter:

| Variant | Convolutional Backbone | LSTM Depth / Units | Parameters | Best Val Loss | Val Acc @ Best | Best Epoch | Epochs Run | Hardware Duration |
| :--- | :---: | :---: | ---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 2 × Conv1D(64), MaxPool(2) | 1 × LSTM(64) | **52,230** | **0.3573** | 87.14% | **4** | **14** | ~137 s |
| **Lightweight** | 2 × Conv1D(32), MaxPool(2) | 1 × LSTM(32) | 15,270 | 0.4176 | 87.79% | 7 | 17 | ~112 s |
| **Deeper Recurrent** | 2 × Conv1D(64), MaxPool(2) | 2 × LSTM(64) | 85,254 | 0.3649 | 87.36% | 5 | 15 | ~210 s |

#### Findings & Selection Rationale:
1. **Validation Loss Superiority:** The Baseline architecture achieved the lowest validation loss (**0.3573** vs 0.4176 for Lightweight and 0.3649 for Deeper Recurrent), indicating superior probability calibration.
2. **Diminishing Returns of Stacking LSTMs:** Adding a second recurrent layer increased parameter count by 63% (from 52K to 85K) and almost doubled training duration without yielding a validation loss improvement.
3. **Selection:** The **Baseline variant** was chosen as the primary model. On the 9 unseen test subjects (2,947 samples), the model achieved:
   * **Test Accuracy:** **88.87%**
   * **Macro F1-Score:** **0.8872**
   * **Weighted F1-Score:** **0.8885**

---

### 7.5 Convergence & The Role of Early Stopping

The training dynamics of the CNN-LSTM reveal rapid convergence:

* **Initial Acceleration (Epochs 1–4):** The model progresses from 61.7% training accuracy and 0.788 val loss in Epoch 1 to **87.14% validation accuracy and 0.3573 validation loss at Epoch 4**.
* **Overfitting Boundary:** After Epoch 4, training loss continues to decrease monotonically (from 0.155 to 0.089), while validation loss begins to fluctuate upward into the 0.40–0.44 range. This divergence represents the "subject gap"—the network fitting the idiosyncrasies of the 17 training subjects.
* **Early Stopping Intervention:** `ReduceLROnPlateau` halved the learning rate at Epoch 9 ($5 \times 10^{-4} \to 2.5 \times 10^{-4}$) and again at Epoch 14. At Epoch 14, `EarlyStopping` terminated the training run, avoiding unnecessary computation.
* **Weight Restoration:** `restore_best_weights=True` rewound model parameters back to the Epoch 4 checkpoint. Without this mechanism, evaluating the Epoch 14 model would result in a degraded test accuracy due to subject-specific memorization.
