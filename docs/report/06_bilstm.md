## 6. Bidirectional LSTM

### 6.1 Motivation

An activity is defined by how the motion evolves across the window. Walking produces a periodic pattern, while sitting and standing produce a nearly flat one. An LSTM's gated cell state can carry information across all 128 time steps, and its forget gate lets it drop what is irrelevant. We read the window in both directions because each 2.56 s window is classified as a whole after it has been recorded, not streamed. The second half of the window is therefore legitimately available when interpreting the first half.

### 6.2 Architecture

The model (`src/models/bilstm.py`) has two stacked bidirectional LSTM layers with 64 units per direction, followed by a small dense head. The forward and backward outputs are concatenated, so each BiLSTM layer produces 2 × 64 = 128 features. The first layer returns its full sequence so that the second layer can model higher-level temporal patterns. The second layer returns only its final states, which summarise the whole window.

| # | Layer | Output shape | Parameters |
| :---: | :--- | :--- | ---: |
| 1 | Input | (N, 128, 9) | 0 |
| 2 | Bidirectional(LSTM(64, return_sequences=True)) | (N, 128, 128) | 37,888 |
| 3 | Dropout(0.4) | (N, 128, 128) | 0 |
| 4 | Bidirectional(LSTM(64)) | (N, 128) | 98,816 |
| 5 | Dropout(0.4) | (N, 128) | 0 |
| 6 | Dense(64, ReLU) | (N, 64) | 8,256 |
| 7 | Dense(6, softmax) | (N, 6) | 390 |
| | **Total (all trainable)** | | **145,350** |

An LSTM layer with *u* units and *d* input features has 4 · (*u*(*d* + *u*) + *u*) parameters: one weight set for each of its four gates. A bidirectional layer doubles that. The first BiLSTM therefore has 2 · 4 · (64 · (9 + 64) + 64) = 37,888 parameters. The second has 2 · 4 · (64 · (128 + 64) + 64) = 98,816, because its input is the 128-feature output of the first layer. The recurrent layers hold 94% of the model's parameters.

Dropout of 0.4 after each recurrent layer is higher than the Transformer's 0.2. We chose it because the recurrent layers overfit quickly when there are only 17 training subjects.

**Activation functions.** Each layer's activation is chosen for its role:

| Where | Activation | Why |
| :--- | :--- | :--- |
| LSTM gates (input, forget, output) | Sigmoid | Outputs lie in (0, 1), so each gate acts as a soft switch that decides what fraction of information is written, kept or exposed. |
| LSTM cell candidate and cell output | tanh | Outputs lie in (−1, 1) and are centred on zero. Because the cell state is updated additively and the gates stay below 1, it remains bounded over 128 steps, while the zero-centred values keep gradients well-scaled. |
| Dense head | ReLU | Cheap, non-saturating for positive inputs, and it lets the head combine the 128 recurrent features non-linearly without vanishing gradients. |
| Output layer | Softmax | Turns the 6 scores into a probability distribution over mutually exclusive activities, which is exactly what categorical cross-entropy expects. |

The sigmoid and tanh choices are the standard Keras LSTM defaults (`recurrent_activation="sigmoid"`, `activation="tanh"`), and we kept them deliberately. The gating mechanism relies on the bounded (0, 1) range of the sigmoid, so replacing it (for example with ReLU) would break the LSTM's ability to control its memory.

### 6.3 Training setup

| Setting | Value |
| :--- | :--- |
| Loss | Sparse categorical cross-entropy (the same as the other models) |
| Optimiser | Adam, learning rate 5 × 10⁻⁴ (default in `configs/bilstm.json`), `clipnorm = 1.0` |
| Batch size | 64 (93 steps per epoch) |
| Maximum epochs | 60 |
| Seed | 42 (Python, NumPy, TensorFlow and weight initialisation) |
| Hardware | Google Colab T4 GPU (TensorFlow 2.20, Keras 3.13), about 2.7 s per epoch for the 2-layer model |

The default learning rate is lower than the Transformer's 10⁻³ because recurrent networks are more sensitive to step size. Gradient-norm clipping at 1.0 guards against the exploding gradients that backpropagation through 128 time steps can produce. No run in our experiments showed a loss spike or a NaN, including the lr 10⁻³ variant.

**Callbacks** (all four monitor `val_loss`, so the test split plays no part in training):

| Callback | Configuration | Role |
| :--- | :--- | :--- |
| EarlyStopping | patience 10, `restore_best_weights=True` | Ends the run after 10 epochs without improvement and restores the weights from the best epoch |
| ModelCheckpoint | `save_best_only=True` | Writes the best epoch to `best_model.keras` |
| ReduceLROnPlateau | factor 0.5, patience 5, min 10⁻⁶ | Halves the learning rate when validation loss stalls |
| CSVLogger | `history.csv` | Logs metrics for every epoch |

### 6.4 Hyperparameter tuning

Starting from the baseline above, we trained three variants. Each changes exactly **one** hyperparameter, so any difference can be attributed to that change. All four runs share the same data, seed, epoch budget and callbacks. They are ranked by best validation loss, and the validation accuracy at that epoch is reported alongside. The test split was not used for tuning.

| Variant | Units / dir | BiLSTM layers | Learning rate | Params | Best val loss | Val acc at best epoch | Best epoch | Epochs run | s / epoch (T4 GPU) |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | :---: | :---: | ---: |
| **baseline** | 64 | 2 | 5 × 10⁻⁴ | 145,350 | **0.3332** | 88.14% | 2 | 12 | 2.7 |
| layers_1 | 64 | 1 | 5 × 10⁻⁴ | 46,534 | 0.3350 | 88.86% | 5 | 15 | 1.6 |
| lr_1e-3 | 64 | 2 | 1 × 10⁻³ | 145,350 | 0.3442 | **89.50%** | 3 | 13 | 2.9 |
| units_32 | 32 | 2 | 5 × 10⁻⁴ | 40,134 | 0.4234 | 87.50% | 2 | 12 | 2.6 |

Three findings stand out:

- **Width matters more than depth.** Halving the units per direction (`units_32`) gave clearly the worst result: the highest validation loss (0.423) and the lowest validation accuracy (87.5%). Dropping the second layer (`layers_1`) removed 68% of the parameters and 41% of the time per epoch, yet it reached almost the same validation loss as the baseline (0.335 vs 0.333) and a slightly higher validation accuracy. On a GPU, halving the units barely saves time (2.6 vs 2.7 s per epoch), because the 128 sequential time steps, not the layer width, dominate the cost.
- **A higher learning rate trains stably but does not win on loss.** With clipping and ReduceLROnPlateau in place, lr 10⁻³ gave the highest validation accuracy (89.5%) but a higher best validation loss (0.344) than the baseline.
- **The ranking is within noise.** We selected the `baseline` for the final evaluation, because it had the lowest validation loss, the rule fixed before tuning. The top three variants are only 0.011 apart in validation loss, and all four lie within 2 points of validation accuracy. An earlier CPU run with the same seed ranked `lr_1e-3` first, because GPU (cuDNN) kernels are not fully deterministic and the TensorFlow versions differed. With a single seed and only 4 validation subjects, the choice is therefore a modest preference, not a clear win. Because the selected model is the default configuration, `python src/train_bilstm.py` with `configs/bilstm.json` reproduces it.

On the 9 held-out test subjects, the selected model reaches 88.9% accuracy, 0.888 macro F1 and 0.983 macro one-vs-rest ROC-AUC. ROC-AUC measures how well each class is ranked across all windows, not only the top prediction, so it stays high even though 327 of 2,947 windows are misclassified. The largest error is SITTING predicted as STANDING (109 of 491 sitting windows, 75.6% recall), plus 62 STANDING windows predicted as SITTING. Among the dynamic classes the model over-predicts WALKING_DOWNSTAIRS: it has the highest recall of the dynamic classes (98.6%) but the lowest precision of all classes (81.7%), mainly because of WALKING (59) and WALKING_UPSTAIRS (30) windows predicted as downstairs. LAYING is almost perfect (100% precision, 97.6% recall). The selected model's validation accuracy was 88.1%, so the subject-wise validation split was a faithful proxy for the test set. The full comparison with the other architectures is in the shared evaluation section.

### 6.5 Convergence and the role of early stopping

The BiLSTM converges almost immediately. In the baseline run (the selected model), training accuracy goes from 65.0% after epoch 1 to 93.8% after epoch 2 and 95.8% after epoch 3. Validation loss reaches its minimum (0.333) at **epoch 2**. The best epoch is 2–5 for every variant.

Everything after that point is overfitting to the training subjects. In the baseline run, training loss keeps falling from 0.199 to 0.079 while validation loss climbs from 0.333 to 0.435. Training accuracy settles around 96–97%, and validation accuracy plateaus around 88–90%. The gap of about 7–8 points is a subject gap: the model fits the 17 training people better than it generalises to new ones. ReduceLROnPlateau halves the learning rate after epoch 7 (5 × 10⁻⁴ → 2.5 × 10⁻⁴), but that does not recover validation loss. It fires again at epoch 12, which is also the epoch where early stopping ends the run, so the second reduction never takes effect.

As a result, most of the real regularisation comes from **early stopping**, not from the 60-epoch budget. Every run stops after 12–15 epochs, which is about 24–37 s on a T4 GPU. `restore_best_weights=True` then returns the weights from the validation-loss minimum. Without it, the final model would be the epoch-12 model, whose validation loss is 30% higher than the best (0.435 vs 0.333). The 60-epoch limit is never reached. It acts only as an upper bound.
