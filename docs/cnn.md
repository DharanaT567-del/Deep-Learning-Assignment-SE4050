# 1D-CNN — Training Notes (vdrperera)

## Architecture
See `src/models/cnn.py` for full implementation. Summary:
- Input: 128 time steps × 9 channels
- 2× Conv1D(64, kernel=5) + BatchNorm + ReLU
- MaxPooling1D(2) + Dropout(0.3)
- Conv1D(128, kernel=3) + BatchNorm + ReLU
- GlobalAveragePooling1D
- Dense(128, ReLU) + Dropout(0.5)
- Dense(6, softmax)
- Total parameters: 66,502

## Baseline Run (configs/cnn.json)
- Optimizer: Adam, lr=1e-3, batch_size=64
- Trained 13 epochs before early stopping (patience=10)
- **Best epoch: 3** (val_loss=0.321, val_accuracy=90.4%)
- Training accuracy continued climbing to ~97% while validation loss rose after 
  epoch 3 — clear evidence of overfitting, correctly caught by early stopping with 
  `restore_best_weights=True`.
- Training time: 55.27s total (~4.3s/epoch)

## Tuning experiments

| Variant | Parameters | Best Epoch | Val Loss | Val Accuracy | Train Time |
|---|---|---|---|---|---|
| kernel_size=5, dropout_conv=0.3 (baseline) | 66,502 | 3 | 0.3215 | **90.43%** | 55.3s |
| kernel_size=3, dropout_conv=0.3 | 57,158 | 3 | **0.3175** | 88.21% | 52.7s |
| kernel_size=5, dropout_conv=0.5 | 66,502 | 3 | 0.3571 | 89.50% | 52.4s |

**Selected configuration: kernel_size=5, dropout_conv=0.3 (the original baseline)** — 
it achieves the highest validation accuracy of the three variants tested.

- Reducing kernel_size to 3 slightly lowered parameter count and validation loss, but 
  noticeably reduced accuracy (-2.2 points) — likely because the smaller receptive field 
  captures footstep-length patterns less effectively across the 128-step window.
- Increasing dropout_conv to 0.5 performed worse than the 0.3 baseline on both metrics. 
  This suggests the baseline model wasn't overfitting severely enough to benefit from 
  heavier regularization — early stopping at epoch 3 was already preventing serious 
  overfitting, so the extra dropout mainly added noise to training rather than solving 
  a real generalization problem.

All three variants converged to their best result by epoch 3, with early stopping 
consistently triggering around epoch 13 (patience=10) — indicating the model learns 
the dominant patterns in this dataset very quickly, after which validation performance 
plateaus or degrades due to overfitting on the training set.