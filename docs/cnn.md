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
| kernel_size=5 (baseline) | 66,502 | 3 | 0.3215 | **90.43%** | 55.3s |
| kernel_size=3 | 57,158 | 3 | **0.3175** | 88.21% | 52.7s |

kernel_size=3 achieves a marginally lower validation loss with ~14% fewer parameters, 
but a noticeably lower validation accuracy than kernel_size=5. Since accuracy is the 
primary metric for this classification task and the difference is meaningful (~2.2 
points), **kernel_size=5 is selected as the final configuration**. The smaller kernel 
may struggle slightly more to capture the footstep-length patterns in the dynamic 
activities, which typically span more than 3 time steps within the 128-step window.