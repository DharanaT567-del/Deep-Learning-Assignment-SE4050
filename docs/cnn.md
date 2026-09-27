# 1D-CNN — Training Notes (Member 2)

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
(to be added — testing kernel_size and dropout variants)