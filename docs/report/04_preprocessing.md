## 4. Data Preprocessing and Feature Engineering

All four models are trained and evaluated on the same processed dataset, `data/uci_har_processed.npz`. It is built once by the shared loader (`src/data/uci_har_loader.py`) and must pass the shared data contract (`src/data_contract.py`) before any model sees it. Because every member uses the same file, differences in results between the four architectures come from the models, not from different preprocessing.

### 4.1 Data quality and cleaning

The dataset authors had already filtered the raw inertial signals for noise, with a median filter and a 3rd-order low-pass Butterworth filter at 20 Hz. They also separated body acceleration from gravity with a second Butterworth filter at 0.3 Hz. We therefore apply no additional signal filtering. We checked the 9 raw signal channels of both official splits before any processing:

| Check | Train (7,352 windows) | Test (2,947 windows) |
| :--- | :---: | :---: |
| Missing values (NaN) | 0 | 0 |
| Infinite values | 0 | 0 |
| Exactly duplicated windows | 0 | 0 |
| Labels outside 1–6 | 0 | 0 |

No imputation, row removal or de-duplication was needed. The absence of NaN and infinite values is not assumed after this one-off check: the data contract re-checks it every time the dataset is loaded (Section 4.8).

The one data-quality issue that does need handling is structural: consecutive windows overlap by 50%. It creates no errors inside a split, but it makes a random split leak information (Section 4.5).

### 4.2 Feature engineering: raw signals, not the 561 hand-crafted features

UCI HAR ships two representations of each window: 561 hand-engineered time- and frequency-domain features (means, energies, FFT band statistics and more), and the 9 raw inertial signals they were computed from. We deliberately use the **raw signals**, for three reasons:

1. **The project compares architectures that learn features themselves.** Convolutions, recurrence and self-attention are all designed to extract temporal patterns from a sequence. Feeding them the 561 features would bypass exactly what is being compared.
2. **The 561 features have no time axis.** Each window is reduced to one flat vector, so there is no sequence left for an LSTM or a Transformer to model.
3. **Fair comparison.** All four models receive identical inputs, so no model benefits from domain knowledge built into the features.

Each window is 2.56 s sampled at 50 Hz, which gives 128 time steps. The loader stacks the 9 per-channel signal files along the last axis, so each split becomes a tensor of shape `(N, 128, 9)` (float32). The channel order is fixed by the contract:

`body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z`

Accelerations are in units of standard gravity (g), and angular velocities are in rad/s. No further features (such as magnitudes, jerk signals or FFT coefficients) are added. Any such feature extraction is left to the networks. The loader can still load the 561 features (`load_engineered_features`), for a classical machine-learning baseline only.

### 4.3 Label encoding

UCI HAR stores activity labels as integers 1–6. The loader subtracts 1, so the labels become 0–5 and can be used directly as class indices with sparse categorical cross-entropy. No one-hot encoding is stored, because sparse cross-entropy takes integer labels directly. The loader raises an error if any raw label falls outside 1–6.

| Raw label | Class index | Activity |
| :---: | :---: | :--- |
| 1 | 0 | WALKING |
| 2 | 1 | WALKING_UPSTAIRS |
| 3 | 2 | WALKING_DOWNSTAIRS |
| 4 | 3 | SITTING |
| 5 | 4 | STANDING |
| 6 | 5 | LAYING |

### 4.4 Subject-wise train / validation / test split

The official UCI HAR split puts 21 subjects (7,352 windows) in train and 9 subjects (2,947 windows) in test. We keep the official test set unchanged. We carve the validation set out of the official training subjects by holding out **4 whole subjects**, which gives a 17 / 4 / 9 subject split. The validation subjects are drawn with a fixed seed (42). If a draw leaves any class missing from train or validation, the loader deterministically re-draws with seed + 1, seed + 2, and so on, until all six classes appear in both.

| Split | Subjects | Windows | Subject IDs |
| :--- | :---: | ---: | :--- |
| Train | 17 | 5,952 | 1, 5, 6, 7, 8, 11, 14, 15, 17, 19, 21, 25, 26, 27, 28, 29, 30 |
| Validation | 4 | 1,400 | 3, 16, 22, 23 |
| Test | 9 | 2,947 | 2, 4, 9, 10, 12, 13, 18, 20, 24 |

No subject appears in more than one split. All six classes appear in every split, and the classes are roughly balanced. The largest class (LAYING) has 1.4 times as many training windows as the smallest (WALKING_DOWNSTAIRS), so no resampling or class weighting is applied:

| Class | Train | Val | Test |
| :--- | ---: | ---: | ---: |
| WALKING | 1,012 | 214 | 496 |
| WALKING_UPSTAIRS | 870 | 203 | 471 |
| WALKING_DOWNSTAIRS | 800 | 186 | 420 |
| SITTING | 1,035 | 251 | 491 |
| STANDING | 1,104 | 270 | 532 |
| LAYING | 1,131 | 276 | 537 |

The validation split is used for early stopping, learning-rate scheduling, checkpoint selection and hyperparameter tuning. The test split is used exactly once per model, for the final evaluation.

### 4.5 Why the split must be by subject (leakage)

UCI HAR windows overlap by 50%. Consecutive 128-sample windows are taken with a stride of 64 samples, so each window shares half of its raw readings with the window before it and half with the window after it. If windows were shuffled and split at random, almost every validation window would have a near-duplicate neighbour in the training set that contains 64 of its 128 readings. The same person's gait, body build and phone placement would also appear on both sides of the split. Validation accuracy would then measure how well the model recognises windows it has effectively already seen, not how well it generalises to a new person. Because early stopping and tuning both read the validation score, this inflated score would also bias model selection. Holding out whole subjects removes both problems. The validation set then behaves like the test set: every validation window comes from people the model has never trained on.

### 4.6 Train-only standardisation

The raw channels are on very different scales. In the training split, body acceleration stays within about ±1.4 g, while angular velocity reaches about ±6 rad/s. Without rescaling, the gyroscope channels would dominate the early gradient updates. Each of the 9 channels is therefore standardised to zero mean and unit variance with `apply_training_standardization`. The mean and standard deviation are computed over all training windows and time steps (axes 0 and 1, which gives one value per channel). The same fixed statistics are then applied unchanged to the validation and test splits. If a channel's standard deviation is below 1e-8, it is replaced by 1 to avoid division by zero.

Fitting the statistics on validation or test data would leak information about the held-out subjects into preprocessing. As a check, the per-channel means after standardisation are exactly 0 on the training split but not on validation, where they range from −0.068 to +0.032. That is the expected result when the validation data plays no part in the fit. The saved NPZ is already standardised, so every training config sets `data.normalize: false` to avoid standardising a second time.

### 4.7 No data augmentation

No augmentation (such as jittering, scaling or time-warping) is applied in the shared pipeline. There are three reasons. First, the dataset is balanced and every class has at least 800 training windows. Second, the 50% overlap already gives each movement pattern several shifted views. Third, and most importantly, keeping augmentation out of the shared data means all four architectures are compared on identical inputs. Augmentation remains a candidate improvement. In particular, sensor rotation would simulate different phone placements and could target the subject gap discussed in the results.

### 4.8 Contract validation

Before the NPZ is saved, and again every time a pipeline loads it, `validate_har_dataset` checks the data and raises `DataContractValidationError` in any of these cases:

- a required key (`X_train`, `y_train`, `subject_train`, `X_val`, `y_val`, `subject_val`) is missing;
- `X` is not shaped `(N, 128, 9)`, or `y` or `subject` does not have one entry per window;
- `X` contains NaN or infinite values;
- labels are not integers in 0–5;
- the train or validation split is missing any of the six classes;
- any subject appears in both train and validation, or (when the test split is checked) in both train and test.

Validation subjects are drawn only from the official training subjects, so they cannot overlap with test. The loader's summary also prints all three pairwise overlaps (train/val, train/test and val/test), and all three are empty. Our pipelines leave the leakage check switched on: they fix the split so that it passes, rather than bypassing it.
