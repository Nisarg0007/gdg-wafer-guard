# Wafer Guard — Audit Findings

## Findings (ranked by severity)

1. **Preprocessing and resampling leakage — critical.** In the original notebook's preprocessing section, `SimpleImputer.fit_transform` and `StandardScaler.fit_transform` run before the split, using all rows. SMOTE is then applied before `train_test_split`, allowing synthetic neighbours to be built using observations later assigned to evaluation. The subsequent random split also ignores production time order. **Repair:** sort by timestamp, split chronologically first, fit imputation/scaling on training rows only, and do not resample validation or test data. The repaired notebook reports the actual prevalence in each split.

2. **Corrupted reconstruction-loss gradients — critical.** The original loader adds a channel dimension, producing input tensors `[B, 1, F]`, while the decoder returns `[B, F]`. Subtraction broadcasts to `[B, B, F]`, introducing cross-wafer residuals. **Repair:** use `[B, F]` tensors throughout and assert that reconstruction and input shapes match. The notebook runs a gradient smoke test and checks finite gradients before training.

3. **Invariant blindness / lack of invariant-deviation monitoring — high.** The original pipeline learns imputation and scaling statistics from the entire dataset, rather than defining baselines using only past/training data. That contaminates the temporal boundary and provides no explicit alert when a channel that was invariant in historical training data later departs from its baseline. The original notebook does not explicitly drop constant channels; it would be inaccurate to claim that it does. **Repair:** derive invariant baselines only from the training period, retain sensor channels, and calculate transparent per-row deviation flags for later periods. These flags are a separate monitoring rule, not training features: they are all zero in the training period by construction, so a classifier cannot learn their effects from training examples. Their alerts are reported separately from classifier metrics.

## PS-L legacy sentinel handling

Rows with `sensor_0 == 0.0` are treated as legacy PS-L rows. On those rows, `sensor_0` is treated as a sentinel rather than a measurement; `sensor_1` values 100, 200, and 300 are validated and represented using one-hot categorical indicators. The `sensor_1` code is set missing in the continuous matrix before imputation/scaling. Non-legacy `sensor_1` measurements remain continuous. The notebook asserts that no unexpected firmware codes occur on sentinel rows.

## Evaluation protocol

Rows are ordered by timestamp and split chronologically 70%/15%/15% into training, validation, and test periods. Imputation and scaling are fitted on training rows only. The supervised autoencoder is trained on training data; early stopping and threshold selection use validation data only. No SMOTE is used. The final test period is evaluated once at its observed prevalence. The dataset-wide prevalence is about 6.64%, but temporal test prevalence can differ; the notebook reports both rather than artificially balancing the test set.

Classifier precision, recall, F1, ROC-AUC, average precision, and confusion matrix are calculated from classifier predictions alone using the validation-selected threshold. Invariant-monitor alerts are counted separately so the reported confusion matrix does not mix a learned classifier with a deterministic rule.

## Residual risk / recommendation

**No-go for production on this evidence.** The dataset has only 104 defective lots, and the chronological test period has few positives. Precision/recall/F1 and the chosen threshold are consequently unstable; temporal distribution shift is material. Collect more later-period labelled data, review false alarms with process engineers, and define drift-monitoring and retraining procedures before deployment.
