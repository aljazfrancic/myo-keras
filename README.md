# myo-keras

EMG gesture classification using Keras on Myo armband data.

## What it does

Trains a dense neural network to classify 8 hand gestures from 8-channel EMG signals recorded with a Myo armband. Features are extracted via a causal sliding-window RMS. The notebook supports two data loading modes (curated subset or all sessions) and includes a grokking experiment.

## Dataset

Expects the [myo-readings-dataset](https://github.com/aljazfrancic/myo-readings-dataset) to be located alongside this repo. Specifically, readings are loaded from `_readings_right_hand/` and curated session names from `curated.txt`.

## Setup

```bash
pip install -r requirements.txt
```

## Project structure

| File | Description |
|---|---|
| `myo_utils.py` | Constants and utility functions (RMS, data loading, auto-curation) |
| `myo-keras.ipynb` | Training, evaluation, and grokking experiment |
| `requirements.txt` | Python dependencies |

## Usage

Open `myo-keras.ipynb` and run cells top to bottom.

- **Curated sessions** (default): loads only participants listed in `curated.txt`.
- **All sessions**: uncomment `load_data_all()` and comment out `load_data_curated()`.

Sessions are split deterministically: suffix `-1` for training, `-2` for validation, `-3` for testing.

## Grokking experiment

Grokking (Power et al., 2022) is a phenomenon where a neural network, trained well past memorising its training data, eventually discovers the generalising solution and sees a dramatic improvement in validation accuracy.

| Phase | Train accuracy | Val accuracy | Weight norms |
|-------|---------------|--------------|--------------|
| 1 — Memorisation | Rapid rise to ~100 % | Stays low | Growing |
| 2 — Plateau | Near 100 % | Unchanged | Slowly declining |
| 3 — Grokking | Near 100 % | Sudden jump upward | Declining further |

### Experiment design

The training set is subsampled to ~5 000 samples (stratified, keeping full validation and test sets) so the model memorises before it generalises. Training uses `AdamW` with decoupled weight decay and sweeps over several weight decay values to find the grokking threshold.

| Weight decay | Expected behaviour |
|--------------|--------------------|
| 0 | Memorise and stay memorised (no grokking) |
| 1e-3 | Weak pressure; grokking may appear very late or not at all |
| 1e-2 | Moderate pressure; likely grokking region |
| 5e-2 | Strong pressure; faster grokking, possibly lower final accuracy |
| 1e-1 | Very strong pressure; may prevent memorisation entirely |

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architecture | Dense 200 → 100 → 70 → 8 | ~29 K params; 29:1 ratio with 1 K samples forces memorisation |
| Output activation | `softmax` | Correct for mutually exclusive multi-class |
| Optimiser | `AdamW` | Decoupled weight decay |
| Learning rate | 3e-4 | Slower; allows compression phase to develop |
| Weight decay | [0, 1e-3, 1e-2, 5e-2, 1e-1] | Sweep to find grokking threshold |
| Training subset | 1 000 (stratified) | 29:1 param ratio forces memorisation |
| Epochs | 5 000 | Long enough for delayed generalisation |
| Batch size | 32 (Keras default) | ~31 steps/epoch with 1 K samples |

### Key plots

1. **Validation accuracy overlay** — all weight decay curves on one graph; shows the threshold where grokking appears.
2. **Weight norms overlay** — shows how stronger decay compresses the model.
3. **Per-run loss / accuracy** — train vs val for each weight decay value.
4. **Confusion matrix and classification report** — for the best grokking run, evaluated on the held-out test set.

### References

- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*. arXiv:2201.02177.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019.

## Auto-curation

To regenerate `curated.txt` based on per-participant test accuracy (run after the first notebook cell so `myo_utils` is in scope):

```python
generate_curated()  # no args: uses READINGS_DIR, CURATED_FILE, CURATION_ACCURACY_THRESHOLD from myo_utils
```

Optional args: `generate_curated(readings_dir=..., output_file=..., accuracy_threshold=...)`.

The default accuracy threshold is 0.7 (configurable via `CURATION_ACCURACY_THRESHOLD`).

## Gesture labels

| Index | Gesture |
|---|---|
| 0 | hibernation |
| 1 | flexion |
| 2 | extension |
| 3 | radial deviation |
| 4 | ulnar deviation |
| 5 | pronation |
| 6 | supination |
| 7 | fist |
