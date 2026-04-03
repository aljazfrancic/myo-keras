# myo-keras

EMG gesture classification using Keras on Myo armband data.

## What it does

Trains a dense neural network to classify 8 hand gestures from 8-channel EMG signals recorded with a Myo armband. Features are extracted via a causal sliding-window RMS. The notebook supports two data loading modes (curated subset or all sessions) and includes a grokking experiment.

## Dataset

Expects the [myo-readings-dataset](https://github.com/aljazfrancic/myo-readings-dataset) to be located alongside this repo. Specifically, readings are loaded from `_readings_right_hand/` and curated session names from `curated.txt`.

### Performance note

In our testing, training and the grokking experiment ran **faster on CPU** than on GPU. The workloads here are dominated by small dense models, full-batch updates, and frequent validation rather than large batched operations that GPUs typically accelerate, so the CPU TensorFlow stack in `requirements.txt` is both the documented setup and a practical default.

## Project structure

| File | Description |
|---|---|
| `myo_utils.py` | Constants and utility functions (RMS, data loading, auto-curation) |
| `myo-keras.ipynb` | Training, evaluation, and grokking experiment |
| `requirements.txt` | Python dependencies (TensorFlow, NumPy, etc.) |

## Usage

Open `myo-keras.ipynb` and run cells top to bottom.

- **Curated sessions** (default): loads only the session directories listed in `curated.txt`.
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

The grokking sweep reloads the data using an RMS window of 30 to reduce smoothing and make generalisation harder. The training set is then subsampled to 100 samples (stratified) so the model memorises before it generalises. The validation set is also stratified-subsampled to 8,000 samples for fast periodic evaluation. Training uses `AdamW` with decoupled weight decay and runs the **Cartesian product** of `GROKKING_ARCHITECTURES × GROKKING_LRS × GROKKING_WEIGHT_DECAYS` (18 runs). Each run resets `tf.random.set_seed(GROKKING_INIT_SEED)` before building the model so all runs share the same initial weights; only the listed hyperparameters differ. The notebook prints **per-run and total wall-clock time** for the sweep.

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architectures (hidden layers) | `[64, 32]`, `[100, 50]`, `[200]` | Vary capacity and depth; softmax output unchanged |
| Output activation | `softmax` | Correct for mutually exclusive multi-class |
| Optimiser | `AdamW` | Decoupled weight decay |
| Learning rates | `3e-4`, `1e-4` | Sweep slower vs slightly slower updates |
| RMS window (grokking sweep) | 30 | Less smoothing; increases memorisation/generalisation separation |
| Weight decay | [0.05, 0.1, 0.2] | Moderate decay grid |
| Init seed | `GROKKING_INIT_SEED` (42) | Same `keras.Sequential` init for every sweep run |
| Training subset | 100 (stratified) | Larger parameter-to-sample gap encourages memorisation first |
| Validation subset | 8,000 (stratified) | Reliable metric estimates with much lower validation cost |
| Epochs | 150,000 | Long horizon for delayed generalisation |
| Batch size | full-batch (`len(grok_train)`) | One gradient step per epoch for stable grokking dynamics |
| Metric logging | every 100 epochs | Tracks long-run trends without per-epoch validation overhead |

### Key plots

1. **Validation accuracy overlay** — 3×2 grid (architecture × learning rate); in each panel, all weight-decay curves overlaid.
2. **Weight norms overlay** — same grid; L2 norm of trainable weights vs epoch.
3. **Per-run loss / accuracy** — one row per Cartesian-product run (full **arch**, **lr**, **wd** in titles).
4. **Val accuracy vs weight norm (dual-axis)** — same 3×2 grid; in each panel, solid lines = validation accuracy, dashed = weight norm, matched colors per `wd`.

### References

- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*. [arXiv:2201.02177](https://arxiv.org/abs/2201.02177).
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019.

## Auto-curation

To regenerate `curated.txt` based on per-participant test accuracy (run after the first notebook cell so `myo_utils` is in scope):

```python
generate_curated()  # no args: uses READINGS_DIR, CURATED_FILE, CURATION_ACCURACY_THRESHOLD from myo_utils
```

Optional arguments: `generate_curated(readings_dir=..., output_file=..., accuracy_threshold=...)`.

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
