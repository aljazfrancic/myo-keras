# myo-keras

EMG gesture classification using Keras on Myo armband data.

## What it does

Trains a dense neural network to classify 8 hand gestures from 8-channel EMG signals recorded with a Myo armband. Features are extracted via a causal sliding-window RMS. The notebook supports two data loading modes (curated subset or all sessions) and includes a grokking experiment.

## Dataset

Expects the [myo-readings-dataset](https://github.com/aljazfrancic/myo-readings-dataset) to be located alongside this repo. Specifically, readings are loaded from `_readings_right_hand/` and curated session names from `curated.txt`.

### Performance note

In our testing, training and the grokking experiment ran **faster on CPU** than on GPU. The workloads here are dominated by small dense models, full-batch updates, and periodic validation (every `GROKKING_LOG_EVERY` epochs during the sweep) rather than large batched operations that GPUs typically accelerate, so the CPU TensorFlow stack in `requirements.txt` is both the documented setup and a practical default.

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

The grokking sweep reloads the data using an RMS window of 30 to reduce smoothing and make generalisation harder. The training set is then subsampled to 100 samples (stratified, default subsample seed 42) so the model memorises before it generalises. The validation set is also stratified-subsampled to 8,000 samples for fast periodic evaluation. Training uses `AdamW` with decoupled weight decay and runs the **Cartesian product** of `GROKKING_ARCHITECTURES × GROKKING_LRS × GROKKING_WEIGHT_DECAYS × GROKKING_SEEDS`. With the defaults in `myo_utils.py`, architecture, learning rate, and weight decay are fixed to the values that previously showed grokking on this setup; only **random seeds** vary (**10** runs). Each run calls `tf.random.set_seed(seed)` before building the model so you can compare initialisation sensitivity. The notebook prints **per-run and total wall-clock time** for the sweep. Edit the four lists in `myo_utils.py` for a wider hyperparameter grid (for example, restore multiple weight decays and set `GROKKING_SEEDS` to a single winning seed after you find one).

For reproducibility across machines, confirm `curated.txt` in the dataset repo has not changed since your reference run; regenerating it with `generate_curated()` can change which sessions enter the curated split.

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architectures (hidden layers) | `[200, 100, 70]` | Matches the setup that previously produced grokking-like dynamics |
| Output activation | `softmax` | Correct for mutually exclusive multi-class |
| Optimiser | `AdamW` | Decoupled weight decay |
| Learning rate | `3e-4` | Fixed to match the successful historical run |
| RMS window (grokking sweep) | 30 | Less smoothing; increases memorisation/generalisation separation |
| Weight decay | `0.1` | Fixed to match the successful historical run; expand the list to overlay multiple curves |
| Seeds | `GROKKING_SEEDS` (10 values) | Sweep initialisation; grokking on noisy data may depend on init |
| Training subset | 100 (stratified) | Larger parameter-to-sample gap encourages memorisation first |
| Validation subset | 8,000 (stratified) | Reliable metric estimates with much lower validation cost |
| Epochs | 80,000 | Enough headroom past ~40k where a jump was observed previously |
| Batch size | full-batch (`len(grok_train)`) | One gradient step per epoch for stable grokking dynamics |
| Metric logging | every 100 epochs | Finer curves around late jumps in validation accuracy |

### Key plots

1. **Validation accuracy overlay** — one figure per **(architecture, learning rate, weight decay)**; in each, all **seed** curves overlaid.
2. **Weight norms overlay** — same layout as (1); L2 norm of trainable weights vs epoch, by seed.
3. **Per-run loss / accuracy** — one figure per Cartesian-product run (**arch** × **lr** × **wd** × **seed**); each figure is a **1×2** subplot (loss | accuracy) with train vs validation for that run only.
4. **Val accuracy vs weight norm (dual-axis)** — one figure per sweep run; solid = validation accuracy (left axis), dashed = L2 weight norm (right axis).

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
