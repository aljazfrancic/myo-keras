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
| `grokking.py` | Grokking sweep runner, callback, model build, and plot helpers |
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

All grokking hyperparameters live in [`myo_utils.py`](myo_utils.py). Sweep execution and plotting helpers live in [`grokking.py`](grokking.py). The notebook reloads curated data with RMS window `GROKKING_RMS_WINDOW`, then builds stratified subsets of size `GROKKING_TRAIN_SUBSET` (train) and `GROKKING_VAL_SUBSET` (validation) for faster periodic evaluation. Training uses `AdamW` and runs **one full training job per tuple** in the Cartesian product

`GROKKING_ARCHITECTURES × GROKKING_LRS × GROKKING_WEIGHT_DECAYS × GROKKING_SEEDS`.

Any of those lists may have length 1 (a single choice is still a valid sweep). Before each run, `run_grok_sweep` calls `tf.random.set_seed(seed)` with the current `seed` from `GROKKING_SEEDS`. It prints **per-run and total wall-clock time**. To resize the grid, edit those lists and scalars in `myo_utils.py` (and restart the kernel / re-import if the notebook is already running).

For reproducibility across machines, confirm `curated.txt` in the dataset repo has not changed since your reference run; regenerating it with `generate_curated()` can change which sessions enter the curated split.

### Parameters (where to configure)

| What | Constant | Role |
|------|----------|------|
| Hidden-layer shapes | `GROKKING_ARCHITECTURES` | Each entry is a list of Dense widths (excluding the softmax head, which `build_grok_model` adds). |
| Learning rates | `GROKKING_LRS` | AdamW learning rate per run. |
| Weight decays | `GROKKING_WEIGHT_DECAYS` | AdamW decoupled weight decay per run. |
| Initialisation seeds | `GROKKING_SEEDS` | Passed to `tf.random.set_seed` before each model build. |
| Epochs per run | `GROKKING_EPOCHS` | No early stopping in `run_grok_sweep`. |
| RMS window (grok reload) | `GROKKING_RMS_WINDOW` | Feature extraction window for the grokking data load only. |
| Train subset size | `GROKKING_TRAIN_SUBSET` | Stratified subsample of the curated train split. |
| Validation subset size | `GROKKING_VAL_SUBSET` | Stratified subsample of the curated validation split (used for logged metrics). |
| Log / eval cadence | `GROKKING_LOG_EVERY` | Training still runs every epoch; **logged** train loss, train accuracy, validation metrics, and weight norm share one cadence: multiples of this value and the final epoch. |
| Model output | `softmax` over `NUM_GESTURES` | Mutually exclusive 8-class classification (set in `build_grok_model`). |
| Batch size | Full batch | `run_grok_sweep` uses `batch_size=len(grok_train)` (one optimizer step per epoch). |

Subsampling uses `subsample_data` in [`myo_utils.py`](myo_utils.py); the notebook uses that helper’s default RNG seed unless you change the call.

### Key plots

1. **Validation accuracy overlay** — For each fixed **(architecture, learning rate, weight decay)** triple, one figure overlays **all seeds** from `GROKKING_SEEDS`.
2. **Weight norms overlay** — Same grouping as (1); L2 norm of trainable weights vs epoch.
3. **Per-run loss / accuracy** — One figure per Cartesian-product run; each is a **1×2** panel (loss | accuracy), train vs validation.
4. **Val accuracy vs weight norm (dual-axis)** — One figure per run; solid = validation accuracy, dashed = L2 weight norm.

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
