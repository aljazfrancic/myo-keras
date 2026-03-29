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

### GPU (optional)

**Where it works:** `tensorflow[and-cuda]` (CUDA 12 via pip) is supported on **Linux** and inside **WSL2** (Ubuntu). It does **not** install cleanly on **native Windows** PowerShell/CMD (see *ResolutionImpossible* below).

#### NVIDIA GPU + TensorFlow 2.16+ (Windows via WSL2)

Do this once per machine; then use the Linux-side env for this repo.

1. **Windows driver** — Install the latest [NVIDIA driver for your GPU](https://www.nvidia.com/Download/index.aspx). WSL uses the host driver; CUDA inside Linux comes from pip with `tensorflow[and-cuda]`.

2. **WSL2 + Ubuntu** — [Install WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and an Ubuntu distribution (`wsl --install -d Ubuntu` or similar). Open **Ubuntu** (not PowerShell) for the steps below.

3. **Check the GPU from Linux** — In the Ubuntu terminal:

   ```bash
   nvidia-smi
   ```

   If this fails, fix WSL GPU support (Windows Update, driver, `wsl --shutdown` then retry) before continuing.

4. **Project on the Linux filesystem** — Clone or copy `myo-keras` and `myo-readings-dataset` under your Linux home, e.g. `~/projects/myo-keras`, so paths stay fast and correct. Open the folder from Cursor/VS Code using **WSL: Ubuntu** (Remote-WSL).

5. **Python 3.10–3.12 venv** (recommended):

   ```bash
   cd ~/projects/myo-keras   # your path
   python3 -m venv .venv   # or .venv-gpu if you keep a separate GPU env
   source .venv/bin/activate   # or .venv-gpu/bin/activate
   pip install -U pip wheel
   pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel keras 2>/dev/null || true
   pip install -r requirements-gpu.txt
   ```

   Optional: `pip install ipykernel` and register a kernel for this venv, then pick it in Jupyter/Cursor.

6. **Notebook** — The first cell installs from `_REQ_FILE`. The checked-in notebook defaults to **`"requirements-gpu.txt"`** (WSL/Linux + NVIDIA). For **native Windows CPU** or **Linux without a usable NVIDIA GPU**, set **`"requirements.txt"`** instead before running that cell.

   Restart the kernel and run the cell. With GPU wheels installed you should see at least one GPU in the TensorFlow line. See [Install TensorFlow with pip](https://www.tensorflow.org/install/pip). If you keep CPU and GPU environments separate, use a dedicated venv (e.g. **`.venv-gpu/`**) and select **`.venv-gpu/bin/python`** as the interpreter.

**Native Windows:** do **not** run `requirements-gpu.txt` in PowerShell; pip will fail with `ResolutionImpossible` / missing `nvidia-nccl-cu12`. Use **`.venv`** and `requirements.txt` (CPU), or use WSL.

If `pip install -r requirements-gpu.txt` fails, try installing TensorFlow first, then the rest:

```bash
pip install "tensorflow[and-cuda]>=2.16,<2.17"
pip install matplotlib scikit-learn
```

The **`<2.17`** upper bound matches `requirements-gpu.txt` (TF 2.17+ can fail on WSL2 when the Windows driver is too old for the bundled CUDA stack).

**Do not** run `pip install -r requirements.txt` and then `requirements-gpu.txt` in the same environment without removing the plain `tensorflow` package first — that often triggers conflicting dependency resolution.

#### If you see `ResolutionImpossible` on native Windows

On **Windows**, PyPI’s default `tensorflow` package is **`tensorflow-intel`** (CPU). Pip’s `tensorflow[and-cuda]` still pulls **NVIDIA CUDA wheels** *and* tries to satisfy **`tensorflow-intel`** at the same time. Those stacks need different versions of **`tensorboard`**, **`protobuf`**, etc., so the resolver often reports **ResolutionImpossible** — this is expected on **native** Windows, not a corrupted venv.

**Workarounds:**

1. **WSL2** (recommended for NVIDIA + modern TF): install Ubuntu in WSL2, `pip install -r requirements-gpu.txt` there, and open the notebook from the Linux filesystem (not `/mnt/c/...` for best I/O).
2. **Stay on CPU** on native Windows: `pip install -r requirements.txt` only, and in the notebook’s first cell set `_REQ_FILE = "requirements.txt"` so the cell does not pull GPU wheels.
3. **Older native Windows GPU** is limited to very old TF versions (e.g. 2.10-era) with manual CUDA/cuDNN; not covered here.

## Project structure

| File | Description |
|---|---|
| `myo_utils.py` | Constants and utility functions (RMS, data loading, auto-curation) |
| `myo-keras.ipynb` | Training, evaluation, and grokking experiment |
| `requirements.txt` | Python dependencies (CPU TensorFlow) |
| `requirements-gpu.txt` | TensorFlow with CUDA via pip (Linux / WSL2) |

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

The grokking sweep reloads the data using an RMS window of 30 to reduce smoothing and make generalisation harder. The training set is then subsampled to 100 samples (stratified) so the model memorises before it generalises. The validation set is also stratified-subsampled to 8,000 samples for fast periodic evaluation, while the full test set remains unchanged. Training uses `AdamW` with decoupled weight decay and sweeps over several weight decay values. Each run resets `tf.random.set_seed(GROKKING_INIT_SEED)` before building the model so all curves share the same initial weights (weight decay is the only intentional difference).

| Weight decay | Expected behaviour |
|--------------|--------------------|
| 0.1 | Known grokking-friendly regime; delayed generalisation and norm compression often visible |
| 1.0 | Strong pressure; faster compression; may shorten plateaus or shift grokking timing |
| 2.0 | Very strong pressure; may limit memorisation or cap final val accuracy |

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Architecture | Dense 200 → 100 → 70 → 8 | ~29.5K params; ~295:1 parameter-to-sample ratio with 100 training samples encourages memorisation first |
| Output activation | `softmax` | Correct for mutually exclusive multi-class |
| Optimiser | `AdamW` | Decoupled weight decay |
| Learning rate | 3e-4 | Slower; allows compression phase to develop |
| RMS window (grokking sweep) | 30 | Less smoothing; increases memorisation/generalisation separation |
| Weight decay | [0.1, 1.0, 2.0] | From moderate to very strong decay |
| Init seed | `GROKKING_INIT_SEED` (42) | Same `keras.Sequential` init for every sweep run |
| Training subset | 100 (stratified) | Larger parameter-to-sample gap encourages memorisation first |
| Validation subset | 8,000 (stratified) | Reliable metric estimates with much lower validation cost |
| Epochs | 150 000 | Extra headroom for delayed generalisation at higher decay |
| Batch size | full-batch (`len(grok_train)`) | One gradient step per epoch for stable grokking dynamics |
| Metric logging | every 100 epochs | Tracks long-run trends without per-epoch validation overhead |
| Report weight decay (`GROKKING_REPORT_WD`) | 0.1 | Test-set confusion matrix uses this run (delayed generalisation), not the sweep-wide peak subsampled-val winner |

### Key plots

1. **Validation accuracy overlay** — all weight decay curves on one graph; shows the threshold where grokking appears.
2. **Weight norms overlay** — shows how stronger decay compresses the model.
3. **Per-run loss / accuracy** — train vs val for each weight decay value.
4. **Confusion matrix and classification report** — for `GROKKING_REPORT_WD`, evaluated on the held-out test set (same RMS window as the sweep); peak-val sweep winner is printed for reference only.
5. **Val accuracy vs weight norm (dual-axis)** — for `GROKKING_REPORT_WD`, overlays validation accuracy and L2 weight norm on the same epoch axis.

### References

- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*. [arXiv:2201.02177](https://arxiv.org/abs/2201.02177).
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
